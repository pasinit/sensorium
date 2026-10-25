"""Prediction and planning with a trained sensory layer (B-plan-0, ``docs/model.md``).

Model-side pieces only; the simulator side (restoring a recorded scene and stepping it) is
``sensorium.bodies.restore``. Nothing here reads privileged state: the success criterion uses the
arm-joint positions of the proprio input.

* ``states_at`` / ``chunks_at``: the states of windows ending at given steps of an episode, and the
  action chunks that follow them (the predictor's conditioning), in the layer's channel space;
  ``encode_at`` also returns the pre-projector summary tokens, and ``encode_parts`` every stage
  of one sense (normalised input, stem token, fused token, summary, state);
* ``own_rollout``: open-loop rollout of the layer's own action-conditioned predictor (its context is
  the last ``steps`` positions, observed then predicted);
* ``RidgeDynamics``: a ridge model ``s_{t+1} = W [s_t, chunk_t, 1]`` fitted on encoded episodes, for
  encoders without a predictor (MAE) and as a control for those with one;
* ``RidgeDecoder``: a ridge map from state to arm-joint positions, so arms with different latent
  scales are compared in radians;
* ``cem``: the cross-entropy method over action sequences.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from sensorium.episode import Episode
from sensorium.model.data import build, load_episode, spacing
from sensorium.model.layout import Header
from sensorium.model.probes import choose_alpha
from sensorium.model.sensory_layer import SensoryLayer, _as_batch
from sensorium.vision import eval_view

ARM_JOINTS = (
    "q_left_shoulder_pitch",
    "q_left_shoulder_roll",
    "q_left_shoulder_yaw",
    "q_left_elbow",
    "q_right_shoulder_pitch",
    "q_right_shoulder_roll",
    "q_right_shoulder_yaw",
    "q_right_elbow",
)
"""Arm joints of the success criterion (shoulders and elbows, both arms)."""


def arm_q(episode: Episode, steps: np.ndarray) -> np.ndarray:
    """``[n, 8]`` arm-joint positions (radians) of the proprio input at ``steps``.

    Raises:
        ValueError: if an arm joint is missing from the episode's proprio channels.
    """
    names = list(episode.proprio_names)
    cols = [names.index(n) for n in ARM_JOINTS]
    return np.asarray(episode.proprio[steps][:, cols], dtype=np.float64)


def states_at(layer: SensoryLayer, episode: Episode, ends: np.ndarray) -> np.ndarray:
    """``[n, dim]`` states of the windows whose last observation is at ``ends`` (eval mode,
    centre crop, as ``SensoryLayer.encode``).

    Raises:
        ValueError: if a window starts before the episode.
    """
    return encode_at(layer, episode, ends)[0]


def encode_at(
    layer: SensoryLayer, episode: Episode, ends: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``[n, dim]`` states s (projector output) and summary tokens z (the projector's input) of
    the windows whose last observation is at ``ends``, from one forward pass (see
    ``states_at``).

    Raises:
        ValueError: if a window starts before the episode.
    """
    parts = encode_parts(layer, episode, ends)
    return parts["s"], parts["z"]


@torch.no_grad()
def encode_parts(
    layer: SensoryLayer, episode: Episode, ends: np.ndarray, sense: str = "proprio_body"
) -> dict[str, np.ndarray]:
    """Every stage of one sense on its way to the state, for the windows ending at ``ends``,
    from one forward pass (eval mode, centre crop):

    * ``input_last`` ``[n, U]`` and ``input_window`` ``[n, K * U]``: the sense's normalised
      model input (the window's last observation, and all ``K``);
    * ``stem`` ``[n, dim]``: its stem token, before the fusion;
    * ``fused`` ``[n, dim]``: the same token after the fusion;
    * ``z`` ``[n, K_s * dim]``: the summary tokens; ``s`` ``[n, K_s * dim]``: the state.

    Raises:
        ValueError: if a window starts before the episode, or the sense is not configured or
            has more than one token.
    """
    model = layer.model
    enc = model.encoder
    if sense not in enc.n_tokens or enc.n_tokens[sense] != 1:
        raise ValueError(f"{sense!r} is not a configured one-token sense")
    seen: dict[str, list[torch.Tensor]] = {"z": [], "stem": [], "fused": []}

    def keep(name: str, value: torch.Tensor) -> None:
        seen[name].append(value.detach())

    slot = model.cfg.summary_tokens + sum(
        enc.n_tokens[x] for x in enc.senses[: enc.senses.index(sense)]
    )
    k = model.cfg.substeps
    loaded = load_episode(episode, layer.space, model.senses)
    if np.min(ends) < (k - 1) * spacing(loaded.header.dt):
        raise ValueError("a window starts before the episode")
    was = model.training
    model.eval()
    rgb = (
        eval_view(loaded.inputs["rgb"][ends], layer.loader.view)
        if "vision" in model.senses
        else None
    )
    arrays = build(loaded, layer.space, model.senses, np.asarray(ends), k, rgb=rgb)
    hooks = [
        enc.projector.register_forward_pre_hook(lambda m, a: keep("z", a[0])),
        enc.stems[sense].register_forward_hook(lambda m, a, o: keep("stem", o)),
        enc.fusion.register_forward_hook(lambda m, a, o: keep("fused", o[:, slot])),
    ]
    try:
        states = model.embed(_as_batch(arrays, len(ends))).squeeze(1).cpu().numpy()
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was)
    n = len(states)
    x = np.asarray(arrays[sense], dtype=np.float64)  # [n, K, U], normalised
    out = {name: torch.cat(v).reshape(n, -1).cpu().numpy() for name, v in seen.items()}
    out |= {"s": states, "input_last": x[:, -1], "input_window": x.reshape(n, -1)}
    return {name: np.asarray(v, dtype=np.float64) for name, v in out.items()}


def union_actions(layer: SensoryLayer, header: Header, actions: np.ndarray) -> np.ndarray:
    """Raw actions ``[..., A]`` of an episode in the layer's union action columns ``[..., U]``."""
    cols = layer.space.columns(header)
    out = np.zeros((*actions.shape[:-1], layer.space.size("action")), dtype=np.float32)
    out[..., cols.dst["action"]] = actions[..., cols.src["action"]]
    return out


def chunks_at(layer: SensoryLayer, episode: Episode, ends: np.ndarray) -> np.ndarray:
    """``[n, K, U]`` the action chunks that follow the windows ending at ``ends`` (the
    conditioning of the step from each window to the next), as ``data.build`` cuts them."""
    k = layer.model.cfg.substeps
    s = spacing(episode.dt)
    idx = np.minimum(np.asarray(ends)[:, None] + s * np.arange(k)[None], episode.length - 1)
    loaded = load_episode(episode, layer.space, ())
    return union_actions(layer, loaded.header, np.asarray(episode.action)[idx])


@torch.no_grad()
def own_rollout(layer: SensoryLayer, start: np.ndarray, chunks: np.ndarray) -> np.ndarray:
    """Open-loop rollout of the layer's own predictor.

    Args:
        layer: A layer whose model has an action-conditioned predictor.
        start: ``[B, dim]`` observed states.
        chunks: ``[B, H, K, U]`` the action chunk applied after each step.

    Returns:
        ``[B, H, dim]`` predicted states after 1..H chunks. The predictor's context is the last
        ``steps`` positions (observed, then predicted).
    """
    model = layer.model.eval()
    context = model.cfg.steps
    states = [torch.as_tensor(start, dtype=torch.float32)]
    acts = torch.as_tensor(chunks, dtype=torch.float32)
    out = []
    for h in range(acts.shape[1]):
        lo = max(0, len(states) - context)
        seq = torch.stack(states[lo:], 1)
        a = acts[:, lo : h + 1]
        cond = model.action(a.flatten(0, 1)).view(seq.shape[0], seq.shape[1], -1)
        nxt = model.predict(seq, cond)[:, -1]
        states.append(nxt)
        out.append(nxt)
    return torch.stack(out, 1).numpy().astype(np.float64)


@dataclass
class Ridge:
    """A ridge map fitted on standardised inputs: ``y = ((x - mean) / scale) @ w + bias``."""

    mean: np.ndarray
    scale: np.ndarray
    w: np.ndarray
    bias: np.ndarray
    alpha: float = float("nan")  # the chosen strength (not saved with the weights)

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "Ridge":
        """Fit with the strength chosen by episode-grouped inner folds (``probes.choose_alpha``)."""
        mean, scale = x.mean(0), x.std(0).clip(min=1e-6)
        z = (x - mean) / scale
        alpha = choose_alpha(z, y, groups)
        bias = y.mean(0)
        w = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - bias))
        return cls(mean, scale, w, bias, alpha)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Predictions for the rows of ``x``."""
        return np.asarray(((x - self.mean) / self.scale) @ self.w + self.bias)


class RidgeDynamics:
    """``s_{t+1} = Ridge([s_t, chunk_t])`` (``chunk_t`` flattened)."""

    def __init__(self, ridge: Ridge) -> None:
        self.ridge = ridge

    @classmethod
    def fit(
        cls, states: np.ndarray, chunks: np.ndarray, nxt: np.ndarray, groups: np.ndarray
    ) -> "RidgeDynamics":
        """Fit on transitions ``(states [n, D], chunks [n, K, U]) -> nxt [n, D]``."""
        x = np.concatenate([states, chunks.reshape(len(chunks), -1)], 1)
        return cls(Ridge.fit(x, nxt, groups))

    def rollout(self, start: np.ndarray, chunks: np.ndarray) -> np.ndarray:
        """``[B, H, D]`` states after 1..H chunks, from ``start [B, D]``."""
        s, out = start, []
        for h in range(chunks.shape[1]):
            s = self.ridge(np.concatenate([s, chunks[:, h].reshape(len(s), -1)], 1))
            out.append(s)
        return np.stack(out, 1)


def cem(
    cost: Callable[[np.ndarray], np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    low: np.ndarray | float,
    high: np.ndarray | float,
    rng: np.random.Generator,
    iterations: int = 5,
    samples: int = 128,
    elites: int = 16,
) -> np.ndarray:
    """The cross-entropy method: the mean of the final elite set of action sequences.

    Args:
        cost: ``[N, ...] -> [N]`` cost of each candidate (lower is better).
        mean, std: Initial sampling distribution, shaped like one candidate.
        low, high: Bounds broadcastable to a candidate (samples are clipped to them).
        rng: Random generator.
        iterations, samples, elites: CEM settings.
    """
    mu, sigma = mean.astype(np.float64), std.astype(np.float64)
    for _ in range(iterations):
        draw = mu + sigma * rng.standard_normal((samples, *mu.shape))
        draw = np.clip(draw, low, high)
        best = draw[np.argsort(cost(draw))[:elites]]
        mu, sigma = best.mean(0), best.std(0) + 1e-6
    return np.asarray(mu)


def rms(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Root-mean-square difference over the last axis."""
    return np.asarray(np.sqrt(((a - b) ** 2).mean(-1)))


def sliding_ends(
    length: int, first: int, starts: Sequence[int], stride: int, horizon: int
) -> np.ndarray:
    """``[len(starts), horizon + 1]`` window ends ``t, t + stride, ...`` for each start ``t``
    (all inside ``[first, length - 1]``).

    Raises:
        ValueError: if a start leaves the episode.
    """
    ends = np.asarray(starts)[:, None] + stride * np.arange(horizon + 1)[None]
    if ends.min() < first or ends.max() > length - 1:
        raise ValueError("a start is too close to an end of the episode")
    return ends
