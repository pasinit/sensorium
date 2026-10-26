"""Our sensory layer as a cost model for stable-worldmodel's planners (B-pusht-0, Push-T).

stable-worldmodel's ``CEMSolver`` calls ``model.get_cost(info, candidates)`` and keeps the
candidates of lowest cost; LeWM's ``JEPA.get_cost`` encodes the current observation and the goal,
rolls its predictor over the candidate action blocks and returns the squared distance of the last
predicted embedding to the goal's. ``SensoryCostModel`` does the same with our layer:

* frames: the environment's 224 px RGB (and the dataset's 224 px goal frame) go through the
  training pipeline's path to 64 px (``pusht.upscale`` to 256, then ``vision.resize`` over the full
  frame), so the planner sees what training saw;
* proprio (the agent arm only): the evaluator z-scores it with its own scaler, which is undone
  here; our layer normalises with its own statistics. The environment gives one observation, so
  the window of ``substeps`` observations repeats it;
* actions: candidates are z-scored action blocks ``[B, N, H, action_block * 2]``; undone with the
  evaluator's scaler, divided by the converter's ``action_scale`` (our episodes' units) and placed
  in our action columns, one chunk per planning step;
* cost: ``|s_H - s_goal|^2`` of our own predictor's open-loop rollout from the current state
  (``planning.own_rollout``), as LeWM's ``criterion``.

Nothing here imports stable-worldmodel: the solver only needs ``get_cost`` and ``parameters``.
"""

from collections.abc import Mapping
from typing import Protocol

import numpy as np
import torch

from sensorium import pusht
from sensorium.model.data import Loaded, build
from sensorium.model.layout import Header
from sensorium.model.planning import own_rollout, union_actions
from sensorium.model.sensory_layer import SensoryLayer, _as_batch
from sensorium.vision import resize


class Scaler(Protocol):
    """The part of sklearn's ``StandardScaler`` the evaluator's ``process`` entries provide."""

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Undo the scaling of ``[n, d]`` values."""
        ...


def frames64(pixels: np.ndarray, size: int = pusht.LEWM_SIZE) -> np.ndarray:
    """``[n, 224, 224, 3]`` uint8 frames -> ``[n, size, size, 3]``: the training pipeline's path
    (bilinear upscale to the stored 256 px, then ``vision.resize`` over the full frame).

    Raises:
        ValueError: if the frames are not ``[n, 224, 224, 3]`` uint8.
    """
    if pixels.ndim != 4 or pixels.shape[1:] != (pusht.SOURCE_SIZE, pusht.SOURCE_SIZE, 3):
        raise ValueError(f"want [n, 224, 224, 3] frames, got {pixels.shape}")
    return resize(pusht.upscale(np.asarray(pixels, dtype=np.uint8)), size)


def as_numpy(x: object) -> np.ndarray:
    """A tensor or array as a numpy array."""
    if torch.is_tensor(x):
        return np.asarray(x.detach().cpu().numpy())
    return np.asarray(x)


class SensoryCostModel(torch.nn.Module):
    """A trained sensory layer behind stable-worldmodel's ``Costable`` interface.

    Args:
        layer: The trained layer (eval mode is set here).
        header: The header of the training episodes (Push-T's).
        action_scaler: The evaluator's action scaler (``process['action']``).
        action_scale: The converter's ``action_scale`` (stored actions = raw / scale).
        proprio_scaler: The evaluator's proprio scaler (``process['proprio']``); needed only if
            the layer reads proprio.
    """

    def __init__(
        self,
        layer: SensoryLayer,
        header: Header,
        action_scaler: Scaler,
        action_scale: float = 1.0,
        proprio_scaler: Scaler | None = None,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.model = layer.model.eval()
        self.header = header
        self.action_scaler = action_scaler
        self.action_scale = action_scale
        self.proprio_scaler = proprio_scaler
        self.senses = self.model.senses
        self.uses_proprio = any(s != "vision" for s in self.senses)
        if self.uses_proprio and proprio_scaler is None:
            raise ValueError("this layer reads proprio: give the evaluator's proprio scaler")

    @torch.no_grad()
    def encode(self, pixels: np.ndarray, proprio: np.ndarray | None = None) -> torch.Tensor:
        """States ``[n, D]`` of ``n`` observations: 224 px frames ``[n, 224, 224, 3]`` and, for a
        layer that reads it, the raw agent proprio ``[n, 4]`` (held over the window)."""
        n, k = len(pixels), self.model.cfg.substeps
        rgb = frames64(pixels, self.model.cfg.vision_size)
        width = len(self.header.proprio_names)
        held = np.zeros((n * k, width), dtype=np.float32)
        if proprio is not None:
            held = np.repeat(np.asarray(proprio, dtype=np.float32).reshape(n, width), k, axis=0)
        inputs = {
            "rgb": rgb,
            "proprio": held,
            "action": np.zeros((n * k, len(self.header.action_names)), dtype=np.float32),
        }
        loaded = Loaded(self.header, inputs, self.layer.space.columns(self.header))
        ends = np.arange(n) * k + (k - 1)  # one window of k observations per frame
        arrays = build(loaded, self.layer.space, self.senses, ends, k, rgb=rgb)
        return self.model.embed(_as_batch(arrays, n)).squeeze(1)

    def chunks(self, candidates: torch.Tensor) -> np.ndarray:
        """Candidate action blocks ``[M, H, block * a]`` (z-scored) -> our action chunks
        ``[M, H, block, U]`` in the layer's action columns."""
        m, h, flat = candidates.shape
        a = len(self.header.action_names)
        raw = self.action_scaler.inverse_transform(as_numpy(candidates).reshape(-1, a))
        ours = np.clip(raw / self.action_scale, -1.0, 1.0).reshape(m, h, flat // a, a)
        return union_actions(self.layer, self.header, ours.astype(np.float32))

    def observation(
        self, info: Mapping[str, object], key: str
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """The last frame of ``info[key]`` (``pixels`` or ``goal``) and its raw proprio, one per
        environment (the solver repeats every entry over the candidates: the first is taken)."""
        pixels = as_numpy(info[key])[:, 0]
        pixels = pixels.reshape(pixels.shape[0], -1, *pixels.shape[-3:])[:, -1]
        proprio = None
        if self.uses_proprio:
            pkey = "proprio" if key == "pixels" else "goal_proprio"
            z = as_numpy(info[pkey])[:, 0]
            z = z.reshape(z.shape[0], -1, z.shape[-1])[:, -1]
            assert self.proprio_scaler is not None
            proprio = self.proprio_scaler.inverse_transform(z)
        return pixels, proprio

    @torch.no_grad()
    def get_cost(self, info: Mapping[str, object], candidates: torch.Tensor) -> torch.Tensor:
        """``[B, N]`` costs of candidate action sequences ``[B, N, H, block * a]``: the squared
        distance of the state our predictor reaches after the ``H`` blocks to the goal's state."""
        b, n, h, _ = candidates.shape
        start = self.encode(*self.observation(info, "pixels"))  # [B, D]
        goal = self.encode(*self.observation(info, "goal"))
        chunks = self.chunks(candidates.reshape(b * n, h, -1))
        start_rep = np.repeat(start.numpy(), n, axis=0)
        final = own_rollout(self.layer, start_rep, chunks)[:, -1]  # [B * N, D]
        final_t = torch.as_tensor(final, dtype=torch.float32).view(b, n, -1)
        return ((final_t - goal[:, None]) ** 2).sum(-1)
