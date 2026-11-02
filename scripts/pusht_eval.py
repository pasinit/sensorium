"""B-pusht-0: evaluation (planning success, block-pose probes, report).

Pre-registration: ``docs/model.md``, "B-pusht-0". Planning goes through stable-worldmodel's own
World evaluator and CEM solver (0.1.1), as le-wm's ``eval.py`` (``8edfeb3``) runs them; this script
repeats ``eval.py``'s steps (pair draw, scalers, World, policy, ``evaluate``) with any model:

* ``ours:<arm>-s<seed>``: our layer through ``sensorium.pusht_plan.SensoryCostModel``;
* ``lewm:<run>``: a LeWM-tiny run's final weights (``<run>/weights_epoch_final.pt``, through
  ``swm.wm.utils.load_pretrained``), its frames through our 64 px pipeline then ImageNet
  normalisation, as LeWM-tiny was trained;
* ``lewm224:<HF repo>``: LeWM's published checkpoint with le-wm ``eval.py``'s own 224 px transform:
  the validation anchor. le-wm's ``eval.py`` cannot run on CPU (it moves the model to CUDA), so
  the anchor runs here; it also checks that this script reproduces ``eval.py``.

LeWM's configs instantiate le-wm's modules (``jepa.JEPA``), so le-wm's directory (``LEWM_DIR``,
``/root/le-wm`` on the VM) is put on ``sys.path`` before a LeWM model is loaded.

    python scripts/pusht_eval.py plan ours:a0-pix-s0 --pairs 200 --seed 43
    python scripts/pusht_eval.py plan lewm:lewm-tiny-s0 --pairs 50 --seed 42
    python scripts/pusht_eval.py plan lewm224:quentinll/lewm-pusht --pairs 50 --seed 42
    python scripts/pusht_eval.py pose ours:a0-pix-s0
    python scripts/pusht_eval.py report

Follow-up checks (``docs/model.md``, "B-pusht-0: follow-up checks"): every plan also keeps each
environment's final state and goal state, so the success rule can be split into its block and
agent halves (``success_rules``); ``--goal-proprio`` takes the goal's agent proprio away from the
agent arm (``SensoryCostModel``); ``pred`` measures a run's prediction loss on LeWM's validation
clips with clean, per-frame-masked and sequence-masked inputs:

    python scripts/pusht_eval.py plan ours:a0-pix-agent-s0 --goal-proprio current
    python scripts/pusht_eval.py pred ours:fix1-pix-s0
    python scripts/pusht_eval.py check-report

Mask-ratio follow-up (fix1 at ``mae_ratio`` 0.5 and 0.3, ``pusht_test.ARMS``): the same plans,
``pred`` and ``pose`` on its runs, then

    python scripts/pusht_eval.py ratio-report

stable-worldmodel, hydra and le-wm's modules are needed only by ``plan`` and LeWM's ``pose``, on
the VM; they are imported there.
"""

import argparse
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path("runs/pusht")
EVAL = ROOT / "eval"
POSE = ROOT / "pose"
DATASET = "pusht_expert_train"
"""The full 224 px file the pairs (and goal frames) come from, as LeWM's evaluation."""
GOAL_OFFSET = 25
BUDGET = 50
CEM = {"num_samples": 300, "var_scale": 1.0, "n_steps": 30, "topk": 30}
PLAN = {"horizon": 5, "receding_horizon": 5, "action_block": 5}
PROTOCOLS = {"primary": (200, 43), "published": (50, 42)}
"""``(pairs, seed)``: the pre-registered primary and LeWM's own 50-pair protocol."""
CHECK = ROOT / "check"
PRED_CLIPS = 2048
PRED_SEED = 0
PRED_BATCH = 128
MASKS = ("clean", "per-frame", "shared")
"""Check 1's input conditions: no mask; training's mask (each frame its own); one mask per clip."""
POSE_EPISODES = 400
POSE_SEED = 1
OURS = ("a0-pix", "a0-pix-agent", "fix1-pix")
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # le-wm's image normalisation (stable-pretraining's)
IMAGENET_STD = [0.229, 0.224, 0.225]
SEEDS = (0, 1, 2)


# ---------------------------------------------------------------------------------------------
# pieces of le-wm's eval.py, without its dependencies (tested here)


@dataclass
class ZScore:
    """sklearn's ``StandardScaler`` as le-wm's ``eval.py`` fits it (rows with a NaN dropped;
    population std; a zero std scales by 1)."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "ZScore":
        """Fit on ``[n, d]`` values."""
        x = np.asarray(x, dtype=np.float64)
        x = x[~np.isnan(x).any(axis=1)]
        std = x.std(0)
        return cls(x.mean(0), np.where(std == 0, 1.0, std))

    def transform(self, x: np.ndarray) -> np.ndarray:
        """``(x - mean) / scale``."""
        return (np.asarray(x) - self.mean) / self.scale

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """``x * scale + mean``."""
        return np.asarray(x) * self.scale + self.mean


def eval_pairs(
    episode_idx: np.ndarray, step_idx: np.ndarray, num_eval: int, seed: int, goal_offset: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(rows, episodes, starts)`` of the evaluation pairs, as le-wm's ``eval.py`` draws them:
    every row whose step leaves ``goal_offset`` steps in its episode is valid, and
    ``rng(seed).choice(len(valid) - 1, num_eval, replace=False)`` of them, sorted, are used.

    Raises:
        ValueError: if fewer than ``num_eval`` rows are valid.
    """
    episodes = np.unique(episode_idx)
    lengths = {e: int(step_idx[episode_idx == e].max()) + 1 for e in episodes}
    max_start = np.array([lengths[e] - goal_offset - 1 for e in episode_idx])
    valid = np.nonzero(step_idx <= max_start)[0]
    if len(valid) - 1 < num_eval:
        raise ValueError("not enough valid starting points for evaluation")
    chosen = np.random.default_rng(seed).choice(len(valid) - 1, size=num_eval, replace=False)
    rows = np.sort(valid[chosen])
    return rows, episode_idx[rows], step_idx[rows]


SUCCESS_RADIUS = 20.0
SUCCESS_ANGLE = np.pi / 9
RULES = ("full", "block", "agent")


def success_rules(final: np.ndarray, goal: np.ndarray) -> dict[str, np.ndarray]:
    """Push-T's success rule and its two halves on ``[n, 7]`` env states (agent x, y, block x,
    y, block angle, agent vx, vy; ``PushT._get_obs``):

    * ``full``: ``PushT.eval_state``, the 4-vector of agent and block position within 20 of the
      goal's and the block angle within pi/9 (the evaluator's success);
    * ``block``: the block position within 20 and its angle within pi/9;
    * ``agent``: the agent position within 20.
    """
    final, goal = np.asarray(final, dtype=np.float64), np.asarray(goal, dtype=np.float64)
    angle = np.abs(goal[:, 4] - final[:, 4])
    angle_ok = np.minimum(angle, 2 * np.pi - angle) < SUCCESS_ANGLE

    def near(cols: slice) -> np.ndarray:
        return np.linalg.norm(goal[:, cols] - final[:, cols], axis=1) < SUCCESS_RADIUS

    return {
        "full": near(slice(0, 4)) & angle_ok,
        "block": near(slice(2, 4)) & angle_ok,
        "agent": near(slice(0, 2)),
    }


def plan_name(model: str, pairs: int, seed: int, goal_proprio: str = "goal") -> str:
    """The file stem of a plan's result (``<kind>-<name>[-goal<variant>]-<pairs>-<seed>``)."""
    kind, name = model.split(":", 1)
    variant = "" if goal_proprio == "goal" else f"-goal{goal_proprio}"
    return f"{kind}-{name.replace('/', '_')}{variant}-{pairs}-{seed}"


def val_sample(val: list[list[int]], lengths: list[int]) -> list[tuple[int, int]]:
    """``PRED_CLIPS`` of LeWM's validation clips ``(subset position, LeWM start)`` that our
    windows can take (``pusht_test.our_start`` inside the episode), drawn with
    ``rng(PRED_SEED)``, sorted.

    Raises:
        ValueError: if fewer than ``PRED_CLIPS`` can be taken.
    """
    import pusht_test as train_script  # noqa: PLC0415

    from sensorium import pusht  # noqa: PLC0415
    from sensorium.model.data import spacing  # noqa: PLC0415

    span = train_script.SEQ_STEPS * train_script.SUBSTEPS * spacing(pusht.DT)
    ok = [
        (int(e), int(t))
        for e, t in val
        if 0 <= train_script.our_start(int(t)) <= lengths[int(e)] - span
    ]
    if len(ok) < PRED_CLIPS:
        raise ValueError(f"only {len(ok)} validation clips can be taken")
    chosen = np.random.default_rng(PRED_SEED).choice(len(ok), PRED_CLIPS, replace=False)
    return sorted(ok[i] for i in chosen)


def pose_targets(state: np.ndarray) -> np.ndarray:
    """``[n, 4]`` block x, y, sin(angle), cos(angle) from Push-T's state ``[n, 7]``
    (``sensorium.pusht.STATE_NAMES``)."""
    from sensorium import pusht  # noqa: PLC0415

    names = pusht.STATE_NAMES
    angle = state[:, names.index("block_angle")]
    xy = state[:, [names.index("block_x"), names.index("block_y")]]
    return np.concatenate([xy, np.sin(angle)[:, None], np.cos(angle)[:, None]], 1)


def pose_r2(features: np.ndarray, targets: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """Episode-grouped 5-fold ridge R2 of each pose target (``x``, ``y``, ``sin``, ``cos``)."""
    from sensorium.model.probes import (  # noqa: PLC0415
        choose_alpha,  # noqa: PLC0415
        episode_folds,
        ridge_predict,
    )

    pred = np.zeros_like(targets)
    for tr, te in episode_folds(groups, 5):
        alpha = choose_alpha(features[tr], targets[tr], groups[tr])
        pred[te] = ridge_predict(features[tr], targets[tr], features[te], alpha)
    sst = ((targets - targets.mean(0)) ** 2).sum(0)
    r2 = 1 - ((pred - targets) ** 2).sum(0) / sst
    return dict(zip(("x", "y", "sin", "cos"), (float(v) for v in r2), strict=True))


def pose_episodes(n_episodes: int, subset: list[int]) -> list[int]:
    """``POSE_EPISODES`` source episodes outside the training subset, drawn with
    ``rng(POSE_SEED)``, sorted."""
    rest = np.setdiff1d(np.arange(n_episodes), np.asarray(subset))
    rng = np.random.default_rng(POSE_SEED)
    return np.sort(rng.choice(rest, POSE_EPISODES, replace=False)).tolist()


# ---------------------------------------------------------------------------------------------
# models


def our_cost_model(run: str, process: dict[str, ZScore], goal_proprio: str = "goal") -> object:
    """Our layer of ``runs/pusht/runs/<run>/ckpt-<STEPS>.pt`` as a cost model."""
    import pusht_test as train_script  # noqa: PLC0415 (scripts/ is on sys.path)

    from sensorium.model.pretrain import load_layer  # noqa: PLC0415
    from sensorium.pusht_plan import SensoryCostModel  # noqa: PLC0415

    layer = load_layer(ROOT / "runs" / run / f"ckpt-{train_script.STEPS}.pt")
    return SensoryCostModel(
        layer, train_script.push_t_header(), process["action"], process["proprio"], goal_proprio
    )


LEWM_DIR = Path("/root/le-wm")


def load_lewm(kind: str, name: str) -> object:
    """A LeWM model: a LeWM-tiny run's final weights (``lewm``) or a HF checkpoint
    (``lewm224``), in eval mode without gradients."""
    import os  # noqa: PLC0415

    import stable_worldmodel as swm  # noqa: PLC0415

    sys.path.insert(0, os.environ.get("LEWM_DIR", str(LEWM_DIR)))
    ref = f"{name}/weights_epoch_final.pt" if kind == "lewm" else name
    model = swm.wm.utils.load_pretrained(ref).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model


def lewm224_transform() -> Callable[[object], object]:
    """le-wm ``eval.py``'s ``img_transform`` at 224 px (the published checkpoint's input)."""
    import torch  # noqa: PLC0415
    from torchvision.transforms import v2 as transforms  # noqa: PLC0415

    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.Resize(size=224),
        ]
    )


def lewm_frame(x: object) -> object:
    """One evaluator frame -> LeWM-tiny's input before normalisation: our 64 px pipeline.

    stable-worldmodel's ``_prepare_info`` calls the transform once per frame, on a
    ``tv_tensors.Image`` (``[3, 224, 224]`` uint8, channels first); returns ``[3, 64, 64]``
    float in [0, 1].

    Raises:
        ValueError: if the frame is not ``[3, 224, 224]``.
    """
    import torch  # noqa: PLC0415

    from sensorium.pusht_plan import frames64  # noqa: PLC0415

    chw = torch.as_tensor(x).detach().cpu()
    if chw.ndim != 3 or chw.shape[0] != 3:
        raise ValueError(f"want one [3, H, W] frame, got {tuple(chw.shape)}")
    small = frames64(chw.permute(1, 2, 0).numpy()[None])[0]  # [64, 64, 3] uint8
    return torch.as_tensor(small).permute(2, 0, 1).float() / 255.0


def lewm_transform() -> Callable[[object], object]:
    """LeWM-tiny's frames: our 64 px pipeline (``lewm_frame``), then le-wm's ImageNet
    normalisation."""
    from torchvision.transforms import v2 as transforms  # noqa: PLC0415

    imagenet = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def apply(x: object) -> object:
        return imagenet(lewm_frame(x))

    return apply


# ---------------------------------------------------------------------------------------------
# planning (VM)


def plan(model: str, pairs: int, seed: int, goal_proprio: str = "goal") -> Path:
    """Planning success of one model on ``pairs`` pairs drawn with ``seed``, through
    stable-worldmodel's World evaluator; writes ``runs/pusht/eval/<plan_name>.json`` (success
    rate, per-pair successes, the pairs, and each environment's final and goal state).

    Raises:
        ValueError: if ``goal_proprio`` is not ``goal`` for a LeWM model.
    """
    import stable_worldmodel as swm  # noqa: PLC0415
    import torch  # noqa: PLC0415

    kind, name = model.split(":", 1)
    if kind != "ours" and goal_proprio != "goal":
        raise ValueError("--goal-proprio applies to our layers only")
    out = EVAL / f"{plan_name(model, pairs, seed, goal_proprio)}.json"
    if out.exists():
        return out
    dataset = swm.data.HDF5Dataset(DATASET, keys_to_cache=["action", "proprio", "state"])
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = np.asarray(dataset.get_col_data(col))
    step_idx = np.asarray(dataset.get_col_data("step_idx"))
    rows, episodes, starts = eval_pairs(episode_idx, step_idx, pairs, seed, GOAL_OFFSET)
    process: dict[str, ZScore] = {}
    for key in ("action", "proprio", "state"):
        process[key] = ZScore.fit(np.asarray(dataset.get_col_data(key)))
        if key != "action":
            process[f"goal_{key}"] = process[key]
    world = swm.World(
        env_name="swm/PushT-v1",
        num_envs=pairs,
        image_shape=(224, 224),
        max_episode_steps=2 * BUDGET,
    )
    if kind == "ours":
        cost_model, transform = our_cost_model(name, process, goal_proprio), {}
    else:
        cost_model = load_lewm(kind, name)
        make = lewm_transform if kind == "lewm" else lewm224_transform
        transform = {"pixels": make(), "goal": make()}
    solver = swm.solver.CEMSolver(model=cost_model, device="cpu", seed=seed, **CEM)
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=swm.PlanConfig(**PLAN), process=process, transform=transform
    )
    world.set_policy(policy)
    torch.set_grad_enabled(False)
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=starts.tolist(),
        goal_offset=GOAL_OFFSET,
        eval_budget=BUDGET,
        episodes_idx=episodes.tolist(),
        callables=[
            {"method": "_set_state", "args": {"state": {"value": "state"}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
        ],
    )
    # 'wait' mode freezes an environment once it succeeds, so its final state is the success state
    envs = [env.unwrapped for env in world.envs.envs]
    final = np.stack([np.asarray(env._get_obs(), dtype=np.float64) for env in envs])
    goal = np.stack([np.asarray(env.goal_state, dtype=np.float64) for env in envs])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": model,
                "goal_proprio": goal_proprio,
                "pairs": pairs,
                "seed": seed,
                "success_rate": float(metrics["success_rate"]),
                "episode_successes": np.asarray(metrics["episode_successes"]).astype(int).tolist(),
                "rows": rows.tolist(),
                "episodes": episodes.tolist(),
                "starts": starts.tolist(),
                "final_states": final.tolist(),
                "goal_states": goal.tolist(),
            }
        )
    )
    logging.info(
        "plan %s on %d pairs (seed %d): %.1f%%", model, pairs, seed, metrics["success_rate"]
    )
    return out


# ---------------------------------------------------------------------------------------------
# block-pose probes (VM)


def pose(model: str) -> Path:
    """Block-pose probes (x, y, sin and cos of the angle) from a model's planning state and the
    state before its projector, on ``POSE_EPISODES`` episodes outside the training subset, at
    every model step; writes ``runs/pusht/pose/<model>.json``."""
    import pusht_test as train_script  # noqa: PLC0415

    from sensorium.episode import Episode  # noqa: PLC0415

    kind, name = model.split(":", 1)
    out = POSE / f"{kind}-{name}.json"
    if out.exists():
        return out
    subset = json.loads((ROOT / "subset.json").read_text())
    chosen = pose_episodes(train_script.episode_count(), subset)
    encode = our_pose_features(name) if kind == "ours" else lewm_pose_features(name)
    feats: dict[str, list[np.ndarray]] = {"state": [], "pre": []}
    targets, groups = [], []
    for i, index in enumerate(chosen):
        ep = Episode.load(train_script.episode_path(index))
        ends = np.arange(train_script.SUBSTEPS - 1, ep.length, train_script.SUBSTEPS)
        s, pre = encode(ep, ends)
        feats["state"].append(s)
        feats["pre"].append(pre)
        targets.append(pose_targets(np.asarray(ep.privileged["state"])[ends]))
        groups.append(np.full(len(ends), i))
    y, g = np.concatenate(targets), np.concatenate(groups)
    result = {k: pose_r2(np.concatenate(v), y, g) for k, v in feats.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": model, "episodes": chosen, **result}))
    return out


def our_pose_features(run: str) -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    """``(s, z)`` of our layer at given window ends of an episode."""
    import pusht_test as train_script  # noqa: PLC0415

    from sensorium.model.planning import encode_at  # noqa: PLC0415
    from sensorium.model.pretrain import load_layer  # noqa: PLC0415

    layer = load_layer(ROOT / "runs" / run / f"ckpt-{train_script.STEPS}.pt")
    return lambda ep, ends: encode_at(layer, ep, ends)


def lewm_pose_features(name: str) -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    """``(projected embedding, CLS token)`` of a LeWM checkpoint at given steps of an episode
    (its 64 px frames, as it was trained)."""
    import torch  # noqa: PLC0415

    from sensorium import pusht  # noqa: PLC0415

    model = load_lewm("lewm", name)
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]

    @torch.no_grad()
    def features(ep: object, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        frames = pusht.lewm_frames(ep)[ends]  # type: ignore[arg-type]
        x = torch.as_tensor(frames).permute(0, 3, 1, 2).float() / 255.0
        out = model.encoder((x - mean) / std, interpolate_pos_encoding=True)
        cls = out.last_hidden_state[:, 0]
        return model.projector(cls).numpy(), cls.numpy()

    return features


# ---------------------------------------------------------------------------------------------
# check 1: prediction loss with clean and masked inputs (VM)


def pred_batches(layer: object, clips: list[tuple[int, int]]) -> list[dict[str, object]]:
    """Batches of ``PRED_BATCH`` sequences, one per clip, cut as training cut them: the run's
    loader configuration, its frame cache, our start for LeWM's clip (``pusht_test.our_start``),
    the centre view (the full frame at crop 1)."""
    import pusht_test as train_script  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from sensorium.model.data import SequenceDataset, sequence_with_ends  # noqa: PLC0415

    paths = train_script.subset_paths()
    model = layer.model  # type: ignore[attr-defined]
    senses = model.data_senses
    data = SequenceDataset(paths, layer.space, senses, layer.loader, train=False)  # type: ignore[attr-defined]
    rng = np.random.default_rng(0)  # unused: no crop, no touch noise, fixed starts
    seqs = [
        sequence_with_ends(
            data.load(paths[e]),
            layer.space,  # type: ignore[attr-defined]
            senses,
            layer.loader,  # type: ignore[attr-defined]
            rng,
            train=False,
            start=train_script.our_start(t),
        )[0]
        for e, t in clips
    ]
    return [
        {k: torch.as_tensor(np.stack([q[k] for q in seqs[i : i + PRED_BATCH]])) for k in seqs[0]}
        for i in range(0, len(seqs), PRED_BATCH)
    ]


def pred_loss(model: object, batches: list[dict[str, object]], mask: str, bn: str) -> float:
    """Mean ``pred`` of ``SensoryModel.forward`` (the training loss's next-state MSE) over the
    batches, with the encoder's input ``mask``ed (``MASKS``) by ``reference.mask_batch``, the
    function training used.

    The decoder is detached for the call, so the forward pass takes its plain path
    (``states = embed(batch)``). With training's per-frame mask that is the same computation as
    fix1's one masked pass (``project(summary_token(masked))``); ``tests`` check it.
    ``bn = "eval"`` uses the projector's running BatchNorm statistics (as planning does),
    ``"batch"`` each batch's own (as the training log did; the running statistics are kept).
    """
    import copy  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from sensorium.model import reference  # noqa: PLC0415

    net = copy.deepcopy(model)
    net.recon = None  # type: ignore[attr-defined]
    net.train(bn == "batch")  # type: ignore[attr-defined]
    for m in net.modules():  # type: ignore[attr-defined]
        if isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = 0.0  # batch statistics in "batch" mode, but running ones unchanged
    c = net.cfg  # type: ignore[attr-defined]
    torch.manual_seed(PRED_SEED)  # the same masks for every model
    total = 0.0
    with torch.no_grad():
        for batch in batches:
            if mask != "clean":
                batch, _, _ = reference.mask_batch(
                    batch,
                    net.senses,  # type: ignore[attr-defined]
                    c.vision_patch,
                    c.mae_ratio,
                    c.mae_proprio_ratio,
                    shared=mask == "shared",
                )
            total += float(net(batch)["pred"])  # type: ignore[operator]
    return total / len(batches)


def pred(model: str) -> Path:
    """Check 1 for one of our runs: its prediction loss on ``PRED_CLIPS`` of LeWM's validation
    clips (``val_sample``) under every ``MASKS`` condition, with eval and batch BatchNorm
    statistics; writes ``runs/pusht/check/pred-<run>.json``.

    Raises:
        ValueError: if the model is not one of ours.
    """
    import pusht_test as train_script  # noqa: PLC0415

    from sensorium.model.pretrain import load_layer  # noqa: PLC0415

    kind, name = model.split(":", 1)
    if kind != "ours":
        raise ValueError("pred measures our layers only")
    out = CHECK / f"pred-{name}.json"
    if out.exists():
        return out
    split = json.loads((ROOT / "clips.json").read_text())
    lengths = [train_script.episode_length(p) for p in train_script.subset_paths()]
    clips = val_sample(split["val"], lengths)
    layer = load_layer(ROOT / "runs" / name / f"ckpt-{train_script.STEPS}.pt")
    batches = pred_batches(layer, clips)
    result = {
        bn: {mask: pred_loss(layer.model, batches, mask, bn) for mask in MASKS}
        for bn in ("eval", "batch")
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": model, "clips": clips, **result}))
    logging.info("pred %s: %s", model, result)
    return out


# ---------------------------------------------------------------------------------------------
# report


def load_plans(protocol: str) -> dict[str, np.ndarray]:
    """``{arm: [seeds, pairs] successes}`` of every arm on a protocol (missing seeds left out)."""
    pairs, seed = PROTOCOLS[protocol]
    arms = {"lewm-tiny": [f"lewm-lewm-tiny-s{s}" for s in SEEDS]}
    arms |= {arm: [f"ours-{arm}-s{s}" for s in SEEDS] for arm in OURS}
    out = {}
    for arm, names in arms.items():
        rows = [EVAL / f"{n}-{pairs}-{seed}.json" for n in names]
        found = [json.loads(p.read_text())["episode_successes"] for p in rows if p.exists()]
        if found:
            out[arm] = np.asarray(found, dtype=float)
    return out


LEWM_LOGS = Path("lewm_logs")
"""LeWM-tiny's CSV logs (``scripts/lewm_launch.py``), copied next to ``runs/pusht`` on the VM."""


def our_curve(arm: str, seed: int) -> dict[str, np.ndarray]:
    """``{clips, pred, reg, recon}`` of one of our runs (its ``log.jsonl``)."""
    import pusht_test as train_script  # noqa: PLC0415

    path = ROOT / "runs" / f"{arm}-s{seed}" / "log.jsonl"
    if not path.exists():
        return {}
    log = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    out = {"clips": np.array([r["step"] for r in log]) * train_script.BATCH}
    for key in ("pred", "reg", "recon"):
        out[key] = np.array([r.get(key, np.nan) for r in log], dtype=float)
    return out


LEWM_COLUMNS = {
    "step": ("step", "global_step", "trainer/global_step"),
    "pred": ("fit/pred_loss", "train/pred_loss", "train/pred_loss_step", "pred_loss"),
    "sigreg": ("fit/sigreg_loss", "train/sigreg_loss", "train/sigreg_loss_step", "sigreg_loss"),
}
"""Accepted column names of LeWM's logged metrics (Lightning's CSVLogger or stable-pretraining's
RegistryLogger), first match wins."""


def lewm_metrics(seed: int) -> Path | None:
    """A LeWM-tiny run's ``metrics.csv``: in ``lewm_logs/lewm-tiny-s<seed>/`` (as the VM copies
    stable-pretraining's RegistryLogger output) or a Lightning ``version_*`` folder under it."""
    run = LEWM_LOGS / f"lewm-tiny-s{seed}"
    found = sorted(run.glob("metrics.csv")) + sorted(run.glob("version_*/metrics.csv"))
    return found[0] if found else None


def column(header: list[str], key: str) -> str:
    """The first of ``LEWM_COLUMNS[key]`` in ``header``.

    Raises:
        KeyError: if none is (the names are listed in the message).
    """
    for name in LEWM_COLUMNS[key]:
        if name in header:
            return name
    raise KeyError(f"no {key} column among {header}")


def lewm_curve(seed: int) -> dict[str, np.ndarray]:
    """``{clips, pred, reg}`` of a LeWM-tiny run (its ``metrics.csv``; ``reg`` is the weighted
    SIGReg term, as ours)."""
    import csv  # noqa: PLC0415

    import pusht_test as train_script  # noqa: PLC0415

    path = lewm_metrics(seed)
    if path is None:
        return {}
    reader = csv.DictReader(path.open())
    header = list(reader.fieldnames or [])
    step, pred, sig = (column(header, k) for k in ("step", "pred", "sigreg"))
    rows = [r for r in reader if r.get(pred) not in (None, "") and r.get(step) not in (None, "")]
    weight = 0.09  # LeWM's SIGReg weight
    return {
        "clips": np.array([int(float(r[step])) for r in rows]) * train_script.BATCH,
        "pred": np.array([float(r[pred]) for r in rows]),
        "reg": np.array([weight * float(r[sig]) if r.get(sig) else np.nan for r in rows]),
    }


def curves_figure() -> None:
    """``runs/pusht/figures/pusht_curves.png``: prediction loss, weighted SIGReg and (fix1)
    reconstruction against training clips seen, every seed of every arm (each in its own units:
    LeWM-tiny's and ours are not on one scale)."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    keys = ("pred", "reg", "recon")
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3.6))
    arms = [("lewm-tiny", lewm_curve), *((a, lambda s, a=a: our_curve(a, s)) for a in OURS)]
    for i, (arm, curve) in enumerate(arms):
        for seed in SEEDS:
            c = curve(seed)
            for ax, key in zip(axes, keys, strict=True):
                if key in c and not np.isnan(c[key]).all():
                    ax.plot(
                        c["clips"], c[key], color=f"C{i}", lw=0.8, label=arm if seed == 0 else None
                    )
    for ax, key in zip(axes, keys, strict=True):
        ax.set(title=key, xlabel="training clips seen", yscale="log")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    (ROOT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "figures" / "pusht_curves.png", dpi=130)
    plt.close(fig)


def report() -> None:
    """Success per arm on both protocols, the pre-registered contrasts, and the pose probes."""
    from sensorium.model.paired import contrast_values, describe  # noqa: PLC0415

    for protocol, (pairs, seed) in PROTOCOLS.items():
        plans = load_plans(protocol)
        print(f"\n## Planning success, {protocol} ({pairs} pairs, rng {seed})\n")  # noqa: T201
        print("| arm | success (mean +- sd over seeds) | seeds |\n|---|---|---|")  # noqa: T201
        for arm, s in plans.items():
            rates = s.mean(1)
            print(f"| {arm} | {rates.mean():.3f} +- {rates.std():.3f} | {len(s)} |")  # noqa: T201
        groups = np.arange(pairs)
        print("\n| contrast | value | |\n|---|---|---|")  # noqa: T201
        for a, b, paired in (
            ("a0-pix", "lewm-tiny", False),
            ("fix1-pix", "a0-pix", True),
            ("a0-pix-agent", "a0-pix", True),
        ):
            if a in plans and b in plans:
                c = contrast_values(plans[a], plans[b], groups, paired=paired)
                label = "paired" if paired else "seeds unpaired, pairs shared"
                print(f"| {a} - {b} | {describe(c)} | {label} |")  # noqa: T201
    print("\n## Block pose from the state (episode-grouped 5-fold ridge R2)\n")  # noqa: T201
    print("| model | layer | x | y | sin | cos |\n|---|---|---|---|---|---|")  # noqa: T201
    for path in sorted(POSE.glob("*.json")):
        r = json.loads(path.read_text())
        for layer in ("state", "pre"):
            cells = " | ".join(f"{r[layer][k]:.3f}" for k in ("x", "y", "sin", "cos"))
            print(f"| {r['model']} | {layer} | {cells} |")  # noqa: T201
    curves_figure()


CHECK_ARMS = {
    "lewm-tiny": ("lewm-tiny", "goal"),
    "a0-pix": ("a0-pix", "goal"),
    "a0-pix-agent": ("a0-pix-agent", "goal"),
    "a0-pix-agent-cur": ("a0-pix-agent", "current"),
    "a0-pix-agent-mean": ("a0-pix-agent", "mean"),
}
"""Check 2's arms: ``(run family, goal_proprio)``."""
CHECK_CONTRASTS = (
    ("a0-pix-agent", "a0-pix"),
    ("a0-pix-agent-cur", "a0-pix"),
    ("a0-pix-agent-mean", "a0-pix"),
)


def check_paths(eval_dir: Path, family: str, variant: str) -> list[Path]:
    """The primary plans of every seed of a run family with a ``goal_proprio`` variant."""
    pairs, seed = PROTOCOLS["primary"]
    kind = "lewm" if family == "lewm-tiny" else "ours"
    return [
        eval_dir / f"{plan_name(f'{kind}:{family}-s{s}', pairs, seed, variant)}.json" for s in SEEDS
    ]


def check_plans(
    eval_dir: Path | None = None, arms: dict[str, tuple[str, str]] | None = None
) -> dict[str, dict[str, np.ndarray]]:
    """``{arm: {rule: [seeds, pairs] successes, "evaluator": the evaluator's own}}`` of ``arms``
    (default check 2's, ``CHECK_ARMS``) on the primary protocol, from ``eval_dir`` (default
    ``EVAL``; arms with a missing seed left out)."""
    eval_dir = EVAL if eval_dir is None else eval_dir
    out: dict[str, dict[str, np.ndarray]] = {}
    for arm, (family, variant) in (CHECK_ARMS if arms is None else arms).items():
        paths = check_paths(eval_dir, family, variant)
        if not all(p.exists() for p in paths):
            continue
        runs = [json.loads(p.read_text()) for p in paths]
        rules = [success_rules(r["final_states"], r["goal_states"]) for r in runs]
        out[arm] = {k: np.asarray([r[k] for r in rules], dtype=float) for k in RULES}
        out[arm]["evaluator"] = np.asarray([r["episode_successes"] for r in runs], dtype=float)
    return out


def success_figure(
    plans: dict[str, dict[str, np.ndarray]], name: str = "pusht_check_rules.png"
) -> None:
    """``runs/pusht/figures/<name>``: success under each rule per arm (bars: mean over seeds;
    dots: each seed)."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    arms = list(plans)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    width = 0.8 / len(RULES)
    for j, rule in enumerate(RULES):
        x = np.arange(len(arms)) + (j - (len(RULES) - 1) / 2) * width
        per_seed = np.asarray([plans[a][rule].mean(1) for a in arms])  # [arms, seeds]
        ax.bar(x, per_seed.mean(1), width * 0.9, color=f"C{j}", label=rule)
        for i in range(len(arms)):
            ax.plot([x[i]] * per_seed.shape[1], per_seed[i], "k.", ms=3)
    ax.set_xticks(np.arange(len(arms)), arms, fontsize=8)
    ax.set(ylabel="success (200 pairs)", ylim=(0, 1))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="rule", fontsize=8, frameon=False)
    fig.tight_layout()
    (ROOT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "figures" / name, dpi=130)
    plt.close(fig)


def rule_tables(
    plans: dict[str, dict[str, np.ndarray]],
    contrasts: tuple[tuple[str, str], ...],
    first: dict[str, np.ndarray],
) -> None:
    """Print success under each rule per arm, the ``full`` rule's match with the evaluator (and
    with ``first``'s plans), and the paired contrasts under each rule."""
    from sensorium.model.paired import contrast_values, describe  # noqa: PLC0415

    pairs, seed = PROTOCOLS["primary"]
    print(f"\n## Success by rule ({pairs} pairs, rng {seed})\n")  # noqa: T201
    print("| arm | full | block | agent | seeds |\n|---|---|---|---|---|")  # noqa: T201
    for arm, rules in plans.items():
        cells = " | ".join(
            f"{r.mean(1).mean():.3f} +- {r.mean(1).std():.3f}" for r in (rules[k] for k in RULES)
        )
        print(f"| {arm} | {cells} | {len(rules['full'])} |")  # noqa: T201
    print("\n| arm | full rule = evaluator | full rule = first run |\n|---|---|---|")  # noqa: T201
    for arm, rules in plans.items():
        same = (rules["full"] == rules["evaluator"]).mean()
        then = first.get(arm)
        again = f"{(rules['full'] == then).mean():.3f}" if then is not None else "n/a"
        print(f"| {arm} | {same:.3f} | {again} |")  # noqa: T201
    groups = np.arange(pairs)
    print("\n| rule | contrast | value |\n|---|---|---|")  # noqa: T201
    for rule in RULES:
        for a, b in contrasts:
            if a in plans and b in plans:
                c = contrast_values(plans[a][rule], plans[b][rule], groups, paired=True)
                print(f"| {rule} | {a} - {b} | {describe(c)} |")  # noqa: T201


def pred_table(runs: tuple[str, ...] | None = None) -> None:
    """Print check 1's prediction losses of every ``runs/pusht/check/pred-<run>.json`` (or only
    those of ``runs``)."""
    print("\n## Prediction loss on LeWM's validation clips\n")  # noqa: T201
    print("| run | BN | clean | per-frame | shared |\n|---|---|---|---|---|")  # noqa: T201
    for path in sorted(CHECK.glob("pred-*.json")):
        if runs is not None and path.stem.removeprefix("pred-") not in runs:
            continue
        r = json.loads(path.read_text())
        for bn in ("eval", "batch"):
            cells = " | ".join(f"{r[bn][m]:.4f}" for m in MASKS)
            print(f"| {r['model']} | {bn} | {cells} |")  # noqa: T201


def check_report() -> None:
    """Both checks: success under each rule per arm, the contrasts with a0-pix, the match of the
    ``full`` rule with the evaluator (and with the first run's plans, if copied to
    ``runs/pusht/eval_first``), the bar figure, and check 1's prediction losses."""
    plans = check_plans()
    rule_tables(plans, CHECK_CONTRASTS, load_plans_from(ROOT / "eval_first"))
    if plans:
        success_figure(plans)
    pred_table()


def load_plans_from(
    eval_dir: Path, arms: dict[str, tuple[str, str]] | None = None
) -> dict[str, np.ndarray]:
    """``{arm: [seeds, pairs]}`` evaluator successes of a previous run's primary plans of ``arms``
    (default ``CHECK_ARMS``) in ``eval_dir`` (arms without all seeds left out; empty if the
    folder is absent)."""
    out = {}
    for arm, (family, variant) in (CHECK_ARMS if arms is None else arms).items():
        paths = check_paths(eval_dir, family, variant)
        if all(p.exists() for p in paths):
            out[arm] = np.asarray(
                [json.loads(p.read_text())["episode_successes"] for p in paths], dtype=float
            )
    return out


RATIO_ARMS = {
    "a0-pix": ("a0-pix", "goal"),
    "fix1-pix": ("fix1-pix", "goal"),
    "fix1-pix-r50": ("fix1-pix-r50", "goal"),
    "fix1-pix-r30": ("fix1-pix-r30", "goal"),
}
"""The mask-ratio follow-up's arms (fix1-pix: ratio 0.75)."""
RATIO_CONTRASTS = (
    ("fix1-pix", "a0-pix"),
    ("fix1-pix-r50", "a0-pix"),
    ("fix1-pix-r30", "a0-pix"),
    ("fix1-pix-r50", "fix1-pix"),
    ("fix1-pix-r30", "fix1-pix"),
)
RATIO_FIX1 = ("fix1-pix", "fix1-pix-r50", "fix1-pix-r30")


def ratio_curves_figure() -> None:
    """``runs/pusht/figures/pusht_ratio_curves.png``: prediction loss, weighted SIGReg and
    reconstruction against training clips seen, every seed of fix1 at each mask ratio, with
    a0-pix for reference."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    keys = ("pred", "reg", "recon")
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3.6))
    for i, arm in enumerate(("a0-pix", *RATIO_FIX1)):
        for seed in SEEDS:
            c = our_curve(arm, seed)
            for ax, key in zip(axes, keys, strict=True):
                if key in c and not np.isnan(c[key]).all():
                    label = arm if seed == 0 else None
                    ax.plot(c["clips"], c[key], color=f"C{i}", lw=0.8, label=label)
    for ax, key in zip(axes, keys, strict=True):
        ax.set(title=key, xlabel="training clips seen", yscale="log")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    (ROOT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "figures" / "pusht_ratio_curves.png", dpi=130)
    plt.close(fig)


def ratio_report() -> None:
    """The mask-ratio follow-up: success under each rule (with the first run's evaluator
    successes, if copied to ``runs/pusht/eval_first``), the contrasts with a0-pix and with fix1
    at 0.75, the bar figure, the prediction losses, the pose probes and the training curves."""
    plans = check_plans(arms=RATIO_ARMS)
    rule_tables(plans, RATIO_CONTRASTS, load_plans_from(ROOT / "eval_first", RATIO_ARMS))
    if plans:
        success_figure(plans, "pusht_ratio_rules.png")
    runs = tuple(f"{a}-s{s}" for a in ("a0-pix", *RATIO_FIX1) for s in SEEDS)
    pred_table(runs)
    print("\n## Block pose from the state (episode-grouped 5-fold ridge R2)\n")  # noqa: T201
    print("| model | layer | x | y | sin | cos |\n|---|---|---|---|---|---|")  # noqa: T201
    for run in runs:
        path = POSE / f"ours-{run}.json"
        if path.exists():
            r = json.loads(path.read_text())
            for layer in ("state", "pre"):
                cells = " | ".join(f"{r[layer][k]:.3f}" for k in ("x", "y", "sin", "cos"))
                print(f"| {r['model']} | {layer} | {cells} |")  # noqa: T201
    ratio_curves_figure()


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "command", choices=("plan", "pose", "report", "pred", "check-report", "ratio-report")
    )
    parser.add_argument("model", nargs="?", default="")
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--goal-proprio", default="goal", choices=("goal", "current", "mean"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    sys.path.insert(0, str(Path(__file__).parent))
    if args.command == "plan":
        plan(args.model, args.pairs, args.seed, args.goal_proprio)
    elif args.command == "pose":
        pose(args.model)
    elif args.command == "pred":
        pred(args.model)
    elif args.command == "check-report":
        check_report()
    elif args.command == "ratio-report":
        ratio_report()
    else:
        report()


if __name__ == "__main__":
    main()
