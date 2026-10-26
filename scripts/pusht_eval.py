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


def our_cost_model(run: str, process: dict[str, ZScore]) -> object:
    """Our layer of ``runs/pusht/runs/<run>/ckpt-<STEPS>.pt`` as a cost model."""
    import pusht_test as train_script  # noqa: PLC0415 (scripts/ is on sys.path)

    from sensorium.model.pretrain import load_layer  # noqa: PLC0415
    from sensorium.pusht_plan import SensoryCostModel  # noqa: PLC0415

    layer = load_layer(ROOT / "runs" / run / f"ckpt-{train_script.STEPS}.pt")
    return SensoryCostModel(
        layer, train_script.push_t_header(), process["action"], process["proprio"]
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


def lewm_transform() -> Callable[[object], object]:
    """LeWM-tiny's frames: our 64 px pipeline, then le-wm's ImageNet normalisation."""
    import torch  # noqa: PLC0415
    from torchvision.transforms import v2 as transforms  # noqa: PLC0415

    from sensorium.pusht_plan import frames64  # noqa: PLC0415

    imagenet = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def apply(x: object) -> object:
        frames = np.asarray(x.cpu().numpy() if torch.is_tensor(x) else x)  # [n, H, W, 3] uint8
        small = torch.as_tensor(frames64(frames)).permute(0, 3, 1, 2).float() / 255.0
        return imagenet(small)

    return apply


# ---------------------------------------------------------------------------------------------
# planning (VM)


def plan(model: str, pairs: int, seed: int) -> Path:
    """Planning success of one model on ``pairs`` pairs drawn with ``seed``, through
    stable-worldmodel's World evaluator; writes ``runs/pusht/eval/<model>-<pairs>-<seed>.json``
    (success rate, per-pair successes, the pairs)."""
    import stable_worldmodel as swm  # noqa: PLC0415
    import torch  # noqa: PLC0415

    kind, name = model.split(":", 1)
    out = EVAL / f"{kind}-{name.replace('/', '_')}-{pairs}-{seed}.json"
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
        cost_model, transform = our_cost_model(name, process), {}
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
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": model,
                "pairs": pairs,
                "seed": seed,
                "success_rate": float(metrics["success_rate"]),
                "episode_successes": np.asarray(metrics["episode_successes"]).astype(int).tolist(),
                "rows": rows.tolist(),
                "episodes": episodes.tolist(),
                "starts": starts.tolist(),
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


def lewm_curve(seed: int) -> dict[str, np.ndarray]:
    """``{clips, pred, reg}`` of a LeWM-tiny run (Lightning's ``metrics.csv``; ``reg`` is the
    weighted SIGReg term, as ours)."""
    import csv  # noqa: PLC0415

    import pusht_test as train_script  # noqa: PLC0415

    found = sorted((LEWM_LOGS / f"lewm-tiny-s{seed}").glob("version_*/metrics.csv"))
    if not found:
        return {}
    rows = [r for r in csv.DictReader(found[-1].open()) if r.get("train/pred_loss")]
    weight = 0.09  # LeWM's SIGReg weight
    return {
        "clips": np.array([int(r["step"]) for r in rows]) * train_script.BATCH,
        "pred": np.array([float(r["train/pred_loss"]) for r in rows]),
        "reg": np.array([weight * float(r["train/sigreg_loss"]) for r in rows]),
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


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=("plan", "pose", "report"))
    parser.add_argument("model", nargs="?", default="")
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    sys.path.insert(0, str(Path(__file__).parent))
    if args.command == "plan":
        plan(args.model, args.pairs, args.seed)
    elif args.command == "pose":
        pose(args.model)
    else:
        report()


if __name__ == "__main__":
    main()
