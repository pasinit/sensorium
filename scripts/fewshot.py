"""The few-shot harness on the command line (``sensorium.fewshot``, ``bodies.h1_closed_loop``).

Usage:
    uv run python scripts/fewshot.py cache <out.npz> --demos 'data/tasks_demos/humanoid/*.npz'
        --encoder layer:<checkpoint.pt> | dino:<weights.pth> [--no-proprio] [--drop SENSE ...]
        [--recal-bn '<pool episodes glob>'] [--horizon H] [--stride S]
    uv run python scripts/fewshot.py sweep <cache.npz> <out_dir> [--k 1 5 10 50] [--seeds 0 1 2]
    uv run python scripts/fewshot.py closed-loop <head.pt> <out.json> --config <demos.toml>
        --block <name> --encoder ... [--first I] [--episodes N] [--steps 600]
    uv run python scripts/fewshot.py floor <out.json> --config <demos.toml> --block <name>
        --demos '<glob>' [--first I] [--episodes N] [--steps 600] [--no-render]
        [--kinds hold replay]

- ``cache``: features and action chunks of the successful demonstrations.
  - The encoder is a pretrained layer's checkpoint (``model.pretrain.save_checkpoint``; its windows
    and senses), or the frozen DINOv2 baseline (``model.controls.load_dino``; the frame at each
    step, plus proprio unless ``--no-proprio``).
  - ``--recal-bn``: a layer's BatchNorm statistics re-estimated on clean sequences of the given
    pool episodes first (``model.recalibrate``; B-pusht-0's BatchNorm rule). Pass the same value
    to ``closed-loop``, so the head runs on the features it was trained on.
  - ``--horizon`` is the chunk length: the head is asked again after that many steps (default 4,
    0.2 s at 20 Hz).
- ``sweep``: for each k and seed, a head trained on k demonstrations drawn by the seed, scored by
  its action RMSE on the remaining demonstrations. Writes ``results.json`` (one row per k and seed,
  and a mean and spread per k) and the heads.
- ``closed-loop``: the head in the simulator on the config block's task (its settings as for the
  demonstrations). Episode ``i`` has seed ``generate.episode_seed(master_seed, "humanoid",
  "closed_loop", i, block)``: the demonstrations' seeds hash the policy ``experiment`` instead, so
  the start states are new. Writes each episode's success and sub-goals (``milestones``) and the
  success rate with a Wilson 95% interval.
- ``floor``: the task's null-policy floors on the same start states. ``hold`` keeps the start
  posture; ``replay`` plays an unrelated demonstration's actions open loop (episode i replays the
  (i mod n)-th of the demonstrations in ``--demos``, recorded from other seeds). A task one of them
  solves cannot rank encoders. Both are open loop and judged on the physics state, so
  ``--no-render`` skips the cameras (``HumanoidRecorder.render_eyes``; the outcomes are the same).
  ``--kinds`` runs only some of them (a VM shard of one start state and one floor).

``closed-loop`` and ``floor`` print one line per finished episode (for a VM's stall rule).
``closed-loop`` renders the model's eye only: the second eye is privileged (``RestoredScene``).
"""

import argparse
import json
import math
import time
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from sensorium import fewshot
from sensorium.bodies.h1_closed_loop import Outcome, ReplayPolicy, evaluate, evaluate_with
from sensorium.bodies.humanoid import DT, HumanoidRecorder
from sensorium.episode import Episode
from sensorium.generate import episode_seed
from sensorium.model.controls import load_dino
from sensorium.model.pretrain import expand, load_layer
from sensorium.model.recalibrate import recalibrated


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval of ``k`` successes in ``n``."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def make_encoder(
    spec: str, proprio: bool, drop: Sequence[str], recal: str = ""
) -> tuple[fewshot.FeatureFn, int]:
    """``(features, substeps)`` of ``layer:<checkpoint>`` or ``dino:<weights>``. With ``recal`` (a
    glob of pool episodes), a layer's BatchNorm statistics are re-estimated on clean sequences of
    them first (``model.recalibrate``).

    Raises:
        ValueError: for another kind, or ``recal`` with DINOv2.
    """
    kind, _, path = spec.partition(":")
    if kind == "layer":
        layer = load_layer(Path(path))
        if recal:
            layer = recalibrated(layer, sorted(expand(recal)))
        return fewshot.layer_features(layer, drop), int(layer.model.cfg.substeps)
    if kind == "dino":
        if recal:
            raise ValueError("--recal-bn applies to a sensory layer only")
        return fewshot.dino_features_fn(load_dino(Path(path)), proprio), 1
    raise ValueError(f"encoder must be layer:<checkpoint> or dino:<weights>, got {spec!r}")


def cache(args: argparse.Namespace) -> None:
    """Build and save the cache."""
    features, substeps = make_encoder(args.encoder, not args.no_proprio, args.drop, args.recal_bn)
    paths = sorted(expand(args.demos))
    built = fewshot.build_cache(paths, features, args.horizon, substeps, args.stride)
    built.save(args.out)
    print(f"{len(built.sources)} successful demonstrations of {len(paths)}; {len(built.ends)} rows")


def sweep(args: argparse.Namespace) -> None:
    """Train and score heads for every k and seed."""
    data = fewshot.DemoCache.load(args.cache)
    cfg = fewshot.BCConfig()
    n = len(data.sources)
    rows = []
    for k in args.k:
        if k >= n:
            print(f"k = {k}: only {n} demonstrations (one must be left to score), skipped")
            continue
        for seed in args.seeds:
            demos = fewshot.choose_demos(n, k, seed)
            held = [i for i in range(n) if i not in demos]
            head = fewshot.train_head(data, demos, cfg, seed)
            rmse = fewshot.action_rmse(head, data, held)
            path = args.out_dir / f"head_k{k}_s{seed}.pt"
            fewshot.save_head(head, cfg, path, {"cache": str(args.cache), "k": k, "seed": seed})
            rows.append({"k": k, "seed": seed, "demos": demos, "rmse": rmse, "head": str(path)})
            print(f"k {k} seed {seed}: action RMSE {rmse:.4f} on {len(held)} demonstrations")
    summary = {}
    for k in sorted({r["k"] for r in rows}):
        values = np.array([r["rmse"] for r in rows if r["k"] == k])
        summary[str(k)] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "n": len(values),
        }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(
        json.dumps({"rows": rows, "by_k": summary}, indent=1)
    )


def eval_seeds(config: dict[str, object], block: str, episodes: int, first: int = 0) -> list[int]:
    """Closed-loop start states ``first .. first + episodes`` of a block: seeds hashing the policy
    ``closed_loop`` (a shard of the 50 when ``first`` > 0)."""
    master = int(config["master_seed"])  # type: ignore[call-overload]
    return [
        episode_seed(master, "humanoid", "closed_loop", i, block)
        for i in range(first, first + episodes)
    ]


def progress(label: str) -> Callable[[Outcome], None]:
    """An ``on_episode`` callback: print one line per finished episode, with its wall time."""
    last = [time.monotonic()]

    def report(o: Outcome) -> None:
        now = time.monotonic()
        print(
            f"{label} episode seed {o.seed}: success {o.success}, {o.steps} steps, "
            f"{now - last[0]:.0f} s",
            flush=True,
        )
        last[0] = now

    return report


def summary(outcomes: Sequence[Outcome]) -> dict[str, object]:
    """Success count, rate, Wilson interval and sub-goal counts of some outcomes."""
    k = sum(o.success for o in outcomes)
    low, high = wilson(k, len(outcomes))
    reached: dict[str, int] = {}
    for o in outcomes:
        for name, value in o.milestones.items():
            reached[name] = reached.get(name, 0) + int(value)
    return {
        "episodes": [o.__dict__ for o in outcomes],
        "success": k,
        "n": len(outcomes),
        "rate": k / max(len(outcomes), 1),
        "wilson95": [low, high],
        "milestones": reached,
    }


def floor(args: argparse.Namespace) -> None:
    """Run the null-policy floors (``--kinds``, both by default) and write their outcomes."""
    config = tomllib.loads(args.config.read_text())
    spec = config["humanoid"][args.block]
    seeds = eval_seeds(config, args.block, args.episodes, args.first)
    demos = []
    if "replay" in args.kinds:
        demos = [Episode.load(p, privileged=False).action for p in sorted(expand(args.demos))]
        if not demos:
            raise SystemExit(f"no demonstrations match {args.demos}")
    result: dict[str, object] = {"block": args.block, "rendered": not args.no_render}
    with HumanoidRecorder(dict(spec["settings"])) as rec:
        if args.no_render:
            rec.render_eyes = False
        if "hold" in args.kinds:
            result["hold"] = summary(
                evaluate_with(
                    rec,
                    seeds,
                    args.steps,
                    lambda judge: ReplayPolicy(judge),
                    progress(f"{args.block} hold"),
                )
            )
        if "replay" in args.kinds:
            replayed = []
            report = progress(f"{args.block} replay")
            for i, seed in enumerate(seeds, start=args.first):
                actions = demos[i % len(demos)]
                replayed += evaluate_with(
                    rec, [seed], args.steps, lambda judge, a=actions: ReplayPolicy(judge, a), report
                )
            result["replay"] = summary(replayed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    for name in args.kinds:
        r = result[name]
        assert isinstance(r, dict)
        print(f"{args.block} floor {name}: {r['success']}/{r['n']} = {r['rate']:.0%}")


def closed_loop(args: argparse.Namespace) -> None:
    """Run the head on the task and write the outcomes."""
    config = tomllib.loads(args.config.read_text())
    spec = config["humanoid"][args.block]
    features, substeps = make_encoder(args.encoder, not args.no_proprio, args.drop, args.recal_bn)
    head = fewshot.load_head(args.head)
    seeds = eval_seeds(config, args.block, args.episodes, args.first)
    with HumanoidRecorder(dict(spec["settings"])) as rec:
        rec.render_right = False  # the privileged second eye: never a model input, half the render
        first = fewshot.first_step(DT, substeps)
        outcomes = evaluate(
            rec, seeds, args.steps, features, head, first, progress(f"{args.block} closed-loop")
        )
    result = {"head": str(args.head), "block": args.block, "encoder": args.encoder}
    result |= summary(outcomes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(f"{args.block}: {result['success']}/{result['n']} = {result['rate']:.0%}")


def main() -> None:
    """Dispatch the subcommands."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    def encoder_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--encoder", required=True)
        p.add_argument("--no-proprio", action="store_true")
        p.add_argument("--drop", nargs="*", default=[])
        p.add_argument("--recal-bn", default="", help="glob of pool episodes: re-estimate BN first")

    p = sub.add_parser("cache")
    p.add_argument("out", type=Path)
    p.add_argument("--demos", required=True)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--stride", type=int, default=1)
    encoder_args(p)
    p = sub.add_parser("sweep")
    p.add_argument("cache", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 50])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p = sub.add_parser("closed-loop")
    p.add_argument("head", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--block", required=True)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--steps", type=int, default=600)
    encoder_args(p)
    p = sub.add_parser("floor")
    p.add_argument("out", type=Path)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--block", required=True)
    p.add_argument("--demos", required=True)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--no-render", action="store_true", help="skip the cameras (open loop only)")
    p.add_argument("--kinds", nargs="+", choices=["hold", "replay"], default=["hold", "replay"])
    args = parser.parse_args()
    {"cache": cache, "sweep": sweep, "closed-loop": closed_loop, "floor": floor}[args.command](args)


if __name__ == "__main__":
    main()
