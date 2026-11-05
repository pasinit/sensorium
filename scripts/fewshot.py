"""The few-shot harness on the command line (``sensorium.fewshot``, ``bodies.h1_closed_loop``).

Usage:
    uv run python scripts/fewshot.py cache <out.npz> --demos 'data/tasks_demos/humanoid/*.npz'
        --encoder layer:<checkpoint.pt> | dino:<weights.pth> [--no-proprio] [--drop SENSE ...]
        [--horizon H] [--stride S]
    uv run python scripts/fewshot.py sweep <cache.npz> <out_dir> [--k 1 5 10 50] [--seeds 0 1 2]
    uv run python scripts/fewshot.py closed-loop <head.pt> <out.json> --config <demos.toml>
        --block <name> --encoder ... [--episodes N] [--steps 600]

- ``cache``: features and action chunks of the successful demonstrations.
  - The encoder is a pretrained layer's checkpoint (``model.pretrain.save_checkpoint``; its windows
    and senses), or the frozen DINOv2 baseline (``model.controls.load_dino``; the frame at each
    step, plus proprio unless ``--no-proprio``).
  - ``--horizon`` is the chunk length: the head is asked again after that many steps (default 4,
    0.2 s at 20 Hz).
- ``sweep``: for each k and seed, a head trained on k demonstrations drawn by the seed, scored by
  its action RMSE on the remaining demonstrations. Writes ``results.json`` (one row per k and seed,
  and a mean and spread per k) and the heads.
- ``closed-loop``: the head in the simulator on the config block's task (its settings as for the
  demonstrations). Episode ``i`` has seed ``generate.episode_seed(master_seed, "humanoid",
  "closed_loop", i, block)``: the demonstrations' seeds hash the policy ``experiment`` instead, so
  the start states are new. Writes each episode's success and the success rate with a Wilson 95%
  interval.
"""

import argparse
import json
import math
import tomllib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from sensorium import fewshot
from sensorium.bodies.h1_closed_loop import evaluate
from sensorium.bodies.humanoid import DT, HumanoidRecorder
from sensorium.generate import episode_seed
from sensorium.model.controls import load_dino
from sensorium.model.pretrain import expand, load_layer


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval of ``k`` successes in ``n``."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def make_encoder(spec: str, proprio: bool, drop: Sequence[str]) -> tuple[fewshot.FeatureFn, int]:
    """``(features, substeps)`` of ``layer:<checkpoint>`` or ``dino:<weights>``.

    Raises:
        ValueError: for another kind.
    """
    kind, _, path = spec.partition(":")
    if kind == "layer":
        layer = load_layer(Path(path))
        return fewshot.layer_features(layer, drop), int(layer.model.cfg.substeps)
    if kind == "dino":
        return fewshot.dino_features_fn(load_dino(Path(path)), proprio), 1
    raise ValueError(f"encoder must be layer:<checkpoint> or dino:<weights>, got {spec!r}")


def cache(args: argparse.Namespace) -> None:
    """Build and save the cache."""
    features, substeps = make_encoder(args.encoder, not args.no_proprio, args.drop)
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


def closed_loop(args: argparse.Namespace) -> None:
    """Run the head on the task and write the outcomes."""
    config = tomllib.loads(args.config.read_text())
    spec = config["humanoid"][args.block]
    features, substeps = make_encoder(args.encoder, not args.no_proprio, args.drop)
    head = fewshot.load_head(args.head)
    master = int(config["master_seed"])
    seeds = [
        episode_seed(master, "humanoid", "closed_loop", i, args.block) for i in range(args.episodes)
    ]
    with HumanoidRecorder(dict(spec["settings"])) as rec:
        first = fewshot.first_step(DT, substeps)
        outcomes = evaluate(rec, seeds, args.steps, features, head, first)
    k = sum(o.success for o in outcomes)
    low, high = wilson(k, len(outcomes))
    result = {
        "head": str(args.head),
        "block": args.block,
        "encoder": args.encoder,
        "episodes": [o.__dict__ for o in outcomes],
        "success": k,
        "n": len(outcomes),
        "rate": k / max(len(outcomes), 1),
        "wilson95": [low, high],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(f"{args.block}: {k}/{len(outcomes)} = {result['rate']:.0%} [{low:.0%}, {high:.0%}]")


def main() -> None:
    """Dispatch the subcommands."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    def encoder_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--encoder", required=True)
        p.add_argument("--no-proprio", action="store_true")
        p.add_argument("--drop", nargs="*", default=[])

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
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--steps", type=int, default=600)
    encoder_args(p)
    args = parser.parse_args()
    {"cache": cache, "sweep": sweep, "closed-loop": closed_loop}[args.command](args)


if __name__ == "__main__":
    main()
