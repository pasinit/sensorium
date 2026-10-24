"""Compare the SIGReg and EMA + VICReg objectives at CPU scale on the H1 sample.

One run per (variant, batch size, seed) of ``configs/model/cpu_tiny_a0.toml``; each writes
``runs/compare/<name>.json`` with the in-batch std and effective rank (mean of the last
four logged steps) and the eval-mode ones on a fixed sample of encoded windows (the
latter do not depend on the batch size). ``--table`` prints the aggregate as markdown.

    uv run python scripts/compare_objectives.py --variant ema-vicreg --seed 0 --batches 8 32
    uv run python scripts/compare_objectives.py --table
"""

import argparse
import json
import logging
import re
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from sensorium.model.pretrain import (
    RunConfig,
    TrainConfig,
    build_space,
    load_config,
    load_layer,
    train,
)
from sensorium.model.probes import run_probes
from sensorium.model.sensory_layer import SensoryLayer, SensoryModel

OUT = Path("runs/compare")
NO_VISION = ("touch", "audio", "proprio_body", "vestibular", "proprio_effector")
VARIANTS = {
    "sigreg-step": {"objective": "sigreg", "sigreg_pool_time": False},
    "sigreg-pooled": {"objective": "sigreg", "sigreg_pool_time": True},
    "ema-vicreg": {"objective": "ema_vicreg"},
    # No frames: does the per-episode look (colours, lights) let the model identify episodes?
    "sigreg-pooled-novis": {"objective": "sigreg", "sigreg_pool_time": True, "senses": NO_VISION},
    "ema-vicreg-novis": {"objective": "ema_vicreg", "senses": NO_VISION},
}


def run(variant: str, batch: int, seed: int, steps: int) -> None:
    """Train one configuration and write its summary."""
    base = load_config(Path("configs/model/cpu_tiny_a0.toml"))
    name = f"{variant}-b{batch}-s{seed}" + ("" if steps == 1000 else f"-n{steps}")
    train_cfg = TrainConfig(
        **{
            **asdict(base.train),
            "steps": steps,
            "batch_size": batch,
            "log_every": 50,
            "out_dir": str(OUT),
            "eval_windows": 512,
        }
    )
    cfg = RunConfig(
        name, seed, base.data, base.augment, replace(base.model, **VARIANTS[variant]), train_cfg
    )
    final = train(cfg)
    rows = [json.loads(line) for line in (OUT / name / "log.jsonl").read_text().splitlines()]
    tail = [r for r in rows if "erank" in r and "eval_erank" not in r][-4:]
    summary = {
        "variant": variant,
        "batch": batch,
        "seed": seed,
        "steps": steps,
        "loss_first": rows[0]["loss"],
        "loss_last": final["loss"],
        "std": float(np.mean([r["std"] for r in tail])),
        "erank": float(np.mean([r["erank"] for r in tail])),
        "eval_std": final["eval_std"],
        "eval_erank": final["eval_erank"],
        "seq_per_s": final["seq_per_s"],
    }
    (OUT / f"{name}.json").write_text(json.dumps(summary))


def probe_all() -> None:
    """Probe every trained run's checkpoint, and untrained encoders as the floor.

    Writes ``<name>.probes.json`` next to each run: the privileged-label probe scores of the
    encoder's states (``state``) and of the raw non-visual inputs of the same windows (``raw``).
    The floor is the same tiny model at its random initialisation, one per seed.
    """
    paths = sorted(Path("data/sample_h1/humanoid").glob("*.npz"))
    base = load_config(Path("configs/model/cpu_tiny_a0.toml"))
    layers = {p.parent.name: load_layer(p) for p in sorted(OUT.glob("*/ckpt.pt"))}
    for seed in (0, 1):
        for label, senses in (("floor", base.model.senses), ("floor-novis", NO_VISION)):
            torch.manual_seed(seed)
            space = build_space(paths, 64)
            model = SensoryModel(replace(base.model, senses=senses), space)
            layers[f"{label}-b0-s{seed}"] = SensoryLayer(model, space, base.loader())
    for name, layer in layers.items():
        if (OUT / f"{name}.probes.json").exists():
            continue
        result = run_probes(layer, paths)
        (OUT / f"{name}.probes.json").write_text(json.dumps(result))
        print(name, round(result["state"]["mean"], 3), round(result["raw"]["mean"], 3))  # noqa: T201


def table() -> None:
    """Print the mean over seeds of every (variant, batch) as a markdown table."""
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json")) if "probes" not in p.name]
    rows = [r for r in rows if r["steps"] == 1000]
    print(
        "| variant | batch | seeds | in-batch std | in-batch erank | eval std | eval erank "
        "| loss first -> last | seq/s |"
    )  # noqa: T201
    print("|---|---|---|---|---|---|---|---|---|")  # noqa: T201
    for variant in VARIANTS:
        for batch in sorted({r["batch"] for r in rows}):
            sel = [r for r in rows if r["variant"] == variant and r["batch"] == batch]
            if not sel:
                continue
            m = {k: np.mean([r[k] for r in sel]) for k in sel[0] if isinstance(sel[0][k], float)}
            print(  # noqa: T201
                f"| {variant} | {batch} | {len(sel)} | {m['std']:.2f} | {m['erank']:.1f} | "
                f"{m['eval_std']:.2f} | {m['eval_erank']:.1f} | "
                f"{m['loss_first']:.2f} -> {m['loss_last']:.2f} | {m['seq_per_s']:.0f} |"
            )


def probe_table() -> None:
    """Print probe scores (mean over seeds): trained variants, the random-init floor, raw inputs."""
    keys = [
        "contact_left_auroc",
        "contact_right_auroc",
        "contact_object_auroc",
        "log_force_r2",
        "tip_r2",
        "mean",
    ]
    results = {
        p.name.removesuffix(".probes.json"): json.loads(p.read_text())
        for p in OUT.glob("*.probes.json")
    }
    print("| encoder | batch | seeds | eval erank | " + " | ".join(keys) + " |")  # noqa: T201
    print("|---" * (4 + len(keys)) + "|")  # noqa: T201
    summaries = {
        p.stem: json.loads(p.read_text()) for p in OUT.glob("*-s?.json") if "probes" not in p.name
    }
    for variant in (*VARIANTS, "floor", "floor-novis"):
        for batch in (0, 8, 32):
            names = [n for n in results if re.fullmatch(rf"{variant}-b{batch}-s\d+", n)]
            if not names:
                continue
            vals = [[results[n]["state"][k] for k in keys] for n in names]
            ranks = [summaries[n]["eval_erank"] for n in names if n in summaries]
            erank = f"{np.mean(ranks):.1f}" if ranks else "-"
            print(
                f"| {variant} | {batch or '-'} | {len(names)} | {erank} | "
                + " | ".join(f"{v:.2f}" for v in np.nanmean(vals, axis=0))
                + " |"
            )  # noqa: T201
    raw = np.nanmean([[r["raw"][k] for k in keys] for r in results.values()], axis=0)
    print("| raw non-visual inputs | - | all | - | " + " | ".join(f"{v:.2f}" for v in raw) + " |")  # noqa: T201


def steps_table() -> None:
    """Probe mean and eval erank against the number of training steps (seed 0, batch 8)."""
    print("| objective | 0 (floor) | 100 | 300 | 1000 |")  # noqa: T201
    print("|---|---|---|---|---|")  # noqa: T201
    floor = json.loads((OUT / "floor-b0-s0.probes.json").read_text())["state"]["mean"]
    for variant in VARIANTS:
        cells = []
        for suffix in ("-n100", "-n300", ""):
            name = f"{variant}-b8-s0{suffix}"
            probe = json.loads((OUT / f"{name}.probes.json").read_text())["state"]["mean"]
            rank = json.loads((OUT / f"{name}.json").read_text())["eval_erank"]
            cells.append(f"{probe:.2f} (erank {rank:.0f})")
        print(f"| {variant} | {floor:.2f} | " + " | ".join(cells) + " |")  # noqa: T201


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batches", type=int, nargs="*", default=[8, 32])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--probe", action="store_true", help="probe all trained runs and the floor")
    parser.add_argument("--probe-table", action="store_true")
    parser.add_argument("--steps-table", action="store_true")
    args = parser.parse_args()
    if args.table:
        table()
        return
    if args.probe:
        probe_all()
        return
    if args.probe_table:
        probe_table()
        return
    if args.steps_table:
        steps_table()
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for batch in args.batches:
        run(args.variant, batch, args.seed, args.steps)


if __name__ == "__main__":
    main()
