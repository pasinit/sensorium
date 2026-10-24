"""The decisive held-out-episode probe test (Track B, ``docs/model.md``).

Does pretraining keep less transferable contact information than a random encoder only because
there are 8 episodes? Data: ``configs/probe_test.toml`` (120 seated babbling episodes of 600 steps,
60 hand + 60 pincer, generated with Track A's generator). The encoder is pretrained on 100 of
them (50 + 50) and **probed only on the 20 it never saw** (leave-one-episode-out inside that set);
the same probes on 20 pretraining episodes ("seen") show what generalisation costs.

    uv run python scripts/decisive_test.py split
    uv run python scripts/decisive_test.py train --objective ema --seed 0     # also: sigreg
    uv run python scripts/decisive_test.py probe                              # all checkpoints
    uv run python scripts/decisive_test.py report                             # tables + figure

Encoders: random initialisation (the floor, 2 seeds), SIGReg pooled over time and EMA + VICReg
(tiny model, batch 32, 3,000 steps, 2 seeds; checkpoints at 100 / 300 / 1,000 / 3,000), and the
raw non-visual inputs of the same windows as a reference. Outputs go to ``runs/decisive/``
(git-ignored).
"""

import argparse
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sensorium.model.diagnostics import heldout_loss, sense_reliance  # noqa: E402
from sensorium.model.pretrain import (  # noqa: E402
    RunConfig,
    TrainConfig,
    build_space,
    load_config,
    load_layer,
    train,
)
from sensorium.model.probes import run_probes  # noqa: E402
from sensorium.model.sensory_layer import SensoryLayer, SensoryModel  # noqa: E402

DATA = Path("data/probe_test/humanoid")
ROOT = Path("runs/decisive")
STEPS = (100, 300, 1000, 2000, 3000)
"""Steps at which a checkpoint is kept when the run is long enough."""
SEEDS = (0, 1)
OBJECTIVES = {
    "sigreg": {"objective": "sigreg", "sigreg_pool_time": True},
    "ema": {"objective": "ema_vicreg"},
    # An intervention for the diagnosis: drop each sense for a whole sequence with probability 0.5.
    "sigreg-sd": {"objective": "sigreg", "sigreg_pool_time": True, "sense_dropout": 0.5},
    "ema-sd": {"objective": "ema_vicreg", "sense_dropout": 0.5},
}
STRIDE = 8
ABLATIONS = {
    "touch": ("touch",),
    "vision": ("vision",),
    "proprio": ("proprio_body", "vestibular", "proprio_effector"),
    "audio": ("audio",),
}
TERMS = (
    "contact_left_auroc",
    "contact_right_auroc",
    "contact_object_auroc",
    "log_force_r2",
    "tip_r2",
)


def split() -> tuple[list[Path], list[Path], list[Path]]:
    """``(pretrain, heldout, seen)`` files: 50 + 50, 10 + 10, and 20 of the pretraining ones."""
    files = sorted(DATA.glob("*.npz"))
    rng = np.random.default_rng(0)
    heldout: list[Path] = []
    for effector in ("hand", "pincer"):
        group = [f for f in files if f"-{effector}_" in f.name]
        heldout += [group[i] for i in sorted(rng.choice(len(group), 10, replace=False))]
    pretrain = [f for f in files if f not in heldout]
    seen = [pretrain[i] for i in sorted(rng.choice(len(pretrain), 20, replace=False))]
    return pretrain, heldout, seen


def write_split() -> None:
    """Record the split (``runs/decisive/split.json``)."""
    pretrain, heldout, seen = split()
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "split.json").write_text(
        json.dumps(
            {
                k: [p.name for p in v]
                for k, v in {"pretrain": pretrain, "heldout": heldout, "seen": seen}.items()
            },
            indent=1,
        )
    )
    logging.info("pretrain %d, held out %d, seen-probe %d", len(pretrain), len(heldout), len(seen))


def train_run(objective: str, seed: int, steps: int, batch: int, workers: int) -> None:
    """Pretrain the tiny model on the pretraining episodes, keeping the numbered checkpoints."""
    pretrain, _, _ = split()
    base = load_config(Path("configs/model/cpu_tiny_a0.toml"))
    tc = TrainConfig(
        **{
            **asdict(base.train),
            "steps": steps,
            "batch_size": batch,
            "out_dir": str(ROOT),
            "log_every": 50,
            "eval_windows": 0,
            "num_workers": 4,
            "save_steps": tuple(s for s in STEPS if s <= steps),
        }
    )
    data = replace(base.data, paths=tuple(str(p) for p in pretrain))
    augment = replace(base.augment, cache_episodes=4)
    model = replace(base.model, **OBJECTIVES[objective])
    train(RunConfig(f"{objective}-s{seed}", seed, data, augment, model, tc))


def floor_layer(seed: int) -> SensoryLayer:
    """The tiny model at a random initialisation, on the pretraining episodes' channel space."""
    pretrain, _, _ = split()
    base = load_config(Path("configs/model/cpu_tiny_a0.toml"))
    torch.manual_seed(seed)
    space = build_space(pretrain, base.data.stats_episodes)
    return SensoryLayer(SensoryModel(base.model, space), space, base.loader())


def probe_layer(layer: SensoryLayer, final: bool) -> dict[str, object]:
    """Held-out and seen probes, label-free diagnostics, and (``final``) sense ablations."""
    _, heldout, seen = split()
    out: dict[str, object] = {
        "heldout": run_probes(layer, heldout, STRIDE, raw=False)["state"],
        "seen": run_probes(layer, seen, STRIDE, raw=False)["state"],
        **heldout_loss(layer, heldout),
        **sense_reliance(layer, heldout),
    }
    if final:
        out["ablate"] = {
            name: run_probes(layer, heldout, STRIDE, drop=drop, raw=False)["state"]
            for name, drop in ABLATIONS.items()
        }
    return out


def probe_all(runs: bool = True, final: int = 2000, baselines: bool = True) -> None:
    """Probe the raw inputs, the floors and (``runs``) every checkpoint; skips what is done."""
    out = ROOT / "probes"
    out.mkdir(parents=True, exist_ok=True)
    _, heldout, seen = split()
    if baselines and not (out / "raw.json").exists():
        raw = {
            "heldout": run_probes(floor_layer(0), heldout, STRIDE)["raw"],
            "seen": run_probes(floor_layer(0), seen, STRIDE)["raw"],
        }
        (out / "raw.json").write_text(json.dumps(raw))
    for seed in SEEDS:
        if baselines and not (out / f"floor-s{seed}.json").exists():
            (out / f"floor-s{seed}.json").write_text(
                json.dumps(probe_layer(floor_layer(seed), True))
            )
    for run in sorted(ROOT.glob("*-s?")) if runs else []:
        kept = [s for s in STEPS if (run / f"ckpt-{s}.pt").exists()]
        for step in kept:
            target = out / f"{run.name}-n{step}.json"
            if not target.exists():
                layer = load_layer(run / f"ckpt-{step}.pt")
                target.write_text(json.dumps(probe_layer(layer, step == final)))
                logging.info("probed %s", target.name)


def load_probes() -> dict[str, dict[str, object]]:
    """All probe results by name."""
    return {p.stem: json.loads(p.read_text()) for p in (ROOT / "probes").glob("*.json")}


def mean_sd(values: list[float]) -> str:
    """``mean +- sd`` of seed values."""
    return (
        f"{np.mean(values):.2f} +- {np.std(values):.2f}" if len(values) > 1 else f"{values[0]:.2f}"
    )


def report() -> None:
    """Print the tables and write the figure."""
    res = load_probes()
    steps = [s for s in STEPS if any(k.endswith(f"-n{s}") for k in res)]
    raw = res.pop("raw")
    floors = [res.pop(f"floor-s{s}") for s in SEEDS if f"floor-s{s}" in res]
    print(
        "Held-out probe mean (seen-episode probe mean in brackets); floor and raw for reference\n"
    )  # noqa: T201
    print("| encoder | " + " | ".join(f"{s} steps" for s in steps) + " |")  # noqa: T201
    print("|---|" + "---|" * len(steps))  # noqa: T201
    for name in OBJECTIVES:
        cells = []
        for step in steps:
            runs = [res[f"{name}-s{s}-n{step}"] for s in SEEDS if f"{name}-s{s}-n{step}" in res]
            if not runs:
                cells.append("-")
                continue
            cells.append(
                f"{mean_sd([r['heldout']['mean'] for r in runs])} "
                f"({mean_sd([r['seen']['mean'] for r in runs])})"
            )
        print(f"| {name} | " + " | ".join(cells) + " |")  # noqa: T201
    print(
        f"| floor | {mean_sd([f['heldout']['mean'] for f in floors])} "
        f"({mean_sd([f['seen']['mean'] for f in floors])}) |||"
    )  # noqa: T201
    print(f"| raw inputs | {raw['heldout']['mean']:.2f} ({raw['seen']['mean']:.2f}) |||")  # noqa: T201
    print("\nTerms, held-out, final checkpoint\n")  # noqa: T201
    print("| encoder | " + " | ".join(TERMS) + " |")  # noqa: T201
    print("|---|" + "---|" * len(TERMS))  # noqa: T201
    rows = {
        name: [
            res[f"{name}-s{s}-n{steps[-1]}"]["heldout"]
            for s in SEEDS
            if f"{name}-s{s}-n{steps[-1]}" in res
        ]
        for name in OBJECTIVES
    }
    rows["floor"] = [f["heldout"] for f in floors]
    for name, runs in rows.items():
        if not runs:
            continue
        print(f"| {name} | " + " | ".join(mean_sd([r[t] for r in runs]) for t in TERMS) + " |")  # noqa: T201
    print("| raw | " + " | ".join(f"{raw['heldout'][t]:.2f}" for t in TERMS) + " |")  # noqa: T201
    diagnostics(res, floors, steps)
    figure(res, floors, raw, steps)


def diagnostics(
    res: dict[str, dict[str, object]], floors: list[dict[str, object]], steps: list[int]
) -> None:
    """Held-out loss, sense reliance and sense-ablation tables."""
    print("\nHeld-out loss and sense reliance (mean over seeds)\n")  # noqa: T201
    senses = [k.removeprefix("reliance_") for k in floors[0] if k.startswith("reliance_")]
    print(
        "| encoder | steps | heldout pred | heldout reg | probe mean | "
        + " | ".join(f"rel {s}" for s in senses)
        + " |"
    )  # noqa: T201
    print("|---|---|---|---|---|" + "---|" * len(senses))  # noqa: T201
    groups = [("floor", 0, floors)]
    for name in OBJECTIVES:
        for step in steps:
            runs = [res[f"{name}-s{s}-n{step}"] for s in SEEDS if f"{name}-s{s}-n{step}" in res]
            if runs:
                groups.append((name, step, runs))
    for name, step, runs in groups:
        pred, reg = (np.mean([r[k] for r in runs]) for k in ("heldout_pred", "heldout_reg"))
        probe = np.mean([r["heldout"]["mean"] for r in runs])
        rel = " | ".join(f"{np.mean([r[f'reliance_{s}'] for r in runs]):.2f}" for s in senses)
        print(f"| {name} | {step} | {pred:.3f} | {reg:.3f} | {probe:.2f} | {rel} |")  # noqa: T201
    print("\nSense-ablated probe mean (held-out, final step; sense masked when encoding)\n")  # noqa: T201
    print("| encoder | none | " + " | ".join(f"-{a}" for a in ABLATIONS) + " |")  # noqa: T201
    print("|---|---|" + "---|" * len(ABLATIONS))  # noqa: T201
    for name, _step, runs in groups:
        if name != "floor" and not all("ablate" in r for r in runs):
            continue
        cells = [mean_sd([r["heldout"]["mean"] for r in runs])]
        cells += [mean_sd([r["ablate"][a]["mean"] for r in runs]) for a in ABLATIONS]
        print(f"| {name} | " + " | ".join(cells) + " |")  # noqa: T201


def figure(
    res: dict[str, dict[str, object]],
    floors: list[dict[str, object]],
    raw: dict[str, object],
    steps: list[int],
) -> None:
    """Probe mean against training steps for every encoder; floor and raw inputs drawn."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    colors = {
        "sigreg": "#1f77b4",
        "ema": "#d62728",
        "sigreg-sd": "#17becf",
        "ema-sd": "#e377c2",
    }
    for ax, key, title in zip(
        axes,
        ("heldout", "seen", "pred"),
        (
            "held-out episodes: probe mean",
            "seen episodes: probe mean",
            "held-out next-state error (own objective)",
        ),
        strict=True,
    ):
        for name, color in colors.items():
            for s in SEEDS:
                pts = [
                    (st, res[f"{name}-s{s}-n{st}"]) for st in steps if f"{name}-s{s}-n{st}" in res
                ]
                ys = [r["heldout_pred"] if key == "pred" else r[key]["mean"] for _, r in pts]
                ax.plot(
                    [p[0] for p in pts],
                    ys,
                    "o--" if name.endswith("-sd") else "o-",
                    color=color,
                    alpha=0.5,
                    label=name if s == SEEDS[0] else None,
                )
        if key != "pred":
            vals = [f[key]["mean"] for f in floors]
            ax.axhspan(min(vals), max(vals), color="gray", alpha=0.3, label="random init (2 seeds)")
            ax.axhline(raw[key]["mean"], color="k", ls=":", label="raw inputs")
        ax.set_xscale("log")
        ax.set_xlabel("training steps")
        ax.set_title(title, fontsize=9)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "probe_vs_steps.png", dpi=130)
    print(f"\nfigure: {ROOT / 'probe_vs_steps.png'}")  # noqa: T201


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=("split", "train", "probe", "report"))
    parser.add_argument("--objective", choices=sorted(OBJECTIVES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--runs-only", action="store_true", help="probe: skip raw and floors")
    parser.add_argument("--floors-only", action="store_true", help="probe: skip the trained runs")
    parser.add_argument("--workers", type=int, default=2, help="loader processes per training run")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.command == "split":
        write_split()
    elif args.command == "train":
        train_run(args.objective, args.seed, args.steps, args.batch_size, args.workers)
    elif args.command == "probe":
        probe_all(not args.floors_only, args.steps, not args.runs_only)
    else:
        report()


if __name__ == "__main__":
    main()
