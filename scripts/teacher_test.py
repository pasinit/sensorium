"""B-teach-0: is touch an *input* or a *teacher*? A probe-level test on the 120-episode data.

Same episodes and split as ``scripts/decisive_test.py`` (``runs/decisive/split.json``: pretrain on
100, probe the 20 never seen), same tiny model, pooled SIGReg, batch 16, 2,000 steps. Arms:

* ``sd-all`` / ``sd-touch``: all six senses, sense dropout 0.5 on every sense / on touch only
  (confirming or killing PR #18's dropout lead; 5 and 2 seeds);
* ``a-input``: vision + touch + proprio, touch dropped 50% of sequences (touch as input and target);
* ``b-teacher``: vision + proprio only; an auxiliary head predicts the touch latent of an EMA touch
  encoder (touch is never an input of the state encoder);
* ``c-none``: vision + proprio only, no touch anywhere;
* ``d-placebo``: as ``b-teacher`` but the touch comes from another episode of the batch.

Every arm is probed on its **vision + proprio state** (touch masked or absent at probe time), on
the held-out episodes only, against the random-init floor.

    uv run python scripts/teacher_test.py train --arm b-teacher --seed 0
    uv run python scripts/teacher_test.py probe
    uv run python scripts/teacher_test.py report
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

from sensorium.model.diagnostics import sense_reliance, touch_latent_r2  # noqa: E402
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
SPLIT = Path("runs/decisive/split.json")
ROOT = Path("runs/teacher")
STRIDE = 8
STEPS = 2000
NO_AUDIO = ("vision", "touch", "proprio_body", "vestibular", "proprio_effector")
NO_TOUCH = ("vision", "proprio_body", "vestibular", "proprio_effector")
POOLED = {"objective": "sigreg", "sigreg_pool_time": True}
ARMS: dict[str, dict[str, object]] = {
    "sd-all": POOLED | {"sense_dropout": 0.5},
    "sd-touch": POOLED | {"sense_dropout": 0.5, "sense_dropout_senses": ("touch",)},
    "a-input": POOLED
    | {"senses": NO_AUDIO, "sense_dropout": 0.5, "sense_dropout_senses": ("touch",)},
    "b-teacher": POOLED | {"senses": NO_TOUCH, "teacher": "touch"},
    "c-none": POOLED | {"senses": NO_TOUCH},
    "d-placebo": POOLED | {"senses": NO_TOUCH, "teacher": "touch_placebo"},
}
FULL_ARMS = ("sd-all", "sd-touch")
"""Arms with all six senses, probed with and without touch (and audio) present."""
TERMS = (
    "contact_left_auroc",
    "contact_right_auroc",
    "contact_object_auroc",
    "log_force_r2",
    "tip_r2",
)


def files(key: str) -> list[Path]:
    """Episode files of a split (``pretrain`` or ``heldout``) from ``decisive_test.py split``."""
    return [DATA / name for name in json.loads(SPLIT.read_text())[key]]


def train_arm(arm: str, seed: int, steps: int, batch: int, workers: int) -> None:
    """Pretrain one arm on the 100 pretraining episodes; checkpoints at 1,000 and the end."""
    base = load_config(Path("configs/model/cpu_tiny.toml"))
    tc = TrainConfig(
        **{
            **asdict(base.train),
            "steps": steps,
            "batch_size": batch,
            "out_dir": str(ROOT),
            "log_every": 50,
            "eval_windows": 0,
            "num_workers": workers,
            "save_steps": tuple(s for s in (1000, steps) if s <= steps),
        }
    )
    data = replace(base.data, paths=tuple(str(p) for p in files("pretrain")))
    model = replace(base.model, **ARMS[arm])
    train(
        RunConfig(f"{arm}-s{seed}", seed, data, replace(base.augment, cache_episodes=4), model, tc)
    )


def floor_layer(seed: int, senses: tuple[str, ...]) -> SensoryLayer:
    """The arm's encoder at a random initialisation."""
    base = load_config(Path("configs/model/cpu_tiny.toml"))
    torch.manual_seed(seed)
    space = build_space(files("pretrain"), base.data.stats_episodes)
    model = SensoryModel(replace(base.model, senses=senses), space)
    return SensoryLayer(model, space, base.loader())


def probe_layer(layer: SensoryLayer, full: bool) -> dict[str, object]:
    """Probes of the vision + proprio state (touch and audio masked), and for full-sense arms also
    of the state with every sense present; plus the sense-reliance profile."""
    heldout = files("heldout")
    masks = {"vp": tuple(s for s in ("touch", "audio") if s in layer.model.senses)}
    if full:
        masks["all"] = ()
    out: dict[str, object] = {
        name: run_probes(layer, heldout, STRIDE, drop=drop, raw=False)["state"]
        for name, drop in masks.items()
    }
    out |= sense_reliance(layer, heldout)
    out |= touch_latent_r2(layer, heldout)
    return out


def probe_all(refresh_r2: bool = False) -> None:
    """Probe the floors and every checkpoint (skips what is done)."""
    out = ROOT / "probes"
    out.mkdir(parents=True, exist_ok=True)
    for senses_name, senses in (
        ("vp", NO_TOUCH),
        ("all", load_config(Path("configs/model/cpu_tiny.toml")).model.senses),
    ):
        for seed in range(5):
            target = out / f"floor-{senses_name}-s{seed}.json"
            if not target.exists():
                layer = floor_layer(seed, senses)
                target.write_text(json.dumps(probe_layer(layer, senses_name == "all")))
    for run in sorted(ROOT.glob("*-s?")):
        arm = run.name.rsplit("-s", 1)[0]
        for step in (1000, STEPS):
            ckpt, target = run / f"ckpt-{step}.pt", out / f"{run.name}-n{step}.json"
            if ckpt.exists() and not target.exists():
                target.write_text(json.dumps(probe_layer(load_layer(ckpt), arm in FULL_ARMS)))
                logging.info("probed %s", target.name)
            elif ckpt.exists() and arm.endswith(("teacher", "placebo")):
                done = json.loads(target.read_text())
                if refresh_r2 or "touch_r2_now" not in done:
                    done |= touch_latent_r2(load_layer(ckpt), files("heldout"))
                    target.write_text(json.dumps(done))


def mean_sd(values: list[float]) -> str:
    """``mean +- sd`` of seed values."""
    arr = np.array(values)
    return f"{arr.mean():.2f} +- {arr.std():.2f}" if len(arr) > 1 else f"{arr[0]:.2f}"


def runs_of(res: dict[str, dict[str, object]], arm: str, step: int) -> list[dict[str, object]]:
    """Probe results of an arm's seeds at a step."""
    return [v for k, v in res.items() if k.startswith(f"{arm}-s") and k.endswith(f"-n{step}")]


def report() -> None:
    """Print the 2 x 2 table, the contact terms, the controls and the reliance profile."""
    res = {p.stem: json.loads(p.read_text()) for p in (ROOT / "probes").glob("*.json")}
    floors = {k: v for k, v in res.items() if k.startswith("floor")}
    print("Held-out probe mean of the vision + proprio state (touch and audio masked)\n")  # noqa: T201
    print("| arm | seeds | step 1000 | step 2000 | " + " | ".join(TERMS) + " |")  # noqa: T201
    print("|---|---|---|---|" + "---|" * len(TERMS))  # noqa: T201
    for arm in ARMS:
        end, mid = runs_of(res, arm, STEPS), runs_of(res, arm, 1000)
        if end:
            print(table_row(arm, end, mid))  # noqa: T201
    for name, key in (("floor (V+P)", "floor-vp"), ("floor (all senses, V+P mask)", "floor-all")):
        runs = [v for k, v in floors.items() if k.startswith(key)]
        if runs:
            print(table_row(name, runs, []))  # noqa: T201
    print("\nPer-seed probe mean at the end (vision + proprio state)\n")  # noqa: T201
    for arm in ARMS:
        vals = sorted(
            (k.rsplit("-n", 1)[0].rsplit("-s", 1)[1], round(v["vp"]["mean"], 3))
            for k, v in res.items()
            if k.startswith(f"{arm}-s") and k.endswith(f"-n{STEPS}")
        )
        if vals:
            print(f"{arm}: " + ", ".join(f"s{s}={m}" for s, m in vals))  # noqa: T201
    contrasts(res, floors)
    teacher_use(res)
    controls_and_reliance(res)
    figure(res, floors)


def contrasts(res: dict[str, dict[str, object]], floors: dict[str, dict[str, object]]) -> None:
    """Differences of end-of-run probe means between arms, with the standard error of each."""
    groups = {arm: [r["vp"] for r in runs_of(res, arm, STEPS)] for arm in ARMS}
    groups["floor"] = [v["vp"] for k, v in floors.items() if k.startswith("floor-vp")]
    print("\nContrasts (difference of mean +- standard error of the difference, ddof = 1)\n")  # noqa: T201
    print("| contrast | probe mean | " + " | ".join(TERMS) + " |")  # noqa: T201
    print("|---|---|" + "---|" * len(TERMS))  # noqa: T201
    for first, second in (
        ("b-teacher", "c-none"),
        ("b-teacher", "d-placebo"),
        ("d-placebo", "c-none"),
        ("a-input", "c-none"),
        ("b-teacher", "floor"),
        ("c-none", "floor"),
        ("sd-all", "floor"),
    ):
        x, y = groups[first], groups[second]
        if len(x) < 2 or len(y) < 2:
            continue
        cells = []
        for key in ("mean", *TERMS):
            a, b = np.array([r[key] for r in x]), np.array([r[key] for r in y])
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            cells.append(f"{a.mean() - b.mean():+.2f} +- {se:.2f}")
        print(f"| {first} - {second} (n = {len(x)}, {len(y)}) | " + " | ".join(cells) + " |")  # noqa: T201


def teacher_use(res: dict[str, dict[str, object]]) -> None:
    """Does the state predict the real touch latents (held out)? 0 = no better than their mean."""
    print("\nTeacher use: R^2 of the aux heads against the true touch latents, held out\n")  # noqa: T201
    for arm in ("b-teacher", "d-placebo"):
        runs = [r for r in runs_of(res, arm, STEPS) if "touch_r2_now" in r]
        if runs:
            print(  # noqa: T201
                f"{arm}: from the state {mean_sd([r['touch_r2_now'] for r in runs])}, "
                f"next touch from the predictor {mean_sd([r['touch_r2_next'] for r in runs])}"
            )


def table_row(name: str, end: list[dict[str, object]], mid: list[dict[str, object]]) -> str:
    """One table row: mean +- sd of the probe mean (step 1000, end) and of every term."""
    first = mean_sd([r["vp"]["mean"] for r in mid]) if mid else "-"
    cells = [mean_sd([r["vp"][t] for r in end]) for t in TERMS]
    return (
        f"| {name} | {len(end)} | {first} | {mean_sd([r['vp']['mean'] for r in end])} | "
        + " | ".join(cells)
        + " |"
    )


def controls_and_reliance(res: dict[str, dict[str, object]]) -> None:
    """Positive control of the full-sense arms, and the sense-reliance profile."""
    print("\nFull-sense arms: probe mean with every sense present (positive control)\n")  # noqa: T201
    for arm in FULL_ARMS:
        runs = runs_of(res, arm, STEPS)
        if runs:
            print(  # noqa: T201
                f"{arm}: all senses {mean_sd([r['all']['mean'] for r in runs])}, "
                f"vision + proprio {mean_sd([r['vp']['mean'] for r in runs])}"
            )
    print("\nSense reliance at the end (mean over seeds)\n")  # noqa: T201
    senses = ["vision", "touch", "audio", "proprio_body", "vestibular", "proprio_effector"]
    print("| arm | " + " | ".join(senses) + " |")  # noqa: T201
    print("|---|" + "---|" * len(senses))  # noqa: T201
    for arm in ARMS:
        runs = runs_of(res, arm, STEPS)
        if runs:
            cells = [
                f"{np.mean([r[f'reliance_{s}'] for r in runs]):.2f}"
                if f"reliance_{s}" in runs[0]
                else "-"
                for s in senses
            ]
            print(f"| {arm} | " + " | ".join(cells) + " |")  # noqa: T201


def figure(res: dict[str, dict[str, object]], floors: dict[str, dict[str, object]]) -> None:
    """Final-step probe mean per arm (every seed a dot) with the floor band."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, arm in enumerate(ARMS):
        vals = [r["vp"]["mean"] for r in runs_of(res, arm, STEPS)]
        ax.plot([i] * len(vals), vals, "o", alpha=0.6)
        if vals:
            ax.plot([i - 0.25, i + 0.25], [np.mean(vals)] * 2, "k-")
    vp = [v["vp"]["mean"] for k, v in floors.items() if k.startswith("floor-vp")]
    if vp:
        ax.axhspan(min(vp), max(vp), color="gray", alpha=0.3, label="random init (V+P)")
    ax.set_xticks(range(len(ARMS)), list(ARMS), rotation=20, fontsize=8)
    ax.set_ylabel("held-out probe mean (V+P state)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "teacher_arms.png", dpi=130)
    print(f"\nfigure: {ROOT / 'teacher_arms.png'}")  # noqa: T201


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=("train", "probe", "report"))
    parser.add_argument("--arm", choices=sorted(ARMS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--refresh-r2", action="store_true", help="probe: recompute touch R^2")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.command == "train":
        train_arm(args.arm, args.seed, args.steps, args.batch_size, args.workers)
    elif args.command == "probe":
        probe_all(args.refresh_r2)
    else:
        report()


if __name__ == "__main__":
    main()
