"""Consequence-rich subsets and objective variants against the frozen probe (B-conseq-0).

Same 120 episodes and split as ``scripts/decisive_test.py`` (pretrain on 100, probe the 20 never
seen), tiny model at 64 px, pooled SIGReg, batch 16, **1,000 steps** (about 16,000 sequences seen),
checkpoints and probes every 250 steps. Arms:

* ``sub-rand``: V+P, sequences drawn uniformly (the baseline);
* ``sub-conseq``: V+P, only sequences where hand-object contact moves the object (>= 1 cm within
  0.5 s) in at least 25% of their steps; ``sub-contact``: only sequences with hand-object contact
  but little consequence (the control). The object motion picks sequences offline, never an input;
* ``c-pdrop``: V+P, all proprio groups dropped together in 50% of the sequences;
* ``c-err``: V+P, next-state loss weighted by the model's own prediction error;
* ``a-base`` / ``a-touch``: all six senses, plain / loss weighted by the change of the touch input;
* ``a-both``: all six senses and a touch teacher: touch is both an input and a prediction target
  (the H1 role grid's input + target cell; ``model.teacher = touch`` with touch in ``senses``).

    uv run python scripts/objective_test.py index
    uv run python scripts/objective_test.py train --arm sub-conseq --seed 0
    uv run python scripts/objective_test.py probe
    uv run python scripts/objective_test.py report
"""

import argparse
import json
import logging
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np

from sensorium.model.diagnostics import heldout_loss, state_geometry, touch_latent_r2
from sensorium.model.pretrain import RunConfig, TrainConfig, load_config, load_layer, train
from sensorium.model.probes import collect, predict_probes
from sensorium.model.subsets import build_index, save_index, summarise

DATA = Path("data/probe_test/humanoid")
SPLIT = Path("runs/decisive/split.json")
ROOT = Path("runs/objective")
INDEX = ROOT / "index.json"
STEPS = 1000
CKPTS = (0, 50, 100, 250, 500, 750, 1000, 2500, 5000, 10000)
LONG_STEPS = (0, 250, 1000, 2500, 5000, 10000)
LONG_ARMS = ("long-jepa", "long-mae", "long-teach", "long-placebo")
SHORT_STEPS = (0, 50, 100, 250, 500, 750, 1000)
STRIDE = 8
NO_TOUCH = ("vision", "proprio_body", "vestibular", "proprio_effector")
ALL = None  # every sense of the tiny config
ARMS: dict[str, dict[str, object]] = {
    "sub-rand": {"senses": NO_TOUCH},
    "sub-conseq": {"senses": NO_TOUCH},
    "sub-contact": {"senses": NO_TOUCH},
    "c-pdrop": {"senses": NO_TOUCH, "proprio_dropout": 0.5},
    "c-err": {"senses": NO_TOUCH, "surprise": "error"},
    "a-base": {},
    "a-touch": {"surprise": "touch"},
    "a-both": {"teacher": "touch"},
    "ref-jepa": {"senses": NO_TOUCH},
    "ref-mae": {"senses": NO_TOUCH, "objective": "mae"},
    "ref-nextobs": {"senses": NO_TOUCH, "objective": "nextobs"},
    "long-jepa": {"senses": NO_TOUCH},
    "long-mae": {"senses": NO_TOUCH, "objective": "mae"},
    "long-teach": {"senses": NO_TOUCH, "teacher": "touch"},
    "long-placebo": {"senses": NO_TOUCH, "teacher": "touch_placebo"},
}
SUBSET = {"sub-conseq": "conseq", "sub-contact": "contact"}
ALL_SENSE_ARMS = ("a-base", "a-touch", "a-both")
TERMS = (
    "contact_left_auroc",
    "contact_right_auroc",
    "contact_object_auroc",
    "log_force_r2",
    "tip_r2",
)


def files(key: str) -> list[Path]:
    """Episode files of a split from ``decisive_test.py split``."""
    return [DATA / n for n in json.loads(SPLIT.read_text())[key]]


def make_index() -> None:
    """Build the subset index over the pretraining episodes (reads privileged object motion)."""
    cfg = load_config(Path("configs/model/cpu_tiny.toml"))
    span = cfg.model.steps * cfg.model.substeps
    index = build_index(files("pretrain"), span)
    save_index(index, INDEX)
    logging.info("index: %s", summarise(index))


def train_arm(arm: str, seed: int, steps: int, workers: int) -> None:
    """Pretrain one arm; checkpoints every 250 steps."""
    base = load_config(Path("configs/model/cpu_tiny.toml"))
    tc = TrainConfig(
        steps=steps,
        batch_size=16,
        lr=base.train.lr,
        log_every=50,
        eval_windows=0,
        num_workers=workers,
        out_dir=str(ROOT),
        save_steps=tuple(
            s for s in (LONG_STEPS if arm in LONG_ARMS else SHORT_STEPS) if s <= steps
        ),
    )
    data = replace(base.data, paths=tuple(str(p) for p in files("pretrain")))
    if arm in SUBSET:
        data = replace(data, subset_index=str(INDEX), subset_key=SUBSET[arm])
    model = replace(base.model, **ARMS[arm])
    train(
        RunConfig(f"{arm}-s{seed}", seed, data, replace(base.augment, cache_episodes=4), model, tc)
    )


def probe_json(arm: str, ckpt: Path) -> dict[str, object]:
    """Held-out probe scores of one checkpoint, its held-out loss and the geometry of its states.

    ``heldout`` is the state with the arm's natural inputs (V+P for V+P arms, every sense for the
    all-sense arms); ``vp`` masks touch and audio; ``geometry`` is the effective rank and std of
    the ``heldout`` states.
    """
    layer = load_layer(ckpt)
    masked = tuple(s for s in ("touch", "audio") if s in layer.model.senses)
    natural = () if arm in ALL_SENSE_ARMS else masked
    out: dict[str, object] = {}
    seen: dict[tuple[str, ...], tuple[dict[str, float], dict[str, float]]] = {}
    for key, drop in (("vp", masked), ("heldout", natural)):
        if drop not in seen:
            data = collect(layer, files("heldout"), STRIDE, drop, raw=False)
            seen[drop] = (
                predict_probes(data, "state").scores(),
                state_geometry(data.features["state"]),
            )
        out[key] = seen[drop][0]
    out["geometry"] = seen[natural][1]
    return out | heldout_loss(layer, files("heldout")) | touch_latent_r2(layer, files("heldout"))


def probe_all() -> None:
    """Probe every checkpoint (skips what exists) and copy the floors of ``runs/controls``."""
    out = ROOT / "probes"
    out.mkdir(parents=True, exist_ok=True)
    for path in Path("runs/controls/probes_json").glob("floor-*-s?.json"):
        if not (out / path.name).exists():
            shutil.copy(path, out / path.name)
    for run in sorted(ROOT.glob("*-s?")):
        arm = run.name.rsplit("-s", 1)[0]
        for step in CKPTS:
            ckpt, target = run / f"ckpt-{step}.pt", out / f"{run.name}-n{step}.json"
            if ckpt.exists() and not target.exists():
                target.write_text(json.dumps(probe_json(arm, ckpt)))
                logging.info("probed %s", target.name)


def mean_sd(values: list[float]) -> str:
    """``mean +- sd`` over seeds (the single value if there is one)."""
    arr = np.array(values)
    return f"{arr.mean():.2f} +- {arr.std():.2f}" if len(arr) > 1 else f"{arr[0]:.2f}"


def load_probes() -> dict[str, dict[str, object]]:
    """All probe JSONs by file stem."""
    return {p.stem: json.loads(p.read_text()) for p in (ROOT / "probes").glob("*.json")}


def geometry_tables(res: dict[str, dict[str, object]]) -> None:
    """Effective rank and std of the held-out states against steps, for arms that logged them."""
    steps = SHORT_STEPS
    for term, label in (("erank", "effective rank"), ("std", "per-dimension std")):
        print(f"\nGeometry of the held-out states: {label} (mean over seeds)\n")  # noqa: T201
        print("| arm | " + " | ".join(f"{s} steps" for s in steps) + " |")  # noqa: T201
        print("|---|" + "---|" * len(steps))  # noqa: T201
        for arm in ARMS:
            cells = []
            for step in steps:
                vals = [
                    v["geometry"][term]
                    for k, v in res.items()
                    if k.startswith(f"{arm}-s") and k.endswith(f"-n{step}") and "geometry" in v
                ]
                cells.append(f"{np.mean(vals):.1f}" if vals else "-")
            if any(c != "-" for c in cells):
                print(f"| {arm} | " + " | ".join(cells) + " |")  # noqa: T201


def report() -> None:
    """Probe mean against steps for every arm, the terms at 1,000 steps, and paired contrasts."""
    res = load_probes()
    print("Held-out probe mean (mean +- sd over seeds; n in brackets) against steps\n")  # noqa: T201
    print("| arm | " + " | ".join(f"{s} steps" for s in SHORT_STEPS) + " |")  # noqa: T201
    print("|---|" + "---|" * len(SHORT_STEPS))  # noqa: T201
    for arm in ARMS:
        cells = []
        for step in SHORT_STEPS:
            runs = [
                v for k, v in res.items() if k.startswith(f"{arm}-s") and k.endswith(f"-n{step}")
            ]
            cells.append(
                f"{mean_sd([r['heldout']['mean'] for r in runs])} ({len(runs)})" if runs else "-"
            )
        print(f"| {arm} | " + " | ".join(cells) + " |")  # noqa: T201
    for name, key in (("floor V+P", "floor-vp"), ("floor all senses", "floor-all")):
        vals = [v["heldout"]["mean"] for k, v in res.items() if k.startswith(key + "-s")]
        if vals:
            print(f"| {name} (n {len(vals)}) | {mean_sd(vals)} | | | |")  # noqa: T201
    geometry_tables(res)
    print("\nTerms at 1,000 steps\n")  # noqa: T201
    print("| arm | n | " + " | ".join(TERMS) + " | held-out pred loss |")  # noqa: T201
    print("|---|---|" + "---|" * (len(TERMS) + 1))  # noqa: T201
    for arm in ARMS:
        runs = [v for k, v in res.items() if k.startswith(f"{arm}-s") and k.endswith("-n1000")]
        if runs:
            cells = [mean_sd([r["heldout"][t] for r in runs]) for t in TERMS]
            print(
                f"| {arm} | {len(runs)} | "
                + " | ".join(cells)  # noqa: T201
                + f" | {mean_sd([r['heldout_pred'] for r in runs])} |"
            )


def run_losses(arm: str) -> dict[int, float]:
    """Training ``pred`` loss of the 100 steps before each checkpoint, mean over seeds."""
    rows = []
    for log in sorted(ROOT.glob(f"{arm}-s?/log.jsonl")):
        rows.append([json.loads(line) for line in log.read_text().splitlines()])
    out = {}
    for step in LONG_STEPS[1:]:
        vals = [np.mean([r["pred"] for r in run if step - 100 < r["step"] <= step]) for run in rows]
        out[step] = float(np.mean(vals)) if vals else float("nan")
    return out


Results = dict[str, dict[str, object]]


def run_rows(res: Results, arm: str, step: int) -> list[dict[str, object]]:
    """Probe JSONs of every seed of ``arm`` at ``step``."""
    return [v for k, v in res.items() if k.startswith(f"{arm}-s") and k.endswith(f"-n{step}")]


def long_table(title: str, cell: Callable[[str, int], str]) -> None:
    """Print a markdown table of one ``cell(arm, step)`` per long arm and step."""
    cols = " | ".join(f"{s} steps" for s in LONG_STEPS)
    print(f"\n{title}\n\n| arm | {cols} |\n|---|" + "---|" * len(LONG_STEPS))  # noqa: T201
    for arm in LONG_ARMS:
        print(f"| {arm} | " + " | ".join(cell(arm, s) for s in LONG_STEPS) + " |")  # noqa: T201


def paired_change(res: Results, arm: str, step: int) -> str:
    """Change of the probe from the same seed's step 0: mean +- SE (MDD = 2.8 SE) over seeds."""
    diff = []
    for seed in range(3):
        a, b = res.get(f"{arm}-s{seed}-n{step}"), res.get(f"{arm}-s{seed}-n0")
        if a and b:
            diff.append(a["heldout"]["mean"] - b["heldout"]["mean"])  # type: ignore[index]
    if len(diff) < 2:
        return "-"
    se = np.std(diff, ddof=1) / np.sqrt(len(diff))
    return f"{np.mean(diff):+.3f} +- {se:.3f} (MDD {2.8 * se:.3f})"


THRESHOLD = 0.032
"""Pre-registered effect threshold of B-teach-long: 2.8 x the standard error (0.012) of the
long-jepa change of the probe from the same seed's step 0 at 10,000 steps."""
CONTRAST_STEPS = (2500, 5000, 10000)


def seed_scores(res: Results, arm: str, step: int) -> dict[int, float]:
    """Probe mean of every seed of ``arm`` at ``step``, by seed."""
    out = {}
    for seed in range(3):
        row = res.get(f"{arm}-s{seed}-n{step}")
        if row:
            out[seed] = row["heldout"]["mean"]  # type: ignore[index]
    return out


def paired_contrast(res: Results, a: str, b: str, step: int) -> tuple[float, float, int]:
    """Mean, standard error and n of ``a - b`` paired by seed (same init) at ``step``."""
    x, y = seed_scores(res, a, step), seed_scores(res, b, step)
    diff = [x[k] - y[k] for k in x if k in y]
    if len(diff) < 2:
        return float("nan"), float("nan"), len(diff)
    return float(np.mean(diff)), float(np.std(diff, ddof=1) / np.sqrt(len(diff))), len(diff)


def resolved(mean: float, se: float) -> bool:
    """Whether a contrast exceeds the threshold and 2.8 of its own standard errors."""
    return abs(mean) > max(THRESHOLD, 2.8 * se)


def teacher_verdict(res: Results, step: int = 10000) -> str:
    """The pre-registered reading of the touch teacher at ``step`` (see ``docs/model.md``)."""
    tj = paired_contrast(res, "long-teach", "long-jepa", step)
    tp = paired_contrast(res, "long-teach", "long-placebo", step)
    pj = paired_contrast(res, "long-placebo", "long-jepa", step)
    if np.isnan(tj[0]) or np.isnan(tp[0]):
        return "incomplete"
    if resolved(*tj[:2]) and tj[0] < 0:
        return "teacher harms"
    if resolved(*tj[:2]) and resolved(*tp[:2]) and tp[0] > 0:
        return "teacher prevents drift"
    if resolved(*tj[:2]) and resolved(*pj[:2]) and pj[0] > 0 and not resolved(*tp[:2]):
        return "extra heads only"
    return "null"


def contrast_table(res: Results) -> None:
    """Paired contrasts of the teacher arms at the late steps, with the pre-registered verdict."""
    print(f"\nPaired contrasts (same seed, same init); threshold {THRESHOLD}\n")  # noqa: T201
    print("| contrast | " + " | ".join(f"{s} steps" for s in CONTRAST_STEPS) + " |")  # noqa: T201
    print("|---|" + "---|" * len(CONTRAST_STEPS))  # noqa: T201
    for a, b in (
        ("long-teach", "long-jepa"),
        ("long-teach", "long-placebo"),
        ("long-placebo", "long-jepa"),
    ):
        cells = []
        for step in CONTRAST_STEPS:
            mean, se, n = paired_contrast(res, a, b, step)
            tag = "resolved" if n > 1 and resolved(mean, se) else "not resolved"
            cells.append(f"{mean:+.3f} +- {se:.3f} (n {n}, {tag})" if n > 1 else "-")
        print(f"| {a} - {b} | " + " | ".join(cells) + " |")  # noqa: T201
    print(f"\nVerdict at 10,000 steps: {teacher_verdict(res)}")  # noqa: T201


def long_report() -> None:
    """The steps curve: probe, change from step 0, held-out against training loss, geometry."""
    res = load_probes()
    floor = [v["heldout"]["mean"] for k, v in res.items() if k.startswith("floor-vp-s")]  # type: ignore[index]

    def probe(arm: str, step: int) -> str:
        runs = run_rows(res, arm, step)
        values = [r["heldout"]["mean"] for r in runs]  # type: ignore[index]
        return f"{mean_sd(values)} ({len(runs)})" if runs else "-"

    def losses(arm: str, step: int) -> str:
        runs = run_rows(res, arm, step)
        held = f"{np.mean([r['heldout_pred'] for r in runs]):.3f}" if runs else "-"  # type: ignore[misc]
        train = run_losses(arm).get(step)
        return f"{held} / {train:.3f}" if train is not None else held

    def geometry(term: str) -> Callable[[str, int], str]:
        def cell(arm: str, step: int) -> str:
            vals = [r["geometry"][term] for r in run_rows(res, arm, step) if "geometry" in r]  # type: ignore[index]
            return f"{np.mean(vals):.1f}" if vals else "-"

        return cell

    long_table(
        f"Probe mean, mean +- sd over seeds (n); floor V+P {mean_sd(floor)} (n {len(floor)})", probe
    )
    long_table(
        "Change from the encoder's own step 0 (paired by seed)",
        lambda a, s: paired_change(res, a, s),
    )
    long_table("Held-out loss (prediction term) / training loss of the 100 steps before", losses)

    def touch_r2(arm: str, step: int) -> str:
        rows = [r for r in run_rows(res, arm, step) if "touch_r2_now" in r]
        if not rows:
            return "-"
        now = np.mean([r["touch_r2_now"] for r in rows])  # type: ignore[type-var]
        nxt = np.mean([r["touch_r2_next"] for r in rows])  # type: ignore[type-var]
        return f"{now:.2f} / {nxt:.2f}"

    long_table("Touch-latent R2 of the teacher heads on held-out episodes (now / next)", touch_r2)
    long_table("Effective rank of the held-out states", geometry("erank"))
    long_table("Per-dimension std of the held-out states", geometry("std"))
    contrast_table(res)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=("index", "train", "probe", "report", "longreport"))
    parser.add_argument("--arm", choices=sorted(ARMS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0, help="default 1,000 (10,000 for long-*)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.command == "index":
        make_index()
    elif args.command == "train":
        default = 10000 if args.arm in LONG_ARMS else STEPS
        train_arm(args.arm, args.seed, args.steps or default, args.workers)
    elif args.command == "probe":
        probe_all()
    elif args.command == "longreport":
        long_report()
    else:
        report()


if __name__ == "__main__":
    main()
