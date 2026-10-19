"""B-ft-curve: does a pretrained encoder make supervised training on the contact labels more
label-efficient?

A label-efficiency curve: the B-ctrl-0 fine-tune path (``controls.train_supervised``, V+P, tiny
model at 64 px) from four starting encoders at seven label budgets, three seeds each:

* starts: ``scratch`` (random init), ``jepa1k`` / ``jepa10k`` (``long-jepa`` of B-steps-0 at 1,000
  and 10,000 steps), ``mae10k`` (``long-mae`` at 10,000 steps), the pretrained run of the same seed;
* budgets: 1, 2, 4, 8, 16, 32 and 100 labelled pretraining episodes, a nested subset per seed
  (``subset``), so every start sees the same episodes at a budget and a larger budget contains the
  smaller ones;
* schedule (pre-declared): 600 steps x 16 sequences up to 16 episodes, 1,000 from 32 on (the
  supervised-ceiling recipe at 100), with a checkpoint at 300 steps (the B-ctrl-0 fine-tune length).

Evaluation on the 20 held-out episodes: the B-ctrl-0 probe (leave-one-episode-out ridge on the
fine-tuned state) and the run's own linear head, both saved as out-of-fold predictions so that
contrasts are paired by seed and by held-out episode (``contrast``). Pre-registration and outcome
rules: ``docs/model.md``, "B-ft-curve".

    uv run python scripts/ft_curve.py train --start jepa1k --budget 4 --seed 0
    uv run python scripts/ft_curve.py probe
    uv run python scripts/ft_curve.py report
"""

import argparse
import functools
import json
import logging
import os
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import torch

from sensorium.model import paired
from sensorium.model.controls import SupervisedConfig, head_predictions, train_supervised
from sensorium.model.paired import Contrast
from sensorium.model.pretrain import build_space, load_config, load_layer
from sensorium.model.probes import Predictions, collect, predict_probes

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = Path("data/probe_test/humanoid")
SPLIT = Path("runs/decisive/split.json")
ROOT = Path("runs/ftcurve")
INIT_ROOT = Path("runs/steps0")
FIGURES = Path(".tmp/phd-b")
STRIDE = 8
DRAWS = paired.DRAWS
NO_TOUCH = ("vision", "proprio_body", "vestibular", "proprio_effector")
STARTS: dict[str, tuple[str, int] | None] = {
    "scratch": None,
    "jepa1k": ("long-jepa", 1000),
    "jepa10k": ("long-jepa", 10000),
    "mae10k": ("long-mae", 10000),
}
BUDGETS = (1, 2, 4, 8, 16, 32, 100)
SEEDS = (0, 1, 2)
SMALL = (1, 2, 4, 8)
MID_STEP = 300
REACH = (0.40, 0.50)
TERMS = (
    "contact_left_auroc",
    "contact_right_auroc",
    "contact_object_auroc",
    "log_force_r2",
    "tip_r2",
)
COLORS = {"scratch": "#2a78d6", "jepa1k": "#eb6834", "jepa10k": "#1baf7a", "mae10k": "#eda100"}


def files(key: str) -> list[Path]:
    """Episode files of a split from ``decisive_test.py split``."""
    return [DATA / n for n in json.loads(SPLIT.read_text())[key]]


def steps_for(budget: int) -> int:
    """The pre-declared fine-tune length at a label budget."""
    return 600 if budget <= 16 else 1000


def subset(seed: int, budget: int, n: int = 100) -> list[int]:
    """Indices of the labelled pretraining episodes at ``budget``: the first ``budget`` of one
    permutation per seed, so subsets are nested across budgets and shared by every start."""
    if not 1 <= budget <= n:
        raise ValueError(f"budget {budget} outside 1..{n}")
    order = np.random.default_rng([seed, 7]).permutation(n)
    return sorted(int(i) for i in order[:budget])


def run_name(start: str, budget: int, seed: int) -> str:
    """Directory and prediction name of one cell."""
    return f"{start}-b{budget}-s{seed}"


def init_path(start: str, seed: int, root: Path = INIT_ROOT) -> Path | None:
    """The pretrained checkpoint a start begins from (``None`` for ``scratch``)."""
    source = STARTS[start]
    if source is None:
        return None
    arm, step = source
    return root / f"{arm}-s{seed}" / f"ckpt-{step}.pt"


# ---------------------------------------------------------------------------------------------
# train and probe


def train_cell(
    start: str, budget: int, seed: int, steps: int, workers: int, init_root: Path
) -> None:
    """Fine-tune one cell into ``runs/ftcurve/runs/<start>-b<budget>-s<seed>``."""
    cfg = load_config(Path("configs/model/cpu_tiny.toml"))
    episodes = files("pretrain")
    space = build_space(episodes, cfg.data.stats_episodes)
    init = init_path(start, seed, init_root)
    if init is not None and not init.exists():
        raise FileNotFoundError(f"{init}: copy the B-steps-0 checkpoints first (see the docstring)")
    steps = steps or steps_for(budget)
    out = ROOT / "runs" / run_name(start, budget, seed)
    began = time.perf_counter()
    train_supervised(
        SupervisedConfig(
            model=replace(cfg.model, senses=NO_TOUCH),
            steps=steps,
            seed=seed,
            workers=workers,
            save_steps=(MID_STEP,) if steps > MID_STEP else (),
        ),
        [episodes[i] for i in subset(seed, budget, len(episodes))],
        space,
        out,
        logging.info,
        init=init,
    )
    meta = {
        "start": start,
        "budget": budget,
        "seed": seed,
        "steps": steps,
        "init": str(init),
        "episodes": subset(seed, budget, len(episodes)),
        "seconds": time.perf_counter() - began,
        "threads": torch.get_num_threads(),
    }
    (out / "cell.json").write_text(json.dumps(meta))


def save_predictions(name: str, pred: Predictions) -> None:
    """Out-of-fold predictions of one cell (for paired bootstraps)."""
    out = ROOT / "preds"
    out.mkdir(parents=True, exist_ok=True)
    edges = np.array([[b.start, b.stop] for b in pred.blocks])
    np.savez(
        out / f"{name}.npz", pred=pred.pred, target=pred.target, groups=pred.groups, edges=edges
    )


@functools.cache
def load_predictions(name: str) -> Predictions | None:
    """Inverse of ``save_predictions``; ``None`` if the cell has not been probed (memoised: call
    it after probing, not while predictions are still being written)."""
    path = ROOT / "preds" / f"{name}.npz"
    if not path.exists():
        return None
    raw = np.load(path)
    blocks = [slice(int(a), int(b)) for a, b in raw["edges"]]
    return Predictions(raw["pred"], raw["target"], blocks, raw["groups"])


def probe_all() -> None:
    """Ridge-probe and head predictions of every trained cell, at 300 steps and at the end.

    Writes ``preds/<cell>-n<step>.npz`` (ridge probe on the state) and ``-head.npz`` (the run's
    own head); cells already probed are skipped, so it can be re-run while training continues.
    """
    heldout = files("heldout")
    for run in sorted((ROOT / "runs").glob("*-b*-s*")):
        if not (run / "cell.json").exists():
            continue  # still training
        steps = json.loads((run / "cell.json").read_text())["steps"]
        for ckpt, head, step in (
            (run / f"ckpt-{MID_STEP}.pt", run / f"head-{MID_STEP}.pt", MID_STEP),
            (run / "ckpt.pt", run / "head.pt", steps),
        ):
            name = f"{run.name}-n{step}"
            if not ckpt.exists() or (ROOT / "preds" / f"{name}-head.npz").exists():
                continue
            data = collect(load_layer(ckpt), heldout, STRIDE, (), raw=False)
            save_predictions(name, predict_probes(data, "state"))
            save_predictions(f"{name}-head", head_predictions(data, head))
            logging.info("probed %s", name)


# ---------------------------------------------------------------------------------------------
# paired contrasts (``sensorium.model.paired``)


def contrast(pairs: list[tuple[Predictions, Predictions]], draws: int = DRAWS) -> Contrast:
    """Paired contrast from ``(a, b)`` predictions of the same seed on the same windows.

    Raises:
        ValueError: if the predictions were not all scored on the same held-out windows.
    """
    return paired.contrast([a for a, _ in pairs], [b for _, b in pairs], True, draws)


def verdict(cells: dict[int, Contrast]) -> str:
    """The pre-registered reading of one pretrained start against scratch (``docs/model.md``).

    ``cells`` maps every budget to its contrast; a budget is a *gain* (*loss*) when the contrast
    is resolved and positive (negative).
    """
    if any(b not in cells or cells[b].n_a < 2 for b in BUDGETS):
        return "incomplete"
    gains = {b for b in BUDGETS if cells[b].resolved and cells[b].point > 0}
    losses = {b for b in BUDGETS if cells[b].resolved and cells[b].point < 0}
    if gains and losses:
        return "mixed"
    if losses:
        return "harms"
    if not gains:
        return "no benefit"
    if gains <= set(SMALL):
        return "helps at small budgets"
    if gains & set(SMALL):
        return "helps at small and large budgets"
    return "helps at large budgets only"


def labels_to_reach(scores: dict[int, float], level: float) -> float:
    """Budget at which a curve first reaches ``level``, log-linear between budgets.

    NaN if the curve never reaches it; the smallest budget if it starts above it.
    """
    budgets = sorted(scores)
    if scores[budgets[0]] >= level:
        return float(budgets[0])
    for lo, hi in zip(budgets[:-1], budgets[1:], strict=True):
        if scores[hi] >= level > scores[lo]:
            t = (level - scores[lo]) / (scores[hi] - scores[lo])
            return float(np.exp(np.log(lo) + t * (np.log(hi) - np.log(lo))))
    return float("nan")


# ---------------------------------------------------------------------------------------------
# report


def step_of(budget: int, final: bool) -> int:
    """The checkpoint step reported: the end of the schedule, or the 300-step one."""
    return steps_for(budget) if final else MID_STEP


def cell_preds(start: str, budget: int, final: bool, head: bool) -> dict[int, Predictions]:
    """Predictions of every probed seed of one cell, by seed."""
    suffix = "-head" if head else ""
    out = {}
    for seed in SEEDS:
        name = f"{run_name(start, budget, seed)}-n{step_of(budget, final)}{suffix}"
        pred = load_predictions(name)
        if pred is not None:
            out[seed] = pred
    return out


def curve(start: str, final: bool, head: bool) -> dict[int, list[float]]:
    """Probe mean of every seed, by budget."""
    return {
        b: [p.scores()["mean"] for p in cell_preds(start, b, final, head).values()] for b in BUDGETS
    }


def contrasts(start: str, final: bool, head: bool) -> dict[int, Contrast]:
    """``start - scratch`` at every budget with at least one paired seed."""
    out = {}
    for b in BUDGETS:
        a, s = cell_preds(start, b, final, head), cell_preds("scratch", b, final, head)
        pairs = [(a[k], s[k]) for k in a if k in s]
        if pairs:
            out[b] = contrast(pairs)
    return out


def score_table(title: str, final: bool, head: bool) -> None:
    """Probe mean +- seed sd for every start and budget."""
    print(f"\n{title}\n")  # noqa: T201
    print("| start | " + " | ".join(f"{b} ep" for b in BUDGETS) + " |")  # noqa: T201
    print("|---|" + "---|" * len(BUDGETS))  # noqa: T201
    for start in STARTS:
        cells = []
        for values in curve(start, final, head).values():
            if not values:
                cells.append("-")
            else:
                sd = f" +- {np.std(values, ddof=1):.2f}" if len(values) > 1 else ""
                cells.append(f"{np.mean(values):.2f}{sd} ({len(values)})")
        print(f"| {start} | " + " | ".join(cells) + " |")  # noqa: T201


def contrast_table(title: str, final: bool, head: bool) -> None:
    """Pretrained - scratch at every budget, its SE components, MDD and the verdict."""
    print(f"\n{title}\n")  # noqa: T201
    print("| start - scratch | budget | difference | seed SE | episode SE | combined SE | MDD | |")  # noqa: T201
    print("|---|---|---|---|---|---|---|---|")  # noqa: T201
    verdicts = {}
    for start in list(STARTS)[1:]:
        cells = contrasts(start, final, head)
        verdicts[start] = verdict(cells)
        for b, c in cells.items():
            tag = "resolved" if c.resolved and c.n_a > 1 else ""
            print(  # noqa: T201
                f"| {start} | {b} | {c.point:+.3f} | {c.seed_se:.3f} | {c.episode_se:.3f} | "
                f"{c.se:.3f} | {c.mdd:.3f} | {tag} |"
            )
    print("\nVerdicts: " + "; ".join(f"{k}: {v}" for k, v in verdicts.items()))  # noqa: T201


def reach_table() -> None:
    """Labelled episodes needed to reach fixed probe means, and the utility of each start."""
    print("\nLabelled episodes to reach a probe mean (seed-mean curve, log-linear)\n")  # noqa: T201
    print("| start | " + " | ".join(f"to {r:.2f}" for r in REACH) + " |")  # noqa: T201
    print("|---|" + "---|" * len(REACH))  # noqa: T201
    means = {}
    for start in STARTS:
        c = {b: float(np.mean(v)) for b, v in curve(start, True, False).items() if v}
        means[start] = c
        cells = [f"{labels_to_reach(c, r):.1f}" if c else "-" for r in REACH]
        print(f"| {start} | " + " | ".join(cells) + " |")  # noqa: T201
    if not means["scratch"]:
        return
    print(  # noqa: T201
        "\nUtility (Newell and Deng 2020): scratch episodes to match / episodes used - 1\n"
    )
    print("| start | " + " | ".join(f"{b} ep" for b in BUDGETS) + " |")  # noqa: T201
    print("|---|" + "---|" * len(BUDGETS))  # noqa: T201
    for start in list(STARTS)[1:]:
        cells = []
        for b in BUDGETS:
            if b not in means[start]:
                cells.append("-")
                continue
            matched = labels_to_reach(means["scratch"], means[start][b])
            cells.append(f"{matched / b - 1:+.2f}" if np.isfinite(matched) else "beyond 100")
        print(f"| {start} | " + " | ".join(cells) + " |")  # noqa: T201


def terms_table() -> None:
    """The five probe terms of every start and budget (seed means, end of schedule)."""
    print("\nTerms of the probe (seed means, end of schedule)\n")  # noqa: T201
    print("| start | budget | " + " | ".join(TERMS) + " |")  # noqa: T201
    print("|---|---|" + "---|" * len(TERMS))  # noqa: T201
    for start in STARTS:
        for b in BUDGETS:
            preds = cell_preds(start, b, True, False).values()
            if preds:
                scores = [p.scores() for p in preds]
                cells = [f"{np.mean([s[t] for s in scores]):.2f}" for t in TERMS]
                print(f"| {start} | {b} | " + " | ".join(cells) + " |")  # noqa: T201


def plot_curves(out: Path) -> None:
    """Score against budget (log x) per start, the paired contrasts, and the head readout."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    panels = (
        (axes[0], False, "probe mean, ridge on the fine-tuned state"),
        (axes[2], True, "probe mean, the run's own linear head"),
    )
    for ax, head, title in panels:
        for start in STARTS:
            pts = [(b, v) for b, v in curve(start, True, head).items() if v]
            if not pts:
                continue
            x = [b for b, _ in pts]
            y = [np.mean(v) for _, v in pts]
            err = [np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0 for _, v in pts]
            ax.errorbar(x, y, err, marker="o", ms=5, lw=2, capsize=3, color=COLORS[start])
            ax.plot([], [], color=COLORS[start], lw=2, label=start)
        ax.set(xscale="log", xlabel="labelled episodes", title=title, xticks=BUDGETS)
        ax.set_xticklabels([str(b) for b in BUDGETS])
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    for k, start in enumerate(list(STARTS)[1:]):
        cells = contrasts(start, True, False)
        if not cells:
            continue
        # a small offset per start so the error bars do not overlap
        x = np.array(list(cells)) * (1 + 0.06 * (k - 1))
        y = [c.point for c in cells.values()]
        ax.errorbar(
            x,
            y,
            [c.se for c in cells.values()],
            marker="o",
            ms=5,
            lw=2,
            capsize=3,
            color=COLORS[start],
            label=start,
        )
    ax.axhline(0, color="#888888", lw=1)
    ax.set(
        xscale="log",
        xlabel="labelled episodes",
        xticks=BUDGETS,
        title="pretrained - scratch (paired; bars: combined SE)",
    )
    ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle("B-ft-curve: 20 held-out episodes, 3 seeds (score bars: seed SE)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    logging.info("wrote %s", out)


def plot_losses(out: Path) -> None:
    """Training loss against step at four budgets, one line per start (seeds thin, mean thick)."""
    shown = (1, 4, 16, 100)
    fig, axes = plt.subplots(1, len(shown), figsize=(16, 4), sharey=True)
    for ax, b in zip(axes, shown, strict=True):
        for start in STARTS:
            runs = []
            for seed in SEEDS:
                log = ROOT / "runs" / run_name(start, b, seed) / "log.jsonl"
                if log.exists():
                    rows = [json.loads(line) for line in log.read_text().splitlines()]
                    runs.append(([r["step"] for r in rows], [r["loss"] for r in rows]))
                    ax.plot(*runs[-1], color=COLORS[start], lw=0.7, alpha=0.35)
            if runs:
                n = min(len(r[0]) for r in runs)
                mean = np.mean([r[1][:n] for r in runs], axis=0)
                ax.plot(runs[0][0][:n], mean, color=COLORS[start], lw=2, label=start)
        ax.set(title=f"{b} labelled episodes", xlabel="step", yscale="log")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("supervised training loss")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    logging.info("wrote %s", out)


def report() -> None:
    """Every table of the study, then the two figures under ``.tmp/phd-b``."""
    score_table("Probe mean (ridge on the fine-tuned state), end of schedule", True, False)
    contrast_table(
        "Primary contrast: pretrained - scratch, ridge probe, end of schedule", True, False
    )
    score_table("Probe mean of the run's own linear head, end of schedule", True, True)
    contrast_table("Pretrained - scratch, own head, end of schedule", True, True)
    score_table(f"Probe mean (ridge), at {MID_STEP} steps (the B-ctrl-0 length)", False, False)
    contrast_table(f"Pretrained - scratch, ridge probe, at {MID_STEP} steps", False, False)
    reach_table()
    terms_table()
    plot_curves(FIGURES / "ftcurve_scores.png")
    plot_losses(FIGURES / "ftcurve_losses.png")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=("train", "probe", "report"))
    parser.add_argument("--start", choices=sorted(STARTS))
    parser.add_argument("--budget", type=int, choices=BUDGETS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0, help="default: the pre-declared schedule")
    parser.add_argument("--workers", type=int, default=2, help="loader processes")
    parser.add_argument("--init-root", type=Path, default=INIT_ROOT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if "OMP_NUM_THREADS" in os.environ:
        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    if args.command == "train":
        if args.start is None or args.budget is None:
            parser.error("train needs --start and --budget")
        train_cell(args.start, args.budget, args.seed, args.steps, args.workers, args.init_root)
    elif args.command == "probe":
        probe_all()
    else:
        report()


if __name__ == "__main__":
    main()
