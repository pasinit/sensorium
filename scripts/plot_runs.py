"""Plot a training study: loss terms, held-out loss and held-out probe score against steps.

Reads ``<runs>/<arm>-s<seed>/log.jsonl`` (one JSON object per logged step, written by
``pretrain.train`` and the supervised controls) and probe JSONs
``<probes>/<arm>-s<seed>-n<step>.json`` (nested dicts). ``--metric`` is a dotted path to the
score (``vp.mean``, ``heldout.mean``); the held-out loss comes from ``heldout_pred`` and
``heldout_reg`` in the same files when present. One row of three panels per arm, every seed
drawn thin, the seed mean thick:

1. training loss terms (``loss``, ``pred``, ``reg``, ``aux``, ``teacher_own``, as logged);
2. held-out loss against the checkpoint step;
3. held-out probe score against the checkpoint step, with the random-init floor as dashed lines
   (``--floor arm=prefix`` reads ``<probes>/<prefix>-s<seed>.json``, one line per floor seed).

    uv run python scripts/plot_runs.py --runs runs/teacher --probes runs/teacher/probes \\
        --metric vp.mean --floor b-teacher=floor-vp --floor c-none=floor-vp \\
        --arms b-teacher c-none --title "B-teach-0: 100 episodes, 2,000 steps x 16" \\
        --out .tmp/phd-b/teacher_curves.png
"""

import argparse
import json
import re
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOSS_KEYS = ("loss", "pred", "reg", "aux", "teacher_own")
COLORS = {
    "loss": "#222222",
    "pred": "#1f77b4",
    "reg": "#ff7f0e",
    "aux": "#2ca02c",
    "teacher_own": "#9467bd",
}
RUN = re.compile(r"^(?P<arm>.+)-s(?P<seed>\d+)$")
PROBE = re.compile(r"^(?P<arm>.+)-s(?P<seed>\d+)-n(?P<step>\d+)$")


def load_logs(root: Path) -> dict[str, dict[int, list[dict[str, float]]]]:
    """``{arm: {seed: [log rows]}}`` from the run directories under ``root``."""
    out: dict[str, dict[int, list[dict[str, float]]]] = defaultdict(dict)
    for path in sorted(root.glob("*-s*/log.jsonl")):
        match = RUN.match(path.parent.name)
        if match:
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            out[match["arm"]][int(match["seed"])] = [r for r in rows if "step" in r and "loss" in r]
    return out


def dotted(obj: dict[str, object], path: str) -> float | None:
    """Value at a dotted path of nested dicts, or ``None``."""
    cur: object = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return float(cur) if isinstance(cur, int | float) else None


def load_probes(probes: Path) -> dict[str, dict[int, dict[int, dict[str, object]]]]:
    """``{arm: {seed: {step: probe json}}}``."""
    out: dict[str, dict[int, dict[int, dict[str, object]]]] = defaultdict(lambda: defaultdict(dict))
    for path in sorted(probes.glob("*.json")):
        match = PROBE.match(path.stem)
        if match:
            out[match["arm"]][int(match["seed"])][int(match["step"])] = json.loads(path.read_text())
    return out


def mean_curve(curves: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Mean over seeds on the steps all seeds share."""
    common = sorted(set.intersection(*(set(x.tolist()) for x, _ in curves)))
    if not common:
        return np.array([]), np.array([])
    ys = [np.interp(common, x, y) for x, y in curves]
    return np.array(common), np.mean(ys, axis=0)


def plot_losses(ax: plt.Axes, logs: dict[int, list[dict[str, float]]]) -> None:
    """Training loss terms against steps, every seed."""
    keys = [k for k in LOSS_KEYS if any(k in r for rows in logs.values() for r in rows)]
    for key in keys:
        curves = []
        for rows in logs.values():
            pts = [(r["step"], r[key]) for r in rows if key in r]
            x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
            ax.plot(x, y, color=COLORS[key], alpha=0.3, lw=0.8)
            curves.append((x, y))
        x, y = mean_curve(curves)
        ax.plot(x, y, color=COLORS[key], lw=2, label=key)
    ax.set_xlabel("training step")
    ax.set_title("training loss terms", fontsize=9)
    ax.legend(fontsize=7)


def plot_heldout_loss(ax: plt.Axes, probes: dict[int, dict[int, dict[str, object]]]) -> None:
    """Held-out loss terms against the checkpoint step, every seed."""
    drawn = False
    for key, color in (("heldout_pred", COLORS["pred"]), ("heldout_reg", COLORS["reg"])):
        curves = []
        for per_step in probes.values():
            pts = sorted(
                (s, dotted(j, key)) for s, j in per_step.items() if dotted(j, key) is not None
            )
            if pts:
                x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
                ax.plot(x, y, "o-", color=color, alpha=0.3, lw=0.8, ms=3)
                curves.append((x, y))
        if curves:
            x, y = mean_curve(curves)
            ax.plot(x, y, "o-", color=color, lw=2, label=key.removeprefix("heldout_"))
            drawn = True
    if drawn:
        ax.legend(fontsize=7)
    else:
        ax.text(
            0.5,
            0.5,
            "no held-out loss in the probe files",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
    ax.set_xlabel("checkpoint step")
    ax.set_title("held-out loss (held-out episodes)", fontsize=9)


def plot_probe(
    ax: plt.Axes,
    probes: dict[int, dict[int, dict[str, object]]],
    metric: str,
    floors: list[float],
    raw_floor_label: str,
) -> None:
    """Held-out probe score against the checkpoint step, every seed, with the floor lines."""
    curves = []
    for per_step in probes.values():
        pts = sorted(
            (s, dotted(j, metric)) for s, j in per_step.items() if dotted(j, metric) is not None
        )
        if pts:
            x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
            ax.plot(x, y, "o-", color="#d62728", alpha=0.35, lw=0.8, ms=3)
            curves.append((x, y))
    if curves:
        x, y = mean_curve(curves)
        ax.plot(x, y, "o-", color="#d62728", lw=2, label=f"trained (n = {len(curves)})")
    for i, level in enumerate(floors):
        ax.axhline(level, color="gray", ls="--", lw=1, label=raw_floor_label if i == 0 else None)
    ax.set_xlabel("checkpoint step")
    ax.set_title(f"held-out probe score ({metric})", fontsize=9)
    ax.legend(fontsize=7)


def floor_levels(probes: Path, prefix: str, metric: str) -> list[float]:
    """One level per floor seed: ``<probes>/<prefix>-s<seed>.json``."""
    levels = []
    for path in sorted(probes.glob(f"{prefix}-s*.json")):
        value = dotted(json.loads(path.read_text()), metric)
        if value is not None:
            levels.append(value)
    return levels


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--metric", default="heldout.mean")
    parser.add_argument("--arms", nargs="*", help="arms to draw (default: all with a log)")
    parser.add_argument(
        "--floor", action="append", default=[], help="arm=prefix of the floor files"
    )
    parser.add_argument("--title", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logs, probes = load_logs(args.runs), load_probes(args.probes)
    arms = args.arms or sorted(set(logs) | set(probes))
    prefixes = dict(f.split("=", 1) for f in args.floor)
    fig, axes = plt.subplots(len(arms), 3, figsize=(14, 3.4 * len(arms)), squeeze=False)
    for row, arm in enumerate(arms):
        if arm in logs:
            plot_losses(axes[row][0], logs[arm])
        else:
            axes[row][0].text(
                0.5, 0.5, "no log.jsonl", ha="center", transform=axes[row][0].transAxes
            )
        axes[row][0].set_ylabel(arm, fontsize=10)
        plot_heldout_loss(axes[row][1], probes.get(arm, {}))
        floors = floor_levels(args.probes, prefixes[arm], args.metric) if arm in prefixes else []
        plot_probe(
            axes[row][2],
            probes.get(arm, {}),
            args.metric,
            floors,
            f"random-init floor (n = {len(floors)})",
        )
    title = textwrap.fill(args.title, 130)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.045 * (title.count("\n") + 1) / len(arms)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(args.out)  # noqa: T201


if __name__ == "__main__":
    main()
