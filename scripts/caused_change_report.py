"""Caused-change events per minute of generated episodes, by outcome type, with a bar chart.

Usage: python scripts/caused_change_report.py <generated dir> [--plot out.png] [--vs <dir>]

Reads ``<dir>/humanoid/*.npz``, groups them by ``meta.effector`` (and ``meta.program`` when the
episodes have one), prints events per minute with 95% bootstrap intervals over episodes and the
outcome-type coverage (``sensorium.caused_change``), and with ``--plot`` saves a bar chart. With
``--vs`` a second directory (e.g. the random play) is drawn next to the first.
"""

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sensorium.caused_change import EVENT_TYPES, CausedChange, caused_change, summarise_events  # noqa: E402
from sensorium.episode import Episode  # noqa: E402


def load_groups(directory: Path) -> dict[str, list[CausedChange]]:
    """Caused-change counts of every episode, grouped by ``effector`` or ``effector/program``."""
    groups: dict[str, list[CausedChange]] = defaultdict(list)
    for path in sorted((directory / "humanoid").glob("*.npz")):
        episode = Episode.load(path)
        if "object_pose" not in episode.privileged:
            continue
        key = str(episode.meta.get("effector"))
        program = episode.meta.get("program")
        groups[f"{key}/{program}" if program else key].append(caused_change(episode))
    return groups


def table(groups: dict[str, list[CausedChange]]) -> str:
    """Markdown table: one row per group, events per minute [interval] by type, plus coverage."""
    header = ["group", "n", "min", *EVENT_TYPES, "coverage"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, results in groups.items():
        s = summarise_events(results)
        cells = [f"{m:.1f} [{lo:.1f}, {hi:.1f}]" for m, lo, hi in s["per_minute"].values()]
        lines.append(
            f"| {name} | {s['n']} | {s['minutes']:.1f} | "
            + " | ".join(cells)
            + f" | {s['coverage']} |"
        )
    return "\n".join(lines)


def plot(sets: dict[str, dict[str, list[CausedChange]]], out: Path) -> None:
    """One panel per effector group; bars by outcome type, one colour per data set."""
    names = sorted({g for groups in sets.values() for g in groups})
    fig, axes = plt.subplots(1, len(names), figsize=(6.5 * len(names), 4.2), squeeze=False)
    width = 0.8 / len(sets)
    for ax, name in zip(axes[0], names, strict=True):
        for k, (label, groups) in enumerate(sets.items()):
            if name not in groups:
                continue
            s = summarise_events(groups[name])
            mean = [s["per_minute"][t][0] for t in EVENT_TYPES]
            err = [
                [m - s["per_minute"][t][1] for m, t in zip(mean, EVENT_TYPES, strict=True)],
                [s["per_minute"][t][2] - m for m, t in zip(mean, EVENT_TYPES, strict=True)],
            ]
            xs = [i + k * width for i in range(len(EVENT_TYPES))]
            ax.bar(xs, mean, width, yerr=err, label=f"{label} (n={s['n']}, {s['coverage']} types)")
        ax.set_xticks([i + width * (len(sets) - 1) / 2 for i in range(len(EVENT_TYPES))])
        ax.set_xticklabels(EVENT_TYPES, rotation=45, ha="right")
        ax.set_ylabel("events per minute")
        ax.set_title(name)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def main() -> None:
    """Print the table; optionally plot."""
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    args = sys.argv[1:]
    directory = Path(args[0])
    sets = {directory.name: load_groups(directory)}
    if "--vs" in args:
        other = Path(args[args.index("--vs") + 1])
        sets[other.name] = load_groups(other)
    for label, groups in sets.items():
        sys.stdout.write(f"\n{label}\n{table(groups)}\n")
    if "--plot" in args:
        plot(sets, Path(args[args.index("--plot") + 1]))


if __name__ == "__main__":
    main()
