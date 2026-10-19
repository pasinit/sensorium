"""Tables and figures for a batch of experiment episodes, against random play.

Usage: python scripts/experiments_report.py <batch dir> <random-play dir> <out prefix>

Reads ``<dir>/humanoid/*.npz``, groups them by ``effector`` and ``program`` (random play by
effector), prints a markdown table (episodes, steps, wait share, object contact, largest
effector-object peak, falls, events per minute by outcome type, coverage) and writes
``<out prefix>_events.png`` (stacked events per minute, one bar per program and one for random
play), ``<out prefix>_hist.png`` (per-episode wait share, forces and caused-change events) and
``<out prefix>_strips.png`` (one frame strip per program and effector).
"""

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from sensorium.babble_stats import A1Stats, a1_stats  # noqa: E402
from sensorium.caused_change import EVENT_TYPES, CausedChange, caused_change, summarise_events  # noqa: E402
from sensorium.episode import Episode  # noqa: E402


def load(
    directory: Path, by_program: bool
) -> dict[str, list[tuple[Episode, A1Stats, CausedChange]]]:
    """Episodes with their A1 statistics and events, grouped by effector (and program)."""
    groups: dict[str, list[tuple[Episode, A1Stats, CausedChange]]] = defaultdict(list)
    for path in sorted((directory / "humanoid").glob("*.npz")):
        e = Episode.load(path)
        if "object_pose" not in e.privileged:
            continue
        key = str(e.meta["effector"]) + (f"/{e.meta['program']}" if by_program else "")
        groups[key].append((e, a1_stats(e), caused_change(e)))
    return groups


def table(groups: dict[str, list[tuple[Episode, A1Stats, CausedChange]]]) -> str:
    """Markdown table, one row per group."""
    header = [
        "group",
        "n",
        "steps",
        "wait",
        "contact",
        "peak N (max)",
        "falls",
        *EVENT_TYPES,
        "types",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, items in sorted(groups.items()):
        s = summarise_events([c for _, _, c in items])
        stats = [a for _, a, _ in items]
        cells = [f"{s['per_minute'][t][0]:.1f}" for t in EVENT_TYPES]
        lines.append(
            f"| {name} | {len(items)} | {sum(e.length for e, _, _ in items)} "
            f"| {np.mean([a.wait_share for a in stats]):.2f} "
            f"| {np.mean([a.object_contact_rate for a in stats]):.2f} "
            f"| {max(a.object_peak_n for a in stats):.0f} | {sum(a.fell for a in stats)} "
            f"| {' | '.join(cells)} | {s['coverage']} |"
        )
    return "\n".join(lines)


def stacked(
    experiments: dict[str, list[tuple[Episode, A1Stats, CausedChange]]],
    random_play: dict[str, list[tuple[Episode, A1Stats, CausedChange]]],
    out: Path,
) -> None:
    """Stacked events per minute: random play and each program, one panel per effector."""
    effectors = sorted({k.split("/")[0] for k in experiments})
    fig, axes = plt.subplots(1, len(effectors), figsize=(7 * len(effectors), 4.5), squeeze=False)
    colours = plt.get_cmap("tab10")
    for ax, eff in zip(axes[0], effectors, strict=True):
        names = [f"random play ({eff})", *sorted(k for k in experiments if k.startswith(eff + "/"))]
        rows = [random_play.get(eff, [])] + [experiments[k] for k in names[1:]]
        bottoms = np.zeros(len(names))
        for k, event in enumerate(EVENT_TYPES):
            values = np.array(
                [
                    summarise_events([c for _, _, c in r], resamples=1)["per_minute"][event][0]
                    if r
                    else 0.0
                    for r in rows
                ]
            )
            ax.bar(range(len(names)), values, bottom=bottoms, color=colours(k), label=event)
            bottoms += values
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.split("/")[-1] for n in names], rotation=30, ha="right")
        ax.set_ylabel("caused-change events per minute")
        ax.set_title(f"{eff}: scripted experiments vs random play")
    axes[0][0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def histograms(
    experiments: dict[str, list[tuple[Episode, A1Stats, CausedChange]]], out: Path
) -> None:
    """Per-episode wait share, largest effector-object force and events per minute, by program."""
    effectors = sorted({k.split("/")[0] for k in experiments})
    fig, axes = plt.subplots(len(effectors), 3, figsize=(14, 4 * len(effectors)), squeeze=False)
    for row, eff in enumerate(effectors):
        keys = sorted(k for k in experiments if k.startswith(eff + "/"))
        for k in keys:
            items = experiments[k]
            axes[row][0].hist(
                [a.wait_share * 100 for _, a, _ in items],
                bins=np.linspace(0, 100, 11),
                alpha=0.5,
                label=k.split("/")[1],
            )
            axes[row][1].hist(
                [a.object_peak_n for _, a, _ in items], bins=np.linspace(0, 600, 13), alpha=0.5
            )
            axes[row][2].hist(
                [sum(c.counts.values()) / c.minutes for _, _, c in items],
                bins=np.linspace(0, 120, 13),
                alpha=0.5,
            )
        for ax, label in zip(
            axes[row],
            ("arm steps waiting (%)", "per-episode peak object force (N)", "all events per minute"),
            strict=True,
        ):
            ax.set_xlabel(label)
            ax.set_ylabel(f"{eff}: episodes")
        axes[row][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def strips(experiments: dict[str, list[tuple[Episode, A1Stats, CausedChange]]], out: Path) -> None:
    """One strip (4 frames of the episode with the most events) per group."""
    rows = []
    for name in sorted(experiments):
        e, _, _ = max(experiments[name], key=lambda item: sum(item[2].counts.values()))
        idx = np.linspace(0, e.length - 1, 4).astype(int)
        rows.append(np.concatenate([e.rgb[i] for i in idx], axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(out)
    sys.stdout.write("strip rows: " + ", ".join(sorted(experiments)) + "\n")


def main() -> None:
    """Print the table and write the figures."""
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    batch, random_dir, prefix = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    experiments = load(batch, True)
    random_play = load(random_dir, False)
    sys.stdout.write(table(experiments) + "\n\nrandom play\n" + table(random_play) + "\n")
    stacked(experiments, random_play, Path(f"{prefix}_events.png"))
    histograms(experiments, Path(f"{prefix}_hist.png"))
    strips(experiments, Path(f"{prefix}_strips.png"))


if __name__ == "__main__":
    main()
