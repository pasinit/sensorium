"""Print the A1 statistics of generated seated H1 episodes, one table row per variant.

Usage: python scripts/a1_report.py <generated dir> [--md]

Reads ``<dir>/humanoid/humanoid-<variant>-<policy>-<index>.npz`` (see
``configs/a1_compare.toml``), computes ``sensorium.babble_stats.a1_stats`` of every
episode and prints the mean over a variant's episodes with its 95% bootstrap
interval; ``--md`` prints markdown.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

from sensorium.babble_stats import A1Stats, a1_stats, summarise
from sensorium.episode import Episode

COLUMNS = (
    ("object_contact_rate", "obj contact", "{:.2f}"),
    ("bouts_per_minute", "bouts/min", "{:.0f}"),
    ("objects_touched", "objects", "{:.1f}"),
    ("object_motion_rate", "obj moved", "{:.2f}"),
    ("unattributed_motion_rate", "rolled alone", "{:.2f}"),
    ("in_air_rate", "in air", "{:.3f}"),
    ("lifts", "lifts/ep", "{:.2f}"),
    ("wait_share", "wait share", "{:.2f}"),
    ("free_space_rate", "free space", "{:.2f}"),
    ("voxel_coverage", "voxels", "{:.2f}"),
    ("tip_entropy_gap", "tip entropy gap", "{:.2f}"),
    ("object_peak_n", "object peak N", "{:.0f}"),
    ("limb_peak_n", "limb peak N", "{:.0f}"),
    ("limb_slam_steps", "slam steps", "{:.1f}"),
    ("self_peak_n", "self peak N", "{:.0f}"),
    ("self_rate", "self rate", "{:.3f}"),
    ("action_jerk", "jerk", "{:.3f}"),
)


def load_variants(directory: Path) -> dict[str, list[A1Stats]]:
    """A1 statistics of every episode under ``directory``, grouped by variant name."""
    groups: dict[str, list[A1Stats]] = defaultdict(list)
    for path in sorted((directory / "humanoid").glob("humanoid-*.npz")):
        match = re.fullmatch(r"humanoid-(.+)-seated_babble-\d+\.npz", path.name)
        if match:
            groups[match.group(1)].append(a1_stats(Episode.load(path)))
    return groups


def main() -> None:
    """Print the table."""
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    markdown = "--md" in sys.argv
    groups = load_variants(Path(sys.argv[1]))
    header = [
        "variant",
        "n",
        *(title for _, title, _ in COLUMNS),
        "falls",
        "cell cov",
        "skin cov",
        "sound cov",
    ]
    rows = []
    for name, stats in groups.items():
        summary = summarise(stats)
        cells = []
        for key, _, fmt in COLUMNS:
            mean, low, high = (fmt.format(v) for v in summary[key])
            cells.append(f"{mean} [{low}, {high}]")
        rows.append(
            [
                name,
                str(len(stats)),
                *cells,
                str(sum(s.fell for s in stats)),
                f"{summary['object_cell_coverage'][0]:.2f}",
                f"{summary['skin_region_coverage'][0]:.2f}",
                f"{summary['sound_pair_coverage'][0]:.2f}",
            ]
        )
    sep = " | " if markdown else "  "
    lines = [sep.join(header)] + ([sep.join(["---"] * len(header))] if markdown else [])
    lines += [sep.join(row) for row in rows]
    sys.stdout.write("\n".join(f"| {line} |" if markdown else line for line in lines) + "\n")


if __name__ == "__main__":
    main()
