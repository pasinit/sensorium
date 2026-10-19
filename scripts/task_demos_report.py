"""Success rate of the scripted demonstrators of the held-out tasks.

Usage: python scripts/task_demos_report.py <generated dir>

Reads ``<dir>/humanoid/*.npz`` made in task mode (``settings.task``): per (task, effector) the
number
of episodes, the success rate with a 95% Wilson interval, the steps to success (successful episodes
only, 20 Hz), the share of arm steps spent waiting for a plan, and the largest effector-object
force. A demonstrator that succeeds in few episodes gives few usable few-shot demonstrations: only
the successful episodes should be offered as demonstrations.
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from sensorium.babble_stats import PEAK_COLUMNS
from sensorium.episode import Episode


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval of a proportion."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def main() -> None:
    """Print the table."""
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    groups: dict[str, list[Episode]] = defaultdict(list)
    for path in sorted((Path(sys.argv[1]) / "humanoid").glob("*.npz")):
        e = Episode.load(path)
        groups[f"{e.meta['task']} / {e.meta['effector']}"].append(e)
    sys.stdout.write(
        "| task / effector | n | success | 95% interval | steps to success | waiting "
        "| max peak N |\n"
    )
    sys.stdout.write("|---|---|---|---|---|---|---|\n")
    for key, eps in sorted(groups.items()):
        wins = [e for e in eps if e.meta.get("success")]
        low, high = wilson(len(wins), len(eps))
        steps = float(np.mean([e.length for e in wins])) if wins else float("nan")
        stats = [e.meta["plan_stats"] for e in eps]
        wait = float(
            np.mean([s["wait_steps"] / (2 * e.length) for s, e in zip(stats, eps, strict=True)])
        )
        peak = max(
            float(e.privileged["peak_force"][:, PEAK_COLUMNS["effector_object"]].max()) for e in eps
        )
        sys.stdout.write(
            f"| {key} | {len(eps)} | {len(wins)}/{len(eps)} = {len(wins) / len(eps):.2f} "
            f"| [{low:.2f}, {high:.2f}] | {steps:.0f} | {wait:.2f} | {peak:.0f} |\n"
        )


if __name__ == "__main__":
    main()
