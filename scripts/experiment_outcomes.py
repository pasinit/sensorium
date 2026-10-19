"""Per experiment variant: do the attempts reach their intended outcome, and where does the time go?

Usage: python scripts/experiment_outcomes.py <generated dir> [--plot out.png]

For every (effector, program) of ``<dir>/humanoid/*.npz`` it reports

* episodes, attempts (planned movements, ``meta.plan_stats``), intended outcomes and their share
  of the attempts, and the share of episodes with at least one intended outcome. The intended
  outcome of ``push`` is an object displaced, of ``topple`` an object tipped over, of ``press`` a
  button pressed or an object held pressed (12 steps of contact), of ``slide`` a tip dragged along
  the table (8 steps of table contact), of ``knock`` a strike (a new contact of 1 N or more) and of
  ``door`` a flap opened;
* the share of steps with an effector on an object, with an effector on the table (hands rest on
  it much of the time), and waiting (an arm without a plan, of both arms' steps); the three
  overlap and need not sum to 1;
* caused-change events per minute (all types).

Outcomes and attempts are counted over the same episodes, so the ratio is an average, not a
per-attempt match: an outcome may come from the other arm or from a bump.
"""

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sensorium.babble_stats import CONTACT_N, PEAK_COLUMNS, runs  # noqa: E402
from sensorium.caused_change import caused_change  # noqa: E402
from sensorium.episode import Episode  # noqa: E402

PRESS_STEPS = 12
SLIDE_STEPS = 8
STRIKE_N = 1.0


def outcomes(episode: Episode, program: str) -> int:
    """Intended outcomes of ``program`` in the episode (see the module docstring)."""
    counts = caused_change(episode).counts
    peak = episode.privileged["peak_force"]
    touching = episode.privileged["object_contact"] >= CONTACT_N
    if program == "push":
        return counts.get("displaced", 0)
    if program == "topple":
        return counts.get("toppled", 0)
    if program == "door":
        return counts.get("opened", 0)
    if program == "press":
        held = sum(len(runs(touching[:, i], PRESS_STEPS, 3)) for i in range(touching.shape[1]))
        return counts.get("pressed", 0) + held
    if program == "slide":
        return len(runs(peak[:, PEAK_COLUMNS["effector_table"]] >= CONTACT_N, SLIDE_STEPS, 3))
    if program == "knock":
        strikes = episode.privileged["impact_pairs"].reshape(episode.length, -1).max(axis=1)
        return len(runs(strikes >= STRIKE_N, 1, 3))
    raise ValueError(program)


def summarise(directory: Path) -> dict[str, dict[str, float]]:
    """Per ``effector/program``: attempts, outcomes, shares, events per minute."""
    rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    for path in sorted((directory / "humanoid").glob("*.npz")):
        e = Episode.load(path)
        program = e.meta.get("program")
        stats = e.meta.get("plan_stats")
        if not program or not isinstance(stats, dict):
            continue
        attempts = int(sum(stats["attempts"].values()))
        peak = e.privileged["peak_force"]
        on_object = e.privileged["object_contact"].max(axis=1) >= CONTACT_N
        on_table = peak[:, PEAK_COLUMNS["effector_table"]] >= CONTACT_N
        counts = caused_change(e).counts
        rows[f"{e.meta['effector']}/{program}"].append(
            {
                "attempts": attempts,
                "outcomes": outcomes(e, str(program)),
                "object": float(on_object.mean()),
                "table": float(on_table.mean()),
                "wait": float(stats["wait_steps"]) / (2 * e.length),
                "events_per_minute": sum(counts.values()) / (e.length * e.dt / 60),
            }
        )
    out = {}
    for key, items in rows.items():
        attempts = sum(i["attempts"] for i in items)
        outcome = sum(i["outcomes"] for i in items)
        out[key] = {
            "episodes": len(items),
            "attempts_per_episode": attempts / len(items),
            "outcomes_per_episode": outcome / len(items),
            "outcomes_per_attempt": outcome / max(attempts, 1),
            "episodes_with_outcome": float(np.mean([i["outcomes"] > 0 for i in items])),
            "object": float(np.mean([i["object"] for i in items])),
            "table": float(np.mean([i["table"] for i in items])),
            "wait": float(np.mean([i["wait"] for i in items])),
            "events_per_minute": float(np.mean([i["events_per_minute"] for i in items])),
        }
    return out


def table(result: dict[str, dict[str, float]]) -> str:
    """Markdown table of ``summarise``."""
    header = (
        "| variant | n | attempts / ep | outcomes / ep | per attempt | eps with outcome "
        "| on object | on table | waiting | events / min |"
    )
    lines = [header, "|" + "---|" * 10]
    for key, r in sorted(result.items()):
        lines.append(
            f"| {key} | {r['episodes']:.0f} | {r['attempts_per_episode']:.1f} "
            f"| {r['outcomes_per_episode']:.1f} | {r['outcomes_per_attempt']:.2f} "
            f"| {r['episodes_with_outcome']:.2f} | {r['object']:.2f} | {r['table']:.2f} "
            f"| {r['wait']:.2f} | {r['events_per_minute']:.1f} |"
        )
    return "\n".join(lines)


def plot(result: dict[str, dict[str, float]], out: Path) -> None:
    """Left: where the steps go; right: outcomes per attempt."""
    keys = sorted(result)
    x = np.arange(len(keys))
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.5))
    for k, (name, label) in enumerate(
        (
            ("object", "an effector on an object"),
            ("table", "an effector on the table"),
            ("wait", "arm waiting"),
        )
    ):
        a.bar(x + (k - 1) * 0.27, [result[v][name] for v in keys], 0.27, label=label)
    a.set_xticks(x, keys, rotation=60)
    a.set_ylabel("share of steps")
    a.legend(fontsize=8)
    b.bar(x, [result[v]["outcomes_per_attempt"] for v in keys], color="tab:green")
    b.set_xticks(x, keys, rotation=60)
    b.set_ylabel("intended outcomes per attempt")
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def main() -> None:
    """Print the table; optionally plot."""
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    result = summarise(Path(sys.argv[1]))
    sys.stdout.write(table(result) + "\n")
    if "--plot" in sys.argv:
        plot(result, Path(sys.argv[sys.argv.index("--plot") + 1]))


if __name__ == "__main__":
    main()
