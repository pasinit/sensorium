"""The A1 gate: pass/fail lines on a generated babbling run, for what the data should show.

Usage: ``python -m sensorium.a1_gate <generated dir>`` (``make a1-gate A1_DIR=<dir>``). It reads the
seated H1 episodes under ``<dir>/humanoid``, groups them by effector, prints one line per check and
exits 1 if any check fails. Run it after every regeneration of the babbling data (docs/touch.md).

Thresholds, each with its reason. Change one only with a line in the changelog below, never after
looking at a result to make it pass.

* pincer forces: every episode's peak effector-object force <= 2 x 235 N. 235 N is the 2F-85's
  maximum grip force (Robotiq sheet); the factor 2 allows the transient of an impact.
* hand forces: Shadow publishes no fingertip force, so the yardstick is what an H1 arm can push
  with statically, 70 N (shoulder and elbow ``forcerange`` over the lever arm; docs/touch.md): the
  median of the episodes' peaks <= 70 N, the 90th percentile <= 2.5 x 70 N, the maximum <= 4 x 70 N.
* contact: share of steps with an effector on an object >= the mean of D0.3 step 3 at the 0.82 m
  table (hands 0.23, pincers 0.28); contact bouts per minute >= 10 (step 3's lowest, pincers).
* lifts: pincers >= 1 per episode (mean), the bar of the step-3 review; hands INFO (open problem).
* visibility: each tip outside the head camera's image in <= 10% of steps (the step-3 review).
* waiting: an arm without a plan (parked, holding) in <= 15% of its steps (the step-4 review).
* repertoire: every primitive of the effector is planned at least once per episode on average
  (count-based choice should spread over all of them; a primitive that never plans is broken).

Changelog:

* 2026-10-14, v1: first version. Hands: median <= 70 N and p90 <= 175 N; contact >= 0.20 (a guess
  below step 3's means); no waiting or repertoire checks.
* 2026-10-14, v2 (after the step-3 review's comments, same day): hands' max <= 280 N added next to
  the p90 (the 4 x yardstick of docs/touch.md); contact raised to step 3's means; waiting and
  repertoire checks added; the median criterion, dropped for a while after the first run, is back.
"""

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from sensorium.babble_stats import A1Stats, a1_stats
from sensorium.bodies.h1_babble import PRIMITIVES
from sensorium.bodies.h1_camera import tip_outside_shares
from sensorium.bodies.h1_scene import build_scene
from sensorium.episode import Episode

PINCER_FORCE_N = 2 * 235.0  # per episode: twice the 2F-85's maximum grip force (Robotiq sheet)
ARM_STATIC_N = 70.0  # what an H1 arm pushes with statically (docs/touch.md)
HAND_P90_N = 2.5 * ARM_STATIC_N  # hands: the 90th percentile of the per-episode peaks ...
HAND_MAX_N = 4.0 * ARM_STATIC_N  # ... and the largest; the median must be under ARM_STATIC_N
OBJECT_CONTACT_MIN = {"hand": 0.23, "pincer": 0.28}  # step 3 means, 0.82 m table
BOUTS_MIN = 10.0  # contact bouts per minute (step 3: 10 to 24)
PINCER_LIFTS_MIN = 1.0  # lifts per episode, mean (hands lift nothing yet: INFO)
OUTSIDE_MAX = 0.10  # share of steps a tip is outside the head camera's image, per tip
WAIT_MAX = 0.15  # share of an arm's steps spent without a plan


@dataclass(frozen=True)
class Check:
    """One gate line: ``status`` is PASS, FAIL or INFO."""

    status: str
    name: str
    detail: str


def forces(effector: str, stats: list[A1Stats]) -> Check:
    """Effector-object peak force per episode against the published range."""
    peaks = np.array([s.object_peak_n for s in stats])
    median, p90, top = (float(np.percentile(peaks, q)) for q in (50, 90, 100))
    detail = f"object peak N median {median:.0f}, p90 {p90:.0f}, max {top:.0f}"
    if effector == "pincer":
        return Check(
            "PASS" if top <= PINCER_FORCE_N else "FAIL",
            "forces",
            detail + f" (<= {PINCER_FORCE_N:.0f})",
        )
    ok = median <= ARM_STATIC_N and p90 <= HAND_P90_N and top <= HAND_MAX_N
    limits = f"median <= {ARM_STATIC_N:.0f}, p90 <= {HAND_P90_N:.0f}, max <= {HAND_MAX_N:.0f}"
    return Check("PASS" if ok else "FAIL", "forces", detail + f" ({limits})")


def contact(effector: str, stats: list[A1Stats]) -> list[Check]:
    """Object contact share and contact bouts per minute against the step-3 level."""
    floor = OBJECT_CONTACT_MIN[effector]
    rate = float(np.mean([s.object_contact_rate for s in stats]))
    bouts = float(np.mean([s.bouts_per_minute for s in stats]))
    return [
        Check(
            "PASS" if rate >= floor else "FAIL",
            "contact",
            f"object contact {rate:.2f} (>= {floor})",
        ),
        Check(
            "PASS" if bouts >= BOUTS_MIN else "FAIL",
            "bouts",
            f"{bouts:.1f} per minute (>= {BOUTS_MIN:g})",
        ),
    ]


def lifts(effector: str, stats: list[A1Stats]) -> Check:
    """Lifts per episode: gated for pincers; hands do not lift yet (reported)."""
    mean = float(np.mean([s.lifts for s in stats]))
    if effector == "pincer":
        return Check(
            "PASS" if mean >= PINCER_LIFTS_MIN else "FAIL",
            "lifts",
            f"{mean:.2f} per episode (>= {PINCER_LIFTS_MIN:g})",
        )
    return Check("INFO", "lifts", f"{mean:.2f} per episode (hands: open problem, not gated)")


def visibility(shares: list[np.ndarray]) -> Check:
    """Share of steps each tip is outside the head camera's image, over the group's episodes."""
    left, right = np.mean(shares, axis=0)
    ok = max(left, right) <= OUTSIDE_MAX
    detail = f"tips outside the image: left {left:.1%}, right {right:.1%} (<= {OUTSIDE_MAX:.0%})"
    return Check("PASS" if ok else "FAIL", "visibility", detail)


def waiting(stats: list[A1Stats]) -> Check:
    """Share of an arm's steps spent without a plan, over the group's episodes."""
    share = float(np.mean([s.wait_share for s in stats]))
    return Check(
        "PASS" if share <= WAIT_MAX else "FAIL",
        "waiting",
        f"{share:.1%} of arm steps (<= {WAIT_MAX:.0%})",
    )


def repertoire(effector: str, stats: list[A1Stats]) -> Check:
    """Every primitive planned at least once per episode on average."""
    totals: dict[str, list[int]] = defaultdict(list)
    for s in stats:
        counts = dict(s.plans_per_primitive)
        for name in PRIMITIVES[effector]:
            totals[name].append(counts.get(name, 0))
    means = {name: float(np.mean(values)) for name, values in totals.items()}
    detail = "plans per episode " + ", ".join(f"{k} {v:.1f}" for k, v in means.items())
    if not any(any(v) for v in totals.values()):
        return Check("FAIL", "repertoire", "no plan statistics in the episodes: regenerate")
    return Check(
        "PASS" if min(means.values()) >= 1.0 else "FAIL", "repertoire", detail + " (each >= 1)"
    )


def evaluate(directory: Path) -> dict[str, list[Check]]:
    """Every check for every effector found under ``directory/humanoid`` (seated episodes)."""
    stats: dict[str, list[A1Stats]] = defaultdict(list)
    outside: dict[str, list[np.ndarray]] = defaultdict(list)
    models: dict[tuple[str, float], mujoco.MjModel] = {}
    for path in sorted((directory / "humanoid").glob("humanoid-*-seated_babble-*.npz")):
        episode = Episode.load(path)
        effector = str(episode.meta["effector"])
        key = (effector, float(episode.meta["table_top"]))  # type: ignore[arg-type]
        if key not in models:
            scene = build_scene(effector, "seated", table_top=key[1])
            models[key] = mujoco.MjModel.from_xml_path(str(scene))
        stats[effector].append(a1_stats(episode))
        outside[effector].append(tip_outside_shares(episode, models[key]))
    if not stats:
        raise FileNotFoundError(f"no seated_babble episodes under {directory}/humanoid")
    return {
        effector: [
            forces(effector, stats[effector]),
            *contact(effector, stats[effector]),
            lifts(effector, stats[effector]),
            visibility(outside[effector]),
            waiting(stats[effector]),
            repertoire(effector, stats[effector]),
        ]
        for effector in sorted(stats)
    }


def main() -> None:
    """Print the gate lines; exit 1 if any check fails."""
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    failed = False
    for effector, checks in evaluate(Path(sys.argv[1])).items():
        for check in checks:
            sys.stdout.write(f"{check.status:<4}  {effector:<6}  {check.name:<10} {check.detail}\n")
            failed = failed or check.status == "FAIL"
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
