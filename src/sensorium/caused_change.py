"""The caused-change metric: what the robot made happen, counted per minute by outcome type.

The headline of Track A's experiments (docs/experiments.md): contact share and the share of
steps with something moving both saturate in random play, so the data is judged by **events per
minute of each outcome type** and by **outcome-type coverage** (how many types occur at least
once a minute). Everything is computed from privileged state (``object_pose``, ``object_contact``,
prop joints, proprio) at analysis time and never reaches a model.

The thresholds are fixed here with their reasons, and change only with a line in the changelog:

* displaced: net xy displacement of an object from the onset of a contact bout to ``WINDOW_STEPS``
  after it ends > 3 cm (a third of the smallest object, above resting noise);
* lifted: ``babble_stats.count_lifts`` (5 cm held for 0.5 s with contact);
* toppled: a *non-spherical* object's tilt crosses 45 degrees inside a bout plus the window (a
  sphere's tilt is meaningless while it rolls; a first baseline that counted it was an upper bound);
* dropped: an object 3 cm above its rest height with no effector on it, for at least 2 steps;
* stacked / unstacked: an object resting (under 2 cm/s for 1 s) with its centre within 2 cm in xy
  of another and more than 2 cm above it (objects cannot interpenetrate, so this is on top of);
  unstacked is the same state ending, both with contact inside the window;
* opened / closed: the drawer or the flap (``prop_state`` columns named ``prop_drawer`` and
  ``prop_flap_*``) travels at least half its range, contact within the window; pressed: the button
  (``prop_button_*``) goes past half its travel, same rule; zero when the episode has no props;
* pushed to target: an object ends within 3 cm of the target marker having started 15 cm or more
  away (``target`` in ``privileged``); zero without a marker;
* own-body change: waist yaw or pelvis lean moves more than 10 degrees from its start, counted
  once per excursion (5 degrees of hysteresis).

Changelog:

* 2026-10-15, v1: first version, written before any experiment program exists.
* 2026-10-15, v3 (before the S2 data is judged): the coverage count (types at 1 per minute or more)
  leaves out ``body_moved``, which needs waist work that is deferred, and the bar is 6 of the 9
  remaining types (``COVERAGE_BAR``); ``summarise_events`` reports the count.
* 2026-10-15, v2 (S1, still before any experiment episode): ``pressed`` added for the spring button,
  so that a button is not counted as an opening; opened and closed use the drawer and the flap only.
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from sensorium.babble_stats import (
    BOUT_GAP,
    CONTACT_N,
    MOVING_MPS,
    count_lifts,
    runs,
    tilt_degrees,
)
from sensorium.episode import Episode

EVENT_TYPES = (
    "displaced",
    "lifted",
    "toppled",
    "dropped",
    "stacked",
    "unstacked",
    "opened",
    "closed",
    "pressed",
    "pushed_to_target",
    "body_moved",
)
WINDOW_STEPS = 20  # 1 s at 20 Hz: the attribution window of babble_stats
DISPLACED_M = 0.03
TOPPLE_DEG = 45.0
FREE_FALL_M = 0.03  # height above rest of a dropped object
STACK_XY_M = 0.02
STACK_DZ_M = 0.02
REST_STEPS = 20  # a stacked object rests this long
TARGET_M = 0.03  # an object this close to the marker is on it
TARGET_START_M = 0.15  # ... having started this far away
JOINT_FRACTION = 0.5  # of its range a prop joint must travel
BODY_DEG = 10.0
BODY_HYSTERESIS_DEG = 5.0
COVERAGE_PER_MINUTE = 1.0  # a type counts toward coverage at this rate
# Coverage counts the types the repertoire can reach: body_moved needs the waist or the pelvis to
# move, which is not built (the supervisor's decision of 2026-10-15), so it is left out. The bar is
# 6 of these 9 types (it was "6 of 8" in the design memo, before unstacked and closed were added).
COVERAGE_TYPES = tuple(t for t in EVENT_TYPES if t != "body_moved")
COVERAGE_BAR = 6


def spherical(names: Sequence[str]) -> np.ndarray:
    """``[n]`` True for objects whose orientation means nothing (spheres)."""
    return np.array(["sphere" in str(n) for n in names], dtype=bool)


def contact_bouts(touching: np.ndarray) -> list[list[tuple[int, int]]]:
    """Per object, its contact bouts ``(start, stop)`` (gaps of ``BOUT_GAP`` steps bridged)."""
    return [runs(touching[:, i], 2, BOUT_GAP) for i in range(touching.shape[1])]


def displaced_events(
    pose: np.ndarray, bouts: list[list[tuple[int, int]]], window: int = WINDOW_STEPS
) -> int:
    """Contact bouts after which the object ends up more than ``DISPLACED_M`` from where it began.

    Args:
        pose: ``[T, n, 7]`` object poses.
        bouts: ``contact_bouts`` of the same episode.
    """
    last = pose.shape[0] - 1
    count = 0
    for i, object_bouts in enumerate(bouts):
        for start, stop in object_bouts:
            end = min(stop + window, last)
            count += int(np.linalg.norm(pose[end, i, :2] - pose[start, i, :2]) > DISPLACED_M)
    return count


def toppled_events(
    pose: np.ndarray,
    bouts: list[list[tuple[int, int]]],
    round_objects: np.ndarray,
    window: int = WINDOW_STEPS,
) -> int:
    """Bouts during which (or within the window after) a non-spherical object tips past 45 deg."""
    tilt = tilt_degrees(pose[:, :, 3:7])
    last = pose.shape[0] - 1
    count = 0
    for i, object_bouts in enumerate(bouts):
        if round_objects[i]:
            continue
        for start, stop in object_bouts:
            end = min(stop + window, last)
            count += int(tilt[start, i] < TOPPLE_DEG <= tilt[start : end + 1, i].max())
    return count


def dropped_events(pose: np.ndarray, touching: np.ndarray) -> int:
    """Free-fall bouts: an object at least ``FREE_FALL_M`` above its rest height, untouched."""
    rest = pose[:3, :, 2].mean(axis=0)
    airborne = (pose[:, :, 2] - rest >= FREE_FALL_M) & ~touching
    return sum(len(runs(airborne[:, i], 2)) for i in range(pose.shape[1]))


def stack_states(pose: np.ndarray, dt: float) -> np.ndarray:
    """``[T, n]`` True while object ``i`` rests on another (resting for ``REST_STEPS`` steps)."""
    steps, n = pose.shape[:2]
    speed = np.zeros((steps, n))
    speed[1:] = np.linalg.norm(np.diff(pose[:, :, :3], axis=0), axis=-1) / dt
    resting = speed < MOVING_MPS
    on_top = np.zeros((steps, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            gap = np.linalg.norm(pose[:, i, :2] - pose[:, j, :2], axis=-1)
            on_top[:, i] |= (gap < STACK_XY_M) & (pose[:, i, 2] - pose[:, j, 2] > STACK_DZ_M)
    state = np.zeros((steps, n), dtype=bool)
    for i in range(n):
        for start, stop in runs(on_top[:, i] & resting[:, i], REST_STEPS):
            state[start:stop, i] = True
    return state


def stacking_events(
    state: np.ndarray, touching: np.ndarray, window: int = WINDOW_STEPS
) -> tuple[int, int]:
    """``(stacked, unstacked)``: onsets and ends of the stacked state with contact in the window."""
    stacked = unstacked = 0
    for i in range(state.shape[1]):
        for start, stop in runs(state[:, i], 1):
            if touching[max(start - window, 0) : start + 1, i].any():
                stacked += 1
            if stop < state.shape[0] and touching[stop : stop + window + 1, i].any():
                unstacked += 1
    return stacked, unstacked


def joint_events(
    travel: np.ndarray, touching_any: np.ndarray, window: int = WINDOW_STEPS
) -> tuple[int, int]:
    """``(up, down)`` crossings of ``JOINT_FRACTION`` by joints' travel in ``[0, 1]``, ``[T, k]``.

    Each crossing needs an effector contact on any object or prop inside the window around it.
    """
    opened = closed = 0
    near = np.convolve(touching_any.astype(float), np.ones(2 * window + 1), mode="same") > 0
    above = travel >= JOINT_FRACTION
    for k in range(travel.shape[1]):
        crossing = np.flatnonzero(np.diff(above[:, k].astype(int)) != 0) + 1
        for t in crossing:
            if near[t]:
                opened += int(above[t, k])
                closed += int(not above[t, k])
    return opened, closed


def pushed_to_target_events(pose: np.ndarray, target: np.ndarray, touching: np.ndarray) -> int:
    """Objects that end on the marker having started far from it, with contact on the way."""
    count = 0
    for i in range(pose.shape[1]):
        start = np.linalg.norm(pose[0, i, :2] - target[:2])
        end = np.linalg.norm(pose[-1, i, :2] - target[:2])
        count += int(start >= TARGET_START_M and end <= TARGET_M and touching[:, i].any())
    return count


def excursions(angle_deg: np.ndarray) -> int:
    """Times ``angle_deg`` leaves ``+-BODY_DEG`` of its start; it must come back within
    ``BODY_HYSTERESIS_DEG`` of the band's edge to count again."""
    away = np.abs(angle_deg - angle_deg[0])
    count, out = 0, False
    for value in away:
        if not out and value > BODY_DEG:
            count, out = count + 1, True
        elif out and value < BODY_DEG - BODY_HYSTERESIS_DEG:
            out = False
    return count


def body_moved_events(episode: Episode) -> int:
    """Waist-yaw and pelvis-lean excursions of the episode's own body."""
    names = episode.proprio_names
    count = 0
    if "q_torso" in names:
        count += excursions(np.degrees(episode.proprio[:, names.index("q_torso")]))
    if all(f"gravity_{a}" in names for a in "xyz"):
        gravity = np.stack([episode.proprio[:, names.index(f"gravity_{a}")] for a in "xyz"], axis=1)
        lean = np.degrees(
            np.arccos(np.clip(-gravity[:, 2] / np.linalg.norm(gravity, axis=1), -1, 1))
        )
        count += excursions(lean)
    return count


@dataclass(frozen=True)
class CausedChange:
    """Event counts of one episode, and its length in minutes."""

    minutes: float
    counts: dict[str, int]

    def per_minute(self) -> dict[str, float]:
        """Events per minute by type."""
        return {t: self.counts.get(t, 0) / self.minutes for t in EVENT_TYPES}


def caused_change(episode: Episode) -> CausedChange:
    """Count the caused-change events of one episode (see the module docstring).

    Raises:
        KeyError: if the episode lacks ``privileged/object_pose`` or ``object_contact``.
    """
    priv = episode.privileged
    for key in ("object_pose", "object_contact"):
        if key not in priv:
            raise KeyError(f"episode has no privileged/{key}: regenerate it")
    pose = np.asarray(priv["object_pose"], dtype=float)
    touching = np.asarray(priv["object_contact"]) >= CONTACT_N
    names = episode.meta.get("object_names")
    round_objects = spherical(names if isinstance(names, list) else ["obj"] * pose.shape[1])
    bouts = contact_bouts(touching)
    rest = pose[:3, :, 2].mean(axis=0)
    counts: Counter[str] = Counter()
    counts["displaced"] = displaced_events(pose, bouts)
    counts["lifted"] = sum(
        count_lifts(pose[:, i, 2] - rest[i], touching[:, i]) for i in range(pose.shape[1])
    )
    counts["toppled"] = toppled_events(pose, bouts, round_objects)
    counts["dropped"] = dropped_events(pose, touching)
    counts["stacked"], counts["unstacked"] = stacking_events(
        stack_states(pose, episode.dt), touching
    )
    prop_state = priv.get("prop_state")
    prop_names = episode.meta.get("prop_names")
    if prop_state is not None and isinstance(prop_names, list):
        travel = np.asarray(prop_state, dtype=float)
        # a prop's own contacts are not in object_contact: the prop's effector contact is its
        # peak-force class (effector_table), so use any contact of the effector with a table geom
        near = touching.any(axis=1) | (np.asarray(priv["peak_force"])[:, 1] >= CONTACT_N)
        cols = [k for k, n in enumerate(prop_names) if n.startswith(("prop_drawer", "prop_flap"))]
        if cols:
            counts["opened"], counts["closed"] = joint_events(travel[:, cols], near)
        cols = [k for k, n in enumerate(prop_names) if n.startswith("prop_button")]
        if cols:
            counts["pressed"] = joint_events(travel[:, cols], near)[0]
    target = priv.get("target")
    if target is not None:
        counts["pushed_to_target"] = pushed_to_target_events(
            pose,
            np.asarray(target, dtype=float)[0],
            touching,  # [T, 3], the same every step
        )
    counts["body_moved"] = body_moved_events(episode)
    return CausedChange(minutes=episode.length * episode.dt / 60.0, counts=dict(counts))


def summarise_events(
    results: Iterable[CausedChange], resamples: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """Events per minute by type with a 95% bootstrap interval over episodes, and coverage.

    Returns:
        ``{"n", "minutes", "per_minute": {type: (mean, low, high)}, "coverage"}`` where the mean
        is total events over total minutes, and ``coverage`` is the number of types at or above
        ``COVERAGE_PER_MINUTE`` events per minute.
    """
    items = list(results)
    rng = np.random.default_rng(seed)
    counts = np.array([[r.counts.get(t, 0) for t in EVENT_TYPES] for r in items], dtype=float)
    minutes = np.array([r.minutes for r in items])
    mean = counts.sum(axis=0) / minutes.sum()
    boot = np.empty((resamples, len(EVENT_TYPES)))
    for b in range(resamples):
        pick = rng.integers(len(items), size=len(items))
        boot[b] = counts[pick].sum(axis=0) / minutes[pick].sum()
    low, high = np.percentile(boot, [2.5, 97.5], axis=0)
    return {
        "n": len(items),
        "minutes": float(minutes.sum()),
        "per_minute": {
            t: (float(mean[k]), float(low[k]), float(high[k])) for k, t in enumerate(EVENT_TYPES)
        },
        "coverage": int(
            sum(
                mean[k] >= COVERAGE_PER_MINUTE
                for k, t in enumerate(EVENT_TYPES)
                if t in COVERAGE_TYPES
            )
        ),
    }
