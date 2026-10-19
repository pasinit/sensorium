"""The caused-change detectors on hand-built episodes (no simulator needed)."""

import numpy as np
import pytest
from test_babble_stats import OBJECTS, STEPS, TOP, episode, resting_pose

from sensorium.caused_change import (
    EVENT_TYPES,
    CausedChange,
    caused_change,
    contact_bouts,
    excursions,
    joint_events,
    pushed_to_target_events,
    stack_states,
    stacking_events,
    summarise_events,
)

DT = 0.05


def touching_between(start: int, stop: int, index: int = 0) -> np.ndarray:
    """``[STEPS, OBJECTS]`` object-contact forces: object ``index`` touched in ``[start, stop)``."""
    force = np.zeros((STEPS, OBJECTS))
    force[start:stop, index] = 3.0
    return force


def test_a_pushed_object_is_displaced_but_a_touched_one_is_not() -> None:
    pose = resting_pose()
    pose[12:32, 0, 0] += np.linspace(0, 0.06, 20)  # slides 6 cm after the touch starts
    pose[32:, 0, 0] += 0.06
    moved = caused_change(episode(pose, np.zeros((STEPS, 4)), touching_between(10, 14)))
    assert moved.counts["displaced"] == 1
    still = caused_change(episode(resting_pose(), np.zeros((STEPS, 4)), touching_between(10, 14)))
    assert still.counts["displaced"] == 0


def test_a_sphere_that_rolls_is_not_a_topple_but_a_box_that_tips_is() -> None:
    names = ["obj_wood_box", "obj_plastic_sphere", "obj_metal_box"]
    pose = resting_pose()
    half = np.radians(60.0) / 2
    for index in (0, 1):  # a box (0) and a sphere (1) rotate about x by 60 degrees
        pose[15:, index, 3:7] = [np.cos(half), np.sin(half), 0.0, 0.0]
    force = touching_between(10, 14, 0) + touching_between(10, 14, 1)
    ep = episode(pose, np.zeros((STEPS, 4)), force)
    ep.meta["object_names"] = names
    assert caused_change(ep).counts["toppled"] == 1  # the box only


def test_lift_and_drop_are_counted() -> None:
    pose = resting_pose()
    pose[12:30, 0, 2] += 0.06  # held 6 cm up for 18 steps, then released
    force = touching_between(10, 30)
    result = caused_change(episode(pose, np.zeros((STEPS, 4)), force))
    assert result.counts["lifted"] == 1
    airborne = resting_pose()
    airborne[30:36, 0, 2] += 0.05  # then falls with nobody touching it
    free = caused_change(episode(airborne, np.zeros((STEPS, 4)), touching_between(10, 30)))
    assert free.counts["dropped"] == 1


def test_stacking_needs_rest_on_top_and_contact_and_unstacking_the_reverse() -> None:
    pose = resting_pose()
    pose[:, 1, :2] = pose[:, 0, :2]  # object 1 over object 0 from step 20 on, resting
    pose[:20, 1, :2] += 0.2
    pose[20:, 1, 2] = pose[20:, 0, 2] + 0.07
    pose[45:, 1, :2] += 0.2  # lifted off again
    state = stack_states(pose, DT)
    assert state[20:45, 1].all() or state[22:44, 1].all()
    force = touching_between(17, 21, 1) + touching_between(45, 48, 1)
    stacked, unstacked = stacking_events(state, force >= 0.5)
    assert (stacked, unstacked) == (1, 1)
    # without any contact it is not caused by the robot
    assert stacking_events(state, np.zeros((STEPS, OBJECTS), dtype=bool)) == (0, 0)


def test_prop_joints_and_target() -> None:
    travel = np.zeros((STEPS, 1))
    travel[30:, 0] = 1.0  # a drawer opens at step 30
    touched = np.zeros(STEPS, dtype=bool)
    touched[26:34] = True
    assert joint_events(travel, touched) == (1, 0)
    assert joint_events(travel, np.zeros(STEPS, dtype=bool)) == (0, 0)  # nobody touched it
    pose = resting_pose()
    target = np.array([0.7, 0.0, TOP])
    pose[-1, 0, :2] = target[:2]  # object 0 ends on the marker; it started 0.25+ m away
    pose[0, 0, :2] = [0.45, 0.1]
    assert pushed_to_target_events(pose, target, touching_between(10, 14) >= 0.5) == 1
    assert pushed_to_target_events(pose, target, np.zeros((STEPS, OBJECTS), dtype=bool)) == 0


def test_excursions_count_once_per_leave_and_return() -> None:
    angle = np.concatenate([np.zeros(5), np.full(5, 12.0), np.zeros(5), np.full(5, -12.0)])
    assert excursions(angle) == 2
    assert excursions(np.full(20, 8.0)) == 0  # never leaves the 10 degree band
    wobble = np.tile([0.0, 11.0], 10)  # comes back to 0, under the 5 degree hysteresis: each counts
    assert excursions(wobble) == 10


def test_summary_is_per_minute_with_coverage() -> None:
    items = [
        CausedChange(minutes=0.5, counts={"displaced": 10, "lifted": 1}),
        CausedChange(minutes=0.5, counts={"displaced": 14}),
    ]
    summary = summarise_events(items, resamples=200)
    assert summary["n"] == 2 and summary["minutes"] == pytest.approx(1.0)
    mean, low, high = summary["per_minute"]["displaced"]
    assert mean == pytest.approx(24.0) and low <= mean <= high
    assert summary["per_minute"]["lifted"][0] == pytest.approx(1.0)
    assert summary["coverage"] == 2  # displaced and lifted reach one per minute
    assert set(summary["per_minute"]) == set(EVENT_TYPES)


def test_bouts_bridge_short_gaps() -> None:
    touching = np.zeros((STEPS, OBJECTS), dtype=bool)
    touching[10:13, 0] = True
    touching[15:18, 0] = True  # a gap of two steps is bridged
    assert contact_bouts(touching)[0] == [(10, 18)]
