"""The event-rate metric on a scripted object history, and the scene director (assets)."""

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from sensorium.episode import Episode
from sensorium.event_rate import event_rate, event_steps

TOP = 0.82


def scripted(steps: int = 200) -> Episode:
    """One cube that rests, is lifted 10 cm with contact, let go, falls back; one impact."""
    z = np.full(steps, TOP + 0.03)
    z[40:80] += np.linspace(0, 0.10, 40)  # lifted with contact
    z[80:90] = TOP + 0.13 - 0.5 * 9.81 * (np.arange(10) * 0.05) ** 2  # falling, untouched
    z[90:] = TOP + 0.03
    z = np.maximum(z, TOP + 0.03)
    pose = np.zeros((steps, 1, 7))
    pose[:, 0, 2], pose[:, 0, 3] = z, 1.0
    contact = np.zeros((steps, 1))
    contact[35:80] = 5.0
    peaks = np.zeros((steps, 4))
    peaks[150, 2] = 50.0  # an arm hits the table: one impact
    episode = SimpleNamespace(
        dt=0.05,
        length=steps,
        privileged={"object_pose": pose, "object_contact": contact, "peak_force": peaks},
        meta={
            "table_top": TOP,
            "object_names": ["obj_wood_box"],
            "peak_force_columns": ["effector_object", "effector_table", "limb_scene", "self"],
        },
    )
    return cast(Episode, episode)


def test_scripted_events_are_found_once_each() -> None:
    counts, motion = event_steps(scripted())
    totals = {k: int(v.sum()) for k, v in counts.items()}
    assert totals["contact_start"] == 1 and totals["contact_end"] == 1
    assert totals["grasp"] == 1 and totals["lift"] == 1
    assert totals["drop"] == 1 and totals["impact"] == 1
    assert totals["fall"] == 0 and totals["topple"] == 0
    assert motion[50] and not motion[20] and not motion[120]


def test_windows_with_nothing_happening_are_rest() -> None:
    rate = event_rate(scripted())
    assert len(rate.windows) == 10  # 200 steps of 50 ms in 1 s windows
    assert 0.0 < rate.rest < 1.0
    assert rate.eventful == pytest.approx(1.0 - rate.rest)
    assert rate.per_minute["lift"] == pytest.approx(1 / (200 * 0.05 / 60))
