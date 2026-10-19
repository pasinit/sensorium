"""The A1 gate's checks on hand-built statistics (no simulator needed)."""

from dataclasses import replace
from typing import Any

import numpy as np
from test_babble_stats import OBJECTS, STEPS, episode, resting_pose

from sensorium.a1_gate import contact, forces, lifts, repertoire, visibility, waiting
from sensorium.babble_stats import PEAK_COLUMNS, A1Stats, a1_stats


def stats_with(**changes: Any) -> list[A1Stats]:
    """One scripted episode's statistics with some fields replaced."""
    base = a1_stats(episode(resting_pose(), np.zeros((STEPS, 4)), np.zeros((STEPS, OBJECTS))))
    return [replace(base, **changes)]


def test_pincer_forces_are_gated_at_twice_the_maximum_grip() -> None:
    assert forces("pincer", stats_with(object_peak_n=300.0)).status == "PASS"
    assert forces("pincer", stats_with(object_peak_n=500.0)).status == "FAIL"


def test_hand_forces_are_gated_on_the_arm_bound() -> None:
    assert forces("hand", stats_with(object_peak_n=60.0)).status == "PASS"
    assert forces("hand", stats_with(object_peak_n=150.0)).status == "FAIL"  # median over 70 N
    assert forces("hand", stats_with(object_peak_n=300.0)).status == "FAIL"


def test_contact_floors() -> None:
    good = contact("pincer", stats_with(object_contact_rate=0.30, bouts_per_minute=12.0))
    assert [c.status for c in good] == ["PASS", "PASS"]
    assert contact("pincer", stats_with(object_contact_rate=0.25))[0].status == "FAIL"  # < 0.28
    assert contact("hand", stats_with(object_contact_rate=0.25))[0].status == "PASS"  # >= 0.23
    bad = contact("hand", stats_with(object_contact_rate=0.1, bouts_per_minute=5.0))
    assert [c.status for c in bad] == ["FAIL", "FAIL"]


def test_lifts_gate_pincers_and_only_report_for_hands() -> None:
    assert lifts("pincer", stats_with(lifts=2)).status == "PASS"
    assert lifts("pincer", stats_with(lifts=0)).status == "FAIL"
    assert lifts("hand", stats_with(lifts=0)).status == "INFO"


def test_visibility_is_gated_per_tip() -> None:
    assert visibility([np.array([0.05, 0.08])]).status == "PASS"
    assert visibility([np.array([0.05, 0.14])]).status == "FAIL"


def test_object_peak_is_the_effector_object_column() -> None:
    peak = np.zeros((STEPS, 4))
    peak[3, PEAK_COLUMNS["effector_object"]] = 42.0
    peak[4, PEAK_COLUMNS["limb_scene"]] = 99.0
    stats = a1_stats(episode(resting_pose(), peak, np.zeros((STEPS, OBJECTS))))
    assert stats.object_peak_n == 42.0


def test_waiting_and_repertoire_are_gated() -> None:
    assert waiting(stats_with(wait_share=0.10)).status == "PASS"
    assert waiting(stats_with(wait_share=0.30)).status == "FAIL"
    full = (("grasp", 2), ("hover", 3), ("poke", 1), ("slide", 2), ("touch", 4))
    assert repertoire("pincer", stats_with(plans_per_primitive=full)).status == "PASS"
    missing = (("grasp", 0), ("hover", 3), ("poke", 1), ("slide", 2), ("touch", 4))
    assert repertoire("pincer", stats_with(plans_per_primitive=missing)).status == "FAIL"
    assert repertoire("pincer", stats_with(plans_per_primitive=())).status == "FAIL"  # old episode
