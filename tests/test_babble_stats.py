"""A1 babbling statistics on hand-built episodes (no simulator needed)."""

import numpy as np
import pytest

from sensorium.babble_stats import (
    BOUT_GAP,
    PEAK_COLUMNS,
    a1_stats,
    attributed_motion,
    count_lifts,
    knn_entropy,
    moving_objects,
    object_cell_ids,
    runs,
    skin_regions,
    summarise,
    table_cell_count,
    tilt_degrees,
    voxel_ids,
)
from sensorium.episode import Episode

STEPS, OBJECTS, TOP = 60, 3, 0.8


def episode(
    pose: np.ndarray, peak: np.ndarray, by_object: np.ndarray, tips: np.ndarray | None = None
) -> Episode:
    """A minimal seated episode: only the fields ``a1_stats`` reads are meaningful."""
    t = pose.shape[0]
    tips = (
        np.tile([[0.5, 0.2, TOP + 0.2], [0.5, -0.2, TOP + 0.2]], (t, 1, 1))
        if tips is None
        else tips
    )
    return Episode(
        body="humanoid",
        dt=0.05,
        rgb=np.zeros((t, 1, 1, 3), dtype=np.uint8),
        audio=np.zeros((t, 1), dtype=np.int16),
        audio_logmel=np.zeros((t, 1), dtype=np.float16),
        proprio=np.zeros((t, 1), dtype=np.float32),
        touch=np.zeros((t, 1), dtype=np.float32),
        action=np.zeros((t, 2), dtype=np.float32),
        proprio_names=["p"],
        touch_names=["t"],
        action_names=["a", "b"],
        privileged={
            "peak_force": peak.astype(np.float32),
            "object_contact": by_object.astype(np.float32),
            "object_pose": pose.astype(np.float32),
            "tip_position": tips.astype(np.float32),
        },
        meta={
            "table_top": TOP,
            "reach_x": [0.4, 0.62],
            "object_names": ["obj_wood_box", "obj_plastic_sphere", "obj_metal_box"],
            "fell": False,
        },
    )


def resting_pose() -> np.ndarray:
    """Three objects resting on the table, upright (wxyz quaternion)."""
    pose = np.zeros((STEPS, OBJECTS, 7))
    pose[:, :, :3] = [[0.45, 0.1, TOP + 0.03], [0.5, -0.1, TOP + 0.03], [0.55, 0.0, TOP + 0.03]]
    pose[:, :, 3] = 1.0
    return pose


def test_runs_bridge_gaps_and_drop_short_runs() -> None:
    flags = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1], dtype=bool)
    assert runs(flags) == [(1, 3), (5, 6), (11, 12)]
    assert runs(flags, min_length=2, max_gap=2) == [(1, 6)]


def test_tilt_and_motion() -> None:
    upright = np.array([1.0, 0, 0, 0])
    on_side = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])  # 90 degrees about x
    assert tilt_degrees(upright) == pytest.approx(0.0)
    assert tilt_degrees(on_side) == pytest.approx(90.0)
    pose = resting_pose()
    pose[10:, 0, 0] += 0.1  # object 0 jumps 10 cm in one step, then rests
    moving = moving_objects(pose, 0.05)
    assert moving[10, 0] and not moving[11:, 0].any() and not moving[:, 1:].any()


def test_count_lifts_needs_height_duration_and_contact() -> None:
    height = np.zeros(40)
    height[10:25] = 0.06
    touched = np.zeros(40, dtype=bool)
    assert count_lifts(height, touched) == 0  # lifted, but nothing touched it
    touched[9:12] = True
    assert count_lifts(height, touched) == 1
    height[10:25] = 0.06
    height[15:25] = 0.0  # held for only 5 steps
    assert count_lifts(height, touched) == 0


def test_a1_stats_on_a_scripted_episode() -> None:
    pose = resting_pose()
    peak = np.zeros((STEPS, 4))
    by_object = np.zeros((STEPS, OBJECTS))
    by_object[10:20, 0] = 3.0  # a bout on object 0
    by_object[30:34, 2] = 3.0  # a bout on object 2
    pose[12:30, 0, 2] += 0.06  # object 0 is lifted while held, then dropped
    pose[12:30, 0, 0] += np.linspace(0, 0.05, 18)
    peak[10:20, PEAK_COLUMNS["effector_object"]] = 3.0
    peak[5, PEAK_COLUMNS["limb_scene"]] = 150.0  # one slam
    peak[7, PEAK_COLUMNS["self"]] = 20.0
    tips = np.tile([[0.5, 0.2, TOP + 0.05], [0.5, -0.2, TOP + 0.2]], (STEPS, 1, 1))
    stats = a1_stats(episode(pose, peak, by_object, tips))
    assert stats.steps == STEPS
    assert stats.object_contact_rate == pytest.approx(14 / STEPS)
    assert stats.objects_touched == 2
    assert stats.bouts_per_minute == pytest.approx(2 / (STEPS * 0.05 / 60))
    assert stats.lifts == 1
    assert stats.in_air_rate == pytest.approx(18 / STEPS)
    assert stats.limb_peak_n == 150.0 and stats.limb_slam_steps == 1
    assert stats.self_peak_n == 20.0 and stats.self_rate == pytest.approx(1 / STEPS)
    assert stats.free_space_rate == 0.5  # one tip hovers 20 cm up, the other 5 cm
    assert 0 < stats.voxel_coverage < 0.1
    assert len(stats.object_cells) >= 3
    assert BOUT_GAP >= 1


def test_a1_stats_rejects_old_episodes() -> None:
    old = episode(resting_pose(), np.zeros((STEPS, 4)), np.zeros((STEPS, OBJECTS)))
    del old.privileged["peak_force"]
    with pytest.raises(KeyError, match="regenerate"):
        a1_stats(old)
    old.meta.pop("table_top")
    old.privileged["peak_force"] = np.zeros((STEPS, 4), dtype=np.float32)
    with pytest.raises(KeyError, match="table_top"):
        a1_stats(old)


def test_coverage_helpers() -> None:
    tips = np.array([[[0.45, 0.0, TOP + 0.01], [0.45, 0.01, TOP + 0.01]]] * 3)  # one voxel
    assert len(set(voxel_ids(tips, TOP, (0.4, 0.62)).tolist())) == 1
    assert len(voxel_ids(tips + [0.0, 0.0, 1.0], TOP, (0.4, 0.62))) == 0  # above the box
    pose = resting_pose()
    edges = (0.24, 0.76)
    assert len(object_cell_ids(pose, edges)) == 3
    pose[:, 0, 0] = 5.0  # off the table: dropped
    assert len(object_cell_ids(pose, edges)) == 2
    assert table_cell_count(edges) == 11 * 20


def test_sound_pairs_from_logged_impacts() -> None:
    voices = [
        "feet",
        "legs",
        "torso",
        "arms",
        "hands",
        "floor",
        "table",
        "wood",
        "plastic",
        "metal",
        "rubber",
    ]
    episode_ = episode(resting_pose(), np.zeros((STEPS, 4)), np.zeros((STEPS, OBJECTS)))
    impacts = np.zeros((STEPS, len(voices), len(voices)), dtype=np.float32)
    impacts[5, voices.index("hands"), voices.index("metal")] = 3.0
    impacts[9, voices.index("arms"), voices.index("table")] = 40.0
    impacts[9, voices.index("feet"), voices.index("floor")] = 40.0  # not a table pair: ignored
    impacts[12, voices.index("hands"), voices.index("wood")] = 0.1  # too weak
    episode_.privileged["impact_pairs"] = impacts
    episode_.meta["impact_voices"] = voices
    stats = a1_stats(episode_)
    assert stats.sound_pairs == {("hands", "metal"), ("arms", "table")}
    assert summarise([stats], resamples=5)["sound_pair_coverage"][0] == pytest.approx(2 / 10)
    assert (
        a1_stats(
            episode(resting_pose(), np.zeros((STEPS, 4)), np.zeros((STEPS, OBJECTS)))
        ).sound_pairs
        == frozenset()
    )


def test_summarise_gives_mean_and_interval() -> None:
    pose = resting_pose()
    quiet = a1_stats(episode(pose, np.zeros((STEPS, 4)), np.zeros((STEPS, OBJECTS))))
    summary = summarise([quiet, quiet, quiet], resamples=50)
    mean, low, high = summary["object_contact_rate"]
    assert mean == low == high == 0.0
    assert "object_cell_coverage" in summary and "object_cells" not in summary


def test_attributed_motion_needs_a_preceding_contact() -> None:
    moving = np.zeros((60, 2), dtype=bool)
    touching = np.zeros((60, 2), dtype=bool)
    moving[10:30, 0] = True  # object 0 is poked, then rolls on
    touching[8:11, 0] = True
    moving[20:50, 1] = True  # object 1 rolls by itself
    touching[55, 1] = True  # touched only long after
    attributed, unattributed = attributed_motion(moving, touching)
    assert attributed[:, 0].sum() == 20 and not attributed[:, 1].any()
    assert unattributed[:, 1].sum() == 30 and not unattributed[:, 0].any()
    assert (attributed | unattributed).tolist() == moving.tolist()


def test_unattributed_rolling_is_not_object_motion() -> None:
    pose = resting_pose()
    pose[5:55, 1, 0] += np.linspace(0, 0.5, 50)  # a ball rolls for 2.5 s, nobody touched it
    stats = a1_stats(episode(pose, np.zeros((STEPS, 4)), np.zeros((STEPS, OBJECTS))))
    assert stats.object_motion_rate == 0.0 and stats.unattributed_motion_rate > 0.7


def test_knn_entropy_ranks_spread() -> None:
    rng = np.random.default_rng(0)
    spread = rng.uniform(0, 1, (600, 3))
    tight = rng.uniform(0, 0.1, (600, 3))
    assert abs(knn_entropy(spread)) < 0.4  # a unit cube: log volume 0, a little boundary bias
    assert knn_entropy(tight) == pytest.approx(knn_entropy(spread) + 3 * np.log(0.1), abs=0.3)
    still = np.tile([0.5, 0.0, 0.8], (50, 1))
    assert np.isfinite(knn_entropy(still))  # repeated positions do not give -inf


def test_skin_regions_group_dense_channels() -> None:
    names = ["body/pelvis/0", "hand/left/ffdistal/0", "hand/left/ffdistal/1", "hand/left/palm/0",
             "pincer/left/left_pad/0", "dynamics/hand/left/ffdistal/slip"]  # fmt: skip
    touch = np.zeros((4, len(names)), dtype=np.float32)
    touch[2, 2] = 1.0  # one channel of the fingertip
    touch[3, 0] = 5.0  # the body grid does not count
    touch[1, 5] = 9.0  # nor a dynamics channel
    touched, total = skin_regions(names, touch)
    assert touched == {"hand/left/ffdistal"} and total == 3
