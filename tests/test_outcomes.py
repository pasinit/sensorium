"""Outcome labels and probes (``sensorium.model.outcomes``)."""

import numpy as np
import pytest

pytest.importorskip("torch", reason="needs the model extra: uv sync --extra model")

from test_model import privileged_episode  # noqa: E402

from sensorium.episode import Episode  # noqa: E402
from sensorium.model import outcomes, probes  # noqa: E402


def scripted(steps: int = 120) -> Episode:
    """Three objects at rest on a table (z 0.85): a box pushed 10 cm at steps 30-40 while
    touched, a sphere rolling untouched from step 60, a tall box lifted at 80-90 and tipped over
    at step 100."""
    ep = privileged_episode(0, steps=steps)
    pose = np.zeros((steps, 3, 7))
    pose[:, :, 2] = 0.85
    pose[:, :, 3] = 1.0  # upright, quaternion wxyz
    pose[:, 1, 0] = 0.3
    pose[:, 2, 0] = -0.3
    pose[30:, 0, 0] = np.minimum(np.arange(steps - 30), 10) * 0.01  # box: 10 cm in 10 steps
    pose[60:, 1, 0] += np.arange(steps - 60) * 0.01  # sphere: 20 cm/s, never touched
    pose[80:90, 2, 2] = 0.95  # tall box held 10 cm up
    pose[100:, 2, 3:7] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]  # tipped 90 degrees
    contact = np.zeros((steps, 3))
    contact[30:40, 0] = 2.0
    contact[78:92, 2] = 2.0
    ep.privileged |= {"object_pose": pose, "object_contact": contact}
    ep.meta |= {"object_names": ["obj_box", "obj_sphere", "obj_tallbox"]}
    return ep


def test_outcome_steps_follow_the_scripted_events() -> None:
    steps = probes.outcome_steps(scripted())
    displaced = np.flatnonzero(steps["displaced"])
    assert displaced.min() >= 30 - probes.OUTCOME_HORIZON and displaced.max() < 40
    assert not steps["displaced"][70:].any()  # the untouched sphere rolls: not a displacement
    assert np.array_equal(np.flatnonzero(steps["lifted"]), np.arange(80, 90))
    toppled = np.flatnonzero(steps["toppled"])
    assert toppled.min() == 100 - probes.OUTCOME_HORIZON and toppled.max() == 99
    assert steps["log_speed"][65] == pytest.approx(np.log1p(0.2 / probes.SPEED_SCALE))
    assert steps["log_speed"][10] == 0.0


def test_outcome_window_labels_and_probes(tmp_path: object) -> None:
    ep = scripted()
    ends = np.array([20, 35, 85, 110])
    lab = outcomes.outcome_labels(probes.outcome_steps(ep), ends, 4, 1)
    assert lab["lifted"].tolist() == [0, 0, 1, 0] and lab["displaced"].tolist()[:2] == [1, 1]
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(6), 40)
    y = {n: (rng.random(240) < 0.3).astype(float) for n in outcomes.OUTCOME_BINARY}
    y["log_speed"] = rng.normal(size=240)
    x = np.stack([y["displaced"], y["lifted"], y["toppled"], y["log_speed"]], 1)
    data = probes.ProbeData({"state": x + 0.1 * rng.normal(size=x.shape)}, y, groups)
    scores = outcomes.predict_outcomes(data, "state").scores()
    assert set(scores) == {
        "displaced_auroc",
        "lifted_auroc",
        "toppled_auroc",
        "log_speed_r2",
        "mean",
    }
    assert scores["mean"] > 0.8
    boot = outcomes.predict_outcomes(data, "state").bootstrap(3, np.random.default_rng(0))
    assert all("log_speed_r2" in b for b in boot)
