"""``reach_free`` (HumanoidBench's reach policy in numpy) and ``stand_supported``.

Needs the assets and the converted policy from ``make h1-assets`` (skipped otherwise).
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="needs the simulator extra: make setup")

import mujoco  # noqa: E402

from sensorium.bodies.h1_policies import Policy  # noqa: E402
from sensorium.bodies.h1_reach import ReachFreePolicy, ReachWeights, weights_available  # noqa: E402
from sensorium.bodies.h1_scene import (  # noqa: E402
    SUPPORTED_PELVIS_Z,
    TABLE_TOP_Z,
    assets_available,
    default_table_top,
    edge_base,
    external_dir,
    table_edges,
)
from sensorium.bodies.humanoid import HumanoidRecorder  # noqa: E402

needs_policy = pytest.mark.skipif(
    not (assets_available() and weights_available()),
    reason="reach policy missing: run `make h1-assets`",
)
ROOT = Path(__file__).resolve().parent.parent


def load_script(name: str) -> ModuleType:
    """A module loaded from ``scripts/`` (not a package)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@needs_policy
def test_converted_policy_is_well_formed() -> None:
    weights = ReachWeights.load()
    assert weights.w1.shape == (256, 61) and weights.w3.shape == (19, 256)
    assert weights.mean.shape == weights.var.shape == (61,) and (weights.var > 0).all()
    rng = np.random.default_rng(0)
    action = weights.action(weights.mean + np.sqrt(weights.var) * rng.normal(0, 1, 61))
    assert action.shape == (19,) and np.abs(action).max() <= 1.0 and np.isfinite(action).all()


@needs_policy
def test_checkpoint_reader_matches_the_converted_file() -> None:
    convert = load_script("convert_reach_policy")
    root = external_dir() / "humanoid-bench"
    checkpoint = root / "data" / "reach_two_hands" / "torch_model.pt"
    if not checkpoint.is_file():
        pytest.skip("the raw checkpoint is not in the checkout")
    state = convert.load_state_dict(checkpoint)
    weights = ReachWeights.load()
    np.testing.assert_allclose(state["dense1.weight"], weights.w1, atol=1e-7)
    np.testing.assert_allclose(state["dense3.bias"], weights.b3, atol=1e-7)


@needs_policy
def test_reach_free_ticks_at_50_hz_and_stays_finite() -> None:
    with HumanoidRecorder({"effector": "hand", "posture": "standing"}) as recorder:
        policy = recorder.make_policy("reach_free", np.random.default_rng(0))
        assert isinstance(policy, ReachFreePolicy)
        ticks: list[int] = []
        original = policy.tick

        def counting_tick(data: mujoco.MjData) -> np.ndarray:
            ticks.append(1)
            return original(data)

        def make(policy_name: str, rng: np.random.Generator) -> Policy:
            return policy

        policy.tick = counting_tick  # type: ignore[method-assign]
        recorder.make_policy = make  # type: ignore[method-assign,assignment]
        episode = recorder.record(3, "reach_free", 12)
        assert episode.meta["task"] == "stand" and episode.meta["end_reason"] in ("steps", "fall")
        steps = episode.length
        assert abs(len(ticks) - steps * 25 / 10) <= 3  # one tick per 10 physics steps
        assert np.isfinite(episode.action).all() and np.abs(episode.action).max() <= 1.0
        with pytest.raises(ValueError, match="standing"):
            HumanoidRecorder({"effector": "hand", "posture": "stand_supported"}).record(
                1, "reach_free", 2
            )


def test_supported_table_keeps_the_arm_geometry() -> None:
    assert default_table_top("stand_supported") == pytest.approx(
        TABLE_TOP_Z + SUPPORTED_PELVIS_Z - 0.64
    )
    near, far = table_edges(default_table_top("stand_supported"), edge_base("stand_supported"))
    assert near == pytest.approx(0.34) and far == 0.76


@needs_policy
def test_stand_supported_is_a_table_scene_at_standing_height() -> None:
    settings: dict[str, object] = {"effector": "pincer", "posture": "stand_supported"}
    with HumanoidRecorder(settings) as recorder:
        episode = recorder.record(1, "seated_babble", 6)
        assert episode.meta["table_top"] == pytest.approx(default_table_top("stand_supported"))
        assert episode.meta["task"] == "stand_table" and not episode.meta["fell"]
        height = episode.proprio[:, episode.proprio_names.index("pelvis_height")]
        assert abs(height.mean() - SUPPORTED_PELVIS_Z) < 0.02
        assert episode.privileged["object_pose"][0, :, 2].min() > episode.meta["table_top"]
        assert "stand supported" in " ".join(episode.text) or all(episode.text)
        with pytest.raises(ValueError, match="table posture"):
            HumanoidRecorder({"effector": "pincer", "posture": "standing"}).record(
                1, "seated_babble", 2
            )
