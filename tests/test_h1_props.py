"""Experiment props, hidden properties and the effort channels (needs the assets)."""

from collections.abc import Iterator

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="needs the simulator extra: make setup")

import mujoco  # noqa: E402

from sensorium.bodies.h1_props import (  # noqa: E402
    EXCLUSION,
    FRICTION,
    HIDDEN_COLUMNS,
    MASS_KG,
    SOLREF_S,
    draw_hidden,
    in_exclusion,
    prop_xml,
)
from sensorium.bodies.h1_scene import assets_available  # noqa: E402
from sensorium.bodies.humanoid import HumanoidRecorder  # noqa: E402
from sensorium.episode import Episode  # noqa: E402

needs_assets = pytest.mark.skipif(not assets_available(), reason="run `make h1-assets`")
SETTINGS: dict[str, object] = {
    "effector": "pincer",
    "posture": "seated",
    "props": ["friction_patches", "button", "hinged_flap", "target_disc"],  # a realistic set
    "hidden_properties": True,
    "effort": True,
}


def test_hidden_properties_are_drawn_inside_their_ranges() -> None:
    hidden = draw_hidden(np.random.default_rng(0), 2000)
    assert hidden.shape == (2000, len(HIDDEN_COLUMNS))
    for column, (low, high) in enumerate((MASS_KG, FRICTION, SOLREF_S)):
        assert hidden[:, column].min() >= low and hidden[:, column].max() <= high
    mass = np.log(hidden[:, 0])  # log-uniform: the log is symmetric about its middle
    assert abs(mass.mean() - (np.log(MASS_KG[0]) + np.log(MASS_KG[1])) / 2) < 0.1
    assert not np.array_equal(hidden, draw_hidden(np.random.default_rng(1), 2000))


def test_unknown_prop_is_rejected_and_zones_cover_the_big_props() -> None:
    with pytest.raises(ValueError, match="unknown props"):
        prop_xml(("trampoline",), 0.82)
    with pytest.raises(ValueError, match="overlap"):
        prop_xml(("drawer", "hinged_flap"), 0.82)
    assert in_exclusion(np.array([0.50, 0.22]), ("drawer",))
    assert in_exclusion(np.array([0.50, -0.22]), ("drawer",))  # mirrored
    assert not in_exclusion(np.array([0.60, 0.22]), ())
    assert set(EXCLUSION) == {"drawer", "hinged_flap", "button"}


def meta_list(episode: Episode, key: str) -> list[str]:
    """A list of strings from an episode's meta."""
    value = episode.meta[key]
    assert isinstance(value, list)
    return [str(v) for v in value]


@pytest.fixture(scope="module")
def recorder() -> Iterator[HumanoidRecorder]:
    """One recorder with every prop, hidden properties and the effort channels."""
    with HumanoidRecorder(SETTINGS) as made:
        yield made


@needs_assets
def test_props_move_under_force_and_stay_in_their_range(recorder: HumanoidRecorder) -> None:
    model, data = recorder.model, recorder.data
    for name, force in (("prop_flap_l", 0.3), ("prop_flap_r", 0.3), ("prop_button_l", 12.0)):
        rng = np.random.default_rng(
            0
        )  # the same start every time: an arm over a prop would block it
        recorder.reset(recorder.make_policy("seated_babble", rng), rng)
        joint = model.joint(name)
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[joint.dofadr[0]] = force
        for _ in range(500):  # 1 s
            mujoco.mj_step(model, data)
        low, high = model.jnt_range[joint.id]
        travel = (data.qpos[joint.qposadr[0]] - low) / (high - low)
        assert 0.5 < travel <= 1.05, f"{name} moved {travel:.2f} of its range"
        data.qfrc_applied[:] = 0.0


@needs_assets
def test_the_drawer_slides_under_force() -> None:
    settings = {**SETTINGS, "props": ["drawer", "target_disc"], "program": "drawer_open"}
    with HumanoidRecorder(settings) as drawer:
        model, data, rng = drawer.model, drawer.data, np.random.default_rng(0)
        drawer.reset(drawer.make_policy("experiment", rng), rng)  # the arms start clear of it
        joint = model.joint("prop_drawer_r")
        data.qfrc_applied[joint.dofadr[0]] = 6.0
        for _ in range(500):
            mujoco.mj_step(model, data)
        low, high = model.jnt_range[joint.id]
        assert 0.5 < (data.qpos[joint.qposadr[0]] - low) / (high - low) <= 1.05


@needs_assets
def test_episode_records_props_target_hidden_properties_and_effort(
    recorder: HumanoidRecorder,
) -> None:
    with HumanoidRecorder({**SETTINGS, "props": ["tall_block", "button", "target_disc"]}) as tall:
        names = meta_list(tall.record(3, "seated_babble", 2), "object_names")
    assert len(names) == 8 and "obj_wood_tallbox" in names and "obj_plastic_tallcylinder" in names
    episode = recorder.record(3, "seated_babble", 12)
    n = len(meta_list(episode, "object_names"))
    assert meta_list(episode, "prop_names") == [
        "prop_flap_l",
        "prop_flap_r",
        "prop_button_l",
        "prop_button_r",
    ]
    priv = episode.privileged
    assert priv["prop_state"].shape == (12, 4)
    assert priv["target"].shape == (12, 3) and np.allclose(priv["target"], priv["target"][0])
    assert priv["hidden_properties"].shape == (12, n, 3)
    assert episode.meta["hidden_property_columns"] == HIDDEN_COLUMNS
    eff = [i for i, name in enumerate(episode.proprio_names) if name.startswith("eff_")]
    assert len(eff) == recorder.model.nu and eff == list(range(eff[0], eff[0] + len(eff)))
    assert episode.proprio.shape[1] == len(episode.proprio_names)


@needs_assets
def test_hidden_properties_follow_the_seed_and_are_restored(recorder: HumanoidRecorder) -> None:
    first = recorder.record(3, "seated_babble", 4).privileged["hidden_properties"][0]
    again = recorder.record(3, "seated_babble", 4).privileged["hidden_properties"][0]
    other = recorder.record(4, "seated_babble", 4).privileged["hidden_properties"][0]
    assert np.array_equal(first, again) and not np.array_equal(first, other)
    objects = recorder.objects
    assert np.allclose(recorder.model.body_mass[objects], other[:, 0], rtol=1e-5)
    recorder.hidden_properties = False  # the next reset restores the nominal values
    recorder.apply_props()
    assert np.allclose(recorder.model.body_mass[objects], recorder.nominal["mass"])
    assert np.allclose(
        recorder.model.geom_friction[recorder.object_geoms, 0], recorder.nominal["friction"]
    )
    recorder.hidden_properties = True


@needs_assets
def test_objects_keep_out_of_the_prop_zones(recorder: HumanoidRecorder) -> None:
    """No object starts deep inside a zone (3 cm in on every side); a slippery one may settle
    over the edge."""
    props = ("hinged_flap", "button")
    around = np.array([[0.03, 0.0], [-0.03, 0.0], [0.0, 0.03], [0.0, -0.03]])
    for seed in range(5):
        poses = recorder.record(seed, "seated_babble", 2).privileged["object_pose"][0, :, :2]
        for i, xy in enumerate(poses):
            deep = in_exclusion(xy, props) and all(in_exclusion(xy + d, props) for d in around)
            assert not deep, (seed, i, xy)


@needs_assets
def test_effort_is_noisy_clipped_and_quantised(recorder: HumanoidRecorder) -> None:
    recorder.effort_rng = np.random.default_rng(0)
    recorder.data.actuator_force[:] = 0.0
    readings = np.stack([recorder.effort_reading() for _ in range(300)])
    rng = recorder.effort_range
    assert (np.abs(readings) <= rng + 1e-9).all()
    sigma = readings.std(axis=0) / rng
    assert 0.005 < sigma[sigma > 0].mean() < 0.02  # about 1% of the range
    step = 2 * rng / 4095  # quantised to 12 bits
    assert np.allclose(readings / step, np.round(readings / step))


@needs_assets
def test_default_scene_is_unchanged_by_the_new_settings() -> None:
    with HumanoidRecorder({"effector": "pincer", "posture": "seated"}) as plain:
        episode = plain.record(2, "seated_babble", 6)
        assert len(meta_list(episode, "object_names")) == 6
        assert "props" not in plain.settings and "effort" not in plain.settings
        assert not any(name.startswith("eff_") for name in episode.proprio_names)
        assert "prop_state" not in episode.privileged and not episode.meta["prop_names"]
