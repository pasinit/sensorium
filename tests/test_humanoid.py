"""The H1 humanoid body: skin naming, contact sound, and short end-to-end recordings.

The recording tests need the assets from ``make h1-assets`` and the simulator
extra (``make setup``); they are skipped, with the reason, when either is missing.
"""

from collections.abc import Iterator

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="needs the simulator extra: make setup")

from sensorium.audio import ContactSound, logmel, to_int16  # noqa: E402
from sensorium.bodies.h1_scene import SCENE_EXTENT, assets_available  # noqa: E402
from sensorium.bodies.h1_sense import VOICES, body_voice, surface_voice, taxel_prefix  # noqa: E402
from sensorium.bodies.humanoid import HumanoidRecorder  # noqa: E402
from sensorium.describe import describe_humanoid  # noqa: E402
from sensorium.episode import IMAGE_SIZE, MODEL_INPUT_KEYS, SAMPLE_RATE, model_inputs  # noqa: E402
from sensorium.generate import content_hash, episode_seed  # noqa: E402

needs_assets = pytest.mark.skipif(
    not assets_available(), reason="H1 assets missing: run `make h1-assets`"
)
STEPS = 20
VARIANTS = [
    ("hand", "standing", "hold"),
    ("hand", "seated", "seated_babble"),
    ("pincer", "standing", "hold"),
    ("pincer", "seated", "seated_babble"),
]
# Touch channels, measured (see docs/touch.md): the body's touch_grid taxels, the dense
# skin points of the hands or pincer pads, and 3 dynamics features per skin region.
CHANNELS = {
    "hand": {"grid": 184, "dense": 2429, "dynamics": 162},
    "pincer": {"grid": 184, "dense": 690, "dynamics": 72},
}
GRID_TAXELS = {"hand": 448, "pincer": 184}  # every touch_grid patch, in privileged/touch_xyz


def test_taxel_prefixes() -> None:
    assert taxel_prefix("left_hip_pitch_touch") == "body/left_hip_pitch"
    assert taxel_prefix("pelvis_touch") == "body/pelvis"
    assert taxel_prefix("lh_ffdistal_touch") == "hand/left/ffdistal"
    assert taxel_prefix("rh_palm_touch") == "hand/right/palm"
    assert taxel_prefix("lp_left_pad") == "pincer/left/left_pad"


def test_voices_and_labels() -> None:
    assert body_voice("left_ankle_link") == ("feet", "left foot")
    assert body_voice("rh_ffdistal") == ("hands", "right hand")
    assert body_voice("lp_right_pad") == ("hands", "left hand")
    assert body_voice("left_elbow_link") == ("arms", "left arm")
    assert body_voice("torso_link") == ("torso", "torso")
    assert surface_voice("obj_metal_box") == ("metal", "metal object")
    assert surface_voice("table_top") == ("table", "table")
    assert surface_voice("floor") == ("floor", "floor")
    assert surface_voice("wall2") == ("floor", "wall")
    assert {body_voice("pelvis")[0], surface_voice("obj_wood_box")[0]} <= set(VOICES)


def test_describe_humanoid() -> None:
    assert describe_humanoid("standing", [], False) == "standing"
    assert describe_humanoid("standing", [], True) == "falling"
    text = describe_humanoid(
        "seated", [("left hand", "table"), ("right arm", "metal object")], False
    )
    assert text == "seated; left hand touching table, right arm touching metal object"


def test_episode_seed_variants() -> None:
    assert episode_seed(0, "humanoid", "hold", 0) == episode_seed(0, "humanoid", "hold", 0, "")
    seeds = {episode_seed(0, "humanoid", "seated_babble", 0, v) for v in ("hand", "pincer", "")}
    assert len(seeds) == 3


def contact_chunks(inputs: list[tuple[float, float, str]], seed: int = 0) -> np.ndarray:
    """Render (impact N, slide, voice) steps with ContactSound into int16 ``[T, 800]``."""
    sound = ContactSound(np.random.default_rng(seed), n_samples=800)
    chunks = []
    for impact, slide, voice in inputs:
        hits, slides = np.zeros(len(VOICES)), np.zeros(len(VOICES))
        hits[VOICES.index(voice)], slides[VOICES.index(voice)] = impact, slide
        chunks.append(sound.step(hits, slides))
    return to_int16(chunks)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt((audio.astype(float) ** 2).mean()))


def test_contact_sound_is_deterministic_and_responds_to_contact() -> None:
    quiet = [(0.0, 0.0, "hands")] * 3
    hit = [(0.0, 0.0, "hands"), (30.0, 0.0, "hands"), (0.0, 0.0, "hands")]
    slide = [(0.0, 0.5, "table")] * 3
    np.testing.assert_array_equal(contact_chunks(hit, 1), contact_chunks(hit, 1))
    assert not np.array_equal(contact_chunks(hit, 1), contact_chunks(hit, 2))
    assert rms(contact_chunks(quiet)) < 50  # room tone only
    assert rms(contact_chunks(hit)[1:2]) > 10 * rms(contact_chunks(quiet))
    assert rms(contact_chunks(slide)) > 10 * rms(contact_chunks(quiet))
    assert logmel(contact_chunks(hit)).shape == (3, 32)


def test_materials_sound_different() -> None:
    wood = contact_chunks([(30.0, 0.0, "wood")], 5)
    metal = contact_chunks([(30.0, 0.0, "metal")], 5)
    spectrum = [np.abs(np.fft.rfft(x[0].astype(float))) for x in (wood, metal)]
    freqs = np.fft.rfftfreq(800, 1 / SAMPLE_RATE)
    centroid = [float((s * freqs).sum() / s.sum()) for s in spectrum]
    assert centroid[1] > centroid[0]


@pytest.fixture(scope="module")
def recorders() -> Iterator[dict[tuple[str, str], HumanoidRecorder]]:
    """One recorder per (effector, posture); built lazily, closed at the end."""
    made = {
        (effector, posture): HumanoidRecorder({"effector": effector, "posture": posture})
        for effector, posture, _ in VARIANTS
    }
    yield made
    for recorder in made.values():
        recorder.close()


@needs_assets
@pytest.mark.parametrize(("effector", "posture", "policy"), VARIANTS)
def test_record_validates_and_is_deterministic(
    recorders: dict[tuple[str, str], HumanoidRecorder], effector: str, posture: str, policy: str
) -> None:
    recorder = recorders[(effector, posture)]
    first = recorder.record(7, policy, STEPS)
    again = recorder.record(7, policy, STEPS)
    first.validate()
    assert first.body == "humanoid" and first.length == STEPS
    assert first.audio.shape == (STEPS, round(first.dt * SAMPLE_RATE))
    assert np.abs(first.action).max() <= 1.0
    channels = CHANNELS[effector]
    assert first.meta["touch_channels"] == channels
    assert first.touch.shape == (STEPS, sum(channels.values())) and (first.touch >= 0).all()
    assert len(set(first.touch_names)) == first.touch.shape[1]
    assert first.touch_names[0].startswith("body/") and "body/left_hip_pitch/3" in first.touch_names
    dense = "hand/left/ffdistal/2" if effector == "hand" else "pincer/left/left_pad/5"
    region = dense.rsplit("/", 1)[0]
    assert dense in first.touch_names and f"dynamics/{region}/slip" in first.touch_names
    assert all(not n.startswith("hand/") for n in first.touch_names[: channels["grid"]])
    assert first.touch_names[-1].startswith("dynamics/")
    assert {"vest_gyro_x", "vest_acc_z", "pelvis_height", "gravity_z"} <= set(first.proprio_names)
    assert first.privileged["touch_xyz"].shape == (STEPS, GRID_TAXELS[effector], 3)
    assert first.meta["touch_xyz_names"] == recorder.skin.names
    assert first.privileged["rgb_right"].shape == first.rgb.shape
    for array in (first.rgb, first.proprio, first.touch, first.action):
        assert np.isfinite(array.astype(float)).all()
    assert all(first.text) and first.meta["end_reason"] == "steps"
    assert first.meta["taxels"] == channels["grid"] + channels["dense"]
    assert not first.meta["fell"]
    assert content_hash(first) == content_hash(again)
    assert first.touch_names == again.touch_names
    assert set(model_inputs(first)) == set(MODEL_INPUT_KEYS)


@needs_assets
def test_seeds_differ_and_policies_are_checked(
    recorders: dict[tuple[str, str], HumanoidRecorder],
) -> None:
    seated = recorders[("hand", "seated")]
    a = seated.record(1, "seated_babble", STEPS)
    b = seated.record(2, "seated_babble", STEPS)
    assert content_hash(a) != content_hash(b)
    with pytest.raises(ValueError, match="seated"):
        recorders[("hand", "standing")].record(1, "seated_babble", STEPS)
    with pytest.raises(ValueError, match="policy"):
        seated.record(1, "dance", STEPS)


@needs_assets
@pytest.mark.parametrize("effector", ["hand", "pincer"])
def test_standing_rest_has_the_hands_at_the_sides(
    recorders: dict[tuple[str, str], HumanoidRecorder], effector: str
) -> None:
    """Geometry, not joint angles: fingertips beside the thighs, below the pelvis, no falls."""
    recorder = recorders[(effector, "standing")]
    for seed in (3, 4):
        episode = recorder.record(seed, "hold", 300)
        assert episode.length == 300 and episode.meta["end_reason"] == "steps"
        assert not episode.privileged["fall"].any()
        height = episode.proprio[:, episode.proprio_names.index("pelvis_height")]
        assert height.min() > 0.9
        assert float(str(episode.meta["touch_rate"])) < 0.01  # thighs included
        pelvis = episode.privileged["qpos"][:, :3]
        tip = episode.privileged["tip_position"] - pelvis[:, None, :]
        # within 25 cm of the pelvis in x on average (the arm noise swings the elbow by 0.1 rad)
        assert np.abs(tip[:, :, 0]).mean() < 0.25 and np.abs(tip[:, :, 0]).max() < 0.40
        assert tip[:, :, 2].max() < 0.0  # below the pelvis
        assert np.abs(tip[:, :, 1]).min() > 0.15  # off the midline, beside the legs


@needs_assets
def test_pincer_pads_are_made_of_cubes(recorders: dict[tuple[str, str], HumanoidRecorder]) -> None:
    model = recorders[("pincer", "seated")].model
    pad = model.body("lp_left_pad").id
    cubes = [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == pad and "pad_cube" in (model.geom(g).name or "")
    ]
    assert len(cubes) == 40
    depth = model.geom_size[cubes, 1]
    assert depth.max() > depth.min()  # crowned: the centre cubes stand proud


@needs_assets
def test_appearance_changes_per_episode(
    recorders: dict[tuple[str, str], HumanoidRecorder],
) -> None:
    seated = recorders[("hand", "seated")]
    a, b, again = (seated.record(seed, "seated_babble", 2) for seed in (1, 2, 1))
    assert a.rgb.shape[1:] == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert a.meta["appearance"] == again.meta["appearance"] != b.meta["appearance"]
    assert set(seated.appearance.tints) >= {"floor", "wall", "table_wood", "metal"}
    assert np.abs(a.rgb[0].astype(int) - b.rgb[0].astype(int)).mean() > 5
    # the object layout comes from its own random stream, untouched by the appearance
    np.testing.assert_array_equal(
        a.privileged["object_pose"][0], again.privileged["object_pose"][0]
    )
    model = seated.model
    floor = model.material("floor").id
    np.testing.assert_allclose(
        model.mat_rgba[floor, :3], seated.appearance.tints["floor"], atol=1e-6
    )
    assert model.stat.extent == SCENE_EXTENT and model.ngeom > 0 and model.geom("wall0").id >= 0
