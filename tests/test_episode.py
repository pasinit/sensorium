"""Episode format: round-trip, validation, and the no-leak guarantee."""

from pathlib import Path

import numpy as np
import pytest

from sensorium.episode import IMAGE_SIZE, MODEL_INPUT_KEYS, SAMPLE_RATE, Episode, model_inputs


def frames(steps: int) -> np.ndarray:
    """Smooth moving gradients: like camera images, unlike noise, they compress well."""
    rows, cols = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    return np.stack(
        [
            np.stack([rows + 3 * t, cols, (rows + cols) // 2 + t], axis=-1) % 256
            for t in range(steps)
        ]
    ).astype(np.uint8)


def make_episode(steps: int = 5, dt: float = 0.1) -> Episode:
    rng = np.random.default_rng(0)
    return Episode(
        body="test",
        dt=dt,
        rgb=frames(steps),
        audio=rng.integers(-100, 100, (steps, round(dt * SAMPLE_RATE)), dtype=np.int16),
        audio_logmel=np.zeros((steps, 32), dtype=np.float16),
        proprio=rng.standard_normal((steps, 2)).astype(np.float32),
        touch=np.zeros((steps, 0), dtype=np.float32),
        action=rng.uniform(-1, 1, (steps, 3)).astype(np.float32),
        proprio_names=["a", "b"],
        touch_names=[],
        action_names=["x", "y", "z"],
        privileged={
            "position": np.zeros((steps, 3), dtype=np.float32),
            "eye": frames(steps)[:, ::-1],
        },
        text=["hello"] * steps,
        meta={"seed": 1},
    )


def test_round_trip(tmp_path: Path) -> None:
    episode = make_episode()
    episode.save(tmp_path / "e.npz")
    loaded = Episode.load(tmp_path / "e.npz")
    for name in ("audio_logmel", "proprio", "touch", "action", "audio"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(episode, name))
    np.testing.assert_array_equal(loaded.privileged["position"], episode.privileged["position"])
    # frames are stored as lossy video: same shape and type, close in value
    for stored, original in (
        (loaded.rgb, episode.rgb),
        (loaded.privileged["eye"], episode.privileged["eye"]),
    ):
        assert stored.shape == original.shape and stored.dtype == np.uint8
        assert np.abs(stored.astype(int) - original.astype(int)).mean() < 3
    assert loaded.text == episode.text
    assert loaded.meta == episode.meta
    assert loaded.action_names == episode.action_names


def test_model_inputs_exclude_privileged_and_text() -> None:
    inputs = model_inputs(make_episode())
    assert set(inputs) == set(MODEL_INPUT_KEYS)
    assert not any(key.startswith("privileged") or key == "text" for key in inputs)


def test_model_inputs_senses_subset() -> None:
    inputs = model_inputs(make_episode(), senses=("rgb", "touch"))
    assert set(inputs) == {"rgb", "touch", "action"}  # action is always kept
    assert set(model_inputs(make_episode(), senses=())) == {"action"}


def test_model_inputs_rejects_unknown_sense() -> None:
    with pytest.raises(ValueError, match="privileged"):
        model_inputs(make_episode(), senses=("rgb", "privileged"))


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("action", np.full((5, 3), 2.0, dtype=np.float32), "outside"),
        ("rgb", np.zeros((5, 32, 32, 3), dtype=np.uint8), "rgb"),
        ("proprio", np.full((5, 2), np.nan, dtype=np.float32), "NaN"),
        ("text", ["only one"], "text"),
    ],
)
def test_validate_rejects(field: str, bad_value: object, message: str) -> None:
    episode = make_episode()
    setattr(episode, field, bad_value)
    with pytest.raises(ValueError, match=message):
        episode.validate()


def test_invalid_episode_is_not_written(tmp_path: Path) -> None:
    episode = make_episode()
    episode.action[0, 0] = 5.0
    with pytest.raises(ValueError):
        episode.save(tmp_path / "e.npz")
    assert not (tmp_path / "e.npz").exists()


def test_frames_are_stored_as_video(tmp_path: Path) -> None:
    episode = make_episode(steps=20)
    episode.save(tmp_path / "e.npz")
    with np.load(tmp_path / "e.npz") as data:
        assert {"rgb.mp4", "privileged/eye.mp4"} <= set(data.files) and "rgb" not in data.files
        assert data["rgb.mp4"].nbytes < episode.rgb.nbytes / 20
    episode.save(tmp_path / "again.npz")
    assert (tmp_path / "e.npz").read_bytes() == (tmp_path / "again.npz").read_bytes()


def test_load_can_skip_privileged(tmp_path: Path) -> None:
    episode = make_episode()
    episode.save(tmp_path / "e.npz")
    loaded = Episode.load(tmp_path / "e.npz", privileged=False)
    assert loaded.privileged == {}
    np.testing.assert_array_equal(loaded.action, episode.action)
