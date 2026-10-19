"""Load-time image views: shapes, one crop per clip, centre crop, jitter, validation."""

import numpy as np
import pytest

from sensorium.vision import View, eval_view, train_view


def clip(steps: int = 4, side: int = 256) -> np.ndarray:
    """Frames whose pixel values encode their (row, column), different per frame."""
    rows, cols = np.mgrid[0:side, 0:side]
    frame = np.stack([rows, cols, (rows + cols) // 2], axis=-1) % 256
    return np.stack([(frame + t) % 256 for t in range(steps)]).astype(np.uint8)


def test_train_view_shape_determinism_and_input_untouched() -> None:
    frames = clip()
    before = frames.copy()
    a = train_view(frames, np.random.default_rng(0), View(size=128))
    b = train_view(frames, np.random.default_rng(0), View(size=128))
    assert a.shape == (4, 128, 128, 3) and a.dtype == np.uint8
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(frames, before)


def test_one_crop_for_the_whole_clip() -> None:
    frames = clip(side=64)
    out = train_view(frames, np.random.default_rng(3), View(size=48, crop=0.75))
    # with crop 48 of 64 and no resize, every frame is the same window shifted by t in value
    top, left = int(out[0, 0, 0, 0]), int(out[0, 0, 0, 1])
    for t in range(len(frames)):
        np.testing.assert_array_equal(out[t], frames[t, top : top + 48, left : left + 48])


def test_crops_vary_across_draws() -> None:
    frames = clip(steps=1, side=64)
    corners = {
        tuple(train_view(frames, np.random.default_rng(seed), View(size=48, crop=0.75))[0, 0, 0])
        for seed in range(20)
    }
    assert len(corners) > 5


def test_eval_view_is_the_centre_crop() -> None:
    frames = clip(side=64)
    np.testing.assert_array_equal(
        eval_view(frames, View(size=48, crop=0.75)), frames[:, 8:56, 8:56]
    )
    np.testing.assert_array_equal(eval_view(frames, View(size=64, crop=1.0)), frames)


def test_jitter_is_off_by_default_and_changes_pixels_when_on() -> None:
    frames = clip(side=64)
    plain = View(size=64, crop=1.0)
    np.testing.assert_array_equal(train_view(frames, np.random.default_rng(0), plain), frames)
    jittered = train_view(
        frames, np.random.default_rng(0), View(size=64, crop=1.0, brightness=0.4, saturation=0.4)
    )
    assert np.abs(jittered.astype(int) - frames.astype(int)).mean() > 1


@pytest.mark.parametrize(
    "kwargs", [{"size": 0}, {"crop": 0.0}, {"crop": 1.5}, {"brightness": -0.1}]
)
def test_invalid_views_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        View(**kwargs)  # type: ignore[arg-type]


def test_frames_must_be_uint8_rgb() -> None:
    with pytest.raises(ValueError, match="uint8"):
        eval_view(np.zeros((2, 8, 8, 3), dtype=np.float32), View(size=8))
