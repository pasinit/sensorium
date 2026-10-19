"""Load-time touch perturbations: deterministic per seed, force channels only where stated."""

import numpy as np
import pytest

from sensorium.touch_noise import TouchNoise, perturb_touch

NAMES = [f"hand/left/ffdistal/{k}" for k in range(200)] + [
    f"dynamics/hand/left/ffdistal/{f}" for f in ("onset", "slip", "vibration")
]


def touch(steps: int = 50) -> np.ndarray:
    values = np.zeros((steps, len(NAMES)), dtype=np.float32)
    values[:, :200] = np.linspace(0.0, 2.0, 200)
    values[:, 200:] = [1.0, 0.3, 0.5]
    return values


def test_same_seed_same_result_and_input_untouched() -> None:
    clean = touch()
    before = clean.copy()
    a = perturb_touch(clean, NAMES, np.random.default_rng(3), TouchNoise())
    b = perturb_touch(clean, NAMES, np.random.default_rng(3), TouchNoise())
    c = perturb_touch(clean, NAMES, np.random.default_rng(4), TouchNoise())
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    np.testing.assert_array_equal(clean, before)
    assert a.dtype == np.float32 and a.shape == clean.shape


def test_no_perturbation_is_identity() -> None:
    off = TouchNoise(gain_sd=0.0, dropout=0.0, offset_sd=0.0, noise_sd=0.0)
    np.testing.assert_array_equal(
        perturb_touch(touch(), NAMES, np.random.default_rng(0), off), touch()
    )


def test_dropout_kills_whole_channels_and_forces_stay_non_negative() -> None:
    out = perturb_touch(touch(), NAMES, np.random.default_rng(0), TouchNoise(dropout=0.5))
    dead = (out == 0).all(axis=0)
    assert 0.3 < dead[:200].mean() < 0.7
    assert (out[:, :200] >= 0).all()


def test_dynamics_channels_get_gain_but_no_noise() -> None:
    noisy = TouchNoise(gain_sd=0.0, dropout=0.0, noise_sd=0.5, offset_sd=0.5)
    out = perturb_touch(touch(), NAMES, np.random.default_rng(0), noisy)
    np.testing.assert_array_equal(out[:, 200:], touch()[:, 200:])
    assert out[:, :200].std(axis=0).mean() > 0.1


def test_binarize_thresholds_forces_only() -> None:
    out = perturb_touch(
        touch(), NAMES, np.random.default_rng(0), TouchNoise(0.0, 0.0, 0.0, 0.0, binarize=1.0)
    )
    assert set(np.unique(out[:, :200])) == {0.0, 1.0}
    assert out[0, :200].sum() == (np.linspace(0.0, 2.0, 200) > 1.0).sum()
    np.testing.assert_array_equal(out[:, 200:], touch()[:, 200:])


@pytest.mark.parametrize("kwargs", [{"dropout": 1.0}, {"dropout": -0.1}, {"noise_sd": -1.0}])
def test_invalid_settings_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        TouchNoise(**kwargs)


def test_names_must_match_channels() -> None:
    with pytest.raises(ValueError, match="names"):
        perturb_touch(touch(), NAMES[:-1], np.random.default_rng(0), TouchNoise())
