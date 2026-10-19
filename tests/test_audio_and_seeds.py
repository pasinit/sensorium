"""Synthetic audio, descriptions and seeds: shapes, determinism, sanity."""

import numpy as np

from sensorium.audio import N_MELS, CarSound, DroneSound, PianoSound, logmel, to_int16
from sensorium.describe import describe_piano, key_name
from sensorium.generate import episode_seed

N = 1600  # samples per 0.1 s step


def render(sound_cls: type, inputs: list[object], seed: int = 0) -> np.ndarray:
    sound = sound_cls(np.random.default_rng(seed), N)
    return to_int16([sound.step(*x) if isinstance(x, tuple) else sound.step(x) for x in inputs])


def test_same_seed_same_audio() -> None:
    inputs: list[object] = [(10.0, 0.5, 0.0)] * 5
    np.testing.assert_array_equal(render(CarSound, inputs, 3), render(CarSound, inputs, 3))
    assert not np.array_equal(render(CarSound, inputs, 3), render(CarSound, inputs, 4))


def test_car_louder_when_faster() -> None:
    slow = render(CarSound, [(1.0, 0.1, 0.0)] * 5).astype(float)
    fast = render(CarSound, [(20.0, 0.9, 0.0)] * 5).astype(float)
    assert np.sqrt((fast**2).mean()) > np.sqrt((slow**2).mean())


def test_drone_silent_when_motors_off() -> None:
    audio = render(DroneSound, [np.zeros(4)] * 3)
    assert np.abs(audio).max() == 0


def test_piano_sounds_only_after_key_press() -> None:
    silent, pressed = np.zeros(88, bool), np.zeros(88, bool)
    pressed[39] = True  # C4
    audio = render(PianoSound, [silent, silent, pressed, pressed])
    assert np.abs(audio[:2]).max() == 0
    assert np.abs(audio[2:]).max() > 1000


def test_logmel_shape_and_dtype() -> None:
    features = logmel(render(DroneSound, [np.full(4, 15_000.0)] * 4))
    assert features.shape == (4, N_MELS) and features.dtype == np.float16
    assert np.isfinite(features).all()


def test_key_names() -> None:
    assert key_name(0) == "A0" and key_name(39) == "C4" and key_name(87) == "C8"
    assert describe_piano([("right", "index", 39)]) == "right index presses C4"
    assert describe_piano([]) == "no keys pressed"


def test_episode_seed_is_stable_and_distinct() -> None:
    assert episode_seed(0, "car", "expert", 0) == episode_seed(0, "car", "expert", 0)
    seeds = {
        episode_seed(0, b, p, i) for b in ("car", "drone") for p in ("a", "b") for i in range(50)
    }
    assert len(seeds) == 200
