"""Synthetic audio from simulator state, and log-mel features.

None of the simulators produce sound, so each body's sound is synthesised from
its privileged state, one chunk of ``dt * SAMPLE_RATE`` samples per step:

* car:   engine harmonics (pitch follows speed) + wind and tyre noise.
* drone: rotor tones at each motor's blade-pass frequency + broadband noise.
* piano: a decaying harmonic tone for every sounding key.
* humanoid: impact "ticks" and sliding noise from skin contacts (``ContactSound``).

Every synthesiser takes a ``numpy.random.Generator``, so audio is reproducible
from the episode seed. A random per-episode "timbre" (gains, noise level)
keeps sound from being an exact, trivially invertible copy of state.
"""

import numpy as np

from sensorium.episode import SAMPLE_RATE

N_MELS = 32


class PhaseOscillator:
    """Sinusoid bank whose phases carry over between chunks (no clicks).

    Each call renders one chunk; frequencies and amplitudes are linearly
    interpolated across the chunk from the previous call's values.
    """

    def __init__(self, n: int) -> None:
        self.phase = np.zeros(n)
        self.prev_freq = np.zeros(n)
        self.prev_amp = np.zeros(n)

    def render(self, freq: np.ndarray, amp: np.ndarray, n_samples: int) -> np.ndarray:
        """Render ``n_samples`` of the summed sinusoids; returns float64 ``[n_samples]``."""
        ramp = np.linspace(0.0, 1.0, n_samples, endpoint=False)[:, None]
        f = self.prev_freq + (freq - self.prev_freq) * ramp
        a = self.prev_amp + (amp - self.prev_amp) * ramp
        phase = self.phase + np.cumsum(2 * np.pi * f / SAMPLE_RATE, axis=0)
        self.phase = phase[-1] % (2 * np.pi)
        self.prev_freq, self.prev_amp = freq.astype(float), amp.astype(float)
        chunk: np.ndarray = (a * np.sin(phase)).sum(axis=1)
        return chunk


def shaped_noise(rng: np.random.Generator, n_samples: int, smoothing: int) -> np.ndarray:
    """White noise low-passed by a moving average of ``smoothing`` samples."""
    noise = rng.standard_normal(n_samples + smoothing)
    kernel = np.ones(smoothing) / np.sqrt(smoothing)
    return np.convolve(noise, kernel, mode="valid")[:n_samples]


class CarSound:
    """Engine + wind + tyres. Call ``step(speed_mps, throttle, slip)`` once per step."""

    def __init__(self, rng: np.random.Generator, n_samples: int) -> None:
        self.rng, self.n = rng, n_samples
        self.engine = PhaseOscillator(4)
        self.idle_hz = rng.uniform(25, 40)  # firing frequency at idle
        self.hz_per_mps = rng.uniform(2.0, 3.5)
        self.harmonic_gain = rng.uniform(0.3, 1.0, size=4) / np.arange(1, 5)

    def step(self, speed_mps: float, throttle: float, slip: float) -> np.ndarray:
        """One chunk of car sound (float64, roughly in [-1, 1])."""
        f0 = self.idle_hz + self.hz_per_mps * speed_mps
        load = 0.3 + 0.7 * max(throttle, 0.0)
        engine = self.engine.render(f0 * np.arange(1, 5), 0.25 * load * self.harmonic_gain, self.n)
        wind = 0.002 * speed_mps * np.sqrt(speed_mps) * shaped_noise(self.rng, self.n, 12)
        tyres = (0.01 * speed_mps + 0.3 * slip) * shaped_noise(self.rng, self.n, 4)
        chunk: np.ndarray = engine + wind + tyres
        return chunk


class DroneSound:
    """Four rotors. Call ``step(rpm[4])`` once per step."""

    BLADES = 2

    def __init__(self, rng: np.random.Generator, n_samples: int) -> None:
        self.rng, self.n = rng, n_samples
        self.rotors = PhaseOscillator(8)  # fundamental + 2nd harmonic for 4 motors
        self.imbalance = rng.uniform(0.9, 1.1, size=4)
        self.hiss = rng.uniform(0.02, 0.06)

    def step(self, rpm: np.ndarray) -> np.ndarray:
        """One chunk of rotor sound (float64)."""
        blade_pass = rpm / 60.0 * self.BLADES * self.imbalance
        freq = np.concatenate([blade_pass, 2 * blade_pass])
        level = np.clip(rpm / 20_000.0, 0.0, 1.5)
        amp = np.concatenate([0.12 * level, 0.04 * level])
        hiss = self.hiss * level.mean() * shaped_noise(self.rng, self.n, 2)
        return self.rotors.render(freq, amp, self.n) + hiss


class PianoSound:
    """88 keys. Call ``step(key_down[88])`` once per step; onsets start a note."""

    def __init__(self, rng: np.random.Generator, n_samples: int) -> None:
        self.rng, self.n = rng, n_samples
        self.keys = PhaseOscillator(88 * 3)  # 3 partials per key
        midi = np.arange(21, 109)  # A0..C8
        f0 = 440.0 * 2.0 ** ((midi - 69) / 12)
        self.freq = np.concatenate([f0, 2 * f0, 3 * f0])
        self.partial_gain = np.repeat(rng.uniform([0.6, 0.2, 0.08], [1.0, 0.4, 0.2]), 88)
        self.envelope = np.zeros(88)
        self.was_down = np.zeros(88, dtype=bool)
        self.decay_held, self.decay_released = 0.995, 0.6  # per-step multipliers

    def step(self, key_down: np.ndarray) -> np.ndarray:
        """One chunk of piano sound (float64)."""
        down = key_down.astype(bool)
        onset = down & ~self.was_down
        self.envelope[onset] = 1.0
        self.envelope *= np.where(down, self.decay_held, self.decay_released)
        self.was_down = down
        amp = 0.15 * np.tile(self.envelope, 3) * self.partial_gain
        audible = self.freq < SAMPLE_RATE / 2
        return self.keys.render(np.where(audible, self.freq, 0.0), amp * audible, self.n)


class ContactSound:
    """Contact sound for the humanoid: impacts ring, slides rasp.

    Sound sources are *voices*: five body parts (feet, legs, torso, arms,
    hands) and six surfaces (floor, table and the four object materials). A
    contact between a body part and a surface excites both voices, so the same
    fingertip sounds different on wood and on metal. Per step the caller gives
    two vectors of length ``len(VOICES)``:

    * ``impact[v]``: strength of *new* contacts on voice ``v`` (normal force, N);
    * ``slide[v]``: sliding intensity on voice ``v`` (sum over sliding contacts of
      tangential speed in m/s times the square root of the normal force).

    An impact renders three damped sinusoids (the voice's modes) starting at the
    chunk's first sample; a slide renders band-limited noise whose brightness is
    the voice's. A random per-episode timbre (mode frequencies, decay, gains,
    noise brightness, room tone) comes from ``rng``.
    """

    # voice: (lowest mode Hz, highest mode Hz, decay seconds)
    VOICES = {
        "feet": (80.0, 300.0, 0.020),
        "legs": (120.0, 400.0, 0.015),
        "torso": (100.0, 350.0, 0.020),
        "arms": (200.0, 700.0, 0.012),
        "hands": (400.0, 1500.0, 0.008),
        "floor": (150.0, 600.0, 0.010),
        "table": (250.0, 900.0, 0.020),
        "wood": (500.0, 1800.0, 0.012),
        "plastic": (900.0, 3000.0, 0.006),
        "metal": (1500.0, 5000.0, 0.040),
        "rubber": (150.0, 500.0, 0.004),
    }
    N_MODES = 3

    def __init__(self, rng: np.random.Generator, n_samples: int) -> None:
        self.rng, self.n = rng, n_samples
        count = len(self.VOICES)
        lo, hi, decay = (np.array(column) for column in zip(*self.VOICES.values(), strict=True))
        # Modes spread between the voice's bounds, jittered per episode.
        spread = np.sort(rng.uniform(0.0, 1.0, size=(count, self.N_MODES)), axis=1)
        self.freq = np.clip(lo[:, None] * (hi / lo)[:, None] ** spread, 50.0, SAMPLE_RATE / 2.2)
        self.decay = decay * rng.uniform(0.7, 1.4, size=count)
        self.gain = rng.uniform(0.5, 1.0, size=(count, self.N_MODES)) / np.arange(1, 4)
        self.phase = rng.uniform(0, 2 * np.pi, size=(count, self.N_MODES))
        self.smoothing = np.clip((SAMPLE_RATE / (2.0 * hi)).astype(int) + 1, 1, 24)
        self.room_tone = rng.uniform(0.0002, 0.001)
        self.t = np.arange(n_samples) / SAMPLE_RATE

    def step(self, impact: np.ndarray, slide: np.ndarray) -> np.ndarray:
        """One chunk (float64) for the contacts of the last control interval."""
        chunk = self.room_tone * shaped_noise(self.rng, self.n, 8)
        for voice in np.flatnonzero(impact > 0.0):
            amp = 0.3 * (1.0 - np.exp(-impact[voice] / 20.0))
            envelope = np.exp(-self.t / self.decay[voice])
            modes = np.sin(
                2 * np.pi * self.freq[voice][:, None] * self.t + self.phase[voice][:, None]
            )
            chunk = chunk + amp * envelope * (self.gain[voice][:, None] * modes).sum(axis=0)
        for voice in np.flatnonzero(slide > 0.0):
            amp = min(0.015 * slide[voice], 0.05)
            chunk = chunk + amp * shaped_noise(self.rng, self.n, int(self.smoothing[voice]))
        return chunk


def to_int16(chunks: list[np.ndarray]) -> np.ndarray:
    """Stack float chunks into ``[T, n]`` int16 with a soft limiter (tanh)."""
    return (np.tanh(np.stack(chunks)) * 32767 * 0.9).astype(np.int16)


def logmel(audio: np.ndarray, n_mels: int = N_MELS) -> np.ndarray:
    """Log-mel energy of each chunk: ``[T, n]`` int16 -> ``[T, n_mels]`` float16.

    One FFT over each whole (Hann-windowed) chunk, so a step has exactly one
    feature vector, aligned with its RGB frame.
    """
    x = audio.astype(np.float32) / 32768.0
    n = x.shape[1]
    spectrum = np.abs(np.fft.rfft(x * np.hanning(n), axis=1)) ** 2
    freqs = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
    mel = 2595 * np.log10(1 + freqs / 700)
    edges = np.linspace(mel[1], mel[-1], n_mels + 2)
    weights = np.zeros((n_mels, freqs.size), dtype=np.float32)
    for i in range(n_mels):  # triangular filters
        lo, mid, hi = edges[i : i + 3]
        weights[i] = np.clip(np.minimum((mel - lo) / (mid - lo), (hi - mel) / (hi - mid)), 0, None)
    return np.log(spectrum @ weights.T + 1e-8).astype(np.float16)
