"""Load-time touch perturbations: per-episode gain, dead channels, sensor noise, binarisation.

Episodes store clean simulated touch; the training loader perturbs it on the
fly, with a fresh draw per episode, so the stored data never changes and the
perturbation can be switched off or varied per run. The defaults follow the
sim-to-real tactile literature (see ``docs/touch.md``):

* per-channel gain ``N(1, gain_sd)`` with ``dropout`` of channels dead for the
  whole episode: Han, Joshi and Deng 2025 (arXiv:2505.02915) use
  ``N(1.0, 0.25)`` per taxel with 20% dropout;
* a per-channel offset and per-step Gaussian noise on force channels:
  Kasolowsky and Bäuml 2024 (arXiv:2409.12735) draw an offset and a noise range
  per taxel at the start of each episode;
* ``binarize``: touch / no touch at a force threshold, as in Yin et al. 2023
  (arXiv:2303.10880). Off by default; an ablation, since it discards force.

Only force channels (normal force in newtons) get offsets, noise, clipping
and binarisation. ``dynamics/*`` channels (onset, slip, vibration) get gain and
dropout only.
"""

from dataclasses import dataclass

import numpy as np

DYNAMICS_PREFIX = "dynamics/"


@dataclass(frozen=True)
class TouchNoise:
    """How to perturb one episode's touch.

    Attributes:
        gain_sd: Standard deviation of the per-channel gain around 1 (clipped at 0).
        dropout: Probability that a channel is dead (always 0) for the episode.
        offset_sd: Standard deviation of a per-channel constant offset (N).
        noise_sd: Standard deviation of per-step Gaussian noise on force channels (N).
        binarize: Force threshold (N) for touch / no touch, or ``None`` to keep forces.
    """

    gain_sd: float = 0.25
    dropout: float = 0.2
    offset_sd: float = 0.0
    noise_sd: float = 0.01
    binarize: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if min(self.gain_sd, self.offset_sd, self.noise_sd) < 0.0:
            raise ValueError("gain_sd, offset_sd and noise_sd must be >= 0")


def perturb_touch(
    touch: np.ndarray, names: list[str], rng: np.random.Generator, noise: TouchNoise
) -> np.ndarray:
    """A perturbed copy of one episode's ``[T, C]`` touch (float32); ``touch`` is unchanged.

    Raises:
        ValueError: if ``names`` does not name every column of ``touch``.
    """
    steps, channels = touch.shape
    if len(names) != channels:
        raise ValueError(f"{channels} touch channels but {len(names)} names")
    force = np.array([not name.startswith(DYNAMICS_PREFIX) for name in names])
    gain = np.maximum(rng.normal(1.0, noise.gain_sd, channels), 0.0)
    gain[rng.random(channels) < noise.dropout] = 0.0
    out = touch.astype(np.float32) * gain.astype(np.float32)
    offset = rng.normal(0.0, noise.offset_sd, channels) * force
    jitter = rng.normal(0.0, noise.noise_sd, (steps, channels)) * force
    out += (offset + jitter).astype(np.float32) * (gain > 0)
    out[:, force] = np.maximum(out[:, force], 0.0)
    if noise.binarize is not None:
        out[:, force] = (out[:, force] > noise.binarize).astype(np.float32)
    return np.asarray(out, dtype=np.float32)
