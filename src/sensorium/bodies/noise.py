"""Smooth random actions, for "noisy" and "explore" policies."""

import numpy as np


class SmoothNoise:
    """Ornstein–Uhlenbeck noise: random but temporally correlated actions.

    ``sample()`` returns a ``[dim]`` vector that drifts toward ``mean`` with
    per-step persistence ``theta`` and stationary std of about ``scale``.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        dim: int,
        scale: float,
        theta: float = 0.15,
        mean: np.ndarray | float = 0.0,
    ) -> None:
        self.rng, self.scale, self.theta = rng, scale, theta
        self.mean = np.broadcast_to(np.asarray(mean, dtype=float), (dim,))
        self.state = self.mean.copy()

    def sample(self) -> np.ndarray:
        """Advance one step and return the current noise vector."""
        sigma = self.scale * np.sqrt(2 * self.theta)
        self.state = (
            self.state
            + self.theta * (self.mean - self.state)
            + sigma * self.rng.standard_normal(self.state.shape)
        )
        return self.state.copy()
