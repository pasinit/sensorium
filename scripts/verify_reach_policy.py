"""Check the numpy reach policy against torch on random observations.

Usage: uv run --extra model python scripts/verify_reach_policy.py [humanoid-bench dir]

Loads the released ``torch_model.pt`` with torch, runs it and ``ReachWeights.action`` on
the same observations and reports the largest difference (torch computes in float32, ours in
float64, so expect ~1e-6). Lives outside the package: generation never imports torch.
"""

import sys
from pathlib import Path

import numpy as np
import torch

from sensorium.bodies.h1_reach import ReachWeights
from sensorium.bodies.h1_scene import external_dir


def main() -> None:
    """Compare on 2,000 observations and fail if the outputs differ by more than 1e-4."""
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else external_dir() / "humanoid-bench"
    weights = ReachWeights.load(root / "reach_two_hands.npz")
    net = torch.nn.Sequential(
        torch.nn.Linear(61, 256), torch.nn.Tanh(), torch.nn.Linear(256, 256),
        torch.nn.Tanh(), torch.nn.Linear(256, 19),
    )  # fmt: skip
    state = torch.load(root / "data" / "reach_two_hands" / "torch_model.pt", weights_only=True)
    net.load_state_dict(
        {
            f"{2 * i}.{kind}": state[f"dense{i + 1}.{kind}"]
            for i in range(3)
            for kind in ("weight", "bias")
        }
    )
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(2000):
        observation = weights.mean + np.sqrt(weights.var) * rng.normal(0, 2.0, 61)
        normalised = (observation - weights.mean) / np.sqrt(weights.var + 1e-8)
        with torch.no_grad():
            expected = np.clip(net(torch.from_numpy(normalised).float()).numpy(), -1, 1)
        worst = max(worst, float(np.abs(expected - weights.action(observation)).max()))
    sys.stdout.write(f"largest difference over 2000 observations: {worst:.2e}\n")
    if worst > 1e-4:
        raise SystemExit("numpy and torch disagree")


if __name__ == "__main__":
    main()
