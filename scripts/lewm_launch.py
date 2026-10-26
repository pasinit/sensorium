"""Run LeWorldModel's ``train.py`` with a run seed (B-pusht-0, LeWM-tiny arm).

LeWM's ``cfg.seed`` (3072) seeds its clip split and its loader order, so B-pusht-0 keeps it (the
split is LeWM's). The run seed is torch's (and numpy's, Python's) global seed, set here before
LeWM's own ``run()`` builds the model, so the three seeds start from different weights. Copied
next to le-wm's ``train.py`` (commit ``8edfeb3``) on the VM and run from there:

    python lewm_launch.py --run-seed=0 <LeWM's Hydra overrides ...>

Everything after ``--run-seed`` goes to LeWM's Hydra command line unchanged.
"""

import random
import sys

import numpy as np
import torch


def split_seed(argv: list[str]) -> tuple[int, list[str]]:
    """``(run seed, the remaining arguments)`` of a command line with ``--run-seed=N``.

    Raises:
        ValueError: if ``--run-seed`` is missing.
    """
    seeds = [a for a in argv if a.startswith("--run-seed=")]
    if len(seeds) != 1:
        raise ValueError("give exactly one --run-seed=N")
    return int(seeds[0].split("=", 1)[1]), [a for a in argv if a not in seeds]


def main() -> None:
    """Seed everything, then hand over to LeWM's Hydra entry point."""
    seed, rest = split_seed(sys.argv[1:])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    sys.argv = [sys.argv[0], *rest]
    import train  # noqa: PLC0415 (le-wm's train.py, next to this file on the VM)

    train.run()


if __name__ == "__main__":
    main()
