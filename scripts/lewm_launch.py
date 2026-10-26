"""Run LeWorldModel's ``train.py`` with a run seed (B-pusht-0, LeWM-tiny arm).

LeWM's ``cfg.seed`` (3072) seeds its clip split and its loader order, so B-pusht-0 keeps it (the
split is LeWM's). The run seed is torch's (and numpy's, Python's) global seed, set here before
LeWM's own ``run()`` builds the model, so the three seeds start from different weights. Copied
next to le-wm's ``train.py`` (commit ``8edfeb3``) on the VM and run from there:

    python lewm_launch.py --run-seed=0 <LeWM's Hydra overrides ...>

Everything after ``--run-seed`` goes to LeWM's Hydra command line unchanged.

LeWM logs its losses only to Weights & Biases; with that off its Lightning ``Trainer`` gets no
logger. The launcher gives it a ``CSVLogger`` (``lewm_logs/<run>/``, every 50 steps), so every
seed's loss curves are kept, as for our arms.
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


def run_name(argv: list[str]) -> str:
    """The run's ``output_model_name`` from its Hydra overrides (``lewm`` if absent)."""
    names = [a.split("=", 1)[1] for a in argv if a.startswith("output_model_name=")]
    return names[-1] if names else "lewm"


def with_csv_logger(name: str) -> None:
    """Make Lightning's ``Trainer`` log to CSV when LeWM gives it no logger."""
    import lightning as pl  # noqa: PLC0415
    from lightning.pytorch.loggers import CSVLogger  # noqa: PLC0415

    init = pl.Trainer.__init__

    def patched(self: object, *args: object, **kwargs: object) -> None:
        if kwargs.get("logger") is None:
            kwargs["logger"] = CSVLogger("lewm_logs", name=name)
            kwargs.setdefault("log_every_n_steps", 50)
        init(self, *args, **kwargs)  # type: ignore[arg-type]

    pl.Trainer.__init__ = patched  # type: ignore[method-assign]


def main() -> None:
    """Seed everything, then hand over to LeWM's Hydra entry point."""
    seed, rest = split_seed(sys.argv[1:])
    with_csv_logger(run_name(rest))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    sys.argv = [sys.argv[0], *rest]
    import train  # noqa: PLC0415 (le-wm's train.py, next to this file on the VM)

    train.run()


if __name__ == "__main__":
    main()
