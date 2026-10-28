"""Run LeWorldModel's ``train.py`` with a run seed (B-pusht-0, LeWM-tiny arm).

LeWM's ``cfg.seed`` (3072) seeds its clip split and its loader order, so B-pusht-0 keeps it (the
split is LeWM's). The run seed is torch's (and numpy's, Python's) global seed, set here before
LeWM's own ``run()`` builds the model, so the three seeds start from different weights. Copied
next to le-wm's ``train.py`` (commit ``8edfeb3``) on the VM and run from there:

    python lewm_launch.py --run-seed=0 <LeWM's Hydra overrides ...>

Everything after ``--run-seed`` goes to LeWM's Hydra command line unchanged, and ``train.py``
runs as ``__main__``, as ``python train.py`` would (Hydra resolves its config path next to it).

LeWM logs its losses only to Weights & Biases; with that off its Lightning ``Trainer`` gets no
logger. The launcher gives it a ``CSVLogger`` (``lewm_logs/<run>/``, every 50 steps), so every
seed's loss curves are kept, as for our arms.

LeWM saves its weights at the end of each epoch. B-pusht-0 stops at a step budget, inside an
epoch, so the launcher also saves them when training ends (``weights_final.pt``, beside LeWM's
per-epoch files), and logs a checksum of the initial weights when training starts (the seeds must
differ).
"""

import random
import runpy
import sys
from pathlib import Path

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


def checksum(model: torch.nn.Module) -> float:
    """A checksum of a model's parameters (their sum of absolute values)."""
    return float(sum(p.detach().abs().sum() for p in model.parameters()))


def with_final_save() -> None:
    """Make LeWM's checkpoint callback also log the initial weights' checksum at the start and
    save ``weights_final.pt`` at the end of training."""
    import utils  # noqa: PLC0415 (le-wm's utils.py, next to this file on the VM)

    def on_train_start(self: object, trainer: object, pl_module: object) -> None:
        print(f"initial weights checksum {checksum(pl_module.model):.6f}", flush=True)  # type: ignore[attr-defined]  # noqa: T201

    def on_train_end(self: object, trainer: object, pl_module: object) -> None:
        self._save(pl_module.model, "final")  # type: ignore[attr-defined]

    utils.SaveCkptCallback.on_train_start = on_train_start
    utils.SaveCkptCallback.on_train_end = on_train_end


def main() -> None:
    """Seed everything, then hand over to LeWM's Hydra entry point."""
    seed, rest = split_seed(sys.argv[1:])
    with_csv_logger(run_name(rest))
    with_final_save()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    script = Path(__file__).resolve().parent / "train.py"  # le-wm's, next to this file on the VM
    sys.argv = [str(script), *rest]
    # As ``python train.py``: imported as a module, train.py's @hydra.main would look for its
    # config_path ("./config/train") as a package-relative module and fail; run as __main__, Hydra
    # resolves it next to train.py.
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
