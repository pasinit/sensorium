"""Few-shot fine-tune yardstick with convergence curves (B-conseq-0, review item 2).

The k = 16 labelled-episode fine-tune of B-ctrl-0 plateaued at 300 steps and may have been
under-trained. Here every run is 600 steps with checkpoints at 100, 200, 400 and 600, each probed
with the same held-out battery (V+P encoder), for three inits and two recipes:

* init: ``random`` (from scratch), ``sub-rand`` and ``sub-conseq`` (the 1,000-step SSL encoders of
  ``scripts/objective_test.py``, same seed);
* recipe ``std``: lr 3e-4 for everything (the B-ctrl-0 recipe); ``lpft``: linear-probe-then-
  fine-tune, the encoder frozen for 100 steps while the head fits, then at 0.1 of the head's lr.

    uv run python scripts/finetune_test.py train --init sub-conseq --recipe lpft --seed 0
    uv run python scripts/finetune_test.py probe
    uv run python scripts/finetune_test.py report
"""

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np

from sensorium.model.controls import SupervisedConfig, train_supervised
from sensorium.model.pretrain import build_space, load_config, load_layer
from sensorium.model.probes import collect, predict_probes

DATA = Path("data/probe_test/humanoid")
SPLIT = Path("runs/decisive/split.json")
ROOT = Path("runs/finetune")
SSL = Path("runs/objective")
STEPS = 600
CKPTS = (100, 200, 400, 600)
K = 16
STRIDE = 8
NO_TOUCH = ("vision", "proprio_body", "vestibular", "proprio_effector")
INITS = ("random", "sub-rand", "sub-conseq")
RECIPES: dict[str, dict[str, float]] = {
    "std": {},
    "lpft": {"freeze_steps": 100, "encoder_lr_scale": 0.1},
}
TERMS = (
    "contact_left_auroc",
    "contact_right_auroc",
    "contact_object_auroc",
    "log_force_r2",
    "tip_r2",
)


def files(key: str) -> list[Path]:
    """Episode files of a split from ``decisive_test.py split``."""
    return [DATA / n for n in json.loads(SPLIT.read_text())[key]]


def train_run(init: str, recipe: str, seed: int, workers: int) -> None:
    """Fine-tune on ``K`` labelled pretraining episodes (a seeded subset) from ``init``."""
    base = load_config(Path("configs/model/cpu_tiny.toml"))
    episodes = files("pretrain")
    space = build_space(episodes, base.data.stats_episodes)
    picked = np.random.default_rng(seed).choice(len(episodes), K, replace=False)
    ckpt = None if init == "random" else SSL / f"{init}-s{seed}" / "ckpt-1000.pt"
    cfg = SupervisedConfig(
        model=replace(base.model, senses=NO_TOUCH),
        steps=STEPS,
        seed=seed,
        workers=workers,
        save_steps=CKPTS,
        **RECIPES[recipe],  # type: ignore[arg-type]
    )
    train_supervised(
        cfg,
        [episodes[i] for i in sorted(picked)],
        space,
        ROOT / f"ft-{init}-{recipe}-s{seed}",
        logging.info,
        init=ckpt,
    )


def probe_all() -> None:
    """Probe every checkpoint that exists and has no JSON yet."""
    out = ROOT / "probes"
    out.mkdir(parents=True, exist_ok=True)
    for run in sorted(ROOT.glob("ft-*-s?")):
        for step in CKPTS:
            ckpt = run / ("ckpt.pt" if step == STEPS else f"ckpt-{step}.pt")
            target = out / f"{run.name}-n{step}.json"
            if ckpt.exists() and not target.exists():
                scores = predict_probes(
                    collect(load_layer(ckpt), files("heldout"), STRIDE, (), raw=False), "state"
                ).scores()
                target.write_text(json.dumps({"heldout": scores, "vp": scores}))
                logging.info("probed %s", target.name)


def mean_sd(values: list[float]) -> str:
    """``mean +- sd`` over seeds."""
    arr = np.array(values)
    return f"{arr.mean():.2f} +- {arr.std():.2f}" if len(arr) > 1 else f"{arr[0]:.2f}"


def report() -> None:
    """Held-out probe mean against fine-tune steps, per init and recipe, with floor and ceiling."""
    res = {p.stem: json.loads(p.read_text()) for p in (ROOT / "probes").glob("*.json")}
    ctrl = Path("runs/controls/probes_json")
    print(f"Held-out probe mean of the V+P state; {K} labelled episodes; mean +- sd (n seeds)\n")  # noqa: T201
    print("| init | recipe | " + " | ".join(f"{s} steps" for s in CKPTS) + " |")  # noqa: T201
    print("|---|---|" + "---|" * len(CKPTS))  # noqa: T201
    for init in INITS:
        for recipe in RECIPES:
            cells = []
            for step in CKPTS:
                runs = [
                    v["heldout"]["mean"]
                    for k, v in res.items()
                    if k.startswith(f"ft-{init}-{recipe}-s") and k.endswith(f"-n{step}")
                ]
                cells.append(f"{mean_sd(runs)} ({len(runs)})" if runs else "-")
            print(f"| {init} | {recipe} | " + " | ".join(cells) + " |")  # noqa: T201
    for name, glob in (
        ("floor V+P", "floor-vp-s?.json"),
        ("supervised ceiling V+P", "sup-vp-s?-n1000.json"),
    ):
        vals = [json.loads(p.read_text())["heldout"]["mean"] for p in sorted(ctrl.glob(glob))]
        print(f"\n{name}: {mean_sd(vals)} (n {len(vals)})")  # noqa: T201
    print("\nTerms at the last step\n")  # noqa: T201
    print("| init | recipe | n | " + " | ".join(TERMS) + " |")  # noqa: T201
    print("|---|---|---|" + "---|" * len(TERMS))  # noqa: T201
    for init in INITS:
        for recipe in RECIPES:
            runs = [
                v["heldout"]
                for k, v in res.items()
                if k.startswith(f"ft-{init}-{recipe}-s") and k.endswith(f"-n{STEPS}")
            ]
            if runs:
                cells = [mean_sd([r[t] for r in runs]) for t in TERMS]
                print(f"| {init} | {recipe} | {len(runs)} | " + " | ".join(cells) + " |")  # noqa: T201


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=("train", "probe", "report"))
    parser.add_argument("--init", choices=INITS)
    parser.add_argument("--recipe", choices=sorted(RECIPES), default="std")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.command == "train":
        train_run(args.init, args.recipe, args.seed, args.workers)
    elif args.command == "probe":
        probe_all()
    else:
        report()


if __name__ == "__main__":
    main()
