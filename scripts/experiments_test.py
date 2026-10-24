"""Random play against scripted experiments at equal sequence count (B-exp-0).

The consequence-rich test of ``scripts/objective_test.py``, rerun on Track A's experiments batch
(``data/experiments_batch``: push, topple, slide, press, knock; hands and pincers; hidden mass,
friction and softness; actuator-effort proprio channels ``eff_*``). Every arm pretrains the tiny
V+P model for 1,000 steps x 16 sequences (the same 16,000 sequences whatever the data) on one shared
channel space built from the random-play and experiment pretraining files (channels an episode
kind lacks read 0 after normalisation) and is probed on **two** held-out sets, same battery:
``rand`` (the 20 held-out random-play episodes of ``runs/decisive/split.json``) and ``exp``
(held-out experiment episodes, a fixed number per variant). Arms:

* ``rand``: uniform windows of 100 random-play episodes (60,000 env steps; the baseline);
* ``exp``: uniform windows of the experiment pretraining episodes;
* ``exp-conseq``: only windows where hand-object contact moves the object (offline, privileged
  object motion picks windows, never an input);
* ``exp-noeff``: as ``exp`` with the effort channels left out of the proprio (``drop_prefix``).

Yardsticks next to the frozen probe: random-init floor (5 seeds), supervised ceiling per set
(``sup-rand``, ``sup-exp``) and a 16-labelled-episode fine-tune from random init, ``rand``,
``exp`` and ``exp-conseq`` encoders (600 steps, probed at 100/200/400/600). Probes every 250 steps.

    uv run python scripts/experiments_test.py split --exp-dir data/experiments_batch/humanoid \\
        --exclude humanoid-pincer_slide humanoid-pincer_press
    uv run python scripts/experiments_test.py index
    uv run python scripts/experiments_test.py train --arm exp --seed 0
    uv run python scripts/experiments_test.py ft --init exp --seed 0 --labels exp
    uv run python scripts/experiments_test.py sup --labels exp --seed 0
    uv run python scripts/experiments_test.py probe
    uv run python scripts/experiments_test.py report
"""

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from sensorium.model.controls import SupervisedConfig, train_supervised
from sensorium.model.data import LoaderConfig
from sensorium.model.diagnostics import heldout_loss
from sensorium.model.pretrain import (
    RunConfig,
    TrainConfig,
    build_space,
    load_config,
    load_layer,
    train,
)
from sensorium.model.probes import collect, predict_probes
from sensorium.model.sensory_layer import SensoryLayer, SensoryModel
from sensorium.model.subsets import build_index, save_index, summarise

ROOT = Path("runs/exp")
SPLIT = ROOT / "split.json"
INDEX = ROOT / "index.json"
RAND_SPLIT = Path("runs/decisive/split.json")
RAND_DIR = Path("data/probe_test/humanoid")
STEPS, FT_STEPS = 1000, 600
CKPTS, FT_CKPTS = (250, 500, 750, 1000), (100, 200, 400, 600)
STRIDE = 8
K = 16
NO_TOUCH = ("vision", "proprio_body", "vestibular", "proprio_effector")
ARMS: dict[str, dict[str, object]] = {
    "rand": {"data": "rand"},
    "exp": {"data": "exp"},
    "exp-conseq": {"data": "exp", "subset": "conseq"},
    "exp-noeff": {"data": "exp", "drop": ("eff_",)},
}
SETS = ("rand", "exp")
TERMS = (
    "contact_left_auroc",
    "contact_right_auroc",
    "contact_object_auroc",
    "log_force_r2",
    "tip_r2",
)


def paths(key: str) -> list[Path]:
    """Files of ``rand_pretrain``, ``rand_heldout``, ``exp_pretrain`` or ``exp_heldout``."""
    return [Path(p) for p in json.loads(SPLIT.read_text())[key]]


def make_split(exp_dir: Path, holdout: int, exclude: list[str], seed: int) -> None:
    """Write the split: ``holdout`` episodes per experiment variant held out, the rest pretrain."""
    rand = json.loads(RAND_SPLIT.read_text())
    rng = np.random.default_rng(seed)
    variants: dict[str, list[Path]] = {}
    for path in sorted(exp_dir.glob("*.npz")):
        variants.setdefault(path.name.rsplit("-experiment-", 1)[0], []).append(path)
    pretrain: list[str] = []
    heldout: list[str] = []
    for name, files in sorted(variants.items()):
        if name in exclude:
            continue
        order = rng.permutation(len(files))
        heldout += [str(files[i].resolve()) for i in sorted(order[:holdout])]
        pretrain += [str(files[i].resolve()) for i in sorted(order[holdout:])]
    split = {
        "rand_pretrain": [str((RAND_DIR / n).resolve()) for n in rand["pretrain"]],
        "rand_heldout": [str((RAND_DIR / n).resolve()) for n in rand["heldout"]],
        "exp_pretrain": pretrain,
        "exp_heldout": heldout,
        "variants": {k: len(v) for k, v in variants.items()},
        "excluded": exclude,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    SPLIT.write_text(json.dumps(split, indent=1))
    logging.info(
        "split: %d random-play + %d experiment pretraining, held out %d + %d; excluded %s",
        len(split["rand_pretrain"]),
        len(pretrain),
        len(split["rand_heldout"]),
        len(heldout),
        exclude,
    )


def space_files() -> list[str]:
    """Pretraining files of both kinds, interleaved, for the shared channel space."""
    a, b = paths("rand_pretrain"), paths("exp_pretrain")
    out: list[Path] = []
    for i in range(max(len(a), len(b))):
        out += [x[i] for x in (a, b) if i < len(x)]
    return [str(p) for p in out]


def make_index() -> None:
    """Build the consequence index over the experiment pretraining episodes."""
    cfg = load_config(Path("configs/model/cpu_tiny_a0.toml"))
    index = build_index(paths("exp_pretrain"), cfg.model.steps * cfg.model.substeps)
    save_index(index, INDEX)
    logging.info("index: %s", summarise(index))


def train_arm(arm: str, seed: int, steps: int, workers: int) -> None:
    """Pretrain one arm on the shared space; checkpoints every 250 steps."""
    spec = ARMS[arm]
    base = load_config(Path("configs/model/cpu_tiny_a0.toml"))
    tc = TrainConfig(
        steps=steps,
        batch_size=16,
        lr=base.train.lr,
        log_every=50,
        eval_windows=0,
        num_workers=workers,
        out_dir=str(ROOT),
        save_steps=tuple(s for s in CKPTS if s <= steps),
    )
    data = replace(
        base.data,
        paths=tuple(str(p) for p in paths(f"{spec['data']}_pretrain")),
        space_paths=tuple(space_files()),
        stats_episodes=1000,
        drop_prefix=tuple(spec.get("drop", ())),  # type: ignore[call-overload]
    )
    if "subset" in spec:
        data = replace(data, subset_index=str(INDEX), subset_key=str(spec["subset"]))
    model = replace(base.model, senses=NO_TOUCH)
    train(
        RunConfig(f"{arm}-s{seed}", seed, data, replace(base.augment, cache_episodes=4), model, tc)
    )


def shared_space(drop: tuple[str, ...] = ()):  # type: ignore[no-untyped-def]
    """The channel space shared by every arm (statistics over both kinds of pretraining file)."""
    return build_space([Path(p) for p in space_files()], 1000, drop)


def labelled(labels: str, seed: int) -> list[Path]:
    """``K`` labelled pretraining episodes of a set (a seeded subset)."""
    files = paths(f"{labels}_pretrain")
    picked = np.random.default_rng(seed).choice(len(files), K, replace=False)
    return [files[i] for i in sorted(picked)]


def finetune(init: str, labels: str, seed: int, workers: int) -> None:
    """Fine-tune the V+P encoder on ``K`` labelled episodes from random or an SSL init."""
    base = load_config(Path("configs/model/cpu_tiny_a0.toml"))
    ckpt = None if init == "random" else ROOT / f"{init}-s{seed}" / "ckpt-1000.pt"
    cfg = SupervisedConfig(
        model=replace(base.model, senses=NO_TOUCH),
        steps=FT_STEPS,
        seed=seed,
        workers=workers,
        save_steps=FT_CKPTS,
    )
    train_supervised(
        cfg,
        labelled(labels, seed),
        shared_space(),
        ROOT / f"ft-{init}-{labels}-s{seed}",
        logging.info,
        init=ckpt,
    )


def supervised(labels: str, seed: int, workers: int) -> None:
    """The supervised ceiling: the V+P encoder on every pretraining episode of a set."""
    base = load_config(Path("configs/model/cpu_tiny_a0.toml"))
    cfg = SupervisedConfig(
        model=replace(base.model, senses=NO_TOUCH), steps=STEPS, seed=seed, workers=workers
    )
    train_supervised(
        cfg,
        paths(f"{labels}_pretrain"),
        shared_space(),
        ROOT / f"sup-{labels}-s{seed}",
        logging.info,
    )


def probe_json(layer: SensoryLayer, with_loss: bool) -> dict[str, object]:
    """Held-out probe scores on both sets (and the held-out loss of an SSL checkpoint)."""
    out: dict[str, object] = {}
    for name in SETS:
        data = collect(layer, paths(f"{name}_heldout"), STRIDE, (), raw=False)
        out[name] = predict_probes(data, "state").scores()
    out["heldout"] = out["exp"]  # default metric of plot_runs.py
    if with_loss:
        out |= heldout_loss(layer, paths("exp_heldout"))
    return out


def probe_all() -> None:
    """Probe every checkpoint and the floors that have no JSON yet."""
    out = ROOT / "probes"
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(Path("configs/model/cpu_tiny_a0.toml"))
    for drop, tag in (((), "floor"), (("eff_",), "floor-noeff")):
        space = shared_space(drop)
        for seed in range(5):
            target = out / f"{tag}-s{seed}.json"
            if not target.exists():
                torch.manual_seed(seed)
                model = replace(cfg.model, senses=NO_TOUCH)
                loader = LoaderConfig(
                    steps=model.steps, substeps=model.substeps, view_size=model.vision_size
                )
                layer = SensoryLayer(SensoryModel(model, space), space, loader)
                target.write_text(json.dumps(probe_json(layer, False)))
                logging.info("probed %s", target.name)
    for run in sorted(ROOT.glob("*-s?")):
        ssl = not run.name.startswith(("ft-", "sup-"))
        steps = CKPTS if ssl else (FT_CKPTS if run.name.startswith("ft-") else (STEPS,))
        for step in steps:
            ckpt = run / (f"ckpt-{step}.pt" if step != steps[-1] or ssl else "ckpt.pt")
            target = out / f"{run.name}-n{step}.json"
            if ckpt.exists() and not target.exists():
                target.write_text(json.dumps(probe_json(load_layer(ckpt), ssl)))
                logging.info("probed %s", target.name)


def mean_sd(values: list[float]) -> str:
    """``mean +- sd`` over seeds."""
    arr = np.array(values)
    return f"{arr.mean():.2f} +- {arr.std():.2f}" if len(arr) > 1 else f"{arr[0]:.2f}"


def table(header: str, columns: tuple[int, ...], rows: list[tuple[str, list[list[float]]]]) -> None:
    """Print a markdown table of ``mean +- sd (n)`` cells."""
    print("| " + header + " | " + " | ".join(f"{s} steps" for s in columns) + " |")  # noqa: T201
    print("|---|" + "---|" * len(columns))  # noqa: T201
    for name, cells in rows:
        shown = [f"{mean_sd(c)} ({len(c)})" if c else "-" for c in cells]
        print(f"| {name} | " + " | ".join(shown) + " |")  # noqa: T201


def report() -> None:
    """Frozen-probe curves, fine-tune curves and terms for both probe sets."""
    res = {p.stem: json.loads(p.read_text()) for p in (ROOT / "probes").glob("*.json")}

    def get(prefix: str, step: int, which: str) -> list[float]:
        return [
            v[which]["mean"]
            for k, v in res.items()
            if k.startswith(prefix + "-s") and k.endswith(f"-n{step}")
        ]

    def refs(which: str) -> list[tuple[str, list[list[float]]]]:
        rows = []
        for tag in ("floor", "floor-noeff", f"sup-{which}"):
            vals = [v[which]["mean"] for k, v in res.items() if k.startswith(tag + "-s")]
            rows.append((tag, [vals] if vals else []))
        return rows

    for which in SETS:
        print(f"\n### Probe set `{which}`: frozen probe mean (mean +- sd over seeds, n)\n")  # noqa: T201
        table("arm", CKPTS, [(a, [get(a, s, which) for s in CKPTS]) for a in ARMS] + refs(which))
        print(f"\nFine-tune on {K} labelled `{which}` episodes, probed on `{which}`\n")  # noqa: T201
        inits = ("random", "rand", "exp", "exp-conseq")
        table(
            "init",
            FT_CKPTS,
            [(i, [get(f"ft-{i}-{which}", s, which) for s in FT_CKPTS]) for i in inits],
        )
        print(f"\nTerms at 1,000 steps (probe set `{which}`)\n")  # noqa: T201
        print("| arm | n | " + " | ".join(TERMS) + " |")  # noqa: T201
        print("|---|---|" + "---|" * len(TERMS))  # noqa: T201
        for arm in ARMS:
            runs = [
                v[which]
                for k, v in res.items()
                if k.startswith(arm + "-s") and k.endswith("-n1000")
            ]
            if runs:
                cells = [mean_sd([r[t] for r in runs]) for t in TERMS]
                print(f"| {arm} | {len(runs)} | " + " | ".join(cells) + " |")  # noqa: T201


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "command", choices=("split", "index", "train", "ft", "sup", "probe", "report")
    )
    parser.add_argument("--arm", choices=sorted(ARMS))
    parser.add_argument("--init", choices=("random", "rand", "exp", "exp-conseq"))
    parser.add_argument("--labels", choices=SETS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--exp-dir", type=Path, default=Path("data/experiments_batch/humanoid"))
    parser.add_argument("--holdout", type=int, default=2)
    parser.add_argument("--exclude", nargs="*", default=[])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.command == "split":
        make_split(args.exp_dir, args.holdout, args.exclude, seed=0)
    elif args.command == "index":
        make_index()
    elif args.command == "train":
        train_arm(args.arm, args.seed, args.steps, args.workers)
    elif args.command == "ft":
        finetune(args.init, args.labels, args.seed, args.workers)
    elif args.command == "sup":
        supervised(args.labels, args.seed, args.workers)
    elif args.command == "probe":
        probe_all()
    else:
        report()


if __name__ == "__main__":
    main()
