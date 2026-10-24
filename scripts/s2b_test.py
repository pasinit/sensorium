"""Hand experiments (S2b) against random play at equal sequence count and size (B-s2b-0).

Track A's ``experiments_s2b`` hand programs (push, topple, knock, press, slide, door; hidden mass,
friction and softness per object and episode; actuator-effort proprio channels ``eff_*``), 2 of 12
episodes per variant held out, the other 60 for pretraining (36,000 env steps). The control is a
seeded random subsample of 60 of the 100 random-play pretraining episodes (``rand60``, 36,000 env
steps, both effectors). Every arm pretrains the tiny model for 1,000 steps x 16 sequences on one
shared channel space (channels a kind lacks read 0) and is probed every 250 steps on two held-out
sets, ``exp`` (the 12 held-out hand-experiment episodes) and ``rand`` (the 20 held-out random-play
episodes, for comparability), with the contact battery and, on ``exp``, the hidden-property probes
(R^2 of log mass, friction and log softness of the touched object, contact windows only). Arms:

* V+P: ``rand60-vp``, ``s2b-vp``, ``s2b-vp-noeff`` (no ``eff_*`` channels);
* all six senses: ``rand60-all``, ``s2b-all``, ``s2b-all-tw`` (loss weighted by the touch change),
  ``s2b-all-noeff``.

Yardsticks: random-init floors (5 seeds, 3 without effort), supervised ceilings trained on the
contact **and** hidden labels (``sup-vp``, ``sup-all``), and a 16-labelled-episode fine-tune
(600 steps, probed at 100/200/400/600 on ``exp``) from random, ``rand60-vp``, ``s2b-vp`` encoders.

    uv run python scripts/s2b_test.py split --exp-dir data/experiments_s2b/humanoid
    uv run python scripts/s2b_test.py train --arm s2b-vp --seed 0
    uv run python scripts/s2b_test.py sup --arm sup-vp --seed 0
    uv run python scripts/s2b_test.py ft --init s2b-vp --seed 0
    uv run python scripts/s2b_test.py probe
    uv run python scripts/s2b_test.py report
"""

import argparse
import json
import logging
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from sensorium.model.controls import SupervisedConfig, train_supervised
from sensorium.model.data import LoaderConfig
from sensorium.model.diagnostics import heldout_loss
from sensorium.model.layout import ChannelSpace
from sensorium.model.pretrain import (
    RunConfig,
    TrainConfig,
    build_space,
    load_config,
    load_layer,
    train,
)
from sensorium.model.probes import collect, predict_hidden, predict_probes
from sensorium.model.sensory_layer import SensoryLayer, SensoryModel

ROOT = Path("runs/s2b")
SPLIT = ROOT / "split.json"
RAND_SPLIT = Path("runs/decisive/split.json")
RAND_DIR = Path("data/probe_test/humanoid")
STEPS, FT_STEPS = 1000, 600
CKPTS, FT_CKPTS = (250, 500, 750, 1000), (100, 200, 400, 600)
STRIDES = {"exp": 4, "rand": 8}
"""Env steps between windows: denser on the 12 experiment episodes (more contact windows for the
hidden-property probes)."""
K = 16
NO_TOUCH = ("vision", "proprio_body", "vestibular", "proprio_effector")
ALL = None  # every sense of the tiny config
EFF = ("eff_",)
ARMS: dict[str, dict[str, object]] = {
    "s2b-vp": {"data": "exp_pretrain", "senses": NO_TOUCH},
    "rand60-vp": {"data": "rand60", "senses": NO_TOUCH},
    "s2b-vp-noeff": {"data": "exp_pretrain", "senses": NO_TOUCH, "drop": EFF},
    "s2b-all": {"data": "exp_pretrain", "senses": ALL},
    "rand60-all": {"data": "rand60", "senses": ALL},
    "s2b-all-tw": {"data": "exp_pretrain", "senses": ALL, "surprise": "touch"},
    "s2b-all-noeff": {"data": "exp_pretrain", "senses": ALL, "drop": EFF},
}
SUPS = {"sup-vp": NO_TOUCH, "sup-all": ALL}
FLOORS = {
    "floor-vp": (NO_TOUCH, (), 5),
    "floor-all": (ALL, (), 5),
    "floor-vp-noeff": (NO_TOUCH, EFF, 3),
    "floor-all-noeff": (ALL, EFF, 3),
}
SETS = ("exp", "rand")
TERMS = (
    "contact_left_auroc",
    "contact_right_auroc",
    "contact_object_auroc",
    "log_force_r2",
    "tip_r2",
)
HIDDEN_TERMS = ("hidden_log_mass_r2", "hidden_friction_r2", "hidden_softness_r2")


def paths(key: str) -> list[Path]:
    """Files of a split key: ``rand_pretrain/heldout``, ``rand60``, ``exp_pretrain/heldout``."""
    return [Path(p) for p in json.loads(SPLIT.read_text())[key]]


def make_split(exp_dir: Path, holdout: int, seed: int) -> None:
    """Hold out ``holdout`` episodes per hand variant (stratified); subsample the control."""
    rand = json.loads(RAND_SPLIT.read_text())
    rng = np.random.default_rng(seed)
    variants: dict[str, list[Path]] = {}
    for path in sorted(exp_dir.glob("*hand_*.npz")):
        variants.setdefault(path.name.rsplit("-experiment-", 1)[0], []).append(path)
    pretrain: list[str] = []
    heldout: list[str] = []
    for _, files in sorted(variants.items()):
        order = rng.permutation(len(files))
        heldout += [str(files[i].resolve()) for i in sorted(order[:holdout])]
        pretrain += [str(files[i].resolve()) for i in sorted(order[holdout:])]
    rand_pretrain = [str((RAND_DIR / n).resolve()) for n in rand["pretrain"]]
    picked = sorted(rng.choice(len(rand_pretrain), len(pretrain), replace=False))
    split = {
        "rand_pretrain": rand_pretrain,
        "rand_heldout": [str((RAND_DIR / n).resolve()) for n in rand["heldout"]],
        "rand60": [rand_pretrain[i] for i in picked],
        "exp_pretrain": pretrain,
        "exp_heldout": heldout,
        "variants": {k: len(v) for k, v in variants.items()},
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    SPLIT.write_text(json.dumps(split, indent=1))
    logging.info(
        "split: %d experiment pretraining (%d held out), %d random-play control",
        len(pretrain),
        len(heldout),
        len(split["rand60"]),
    )


def space_files() -> list[str]:
    """Pretraining files of both kinds, interleaved, for the shared channel space."""
    a, b = paths("rand_pretrain"), paths("exp_pretrain")
    out: list[Path] = []
    for i in range(max(len(a), len(b))):
        out += [x[i] for x in (a, b) if i < len(x)]
    return [str(p) for p in out]


def shared_space(drop: tuple[str, ...] = ()) -> ChannelSpace:
    """The channel space shared by every arm (statistics over both kinds of pretraining file)."""
    return build_space([Path(p) for p in space_files()], 1000, drop)


def tiny() -> RunConfig:
    """The tiny model configuration and loader settings."""
    return load_config(Path("configs/model/cpu_tiny_a0.toml"))


def train_arm(arm: str, seed: int, workers: int) -> None:
    """Pretrain one arm on the shared space; checkpoints every 250 steps."""
    spec, base = ARMS[arm], tiny()
    tc = TrainConfig(
        steps=STEPS,
        batch_size=16,
        lr=base.train.lr,
        log_every=50,
        eval_windows=0,
        num_workers=workers,
        out_dir=str(ROOT),
        save_steps=CKPTS,
    )
    data = replace(
        base.data,
        paths=tuple(str(p) for p in paths(str(spec["data"]))),
        space_paths=tuple(space_files()),
        stats_episodes=1000,
        drop_prefix=tuple(spec.get("drop", ())),  # type: ignore[call-overload]
    )
    senses = spec["senses"] or base.model.senses
    model = replace(base.model, senses=senses, surprise=str(spec.get("surprise", "none")))  # type: ignore[arg-type]
    train(
        RunConfig(f"{arm}-s{seed}", seed, data, replace(base.augment, cache_episodes=4), model, tc)
    )


def supervised(arm: str, seed: int, workers: int) -> None:
    """The supervised ceiling: contact and hidden-property labels on the pretraining episodes."""
    base = tiny()
    senses = SUPS[arm] or base.model.senses
    cfg = SupervisedConfig(
        model=replace(base.model, senses=senses),
        steps=STEPS,
        seed=seed,
        workers=workers,
        hidden=True,
    )
    train_supervised(
        cfg, paths("exp_pretrain"), shared_space(), ROOT / f"{arm}-s{seed}", logging.info
    )


def finetune(init: str, seed: int, workers: int) -> None:
    """Fine-tune the V+P encoder on ``K`` labelled s2b episodes from random or an SSL init."""
    base = tiny()
    ckpt = None if init == "random" else ROOT / f"{init}-s{seed}" / "ckpt-1000.pt"
    files = paths("exp_pretrain")
    picked = np.random.default_rng(seed).choice(len(files), K, replace=False)
    cfg = SupervisedConfig(
        model=replace(base.model, senses=NO_TOUCH),
        steps=FT_STEPS,
        seed=seed,
        workers=workers,
        save_steps=FT_CKPTS,
    )
    train_supervised(
        cfg,
        [files[i] for i in sorted(picked)],
        shared_space(),
        ROOT / f"ft-{init}-s{seed}",
        logging.info,
        init=ckpt,
    )


def probe_json(layer: SensoryLayer, sets: tuple[str, ...], with_loss: bool) -> dict[str, object]:
    """Contact and hidden-property probe scores on the held-out sets.

    ``exp`` and ``rand`` use the encoder's natural inputs; an all-sense encoder is also probed with
    touch and audio masked on ``exp`` (``exp_vp``). Hidden-property scores are ``exp_hidden`` (and
    ``exp_vp_hidden``). ``heldout`` is ``exp`` (the default metric of ``plot_runs.py``).
    """
    out: dict[str, object] = {}
    masked = tuple(s for s in ("touch", "audio") if s in layer.model.senses)
    for name in sets:
        for tag, drop in ((name, ()), (f"{name}_vp", masked)):
            if tag.endswith("_vp") and (not masked or name != "exp"):
                continue
            data = collect(layer, paths(f"{name}_heldout"), STRIDES[name], drop, raw=False)
            out[tag] = predict_probes(data, "state").scores()
            if name == "exp":
                out[f"{tag}_hidden"] = predict_hidden(data, "state")
    out["heldout"] = out.get("exp", {})
    if with_loss and "exp" in sets:
        out |= heldout_loss(layer, paths("exp_heldout"))
    return out


def probe_all() -> None:
    """Probe the floors and every checkpoint that has no JSON yet."""
    out = ROOT / "probes"
    out.mkdir(parents=True, exist_ok=True)
    base = tiny()
    for tag, (senses, drop, seeds) in FLOORS.items():
        space = shared_space(drop)
        for seed in range(seeds):
            target = out / f"{tag}-s{seed}.json"
            if not target.exists():
                torch.manual_seed(seed)
                model = replace(base.model, senses=senses or base.model.senses)
                loader = LoaderConfig(
                    steps=model.steps, substeps=model.substeps, view_size=model.vision_size
                )
                layer = SensoryLayer(SensoryModel(model, space), space, loader)
                target.write_text(json.dumps(probe_json(layer, SETS, False)))
                logging.info("probed %s", target.name)
    for run in sorted(ROOT.glob("*-s?")):
        kind = run.name.rsplit("-s", 1)[0]
        ft, sup = kind.startswith("ft-"), kind.startswith("sup-")
        steps = FT_CKPTS if ft else ((STEPS,) if sup else CKPTS)
        for step in steps:
            ckpt = run / (f"ckpt-{step}.pt" if step != steps[-1] or not (ft or sup) else "ckpt.pt")
            target = out / f"{run.name}-n{step}.json"
            if ckpt.exists() and not target.exists():
                sets = ("exp",) if ft else SETS
                target.write_text(json.dumps(probe_json(load_layer(ckpt), sets, not (ft or sup))))
                logging.info("probed %s", target.name)


def mean_sd(values: list[float]) -> str:
    """``mean +- sd`` over seeds (NaNs dropped)."""
    arr = np.array([v for v in values if np.isfinite(v)])
    if not len(arr):
        return "-"
    return f"{arr.mean():.2f} +- {arr.std():.2f}" if len(arr) > 1 else f"{arr[0]:.2f}"


def table(header: str, columns: tuple[int, ...], rows: list[tuple[str, list[list[float]]]]) -> None:
    """Print a markdown table of ``mean +- sd (n)`` cells."""
    print("| " + header + " | " + " | ".join(f"{s} steps" for s in columns) + " |")  # noqa: T201
    print("|---|" + "---|" * len(columns))  # noqa: T201
    for name, cells in rows:
        shown = [f"{mean_sd(c)} ({len(c)})" if c else "-" for c in cells]
        print(f"| {name} | " + " | ".join(shown) + " |")  # noqa: T201


def report() -> None:
    """Frozen-probe curves, hidden-property scores, fine-tune curves and terms."""
    res = {p.stem: json.loads(p.read_text()) for p in (ROOT / "probes").glob("*.json")}

    def get(name: str, step: int | None, key: str, term: str = "mean") -> list[float]:
        """Scores of every seed of run ``name`` (at ``step``; ``None``: files without a step)."""
        found = []
        for stem, value in res.items():
            m = re.fullmatch(r"(.+)-s(\d+)(?:-n(\d+))?", stem)
            file_step = int(m[3]) if m and m[3] else None
            if m and m[1] == name and file_step == step and key in value:
                found.append(value[key].get(term, float("nan")))
        return found

    def refs(key: str, term: str) -> list[tuple[str, list[list[float]]]]:
        rows = [(t, [get(t, None, key, term)]) for t in FLOORS]
        return rows + [(t, [get(t, STEPS, key, term)]) for t in SUPS]

    for key in ("exp", "exp_vp", "rand"):
        print(f"\n### Probe set `{key}`: frozen probe mean (mean +- sd over seeds, n)\n")  # noqa: T201
        arms = [a for a in ARMS if key != "exp_vp" or "all" in a]
        table("arm", CKPTS, [(a, [get(a, s, key) for s in CKPTS]) for a in arms])
        rows = [r for r in refs(key, "mean") if r[1] != [[]]]
        table("reference (final)", (STEPS,), [(n, c) for n, c in rows])
    for key in ("exp_hidden", "exp_vp_hidden"):
        for term in HIDDEN_TERMS:
            print(f"\n### Hidden property `{term}` on `{key}` (R^2 on contact windows)\n")  # noqa: T201
            arms = [a for a in ARMS if key != "exp_vp_hidden" or "all" in a]
            table("arm", CKPTS, [(a, [get(a, s, key, term) for s in CKPTS]) for a in arms])
            table("reference", (STEPS,), [r for r in refs(key, term) if r[1] != [[]]])
    print("\n### Fine-tune on 16 labelled s2b episodes, probed on `exp`\n")  # noqa: T201
    table(
        "init",
        FT_CKPTS,
        [
            (i, [get(f"ft-{i}", s, "exp") for s in FT_CKPTS])
            for i in ("random", "rand60-vp", "s2b-vp")
        ],
    )
    print("\n### Terms at 1,000 steps (`exp`)\n")  # noqa: T201
    print("| arm | n | " + " | ".join(TERMS) + " |")  # noqa: T201
    print("|---|---|" + "---|" * len(TERMS))  # noqa: T201
    for arm in ARMS:
        cells = [get(arm, 1000, "exp", t) for t in TERMS]
        if cells[0]:
            print(f"| {arm} | {len(cells[0])} | " + " | ".join(mean_sd(c) for c in cells) + " |")  # noqa: T201


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=("split", "train", "sup", "ft", "probe", "report"))
    parser.add_argument("--arm", choices=sorted({*ARMS, *SUPS}))
    parser.add_argument("--init", choices=("random", "rand60-vp", "s2b-vp"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--exp-dir", type=Path, default=Path("data/experiments_s2b/humanoid"))
    parser.add_argument("--holdout", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.command == "split":
        make_split(args.exp_dir, args.holdout, seed=0)
    elif args.command == "train":
        train_arm(args.arm, args.seed, args.workers)
    elif args.command == "sup":
        supervised(args.arm, args.seed, args.workers)
    elif args.command == "ft":
        finetune(args.init, args.seed, args.workers)
    elif args.command == "probe":
        probe_all()
    else:
        report()


if __name__ == "__main__":
    main()
