"""B-variety-0 follow-up (exploratory): do the encoders' states know the outcomes of actions?

The contact battery found that pretraining on the experiment programs (``variety``) hurts MAE and
the supervised ceiling against babbling. The battery scores the robot's own contacts, force and
fingertips, not what happens to the objects, which is what the programs add. This script probes
the same encoders for object outcomes (``sensorium.model.outcomes``: displaced in the next 1 s,
lifted, toppled in the next 1 s, object speed) on both 100-episode held-out sets of B-variety-0.

Encoders: the 10,000-step checkpoints of the 12 SSL runs, the 6 supervised ceilings, the
random-init floor (5 seeds) and the raw proprio K = 4 readout, as in ``variety_test.py`` (same
split, channel space and window stride). ``encode`` loads each test episode once and encodes it
with every encoder, saving states and labels; ``report`` fits the probes (leave one episode out)
and prints the tables and paired contrasts.

    uv run python scripts/outcome_test.py encode --test babble [--threads 2]
    uv run python scripts/outcome_test.py encode --test variety [--threads 2]
    uv run python scripts/outcome_test.py report

Needs ``runs/variety/split.json``, the held-out episodes under ``data/`` and the checkpoints
under ``runs/variety/runs/`` (``ckpt-10000.pt``; ``ckpt.pt`` for the ceilings).
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import variety_test as variety  # scripts/ is on sys.path when this file runs as a script

from sensorium.episode import Episode
from sensorium.model.outcomes import (
    OUTCOME_BINARY,
    OUTCOMES,
    collect_all,
    predict_outcomes,
)
from sensorium.model.paired import contrast, describe
from sensorium.model.pretrain import load_config, load_layer
from sensorium.model.probes import Predictions, ProbeData, predict_probes
from sensorium.model.sensory_layer import SensoryLayer

STEP = variety.STEPS
ARMS = (*variety.ARMS, *variety.SUPERVISED)


def out_dir() -> Path:
    """Where the encoded test sets go."""
    return variety.ROOT / "outcomes"


def step_of(arm: str) -> int:
    """The checkpoint probed: 10,000 for the SSL arms, the end of training for the ceilings."""
    return variety.SUP_STEPS if arm in variety.SUPERVISED else STEP


def use_checkpoint_space() -> None:
    """Take the arms' shared channel space and normalisation from a checkpoint, so the floors and
    the raw proprio readout do not need the 100 pretraining episodes they were fitted on (every
    checkpoint stores that same space; ``variety_test`` builds it identically for every arm)."""
    first = next(iter(sorted((variety.ROOT / "runs").glob(f"*/ckpt-{STEP}.pt"))))
    space = load_layer(first).space
    variety.shared_space = lambda: space


def encoders() -> dict[str, SensoryLayer]:
    """Every encoder by output name: floors, then each arm's seeds (those on disk)."""
    out = {f"floor-vp-s{s}": variety.floor(s) for s in variety.FLOOR_SEEDS}
    for arm in ARMS:
        for seed in variety.SEEDS:
            run = variety.ROOT / "runs" / f"{arm}-s{seed}"
            path = run / ("ckpt.pt" if arm in variety.SUPERVISED else f"ckpt-{STEP}.pt")
            if path.exists():
                out[f"{arm}-s{seed}-n{step_of(arm)}"] = load_layer(path)
    return out


def encode(test: str) -> None:
    """States of every encoder, the raw proprio features and all labels of one held-out set,
    saved to ``runs/variety/outcomes/<test>.npz``."""
    layers = encoders()
    k = load_config(variety.BASE).model.substeps
    paths = variety.files(test, "heldout")
    done = [0]

    def features(episode: Episode, ends: np.ndarray) -> dict[str, np.ndarray]:
        out = variety.raw_features(episode, ends)
        for name, layer in layers.items():
            states, own = layer.encode(episode, stride=variety.STRIDE)
            if not np.array_equal(own, ends):
                raise ValueError(f"{name}: windows differ from the label windows")
            out[name] = states
        done[0] += 1
        logging.info("%s: encoded %d / %d episodes", test, done[0], len(paths))
        return out

    data = collect_all(paths, k, variety.STRIDE, features)
    out_dir().mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir() / f"{test}.npz",
        groups=data.groups,
        **{f"x/{n}": v for n, v in data.features.items()},
        **{f"y/{n}": v for n, v in data.labels.items()},
    )
    logging.info("wrote %s: %d encoders, %d windows", test, len(layers), len(data.groups))


def load(test: str) -> ProbeData:
    """Inverse of ``encode``'s file."""
    raw = np.load(out_dir() / f"{test}.npz")
    return ProbeData(
        {k.removeprefix("x/"): raw[k] for k in raw.files if k.startswith("x/")},
        {k.removeprefix("y/"): raw[k] for k in raw.files if k.startswith("y/")},
        raw["groups"],
    )


def outcome_predictions(test: str, data: ProbeData, name: str) -> Predictions:
    """Out-of-fold outcome predictions of one encoder (fitted once, then read from
    ``runs/variety/outcomes/preds/<test>/<name>.npz``)."""
    path = out_dir() / "preds" / test / f"{name}.npz"
    if path.exists():
        raw = np.load(path)
        blocks = [slice(int(a), int(b)) for a, b in raw["edges"]]
        return Predictions(
            raw["pred"], raw["target"], blocks, raw["groups"], OUTCOMES, OUTCOME_BINARY
        )
    pred = predict_outcomes(data, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    edges = np.array([[b.start, b.stop] for b in pred.blocks])
    np.savez(path, pred=pred.pred, target=pred.target, groups=pred.groups, edges=edges)
    logging.info("outcome probe %s on %s: %.3f", name, test, pred.scores()["mean"])
    return pred


def battery_predictions(test: str, data: ProbeData, name: str) -> Predictions:
    """The contact battery's out-of-fold predictions saved by ``variety_test.py probe`` (the same
    windows), or fitted here if missing."""
    saved = variety.load_predictions(test, "raw-proprio" if name == "p4" else name)
    if saved is not None and np.array_equal(saved.groups, data.groups):
        return saved
    logging.info("battery probe %s on %s: not saved, fitting", name, test)
    return predict_probes(data, name)


def family(preds: dict[str, Predictions], arm: str) -> list[Predictions]:
    """An arm's predictions in seed order (floors: every floor seed)."""
    if arm == "floor":
        return [preds[f"floor-vp-s{s}"] for s in variety.FLOOR_SEEDS if f"floor-vp-s{s}" in preds]
    names = [f"{arm}-s{s}-n{step_of(arm)}" for s in variety.SEEDS]
    return [preds[n] for n in names if n in preds]


def cell(values: list[float]) -> str:
    """``mean +- sd`` (or the single value)."""
    if not values:
        return "-"
    sd = f" +- {np.std(values, ddof=1):.3f}" if len(values) > 1 else ""
    return f"{np.mean(values):.3f}{sd}"


def report_test(test: str) -> None:
    """Tables and contrasts of one held-out set."""
    data = load(test)
    names = [n for n in data.features if n != "p4"]
    outcome = {n: outcome_predictions(test, data, n) for n in [*names, "p4"]}
    battery = {n: battery_predictions(test, data, n) for n in [*names, "p4"]}
    keys = [f"{n}_auroc" if n in OUTCOME_BINARY else f"{n}_r2" for n in OUTCOMES]
    variety.say(
        f"\n## Outcome probes on {test} ({len(np.unique(data.groups))} held-out episodes)\n"
    )
    rates = " / ".join(f"{data.labels[n].mean():.3f}" for n in OUTCOME_BINARY)
    variety.say(f"Positive rates of the windows, displaced / lifted / toppled: {rates}")
    speed = data.labels["log_speed"]
    variety.say(f"log_speed mean {speed.mean():.2f}, sd {speed.std():.2f}\n")
    variety.say("| encoder | " + " | ".join(keys) + " | outcome mean | battery mean |")
    variety.say("|---|" + "---|" * (len(keys) + 2))
    outcome["raw"] = outcome.pop("p4")
    battery["raw"] = battery.pop("p4")
    rows = ["floor", "raw", *ARMS]
    for arm in rows:
        fam = [outcome["raw"]] if arm == "raw" else family(outcome, arm)
        bat = [battery["raw"]] if arm == "raw" else family(battery, arm)
        cells = [cell([p.scores()[k] for p in fam]) for k in (*keys, "mean")]
        variety.say(
            f"| {arm} | " + " | ".join(cells) + f" | {cell([p.scores()['mean'] for p in bat])} |"
        )
    variety.say(f"\nPaired contrasts of the outcome mean on {test} (seed and episode)\n")
    variety.say("| contrast | outcome mean | battery mean (re-probed) |\n|---|---|---|")
    for obj in ("mae", "jepa", "sup"):
        a, b = f"{obj}-variety", f"{obj}-babble"
        co = contrast(family(outcome, a), family(outcome, b), paired=True)
        cb = contrast(family(battery, a), family(battery, b), paired=True)
        variety.say(f"| {a} - {b} | {describe(co)} | {describe(cb)} |")
    for arm in variety.ARMS:
        co = contrast(family(outcome, arm), family(outcome, "floor"), paired=False)
        variety.say(f"| {arm} - floor | {describe(co)} | |")
    co = contrast([outcome["raw"]], family(outcome, "floor"), paired=False)
    variety.say(f"| raw proprio - floor | {describe(co)} | |")
    variety.say(f"\nPer-target variety - babble on {test} (mean over seeds)\n")
    variety.say("| arm pair | " + " | ".join(keys) + " |\n|---|" + "---|" * len(keys))
    for obj in ("mae", "jepa", "sup"):
        va = family(outcome, f"{obj}-variety")
        ba = family(outcome, f"{obj}-babble")
        deltas = [
            np.mean([p.scores()[k] for p in va]) - np.mean([p.scores()[k] for p in ba])
            for k in keys
        ]
        variety.say(f"| {obj} | " + " | ".join(f"{d:+.3f}" for d in deltas) + " |")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=("encode", "report"))
    parser.add_argument("--test", choices=variety.SETS)
    parser.add_argument("--threads", type=int, default=2, help="torch threads")
    parser.add_argument("--data", type=Path, default=variety.DATA)
    parser.add_argument("--root", type=Path, default=variety.ROOT)
    args = parser.parse_args()
    variety.DATA, variety.ROOT = args.data, args.root
    torch.set_num_threads(args.threads)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.command == "encode":
        if args.test is None:
            parser.error("encode needs --test")
        use_checkpoint_space()
        encode(args.test)
    else:
        logging.getLogger().handlers[0].flush()
        variety.say("Exploratory (not pre-registered): outcome probes of B-variety-0's encoders.")
        for test in variety.SETS:
            report_test(test)


if __name__ == "__main__":
    main()
