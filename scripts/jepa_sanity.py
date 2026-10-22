"""Tiny Mac sanity runs for the JEPA diagnosis: which change stops the early drop below the floor?

B-variety-0's JEPA loses contact information within its first 250 steps (-0.026 to -0.077 from
its own step 0). These runs repeat those 250 steps on the Mac with one change at a time and probe
the encoder at step 0 (the floor of that seed) and at the end. Training data: the 100 held-out
babble_v2 episodes; probe set: the 100 held-out variety episodes (never trained on here). The
model, batch and sampler are B-variety-0's (``configs/model/cpu_tiny.toml``, V+P, 16 x 8 steps,
128 px frame cache), except for the variant's change.

Variants: ``base``; ``reuse1`` (each batch from 16 episodes instead of 2: ``augment.reuse`` 1);
``sigreg1``, ``sigreg001``, ``sigreg0`` (SIGReg weight 1.0, 0.01, 0 instead of 0.1); ``dim256``
(state width 256 instead of 128; LeWM reports a drop below about 184); ``teacher`` (touch as a
prediction target only, ``model.teacher``); ``teacher_in`` (touch as input AND target: this feeds
touch to the encoder, so its contact probe is not comparable with the V+P arms); ``stopgrad`` (no
gradient into the next-state targets); ``mae`` (the reference objective); ``recon05`` (JEPA plus
pre-projector reconstruction, weight 0.5); ``drop02`` (proprio dropout 0.2). ``probe`` scores every
checkpoint on unseen variety episodes, ``trainprobe`` on the run's own training episodes.

    uv run python scripts/jepa_sanity.py cache
    uv run python scripts/jepa_sanity.py train --variant reuse1 --seed 0 [--steps 250]
    uv run python scripts/jepa_sanity.py probe
    uv run python scripts/jepa_sanity.py trainprobe
"""

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import variety_test as variety  # scripts/ is on sys.path when this file runs as a script

from sensorium.episode import Episode
from sensorium.model.controls import build_probe_data
from sensorium.model.frame_cache import build_cache
from sensorium.model.pretrain import RunConfig, TrainConfig, load_config, load_layer, train
from sensorium.model.probes import Predictions

ROOT = Path("runs/jepa_sanity")
CACHE = Path("data/sanity_babble_cache128")
VARIANTS: dict[str, tuple[dict[str, object], dict[str, object]]] = {
    "base": ({}, {}),
    "reuse1": ({}, {"reuse": 1}),
    "sigreg1": ({"sigreg_weight": 1.0}, {}),
    "stopgrad": ({"target_grad": False}, {}),
    "mae": ({"objective": "mae"}, {}),
    "sigreg0": ({"sigreg_weight": 0.0}, {}),
    "sigreg001": ({"sigreg_weight": 0.01}, {}),
    "dim256": ({"dim": 256}, {}),
    "teacher": ({"teacher": "touch"}, {}),
    "teacher_in": ({"teacher": "touch", "senses": (*variety.NO_TOUCH, "touch")}, {}),
    "recon05": ({"recon_weight": 0.5}, {}),
    "drop02": ({"proprio_dropout": 0.2}, {}),
}


def train_paths() -> list[Path]:
    """The training episodes: B-variety-0's 100 held-out babble_v2 episodes."""
    return variety.files("babble", "heldout")


def run(variant: str, seed: int, steps: int) -> None:
    """Train one variant into ``runs/jepa_sanity/runs/<variant>-s<seed>`` (steps 0 and end kept)."""
    base = load_config(variety.BASE)
    model_extra, augment_extra = VARIANTS[variant]
    paths = tuple(str(p) for p in train_paths())
    data = replace(base.data, paths=paths, space_paths=paths, drop_prefix=variety.DROP)
    model = replace(base.model, **({"senses": variety.NO_TOUCH} | model_extra))  # type: ignore[arg-type]
    augment = replace(base.augment, cache_episodes=4, frame_cache=str(CACHE), **augment_extra)  # type: ignore[arg-type]
    tc = TrainConfig(
        steps=steps,
        batch_size=16,
        lr=base.train.lr,
        log_every=25,
        eval_windows=0,
        num_workers=1,
        out_dir=str(ROOT / "runs"),
        save_steps=(0, steps),
        threads=2,
    )
    train(RunConfig(f"{variant}-s{seed}", seed, data, augment, model, tc))


def probe() -> None:
    """Contact battery (5-fold ridge, as ``jepa_diagnostics``) of every checkpoint, post and pre
    projector, on the held-out variety episodes; then the table of change from step 0."""
    import jepa_diagnostics as diag  # noqa: PLC0415 (a sibling script)

    paths = variety.files("variety", "heldout")
    layers = {
        f"{ck.parent.name}-n{ck.stem.removeprefix('ckpt-')}": load_layer(ck)
        for ck in sorted((ROOT / "runs").glob("*/ckpt-*.pt"))
    }
    out = ROOT / "probes.json"
    done = json.loads(out.read_text()) if out.exists() else {}
    todo = {n: layer for n, layer in layers.items() if n not in done}
    hooks = {n: diag.summaries(layer) for n, layer in todo.items()}
    k = load_config(variety.BASE).model.substeps

    def features(episode: Episode, ends: np.ndarray) -> dict[str, np.ndarray]:
        feats = {}
        for name, layer in todo.items():
            hooks[name].clear()
            feats[f"post/{name}"] = layer.encode(episode, stride=variety.STRIDE)[0]
            feats[f"pre/{name}"] = np.concatenate(hooks[name])
        return feats

    if todo:
        data = build_probe_data(paths, k, variety.STRIDE, features)
        target, blocks = diag.battery(data.labels)
        for name in todo:
            done[name] = {}
            for where in ("post", "pre"):
                x = data.features[f"{where}/{name}"]
                pred = diag.ridge_cv(x, target, data.groups, blocks, scale=False)
                done[name][where] = Predictions(pred, target, blocks, data.groups).scores()
            logging.info(
                "%s: post %.3f pre %.3f",
                name,
                done[name]["post"]["mean"],
                done[name]["pre"]["mean"],
            )
        out.write_text(json.dumps(done, indent=1))
    table(done)


def train_probe() -> None:
    """The contact battery of every checkpoint on its own training episodes (the 100 held-out
    babble_v2 episodes, all seen in training), next to ``probe``'s scores on the variety set."""
    import jepa_diagnostics as diag  # noqa: PLC0415 (a sibling script)

    out = ROOT / "train_probes.json"
    done = json.loads(out.read_text()) if out.exists() else {}
    layers = {
        f"{ck.parent.name}-n{ck.stem.removeprefix('ckpt-')}": load_layer(ck)
        for ck in sorted((ROOT / "runs").glob("*/ckpt-*.pt"))
    }
    todo = {n: layer for n, layer in layers.items() if n not in done}
    hooks = {n: diag.summaries(layer) for n, layer in todo.items()}
    k = load_config(variety.BASE).model.substeps

    def features(episode: Episode, ends: np.ndarray) -> dict[str, np.ndarray]:
        feats = {}
        for name, layer in todo.items():
            hooks[name].clear()
            feats[f"post/{name}"] = layer.encode(episode, stride=variety.STRIDE)[0]
            feats[f"pre/{name}"] = np.concatenate(hooks[name])
        return feats

    if todo:
        data = build_probe_data(train_paths(), k, variety.STRIDE, features)
        for name in todo:
            done[name] = {
                where: diag.battery_score(
                    data.features[f"{where}/{name}"], data.labels, data.groups
                )
                for where in ("post", "pre")
            }
            logging.info("%s on its training episodes: %s", name, done[name])
        out.write_text(json.dumps(done, indent=1))
    held = json.loads((ROOT / "probes.json").read_text())
    variety.say(
        "\nSanity runs: contact battery on their OWN TRAINING episodes (the 100 held-out babble_v2 "
        "episodes, every one seen in training; 5 episode-grouped folds) against unseen variety "
        "episodes. Change from each run's own step 0, mean +- SE over seeds.\n"
    )
    variety.say("| variant | train eps, post | unseen, post | train eps, pre | unseen, pre |")
    variety.say("|---|---|---|---|---|")
    for variant in VARIANTS:
        runs = sorted({n.rsplit("-n", 1)[0] for n in done if n.startswith(f"{variant}-s")})
        cols: dict[str, list[float]] = {"tp": [], "hp": [], "tq": [], "hq": []}
        for r in runs:
            steps = sorted(int(n.rsplit("-n", 1)[1]) for n in done if n.startswith(f"{r}-n"))
            a, b = f"{r}-n{steps[0]}", f"{r}-n{steps[-1]}"
            if len(steps) < 2 or b not in held:
                continue
            cols["tp"].append(done[b]["post"] - done[a]["post"])
            cols["tq"].append(done[b]["pre"] - done[a]["pre"])
            cols["hp"].append(held[b]["post"]["mean"] - held[a]["post"]["mean"])
            cols["hq"].append(held[b]["pre"]["mean"] - held[a]["pre"]["mean"])
        if cols["tp"]:
            c = {k: diag.cell(v) for k, v in cols.items()}
            variety.say(f"| {variant} | {c['tp']} | {c['hp']} | {c['tq']} | {c['hq']} |")


def table(done: dict[str, dict[str, dict[str, float]]]) -> None:
    """Change of the contact-battery mean from each run's own step 0, per variant."""
    variety.say("| variant | seeds | step 0 post | end post | end - 0 post | end - 0 pre |")
    variety.say("|---|---|---|---|---|---|")
    for variant in VARIANTS:
        runs = sorted({n.rsplit("-n", 1)[0] for n in done if n.startswith(f"{variant}-s")})
        rows = []
        for r in runs:
            steps = sorted(int(n.rsplit("-n", 1)[1]) for n in done if n.startswith(f"{r}-n"))
            if len(steps) < 2:
                continue
            a, b = done[f"{r}-n{steps[0]}"], done[f"{r}-n{steps[-1]}"]
            rows.append(
                (
                    a["post"]["mean"],
                    b["post"]["mean"],
                    b["post"]["mean"] - a["post"]["mean"],
                    b["pre"]["mean"] - a["pre"]["mean"],
                )
            )
        if not rows:
            continue
        m = np.array(rows)
        se = m.std(0, ddof=1) / np.sqrt(len(m)) if len(m) > 1 else np.zeros(4)
        variety.say(
            f"| {variant} | {len(m)} | {m[:, 0].mean():.3f} | {m[:, 1].mean():.3f} | "
            f"{m[:, 2].mean():+.3f} +- {se[2]:.3f} | {m[:, 3].mean():+.3f} +- {se[3]:.3f} |"
        )


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=("cache", "train", "probe", "trainprobe"))
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=250)
    args = parser.parse_args()
    torch.set_num_threads(2)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.command == "cache":
        build_cache(train_paths(), CACHE, variety.CACHE_SIZE, jobs=2)
    elif args.command == "train":
        if args.variant is None:
            parser.error("train needs --variant")
        run(args.variant, args.seed, args.steps)
    elif args.command == "trainprobe":
        train_probe()
    else:
        probe()


if __name__ == "__main__":
    main()
