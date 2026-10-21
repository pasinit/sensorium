"""B-variety-0: at equal volume, does more varied pretraining data help the V+P encoder?

Two data sets of the same layout (600-step seated episodes, half Shadow hand, half pincer, hidden
properties and effort recorded): ``configs/variety.toml`` (the seven experiment programs plus
babbling with props) and ``configs/babble_v2.toml`` (babbling only, as ``scale_test.toml`` on the
current physics). ``split`` draws one split per set with ``scale_test.make_split`` (stratified by
block): 100 held-out episodes, then the pretraining set, the first n of the rest in stratified
order, with n the same for both sets: the smaller set's count of non-held-out episodes, fixed
here before any training (generation is time-boxed, so blocks may be partly filled).

Arms (V+P, as B-scale-0: 10,000 steps x 16 sequences; checkpoints at 0, 250, 1,000, 2,500, 5,000
and 10,000; seeds 0 to 2): ``jepa-variety``, ``jepa-babble``, ``mae-variety``, ``mae-babble``, and
the supervised ceilings ``sup-variety`` and ``sup-babble`` (1,000 steps). Every arm shares one
channel space and normalisation, fitted on 50 + 50 pretraining episodes of the two sets, without the
``eff_`` proprio channels (babbling and experiment episodes then give the model the same input). So
the same seed gives the same initial encoder in every arm and for the floor. Every encoder and bar
(random-init floor, 5 seeds; raw proprio K = 4) is probed on both held-out sets.

The SSL arms read their episodes from a 128 px frame cache per set (opt-in, as B-scale-0;
``--no-cache`` decodes). Probes always decode the held-out episodes at full resolution.

    uv run python scripts/variety_test.py split [--from-config] [--dry-run]
    uv run python scripts/variety_test.py cache --jobs 30   # both sets' pretraining episodes
    uv run python scripts/variety_test.py train --arm mae-variety --seed 0
    uv run python scripts/variety_test.py probe [--shard 0/16]
    uv run python scripts/variety_test.py report

Pre-registration and outcome rules: ``docs/model.md``, "B-variety-0".
"""

import argparse
import functools
import json
import logging
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import scale_test as scale  # scripts/ is on sys.path when this file runs as a script
import torch

from sensorium.episode import Episode
from sensorium.model.controls import SupervisedConfig, build_probe_data, train_supervised
from sensorium.model.data import LoaderConfig, build, load_episode
from sensorium.model.diagnostics import heldout_loss, state_geometry
from sensorium.model.frame_cache import build_cache
from sensorium.model.layout import ChannelSpace
from sensorium.model.paired import Contrast, contrast, describe
from sensorium.model.pretrain import (
    RunConfig,
    TrainConfig,
    build_space,
    load_config,
    load_layer,
    train,
)
from sensorium.model.probes import (
    Predictions,
    ProbeData,
    collect,
    predict_hidden,
    predict_probes,
)
from sensorium.model.sensory_layer import SensoryLayer, SensoryModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SETS = ("babble", "variety")
"""The two data sets; each is also a held-out test set. ``babble`` is the primary test set."""
DIRS = {"babble": "babble_v2", "variety": "variety"}
DATA = Path("data")
CACHE = Path("data")
ROOT = Path("runs/variety")
BASE = scale.BASE
CACHE_SIZE = scale.CACHE_SIZE
HELDOUT = scale.HELDOUT
PRETRAIN = 0
"""Pretraining episodes per set; 0 = the matched available count (``matched_size``)."""
SPACE_PER_SET = 50
DROP = ("eff_",)
STEPS = scale.STEPS
SUP_STEPS = scale.SUP_STEPS
CKPTS = scale.CKPTS
SEEDS = scale.SEEDS
FLOOR_SEEDS = scale.FLOOR_SEEDS
STRIDE = scale.STRIDE
NO_TOUCH = scale.NO_TOUCH
PROPRIO = scale.PROPRIO
ARMS: dict[str, tuple[str, dict[str, object]]] = {
    "jepa-babble": ("babble", {}),
    "jepa-variety": ("variety", {}),
    "mae-babble": ("babble", {"objective": "mae"}),
    "mae-variety": ("variety", {"objective": "mae"}),
}
SUPERVISED = {"sup-babble": "babble", "sup-variety": "variety"}
COLORS = {
    "jepa-babble": "#86b6ef",
    "jepa-variety": "#104281",
    "mae-babble": "#f5a37f",
    "mae-variety": "#b8410f",
    "sup-babble": "#7fd6b4",
    "sup-variety": "#0f7a52",
}
HIDDEN_TERMS = ("hidden_log_mass_r2", "hidden_friction_r2", "hidden_softness_r2")


def config_of(name: str) -> Path:
    """The generator config of a data set."""
    return Path("configs") / f"{DIRS[name]}.toml"


def data_dir(name: str) -> Path:
    """Where a data set's episodes are (``<data>/<out_dir>/humanoid``)."""
    return DATA / DIRS[name] / "humanoid"


def cache_dir(name: str) -> Path:
    """A data set's 128 px frame cache (one per set: episode stems may repeat across sets)."""
    return CACHE / f"{DIRS[name]}_cache{CACHE_SIZE}"


# ---------------------------------------------------------------------------------------------
# split


def program_of(block: str) -> str:
    """A block's program: its name without the effector prefix and the trailing index
    (``hand_lift_drop_00`` -> ``lift_drop``)."""
    parts = block.split("_")[1:]
    return "_".join(parts[:-1] if len(parts) > 1 and parts[-1].isdigit() else parts)


def summary(split: dict[str, list[str]]) -> str:
    """Held-out counts by effector and by program, and the number of blocks each list covers."""
    held = Counter(scale.block_of(n).split("_")[0] for n in split["heldout"])
    programs = Counter(program_of(scale.block_of(n)) for n in split["heldout"])
    blocks = {k: len({scale.block_of(n) for n in v}) for k, v in split.items()}
    return (
        f"{len(split['heldout'])} held-out {dict(held)} {dict(sorted(programs.items()))}, "
        f"{len(split['pretrain'])} pretraining, "
        f"{len(split['spare'])} spare; blocks covered {blocks}"
    )


def make_splits(names: dict[str, list[str]]) -> dict[str, dict[str, list[str]]]:
    """One ``scale_test.make_split`` per set (seed 0, ``HELDOUT`` held-out), both with the same
    pretraining size: ``PRETRAIN`` if set, else ``matched_size``.

    Raises:
        ValueError: if a set has no more than ``HELDOUT`` episodes.
    """
    n = PRETRAIN or matched_size(names)
    return {s: scale.make_split(names[s], heldout=HELDOUT, pretrain=n) for s in SETS}


def matched_size(names: dict[str, list[str]]) -> int:
    """The pretraining size of every arm: the smaller set's count of non-held-out episodes.

    Raises:
        ValueError: if a set has no more than ``HELDOUT`` episodes.
    """
    n = min(len(names[s]) for s in SETS) - HELDOUT
    if n <= 0:
        raise ValueError(
            f"need more than {HELDOUT} episodes per set, got {[len(v) for v in names.values()]}"
        )
    return n


def pretrain_size() -> int:
    """The pretraining size written in the split (every arm's episode count)."""
    return len(json.loads((ROOT / "split.json").read_text())[SETS[0]]["pretrain"])


def write_split(from_config: bool, dry_run: bool) -> None:
    """Draw the splits from the generated files (or, for a preview, from the configs' names)."""
    names = {}
    for s in SETS:
        expected = scale.expected_names(config_of(s)) if config_of(s).exists() else None
        if from_config:
            if expected is None:
                raise FileNotFoundError(f"{config_of(s)} does not exist yet")
            names[s] = expected
            continue
        names[s] = sorted(p.name for p in data_dir(s).glob("humanoid-*.npz"))
        if expected is not None and names[s] != expected:
            logging.warning(
                "%s: %d episodes on disk, the config names %d", s, len(names[s]), len(expected)
            )
    splits = make_splits(names)
    logging.info("pretraining episodes per set: %d (matched)", len(splits[SETS[0]]["pretrain"]))
    for s in SETS:
        logging.info("%s: %s", s, summary(splits[s]))
    if not dry_run:
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / "split.json").write_text(json.dumps(splits, indent=1))


def files(name: str, key: str, n: int | None = None) -> list[Path]:
    """Episode files of a set's split (``pretrain`` truncated to its first ``n``)."""
    names = json.loads((ROOT / "split.json").read_text())[name][key]
    return [data_dir(name) / n_ for n_ in names[:n]]


def space_files() -> list[Path]:
    """The episodes every arm fits its channel space and normalisation on: the first 50
    pretraining episodes of each set, interleaved (so the 64 used for statistics mix both)."""
    pairs = zip(*(files(s, "pretrain", SPACE_PER_SET) for s in SETS), strict=True)
    return [p for pair in pairs for p in pair]


@functools.cache
def shared_space() -> ChannelSpace:
    """The channel space of every arm (no ``eff_`` channels), with its normalisation."""
    return build_space(space_files(), load_config(BASE).data.stats_episodes, DROP)


# ---------------------------------------------------------------------------------------------
# train


def write_cache(jobs: int, names: tuple[str, ...]) -> None:
    """The 128 px frame caches of the pretraining episodes of each set in ``names``."""
    for s in names:
        build_cache(files(s, "pretrain"), cache_dir(s), CACHE_SIZE, jobs)


def train_arm(arm: str, seed: int, steps: int, workers: int, cache: bool = True) -> None:
    """Pretrain one arm into ``runs/variety/runs/<arm>-s<seed>`` (from its set's frame cache
    unless ``cache`` is false), or a supervised ceiling (always decodes: it reads labels)."""
    base = load_config(BASE)
    out = ROOT / "runs"
    if arm in SUPERVISED:
        train_supervised(
            SupervisedConfig(
                model=replace(base.model, senses=NO_TOUCH),
                steps=steps or SUP_STEPS,
                seed=seed,
                workers=workers,
            ),
            files(SUPERVISED[arm], "pretrain"),
            shared_space(),
            out / f"{arm}-s{seed}",
            logging.info,
        )
        return
    name, extra = ARMS[arm]
    steps = steps or STEPS
    tc = TrainConfig(
        steps=steps,
        batch_size=16,
        lr=base.train.lr,
        log_every=50,
        eval_windows=0,
        num_workers=workers,
        out_dir=str(out),
        save_steps=tuple(s for s in CKPTS if s <= steps),
    )
    data = replace(
        base.data,
        paths=tuple(str(p) for p in files(name, "pretrain")),
        space_paths=tuple(str(p) for p in space_files()),
        drop_prefix=DROP,
    )
    model = replace(base.model, senses=NO_TOUCH, **extra)  # type: ignore[arg-type]
    augment = replace(
        base.augment, cache_episodes=4, frame_cache=str(cache_dir(name)) if cache else ""
    )
    train(RunConfig(f"{arm}-s{seed}", seed, data, augment, model, tc))


# ---------------------------------------------------------------------------------------------
# probe


def floor(seed: int) -> SensoryLayer:
    """The V+P encoder at random initialisation, in the arms' shared channel space."""
    model = replace(load_config(BASE).model, senses=NO_TOUCH)
    torch.manual_seed(seed)
    space = shared_space()
    loader = LoaderConfig(steps=model.steps, substeps=model.substeps, view_size=model.vision_size)
    return SensoryLayer(SensoryModel(model, space), space, loader)


def raw_features(episode: Episode, ends: np.ndarray) -> dict[str, np.ndarray]:
    """``{"p4": ...}``: the windows' own proprio (no ``eff_``), the K = 4 observations stacked."""
    space = shared_space()
    k = load_config(BASE).model.substeps
    arrays = build(load_episode(episode, space, PROPRIO), space, PROPRIO, ends, k)
    return {"p4": np.concatenate([arrays[g].reshape(len(ends), -1) for g in PROPRIO], 1)}


def raw_proprio_data(test: str) -> ProbeData:
    """The raw proprio features (``raw_features``) and labels of a test set."""
    k = load_config(BASE).model.substeps
    return build_probe_data(files(test, "heldout"), k, STRIDE, raw_features)


def pred_path(test: str, name: str) -> Path:
    """Out-of-fold predictions of one encoder on one test set."""
    return ROOT / "preds" / test / f"{name}.npz"


def probe_path(test: str, name: str) -> Path:
    """Scores, hidden-property scores, geometry and held-out loss of one encoder on one test set."""
    return ROOT / "probes" / test / f"{name}.json"


def save_predictions(test: str, name: str, pred: Predictions) -> None:
    """Out-of-fold predictions for paired bootstraps."""
    path = pred_path(test, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    edges = np.array([[b.start, b.stop] for b in pred.blocks])
    np.savez(path, pred=pred.pred, target=pred.target, groups=pred.groups, edges=edges)


@functools.cache
def load_predictions(test: str, name: str) -> Predictions | None:
    """Inverse of ``save_predictions``; ``None`` if not probed (memoised)."""
    path = pred_path(test, name)
    if not path.exists():
        return None
    raw = np.load(path)
    blocks = [slice(int(a), int(b)) for a, b in raw["edges"]]
    return Predictions(raw["pred"], raw["target"], blocks, raw["groups"])


def write_probe(
    test: str, name: str, data: ProbeData, feature: str, extra: dict[str, object]
) -> None:
    """Contact-battery predictions and scores plus the (exploratory) hidden-property scores."""
    pred = predict_probes(data, feature)
    save_predictions(test, name, pred)
    out = {"heldout": pred.scores(), "hidden": predict_hidden(data, feature)} | extra
    path = probe_path(test, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    logging.info("probed %s on %s: %.3f", name, test, out["heldout"]["mean"])  # type: ignore[index]


def probe_layer(test: str, name: str, layer: SensoryLayer, loss: bool) -> None:
    """Probe one encoder on one test set (``loss``: also its own objective's held-out loss)."""
    data = collect(layer, files(test, "heldout"), STRIDE, (), raw=False)
    extra: dict[str, object] = {"geometry": state_geometry(data.features["state"])}
    if loss:
        extra |= heldout_loss(layer, files(test, "heldout"))
    write_probe(test, name, data, "state", extra)


def encoders() -> list[str]:
    """Every encoder to probe, by output name (floors, raw proprio, every checkpoint present)."""
    names = [f"floor-vp-s{s}" for s in FLOOR_SEEDS] + ["raw-proprio"]
    for run in sorted((ROOT / "runs").glob("*-s?")):
        if run.name.rsplit("-s", 1)[0] in SUPERVISED:
            if (run / "ckpt.pt").exists():
                names.append(f"{run.name}-n{SUP_STEPS}")
            continue
        names += [f"{run.name}-n{s}" for s in CKPTS if (run / f"ckpt-{s}.pt").exists()]
    return names


def probe_jobs() -> list[tuple[str, str]]:
    """Every (test set, encoder) pair."""
    return [(test, name) for name in encoders() for test in SETS]


def probe_all(shard: int, shards: int) -> None:
    """Run this shard's probe jobs (``i % shards == shard``) that have no predictions yet."""
    for i, (test, name) in enumerate(probe_jobs()):
        if i % shards != shard or pred_path(test, name).exists():
            continue
        if name == "raw-proprio":
            write_probe(test, name, raw_proprio_data(test), "p4", {})
        elif name.startswith("floor-vp-s"):
            probe_layer(test, name, floor(int(name.removeprefix("floor-vp-s"))), loss=True)
        elif name.rsplit("-s", 1)[0] in SUPERVISED:
            run = name.rsplit("-n", 1)[0]
            probe_layer(test, name, load_layer(ROOT / "runs" / run / "ckpt.pt"), loss=False)
        else:
            run, step = name.rsplit("-n", 1)
            probe_layer(test, name, load_layer(ROOT / "runs" / run / f"ckpt-{step}.pt"), loss=True)


# ---------------------------------------------------------------------------------------------
# report


def family(test: str, arm: str, step: int) -> list[Predictions]:
    """Predictions of every probed seed of an arm at a checkpoint on a test set, in seed order."""
    preds = [load_predictions(test, f"{arm}-s{s}-n{step}") for s in SEEDS]
    return [p for p in preds if p is not None]


def floors(test: str) -> list[Predictions]:
    """Predictions of the random-init floors on a test set."""
    preds = [load_predictions(test, f"floor-vp-s{s}") for s in FLOOR_SEEDS]
    return [p for p in preds if p is not None]


def raw(test: str) -> list[Predictions]:
    """The raw-proprio readout on a test set (empty if not probed)."""
    pred = load_predictions(test, "raw-proprio")
    return [pred] if pred is not None else []


def paired_at(test: str, a: str, b: str, step: int, step_b: int | None = None) -> Contrast | None:
    """``a`` at ``step`` minus ``b`` at ``step_b`` (default ``step``) on a test set, paired by
    seed (the seeds both have) and by episode."""
    step_b = step if step_b is None else step_b
    pa = {s: load_predictions(test, f"{a}-s{s}-n{step}") for s in SEEDS}
    pb = {s: load_predictions(test, f"{b}-s{s}-n{step_b}") for s in SEEDS}
    both = [s for s in SEEDS if pa[s] is not None and pb[s] is not None]
    if len(both) < 2:
        return None
    return contrast([pa[s] for s in both], [pb[s] for s in both], paired=True)  # type: ignore[misc]


def against(test: str, arm: str, step: int, bar: list[Predictions]) -> Contrast | None:
    """An arm minus a bar (independent families, paired by episode)."""
    preds = family(test, arm, step)
    return contrast(preds, bar, paired=False) if preds and bar else None


def verdict(c: Contrast | None) -> str:
    """The pre-registered reading of one ``variety - babble`` contrast (``docs/model.md``)."""
    if c is None:
        return "incomplete"
    if not c.resolved:
        return "no effect"
    return "variety helps" if c.point > 0 else "variety harms"


def reading(on_babble: str, on_variety: str) -> str:
    """The pre-registered joint reading of the primary (babble test) and the secondary (variety
    test) verdicts of one objective."""
    if "incomplete" in (on_babble, on_variety):
        return "incomplete"
    pair = {on_babble, on_variety}
    if pair == {"variety helps", "variety harms"}:
        return "mixed"
    if on_babble == "variety harms" or (on_babble == "no effect" and on_variety == "variety harms"):
        return "variety harms"
    if on_babble == on_variety == "variety helps":
        return "variety helps on both test sets"
    if on_babble == "variety helps":
        return "variety helps on babbling only"
    if on_variety == "variety helps":
        return "variety helps on its own distribution only"
    return "no effect"


def level(c: Contrast | None) -> str:
    """Above, at or below a bar (resolved or not)."""
    if c is None:
        return "incomplete"
    if not c.resolved:
        return "at"
    return "above" if c.point > 0 else "below"


def say(text: str) -> None:
    """One line of the report (standard output)."""
    print(text)  # noqa: T201


def probe_json(test: str, name: str) -> dict[str, object]:
    """A probe JSON (empty if missing)."""
    path = probe_path(test, name)
    return json.loads(path.read_text()) if path.exists() else {}


def train_loss(run: str, step: int, key: str = "pred") -> float:
    """Mean training ``key`` over the logged steps in ``(step - 250, step]`` of a run."""
    log = ROOT / "runs" / run / "log.jsonl"
    if not log.exists() or step == 0:
        return float("nan")
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    vals = [r[key] for r in rows if key in r and step - 250 < r["step"] <= step]
    return float(np.mean(vals)) if vals else float("nan")


def hidden_cell(test: str, names: list[str], term: str) -> str:
    """``mean +- sd (n)`` of one hidden-property R^2 over encoders (NaN scores left out)."""
    vals = []
    for name in names:
        hidden = probe_json(test, name).get("hidden", {})
        value = hidden.get(term, float("nan")) if isinstance(hidden, dict) else float("nan")
        if np.isfinite(value):
            vals.append(float(value))
    if not vals:
        return "-"
    sd = f" +- {np.std(vals, ddof=1):.2f}" if len(vals) > 1 else ""
    return f"{np.mean(vals):.2f}{sd} ({len(vals)})"


def bar_names() -> dict[str, list[str]]:
    """The bars' encoder names: floor seeds, raw proprio, supervised seeds."""
    out = {
        "floor V+P (random init)": [f"floor-vp-s{s}" for s in FLOOR_SEEDS],
        "raw proprio K = 4, linear (no eff_)": ["raw-proprio"],
    }
    for arm in SUPERVISED:
        out[f"{arm} (supervised V+P ceiling)"] = [f"{arm}-s{s}-n{SUP_STEPS}" for s in SEEDS]
    return out


def tables(test: str) -> None:
    """Probe curves, bars, losses and geometry on one test set."""
    say(f"\n## Test set: {test} ({HELDOUT} held-out {DIRS[test]} episodes)\n")
    say("Held-out probe mean (mean +- sd over seeds (n))\n")
    say("| arm | " + " | ".join(f"{s:,}" for s in CKPTS) + " |")
    say("|---|" + "---|" * len(CKPTS))
    for arm in ARMS:
        say(f"| {arm} | " + " | ".join(scale.cell(family(test, arm, s)) for s in CKPTS) + " |")
    say("\n| bar | probe mean |\n|---|---|")
    say(f"| floor V+P (random init) | {scale.cell(floors(test))} |")
    say(f"| raw proprio K = 4, linear (no eff_) | {scale.cell(raw(test))} |")
    for arm in SUPERVISED:
        say(f"| {arm} (supervised V+P ceiling) | {scale.cell(family(test, arm, SUP_STEPS))} |")
    say("\nHeld-out / training prediction loss, effective rank (seed means)\n")
    say("| arm | " + " | ".join(f"{s:,}" for s in CKPTS) + " |")
    say("|---|" + "---|" * len(CKPTS))
    for arm in ARMS:
        cells = []
        for s in CKPTS:
            runs = [f"{arm}-s{seed}" for seed in SEEDS]
            js = [probe_json(test, f"{r}-n{s}") for r in runs]
            held = [j["heldout_pred"] for j in js if "heldout_pred" in j]
            rank = [j["geometry"]["erank"] for j in js if "geometry" in j]  # type: ignore[index]
            tr = [v for v in (train_loss(r, s) for r in runs) if np.isfinite(v)]
            cells.append(f"{scale.num(held, 3)} / {scale.num(tr, 3)}, rank {scale.num(rank, 1)}")
        say(f"| {arm} | " + " | ".join(cells) + " |")


def contrasts(test: str) -> dict[str, Contrast | None]:
    """Every contrast on one test set; returns the two ``variety - babble`` contrasts at 10,000."""
    say(f"\nvariety - babble on {test}, every checkpoint (paired by seed and episode)\n")
    say("| objective | " + " | ".join(f"{s:,}" for s in CKPTS[1:]) + " |")
    say("|---|" + "---|" * (len(CKPTS) - 1))
    last: dict[str, Contrast | None] = {}
    for obj in ("mae", "jepa"):
        cs = [paired_at(test, f"{obj}-variety", f"{obj}-babble", s) for s in CKPTS[1:]]
        last[obj] = cs[-1]
        say(f"| {obj} | " + " | ".join(describe(c) if c else "-" for c in cs) + " |")
    sup = paired_at(test, "sup-variety", "sup-babble", SUP_STEPS)
    say(f"\nsup-variety - sup-babble: {describe(sup) if sup else '-'}")
    say(f"\nmae - jepa at {STEPS:,} on {test} (paired by seed and episode)\n")
    say("| data | contrast |\n|---|---|")
    for name in ("babble", "variety"):
        c = paired_at(test, f"mae-{name}", f"jepa-{name}", STEPS)
        say(f"| {name} | {describe(c) if c else '-'} |")
    say(f"\nChange from the same seed's step 0 on {test} (paired by seed and episode)\n")
    say("| arm | " + " | ".join(f"{s:,}" for s in CKPTS[1:]) + " |")
    say("|---|" + "---|" * (len(CKPTS) - 1))
    for arm in ARMS:
        cs = [paired_at(test, arm, arm, s, 0) for s in CKPTS[1:]]
        say(f"| {arm} | " + " | ".join(describe(c) if c else "-" for c in cs) + " |")
    say(f"\nAgainst the bars on {test} (independent families, paired by episode)\n")
    say("| arm | step | minus floor | minus raw proprio |\n|---|---|---|---|")
    rows = [(arm, STEPS) for arm in ARMS] + [(arm, SUP_STEPS) for arm in SUPERVISED]
    for arm, s in rows:
        f, r = against(test, arm, s, floors(test)), against(test, arm, s, raw(test))
        if f or r:
            say(f"| {arm} | {s:,} | {describe(f) if f else '-'} | {describe(r) if r else '-'} |")
    if raw(test) and floors(test):
        say(f"\nraw proprio - floor: {describe(contrast(raw(test), floors(test), paired=False))}")
    say(f"\nHidden-property probes on {test} (EXPLORATORY; contact-window R^2, mean +- sd (n))\n")
    say("| encoder | step | " + " | ".join(HIDDEN_TERMS) + " |")
    say("|---|---|" + "---|" * len(HIDDEN_TERMS))
    hidden_rows = [(arm, f"{STEPS:,}", [f"{arm}-s{s}-n{STEPS}" for s in SEEDS]) for arm in ARMS]
    hidden_rows += [(bar, "-", names) for bar, names in bar_names().items()]
    for label, s, names in hidden_rows:
        cells = " | ".join(hidden_cell(test, names, t) for t in HIDDEN_TERMS)
        say(f"| {label} | {s} | {cells} |")
    return last


def plot_probe(test: str, out: Path) -> None:
    """Probe mean against steps per arm (seed SE bars) on one test set, with the bars."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for k, arm in enumerate(ARMS):
        pts = [(s, [p.scores()["mean"] for p in family(test, arm, s)]) for s in CKPTS]
        pts = [(s, v) for s, v in pts if v]
        if not pts:
            continue
        x = [max(s, 60) * (1 + 0.04 * (k - 1.5)) for s, _ in pts]  # step 0 drawn at 60 (log axis)
        y = [np.mean(v) for _, v in pts]
        err = [np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0 for _, v in pts]
        ax.errorbar(x, y, err, marker="o", ms=5, lw=2, capsize=3, color=COLORS[arm], label=arm)
    bars = [
        ("floor (random init)", floors(test), "#888888", "--"),
        ("raw proprio K=4", raw(test), "#444444", ":"),
    ] + [(f"{arm} ceiling", family(test, arm, SUP_STEPS), COLORS[arm], "-.") for arm in SUPERVISED]
    for label, preds, color, style in bars:
        if preds:
            ax.axhline(scale.seed_mean(preds), color=color, ls=style, lw=1.2, label=label)
    ax.set(xscale="log", xlabel="pretraining step (0 drawn at 60)", ylabel="held-out probe mean")
    ax.set_xticks([60, 250, 1000, 2500, 5000, 10000])
    ax.set_xticklabels(["0", "250", "1k", "2.5k", "5k", "10k"])
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(
        f"B-variety-0 on {HELDOUT} held-out {DIRS[test]} episodes, 3 seeds (bars: seed SE)"
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    logging.info("wrote %s", out)


def plot_curves(test: str, out: Path) -> None:
    """Loss curves per seed per arm, held-out loss and probe on one test set (``plot_runs.py``)."""
    script = Path(__file__).with_name("plot_runs.py")
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(script),
            "--runs",
            str(ROOT / "runs"),
            "--probes",
            str(ROOT / "probes" / test),
            "--metric",
            "heldout.mean",
            *[a for arm in ARMS for a in ("--floor", f"{arm}=floor-vp")],
            "--arms",
            *ARMS,
            "--title",
            f"B-variety-0: V+P on {pretrain_size():,} variety or babble_v2 episodes; "
            f"probe on {DIRS[test]}",
            "--out",
            str(out),
        ],
        check=True,
    )


def report() -> None:
    """Every table, the verdicts and the figures (``runs/variety/figures``)."""
    last = {}
    for test in SETS:
        tables(test)
        last[test] = contrasts(test)
    say(f"\nPretraining episodes per set (every arm): {pretrain_size():,}")
    say(f"\n## Pre-registered outcomes ({STEPS:,} steps; docs/model.md, B-variety-0)\n")
    say("| objective | variety - babble on babble (primary) | on variety (secondary) | reading |")
    say("|---|---|---|---|")
    for obj in ("mae", "jepa"):
        cb, cv = last["babble"][obj], last["variety"][obj]
        vb, vv = verdict(cb), verdict(cv)
        say(
            f"| {obj} | {describe(cb) if cb else '-'}: **{vb}** | "
            f"{describe(cv) if cv else '-'}: {vv} | **{reading(vb, vv)}** |"
        )
    say("\n| arm | test | minus floor | reading |\n|---|---|---|---|")
    for arm in ("jepa-babble", "jepa-variety"):
        for test in SETS:
            c = against(test, arm, STEPS, floors(test))
            say(f"| {arm} | {test} | {describe(c) if c else '-'} | {level(c)} the floor |")
    for test in SETS:
        plot_probe(test, ROOT / "figures" / f"variety_probe_{test}.png")
        if any((ROOT / "runs").glob("*/log.jsonl")):
            plot_curves(test, ROOT / "figures" / f"variety_curves_{test}.png")


def main() -> None:
    """Command-line entry point."""
    global DATA, ROOT, CACHE  # noqa: PLW0603
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=("split", "cache", "train", "probe", "report"))
    parser.add_argument("--arm", choices=(*ARMS, *SUPERVISED))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0, help="default 10,000 (1,000 supervised)")
    parser.add_argument("--workers", type=int, default=1, help="train: loader processes")
    parser.add_argument("--jobs", type=int, default=8, help="cache: decoding processes")
    parser.add_argument("--set", choices=SETS, action="append", help="cache: only this set")
    parser.add_argument("--no-cache", action="store_true", help="train: decode every episode")
    parser.add_argument("--shard", default="0/1", help="probe: i/n, this process's share")
    parser.add_argument("--from-config", action="store_true", help="split: names from the configs")
    parser.add_argument("--dry-run", action="store_true", help="split: print, do not write")
    parser.add_argument("--data", type=Path, default=DATA, help="parent of variety/, babble_v2/")
    parser.add_argument("--cache", type=Path, default=CACHE, help="parent of the frame caches")
    parser.add_argument("--root", type=Path, default=ROOT, help="smoke tests only")
    args = parser.parse_args()
    DATA, ROOT, CACHE = args.data, args.root, args.cache
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.command == "split":
        write_split(args.from_config, args.dry_run)
    elif args.command == "cache":
        write_cache(args.jobs, tuple(args.set or SETS))
    elif args.command == "train":
        if args.arm is None:
            parser.error("train needs --arm")
        train_arm(args.arm, args.seed, args.steps, args.workers, not args.no_cache)
    elif args.command == "probe":
        shard, shards = (int(x) for x in args.shard.split("/"))
        probe_all(shard, shards)
    else:
        report()


if __name__ == "__main__":
    main()
