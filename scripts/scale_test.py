"""B-scale-0: does more pretraining data help the V+P JEPA?

Data: ``configs/scale_test.toml``, 1,110 seated babbling episodes (master seed 2, disjoint from
``probe_test``) in 30 blocks of 37 (15 Shadow-hand, 15 pincer). ``split`` draws once, stratified by
block (``make_split``): 100 held-out episodes (3 or 4 per block, 50 hand and 50 pincer), and an
ordered pretraining list whose first 100, 300 and 1,000 episodes are the nested subsets.

Arms (V+P, pooled SIGReg as ``long-jepa``; 10,000 steps x 16 sequences; checkpoints at 0, 250,
1,000, 2,500, 5,000 and 10,000; seeds 0 to 2): ``jepa-e100``, ``jepa-e300``, ``jepa-e1000``, and
``mae-e1000`` (masked reconstruction). Every arm shares one channel space and normalisation (fitted
on the 100-episode subset), so the same seed gives the same initial encoder at every data size.
Bars on the same 100 held-out episodes: the random-init floor (5 seeds), the raw proprio K = 4
linear readout, and the supervised V+P ceiling on the 1,000 episodes (``sup-e1000``, 1,000 steps).

The JEPA and MAE arms read their episodes from a frame cache (``sensorium.model.frame_cache``,
frames stored at 128 px, the random crop placed in the 256 px grid), which removes the video
decoding that otherwise costs about 93% of a step's CPU. Every arm uses it, so the contrasts are
unaffected; probes and held-out loss decode the held-out episodes at full resolution as always.

    uv run python scripts/scale_test.py split
    uv run python scripts/scale_test.py cache --jobs 30     # 128 px frame cache of the 1,000
    uv run python scripts/scale_test.py train --arm jepa-e1000 --seed 0
    uv run python scripts/scale_test.py probe [--shard 0/6]
    uv run python scripts/scale_test.py report

Pre-registration and outcome rules: ``docs/model.md``, "B-scale-0".
"""

import argparse
import functools
import json
import logging
import subprocess
import sys
import tomllib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import torch

from sensorium.episode import Episode
from sensorium.model.controls import SupervisedConfig, build_probe_data, train_supervised
from sensorium.model.data import LoaderConfig, build, load_episode
from sensorium.model.diagnostics import heldout_loss, state_geometry
from sensorium.model.frame_cache import build_cache
from sensorium.model.paired import Contrast, contrast, describe
from sensorium.model.pretrain import (
    RunConfig,
    TrainConfig,
    build_space,
    load_config,
    load_layer,
    train,
)
from sensorium.model.probes import Predictions, collect, predict_probes
from sensorium.model.sensory_layer import SensoryLayer, SensoryModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CONFIG = Path("configs/scale_test.toml")
BASE = Path("configs/model/cpu_tiny.toml")
DATA = Path("data/scale_test/humanoid")
CACHE = Path("data/scale_test_cache128")
CACHE_SIZE = 128
ROOT = Path("runs/scale")
FIGURES = Path(".tmp/phd-b")
SPLIT_SEED = 0
HELDOUT = 100
SIZES = (100, 300, 1000)
STEPS = 10000
SUP_STEPS = 1000
CKPTS = (0, 250, 1000, 2500, 5000, 10000)
SEEDS = (0, 1, 2)
FLOOR_SEEDS = (0, 1, 2, 3, 4)
STRIDE = 8
NO_TOUCH = ("vision", "proprio_body", "vestibular", "proprio_effector")
PROPRIO = ("proprio_body", "vestibular", "proprio_effector")
ARMS: dict[str, tuple[int, dict[str, object]]] = {
    "jepa-e100": (100, {}),
    "jepa-e300": (300, {}),
    "jepa-e1000": (1000, {}),
    "mae-e1000": (1000, {"objective": "mae"}),
}
SUPERVISED = "sup-e1000"
COLORS = {
    "jepa-e100": "#86b6ef",
    "jepa-e300": "#2a78d6",
    "jepa-e1000": "#104281",
    "mae-e1000": "#eb6834",
}


# ---------------------------------------------------------------------------------------------
# split


def block_of(name: str) -> str:
    """The generator block of an episode file (``humanoid-<block>-<policy>-<index>.npz``)."""
    parts = name.split("-")
    if len(parts) != 4 or parts[0] != "humanoid":
        raise ValueError(f"not a humanoid episode file name: {name}")
    return parts[1]


def expected_names(config: Path = CONFIG) -> list[str]:
    """Episode file names the generator writes for ``config`` (before the data exists)."""
    raw = tomllib.loads(config.read_text())
    names = []
    for block, spec in raw["humanoid"].items():
        for policy in spec["policies"]:
            names += [
                f"humanoid-{block}-{policy}-{i:03d}.npz" for i in range(spec["episodes_per_policy"])
            ]
    return sorted(names)


def stratified_order(blocks: dict[str, list[str]], rng: np.random.Generator) -> list[str]:
    """All episodes of ``blocks`` in rounds: round r takes the r-th episode of every block, the
    blocks visited in a fresh random order that alternates hand and pincer blocks. Any prefix is
    therefore balanced across blocks (within one) and across effectors (within one)."""
    hand = sorted(b for b in blocks if b.startswith("hand"))
    other = sorted(b for b in blocks if not b.startswith("hand"))
    order = []
    for r in range(max(len(v) for v in blocks.values())):
        h, o = list(rng.permutation(hand)), list(rng.permutation(other))
        visit = [b for pair in zip(h, o, strict=False) for b in pair]
        visit += h[len(o) :] + o[len(h) :]
        order += [blocks[b][r] for b in visit if r < len(blocks[b])]
    return order


def make_split(
    names: Sequence[str], seed: int = SPLIT_SEED, heldout: int = HELDOUT, pretrain: int = SIZES[-1]
) -> dict[str, list[str]]:
    """Held-out episodes, the ordered pretraining list (subset n = its first n) and spares.

    Episodes are shuffled within each block, then the held-out set and the pretraining order are
    both drawn with ``stratified_order`` (held-out first, from all episodes; pretraining from the
    rest).

    Raises:
        ValueError: if there are fewer than ``heldout + pretrain`` episodes.
    """
    if len(names) < heldout + pretrain:
        raise ValueError(f"{len(names)} episodes, need {heldout + pretrain}")
    rng = np.random.default_rng(seed)
    blocks: dict[str, list[str]] = defaultdict(list)
    for name in sorted(names):
        blocks[block_of(name)].append(name)
    blocks = {b: [str(n) for n in rng.permutation(v)] for b, v in sorted(blocks.items())}
    held = stratified_order(blocks, rng)[:heldout]
    taken = set(held)
    rest = {b: [n for n in v if n not in taken] for b, v in blocks.items()}
    order = stratified_order(rest, rng)
    return {"heldout": held, "pretrain": order[:pretrain], "spare": order[pretrain:]}


def write_split(from_config: bool) -> None:
    """Draw the split from the generated files (or, for a preview, the config's names)."""
    if from_config:
        names = expected_names()
    else:
        names = sorted(p.name for p in DATA.glob("humanoid-*.npz"))
        expected = expected_names()
        if names != expected:
            logging.warning("%d episodes on disk, the config names %d", len(names), len(expected))
    split = make_split(names)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "split.json").write_text(json.dumps(split, indent=1))
    counts: dict[str, int] = defaultdict(int)
    for n in split["heldout"]:
        counts[block_of(n).split("_")[0]] += 1
    logging.info(
        "split: %d held-out (%s), %d pretraining, %d spare",
        len(split["heldout"]),
        dict(counts),
        len(split["pretrain"]),
        len(split["spare"]),
    )


def files(key: str, n: int | None = None) -> list[Path]:
    """Episode files of the split (``pretrain`` truncated to its first ``n``)."""
    names = json.loads((ROOT / "split.json").read_text())[key]
    return [DATA / name for name in names[:n]]


def space_files() -> list[Path]:
    """The episodes every arm fits its channel space and normalisation on (the 100 subset)."""
    return files("pretrain", SIZES[0])


# ---------------------------------------------------------------------------------------------
# train


def write_cache(jobs: int) -> None:
    """The 128 px frame cache of the 1,000 pretraining episodes (``CACHE``)."""
    build_cache(files("pretrain"), CACHE, CACHE_SIZE, jobs)


def train_arm(arm: str, seed: int, steps: int, workers: int, cache: bool = True) -> None:
    """Pretrain one arm into ``runs/scale/runs/<arm>-s<seed>`` (from the frame cache unless
    ``cache`` is false), or the supervised ceiling (always decodes: it reads labels)."""
    base = load_config(BASE)
    out = ROOT / "runs"
    if arm == SUPERVISED:
        space = build_space(space_files(), base.data.stats_episodes)
        train_supervised(
            SupervisedConfig(
                model=replace(base.model, senses=NO_TOUCH),
                steps=steps or SUP_STEPS,
                seed=seed,
                workers=workers,
            ),
            files("pretrain"),
            space,
            out / f"{arm}-s{seed}",
            logging.info,
        )
        return
    size, extra = ARMS[arm]
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
        paths=tuple(str(p) for p in files("pretrain", size)),
        space_paths=tuple(str(p) for p in space_files()),
    )
    model = replace(base.model, senses=NO_TOUCH, **extra)  # type: ignore[arg-type]
    augment = replace(base.augment, cache_episodes=4, frame_cache=str(CACHE) if cache else "")
    train(RunConfig(f"{arm}-s{seed}", seed, data, augment, model, tc))


# ---------------------------------------------------------------------------------------------
# probe


def floor(seed: int) -> SensoryLayer:
    """The V+P encoder at random initialisation, in the arms' shared channel space."""
    base = load_config(BASE)
    torch.manual_seed(seed)
    space = build_space(space_files(), base.data.stats_episodes)
    model = replace(base.model, senses=NO_TOUCH)
    loader = LoaderConfig(steps=model.steps, substeps=model.substeps, view_size=model.vision_size)
    return SensoryLayer(SensoryModel(model, space), space, loader)


def raw_proprio() -> Predictions:
    """Ridge probe on the window's own proprio, the K = 4 observations stacked (no encoder)."""
    base = load_config(BASE)
    space = build_space(space_files(), base.data.stats_episodes)
    k = base.model.substeps

    def features(episode: Episode, ends: np.ndarray) -> dict[str, np.ndarray]:
        arrays = build(load_episode(episode, space, PROPRIO), space, PROPRIO, ends, k)
        return {"p4": np.concatenate([arrays[g].reshape(len(ends), -1) for g in PROPRIO], 1)}

    return predict_probes(build_probe_data(files("heldout"), k, STRIDE, features), "p4")


def save_predictions(name: str, pred: Predictions) -> None:
    """Out-of-fold predictions (``runs/scale/preds/<name>.npz``) for paired bootstraps."""
    out = ROOT / "preds"
    out.mkdir(parents=True, exist_ok=True)
    edges = np.array([[b.start, b.stop] for b in pred.blocks])
    np.savez(
        out / f"{name}.npz", pred=pred.pred, target=pred.target, groups=pred.groups, edges=edges
    )


@functools.cache
def load_predictions(name: str) -> Predictions | None:
    """Inverse of ``save_predictions``; ``None`` if not probed (memoised)."""
    path = ROOT / "preds" / f"{name}.npz"
    if not path.exists():
        return None
    raw = np.load(path)
    blocks = [slice(int(a), int(b)) for a, b in raw["edges"]]
    return Predictions(raw["pred"], raw["target"], blocks, raw["groups"])


def probe_layer(name: str, layer: SensoryLayer, loss: bool) -> None:
    """Out-of-fold predictions, scores, state geometry and (``loss``) held-out loss of a layer."""
    data = collect(layer, files("heldout"), STRIDE, (), raw=False)
    pred = predict_probes(data, "state")
    save_predictions(name, pred)
    out: dict[str, object] = {
        "heldout": pred.scores(),
        "geometry": state_geometry(data.features["state"]),
    }
    if loss:
        out |= heldout_loss(layer, files("heldout"))
    (ROOT / "probes" / f"{name}.json").write_text(json.dumps(out))
    logging.info("probed %s: %.3f", name, out["heldout"]["mean"])  # type: ignore[index]


def probe_jobs() -> list[str]:
    """Every probe job, by output name (floors, raw proprio, every checkpoint present)."""
    jobs = [f"floor-vp-s{s}" for s in FLOOR_SEEDS] + ["raw-proprio"]
    for run in sorted((ROOT / "runs").glob("*-s?")):
        if run.name.startswith(SUPERVISED):
            if (run / "ckpt.pt").exists():
                jobs.append(f"{run.name}-n{SUP_STEPS}")
            continue
        jobs += [f"{run.name}-n{s}" for s in CKPTS if (run / f"ckpt-{s}.pt").exists()]
    return jobs


def probe_all(shard: int, shards: int) -> None:
    """Run this shard's probe jobs (``i % shards == shard``) that have no predictions yet."""
    (ROOT / "probes").mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(probe_jobs()):
        if i % shards != shard or (ROOT / "preds" / f"{name}.npz").exists():
            continue
        if name == "raw-proprio":
            pred = raw_proprio()
            save_predictions(name, pred)
            (ROOT / "probes" / f"{name}.json").write_text(json.dumps({"heldout": pred.scores()}))
        elif name.startswith("floor-vp-s"):
            probe_layer(name, floor(int(name.removeprefix("floor-vp-s"))), loss=True)
        elif name.startswith(SUPERVISED):
            probe_layer(
                name, load_layer(ROOT / "runs" / name.rsplit("-n", 1)[0] / "ckpt.pt"), False
            )
        else:
            run, step = name.rsplit("-n", 1)
            probe_layer(name, load_layer(ROOT / "runs" / run / f"ckpt-{step}.pt"), loss=True)


# ---------------------------------------------------------------------------------------------
# report


def family(arm: str, step: int) -> list[Predictions]:
    """Predictions of every probed seed of an arm at a checkpoint, in seed order."""
    preds = [load_predictions(f"{arm}-s{s}-n{step}") for s in SEEDS]
    return [p for p in preds if p is not None]


def floors() -> list[Predictions]:
    """Predictions of the random-init floors."""
    preds = [load_predictions(f"floor-vp-s{s}") for s in FLOOR_SEEDS]
    return [p for p in preds if p is not None]


def seed_mean(preds: list[Predictions]) -> float:
    """Mean probe score over a family (NaN if empty)."""
    return float(np.mean([p.scores()["mean"] for p in preds])) if preds else float("nan")


def best_step(a: str = "jepa-e1000", b: str = "jepa-e100") -> int | None:
    """The pre-registered "best" checkpoint: the step after 0 where the average of the two arms'
    seed-mean probe is highest (one common step, so the selection favours neither arm)."""
    means = {s: (seed_mean(family(a, s)) + seed_mean(family(b, s))) / 2 for s in CKPTS[1:]}
    finite = {s: m for s, m in means.items() if np.isfinite(m)}
    return max(finite, key=lambda s: finite[s]) if finite else None


def paired_at(a: str, b: str, step: int | None) -> Contrast | None:
    """``a - b`` at one checkpoint, paired by seed (the seeds both arms have)."""
    if step is None:
        return None
    pa = {s: load_predictions(f"{a}-s{s}-n{step}") for s in SEEDS}
    pb = {s: load_predictions(f"{b}-s{s}-n{step}") for s in SEEDS}
    both = [s for s in SEEDS if pa[s] is not None and pb[s] is not None]
    if len(both) < 2:
        return None
    return contrast([pa[s] for s in both], [pb[s] for s in both], paired=True)  # type: ignore[misc]


def verdict(best: Contrast | None, last: Contrast | None) -> str:
    """The pre-registered reading of ``jepa-e1000 - jepa-e100`` (``docs/model.md``)."""
    if best is None or last is None:
        return "incomplete"
    signs = {
        (c.point > 0) for c in (best, last) if c.resolved
    }  # True: resolved positive, False: resolved negative
    if signs == {True, False}:
        return "mixed"
    if signs == {False}:
        return "more data harms"
    if best.resolved and last.resolved:
        return "data scale helps"
    if best.resolved:
        return "helps only early"
    if last.resolved:
        return "helps only late"
    return "no effect"


def cell(preds: list[Predictions]) -> str:
    """``mean +- sd (n)`` of a family."""
    if not preds:
        return "-"
    v = [p.scores()["mean"] for p in preds]
    sd = f" +- {np.std(v, ddof=1):.2f}" if len(v) > 1 else ""
    return f"{np.mean(v):.2f}{sd} ({len(v)})"


def num(values: Sequence[object], digits: int) -> str:
    """Mean of some numbers, or ``-`` if there are none."""
    return f"{np.mean(np.asarray(values, dtype=float)):.{digits}f}" if values else "-"


def probe_json(name: str) -> dict[str, object]:
    """A probe JSON (empty if missing)."""
    path = ROOT / "probes" / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def train_loss(run: str, step: int, key: str = "pred") -> float:
    """Mean training ``key`` over the logged steps in ``(step - 250, step]`` of a run."""
    log = ROOT / "runs" / run / "log.jsonl"
    if not log.exists() or step == 0:
        return float("nan")
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    vals = [r[key] for r in rows if key in r and step - 250 < r["step"] <= step]
    return float(np.mean(vals)) if vals else float("nan")


def tables() -> tuple[int | None, Contrast | None, Contrast | None]:
    """Probe curves, bars, losses and geometry; returns the best step and the primary contrasts."""
    print("\nHeld-out probe mean (100 held-out episodes; mean +- sd over seeds (n))\n")  # noqa: T201
    print("| arm | " + " | ".join(f"{s:,}" for s in CKPTS) + " |")  # noqa: T201
    print("|---|" + "---|" * len(CKPTS))  # noqa: T201
    for arm in ARMS:
        print(f"| {arm} | " + " | ".join(cell(family(arm, s)) for s in CKPTS) + " |")  # noqa: T201
    raw = load_predictions("raw-proprio")
    print("\n| bar | probe mean |\n|---|---|")  # noqa: T201
    print(f"| floor V+P (random init) | {cell(floors())} |")  # noqa: T201
    print(f"| raw proprio K = 4, linear | {cell([raw] if raw else [])} |")  # noqa: T201
    print(f"| supervised V+P ceiling, 1,000 episodes | {cell(family(SUPERVISED, SUP_STEPS))} |")  # noqa: T201
    print("\nHeld-out / training prediction loss, effective rank (seed means)\n")  # noqa: T201
    print("| arm | " + " | ".join(f"{s:,}" for s in CKPTS) + " |")  # noqa: T201
    print("|---|" + "---|" * len(CKPTS))  # noqa: T201
    for arm in ARMS:
        cells = []
        for s in CKPTS:
            runs = [f"{arm}-s{seed}" for seed in SEEDS]
            js = [probe_json(f"{r}-n{s}") for r in runs]
            held = [j["heldout_pred"] for j in js if "heldout_pred" in j]
            rank = [j["geometry"]["erank"] for j in js if "geometry" in j]  # type: ignore[index]
            tr = [v for v in (train_loss(r, s) for r in runs) if np.isfinite(v)]
            cells.append(f"{num(held, 3)} / {num(tr, 3)}, rank {num(rank, 1)}")
        print(f"| {arm} | " + " | ".join(cells) + " |")  # noqa: T201
    step = best_step()
    best = paired_at("jepa-e1000", "jepa-e100", step)
    last = paired_at("jepa-e1000", "jepa-e100", STEPS)
    return step, best, last


def contrasts(step: int | None, best: Contrast | None, last: Contrast | None) -> None:
    """The primary contrast and its verdict, then the secondary, own-step-0 and bar contrasts."""
    print(f"\nPrimary: jepa-e1000 - jepa-e100 (paired by seed and episode); best step {step}\n")  # noqa: T201
    print(f"* at the best step: {describe(best) if best else '-'}")  # noqa: T201
    print(f"* at 10,000 steps: {describe(last) if last else '-'}")  # noqa: T201
    print(f"\nVerdict: {verdict(best, last)}")  # noqa: T201
    print("\nSecondary contrasts (paired by seed and episode)\n")  # noqa: T201
    print("| contrast | best step | 10,000 |\n|---|---|---|")  # noqa: T201
    for a, b in (
        ("jepa-e300", "jepa-e100"),
        ("jepa-e1000", "jepa-e300"),
        ("mae-e1000", "jepa-e1000"),
    ):
        c1, c2 = paired_at(a, b, step), paired_at(a, b, STEPS)
        print(f"| {a} - {b} | {describe(c1) if c1 else '-'} | {describe(c2) if c2 else '-'} |")  # noqa: T201
    print("\nChange from the same seed's step 0 (paired by seed and episode)\n")  # noqa: T201
    print("| arm | " + " | ".join(f"{s:,}" for s in CKPTS[1:]) + " |")  # noqa: T201
    print("|---|" + "---|" * (len(CKPTS) - 1))  # noqa: T201
    for arm in ARMS:
        cs = [_from_zero(arm, s) for s in CKPTS[1:]]
        print(f"| {arm} | " + " | ".join(describe(c) if c else "-" for c in cs) + " |")  # noqa: T201
    print("\nAgainst the bars (independent families, paired by episode)\n")  # noqa: T201
    print("| arm | step | minus floor | minus raw proprio |\n|---|---|---|---|")  # noqa: T201
    raw = load_predictions("raw-proprio")
    rows = [(arm, s) for arm in ARMS for s in sorted({step or STEPS, STEPS})]
    rows.append((SUPERVISED, SUP_STEPS))
    for arm, s in rows:
        preds = family(arm, s)
        if not preds:
            continue
        f = describe(contrast(preds, floors(), paired=False)) if floors() else "-"
        r = describe(contrast(preds, [raw], paired=False)) if raw else "-"
        print(f"| {arm} | {s:,} | {f} | {r} |")  # noqa: T201
    if raw and floors():
        print(f"\nraw proprio - floor: {describe(contrast([raw], floors(), paired=False))}")  # noqa: T201


def _from_zero(arm: str, step: int) -> Contrast | None:
    """``arm`` at ``step`` minus the same seeds at step 0."""
    pa = {s: load_predictions(f"{arm}-s{s}-n{step}") for s in SEEDS}
    p0 = {s: load_predictions(f"{arm}-s{s}-n0") for s in SEEDS}
    both = [s for s in SEEDS if pa[s] is not None and p0[s] is not None]
    if len(both) < 2:
        return None
    return contrast([pa[s] for s in both], [p0[s] for s in both], paired=True)  # type: ignore[misc]


def plot_curves(out: Path) -> None:
    """Probe mean against steps per arm (seed SE bars), with the floor, raw-proprio and ceiling."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for k, arm in enumerate(ARMS):
        pts = [(s, [p.scores()["mean"] for p in family(arm, s)]) for s in CKPTS]
        pts = [(s, v) for s, v in pts if v]
        if not pts:
            continue
        x = [max(s, 60) * (1 + 0.04 * (k - 1.5)) for s, _ in pts]  # step 0 drawn at 60 (log axis)
        y = [np.mean(v) for _, v in pts]
        err = [np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0 for _, v in pts]
        ax.errorbar(x, y, err, marker="o", ms=5, lw=2, capsize=3, color=COLORS[arm], label=arm)
    bars = (
        ("floor (random init)", floors(), "#888888", "--"),
        ("raw proprio K=4", [p for p in [load_predictions("raw-proprio")] if p], "#444444", ":"),
        ("supervised ceiling", family(SUPERVISED, SUP_STEPS), "#1baf7a", "-."),
    )
    for label, preds, color, style in bars:
        if preds:
            ax.axhline(seed_mean(preds), color=color, ls=style, lw=1.2, label=label)
    ax.set(xscale="log", xlabel="pretraining step (0 drawn at 60)", ylabel="held-out probe mean")
    ax.set_xticks([60, 250, 1000, 2500, 5000, 10000])
    ax.set_xticklabels(["0", "250", "1k", "2.5k", "5k", "10k"])
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("B-scale-0: 100 held-out episodes, 3 seeds (bars: seed SE)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    logging.info("wrote %s", out)


def report() -> None:
    """Every table, the verdict and the figures (``.tmp/phd-b/scale_*.png``)."""
    step, best, last = tables()
    contrasts(step, best, last)
    plot_curves(FIGURES / "scale_probe.png")
    if any((ROOT / "runs").glob("*/log.jsonl")):
        subprocess.run(  # noqa: S603
            [
                sys.executable,
                "scripts/plot_runs.py",
                "--runs",
                str(ROOT / "runs"),
                "--probes",
                str(ROOT / "probes"),
                "--metric",
                "heldout.mean",
                *[a for arm in ARMS for a in ("--floor", f"{arm}=floor-vp")],
                "--arms",
                *ARMS,
                "--title",
                "B-scale-0: V+P JEPA on 100 / 300 / 1,000 episodes, 10,000 steps x 16",
                "--out",
                str(FIGURES / "scale_curves.png"),
            ],
            check=True,
        )


def main() -> None:
    """Command-line entry point."""
    global DATA, ROOT, CACHE  # noqa: PLW0603
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=("split", "cache", "train", "probe", "report"))
    parser.add_argument("--arm", choices=(*ARMS, SUPERVISED))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0, help="default 10,000 (1,000 supervised)")
    parser.add_argument("--workers", type=int, default=1, help="train: loader processes")
    parser.add_argument("--jobs", type=int, default=8, help="cache: decoding processes")
    parser.add_argument("--no-cache", action="store_true", help="train: decode every episode")
    parser.add_argument("--cache", type=Path, default=CACHE, help="frame cache directory")
    parser.add_argument("--shard", default="0/1", help="probe: i/n, this process's share")
    parser.add_argument("--from-config", action="store_true", help="split: names from the config")
    parser.add_argument("--data", type=Path, default=DATA, help="smoke tests only")
    parser.add_argument("--root", type=Path, default=ROOT, help="smoke tests only")
    args = parser.parse_args()
    DATA, ROOT, CACHE = args.data, args.root, args.cache
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.command == "split":
        write_split(args.from_config)
    elif args.command == "cache":
        write_cache(args.jobs)
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
