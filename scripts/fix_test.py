"""B-fix-0 and B-fix-1: can reconstruction or proprio dropout stop the JEPA losing contact?

B-diag-0 found that the pooled-SIGReg JEPA keeps what state and action predict: arm kinematics,
almost no vision, no scene context. Its projector then removes contact information. Two fixes,
each on B-variety-0's recipe (babble_v2 pretraining set, n = 791; V+P; 10,000 steps x 16;
checkpoints 0, 250, 1,000, 2,500, 5,000 and 10,000; seeds 0 to 2; the shared channel space
without ``eff_``; the 128 px frame cache):

* ``recon01``, ``recon05``: ``model.recon_weight`` 0.1 and 0.5. A decoder reads the pre-projector
  state of a masked copy of the batch and reconstructs the masked pixels and proprio (the ``mae``
  objective's targets).
* ``drop01``, ``drop02``: ``model.proprio_dropout`` 0.1 and 0.2. Every proprio sense of a training
  sequence is hidden together, so vision has to carry the state.
* ``recon05drop02``: both.
* ``fix1-recon05`` (B-fix-1): reconstruction weight 0.5 in **one masked pass**
  (``model.recon_input = masked``): the batch is masked once, the encoder runs once on it, the
  next-state loss and SIGReg use its projected states and the decoder its pre-projector states.
  Encoders already probed (their predictions saved) are not encoded again, so B-fix-1 reuses
  B-fix-0's predictions of the references and pairs with them directly.

References, not retrained: B-variety-0's ``jepa-babble`` (A0, the same recipe with both options
off; the code path is unchanged when they are off) and ``mae-babble``, and the random-init floor
(seeds 0 to 2), all from ``runs/variety``.

Evaluation (every encoder, every checkpoint, before and after the projector):

* the three contact AUROCs (the battery without force and tip) and the outcome battery;
* on the 100 held-out babble_v2 episodes, on 100 of the babble_v2 pretraining episodes (seen by
  every trained encoder), and on the 100 held-out variety episodes;
* the probe is an episode-grouped 5-fold ridge.

``encode`` and ``probe`` take ``--shard i/n``.

    uv run python scripts/fix_test.py train --arm recon05 --seed 0
    uv run python scripts/fix_test.py encode --shard 0/16
    uv run python scripts/fix_test.py probe --shard 0/16
    uv run python scripts/fix_test.py report

Pre-registration and outcome rules: ``docs/model.md``, "B-fix-0".
"""

import argparse
import functools
import json
import logging
from dataclasses import replace
from pathlib import Path

import jepa_diagnostics as diag
import matplotlib
import numpy as np
import torch
import variety_test as variety  # scripts/ is on sys.path when this file runs as a script

from sensorium.episode import Episode
from sensorium.model.outcomes import OUTCOME_BINARY, OUTCOMES, collect_all
from sensorium.model.paired import Contrast, contrast, describe
from sensorium.model.pretrain import RunConfig, TrainConfig, load_config, load_layer, train
from sensorium.model.probes import BINARY, CONTINUOUS, Predictions
from sensorium.model.sensory_layer import SensoryLayer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("runs/fix")
DATA_SET = "babble"
ARMS: dict[str, dict[str, object]] = {
    "recon01": {"recon_weight": 0.1, "recon_input": "separate"},
    "recon05": {"recon_weight": 0.5, "recon_input": "separate"},
    "drop01": {"proprio_dropout": 0.1},
    "drop02": {"proprio_dropout": 0.2},
    "recon05drop02": {"recon_weight": 0.5, "proprio_dropout": 0.2, "recon_input": "separate"},
    "fix1-recon05": {"recon_weight": 0.5, "recon_input": "masked"},
}
"""B-fix-0's arms and B-fix-1's single-masked-pass arm (``model.recon_input = masked``)."""
REFERENCES = {"A0": "jepa-babble", "mae": "mae-babble"}
EVAL_SETS = ("held-out babble", "own-pretraining babble", "held-out variety")
SCORES = ("contact AUROCs", "outcome battery")
LAYERS = ("pre", "post")
COLORS = {
    "A0": "#2a78d6",
    "mae": "#e34948",
    "recon01": "#1baf7a",
    "recon05": "#008300",
    "drop01": "#e87ba4",
    "drop02": "#4a3aa7",
    "recon05drop02": "#eda100",
    "fix1-recon05": "#222222",
}


# ---------------------------------------------------------------------------------------------
# train


def train_arm(arm: str, seed: int, steps: int, workers: int) -> None:
    """Train one arm into ``runs/fix/runs/<arm>-s<seed>`` (B-variety-0's recipe plus the arm's
    options)."""
    base = load_config(variety.BASE)
    steps = steps or variety.STEPS
    tc = TrainConfig(
        steps=steps,
        batch_size=16,
        lr=base.train.lr,
        log_every=50,
        eval_windows=0,
        num_workers=workers,
        out_dir=str(ROOT / "runs"),
        save_steps=tuple(s for s in variety.CKPTS if s <= steps),
    )
    data = replace(
        base.data,
        paths=tuple(str(p) for p in variety.files(DATA_SET, "pretrain")),
        space_paths=tuple(str(p) for p in variety.space_files()),
        drop_prefix=variety.DROP,
    )
    model = replace(base.model, senses=variety.NO_TOUCH, **ARMS[arm])  # type: ignore[arg-type]
    augment = replace(base.augment, cache_episodes=4, frame_cache=str(variety.cache_dir(DATA_SET)))
    train(RunConfig(f"{arm}-s{seed}", seed, data, augment, model, tc))


# ---------------------------------------------------------------------------------------------
# encode and probe


def checkpoint(arm: str, seed: int, step: int) -> Path:
    """A trained arm's or a reference's checkpoint."""
    if arm in REFERENCES:
        return variety.ROOT / "runs" / f"{REFERENCES[arm]}-s{seed}" / f"ckpt-{step}.pt"
    return ROOT / "runs" / f"{arm}-s{seed}" / f"ckpt-{step}.pt"


def encoder_names() -> list[str]:
    """Every encoder: floors, then every arm and reference at every checkpoint on disk."""
    names = [f"floor-s{s}" for s in diag.FLOORS]
    for arm in (*REFERENCES, *ARMS):
        for seed in variety.SEEDS:
            names += [
                f"{arm}-s{seed}-n{c}" for c in variety.CKPTS if checkpoint(arm, seed, c).exists()
            ]
    return names


def layer_of(name: str) -> SensoryLayer:
    """Load an encoder by name."""
    if name.startswith("floor-s"):
        return variety.floor(int(name.removeprefix("floor-s")))
    run, step = name.rsplit("-n", 1)
    arm, seed = run.rsplit("-s", 1)
    return load_layer(checkpoint(arm, int(seed), int(step)))


def eval_paths(label: str) -> list[Path]:
    """The episodes of an evaluation set."""
    if label == "held-out babble":
        return variety.files("babble", "heldout")
    if label == "own-pretraining babble":
        return variety.files("babble", "pretrain", 100)
    return variety.files("variety", "heldout")


def slug(label: str) -> str:
    """A file-name form of an evaluation set's label."""
    return label.replace(" ", "_")


def shard_of(items: list[str], shard: str) -> list[str]:
    """This process's share (``i/n``) of a list."""
    i, n = (int(x) for x in shard.split("/"))
    return items[i::n]


def encode(shard: str) -> None:
    """Pre- and post-projector states of this shard's encoders, with all labels, on every
    evaluation set (``runs/fix/enc/<set>/<encoder>.npz``, skipped if present)."""
    k = load_config(variety.BASE).model.substeps
    for label in EVAL_SETS:
        out = ROOT / "enc" / slug(label)
        out.mkdir(parents=True, exist_ok=True)
        names = [
            n
            for n in shard_of(encoder_names(), shard)
            if not (out / f"{n}.npz").exists() and not probed(label, n)
        ]
        if not names:
            continue
        layers = {n: layer_of(n) for n in names}
        hooks = {n: diag.summaries(layer) for n, layer in layers.items()}

        def features(episode: Episode, ends: np.ndarray) -> dict[str, np.ndarray]:
            feats = {}
            for n, layer in layers.items():  # noqa: B023 (used within this iteration only)
                hooks[n].clear()  # noqa: B023
                feats[f"post/{n}"] = layer.encode(episode, stride=variety.STRIDE)[0]
                feats[f"pre/{n}"] = np.concatenate(hooks[n])  # noqa: B023
            return feats

        data = collect_all(eval_paths(label), k, variety.STRIDE, features)
        labels = out / "labels.npz"
        if not labels.exists():  # shards may race here: write a private file, then rename
            tmp = out / f".labels-{shard.replace('/', '-')}.npz"
            np.savez(tmp, groups=data.groups, **data.labels)
            tmp.replace(labels)
        for n in names:
            np.savez(
                out / f"{n}.npz", pre=data.features[f"pre/{n}"], post=data.features[f"post/{n}"]
            )
        logging.info("encoded %d encoders on %s", len(names), label)


def targets(
    y: dict[str, np.ndarray], score: str
) -> tuple[np.ndarray, list[slice], tuple[str, ...], tuple[str, ...]]:
    """Target columns, blocks, names and binary names of a score's probe."""
    names = (*BINARY, *CONTINUOUS) if score == "contact AUROCs" else OUTCOMES
    binary = BINARY if score == "contact AUROCs" else OUTCOME_BINARY
    cols = [y[n].reshape(len(y[n]), -1) for n in names]
    edges = np.cumsum([0, *(c.shape[1] for c in cols)])
    blocks = [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    return np.concatenate(cols, 1), blocks, names, binary


def pred_path(label: str, score: str, layer: str, name: str) -> Path:
    """Saved out-of-fold predictions of one probe."""
    return ROOT / "preds" / slug(label) / f"{slug(score)}-{layer}-{name}.npz"


def probe(shard: str) -> None:
    """Fit this shard's probes (every evaluation set, score and layer of its encoders); an
    encoder whose predictions on a set are all saved is skipped (its encodings may not exist:
    a later study reuses an earlier one's predictions)."""
    for label in EVAL_SETS:
        enc = ROOT / "enc" / slug(label)
        raw = np.load(enc / "labels.npz")
        groups = raw["groups"]
        y = {k: raw[k] for k in raw.files if k != "groups"}
        for name in shard_of(encoder_names(), shard):
            if probed(label, name):
                continue
            feats = np.load(enc / f"{name}.npz")
            for score in SCORES:
                t, blocks, names, binary = targets(y, score)
                for layer in LAYERS:
                    out = pred_path(label, score, layer, name)
                    if out.exists():
                        continue
                    pred = diag.ridge_cv(feats[layer], t, groups, blocks, scale=False)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(out, pred=pred, target=t, groups=groups)
            logging.info("probed %s on %s", name, label)


def probed(label: str, name: str) -> bool:
    """Whether every probe of an encoder on an evaluation set is saved (then it is not encoded
    again: a later study reuses an earlier study's predictions, which pair directly)."""
    return all(pred_path(label, sc, la, name).exists() for sc in SCORES for la in LAYERS)


@functools.cache
def predictions(label: str, score: str, layer: str, name: str) -> Predictions | None:
    """A saved probe as ``Predictions`` (contact: the three AUROCs only); ``None`` if missing."""
    path = pred_path(label, score, layer, name)
    if not path.exists():
        return None
    raw = np.load(path)
    names = (*BINARY, *CONTINUOUS) if score == "contact AUROCs" else OUTCOMES
    binary = BINARY if score == "contact AUROCs" else OUTCOME_BINARY
    widths = {"tip": 6}
    edges = np.cumsum([0, *(widths.get(n, 1) for n in names)])
    blocks = [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    pred = Predictions(raw["pred"], raw["target"], blocks, raw["groups"], names, binary)
    return pred.subset(BINARY) if score == "contact AUROCs" else pred


# ---------------------------------------------------------------------------------------------
# report


def family(label: str, score: str, layer: str, arm: str, step: int) -> list[Predictions]:
    """An arm's (or the floors', ``arm == "floor"``) predictions in seed order."""
    if arm == "floor":
        names = [f"floor-s{s}" for s in diag.FLOORS]
    else:
        names = [f"{arm}-s{s}-n{step}" for s in variety.SEEDS]
    preds = [predictions(label, score, layer, n) for n in names]
    return [p for p in preds if p is not None]


def paired(a: list[Predictions], b: list[Predictions]) -> Contrast | None:
    """``a - b`` paired by seed and episode (``None`` unless both have the same >= 2 seeds)."""
    return contrast(a, b, paired=True) if len(a) == len(b) > 1 else None


def verdict(c: Contrast | None) -> str:
    """The pre-registered reading of ``arm - floor`` (pre-projector contact AUROCs, held-out)."""
    if c is None:
        return "incomplete"
    return "works" if c.resolved and c.point > 0 else "does not work"


def report() -> None:
    """Tables (10,000 steps), contrasts, the verdicts and the figures."""
    step = variety.STEPS
    arms = ("A0", *ARMS, "mae")
    for label in EVAL_SETS:
        variety.say(f"\n## {label}: score at {step:,} steps (mean +- sd over seeds (n))\n")
        variety.say("| arm | contact pre | contact post | outcome pre | outcome post |")
        variety.say("|---|---|---|---|---|")
        for arm in ("floor", *arms):
            cells = [
                variety.scale.cell(family(label, score, layer, arm, step))
                for score in SCORES
                for layer in LAYERS
            ]
            variety.say(f"| {arm} | " + " | ".join(cells) + " |")
        for score in SCORES:
            variety.say(f"\n{label}, {score}: paired contrasts at {step:,} (seed and episode)\n")
            variety.say("| arm | - floor, pre | - floor, post | - A0, pre | - A0, post |")
            variety.say("|---|---|---|---|---|")
            for arm in (*ARMS, "mae"):
                cells = []
                for ref in ("floor", "A0"):
                    for layer in LAYERS:
                        c = paired(
                            family(label, score, layer, arm, step),
                            family(label, score, layer, ref, step),
                        )
                        cells.append(describe(c) if c else "-")
                variety.say(f"| {arm} | " + " | ".join(cells) + " |")
    variety.say(
        f"\n## Pre-registered verdict: arm - floor, pre-projector contact AUROCs, held-out "
        f"babble, {step:,} steps\n"
    )
    variety.say("| arm | arm - floor | verdict |\n|---|---|---|")
    for arm in (*ARMS, "A0", "mae"):
        c = paired(
            family("held-out babble", "contact AUROCs", "pre", arm, step),
            family("held-out babble", "contact AUROCs", "pre", "floor", step),
        )
        variety.say(f"| {arm} | {describe(c) if c else '-'} | {verdict(c)} |")
    fix1()
    for label in EVAL_SETS:
        figure(label)
    curves()
    loss_figure()


def training_summary(arm: str) -> dict[str, float]:
    """Seed means of the last 500 logged steps' ``pred`` and ``recon`` and the run's seq/s."""
    rows = []
    for path in sorted(collect_logs().glob(f"{arm}-s*/log.jsonl")):
        log = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        last = [r for r in log if r["step"] > log[-1]["step"] - 500]
        rows.append(
            {
                "pred": float(np.mean([r["pred"] for r in last])),
                "recon": float(np.mean([r.get("recon", np.nan) for r in last])),
                "seq_per_s": float(log[-1]["seq_per_s"]),
            }
        )
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {}


def fix1() -> None:
    """B-fix-1's pre-registered contrasts: the single masked pass against B-fix-0's two-pass
    ``recon05``, then against MAE and the floor, on every evaluation set, with the training
    summary (next-state loss, reconstruction, speed)."""
    arm, step = "fix1-recon05", variety.STEPS
    counts = {
        f"{label} / {score} / {layer}": len(family(label, score, layer, arm, step))
        for label in EVAL_SETS
        for score in SCORES
        for layer in LAYERS
    }
    if not any(counts.values()):
        variety.say(f"\n## B-fix-1: {arm} has no predictions (not run)\n")
        return
    missing = [where for where, n in counts.items() if n < len(variety.SEEDS)]
    if missing:
        raise RuntimeError(
            f"B-fix-1 is incomplete: {arm} lacks seeds at {step:,} steps on " + "; ".join(missing)
        )
    variety.say(f"\n## B-fix-1: {arm} (one masked pass) at {step:,} steps\n")
    for label in EVAL_SETS:
        variety.say(f"\n{label} (paired by seed and episode)\n")
        variety.say("| contrast | score | pre-projector | post-projector |\n|---|---|---|---|")
        for ref in ("recon05", "mae", "floor"):
            for score in SCORES:
                cells = []
                for layer in LAYERS:
                    c = paired(
                        family(label, score, layer, arm, step),
                        family(label, score, layer, ref, step),
                    )
                    cells.append(describe(c) if c else "-")
                variety.say(f"| {arm} - {ref} | {score} | " + " | ".join(cells) + " |")
    variety.say("\nTraining (seed means; loss terms over the last 500 steps)\n")
    variety.say("| arm | next-state MSE | reconstruction | seq/s |\n|---|---|---|---|")
    for name in ("A0", "recon05", arm):
        t = training_summary(name)
        if t:
            variety.say(f"| {name} | {t['pred']:.3f} | {t['recon']:.3f} | {t['seq_per_s']:.1f} |")


def loss_figure() -> None:
    """Next-state MSE and reconstruction against steps: A0, recon05 and the single pass."""
    runs = collect_logs()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm in ("A0", "recon05", "fix1-recon05"):
        for path in sorted(runs.glob(f"{arm}-s*/log.jsonl")):
            log = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            for ax, term in zip(axes, ("pred", "recon"), strict=True):
                pts = [(r["step"], r[term]) for r in log if term in r]
                if pts:
                    x, y = zip(*pts, strict=True)
                    label = arm if path.parent.name.endswith("-s0") else None
                    ax.plot(x, y, color=COLORS[arm], lw=0.8, alpha=0.7, label=label)
    for ax, title in zip(axes, ("next-state MSE (pred)", "reconstruction (recon)"), strict=True):
        ax.set(xscale="log", xlabel="step", title=title)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("B-fix-1: loss terms per seed, one masked pass against two passes")
    fig.tight_layout()
    out = ROOT / "figures" / "fix1_losses.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def figure(label: str) -> None:
    """Probe score against steps per arm (seed SE bars), per score and layer, with the floor."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for row, score in enumerate(SCORES):
        for col, layer in enumerate(LAYERS):
            ax = axes[row][col]
            for arm in ("A0", *ARMS, "mae"):
                pts = []
                for step in variety.CKPTS:
                    vals = [p.scores()["mean"] for p in family(label, score, layer, arm, step)]
                    if vals:
                        se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
                        pts.append((max(step, 60), np.mean(vals), se))
                if pts:
                    x, y, e = zip(*pts, strict=True)
                    ax.errorbar(
                        x, y, e, marker="o", ms=4, lw=1.6, capsize=2, color=COLORS[arm], label=arm
                    )
            floor = [p.scores()["mean"] for p in family(label, score, layer, "floor", 0)]
            if floor:
                ax.axhline(np.mean(floor), color="#888888", ls="--", lw=1.2, label="floor")
            ax.set(xscale="log", title=f"{score}, {layer}-projector", xlabel="step (0 drawn at 60)")
            ax.grid(alpha=0.25)
    axes[0][0].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle(f"B-fix-0 on {label} (bars: seed SE)")
    fig.tight_layout()
    out = ROOT / "figures" / f"fix_probe_{slug(label)}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    logging.info("wrote %s", out)


def collect_logs() -> Path:
    """Copy every run's ``log.jsonl`` (A0 from ``runs/variety``) to ``runs/fix/curve_runs``."""
    runs = ROOT / "curve_runs"
    runs.mkdir(parents=True, exist_ok=True)
    for seed in variety.SEEDS:
        for arm, source in [("A0", variety.ROOT / "runs" / f"jepa-babble-s{seed}")] + [
            (arm, ROOT / "runs" / f"{arm}-s{seed}") for arm in ARMS
        ]:
            log = source / "log.jsonl"
            if log.exists():
                (runs / f"{arm}-s{seed}").mkdir(exist_ok=True)
                (runs / f"{arm}-s{seed}" / "log.jsonl").write_text(log.read_text())
    return runs


def curves() -> None:
    """Training loss terms per seed (thin) and their seed mean (thick), one row per arm: the
    next-state MSE, the weighted SIGReg, the reconstruction term and the total."""
    runs = collect_logs()
    terms = ("pred", "reg", "recon", "loss")
    arms = ("A0", *ARMS)
    fig, axes = plt.subplots(len(arms), len(terms), figsize=(15, 2.6 * len(arms)), squeeze=False)
    for row, arm in enumerate(arms):
        logs = [
            [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            for path in sorted(runs.glob(f"{arm}-s*/log.jsonl"))
        ]
        for col, term in enumerate(terms):
            ax = axes[row][col]
            series = []
            for rows in logs:
                pts = [(r["step"], r[term]) for r in rows if term in r]
                if pts:
                    x, y = zip(*pts, strict=True)
                    ax.plot(x, y, color=COLORS[arm], lw=0.7, alpha=0.5)
                    series.append(np.array(y))
            if series and len({len(v) for v in series}) == 1:
                ax.plot(x, np.mean(series, 0), color=COLORS[arm], lw=2)
            if not series:
                ax.text(0.5, 0.5, "not in this arm", ha="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            ax.set_xscale("log")
            ax.grid(alpha=0.25)
            if row == 0:
                ax.set_title(term)
        axes[row][0].set_ylabel(arm)
    fig.suptitle("B-fix-0: training loss terms per seed (thin) and seed mean (thick)")
    fig.tight_layout()
    out = ROOT / "figures" / "fix_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logging.info("wrote %s", out)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=("train", "encode", "probe", "report"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    globals()["ROOT"] = args.root
    if args.command == "train":
        if args.arm is None:
            parser.error("train needs --arm")
        train_arm(args.arm, args.seed, args.steps, args.workers)
        return
    found = sorted((variety.ROOT / "runs").glob("*/ckpt-*.pt")) or sorted(
        (ROOT / "runs").glob("*/ckpt-*.pt")
    )
    if not found:
        parser.error("no checkpoint to read the shared channel space from")
    space = load_layer(found[0]).space  # the arms' shared space, stored in every checkpoint
    variety.shared_space = lambda: space
    if args.command == "encode":
        encode(args.shard)
    elif args.command == "probe":
        probe(args.shard)
    else:
        report()


if __name__ == "__main__":
    main()
