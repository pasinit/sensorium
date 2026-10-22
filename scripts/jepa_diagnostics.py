"""Why is the V+P JEPA below its random-init floor? Diagnostics on B-variety-0's encoders.

Mac only, on the 10,000-step checkpoints (and any other ``ckpt-<step>.pt`` present) and the logs
under ``runs/variety/runs``, with the 100 + 100 held-out episodes of ``runs/variety/split.json``:

* ``curves``: training loss terms against steps per arm, with the embedding std and effective rank
  logged by the run, and the prediction loss divided by the embedding variance (is the loss falling
  because the predictor learns, or because the targets shrink?).
* ``encode --test T``: every encoder's states on the held-out windows (stride 8), both after the
  projector (the state the probes read and the loss acts on) and before it (the fusion summary
  token), plus each encoder's next-state prediction error on held-out training-geometry sequences
  against two baselines: copying the current state and predicting the mean.
* ``report``: geometry (covariance spectrum, effective rank, per-dimension std, lag-1 temporal
  correlation of consecutive windows), the contact battery from pre- and post-projector features
  with the battery's ridge, a per-column standardised ridge and a small MLP (episode-grouped
  5-fold), and the figures (``runs/variety/figures/jepa_*.png``).

    uv run python scripts/jepa_diagnostics.py curves
    uv run python scripts/jepa_diagnostics.py encode --test babble
    uv run python scripts/jepa_diagnostics.py report
"""

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import torch
import variety_test as variety  # scripts/ is on sys.path when this file runs as a script

from sensorium.episode import Episode
from sensorium.model.controls import build_probe_data
from sensorium.model.data import SequenceDataset
from sensorium.model.paired import contrast, describe, difference_in_differences
from sensorium.model.pretrain import load_config, load_layer
from sensorium.model.probes import (
    BINARY,
    CONTINUOUS,
    Predictions,
    choose_alphas,
    episode_folds,
    ridge_predict,
)
from sensorium.model.sensory_layer import SensoryLayer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FOLDS = 5
ARMS = ("jepa-babble", "jepa-variety", "mae-babble", "mae-variety")
FLOORS = (0, 1, 2)
SEQ_BATCHES = 8
COLORS = {
    "floor": "#888888",
    "jepa-babble": "#2a78d6",
    "jepa-variety": "#104281",
    "mae-babble": "#eb6834",
    "mae-variety": "#b8410f",
}


def out_dir() -> Path:
    """Where the diagnostics go."""
    return variety.ROOT / "jepa_diag"


# ---------------------------------------------------------------------------------------------
# curves


def read_log(run: str) -> list[dict[str, float]]:
    """The logged rows of a run."""
    path = variety.ROOT / "runs" / run / "log.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def curves() -> None:
    """Loss, regulariser, std, rank and variance-normalised loss against steps per arm."""
    keys = ("pred", "pred_per_var", "reg", "std", "erank")
    fig, axes = plt.subplots(1, len(keys), figsize=(20, 3.8))
    for arm in ARMS:
        for seed in variety.SEEDS:
            rows = [r for r in read_log(f"{arm}-s{seed}") if "std" in r]
            if not rows:
                continue
            step = np.array([r["step"] for r in rows])
            vals = {k: np.array([r.get(k, np.nan) for r in rows]) for k in ("pred", "reg", "std")}
            vals["erank"] = np.array([r.get("erank", np.nan) for r in rows])
            vals["pred_per_var"] = vals["pred"] / vals["std"] ** 2
            for ax, key in zip(axes, keys, strict=True):
                ax.plot(
                    step,
                    vals[key],
                    color=COLORS[arm],
                    lw=1,
                    alpha=0.7,
                    label=arm if seed == 0 else None,
                )
    titles = {
        "pred": "training loss term (JEPA next-state MSE; MAE reconstruction)",
        "pred_per_var": "loss / embedding variance (std^2)",
        "reg": "SIGReg (weighted; 0 for MAE)",
        "std": "embedding std per dimension (batch)",
        "erank": "effective rank of the batch (of 128)",
    }
    for ax, key in zip(axes, keys, strict=True):
        ax.set_title(titles[key], fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("step")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    path = variety.ROOT / "figures" / "jepa_curves.png"
    fig.savefig(path, dpi=130)
    logging.info("wrote %s", path)
    for arm in ARMS:
        rows = [read_log(f"{arm}-s{s}") for s in variety.SEEDS]
        for at in (250, 1000, 10000):
            sel = [r for rs in rows for r in rs if r["step"] == at and "std" in r]
            if sel:
                p = np.mean([r["pred"] for r in sel])
                s = np.mean([r["std"] for r in sel])
                e = np.mean([r["erank"] for r in sel])
                variety.say(f"| {arm} | {at} | {p:.3f} | {s:.3f} | {p / s**2:.3f} | {e:.1f} |")


# ---------------------------------------------------------------------------------------------
# encode


def encoders() -> dict[str, SensoryLayer]:
    """Floors and every probed SSL checkpoint on disk."""
    out = {f"floor-vp-s{s}": variety.floor(s) for s in FLOORS}
    for arm in ARMS:
        for seed in variety.SEEDS:
            for path in sorted((variety.ROOT / "runs" / f"{arm}-s{seed}").glob("ckpt-*.pt")):
                step = int(path.stem.removeprefix("ckpt-"))
                if step > 0:
                    out[f"{arm}-s{seed}-n{step}"] = load_layer(path)
    return out


def summaries(layer: SensoryLayer) -> list[np.ndarray]:
    """A forward hook that records the fusion summary token (the projector's input)."""
    store: list[np.ndarray] = []

    def hook(module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        store.append(args[0].detach().cpu().numpy())

    layer.model.encoder.projector.register_forward_pre_hook(hook)
    return store


@torch.no_grad()
def predictor_errors(layer: SensoryLayer, paths: list[Path]) -> dict[str, float]:
    """Next-state error of the predictor on held-out training-geometry sequences, with the
    copy-the-current-state and predict-the-mean baselines (all per dimension, in state units)."""
    cfg = replace(layer.loader, touch_noise=False, frame_cache="")
    data = SequenceDataset(paths, layer.space, layer.model.data_senses, cfg, seed=0, train=False)
    rng = np.random.default_rng(0)
    model = layer.model.eval()
    sums: dict[str, float] = defaultdict(float)
    for _ in range(SEQ_BATCHES):
        seqs = [data.sample(rng) for _ in range(16)]
        batch = {k: torch.as_tensor(np.stack([s[k] for s in seqs])) for k in seqs[0]}
        states = model.embed(batch)
        cond = model.action(batch["action"].flatten(0, 1)).view_as(states)
        pred = model.predict(states[:, :-1], cond[:, :-1])
        target = states[:, 1:]
        flat = target.reshape(-1, target.size(-1))
        sums["var"] += float(flat.var(0).mean()) / SEQ_BATCHES
        sums["pred"] += float((pred - target).pow(2).mean()) / SEQ_BATCHES
        sums["copy"] += float((states[:, :-1] - target).pow(2).mean()) / SEQ_BATCHES
        within = target - target.mean(1, keepdim=True)
        sums["within_seq_var"] += float(within.pow(2).mean()) / SEQ_BATCHES
    return dict(sums)


def encode(test: str) -> None:
    """States (post and pre projector) and predictor errors of every encoder on one test set."""
    layers = encoders()
    paths = variety.files(test, "heldout")
    hooks = {name: summaries(layer) for name, layer in layers.items()}
    k = load_config(variety.BASE).model.substeps

    def features(episode: Episode, ends: np.ndarray) -> dict[str, np.ndarray]:
        out = {}
        for name, layer in layers.items():
            hooks[name].clear()
            states, own = layer.encode(episode, stride=variety.STRIDE)
            assert np.array_equal(own, ends)
            out[f"post/{name}"] = states
            out[f"pre/{name}"] = np.concatenate(hooks[name])
        return out

    data = build_probe_data(paths, k, variety.STRIDE, features)
    errors = {}
    for name, layer in layers.items():
        errors[name] = predictor_errors(layer, paths)
        logging.info("%s on %s: %s", name, test, errors[name])
    out_dir().mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir() / f"{test}.npz",
        groups=data.groups,
        **{f"x/{n}": v for n, v in data.features.items()},
        **{f"y/{n}": v for n, v in data.labels.items()},
    )
    (out_dir() / f"{test}_predictor.json").write_text(json.dumps(errors, indent=1))


# ---------------------------------------------------------------------------------------------
# report


def load(test: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Features, labels and groups saved by ``encode``."""
    raw = np.load(out_dir() / f"{test}.npz")
    x = {k.removeprefix("x/"): raw[k] for k in raw.files if k.startswith("x/")}
    y = {k.removeprefix("y/"): raw[k] for k in raw.files if k.startswith("y/")}
    return x, y, raw["groups"]


def geometry(x: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """Spectrum summaries and the lag-1 correlation of consecutive windows of an episode."""
    c = x - x.mean(0)
    eig = np.sort(np.linalg.eigvalsh(np.cov(c.T)))[::-1].clip(min=0)
    p = eig / eig.sum()
    same = groups[1:] == groups[:-1]
    z = c / c.std(0).clip(min=1e-9)
    lag1 = float(np.mean((z[1:][same] * z[:-1][same]).mean(0)))
    between = np.array([c[groups == g].mean(0) for g in np.unique(groups)])
    return {
        "erank": float(np.exp(-(p * np.log(p.clip(min=1e-12))).sum())),
        "participation": float(eig.sum() ** 2 / (eig**2).sum()),
        "top1_share": float(p[0]),
        "std": float(c.std(0).mean()),
        "lag1": lag1,
        "between_episode_share": float(between.var(0).sum() / c.var(0).sum()),
    }


def battery(y: dict[str, np.ndarray]) -> tuple[np.ndarray, list[slice]]:
    """The contact battery targets and their column blocks (as ``probes.predict_probes``)."""
    cols = [y[n].reshape(len(y[n]), -1) for n in (*BINARY, *CONTINUOUS)]
    edges = np.cumsum([0, *(c.shape[1] for c in cols)])
    return np.concatenate(cols, 1), [
        slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)
    ]


def ridge_cv(
    x: np.ndarray, t: np.ndarray, groups: np.ndarray, blocks: list[slice], scale: bool
) -> np.ndarray:
    """Episode-grouped 5-fold ridge predictions (``scale``: standardise every column first)."""
    pred = np.zeros_like(t)
    for tr, te in episode_folds(groups, FOLDS):
        xtr, xte = x[tr], x[te]
        if scale:
            m, s = xtr.mean(0), xtr.std(0).clip(min=1e-6)
            xtr, xte = (xtr - m) / s, (xte - m) / s
        alphas = choose_alphas(xtr, t[tr], groups[tr], blocks)
        for block, alpha in zip(blocks, alphas, strict=True):
            pred[te, block] = ridge_predict(xtr, t[tr][:, block], xte, alpha)
    return pred


def mlp_cv(x: np.ndarray, t: np.ndarray, groups: np.ndarray, seed: int = 0) -> np.ndarray:
    """Episode-grouped 5-fold predictions of a one-hidden-layer MLP (256 GELU units, AdamW,
    weight decay 1e-2, 300 full-batch epochs; inputs and targets standardised on the fold)."""
    pred = np.zeros_like(t)
    for tr, te in episode_folds(groups, FOLDS):
        torch.manual_seed(seed)
        m, s = x[tr].mean(0), x[tr].std(0).clip(min=1e-6)
        tm, ts = t[tr].mean(0), t[tr].std(0).clip(min=1e-6)
        xtr = torch.as_tensor((x[tr] - m) / s, dtype=torch.float32)
        ttr = torch.as_tensor((t[tr] - tm) / ts, dtype=torch.float32)
        net = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], 256), torch.nn.GELU(), torch.nn.Linear(256, t.shape[1])
        )
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
        for _ in range(300):
            loss = (net(xtr) - ttr).pow(2).mean()
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
        with torch.no_grad():
            out = net(torch.as_tensor((x[te] - m) / s, dtype=torch.float32)).numpy()
        pred[te] = out * ts + tm
    return pred


def families(names: list[str]) -> list[tuple[str, list[str]]]:
    """``(label, encoder names)``: the floors, then every arm at every probed step."""
    steps = sorted({int(n.rsplit("-n", 1)[1]) for n in names if "-n" in n})
    out = [("floor", [n for n in names if n.startswith("floor-vp-")])]
    for arm in ARMS:
        for step in steps:
            members = [n for n in names if n.startswith(f"{arm}-s") and n.endswith(f"-n{step}")]
            if members:
                out.append((f"{arm} @{step}", members))
    return out


def geometry_table(test: str, x: dict[str, np.ndarray], groups: np.ndarray) -> None:
    """Spectrum, temporal and predictor summaries per family (means over seeds)."""
    names = sorted({k.split("/", 1)[1] for k in x})
    predictor = json.loads((out_dir() / f"{test}_predictor.json").read_text())
    variety.say(f"\n## {test} test set\n")
    variety.say(
        "| encoder | erank post / pre | top eigen share | lag-1 corr post / pre | "
        "between-episode var share | pred / var | copy / var | within-seq var / var |"
    )
    variety.say("|---|---|---|---|---|---|---|---|")
    for label, members in families(names):
        post = [geometry(x[f"post/{n}"], groups) for n in members]
        pre = [geometry(x[f"pre/{n}"], groups) for n in members]
        pe = [predictor[n] for n in members]

        def avg(rows: list[dict[str, float]], key: str) -> float:
            return float(np.mean([r[key] for r in rows]))

        def ratio(rows: list[dict[str, float]], key: str) -> float:
            return float(np.mean([r[key] / r["var"] for r in rows]))

        variety.say(
            f"| {label} | {avg(post, 'erank'):.1f} / {avg(pre, 'erank'):.1f} | "
            f"{avg(post, 'top1_share'):.3f} | "
            f"{avg(post, 'lag1'):.3f} / {avg(pre, 'lag1'):.3f} | "
            f"{avg(post, 'between_episode_share'):.3f} | {ratio(pe, 'pred'):.3f} | "
            f"{ratio(pe, 'copy'):.3f} | {ratio(pe, 'within_seq_var'):.3f} |"
        )


def probe_table(
    test: str, x: dict[str, np.ndarray], y: dict[str, np.ndarray], groups: np.ndarray
) -> None:
    """The contact battery from pre- and post-projector features with three probes."""
    names = sorted({k.split("/", 1)[1] for k in x})
    target, blocks = battery(y)
    variety.say(f"\nContact battery mean on {test} (episode-grouped 5-fold; mean over seeds)\n")
    variety.say("| encoder | ridge post / pre | std-ridge post / pre | MLP post / pre |")
    variety.say("|---|---|---|---|")
    for label, members in families(names):
        cells = []
        for kind in ("ridge", "std-ridge", "mlp"):
            pair = []
            for where in ("post", "pre"):
                vals = []
                for n in members:
                    feats = x[f"{where}/{n}"]
                    if kind == "mlp":
                        p = mlp_cv(feats, target, groups)
                    else:
                        p = ridge_cv(feats, target, groups, blocks, kind == "std-ridge")
                    vals.append(Predictions(p, target, blocks, groups).scores()["mean"])
                pair.append(f"{np.mean(vals):.3f}")
            cells.append(" / ".join(pair))
        variety.say(f"| {label} | " + " | ".join(cells) + " |")
        logging.info("probed %s on %s", label, test)


def report() -> None:
    """Geometry, predictor and probe tables and the spectrum figure."""
    variety.say("Exploratory diagnostics of B-variety-0's encoders (not pre-registered).")
    for test in variety.SETS:
        x, y, groups = load(test)
        geometry_table(test, x, groups)
        probe_table(test, x, y, groups)
        spectrum_figure(test, x, sorted({k.split("/", 1)[1] for k in x}))


def spectrum_figure(test: str, x: dict[str, np.ndarray], names: list[str]) -> None:
    """Normalised covariance eigenvalues (log) of post- and pre-projector features."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, where in zip(axes, ("post", "pre"), strict=True):
        for n in names:
            kind = "floor" if n.startswith("floor") else n.rsplit("-s", 1)[0]
            if "-n" in n and not n.endswith("-n10000"):
                continue
            c = x[f"{where}/{n}"] - x[f"{where}/{n}"].mean(0)
            eig = np.sort(np.linalg.eigvalsh(np.cov(c.T)))[::-1].clip(min=1e-12)
            ax.plot(
                np.arange(1, len(eig) + 1),
                eig / eig.sum(),
                color=COLORS[kind],
                lw=1,
                alpha=0.8,
                label=kind if n.endswith(("s0-n10000", "floor-vp-s0")) else None,
            )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="component",
            ylabel="share of variance",
            title=f"{where}-projector features, {test} test (10k steps)",
        )
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    path = variety.ROOT / "figures" / f"jepa_spectrum_{test}.png"
    fig.savefig(path, dpi=130)
    logging.info("wrote %s", path)


def battery_score(x: np.ndarray, y: dict[str, np.ndarray], groups: np.ndarray) -> float:
    """Contact-battery mean of one feature matrix (episode-grouped 5-fold ridge)."""
    target, blocks = battery(y)
    pred = ridge_cv(x, target, groups, blocks, scale=False)
    return float(Predictions(pred, target, blocks, groups).scores()["mean"])


def cell(values: list[float]) -> str:
    """``mean +- SE (n)``."""
    v = np.asarray(values)
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return f"{v.mean():.3f} +- {se:.3f} ({len(v)})"


def train_probe(n_episodes: int = 100) -> None:
    """The contact battery on each encoder's own pretraining episodes (the first ``n_episodes`` of
    its set's pretraining list, balanced hand/pincer) against the held-out set of the same data,
    pre and post projector, with the floors on the same episodes."""
    k = load_config(variety.BASE).model.substeps
    rows = []
    for name in variety.SETS:
        paths = variety.files(name, "pretrain", n_episodes)
        layers = {f"floor-vp-s{s}": variety.floor(s) for s in FLOORS}
        for obj in ("jepa", "mae"):
            for seed in variety.SEEDS:
                run = variety.ROOT / "runs" / f"{obj}-{name}-s{seed}" / "ckpt-10000.pt"
                layers[f"{obj}-{name}-s{seed}-n10000"] = load_layer(run)
        hooks = {n: summaries(layer) for n, layer in layers.items()}

        def features(episode: Episode, ends: np.ndarray) -> dict[str, np.ndarray]:
            out = {}
            for n, layer in layers.items():  # noqa: B023 (used inside this loop iteration only)
                hooks[n].clear()  # noqa: B023
                out[f"post/{n}"] = layer.encode(episode, stride=variety.STRIDE)[0]
                out[f"pre/{n}"] = np.concatenate(hooks[n])  # noqa: B023
            return out

        data = build_probe_data(paths, k, variety.STRIDE, features)
        np.savez(
            out_dir() / f"train_{name}.npz",
            groups=data.groups,
            **{f"x/{n}": v for n, v in data.features.items()},
            **{f"y/{n}": v for n, v in data.labels.items()},
        )
        held_x, held_y, held_groups = load(name)
        for label, prefix in (
            ("floor", "floor-vp-s"),
            (f"jepa-{name}", f"jepa-{name}-s"),
            (f"mae-{name}", f"mae-{name}-s"),
        ):
            members = [n for n in layers if n.startswith(prefix)]
            cells = {}
            for where in ("post", "pre"):
                tr = [
                    battery_score(data.features[f"{where}/{n}"], data.labels, data.groups)
                    for n in members
                ]
                he = [battery_score(held_x[f"{where}/{n}"], held_y, held_groups) for n in members]
                cells[where] = (tr, he)
            rows.append((name, label, cells))
            logging.info("train-probed %s on %s", label, name)
    variety.say(
        "\nContact battery on the encoder's OWN PRETRAINING episodes (100 of its set's pretraining "
        "list, 50 hand / 50 pincer; every probe fit and test episode was seen in pretraining; the "
        "5 episode-grouped folds only keep the linear probe from testing on an episode it fitted) "
        "against the 100 held-out episodes of the same set. Floor = the random-init encoder on the "
        "same episodes. Mean +- SE over seeds (n).\n"
    )
    variety.say(
        "| data | encoder | train episodes, post | held-out, post | train - held-out, post | "
        "train episodes, pre | held-out, pre |"
    )
    variety.say("|---|---|---|---|---|---|---|")
    for name, label, cells in rows:
        (tr, he), (trp, hep) = cells["post"], cells["pre"]
        gap = [a - b for a, b in zip(tr, he, strict=True)]
        variety.say(
            f"| {name} | {label} | {cell(tr)} | {cell(he)} | {cell(gap)} | "
            f"{cell(trp)} | {cell(hep)} |"
        )


def battery_predictions(x: np.ndarray, y: dict[str, np.ndarray], groups: np.ndarray) -> Predictions:
    """Out-of-fold contact-battery predictions of one feature matrix (5-fold ridge)."""
    target, blocks = battery(y)
    return Predictions(ridge_cv(x, target, groups, blocks, scale=False), target, blocks, groups)


def paired_layers() -> None:
    """Paired contrasts (seed and episode; ``sensorium.model.paired``) of the layer claim on the
    own-pretraining and held-out episodes: arm - floor pre-projector, arm - floor post-projector,
    and the layer effect relative to the floor ``(arm - floor)_pre - (arm - floor)_post``."""
    variety.say(
        "\nPaired by seed (arm seed s and floor seed s share the initial encoder) and by episode; "
        "MDD = 2.8 x combined SE; 10,000 steps; 5-fold episode-grouped ridge. Own-pretraining "
        "episodes: every probe fit and test episode was seen by the encoder in pretraining.\n"
    )
    variety.say(
        "| data | episodes | arm | arm - floor, pre | arm - floor, post | "
        "layer effect (pre - post) |"
    )
    variety.say("|---|---|---|---|---|---|")
    for name in variety.SETS:
        for label, file in (("own pretraining", f"train_{name}.npz"), ("held-out", f"{name}.npz")):
            raw = np.load(out_dir() / file)
            y = {k.removeprefix("y/"): raw[k] for k in raw.files if k.startswith("y/")}
            groups = raw["groups"]

            def fam(prefix: str, where: str) -> list[Predictions]:
                suffix = "" if prefix.startswith("floor") else "-n10000"
                return [
                    battery_predictions(raw[f"x/{where}/{prefix}{s}{suffix}"], y, groups)  # noqa: B023
                    for s in variety.SEEDS
                ]

            floor = {w: fam("floor-vp-s", w) for w in ("pre", "post")}
            for obj in ("jepa", "mae"):
                arm = {w: fam(f"{obj}-{name}-s", w) for w in ("pre", "post")}
                pre = contrast(arm["pre"], floor["pre"], paired=True)
                post = contrast(arm["post"], floor["post"], paired=True)
                layer = difference_in_differences(
                    arm["pre"], floor["pre"], arm["post"], floor["post"]
                )
                variety.say(
                    f"| {name} | {label} | {obj} | {describe(pre)} | {describe(post)} | "
                    f"{describe(layer)} |"
                )


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=("curves", "encode", "report", "trainprobe", "paired"))
    parser.add_argument("--test", choices=variety.SETS)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    first = next(iter(sorted((variety.ROOT / "runs").glob("*/ckpt-10000.pt"))))
    space = load_layer(first).space  # the arms' shared space, stored in every checkpoint
    variety.shared_space = lambda: space
    if args.command == "curves":
        curves()
    elif args.command == "encode":
        if args.test is None:
            parser.error("encode needs --test")
        encode(args.test)
    elif args.command == "trainprobe":
        train_probe()
    elif args.command == "paired":
        paired_layers()
    else:
        report()


if __name__ == "__main__":
    main()
