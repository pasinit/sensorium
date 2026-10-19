"""B-ctrl-0: positive controls for the probe battery, and its variance.

Same 120 episodes and split as ``scripts/decisive_test.py`` (pretrain on 100, probe the 20 never
seen), same probes (leave-one-episode-out ridge on windows at stride 8). Controls:

* supervised ceiling: the tiny encoder trained on the probe labels (``train``);
* frozen DINOv2 ViT-S/14 (+ proprio), and PCA-reduced raw inputs (``features``);
* random-init floors and the trained SSL encoders of PR #20, probed here with out-of-fold
  predictions so every number gets a bootstrap over the 20 held-out episodes (``probe``).

    uv run python scripts/controls_test.py features
    uv run python scripts/controls_test.py train --variant sup-vp --seed 0
    uv run python scripts/controls_test.py probe
    uv run python scripts/controls_test.py report
"""

import argparse
import json
import logging
import re
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

from sensorium.episode import Episode
from sensorium.model.controls import (
    RawFeatures,
    SupervisedConfig,
    block_rms,
    build_probe_data,
    concat_scaled,
    dino_features,
    few_shot_predictions,
    load_dino,
    pca_apply,
    pca_fit,
    train_supervised,
)
from sensorium.model.data import LoaderConfig, SequenceDataset
from sensorium.model.diagnostics import heldout_loss
from sensorium.model.pretrain import (
    RunConfig,
    TrainConfig,
    build_space,
    load_config,
    load_layer,
    train,
)
from sensorium.model.probes import Predictions, ProbeData, collect, predict_probes
from sensorium.model.sensory_layer import ModelConfig, SensoryLayer, SensoryModel

DATA = Path("data/probe_test/humanoid")
SPLIT = Path("runs/decisive/split.json")
ROOT = Path("runs/controls")
DINO_WEIGHTS = Path("data/external/dinov2/dinov2_vits14_pretrain.pth")
STRIDE = 8
DRAWS = 300
NO_TOUCH = ("vision", "proprio_body", "vestibular", "proprio_effector")
VARIANTS = {"sup-vp": NO_TOUCH, "sup-all": None, "sup-vp128": NO_TOUCH}  # None: all senses
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


def base_config():  # type: ignore[no-untyped-def]
    """The tiny model configuration and loader settings."""
    return load_config(Path("configs/model/cpu_tiny.toml"))


# ---------------------------------------------------------------------------------------------
# feature controls


def fit_raw(space, substeps: int):  # type: ignore[no-untyped-def]
    """Block RMS and PCA fits of the raw feature sets, on windows of 50 pretraining episodes."""
    raw = RawFeatures(space, substeps)
    blocks: dict[str, list[np.ndarray]] = {}
    for path in files("pretrain")[::2]:
        episode = Episode.load(path, privileged=False)
        first = (substeps - 1) * 1
        ends = np.arange(first, episode.length, 16)
        for name, value in raw.blocks(episode, ends).items():
            blocks.setdefault(name, []).append(value)
    stacked = {n: np.concatenate(v) for n, v in blocks.items()}
    rms = {n: block_rms(v) for n, v in stacked.items()}
    sets = {
        "raw-vp": ("proprio", "vision"),
        "raw-nonvisual": ("touch", "audio", "proprio"),
        "raw-all": ("touch", "audio", "proprio", "vision"),
    }
    fits = {}
    for name, parts in sets.items():
        x = concat_scaled([stacked[p] for p in parts], [rms[p] for p in parts])
        fits[name] = {k: pca_fit(x, k) for k in (32, 128)}
    return raw, rms, sets, fits


def compute_features() -> None:
    """Held-out probe data with DINOv2 and raw-PCA features (cached in ``runs/controls``)."""
    ROOT.mkdir(parents=True, exist_ok=True)
    cfg = base_config()
    substeps = cfg.model.substeps
    space = build_space(files("pretrain"), cfg.data.stats_episodes)
    raw, rms, sets, fits = fit_raw(space, substeps)
    dino = load_dino(DINO_WEIGHTS)
    torch.set_num_threads(4)

    def features(episode: Episode, ends: np.ndarray) -> dict[str, np.ndarray]:
        blocks = raw.blocks(episode, ends)
        d = dino_features(dino, episode.rgb[ends])
        out = {
            "dino": d,
            "dino+proprio": concat_scaled([d, blocks["proprio"]], [rms["dino"], rms["proprio"]]),
            "proprio": blocks["proprio"],
        }
        for name, parts in sets.items():
            x = concat_scaled([blocks[p] for p in parts], [rms[p] for p in parts])
            for k, fit in fits[name].items():
                out[f"{name}-pca{k}"] = pca_apply(x, fit)
        return out

    # the DINO block RMS is fitted on the held-out frames themselves (an unsupervised scale only)
    first = Episode.load(files("heldout")[0], privileged=False)
    rms["dino"] = block_rms(dino_features(dino, first.rgb[::8]))
    data = build_probe_data(files("heldout"), substeps, STRIDE, features)
    np.savez(
        ROOT / "heldout_features.npz",
        groups=data.groups,
        **{f"x_{k}": v for k, v in data.features.items()},
        **{f"y_{k}": v for k, v in data.labels.items()},
    )
    logging.info("features: %s", {k: v.shape for k, v in data.features.items()})


def load_feature_data() -> ProbeData:
    """The cached held-out feature windows."""
    raw = np.load(ROOT / "heldout_features.npz")
    return ProbeData(
        {k[2:]: raw[k] for k in raw.files if k.startswith("x_")},
        {k[2:]: raw[k] for k in raw.files if k.startswith("y_")},
        raw["groups"],
    )


# ---------------------------------------------------------------------------------------------
# training and probing encoders


def train_variant(variant: str, seed: int, steps: int) -> None:
    """Train a supervised-ceiling encoder, or a few-shot fine-tune (``ft-random-k4``).

    Fine-tunes use ``k`` labelled pretraining episodes (a seeded subset) and the V+P encoder, and
    start from random weights or from the c-none SSL encoder of the same seed (PR #20, 2,000 steps).
    """
    cfg = base_config()
    space = build_space(files("pretrain"), cfg.data.stats_episodes)
    run = ROOT / "runs" / f"{variant}-s{seed}"
    if variant.startswith("ft-"):
        _, init, shots = variant.split("-")
        episodes = files("pretrain")
        picked = np.random.default_rng(seed).choice(
            len(episodes), int(shots.removeprefix("k")), replace=False
        )
        ssl = Path(f"runs/teacher/c-none-s{seed}/ckpt-2000.pt") if init == "ssl" else None
        train_supervised(
            SupervisedConfig(model=replace(cfg.model, senses=NO_TOUCH), steps=steps, seed=seed),
            [episodes[i] for i in sorted(picked)],
            space,
            run,
            logging.info,
            init=ssl,
        )
        return
    senses = VARIANTS[variant] or cfg.model.senses
    size = 128 if variant.endswith("128") else 64
    model = sized(replace(cfg.model, senses=senses), size)
    train_supervised(
        SupervisedConfig(model=model, steps=steps, seed=seed),
        files("pretrain"),
        space,
        run,
        logging.info,
    )


def sized(model: ModelConfig, size: int) -> ModelConfig:
    """The model at a vision size (patch 8 at 64 px, 16 at 128 px: 64 tokens either way)."""
    return replace(model, vision_size=size, vision_patch=size // 8)


def loader_for(model: ModelConfig) -> LoaderConfig:
    """Window cutting and view size implied by a model configuration."""
    return LoaderConfig(steps=model.steps, substeps=model.substeps, view_size=model.vision_size)


def floor(seed: int, senses: tuple[str, ...], size: int = 64) -> SensoryLayer:
    """The tiny encoder at random initialisation (vision at ``size`` px)."""
    cfg = base_config()
    torch.manual_seed(seed)
    space = build_space(files("pretrain"), cfg.data.stats_episodes)
    model = sized(replace(cfg.model, senses=senses), size)
    return SensoryLayer(SensoryModel(model, space), space, loader_for(model))


def save_predictions(name: str, pred: Predictions) -> None:
    """Out-of-fold predictions of an encoder or feature set (for bootstrap and pairing)."""
    out = ROOT / "preds"
    out.mkdir(parents=True, exist_ok=True)
    edges = np.array([[b.start, b.stop] for b in pred.blocks])
    np.savez(
        out / f"{name}.npz", pred=pred.pred, target=pred.target, groups=pred.groups, edges=edges
    )


def load_predictions(name: str) -> Predictions:
    """Inverse of ``save_predictions``."""
    raw = np.load(ROOT / "preds" / f"{name}.npz")
    blocks = [slice(int(a), int(b)) for a, b in raw["edges"]]
    return Predictions(raw["pred"], raw["target"], blocks, raw["groups"])


def teacher_layers() -> dict[str, SensoryLayer]:
    """The SSL arms of PR #20 (2,000-step checkpoints)."""
    layers: dict[str, SensoryLayer] = {}
    for arm in ("c-none", "b-teacher", "d-placebo"):
        for seed in range(6):
            path = Path(f"runs/teacher/{arm}-s{seed}/ckpt-2000.pt")
            if path.exists():
                layers[f"{arm}-s{seed}"] = load_layer(path)
    return layers


def encoder_layers() -> dict[str, SensoryLayer]:
    """Every encoder to probe: floors, the supervised controls, and the SSL arms of PR #20."""
    cfg = base_config()
    layers: dict[str, SensoryLayer] = {}
    for seed in range(5):
        layers[f"floor-vp-s{seed}"] = floor(seed, NO_TOUCH)
        layers[f"floor-all-s{seed}"] = floor(seed, cfg.model.senses)
    for seed in range(5):
        layers[f"floor-vp128-s{seed}"] = floor(seed, NO_TOUCH, 128)
        layers[f"floor-all128-s{seed}"] = floor(seed, cfg.model.senses, 128)
    for ckpt in sorted((ROOT / "runs").glob("*/ckpt.pt")):
        layers[ckpt.parent.name] = load_layer(ckpt)
    for run in sorted(ROOT128.glob("*-s?")):
        if (run / "ckpt-2000.pt").exists():
            layers[run.name] = load_layer(run / "ckpt-2000.pt")
    for seed in range(2):  # the all-senses pooled-SIGReg encoders of the decisive test (64 px)
        path = Path(f"runs/decisive/sigreg-s{seed}/ckpt-2000.pt")
        if path.exists():
            layers[f"all64-s{seed}"] = load_layer(path)
    layers.update(teacher_layers())
    return layers


def probe_all() -> None:
    """Out-of-fold predictions and cached windows of every feature set and encoder."""
    data = load_feature_data()
    for name in data.features:
        if not (ROOT / "preds" / f"feat-{name}.npz").exists():
            save_predictions(f"feat-{name}", predict_probes(data, name))
    (ROOT / "states").mkdir(exist_ok=True)
    for name, layer in encoder_layers().items():
        states, preds = ROOT / "states" / f"{name}.npz", ROOT / "preds" / f"enc-{name}.npz"
        if states.exists() and preds.exists():
            continue
        data_enc = collect(layer, files("heldout"), STRIDE, (), raw=False)
        np.savez(
            states,
            groups=data_enc.groups,
            x_state=data_enc.features["state"],
            **{f"y_{k}": v for k, v in data_enc.labels.items()},
        )
        if not preds.exists():
            save_predictions(f"enc-{name}", predict_probes(data_enc, "state"))
        logging.info("probed %s", name)


# ---------------------------------------------------------------------------------------------
# report


def ci(values: list[dict[str, float]], key: str = "mean") -> tuple[float, float]:
    """95% percentile interval of a score over bootstrap draws."""
    arr = np.array([v[key] for v in values])
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def boot(pred: Predictions) -> list[dict[str, float]]:
    """Bootstrap over episodes, fixed seed (equal seeds pair two encoders on one episode set)."""
    return pred.bootstrap(DRAWS, np.random.default_rng(0))


def row(name: str, names: list[str], kind: str) -> str:
    """One report row: probe mean (95% CI over episodes) and the terms, averaged over ``names``."""
    preds = [load_predictions(f"{kind}-{n}") for n in names]
    scores = [p.scores() for p in preds]
    means = np.array([s["mean"] for s in scores])
    cis = [ci(boot(p)) for p in preds]
    spread = f", seed sd {means.std():.2f}" if len(means) > 1 else ""
    lo, hi = np.mean([c[0] for c in cis]), np.mean([c[1] for c in cis])
    cell = f"{means.mean():.2f} [{lo:.2f}, {hi:.2f}]"
    terms = " | ".join(f"{np.mean([s[t] for s in scores]):.2f}" for t in TERMS)
    return f"| {name} | {len(names)} | {cell}{spread} | {terms} |"


def paired(
    a: list[str], b: list[str], kind_a: str, kind_b: str
) -> tuple[float, float, float, float]:
    """Probe-mean difference a - b (seed-averaged): point, paired-bootstrap SE, 95% CI bounds."""
    pa = [load_predictions(f"{kind_a}-{n}") for n in a]
    pb = [load_predictions(f"{kind_b}-{n}") for n in b]
    da = np.mean([[d["mean"] for d in boot(p)] for p in pa], axis=0)
    db = np.mean([[d["mean"] for d in boot(p)] for p in pb], axis=0)
    diff = da - db
    point = float(
        np.mean([p.scores()["mean"] for p in pa]) - np.mean([p.scores()["mean"] for p in pb])
    )
    return (
        point,
        float(diff.std()),
        float(np.percentile(diff, 2.5)),
        float(np.percentile(diff, 97.5)),
    )  # type: ignore[return-value]


def seeds(prefix: str) -> list[str]:
    """Names of the seeds of an encoder family that have predictions."""
    found = sorted((ROOT / "preds").glob(f"enc-{prefix}-s?.npz"))
    return [p.stem.removeprefix("enc-") for p in found]


def report() -> None:
    """Print the control table, the gaps, the pairing and the minimum detectable difference."""
    print("Probe mean (95% bootstrap CI over the 20 held-out episodes) and terms\n")  # noqa: T201
    print("| control | n | probe mean [CI] | " + " | ".join(TERMS) + " |")  # noqa: T201
    print("|---|---|---|" + "---|" * len(TERMS))  # noqa: T201
    rows = [
        ("floor, V+P (random init)", seeds("floor-vp"), "enc"),
        ("floor, all senses", seeds("floor-all"), "enc"),
        ("SSL c-none (V+P)", seeds("c-none"), "enc"),
        ("SSL b-teacher (V+P)", seeds("b-teacher"), "enc"),
        ("SSL d-placebo (V+P)", seeds("d-placebo"), "enc"),
        ("supervised ceiling, V+P", seeds("sup-vp"), "enc"),
        ("supervised ceiling, all senses", seeds("sup-all"), "enc"),
        ("floor, V+P, 128 px", seeds("floor-vp128"), "enc"),
        ("SSL V+P pooled SIGReg, 128 px", seeds("c128"), "enc"),
        ("supervised ceiling, V+P, 128 px", seeds("sup-vp128"), "enc"),
        ("floor, all senses, 128 px", seeds("floor-all128"), "enc"),
        ("SSL all senses pooled SIGReg, 64 px (decisive test)", seeds("all64"), "enc"),
        ("SSL all senses pooled SIGReg, 128 px", seeds("a128"), "enc"),
        ("fine-tuned from scratch, 4 labelled episodes", seeds("ft-random-k4"), "enc"),
        ("fine-tuned from SSL (c-none), 4 labelled episodes", seeds("ft-ssl-k4"), "enc"),
        ("fine-tuned from scratch, 16 labelled episodes", seeds("ft-random-k16"), "enc"),
        ("fine-tuned from SSL (c-none), 16 labelled episodes", seeds("ft-ssl-k16"), "enc"),
    ]
    for name, names, kind in rows:
        if names:
            print(row(name, names, kind))  # noqa: T201
    for name in (
        "dino",
        "dino+proprio",
        "proprio",
        "raw-vp-pca32",
        "raw-vp-pca128",
        "raw-nonvisual-pca32",
        "raw-nonvisual-pca128",
        "raw-all-pca32",
        "raw-all-pca128",
    ):
        if (ROOT / "preds" / f"feat-{name}.npz").exists():
            print(row(name, [name], "feat"))  # noqa: T201
    gaps_and_power()


def gaps_and_power() -> None:
    """Floor-to-ceiling gaps, DINO against the floor, and the minimum detectable difference."""
    print("\nGaps and pairing (difference of probe means, paired bootstrap over episodes)\n")  # noqa: T201
    pairs = [
        ("supervised V+P - floor V+P", seeds("sup-vp"), seeds("floor-vp"), "enc", "enc"),
        ("supervised all - floor all", seeds("sup-all"), seeds("floor-all"), "enc", "enc"),
        ("dino+proprio - floor V+P", ["dino+proprio"], seeds("floor-vp"), "feat", "enc"),
        ("dino - floor V+P", ["dino"], seeds("floor-vp"), "feat", "enc"),
        ("dino+proprio - proprio", ["dino+proprio"], ["proprio"], "feat", "feat"),
        ("c-none - floor V+P", seeds("c-none"), seeds("floor-vp"), "enc", "enc"),
        ("b-teacher - c-none", seeds("b-teacher"), seeds("c-none"), "enc", "enc"),
        ("128 px - 64 px, SSL V+P", seeds("c128"), seeds("c-none"), "enc", "enc"),
        ("128 px - 64 px, floor V+P", seeds("floor-vp128"), seeds("floor-vp"), "enc", "enc"),
        ("128 px - 64 px, supervised V+P", seeds("sup-vp128"), seeds("sup-vp"), "enc", "enc"),
        ("128 px - 64 px, SSL all senses", seeds("a128"), seeds("all64"), "enc", "enc"),
        ("SSL V+P 128 px - floor V+P 128 px", seeds("c128"), seeds("floor-vp128"), "enc", "enc"),
        ("SSL all 128 px - floor all 128 px", seeds("a128"), seeds("floor-all128"), "enc", "enc"),
        (
            "fine-tune from SSL - from scratch, k = 4",
            seeds("ft-ssl-k4"),
            seeds("ft-random-k4"),
            "enc",
            "enc",
        ),
        (
            "fine-tune from SSL - from scratch, k = 16",
            seeds("ft-ssl-k16"),
            seeds("ft-random-k16"),
            "enc",
            "enc",
        ),
        (
            "fine-tune from scratch k = 16 - k = 4",
            seeds("ft-random-k16"),
            seeds("ft-random-k4"),
            "enc",
            "enc",
        ),
    ]
    for name, a, b, ka, kb in pairs:
        if a and b:
            point, se, lo, hi = paired(a, b, ka, kb)
            print(f"{name}: {point:+.3f}  (paired SE {se:.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}])")  # noqa: T201
    floors = seeds("floor-vp")
    if len(floors) >= 2:
        print("\nVariance components\n")  # noqa: T201
        ses = [
            paired([floors[i]], [floors[j]], "enc", "enc")[1]
            for i in range(len(floors))
            for j in range(i)
        ]
        se = float(np.mean(ses))
        print(f"episode sampling: paired SE of a difference between two floor seeds {se:.3f}")  # noqa: T201
        seed_sd = float(
            np.sqrt(
                np.mean(
                    [
                        np.var(
                            [load_predictions(f"enc-{n}").scores()["mean"] for n in seeds(prefix)],
                            ddof=1,
                        )
                        for prefix in ("floor-vp", "c-none", "b-teacher", "d-placebo")
                        if len(seeds(prefix)) > 1
                    ]
                )
            )
        )
        print(
            f"training seeds: sd of the probe mean (pooled over floors and SSL arms) {seed_sd:.3f}"
        )  # noqa: T201
        print("\nMinimum detectable difference (80% power, 5% two-sided: 2.8 x SE)\n")  # noqa: T201
        print("| seeds per arm | 20 held-out episodes | 80 held-out episodes |")  # noqa: T201
        print("|---|---|---|")  # noqa: T201
        for n in (1, 3, 6, 12, 24):
            cells = [
                f"{2.8 * np.sqrt(se**2 * 20 / eps + 2 * seed_sd**2 / n):.3f}" for eps in (20, 80)
            ]
            print(f"| {n} | " + " | ".join(cells) + " |")  # noqa: T201


SSL128 = {"c128": NO_TOUCH, "a128": None}
STEPS128 = (250, 500, 1000, 2000)
ROOT128 = Path("runs/controls128")


def train_ssl128(arm: str, seed: int, steps: int) -> None:
    """Pretrain a pooled-SIGReg encoder, vision at 128 px (V+P: ``c128``, all senses: ``a128``)."""
    cfg = base_config()
    senses = SSL128[arm] or cfg.model.senses
    model = sized(replace(cfg.model, senses=senses), 128)
    tc = TrainConfig(
        steps=steps,
        batch_size=16,
        lr=cfg.train.lr,
        log_every=50,
        eval_windows=0,
        num_workers=2,
        out_dir=str(ROOT128),
        save_steps=tuple(s for s in STEPS128 if s <= steps),
    )
    data = replace(cfg.data, paths=tuple(str(p) for p in files("pretrain")))
    train(
        RunConfig(f"{arm}-s{seed}", seed, data, replace(cfg.augment, cache_episodes=4), model, tc)
    )


def probe_json(layer: SensoryLayer, all_senses: bool) -> dict[str, object]:
    """Held-out probe scores (``heldout`` = the encoder's natural input, ``vp`` = touch and audio
    masked) and held-out loss of one encoder, in the format ``plot_runs.py`` reads."""
    out: dict[str, object] = {}
    masked = tuple(s for s in ("touch", "audio") if s in layer.model.senses)
    for key, drop in (("vp", masked), ("heldout", () if all_senses else masked)):
        if key not in out:
            out[key] = predict_probes(
                collect(layer, files("heldout"), STRIDE, drop, raw=False), "state"
            ).scores()
    return out | heldout_loss(layer, files("heldout"))


def probe_ssl128() -> None:
    """Per-checkpoint probe JSONs of the 128 px runs and floors (``runs/controls128/probes``)."""
    out = ROOT128 / "probes"
    out.mkdir(parents=True, exist_ok=True)
    cfg = base_config()
    for arm, senses in SSL128.items():
        senses = senses or cfg.model.senses
        for seed in range(5):
            target = out / f"floor-{arm}-s{seed}.json"
            if not target.exists():
                target.write_text(json.dumps(probe_json(floor(seed, senses, 128), arm == "a128")))
    for run in sorted(ROOT128.glob("*-s?")):
        arm = run.name.rsplit("-s", 1)[0]
        for step in STEPS128:
            ckpt, target = run / f"ckpt-{step}.pt", out / f"{run.name}-n{step}.json"
            if ckpt.exists() and not target.exists():
                target.write_text(json.dumps(probe_json(load_layer(ckpt), arm == "a128")))
                logging.info("probed %s", target.name)


def cost_report() -> None:
    """CPU seconds per training step (fwd, bwd, optimiser) at 64 and 128 px, one process."""
    cfg = base_config()
    space = build_space(files("pretrain"), 16)
    for size in (64, 128):
        for label, senses in (("V+P", NO_TOUCH), ("all senses", cfg.model.senses)):
            model_cfg = sized(replace(cfg.model, senses=senses), size)
            torch.manual_seed(0)
            model = SensoryModel(model_cfg, space)
            dataset = SequenceDataset(
                files("pretrain")[:6], space, senses, loader_for(model_cfg), seed=0
            )
            batch = next(iter(DataLoader(dataset, batch_size=16)))
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            for i in range(7):
                if i == 2:
                    start = time.perf_counter()
                opt.zero_grad()
                model(batch)["loss"].backward()
                opt.step()
            step_s = (time.perf_counter() - start) / 5
            with FlopCounterMode(display=False) as counter:
                model(batch)["loss"].backward()
            print(  # noqa: T201
                f"{size} px, {label}: {step_s:.2f} s per step of 16 sequences "
                f"({16 / step_s:.1f} sequences/s), "
                f"{counter.get_total_flops() / 16 / 1e9:.1f} GFLOPs per sequence, "
                f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters"
            )


def export_probe_jsons() -> None:
    """``plot_runs.py`` files for the supervised and fine-tuned controls (final checkpoint only)."""
    out = ROOT / "probes_json"
    out.mkdir(exist_ok=True)
    for path in sorted((ROOT / "preds").glob("enc-*.npz")):
        name = path.stem.removeprefix("enc-")
        match = re.match(r"^(sup-vp|sup-all|ft-\w+-k\d+)(128)?-s(\d)$", name)
        if match:
            step = 300 if name.startswith("ft-") else 1000
            (out / f"{name}-n{step}.json").write_text(
                json.dumps({"heldout": load_predictions(f"enc-{name}").scores()})
            )
        elif re.match(r"^floor-(vp|all)(128)?-s\d$", name):
            (out / f"{name}.json").write_text(
                json.dumps({"heldout": load_predictions(f"enc-{name}").scores()})
            )


SHOTS = (2, 4, 8, 19)
FEW_SHOT = {
    "floor V+P": ("enc", "floor-vp"),
    "SSL c-none": ("enc", "c-none"),
    "SSL b-teacher": ("enc", "b-teacher"),
    "supervised V+P": ("enc", "sup-vp"),
    "floor all senses": ("enc", "floor-all"),
    "supervised all senses": ("enc", "sup-all"),
    "dino+proprio": ("feat", "dino+proprio"),
    "proprio": ("feat", "proprio"),
    "raw-nonvisual-pca128": ("feat", "raw-nonvisual-pca128"),
    "raw-vp-pca128": ("feat", "raw-vp-pca128"),
}


def load_states(name: str) -> ProbeData:
    """Cached windows of an encoder (``probe`` writes them)."""
    raw = np.load(ROOT / "states" / f"{name}.npz")
    return ProbeData(
        {"state": raw["x_state"]},
        {k[2:]: raw[k] for k in raw.files if k.startswith("y_")},
        raw["groups"],
    )


def few_shot_report() -> None:
    """Probe mean when only k labelled held-out episodes are available (ridge, no search)."""
    feats = load_feature_data()
    print("\nFew-shot probes: probe mean with k labelled episodes (mean over seeds and 4 draws)\n")  # noqa: T201
    print("| control | " + " | ".join(f"k = {k}" for k in SHOTS) + " |")  # noqa: T201
    print("|---|" + "---|" * len(SHOTS))  # noqa: T201
    for label, (kind, key) in FEW_SHOT.items():
        if kind == "feat":
            sources = [(feats, key)]
        else:
            sources = [
                (load_states(n), "state")
                for n in seeds(key)
                if (ROOT / "states" / f"{n}.npz").exists()
            ]
        if not sources:
            continue
        cells = []
        for k in SHOTS:
            vals = [
                predicted.scores()["mean"]
                for data, feature in sources
                for d in range(4)
                for predicted in [few_shot_predictions(data, feature, k, np.random.default_rng(d))]
            ]
            cells.append(f"{np.mean(vals):.2f} +- {np.std(vals):.2f}")
        print(f"| {label} | " + " | ".join(cells) + " |")  # noqa: T201


def alpha_report() -> None:
    """Probe mean against the fixed ridge strength (all other episodes as training, no search)."""
    feats = load_feature_data()
    mults = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    print("\nRidge-strength sensitivity: probe mean at a fixed strength (x number of features)\n")  # noqa: T201
    print("| control | search (inner CV) | " + " | ".join(f"{m:g}" for m in mults) + " |")  # noqa: T201
    for label, (kind, key) in FEW_SHOT.items():
        if kind == "feat":
            sources = [(feats, key)]
            searched = [load_predictions(f"feat-{key}").scores()["mean"]]
        else:
            names = [n for n in seeds(key) if (ROOT / "states" / f"{n}.npz").exists()]
            sources = [(load_states(n), "state") for n in names]
            searched = [load_predictions(f"enc-{n}").scores()["mean"] for n in names]
        if not sources:
            continue
        cells = []
        for m in mults:
            means = [
                few_shot_predictions(d, f, 19, np.random.default_rng(0), m).scores()["mean"]
                for d, f in sources
            ]
            cells.append(f"{np.mean(means):.2f}")
        print(f"| {label} | {np.mean(searched):.2f} | " + " | ".join(cells) + " |")  # noqa: T201


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "features",
            "train",
            "train-ssl",
            "probe",
            "probe-ssl",
            "report",
            "fewshot",
            "alpha",
            "cost",
            "export",
        ),
    )
    parser.add_argument("--variant", help="sup-vp, sup-all, or ft-{random,ssl}-k{K}")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.command == "features":
        compute_features()
    elif args.command == "train":
        train_variant(args.variant, args.seed, args.steps)
    elif args.command == "probe":
        probe_all()
    elif args.command == "train-ssl":
        train_ssl128(args.variant, args.seed, args.steps)
    elif args.command == "probe-ssl":
        probe_ssl128()
    elif args.command == "cost":
        cost_report()
    elif args.command == "export":
        export_probe_jsons()
    elif args.command == "fewshot":
        few_shot_report()
    elif args.command == "alpha":
        alpha_report()
    else:
        report()


if __name__ == "__main__":
    main()
