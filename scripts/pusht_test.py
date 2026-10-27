"""B-pusht-0: our sensory layer against LeWorldModel on Push-T, at a matched tiny size (training).

Pre-registration: ``docs/model.md``, "B-pusht-0". Data: phd-a's converter (``sensorium.pusht``,
``scripts/pusht_convert.py``): one episode per source episode in ``data/pusht/pusht/`` and a 64 px
HDF5 in LeWM's layout. Every arm trains on the same fixed subset of ``N_SUBSET`` episodes and on
LeWM's own clip split of it, for ``CLIPS`` training clips (``STEPS`` x ``BATCH``).

Our arms (``ARMS``): A0 on pixels, A0 on pixels + the agent's proprio, fix1 on pixels. Our sequences
are 4 model steps of 5 env steps (3 context frames + 1 target, the action chunk = LeWM's action
block of 5), seen at 64 px over the full frame (no crop, no jitter), from a 64 px frame cache.

LeWM-tiny is LeWM's own ``train.py`` with ``LEWM_OVERRIDES`` (``lewm-cmd`` prints the command).

    uv run python scripts/pusht_test.py subset           # the fixed episode subset
    uv run python scripts/pusht_test.py weights          # LeWM's clip split -> our start weights
    uv run python scripts/pusht_test.py cache --jobs 30  # the 64 px frame cache of the subset
    uv run python scripts/pusht_test.py params           # parameter counts of every arm
    uv run python scripts/pusht_test.py train --arm a0-pix --seed 0
    uv run python scripts/pusht_test.py lewm-cmd --seed 0
"""

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np

from sensorium import pusht
from sensorium.model.data import spacing
from sensorium.model.frame_cache import build_cache
from sensorium.model.layout import ChannelSpace, Header
from sensorium.model.pretrain import AugmentConfig, DataConfig, RunConfig, TrainConfig, train
from sensorium.model.sensory_layer import ModelConfig, SensoryModel, count_parameters

ROOT = Path("runs/pusht")
DATA = Path("data/pusht")
CACHE = Path("data/pusht_cache64")
N_SUBSET = 2000
SUBSET_SEED = 0
STEPS = 16_000
BATCH = 128
CLIPS = STEPS * BATCH
"""Training clips seen by every run (2,048,000)."""
SEEDS = (0, 1, 2)
CKPTS = (0, 1000, 4000, STEPS)
SIZE = 64
SUBSTEPS = 5  # env steps per model step: LeWM's frame-skip
SEQ_STEPS = 4  # model steps per sequence: LeWM's history 3 + 1 prediction

BASE = ModelConfig(
    dim=128,
    fusion_depth=2,
    predictor_depth=2,
    heads=4,
    vision_size=SIZE,
    vision_patch=8,
    vision_dim=64,
    vision_depth=2,
    vision_heads=2,
    vision_tokens=4,
    steps=SEQ_STEPS,
    substeps=SUBSTEPS,
    sigreg_proj=256,
    sigreg_pool_time=False,  # batch 128: SIGReg per time step, as LeWM
)
"""Our ``cpu_tiny`` widths, with Push-T's sequence geometry."""
ARMS: dict[str, dict[str, object]] = {
    "a0-pix": {"senses": ("vision",), "recon_weight": 0.0},
    "a0-pix-agent": {"senses": ("vision", "proprio_effector"), "recon_weight": 0.0},
    "fix1-pix": {"senses": ("vision",), "recon_weight": 0.5, "recon_input": "masked"},
}
"""Our arms (ii), (iii), (iv) of the pre-registration."""

LEWM_DATASET = "pusht_subset64"
"""The 64 px HDF5 of the subset in LeWM's layout (``$STABLEWM_HOME/<name>.h5``)."""
LEWM_OVERRIDES = (
    "data=pusht",
    f"data.dataset.name={LEWM_DATASET}",
    f"img_size={SIZE}",
    "embed_dim=128",
    "model.encoder.patch_size=8",
    "+model.encoder.hidden_size=128",
    "+model.encoder.num_hidden_layers=3",
    "+model.encoder.num_attention_heads=4",
    "+model.encoder.intermediate_size=512",
    "model.predictor.depth=2",
    "model.predictor.heads=4",
    "model.predictor.dim_head=32",
    "model.predictor.mlp_dim=512",
    "model.projector.hidden_dim=256",
    "model.pred_proj.hidden_dim=256",
    "trainer.accelerator=cpu",
    "trainer.precision=32",
    f"+trainer.max_steps={STEPS}",
    "trainer.max_epochs=1000",
    f"loader.batch_size={BATCH}",
    "num_workers=1",
    "loader.prefetch_factor=2",
    "loader.pin_memory=false",
    "++wandb.enabled=false",
)
"""LeWM-tiny: LeWM's ``train.py`` (le-wm ``8edfeb3``) at our tiny size on CPU. Everything not
listed is LeWM's: SIGReg 0.09 (1,024 projections), history 3, frame-skip 5, AdamW 5e-5 and its
schedule, and its 90/10 clip split, seeded by ``cfg.seed`` = 3072 (kept, so the split is LeWM's;
the run seed is set separately, see ``lewm_command``)."""


def lewm_tiny_parameters() -> dict[str, int]:
    """LeWM-tiny's parameter count by part, from le-wm ``8edfeb3``'s modules with
    ``LEWM_OVERRIDES`` (HF ViT encoder; ``module.MLP``, ``Embedder``, ``ARPredictor``)."""
    d, layers, inter, patch, act = 128, 3, 512, 8, 2 * SUBSTEPS
    vit_layer = 2 * 2 * d + 3 * (d * d + d) + (d * d + d) + (d * inter + inter) + (inter * d + d)
    encoder = (3 * patch * patch * d + d) + d + ((SIZE // patch) ** 2 + 1) * d + layers * vit_layer
    encoder += 2 * d  # final layer norm

    def mlp(hidden: int) -> int:  # Linear, BatchNorm1d, GELU, Linear
        return (d * hidden + hidden) + 2 * hidden + (hidden * d + d)

    heads, dim_head, mlp_dim, depth = 4, 32, 512, 2
    inner = heads * dim_head
    block = (2 * d + d * 3 * inner + inner * d + d) + (
        2 * d + d * mlp_dim + mlp_dim + mlp_dim * d + d
    )
    block += d * 6 * d + 6 * d  # AdaLN-zero modulation
    predictor = depth * block + 2 * d + 3 * d  # final norm, positional embedding (history 3)
    action = (act * act + act) + (act * 4 * d + 4 * d) + (4 * d * d + d)
    parts = {
        "encoder": encoder,
        "projector": mlp(256),
        "pred_proj": mlp(256),
        "action": action,
        "predictor": predictor,
    }
    return parts | {"total": sum(parts.values())}


def episode_path(index: int) -> Path:
    """Our episode of a source episode (phd-a's converter layout)."""
    return DATA / pusht.BODY / f"{pusht.BODY}-{index:05d}.npz"


def episode_count() -> int:
    """Converted episodes in ``DATA``."""
    return len(list((DATA / pusht.BODY).glob(f"{pusht.BODY}-*.npz")))


def subset_indices(n_episodes: int) -> np.ndarray:
    """The fixed training subset: ``N_SUBSET`` source episodes drawn with ``rng(SUBSET_SEED)``,
    in increasing order (the order of LeWM's subset file and of its clip split)."""
    rng = np.random.default_rng(SUBSET_SEED)
    return np.sort(rng.choice(n_episodes, N_SUBSET, replace=False))


def episode_length(path: Path) -> int:
    """Steps of an episode file (read from its action column only)."""
    with np.load(path) as raw:
        return int(raw["action"].shape[0])


def our_start(lewm_start: int, dt: float = pusht.DT) -> int:
    """The first env step of our sequence whose windows end on LeWM's clip frames ``t, t + 5,
    ...``: our window of ``SUBSTEPS`` observations ends on the frame, so it starts ``SUBSTEPS - 1``
    spacings earlier. Negative for LeWM's first clips of an episode (we cannot take those)."""
    return lewm_start - (SUBSTEPS - 1) * spacing(dt)


def start_weights(
    names: list[str], lengths: list[int], train_clips: list[tuple[int, int]]
) -> dict[str, list[float]]:
    """Per episode file, weight 1 on each of our sequence starts whose windows end on a LeWM
    training clip's frames, 0 elsewhere (``SequenceDataset``'s ``start_weights``). ``train_clips``
    are ``(position in names, LeWM start)``."""
    span = SEQ_STEPS * SUBSTEPS * spacing(pusht.DT)
    weights = [np.zeros(max(n - span + 1, 0)) for n in lengths]
    for ep, t in train_clips:
        s = our_start(t)
        if 0 <= s < len(weights[ep]):
            weights[ep][s] = 1.0
    return {name: w.tolist() for name, w in zip(names, weights, strict=True)}


def subset() -> Path:
    """Write ``runs/pusht/subset.json`` (the source episode indices of the training subset)."""
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "subset.json"
    path.write_text(json.dumps(subset_indices(episode_count()).tolist()))
    return path


def subset_paths() -> list[Path]:
    """Our episode files of the subset (``subset.json``), in order."""
    return [episode_path(int(i)) for i in json.loads((ROOT / "subset.json").read_text())]


def weights() -> None:
    """LeWM's clip split over the subset (``pusht.clip_split``, as LeWM's ``train.py`` computes
    it on the subset file) -> ``clips.json`` and our ``start_weights.json``."""
    paths = subset_paths()
    lengths = [episode_length(p) for p in paths]
    split = pusht.clip_split(lengths)
    (ROOT / "clips.json").write_text(json.dumps(split))
    w = start_weights([p.name for p in paths], lengths, split["train"])
    taken = sum(sum(x) for x in w.values())
    (ROOT / "start_weights.json").write_text(json.dumps({"weights": w}))
    logging.info(
        "%d LeWM training clips, %d usable by our loader (the rest start < %d steps in)",
        len(split["train"]),
        taken,
        (SUBSTEPS - 1) * spacing(pusht.DT),
    )


def cache(jobs: int) -> None:
    """The 64 px frame cache of the subset (``vision.resize`` of the full stored frame, the same
    function as LeWM's 64 px file)."""
    build_cache(subset_paths(), CACHE, SIZE, jobs)


def push_t_header() -> Header:
    """The header of every Push-T episode (for parameter counts without data)."""
    return Header(
        body=pusht.BODY,
        dt=pusht.DT,
        proprio_names=tuple(pusht.PROPRIO_NAMES),
        touch_names=(),
        action_names=tuple(pusht.ACTION_NAMES),
    )


def model_config(arm: str) -> ModelConfig:
    """One arm's model configuration."""
    return replace(BASE, **ARMS[arm])  # type: ignore[arg-type]


def params() -> dict[str, dict[str, int]]:
    """Parameter counts of every arm (ours: the planning model, the reconstruction decoder
    apart) and LeWM-tiny's; printed as JSON."""
    space = ChannelSpace.from_headers([push_t_header()])
    out = {}
    for arm in ARMS:
        model = SensoryModel(model_config(arm), space)
        counts = count_parameters(model)
        recon = sum(p.numel() for p in model.recon.parameters()) if model.recon else 0
        out[arm] = {"planning model": counts["total"] - recon, "reconstruction decoder": recon}
    out["lewm-tiny"] = {"planning model": lewm_tiny_parameters()["total"]}
    print(json.dumps(out, indent=1))  # noqa: T201
    return out


def run_config(arm: str, seed: int) -> RunConfig:
    """One of our runs: the subset, LeWM's training clips, the 64 px cache, ``STEPS`` x
    ``BATCH``."""
    data = DataConfig(
        paths=tuple(str(p) for p in subset_paths()),
        event_index=str(ROOT / "start_weights.json"),
    )
    augment = AugmentConfig(
        crop=1.0, touch_noise=False, cache_episodes=8, frame_cache=str(CACHE), reuse=1
    )
    tc = TrainConfig(
        steps=STEPS,
        batch_size=BATCH,
        log_every=50,
        save_steps=CKPTS,
        num_workers=1,
        out_dir=str(ROOT / "runs"),
        eval_windows=0,
        threads=2,
    )
    return RunConfig(f"{arm}-s{seed}", seed, data, augment, model_config(arm), tc)


def lewm_command(seed: int) -> list[str]:
    """LeWM-tiny's training command for one seed. LeWM's ``cfg.seed`` (3072) seeds its clip split
    and loader order, so it is kept; the run seed is torch's global seed, set by
    ``scripts/lewm_launch.py`` before LeWM's ``run()``."""
    name = f"lewm-tiny-s{seed}"
    return [
        "python",
        "lewm_launch.py",
        f"--run-seed={seed}",
        *LEWM_OVERRIDES,
        f"output_model_name={name}",
        f"subdir={name}",
    ]


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "command", choices=("subset", "weights", "cache", "params", "train", "lewm-cmd")
    )
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.command == "subset":
        subset()
    elif args.command == "weights":
        weights()
    elif args.command == "cache":
        cache(args.jobs)
    elif args.command == "params":
        params()
    elif args.command == "train":
        if args.arm is None:
            parser.error("train needs --arm")
        train(run_config(args.arm, args.seed))
    else:
        print(" ".join(lewm_command(args.seed)))  # noqa: T201


if __name__ == "__main__":
    main()
