"""Positive controls for the probe battery: can the yardstick see a good representation at all?

* **Supervised ceiling.** The same encoder trained *directly on the probe labels* on the pretraining
  episodes (``train_supervised``), then probed on held-out episodes like any other encoder. This
  is the one place where training reads the labels, on purpose; ``pretrain.train`` never does, and
  nothing here is used by it.
* **Off-the-shelf vision.** Frozen DINOv2 ViT-S/14 on the centre-cropped frames (``dino_features``).
* **Raw inputs, done right.** PCA-reduced raw inputs, with the PCA fitted on pretraining episodes
  (``RawFeatures``), instead of a ridge on thousands of raw channels.

``build_probe_data`` assembles the same windows, labels and episode groups as
``probes.collect`` for any set of feature functions, so every control goes through the same
leave-one-episode-out ridge probes and the same bootstrap.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from sensorium.episode import Episode
from sensorium.model.data import (
    Loaded,
    LoaderConfig,
    SequenceDataset,
    build,
    load_episode,
    sequence_with_ends,
    spacing,
)
from sensorium.model.events import load_start_weights
from sensorium.model.layout import ChannelSpace
from sensorium.model.pretrain import RunConfig, TrainConfig, lr_at, save_checkpoint
from sensorium.model.probes import (
    BINARY,
    CONTINUOUS,
    HIDDEN,
    Predictions,
    ProbeData,
    hand_object_touch,
    hidden_from,
    hidden_properties,
    hidden_window_labels,
    privileged_labels,
    ridge_predict,
    window_labels,
)
from sensorium.model.sensory_layer import ModelConfig, SensoryModel
from sensorium.vision import View, eval_view

DINO_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
"""``facebookresearch/dinov2`` commit of the hub code (code and weights: Apache-2.0)."""
DINO_VIEW = View(size=224, crop=0.875)
"""DINOv2's evaluation transform on our 256 px frames: resize to 256, centre-crop 224."""
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PROPRIO = ("proprio_body", "vestibular", "proprio_effector")


# ---------------------------------------------------------------------------------------------
# Supervised ceiling


class LabelledSequences(SequenceDataset):
    """Training sequences with the probe labels of their windows attached (``y_<label>``).

    Reads the episodes' labels (hence the full ``Episode.load``). Only the supervised control uses
    it. With ``hidden``, also ``y_hidden_*`` (``probes.HIDDEN``, NaN where undefined).
    """

    def __init__(self, *args: object, hidden: bool = False, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.hidden = hidden
        self._labels: dict[Path, dict[str, np.ndarray]] = {}
        self._hidden: dict[Path, tuple[np.ndarray, np.ndarray] | None] = {}

    def load(self, path: Path) -> Loaded:
        """Decode an episode and remember its labels."""
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        episode = Episode.load(path)
        self._labels[path] = privileged_labels(episode)
        if self.hidden:
            try:
                self._hidden[path] = (hidden_properties(episode), hand_object_touch(episode))
            except KeyError:
                self._hidden[path] = None
        loaded = load_episode(episode, self.space, self.senses)
        self._cache[path] = loaded
        while len(self._cache) > self.cfg.cache_episodes:
            self._cache.popitem(last=False)
        return loaded

    def labelled(self, i: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """One sequence from episode ``i`` with ``y_*`` labels per window."""
        path = self.paths[i]
        loaded = self.load(path)
        start = self._start(i, rng)
        seq, ends = sequence_with_ends(
            loaded, self.space, self.senses, self.cfg, rng, self.train, start=start
        )
        labels = window_labels(
            self._labels[path], ends, self.cfg.substeps, spacing(loaded.header.dt)
        )
        if self.hidden:
            known = self._hidden[path]
            labels |= (
                hidden_from(known[0], known[1], ends, self.cfg.substeps, spacing(loaded.header.dt))
                if known
                else {n: np.full(len(ends), np.nan) for n in HIDDEN}
            )
        return seq | {f"y_{k}": v.astype(np.float32) for k, v in labels.items()}

    def sample(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Draw one labelled sequence."""
        return self.labelled(self._pick(rng), rng)

    def __iter__(self):  # type: ignore[no-untyped-def]
        rng = np.random.default_rng([self.seed, 0])
        while True:
            i = self._pick(rng)
            for _ in range(self.cfg.reuse):
                yield self.labelled(i, rng)


def label_scale(paths: Sequence[Path], limit: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of the continuous targets (force, tip x 6) over some episodes' steps."""
    rows = []
    for path in list(paths)[:limit]:
        lab = privileged_labels(Episode.load(path))
        rows.append(np.concatenate([lab["log_force"][:, None], lab["tip"]], axis=1))
    stacked = np.concatenate(rows)
    return stacked.mean(0), np.maximum(stacked.std(0), 1e-6)


@dataclass(frozen=True)
class SupervisedConfig:
    """A supervised-ceiling run.

    Attributes:
        model: The encoder configuration (``senses`` decide what it sees).
        steps, batch_size, lr, weight_decay, warmup, grad_clip: Optimisation.
        seed: Random seed.
        workers: Loader processes.
        save_steps: Steps at which to also write ``ckpt-<step>.pt`` (for held-out score against
            steps).
        freeze_steps: For the first steps only the linear head is trained (the encoder is held
            fixed): linear-probe-then-fine-tune.
        encoder_lr_scale: The encoder's learning rate relative to the head's (after the freeze).
        hidden: Also train on the hidden-property labels.
        event_index: Optional start-weight file of ``sensorium.model.events`` (draw sequences
            toward event windows).
    """

    model: ModelConfig
    steps: int = 1000
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-3
    warmup: int = 20
    grad_clip: float = 1.0
    seed: int = 0
    workers: int = 2
    save_steps: tuple[int, ...] = ()
    freeze_steps: int = 0
    encoder_lr_scale: float = 1.0
    hidden: bool = False
    event_index: str = ""


def hidden_scale(paths: Sequence[Path], limit: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of the hidden-property targets over the objects of some episodes."""
    rows = []
    for path in list(paths)[:limit]:
        try:
            props = hidden_properties(Episode.load(path))
        except KeyError:
            continue
        rows.append(np.stack([np.log(props[:, 0]), props[:, 1], np.log(props[:, 2])], axis=1))
    if not rows:
        return np.zeros(len(HIDDEN)), np.ones(len(HIDDEN))
    stacked = np.concatenate(rows)
    return stacked.mean(0), np.maximum(stacked.std(0), 1e-6)


def supervised_loss(
    heads: Tensor, batch: dict[str, Tensor], scale: tuple[Tensor, Tensor]
) -> Tensor:
    """BCE on the three contact labels plus MSE on the standardised force and tip targets."""
    bce = sum(
        nn.functional.binary_cross_entropy_with_logits(heads[..., i], batch[f"y_{n}"])
        for i, n in enumerate(BINARY)
    )
    cont = torch.cat([batch["y_log_force"][..., None], batch["y_tip"]], dim=-1)
    n = len(BINARY) + cont.shape[-1]
    mse = ((heads[..., len(BINARY) : n] - (cont - scale[0][:7]) / scale[1][:7]) ** 2).mean()
    loss = bce / len(BINARY) + mse
    if "y_hidden_log_mass" in batch:  # hidden-property targets, NaN where nothing is touched
        target = torch.stack([batch[f"y_{name}"] for name in HIDDEN], dim=-1)
        valid = torch.isfinite(target)
        # NaN targets are zeroed first: a NaN in the graph would give NaN gradients even if masked
        clean = torch.where(valid, target, torch.zeros_like(target))
        error = (heads[..., n:] - (clean - scale[0][7:]) / scale[1][7:]) ** 2
        if valid.any():
            loss = loss + error[valid].mean()
    return loss


def train_supervised(
    cfg: SupervisedConfig,
    paths: Sequence[Path],
    space: ChannelSpace,
    out: Path,
    log: Callable[[str], None],
    init: Path | None = None,
) -> Path:
    """Train the encoder and a linear head on the labels; returns the checkpoint path.

    With ``init`` (a checkpoint of ``pretrain.train``) the encoder starts from its encoder weights:
    the few-shot adaptation of a pretrained layer, to compare with training from scratch on the
    same labelled episodes.

    The checkpoint holds the encoder (and an unused predictor) in a ``SensoryModel``, so
    ``pretrain.load_layer`` and the probes treat it like any other encoder. The linear head and
    the target standardisation go next to it in ``head.pt`` (``head-<step>.pt`` at the save
    steps), for ``head_predictions``.
    """
    torch.manual_seed(cfg.seed)
    loader_cfg = LoaderConfig(
        steps=cfg.model.steps,
        substeps=cfg.model.substeps,
        view_size=cfg.model.vision_size,
        cache_episodes=4,
    )
    model = SensoryModel(replace(cfg.model, recon_weight=0.0), space)  # no reconstruction here
    if init is not None:
        saved = torch.load(init, weights_only=True)["model"]
        encoder = {
            k.removeprefix("encoder."): v for k, v in saved.items() if k.startswith("encoder.")
        }
        model.encoder.load_state_dict(encoder)
    head = nn.Linear(cfg.model.dim, len(BINARY) + 1 + 6 + (len(HIDDEN) if cfg.hidden else 0))
    mean, std = label_scale(paths)
    if cfg.hidden:
        h_mean, h_std = hidden_scale(paths)
        mean, std = np.concatenate([mean, h_mean]), np.concatenate([std, h_std])
    scale = (torch.as_tensor(mean, dtype=torch.float32), torch.as_tensor(std, dtype=torch.float32))
    weights = load_start_weights(Path(cfg.event_index)) if cfg.event_index else None
    dataset = LabelledSequences(
        paths,
        space,
        model.data_senses,
        loader_cfg,
        seed=cfg.seed,
        hidden=cfg.hidden,
        start_weights=weights,
    )
    batches = iter(
        DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.workers,
            persistent_workers=cfg.workers > 0,
            prefetch_factor=4 if cfg.workers else None,
        )
    )
    params = [*model.encoder.parameters(), *head.parameters()]
    opt = torch.optim.AdamW(
        [
            {"params": list(model.encoder.parameters())},
            {"params": list(head.parameters())},
        ],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    model.train()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "log.jsonl").open("w") as journal:
        for step in range(1, cfg.steps + 1):
            rate = lr_at(step - 1, _schedule(cfg))
            opt.param_groups[0]["lr"] = (
                0.0 if step <= cfg.freeze_steps else rate * cfg.encoder_lr_scale
            )
            opt.param_groups[1]["lr"] = rate
            batch = next(batches)
            loss = supervised_loss(head(model.embed(batch)), batch, scale)
            opt.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            if step % 50 == 0 or step == cfg.steps:
                value = float(loss.detach())
                journal.write(json.dumps({"step": step, "loss": value}) + "\n")
                log(f"step {step}/{cfg.steps} supervised loss {value:.4f}")
            if step in cfg.save_steps and step != cfg.steps:
                save_checkpoint(
                    out / f"ckpt-{step}.pt",
                    model,
                    space,
                    RunConfig(name=out.name, model=model.cfg),
                    step,
                )
                torch.save({"head": head.state_dict(), "scale": scale}, out / f"head-{step}.pt")
    save_checkpoint(
        out / "ckpt.pt", model, space, RunConfig(name=out.name, model=model.cfg), cfg.steps
    )
    torch.save({"head": head.state_dict(), "scale": scale}, out / "head.pt")
    return out / "ckpt.pt"


def head_predictions(data: ProbeData, head_path: Path) -> Predictions:
    """Held-out predictions of a supervised run's own linear head on encoded windows.

    ``data`` comes from ``probes.collect`` on the run's checkpoint (``state`` = the encoder's
    states, which is what the head was trained on); ``head_path`` is the ``head.pt`` written next
    to it. The contact columns are logits (AUROC is rank-based); force and tip are mapped back to
    label units. Only the labelled training episodes of the run fitted this readout, unlike the
    ridge probe, which is fitted on the held-out episodes themselves.
    """
    saved = torch.load(head_path, weights_only=True)
    weight = saved["head"]["weight"]
    head = nn.Linear(weight.shape[1], weight.shape[0])
    head.load_state_dict(saved["head"])
    mean, std = (np.asarray(t, dtype=np.float64) for t in saved["scale"])
    with torch.no_grad():
        out = head(torch.as_tensor(data.features["state"], dtype=torch.float32)).double().numpy()
    n = len(BINARY) + 7
    pred = np.concatenate([out[:, : len(BINARY)], out[:, len(BINARY) : n] * std[:7] + mean[:7]], 1)
    columns = [data.labels[k].reshape(len(pred), -1) for k in (*BINARY, *CONTINUOUS)]
    target = np.concatenate(columns, axis=1)
    edges = np.cumsum([0, *(c.shape[1] for c in columns)])
    blocks = [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    return Predictions(pred, target, blocks, data.groups)


def _schedule(cfg: SupervisedConfig) -> TrainConfig:
    """A ``TrainConfig`` carrying the learning-rate schedule of ``cfg`` (for ``lr_at``)."""
    return TrainConfig(steps=cfg.steps, lr=cfg.lr, warmup=cfg.warmup)


# ---------------------------------------------------------------------------------------------
# Features without training


def load_dino(weights: Path) -> nn.Module:
    """DINOv2 ViT-S/14 from the pinned hub code and a local weights file (22.06M parameters)."""
    model: nn.Module = torch.hub.load(  # type: ignore[no-untyped-call]
        f"facebookresearch/dinov2:{DINO_COMMIT}", "dinov2_vits14", pretrained=False, trust_repo=True
    )
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return model.eval()


@torch.no_grad()
def dino_features(model: nn.Module, frames: np.ndarray, batch: int = 64) -> np.ndarray:
    """``[N, 768]``: class token and mean patch token of DINOv2 on centre-cropped frames.

    Frames are the stored 256 px ``uint8`` frames; DINOv2's own evaluation transform (resize 256,
    centre-crop 224, ImageNet mean and std) is applied.
    """
    cropped = eval_view(frames, DINO_VIEW)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(cropped), batch):
        x = torch.as_tensor(cropped[i : i + batch]).permute(0, 3, 1, 2).float() / 255.0
        feats = model.forward_features((x - mean) / std)  # type: ignore[operator]
        out.append(
            torch.cat(
                [feats["x_norm_clstoken"], feats["x_norm_patchtokens"].mean(1)], dim=1
            ).numpy()
        )
    return np.concatenate(out)


class RawFeatures:
    """Raw windows as flat vectors: touch (log1p), audio, proprio, and a 16 px vision thumbnail.

    Args:
        space: The channel space (with the pretraining normalisation).
        substeps: K of the windows (the last observation of each is used).
    """

    def __init__(self, space: ChannelSpace, substeps: int) -> None:
        self.space, self.substeps = space, substeps
        self.thumb = View(size=16, crop=0.875)

    def blocks(self, episode: Episode, ends: np.ndarray) -> dict[str, np.ndarray]:
        """Named feature blocks ``touch``, ``audio``, ``proprio``, ``vision`` of the windows."""
        senses = ("touch", "audio", *PROPRIO)
        arrays = build(
            load_episode(episode, self.space, (*senses, "vision")),
            self.space,
            senses,
            ends,
            self.substeps,
        )
        frames = eval_view(episode.rgb[ends], self.thumb).reshape(len(ends), -1) / 255.0
        return {
            "touch": arrays["touch"][:, -1],
            "audio": arrays["audio"][:, -1],
            "proprio": np.concatenate([arrays[s][:, -1] for s in PROPRIO], axis=1),
            "vision": frames.astype(np.float32),
        }


def pca_fit(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and the top ``k`` principal directions ``[D, k]`` of ``x`` (rows are samples)."""
    mean = x.mean(0)
    _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    return mean, vt[:k].T


def pca_apply(x: np.ndarray, fit: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Project ``x`` on a fitted PCA."""
    return (x - fit[0]) @ fit[1]


# ---------------------------------------------------------------------------------------------
# Probe data for arbitrary features


FeatureFn = Callable[[Episode, np.ndarray], dict[str, np.ndarray]]


def window_ends(episode: Episode, substeps: int, stride: int) -> np.ndarray:
    """Last-observation indices of the probe windows of an episode (as ``SensoryLayer.encode``)."""
    first = (substeps - 1) * spacing(episode.dt)
    return np.arange(first, episode.length, stride)


def build_probe_data(
    paths: Sequence[Path], substeps: int, stride: int, features: FeatureFn
) -> ProbeData:
    """Windows, labels and episode groups of ``paths`` with features from ``features``.

    ``features(episode, ends)`` returns named ``[len(ends), D]`` matrices; the windows and labels
    are those of ``probes.collect``, so results are comparable and bootstrap-pairable with any
    encoder probed there.
    """
    feats: dict[str, list[np.ndarray]] = {}
    labels: dict[str, list[np.ndarray]] = {n: [] for n in (*BINARY, *CONTINUOUS, *HIDDEN)}
    groups = []
    for g, path in enumerate(paths):
        episode = Episode.load(path)
        ends = window_ends(episode, substeps, stride)
        lab = window_labels(privileged_labels(episode), ends, substeps, spacing(episode.dt))
        lab |= hidden_window_labels(episode, ends, substeps, spacing(episode.dt))
        for name, value in features(episode, ends).items():
            feats.setdefault(name, []).append(value)
        for name, value in lab.items():
            labels[name].append(value)
        groups.append(np.full(len(ends), g))
    return ProbeData(
        {n: np.concatenate(v) for n, v in feats.items()},
        {n: np.concatenate(v) for n, v in labels.items()},
        np.concatenate(groups),
    )


def block_rms(block: np.ndarray) -> float:
    """Overall RMS of a feature block around its mean (1 if constant)."""
    return float(np.sqrt(((block - block.mean(0)) ** 2).mean())) or 1.0


def concat_scaled(blocks: Sequence[np.ndarray], rms: Sequence[float]) -> np.ndarray:
    """Concatenate feature blocks, each divided by a fixed RMS (so none dominates the ridge)."""
    return np.concatenate([b / r for b, r in zip(blocks, rms, strict=True)], axis=1)


def few_shot_predictions(
    data: ProbeData, feature: str, shots: int, rng: np.random.Generator, alpha_mult: float = 1.0
) -> Predictions:
    """Probe predictions when only ``shots`` labelled episodes are available.

    For every held-out episode, a ridge (strength ``alpha_mult`` x features) is fitted on ``shots``
    other episodes drawn at random and predicts it; predictions are pooled as in the full probe. No
    strength search: with a handful of episodes there is nothing to select on.
    """
    x = data.features[feature]
    columns = [data.labels[n].reshape(len(x), -1) for n in (*BINARY, *CONTINUOUS)]
    y = np.concatenate(columns, axis=1)
    edges = np.cumsum([0, *(c.shape[1] for c in columns)])
    blocks = [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    episodes = np.unique(data.groups)
    pred = np.zeros_like(y, dtype=np.float64)
    for test in episodes:
        others = rng.choice(episodes[episodes != test], shots, replace=False)
        train = np.isin(data.groups, others)
        rows = data.groups == test
        pred[rows] = ridge_predict(x[train], y[train], x[rows], alpha_mult * x.shape[1])
    return Predictions(pred, y, blocks, data.groups)
