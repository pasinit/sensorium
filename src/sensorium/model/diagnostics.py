"""Label-free diagnostics of a trained layer: held-out loss and what the states owe to each sense.

Neither reads privileged state: they use the model's own inputs and loss.
"""

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from sensorium.episode import Episode
from sensorium.model.data import SequenceDataset
from sensorium.model.sensory_layer import SensoryLayer, collapse_metrics


@torch.no_grad()
def heldout_loss(
    layer: SensoryLayer,
    paths: Sequence[Path],
    batches: int = 8,
    batch_size: int = 16,
    seed: int = 0,
) -> dict[str, float]:
    """Mean ``pred`` and ``reg`` of the layer's own objective on sequences of held-out episodes.

    Centre crop, no touch noise, eval mode, a fixed seed. ``pred`` is the next-state error
    against the online states (SIGReg) or the EMA states (EMA + VICReg), so compare it only
    within one objective.
    """
    # held-out episodes are not in a training run's frame cache: decode them
    cfg = replace(layer.loader, touch_noise=False, frame_cache="")
    dataset = SequenceDataset(
        paths, layer.space, layer.model.data_senses, cfg, seed=seed, train=False
    )
    rng = np.random.default_rng(seed)
    device = next(layer.model.parameters()).device
    was_training = layer.model.training
    layer.model.eval()
    totals = {"pred": 0.0, "reg": 0.0}
    for _ in range(batches):
        seqs = [dataset.sample(rng) for _ in range(batch_size)]
        batch = {k: torch.as_tensor(np.stack([s[k] for s in seqs])).to(device) for k in seqs[0]}
        out = layer.model(batch)
        for key in totals:
            totals[key] += float(out[key]) / batches
    layer.model.train(was_training)
    return {f"heldout_{k}": v for k, v in totals.items()}


def sense_reliance(
    layer: SensoryLayer, paths: Sequence[Path], stride: int = 16
) -> dict[str, float]:
    """How much each sense moves the state: ``E|s - s_without_sense|^2 / total variance of s``.

    0 = the state ignores the sense; 1 = removing it moves the state as far as the states
    differ from each other. The layer was not trained with senses removed, so large values can
    also mean the fusion is brittle; compare checkpoints and encoders, not absolute levels.
    """
    full: list[np.ndarray] = []
    shifts: dict[str, list[np.ndarray]] = {s: [] for s in layer.model.senses}
    for path in paths:
        episode = Episode.load(path, privileged=False)
        full.append(layer.encode(episode, stride=stride)[0])
        for sense in layer.model.senses:
            shifts[sense].append(layer.encode(episode, stride=stride, drop=(sense,))[0])
    states = np.concatenate(full)
    total = float(states.var(0).sum())
    return {
        f"reliance_{sense}": float(((states - np.concatenate(parts)) ** 2).sum(1).mean() / total)
        for sense, parts in shifts.items()
    }


@torch.no_grad()
def touch_latent_r2(
    layer: SensoryLayer,
    paths: Sequence[Path],
    batches: int = 8,
    batch_size: int = 16,
    seed: int = 0,
) -> dict[str, float]:
    """How well the aux heads of a touch teacher predict the **true** EMA touch latents (held out).

    ``r2_now``: from the state, the latent of the current touch; ``r2_next``: from the predictor's
    output, the latent of the next touch. 1 - MSE / variance of the targets, so 0 = no better than
    the mean latent (the EMA target encoder runs with batch statistics, as in training; negative
    values mean the heads do not transfer to held-out episodes). For a placebo arm the targets here
    are the real ones of the same sequence (not the rolled ones it was trained on), so a value near
    0 means it learned nothing about touch.
    Empty if the layer has no teacher.
    """
    model = layer.model
    if model.teacher is None:
        return {}
    cfg = replace(layer.loader, touch_noise=False, frame_cache="")
    dataset = SequenceDataset(paths, layer.space, model.data_senses, cfg, seed=seed, train=False)
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    model.teacher.target.train()  # batch statistics, as the heads were trained against
    now_err, next_err, targets = [], [], []
    for _ in range(batches):
        seqs = [dataset.sample(rng) for _ in range(batch_size)]
        batch = {k: torch.as_tensor(np.stack([s[k] for s in seqs])).to(device) for k in seqs[0]}
        states = model.embed(batch)
        cond = model.action(batch["action"].flatten(0, 1)).view_as(states)
        trunk = model.trunk(states[:, :-1], cond[:, :-1])
        target = model.teacher.target(batch["touch"], batch["region_mask"])
        now_err.append((model.teacher.head_now(states) - target).pow(2).mean())
        next_err.append((model.teacher.head_next(trunk) - target[:, 1:]).pow(2).mean())
        targets.append(target.flatten(0, 1))
    model.train(was_training)
    var = torch.cat(targets).var(0).mean()
    return {
        "touch_r2_now": float(1 - torch.stack(now_err).mean() / var),
        "touch_r2_next": float(1 - torch.stack(next_err).mean() / var),
    }


def state_geometry(states: np.ndarray) -> dict[str, float]:
    """Effective rank and mean per-dimension std of ``[N, D]`` states (held-out windows).

    The effective rank is ``exp`` of the entropy of the normalised singular values of the centred
    states: ``D`` for isotropic states, 1 for a collapsed one (``sensory_layer.collapse_metrics``
    on a set of windows instead of a training batch).
    """
    x = torch.as_tensor(np.asarray(states, dtype=np.float32))[None]
    return collapse_metrics(x) | {"windows": float(len(states))}
