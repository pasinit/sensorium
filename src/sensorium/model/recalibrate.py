"""BatchNorm re-calibration of a trained sensory layer (B-pusht-0's BatchNorm rule).

The projector's BatchNorm running statistics are what eval mode, planning and the few-shot
features use. They must describe the inputs the layer is used on: B-pusht-0's one-pass fix1
collected them on masked frames, which mapped every clean state to nearly one point.
``recalibrate`` re-estimates them on clean sequences of a pool, without gradients; no weight
changes. ``pool_batches`` cuts those sequences as training cut them (the layer's loader
configuration, centre view).
"""

import copy
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from sensorium.model.data import SequenceDataset, sequence_with_ends
from sensorium.model.sensory_layer import SensoryLayer, SensoryModel

RECAL_BATCHES = 50
RECAL_BATCH = 128
RECAL_SEED = 1
PER_EPISODE = 8
"""Sequences cut from each decoded episode (an episode is decoded once)."""


def recalibrate(model: SensoryModel, batches: Sequence[dict[str, torch.Tensor]]) -> SensoryModel:
    """A copy of ``model`` whose BatchNorm running statistics are re-estimated: reset, then the
    cumulative average over ``batches`` encoded in training mode without gradients. Every weight is
    unchanged; the copy is returned in eval mode.

    Raises:
        ValueError: if ``batches`` is empty.
    """
    if not batches:
        raise ValueError("recalibrate needs at least one batch")
    net = copy.deepcopy(model)
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.reset_running_stats()
            m.momentum = None  # a cumulative average over the batches
    net.train()
    with torch.no_grad():
        for batch in batches:
            net.embed(batch)
    return net.eval()


def pool_batches(
    layer: SensoryLayer,
    paths: Sequence[Path],
    batches: int = RECAL_BATCHES,
    batch_size: int = RECAL_BATCH,
    seed: int = RECAL_SEED,
) -> list[dict[str, torch.Tensor]]:
    """``batches`` batches of ``batch_size`` clean sequences from the episodes in ``paths``:
    ``PER_EPISODE`` random starts per episode, the episodes drawn by ``rng(seed)`` (with
    replacement only if there are too few), each sequence cut with the layer's loader
    configuration and the centre view (no crop, no touch noise).

    Raises:
        ValueError: if ``paths`` is empty.
    """
    if not paths:
        raise ValueError("pool_batches needs episode files")
    rng = np.random.default_rng(seed)
    total = batches * batch_size
    n_eps = math.ceil(total / PER_EPISODE)
    picks = rng.choice(len(paths), n_eps, replace=n_eps > len(paths))
    senses = layer.model.data_senses
    data = SequenceDataset(list(paths), layer.space, senses, layer.loader, seed=seed, train=False)
    seqs = []
    for i in picks:
        loaded = data.load(Path(paths[int(i)]))
        for _ in range(PER_EPISODE):
            seqs.append(
                sequence_with_ends(loaded, layer.space, senses, layer.loader, rng, False)[0]
            )
    seqs = seqs[:total]
    return [
        {k: torch.as_tensor(np.stack([q[k] for q in seqs[j : j + batch_size]])) for k in seqs[0]}
        for j in range(0, total, batch_size)
    ]


def recalibrated(layer: SensoryLayer, paths: Sequence[Path]) -> SensoryLayer:
    """The layer with ``recalibrate``'d BatchNorm statistics on ``pool_batches`` of ``paths``."""
    return SensoryLayer(
        recalibrate(layer.model, pool_batches(layer, paths)), layer.space, layer.loader
    )
