"""Outcome probes: does an encoder's state know what is happening to the objects? (exploratory)

The contact battery (``probes.py``) scores the robot's own contacts, force and fingertips. The
experiment programs are about consequences, so these labels score the objects instead, from
privileged state (``object_pose``, ``object_contact``) at analysis time only. Per window of K
observations ending at step ``t``:

* ``displaced`` (AUROC): some object that an effector touches within 1 s of ``t`` moves more than
  ``caused_change.DISPLACED_M`` (3 cm) in xy between ``t`` and ``t`` + 1 s: what the robot is about
  to move (a sphere still rolling long after it was hit does not count);
* ``lifted`` (AUROC): at ``t`` some object is at least ``babble_stats.AIR_M`` (3 cm) above its
  resting height with an effector on it (in hand, above the table);
* ``toppled`` (AUROC): some non-spherical object below ``caused_change.TOPPLE_DEG`` (45 degrees)
  of tilt at ``t`` passes it within the next 1 s;
* ``log_speed`` (R^2): ``log1p(v / 1 cm/s)`` of the fastest object's speed, averaged over the
  window's observations.

The per-step labels are computed in ``probes.outcome_steps`` (the one module that reads privileged
state). The probes and their scores are those of the battery (``probes.predict_probes`` with these
names), so the paired contrasts of ``sensorium.model.paired`` apply unchanged.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from sensorium.episode import Episode
from sensorium.model.controls import window_ends
from sensorium.model.data import spacing
from sensorium.model.probes import (
    BINARY,
    CONTINUOUS,
    HIDDEN,
    Predictions,
    ProbeData,
    hidden_window_labels,
    outcome_steps,
    predict_probes,
    privileged_labels,
    window_labels,
)

OUTCOME_BINARY = ("displaced", "lifted", "toppled")
OUTCOME_CONTINUOUS = ("log_speed",)
OUTCOMES = (*OUTCOME_BINARY, *OUTCOME_CONTINUOUS)


def outcome_labels(
    steps: dict[str, np.ndarray], ends: np.ndarray, substeps: int, gap: int
) -> dict[str, np.ndarray]:
    """Window labels from ``outcome_steps``: the binary labels at the window's last step, the
    speed averaged over its ``substeps`` observations ``gap`` apart."""
    idx = ends[:, None] - gap * (substeps - 1 - np.arange(substeps))[None]
    out = {name: steps[name][ends].astype(np.float64) for name in OUTCOME_BINARY}
    out["log_speed"] = steps["log_speed"][idx].mean(axis=1)
    return out


FeatureSet = Callable[[Episode, np.ndarray], dict[str, np.ndarray]]
"""Named ``[len(ends), D]`` feature matrices of one episode's windows."""


def collect_all(
    paths: Sequence[Path], substeps: int, stride: int, features: FeatureSet
) -> ProbeData:
    """Windows of ``paths`` (as ``probes.collect``) with every feature of ``features`` and the
    outcome, contact-battery and hidden-property labels. Each episode is loaded once, so many
    encoders can share one pass (``features`` returns one matrix per encoder).

    Raises:
        KeyError: if an episode lacks the privileged arrays the labels need.
    """
    feats: dict[str, list[np.ndarray]] = {}
    labels: dict[str, list[np.ndarray]] = {
        n: [] for n in (*BINARY, *CONTINUOUS, *HIDDEN, *OUTCOMES)
    }
    groups = []
    for g, path in enumerate(paths):
        episode = Episode.load(path)  # privileged labels are read below, and only there
        ends = window_ends(episode, substeps, stride)
        gap = spacing(episode.dt)
        lab = window_labels(privileged_labels(episode), ends, substeps, gap)
        lab |= hidden_window_labels(episode, ends, substeps, gap)
        lab |= outcome_labels(outcome_steps(episode), ends, substeps, gap)
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


def predict_outcomes(data: ProbeData, feature: str) -> Predictions:
    """Leave-one-episode-out ridge predictions of the outcome labels from one feature matrix."""
    return predict_probes(data, feature, OUTCOMES, OUTCOME_BINARY)
