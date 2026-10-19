"""Choosing training sequences by what they contain: the consequence-rich subset test.

Hand-object contact that *moves* the object ("consequence") is rare in random poking. This module
builds, **offline**, an index of the sequence start positions whose windows contain it, and a
sampler that draws training sequences only from the chosen positions. The privileged object
motion decides which sequences are drawn; it is read by ``probes.consequence_steps`` while the
index is built and never reaches a batch (the sampler loads episodes with ``privileged=False``).

Three classes of start position (a sequence spans ``T * K`` env steps, 32 for the default model):

* ``all``: every valid start (equal to uniform random sampling);
* ``conseq``: at least ``frac`` of the sequence's steps are consequence steps;
* ``contact``: at least ``frac`` of its steps have hand-object contact, but fewer than ``frac``
  are consequence steps (contact without much effect: the control);
* ``event``: at least ``frac`` of its steps are inside a Track A caused-change event
  (``probes.event_steps``: displaced, toppled, dropped, opened, closed, pressed), so "consequence"
  means the same in both tracks.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from torch.utils.data import get_worker_info

from sensorium.episode import Episode
from sensorium.model.data import (
    Loaded,
    LoaderConfig,
    SequenceDataset,
    sequence_with_ends,
    spacing,
)
from sensorium.model.layout import ChannelSpace, read_header
from sensorium.model.probes import consequence_steps

KEYS = ("all", "conseq", "contact", "event")


def window_fraction(flags: np.ndarray, span: int) -> np.ndarray:
    """Fraction of ``flags`` true in every window of ``span`` steps (``len - span + 1`` values)."""
    return np.convolve(flags.astype(float), np.ones(span), "valid") / span


def build_index(
    paths: Sequence[Path], span: int, frac: float = 0.25, horizon: int = 10, min_move: float = 0.01
) -> dict[str, dict[str, list[int]]]:
    """``{episode file name: {class: [start positions]}}`` for the three classes.

    Loads each episode with its privileged arrays (this is the only place it happens).
    """
    index: dict[str, dict[str, list[int]]] = {}
    for path in paths:
        flags = consequence_steps(Episode.load(path), horizon, min_move)
        contact = window_fraction(flags["hand_object_contact"], span)
        conseq = window_fraction(flags["consequence"], span)
        index[path.name] = {
            "all": list(range(len(contact))),
            "conseq": np.flatnonzero(conseq >= frac).tolist(),
            "contact": np.flatnonzero((contact >= frac) & (conseq < frac)).tolist(),
            "event": np.flatnonzero(window_fraction(flags["event"], span) >= frac).tolist(),
        }
    return index


def summarise(index: dict[str, dict[str, list[int]]]) -> dict[str, dict[str, int]]:
    """Distinct start positions and episodes with at least one, per class."""
    return {
        key: {
            "starts": sum(len(v[key]) for v in index.values()),
            "episodes": sum(1 for v in index.values() if v[key]),
        }
        for key in KEYS
    }


class IndexedSequences(SequenceDataset):
    """Training sequences drawn only from the start positions of one class of an index.

    Episodes are drawn with probability proportional to their number of chosen starts (so every
    chosen sequence is equally likely), ``cfg.reuse`` sequences per pick. Episodes are loaded with
    ``privileged=False`` like any training episode.

    Args:
        paths: Episode files (the index must have an entry for each file name).
        space, senses, cfg, seed, train: As ``SequenceDataset``.
        index: The index file written by ``build_index`` (loaded as JSON).
        key: ``conseq``, ``contact`` or ``all``.

    Raises:
        ValueError: if ``key`` is unknown or no episode has a chosen start.
    """

    def __init__(
        self,
        paths: Sequence[Path],
        space: ChannelSpace,
        senses: Sequence[str],
        cfg: LoaderConfig,
        index: dict[str, dict[str, list[int]]],
        key: str,
        seed: int = 0,
        train: bool = True,
    ) -> None:
        if key not in KEYS:
            raise ValueError(f"key must be one of {KEYS}, got {key!r}")
        super().__init__(paths, space, senses, cfg, seed, train)
        self.starts = [np.array(index[p.name][key], dtype=int) for p in self.paths]
        counts = np.array([len(s) for s in self.starts], dtype=float)
        if counts.sum() == 0:
            raise ValueError(f"no sequence start of class {key!r} in these episodes")
        self._weights = counts / counts.sum()
        self._span = {p: cfg.steps * cfg.substeps * spacing(read_header(p).dt) for p in self.paths}

    def _cut_at(self, i: int, loaded: Loaded, rng: np.random.Generator) -> dict[str, np.ndarray]:
        start = int(rng.choice(self.starts[i]))
        return sequence_with_ends(
            loaded, self.space, self.senses, self.cfg, rng, self.train, start=start
        )[0]

    def sample(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Draw one sequence from the chosen starts."""
        i = int(rng.choice(len(self.paths), p=self._weights))
        return self._cut_at(i, self.load(self.paths[i]), rng)

    def __iter__(self):  # type: ignore[no-untyped-def]
        info = get_worker_info()
        rng = np.random.default_rng([self.seed, info.id if info else 0])
        while True:
            i = int(rng.choice(len(self.paths), p=self._weights))
            loaded = self.load(self.paths[i])
            for _ in range(self.cfg.reuse):
                yield self._cut_at(i, loaded, rng)


def save_index(index: dict[str, dict[str, list[int]]], path: Path) -> None:
    """Write an index as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index))


def load_index(path: Path) -> dict[str, dict[str, list[int]]]:
    """Read an index written by ``save_index``."""
    return json.loads(path.read_text())
