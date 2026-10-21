"""From episodes to 5 Hz training sequences (and to the windows ``encode`` reads).

A **model step** covers ``substeps`` (K) observations spaced ``spacing`` env steps apart
(0.2 s for the 20 Hz H1: K = 4, spacing 1; a body with another control rate gets the
spacing that is closest to 0.05 s per observation). The state ``s_t`` of the model is a
function of the window of K observations that *ends* at ``e_t``; the action chunk ``a_t``
is the K actions applied from ``e_t`` on, so ``(s_t, a_t) -> s_{t+1}`` is one transition
(the timing convention of ``sensorium.episode``). Vision uses only the last frame of the
window. Touch, audio and proprio keep all K sub-steps, stacked.

Everything that reaches a batch comes from ``sensorium.episode.model_inputs``; the
privileged state and the text of an episode are never touched (the tests plant a
tripwire on them). Touch noise (``touch_noise``) and the random crop (``vision``) are
applied at load time, once per training sequence.
"""

import logging
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from sensorium.episode import Episode, model_inputs
from sensorium.model.layout import (
    INPUT_KEY,
    N_MELS,
    PROPRIO_GROUPS,
    ChannelSpace,
    Columns,
    Header,
    check_senses,
    read_header,
)
from sensorium.touch_noise import TouchNoise, perturb_touch
from sensorium.vision import View, eval_view, train_view

log = logging.getLogger("sensorium.model.data")

REFERENCE_DT = 0.05
CLIP = 10.0


@dataclass(frozen=True)
class LoaderConfig:
    """How sequences are cut and augmented.

    Attributes:
        steps: Model steps ``T`` per training sequence.
        substeps: Observations ``K`` stacked per model step.
        view_size: Side of the vision input (px).
        crop: Random-crop fraction of the stored 256 px frame (``vision.View.crop``).
        touch_noise: Whether to perturb touch (gain, dead channels, noise) per sequence.
        cache_episodes: Decoded episodes kept in memory per worker.
        reuse: Sequences drawn from an episode each time the stream picks it. Decoding an
            episode costs more than cutting a sequence from it, so a stream that decoded one
            episode per sequence would be decode-bound (``docs/model.md``).
        frame_cache: Optional directory of ``sensorium.model.frame_cache``: episodes are read
            from it (memory-mapped) instead of decoded. Every training episode must be in it.
    """

    steps: int = 16
    substeps: int = 4
    view_size: int = 128
    crop: float = 0.9
    touch_noise: bool = True
    cache_episodes: int = 16
    reuse: int = 8
    frame_cache: str = ""

    @property
    def view(self) -> View:
        """The ``vision.View`` of this configuration."""
        return View(size=self.view_size, crop=self.crop)


@dataclass
class Loaded:
    """One decoded episode: only what ``model_inputs`` returned, plus its names.

    ``rgb_source_side`` is set when ``inputs["rgb"]`` is a downscale (a frame cache) of frames of
    that side; crops are then placed in the original grid (``vision.train_view``).
    """

    header: Header
    inputs: dict[str, np.ndarray]
    columns: Columns
    rgb_source_side: int | None = None

    @property
    def length(self) -> int:
        """Number of steps."""
        return int(self.inputs["action"].shape[0])


def spacing(dt: float) -> int:
    """Env steps between the K observations of a model step, for a control period ``dt``."""
    return max(1, round(REFERENCE_DT / dt))


def header_of(episode: Episode) -> Header:
    """The header of an in-memory episode."""
    return Header(
        body=episode.body,
        dt=episode.dt,
        proprio_names=tuple(episode.proprio_names),
        touch_names=tuple(episode.touch_names),
        action_names=tuple(episode.action_names),
    )


def load_episode(episode: Episode, space: ChannelSpace, senses: Sequence[str]) -> Loaded:
    """Keep the model inputs of an episode that ``senses`` ask for, and nothing else."""
    keys = sorted({INPUT_KEY[s] for s in check_senses(senses)})
    header = header_of(episode)
    return Loaded(header, model_inputs(episode, senses=keys), space.columns(header))


def compute_stats(paths: Sequence[Path], space: ChannelSpace, limit: int = 64) -> None:
    """Fit per-channel mean and std of the proprio groups and audio on some episodes.

    Reads ``proprio`` and ``audio_logmel`` straight from the files (no video decoding);
    these two arrays are model inputs.
    """
    sums = {g: np.zeros((2, space.size(g))) for g in (*PROPRIO_GROUPS, "audio")}
    counts = {g: np.zeros(space.size(g)) for g in sums}
    for path in list(paths)[:limit]:
        header = read_header(path)
        cols = space.columns(header)
        with np.load(path, allow_pickle=False) as data:
            proprio, audio = data["proprio"], data["audio_logmel"].astype(np.float64)
        for g in PROPRIO_GROUPS:
            x = proprio[:, cols.src[g]].astype(np.float64)
            sums[g][0, cols.dst[g]] += x.sum(0)
            sums[g][1, cols.dst[g]] += (x**2).sum(0)
            counts[g][cols.dst[g]] += len(x)
        sums["audio"][0] += audio.sum(0)
        sums["audio"][1] += (audio**2).sum(0)
        counts["audio"] += len(audio)
    for g, s in sums.items():
        n = np.maximum(counts[g], 1.0)
        mean = s[0] / n
        std = np.sqrt(np.maximum(s[1] / n - mean**2, 0.0))
        space.set_stats(g, mean, np.where(std > 1e-6, std, 1.0))


def _normalise(x: np.ndarray, space: ChannelSpace, group: str, dst: np.ndarray) -> np.ndarray:
    """Standardise columns ``dst`` of a group with the fitted statistics (identity if unfitted)."""
    if group not in space.stats:
        return x.astype(np.float32)
    mean = np.asarray(space.stats[group]["mean"])[dst]
    std = np.asarray(space.stats[group]["std"])[dst]
    return np.clip((x - mean) / std, -CLIP, CLIP).astype(np.float32)


def _to_union(x: np.ndarray, dst: np.ndarray, size: int) -> np.ndarray:
    """``[..., C]`` -> ``[..., size]`` with column ``i`` placed at ``dst[i]``; the rest 0."""
    out = np.zeros((*x.shape[:-1], size), dtype=np.float32)
    out[..., dst] = x
    return out


def window_index(ends: np.ndarray, substeps: int, gap: int) -> np.ndarray:
    """``[n, K]`` observation indices of the windows ending at ``ends``."""
    return ends[:, None] - gap * (substeps - 1 - np.arange(substeps))[None]


def build(
    ep: Loaded,
    space: ChannelSpace,
    senses: Sequence[str],
    ends: np.ndarray,
    substeps: int,
    touch: np.ndarray | None = None,
    touch_offset: int = 0,
    rgb: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Arrays for the windows ending at ``ends`` (indices into the episode).

    Args:
        ep: A loaded episode.
        space: The channel space.
        senses: Configured senses; only these are produced.
        ends: ``[n]`` last-observation index of each window.
        substeps: K.
        touch: Optional replacement for the episode's touch (a perturbed copy of the rows
            from ``touch_offset`` on).
        touch_offset: Episode index of the first row of ``touch``.
        rgb: Optional ``[n, H, W, 3]`` frames already cropped for these windows.

    Returns:
        ``vision`` ``[n, H, W, 3]`` uint8, ``touch`` ``[n, K, U]``, ``audio`` ``[n, K, 32]``,
        one ``[n, K, U]`` per proprio sense, ``action`` ``[n, K, Ua]`` (float32), plus
        ``present`` ``[S]`` bool per configured sense and ``region_mask`` ``[R]``.
    """
    s = spacing(ep.header.dt)
    idx = window_index(ends, substeps, s)
    cols, out = ep.columns, {}
    size = space.size
    for sense in check_senses(senses):
        key = INPUT_KEY[sense]
        if sense == "vision":
            out[sense] = rgb if rgb is not None else ep.inputs["rgb"][ends]
        elif sense == "touch":
            rows = ep.inputs["touch"][idx] if touch is None else touch[idx - touch_offset]
            x = rows[..., cols.src["touch"]]
            out[sense] = _to_union(np.log1p(np.maximum(x, 0.0)), cols.dst["touch"], size("touch"))
        elif sense == "audio":
            out[sense] = _normalise(
                ep.inputs[key][idx].astype(np.float32), space, "audio", np.arange(N_MELS)
            )
        else:
            x = ep.inputs[key][idx][..., cols.src[sense]]
            out[sense] = _to_union(
                _normalise(x, space, sense, cols.dst[sense]), cols.dst[sense], size(sense)
            )
    last = ep.length - 1
    aidx = np.minimum(ends[:, None] + s * np.arange(substeps)[None], last)
    action = ep.inputs["action"][aidx][..., cols.src["action"]]
    out["action"] = _to_union(action.astype(np.float32), cols.dst["action"], size("action"))
    out["present"] = np.array([_has(ep, s_) for s_ in check_senses(senses)], dtype=bool)
    out["region_mask"] = cols.region_mask
    return out


def _has(ep: Loaded, sense: str) -> bool:
    """Whether the episode has any channel of this sense."""
    if sense == "vision":
        return ep.inputs["rgb"].size > 0
    if sense == "audio":
        return ep.inputs["audio_logmel"].shape[1] > 0
    group = "touch" if sense == "touch" else sense
    return len(ep.columns.src[group]) > 0


def sequence_from_episode(
    ep: Loaded,
    space: ChannelSpace,
    senses: Sequence[str],
    cfg: LoaderConfig,
    rng: np.random.Generator,
    train: bool = True,
) -> dict[str, np.ndarray]:
    """One random training sequence of ``cfg.steps`` model steps from an episode.

    Raises:
        ValueError: if the episode is shorter than one sequence.
    """
    return sequence_with_ends(ep, space, senses, cfg, rng, train)[0]


def sequence_with_ends(
    ep: Loaded,
    space: ChannelSpace,
    senses: Sequence[str],
    cfg: LoaderConfig,
    rng: np.random.Generator,
    train: bool = True,
    start: int | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """``sequence_from_episode`` that also returns the ``[T]`` last-observation index of each window
    (so a supervised control can attach labels to the same windows).

    Args:
        start: First env step of the sequence; ``None`` draws one uniformly.

    Raises:
        ValueError: if the episode is shorter than one sequence.
    """
    k, s = cfg.substeps, spacing(ep.header.dt)
    span = cfg.steps * k * s
    if ep.length < span:
        raise ValueError(f"episode of {ep.length} steps is shorter than a sequence ({span})")
    start = int(rng.integers(0, ep.length - span + 1)) if start is None else start
    ends = start + np.arange(cfg.steps) * k * s + (k - 1) * s
    touch = None
    if "touch" in senses and len(ep.columns.src["touch"]) and cfg.touch_noise:
        clip = slice(start, start + span)
        names = list(ep.header.touch_names)
        touch = perturb_touch(ep.inputs["touch"][clip], names, rng, TouchNoise())
    rgb = None
    if "vision" in senses:
        frames = np.asarray(ep.inputs["rgb"][ends])
        side = ep.rgb_source_side
        if train:
            rgb = train_view(frames, rng, cfg.view, side)
        else:
            rgb = eval_view(frames, cfg.view, side)
    return build(ep, space, senses, ends, k, touch=touch, touch_offset=start, rgb=rgb), ends


class SequenceDataset(IterableDataset[dict[str, np.ndarray]]):
    """An endless stream of random training sequences from a list of episode files.

    Episodes are drawn with probability proportional to their number of valid start
    positions, so every window of every episode is equally likely; with ``start_weights``,
    proportional to the sum of their start weights, and the start within the episode in
    proportion to its weight (``sensorium.model.events``). Decoded episodes are
    kept in a small per-worker LRU cache. Each worker (and the main process) has its own
    random stream, derived from ``seed``, so a run is reproducible for a fixed
    ``num_workers``.

    Args:
        paths: Episode files.
        space: The channel space (``ChannelSpace.from_headers`` over the same files).
        senses: Configured senses; only their inputs are decoded and returned.
        cfg: Sequence cutting and augmentation.
        seed: Base seed of the random streams.
        train: Random crop and touch noise if true, centre crop if false.
        start_weights: Optional ``{episode file name: [weight per valid start]}``.

    Raises:
        ValueError: if no episode is long enough for one sequence, or ``start_weights`` misses an
            episode or has the wrong number of starts for one.
    """

    def __init__(
        self,
        paths: Sequence[Path],
        space: ChannelSpace,
        senses: Sequence[str],
        cfg: LoaderConfig,
        seed: int = 0,
        train: bool = True,
        start_weights: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        self.paths = [Path(p) for p in paths]
        self.space, self.cfg, self.seed, self.train = space, cfg, seed, train
        self.senses = check_senses(senses)
        self._cache: OrderedDict[Path, Loaded] = OrderedDict()
        weights: list[float] = []
        for path in self.paths:
            header = read_header(path)
            with np.load(path, allow_pickle=False) as data:
                length = int(data["action"].shape[0])
            span = cfg.steps * cfg.substeps * spacing(header.dt)
            weights.append(max(length - span + 1, 0))
        self._start_p: list[np.ndarray | None] = [None] * len(self.paths)
        if start_weights is not None:
            for i, path in enumerate(self.paths):
                w = np.asarray(start_weights[path.name], dtype=np.float64)
                if len(w) != weights[i]:
                    raise ValueError(f"{path.name}: {len(w)} start weights, {weights[i]} starts")
                weights[i] = float(w.sum())
                self._start_p[i] = w / w.sum() if w.sum() > 0 else None
        if sum(weights) == 0:
            raise ValueError("no episode is long enough for one sequence")
        self._weights = np.array(weights, dtype=np.float64) / sum(weights)

    def load(self, path: Path) -> Loaded:
        """Decode (or fetch from the cache) one episode."""
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        if self.cfg.frame_cache:
            from sensorium.model.frame_cache import load_cached  # noqa: PLC0415 (imports data)

            loaded = load_cached(path, Path(self.cfg.frame_cache), self.space, self.senses)
        else:
            loaded = load_episode(Episode.load(path, privileged=False), self.space, self.senses)
        self._cache[path] = loaded
        while len(self._cache) > self.cfg.cache_episodes:
            self._cache.popitem(last=False)
        return loaded

    def sample(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Draw one sequence."""
        i = self._pick(rng)
        return self._cut(i, self.load(self.paths[i]), rng)

    def _pick(self, rng: np.random.Generator) -> int:
        return int(rng.choice(len(self.paths), p=self._weights))

    def _start(self, i: int, rng: np.random.Generator) -> int | None:
        """A weighted start in episode ``i``, or ``None`` (uniform, drawn by the cutter)."""
        p = self._start_p[i]
        return None if p is None else int(rng.choice(len(p), p=p))

    def _cut(self, i: int, loaded: Loaded, rng: np.random.Generator) -> dict[str, np.ndarray]:
        start = self._start(i, rng)
        return sequence_with_ends(
            loaded, self.space, self.senses, self.cfg, rng, self.train, start=start
        )[0]

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        info = get_worker_info()
        rng = np.random.default_rng([self.seed, info.id if info else 0])
        while True:
            i = self._pick(rng)
            loaded = self.load(self.paths[i])
            for _ in range(self.cfg.reuse):
                yield self._cut(i, loaded, rng)


def to_tensors(batch: Mapping[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """numpy -> torch (for a batch that was not made by a ``DataLoader``)."""
    return {k: torch.as_tensor(v) for k, v in batch.items()}
