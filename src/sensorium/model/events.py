"""Event-weighted sequence sampling from label-free signals.

Most windows of seated babbling are rest: the arms move slowly or not at all and nothing on the
table changes. This module weights training sequences toward events, using only what the model
itself receives (``model_inputs``), never contact or force labels (the probes test on those):

* ``body_speed``: mean ``|dq|`` of the body joints (proprio ``dq_*`` without the effector prefixes);
* ``effector_speed``: mean ``|dq|`` of the hand or pincer joints (``dq_lh_*``, ``dq_rh_*``,
  ``dq_lp_*``, ``dq_rp_*``), the label-free stand-in for fingertip velocity;
* ``effort_change``: mean ``|d eff_* / dt|`` (absent when the episode records no effort);
* ``frame_change``: mean absolute difference of consecutive frames (grey levels, every fourth
  pixel), which also sees objects that move with nothing touching them;
* ``effector_effort`` (off by default; ``SIGNALS`` is the default set): mean ``|eff|`` of the
  hand or pincer actuators, the load of a grip or a push.

Each signal is scaled robustly over the training set (median and MAD, ``fit_scales``); a step is an
**event step** when any scaled signal exceeds ``Z`` (3). A sequence start gets the weight
``1 + (cap - 1) * min(1, f / SATURATE)``, with ``f`` the share of event steps in its span, so a
start with no event step keeps weight 1, and starts with a quarter or more event steps get ``cap``
(4). Rest starts are then scaled up, if needed, to hold at least ``REST_SHARE`` (15%) of the
draws, and no start is drawn more than ``cap`` times the uniform rate (``finalise``). This is done
within the hand and the pincer episodes separately (``finalise_by_group``), so each effector keeps
its uniform share of draws: pincer effort runs higher than hand effort, and one pool would
oversample pincers. Test sets are never
weighted: the weights only reach ``SequenceDataset(start_weights=...)`` of training runs.

    uv run python -m sensorium.model.events <out.json> <episode.npz>... [--cache DIR] [--span 32]
"""

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from sensorium.episode import Episode, model_inputs
from sensorium.model.data import spacing
from sensorium.model.frame_cache import META, entry

SIGNALS = ("body_speed", "effector_speed", "effort_change", "frame_change")
"""The default signals (the brief's: proprio and effector speed, effort change, frame change)."""
ALL_SIGNALS = (*SIGNALS, "effector_effort")
EFFECTOR_PREFIXES = ("dq_lh_", "dq_rh_", "dq_lp_", "dq_rp_")
Z = 3.0
"""Robust z-score above which a signal marks an event step."""
CAP = 4.0
"""Most a start is drawn, relative to the uniform rate."""
SATURATE = 0.25
"""Share of event steps in a span at which its weight reaches ``CAP``."""
REST_SHARE = 0.15
"""Least share of draws kept for rest starts (no event step in the span)."""
PIXEL_STRIDE = 4

log = logging.getLogger("sensorium.model.events")


def step_signals(
    proprio: np.ndarray, names: Sequence[str], rgb: np.ndarray, dt: float
) -> dict[str, np.ndarray]:
    """``[T]`` raw signals of one episode (see the module docstring; step 0 is 0).

    Args:
        proprio: ``[T, P]`` proprio of the episode.
        names: The ``P`` proprio channel names.
        rgb: ``[T, H, W, 3]`` uint8 frames (any resolution), or an empty array.
        dt: Seconds per step.
    """
    names = list(names)
    dq = [i for i, n in enumerate(names) if n.startswith("dq_")]
    eff = [i for i in dq if names[i].startswith(EFFECTOR_PREFIXES)]
    body = [i for i in dq if i not in eff]
    effort = [i for i, n in enumerate(names) if n.startswith("eff_")]
    effector_effort = [i for i in effort if names[i][4:].startswith(("lh_", "rh_", "lp_", "rp_"))]
    t = len(proprio)
    out = {
        "body_speed": np.abs(proprio[:, body]).mean(1) if body else np.zeros(t),
        "effector_speed": np.abs(proprio[:, eff]).mean(1) if eff else np.zeros(t),
    }
    if effort:
        change = np.abs(np.diff(proprio[:, effort], axis=0)).mean(1) / dt
        out["effort_change"] = np.concatenate([np.zeros(1), change])
    if effector_effort:
        out["effector_effort"] = np.abs(proprio[:, effector_effort]).mean(1)
    if rgb.size:
        grey = np.asarray(rgb[:, ::PIXEL_STRIDE, ::PIXEL_STRIDE], dtype=np.float32).mean(-1)
        out["frame_change"] = np.concatenate(
            [np.zeros(1), np.abs(np.diff(grey, axis=0)).mean((1, 2))]
        )
    return {k: v.astype(np.float64) for k, v in out.items()}


def effector_kind(names: Sequence[str]) -> str:
    """``hand``, ``pincer`` or ``none``, from the proprio channel names (the body, not a label)."""
    if any(n.startswith(("dq_lh_", "dq_rh_")) for n in names):
        return "hand"
    if any(n.startswith(("dq_lp_", "dq_rp_")) for n in names):
        return "pincer"
    return "none"


def episode_signals(
    path: Path, cache: Path | None = None
) -> tuple[dict[str, np.ndarray], float, str]:
    """Signals, ``dt`` and effector kind of an episode file, from the frame cache if given (no
    privileged state is read either way)."""
    if cache is not None:
        root = entry(cache, path)
        header = json.loads((root / META).read_text())["header"]
        proprio = np.load(root / "proprio.npy", mmap_mode="r")
        rgb = np.load(root / "rgb.npy", mmap_mode="r")
        names = header["proprio_names"]
        signals = step_signals(np.asarray(proprio), names, rgb, header["dt"])
        return signals, header["dt"], effector_kind(names)
    episode = Episode.load(path, privileged=False)
    inputs = model_inputs(episode, ("rgb", "proprio"))
    return (
        step_signals(inputs["proprio"], episode.proprio_names, inputs["rgb"], episode.dt),
        episode.dt,
        effector_kind(episode.proprio_names),
    )


def fit_scales(
    signals: Sequence[Mapping[str, np.ndarray]], use: Sequence[str] = SIGNALS
) -> dict[str, tuple[float, float]]:
    """``{signal: (median, 1.4826 * MAD)}`` over every step of some episodes, for the signals
    ``use`` (``event_flags`` reads only these)."""
    out = {}
    for name in use:
        values = [s[name] for s in signals if name in s]
        if not values:
            continue
        x = np.concatenate(values)
        med = float(np.median(x))
        out[name] = (med, max(1.4826 * float(np.median(np.abs(x - med))), 1e-9))
    return out


def event_flags(
    signals: Mapping[str, np.ndarray], scales: Mapping[str, tuple[float, float]]
) -> np.ndarray:
    """``[T]`` True where any scaled signal exceeds ``Z``."""
    z = [(signals[n] - m) / s for n, (m, s) in scales.items() if n in signals]
    return np.max(z, axis=0) > Z


def raw_weights(flags: np.ndarray, span: int, cap: float = CAP) -> np.ndarray:
    """Per-start weights before ``finalise`` (``len(flags) - span + 1`` values)."""
    share = np.convolve(flags.astype(float), np.ones(span), "valid") / span
    return 1.0 + (cap - 1.0) * np.minimum(1.0, share / SATURATE)


def finalise(
    weights: Mapping[str, np.ndarray], cap: float = CAP, rest_share: float = REST_SHARE
) -> dict[str, np.ndarray]:
    """Scale rest starts (weight 1) up to ``rest_share`` of the total, then clip every start at
    ``cap`` times the mean weight (the uniform rate), a few rounds until both hold."""
    names = list(weights)
    w = np.concatenate([weights[n] for n in names])
    rest = w == 1.0
    for _ in range(20):
        if rest.any() and (~rest).any():
            rest_mass, event_mass = w[rest].sum(), w[~rest].sum()
            if rest_mass / (rest_mass + event_mass) < rest_share:
                w[rest] *= rest_share * event_mass / ((1 - rest_share) * rest_mass)
        limit = cap * w.mean()
        if w.max() <= limit * (1 + 1e-9):
            break
        w = np.minimum(w, limit)
    edges = np.cumsum([0, *(len(weights[n]) for n in names)])
    return {n: w[a:b] for n, a, b in zip(names, edges[:-1], edges[1:], strict=True)}


def finalise_by_group(
    weights: Mapping[str, np.ndarray],
    groups: Mapping[str, str],
    cap: float = CAP,
    rest_share: float = REST_SHARE,
) -> dict[str, np.ndarray]:
    """``finalise`` within each group of episodes (e.g. hand and pincer), each group then scaled
    to a mean weight of 1, so a group keeps the share of draws it has under uniform sampling and
    weighting only moves draws inside it."""
    out: dict[str, np.ndarray] = {}
    for group in sorted(set(groups.values())):
        part = finalise({n: w for n, w in weights.items() if groups[n] == group}, cap, rest_share)
        mean = np.concatenate(list(part.values())).mean()
        out |= {n: w / mean for n, w in part.items()}
    return {n: out[n] for n in weights}


def collect_signals(
    paths: Sequence[Path], cache: Path | None = None
) -> tuple[list[dict[str, np.ndarray]], list[float], list[str]]:
    """Signals, ``dt`` and effector kind of every episode of ``paths`` (``episode_signals``)."""
    signals, dts, kinds = [], [], []
    for i, path in enumerate(paths):
        s, dt, kind = episode_signals(path, cache)
        signals.append(s)
        dts.append(dt)
        kinds.append(kind)
        if (i + 1) % 100 == 0:
            log.info("signals of %d / %d episodes", i + 1, len(paths))
    return signals, dts, kinds


def index_from(
    paths: Sequence[Path],
    signals: Sequence[Mapping[str, np.ndarray]],
    dts: Sequence[float],
    kinds: Sequence[str],
    span_steps: int,
    scales: Mapping[str, tuple[float, float]],
    cap: float = CAP,
) -> dict[str, object]:
    """Start weights of every episode of ``paths`` and how they were made (JSON-ready).

    Args:
        paths, signals, dts, kinds: The training episodes and their ``collect_signals`` (the
            kinds group the episodes for ``finalise_by_group``).
        span_steps: Model steps x substeps of one sequence (its span is this x the episode's
            observation spacing in env steps).
        scales: ``fit_scales`` of these or a larger set of episodes (two training sets weighted
            with the same scales mark the same physical events).
        cap: ``CAP``.
    """
    flags = [event_flags(s, scales) for s in signals]
    raw = {
        p.name: raw_weights(f, span_steps * spacing(dt))
        for p, f, dt in zip(paths, flags, dts, strict=True)
    }
    weights = finalise_by_group(raw, {p.name: k for p, k in zip(paths, kinds, strict=True)}, cap)
    return {
        "scales": dict(scales),
        "z": Z,
        "cap": cap,
        "saturate": SATURATE,
        "rest_share": REST_SHARE,
        "event_step_share": float(np.mean(np.concatenate(flags))),
        "flags": {p.name: np.flatnonzero(f).tolist() for p, f in zip(paths, flags, strict=True)},
        "weights": {k: np.round(v, 5).tolist() for k, v in weights.items()},
    }


def build_event_index(
    paths: Sequence[Path],
    span_steps: int,
    cache: Path | None = None,
    use: Sequence[str] = SIGNALS,
    cap: float = CAP,
) -> dict[str, object]:
    """``index_from`` with the scales fitted on ``paths`` themselves."""
    signals, dts, kinds = collect_signals(paths, cache)
    return index_from(paths, signals, dts, kinds, span_steps, fit_scales(signals, use), cap)


def window_share(
    flags: Mapping[str, np.ndarray],
    weights: Mapping[str, np.ndarray] | None,
    steps: int,
    substeps: int,
    gap: int,
) -> float:
    """Expected share of drawn windows that contain a flagged step.

    A sequence starting at ``s`` has ``steps`` windows of ``substeps`` observations ``gap`` env
    steps apart (as ``data.sequence_with_ends``). Starts are drawn in proportion to ``weights``
    (``None``: every start of every episode equally likely, the uniform sampler).

    Args:
        flags: ``{episode: [T] bool}``, e.g. event steps or privileged events (for a table).
        weights: ``{episode: [n_starts]}`` start weights, or ``None``.
        steps, substeps, gap: The sequence geometry.
    """
    span = steps * substeps * gap
    total, mass = 0.0, 0.0
    for name, f in flags.items():
        csum = np.concatenate([[0], np.cumsum(f.astype(int))])
        n = len(f) - span + 1
        starts = np.arange(n)
        hits = np.zeros(n)
        for j in range(steps):
            last = starts + j * substeps * gap + (substeps - 1) * gap
            first = last - (substeps - 1) * gap
            hits += (csum[last + 1] - csum[first]) > 0
        w = np.ones(n) if weights is None else np.asarray(weights[name], dtype=np.float64)
        total += float((w * hits / steps).sum())
        mass += float(w.sum())
    return total / mass


def load_start_weights(path: Path) -> dict[str, np.ndarray]:
    """``{episode file name: start weights}`` of an index written by ``main``."""
    raw = json.loads(Path(path).read_text())
    return {k: np.asarray(v, dtype=np.float64) for k, v in raw["weights"].items()}


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--cache", type=Path, default=None, help="frame cache to read from")
    parser.add_argument("--span", type=int, default=32, help="model steps x substeps")
    parser.add_argument("--signals", nargs="+", choices=ALL_SIGNALS, default=SIGNALS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    index = build_event_index(args.paths, args.span, args.cache, args.signals)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(index))
    log.info("event steps %.3f of all steps", index["event_step_share"])


if __name__ == "__main__":
    main()
