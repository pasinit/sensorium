"""Linear probes of the encoded states against privileged simulator labels.

**Evaluation only.** The privileged arrays of an episode (``episode.privileged``) are read
here and nowhere else in ``sensorium.model``: the training path loads episodes with
``privileged=False`` and builds batches from ``model_inputs`` (tests enforce both). The labels
never enter a loss; they say what the trained states contain.

Labels per observation window (the K observations a state is a function of), from the H1
episodes' ``privileged/contacts`` ``[T, 16, 6]`` (body1, body2, normal force, x, y, z; the 16
strongest contacts, ``-1`` padded), ``meta.body_names`` and ``privileged/tip_position``:

* ``contact_left`` / ``contact_right``: any contact of a left (``lh_*``, ``lp_*``) / right
  (``rh_*``, ``rp_*``) hand or pincer body in the window;
* ``contact_object``: any contact between an ``obj_*`` body and a hand or pincer body (an object
  resting on the table is always in contact with the table, which is not an interaction; PR #15's
  8-episode runs counted any object contact, positive in 74% of windows, here it would be 100%);
* ``log_force``: mean over the window of ``log1p`` of the summed normal force;
* ``tip``: the 6-d world position of the two hand tips at the window's last step.

A probe is a ridge regression from the (centred and scaled) state to the label, fitted and scored
with **leave-one-episode-out** folds (every window of an episode is on the same side of the
split), the ridge strength chosen by an inner episode-grouped 3-fold on the training episodes.
Binary labels are scored by AUROC of the ridge output (0.5 = chance; threshold-free because the
classes are imbalanced), continuous ones by R^2 of the pooled held-out predictions.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sensorium import caused_change as cc
from sensorium.babble_stats import AIR_M, CONTACT_N, runs, tilt_degrees
from sensorium.episode import Episode
from sensorium.model.data import build, load_episode, spacing
from sensorium.model.sensory_layer import SensoryLayer

BINARY = ("contact_left", "contact_right", "contact_object")
CONTINUOUS = ("log_force", "tip")
HIDDEN = ("hidden_log_mass", "hidden_friction", "hidden_softness")
"""Hidden-property labels of the object a hand or pincer touches in the window (NaN if none, or
if the episode records no ``hidden_properties``): log mass, friction, log contact softness
(``solref_s``). Fixed per object and episode, so a state can only know them by acting on the
object (the effort and touch of the push) and not by looking at it."""
LEFT_PREFIXES = ("lh_", "lp_")
RIGHT_PREFIXES = ("rh_", "rp_")
OBJECT_PREFIX = "obj_"
INNER_FOLDS = 3
"""Episode-grouped folds of the inner search of the ridge strength."""
ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
"""Ridge strengths as multiples of the number of features, searched by the inner folds."""


@dataclass
class ProbeData:
    """States and labels of a set of windows.

    Attributes:
        features: Named feature matrices ``[N, D_name]`` (``"state"``, and ``"raw"`` for the
            non-visual inputs of the same windows).
        labels: Label name -> ``[N]`` or ``[N, 6]`` array.
        groups: ``[N]`` episode index of each window.
    """

    features: dict[str, np.ndarray]
    labels: dict[str, np.ndarray]
    groups: np.ndarray


def privileged_labels(episode: Episode) -> dict[str, np.ndarray]:
    """Per-step labels of an H1 episode (see the module docstring).

    Raises:
        KeyError: if the episode has no ``contacts`` / ``tip_position`` / ``body_names``
            (only the humanoid records them).
    """
    contacts = episode.privileged["contacts"]
    names = list(episode.meta["body_names"])  # type: ignore[call-overload]
    ids = {
        "left": [i for i, n in enumerate(names) if n.startswith(LEFT_PREFIXES)],
        "right": [i for i, n in enumerate(names) if n.startswith(RIGHT_PREFIXES)],
        "object": [i for i, n in enumerate(names) if n.startswith(OBJECT_PREFIX)],
    }
    valid = (contacts[..., 0] >= 0) & (contacts[..., 2] > 0)
    b1, b2 = contacts[..., 0].astype(int), contacts[..., 1].astype(int)

    def touching(group: list[int], other: list[int] | None = None) -> np.ndarray:
        if other is None:
            hit = np.isin(b1, group) | np.isin(b2, group)
        else:
            hit = (np.isin(b1, group) & np.isin(b2, other)) | (
                np.isin(b2, group) & np.isin(b1, other)
            )
        return (hit & valid).any(axis=1)

    force = np.where(valid, contacts[..., 2], 0.0).sum(axis=1)
    return {
        "contact_left": touching(ids["left"]),
        "contact_right": touching(ids["right"]),
        "contact_object": touching(ids["object"], ids["left"] + ids["right"]),
        "log_force": np.log1p(force),
        "tip": episode.privileged["tip_position"].reshape(len(contacts), -1),
    }


def consequence_steps(
    episode: Episode, horizon: int = 10, min_move: float = 0.01
) -> dict[str, np.ndarray]:
    """Per-step ``hand_object_contact``, ``consequence`` and ``event`` flags of an H1 episode.

    ``consequence[t]``: a hand or pincer body touches object k at ``t`` and that object ends
    ``horizon`` steps (0.5 s at 20 Hz) later more than ``min_move`` metres from where it was at
    ``t``. ``event[t]``: inside a Track A caused-change event (``event_steps``; absent when the
    episode lacks ``object_contact``). Used **offline to choose training sequences**
    (``subsets.py``); never an input.

    Raises:
        KeyError: if the episode has no ``contacts`` / ``object_pose`` / ``body_names``.
    """
    pose = episode.privileged["object_pose"][:, :, :3]
    steps = len(pose)
    touched = hand_object_touch(episode)
    motion = np.zeros_like(touched, dtype=float)
    if pose.shape[1]:
        motion[: steps - horizon] = np.linalg.norm(pose[horizon:] - pose[:-horizon], axis=2)
    try:
        event = event_steps(episode)
    except KeyError:
        event = np.zeros(steps, dtype=bool)
    return {
        "hand_object_contact": touched.any(axis=1),
        "consequence": (touched & (motion > min_move)).any(axis=1),
        "event": event,
    }


def hand_object_touch(episode: Episode) -> np.ndarray:
    """``[T, n_objects]`` True where a hand or pincer body touches that object (force > 0)."""
    contacts = episode.privileged["contacts"]
    names = list(episode.meta["body_names"])  # type: ignore[call-overload]
    object_names = list(episode.meta["object_names"])  # type: ignore[call-overload]
    hand = [i for i, n in enumerate(names) if n.startswith(LEFT_PREFIXES + RIGHT_PREFIXES)]
    obj_of = {i: object_names.index(n) for i, n in enumerate(names) if n in object_names}
    touched = np.zeros((len(contacts), max(len(object_names), 1)), dtype=bool)
    for t in range(len(contacts)):
        for b1, b2, force in contacts[t, :, :3]:
            for a, b in ((int(b1), int(b2)), (int(b2), int(b1))):
                if force > 0 and a in obj_of and b in hand:
                    touched[t, obj_of[a]] = True
    return touched


def mark_prop_events(episode: Episode, touching: np.ndarray, out: np.ndarray) -> None:
    """Mark in ``out`` ``WINDOW_STEPS`` around each drawer, flap or button crossing with contact."""
    priv = episode.privileged
    prop_state, prop_names = priv.get("prop_state"), episode.meta.get("prop_names")
    if prop_state is None or not isinstance(prop_names, list) or not len(prop_names):
        return
    travel = np.asarray(prop_state, dtype=float)
    near = touching.any(axis=1) | (np.asarray(priv["peak_force"])[:, 1] >= CONTACT_N)
    near = np.convolve(near.astype(float), np.ones(2 * cc.WINDOW_STEPS + 1), mode="same") > 0
    above = travel >= cc.JOINT_FRACTION
    for k, name in enumerate(prop_names):
        if not name.startswith(("prop_drawer", "prop_flap", "prop_button")):
            continue
        for t in np.flatnonzero(np.diff(above[:, k].astype(int)) != 0) + 1:
            if near[t]:
                out[max(t - cc.WINDOW_STEPS, 0) : t + cc.WINDOW_STEPS + 1] = True


def event_steps(episode: Episode) -> np.ndarray:
    """``[T]`` True inside a caused-change event of Track A (``sensorium.caused_change``).

    Spans, with Track A's thresholds: a contact bout through ``WINDOW_STEPS`` after it ends when the
    object ends more than ``DISPLACED_M`` away (displaced) or a non-spherical object tips past
    ``TOPPLE_DEG`` (toppled); free-fall runs (dropped); and ``WINDOW_STEPS`` around each crossing of
    half the range of a drawer, flap or button with an effector near (opened, closed, pressed).
    Lifted, stacked, pushed-to-target and body-moved are not marked (rare or absent in the
    hand programs). Offline, to choose windows only.

    Raises:
        KeyError: if the episode lacks ``object_pose`` or ``object_contact``.
    """
    priv = episode.privileged
    pose = np.asarray(priv["object_pose"], dtype=float)
    touching = np.asarray(priv["object_contact"]) >= CONTACT_N
    steps, last = pose.shape[0], pose.shape[0] - 1
    names = episode.meta.get("object_names")
    round_objects = cc.spherical(names if isinstance(names, list) else ["obj"] * pose.shape[1])
    tilt = tilt_degrees(pose[:, :, 3:7])
    out = np.zeros(steps, dtype=bool)
    for i, bouts in enumerate(cc.contact_bouts(touching)):
        for start, stop in bouts:
            end = min(stop + cc.WINDOW_STEPS, last)
            moved = np.linalg.norm(pose[end, i, :2] - pose[start, i, :2]) > cc.DISPLACED_M
            tipped = (
                not round_objects[i]
                and tilt[start, i] < cc.TOPPLE_DEG <= tilt[start : end + 1, i].max()
            )
            if moved or tipped:
                out[start : end + 1] = True
    airborne = (pose[:, :, 2] - pose[:3, :, 2].mean(axis=0) >= cc.FREE_FALL_M) & ~touching
    for i in range(pose.shape[1]):
        for start, stop in runs(airborne[:, i], 2):
            out[start:stop] = True
    mark_prop_events(episode, touching, out)
    return out


def hidden_properties(episode: Episode) -> np.ndarray:
    """``[n_objects, 3]`` hidden mass, friction and ``solref_s`` of an episode's objects.

    Raises:
        KeyError: if the episode records no ``hidden_properties``.
    """
    return np.asarray(episode.privileged["hidden_properties"], dtype=float)[0]


def hidden_window_labels(
    episode: Episode, ends: np.ndarray, substeps: int, gap: int
) -> dict[str, np.ndarray]:
    """Hidden-property labels (``HIDDEN``) of the windows ending at ``ends``; NaN without contact.

    The object is the one a hand or pincer touches for the most steps of the window. Episodes
    that record no hidden properties (random play) give NaN everywhere.
    """
    try:
        hidden = hidden_properties(episode)
    except KeyError:
        return {n: np.full(len(ends), np.nan) for n in HIDDEN}
    return hidden_from(hidden, hand_object_touch(episode), ends, substeps, gap)


def hidden_from(
    hidden: np.ndarray, touched: np.ndarray, ends: np.ndarray, substeps: int, gap: int
) -> dict[str, np.ndarray]:
    """``hidden_window_labels`` from the ``[n_obj, 3]`` properties and ``[T, n_obj]`` touch."""
    idx = ends[:, None] - gap * (substeps - 1 - np.arange(substeps))[None]
    counts = touched[idx].sum(axis=1)  # [N, n_obj] steps of the window touching each object
    best = counts.argmax(axis=1)
    has = counts.max(axis=1) > 0
    picked = hidden[np.minimum(best, len(hidden) - 1)]
    nan = np.full(len(ends), np.nan)
    return {
        "hidden_log_mass": np.where(has, np.log(picked[:, 0]), nan),
        "hidden_friction": np.where(has, picked[:, 1], nan),
        "hidden_softness": np.where(has, np.log(picked[:, 2]), nan),
    }


def window_labels(
    labels: Mapping[str, np.ndarray], ends: np.ndarray, substeps: int, gap: int
) -> dict[str, np.ndarray]:
    """Labels of the windows of ``substeps`` observations ``gap`` apart ending at ``ends``."""
    idx = ends[:, None] - gap * (substeps - 1 - np.arange(substeps))[None]
    out: dict[str, np.ndarray] = {}
    for name in BINARY:
        out[name] = labels[name][idx].any(axis=1).astype(np.float64)
    out["log_force"] = labels["log_force"][idx].mean(axis=1)
    out["tip"] = labels["tip"][ends].astype(np.float64)
    return out


def collect(
    layer: SensoryLayer,
    paths: Sequence[Path],
    stride: int,
    drop: Sequence[str] = (),
    raw: bool = True,
) -> ProbeData:
    """Encode windows of episodes and attach their labels (episodes without labels are skipped).

    ``labels`` also holds the ``HIDDEN`` labels (NaN where undefined), read by ``predict_hidden``.

    Args:
        layer: The encoder.
        paths: Episode files (the encoder should not have been trained on them for a
            held-out test).
        stride: Env steps between windows.
        drop: Senses masked while encoding (``SensoryLayer.encode``).
        raw: Also build the raw-input features.

    Raises:
        ValueError: if no episode has privileged labels.
    """
    model = layer.model
    k = model.cfg.substeps
    feats: dict[str, list[np.ndarray]] = {"state": [], "raw": []}
    labels: dict[str, list[np.ndarray]] = {n: [] for n in (*BINARY, *CONTINUOUS, *HIDDEN)}
    groups: list[np.ndarray] = []
    for g, path in enumerate(paths):
        episode = Episode.load(path)  # privileged labels are read below, and only there
        try:
            lab = privileged_labels(episode)
        except KeyError:
            continue
        states, ends = layer.encode(episode, stride=stride, drop=drop)
        feats["state"].append(states)
        if raw:
            feats["raw"].append(_raw_features(layer, episode, ends))
        for name, value in window_labels(lab, ends, k, spacing(episode.dt)).items():
            labels[name].append(value)
        hidden = hidden_window_labels(episode, ends, k, spacing(episode.dt))
        for name, value in hidden.items():
            labels[name].append(value)
        groups.append(np.full(len(ends), g))
    if not groups:
        raise ValueError("no episode has privileged contact labels (only the H1 records them)")
    return ProbeData(
        {n: np.concatenate(v) for n, v in feats.items() if v},
        {n: np.concatenate(v) for n, v in labels.items()},
        np.concatenate(groups),
    )


def _raw_features(layer: SensoryLayer, episode: Episode, ends: np.ndarray) -> np.ndarray:
    """The non-visual model inputs of the windows (last observation of each), flattened.

    A reference for what the states could contain: touch (log1p), proprio groups and audio in
    the model's own normalisation, without the frames.
    """
    senses = tuple(s for s in layer.model.senses if s != "vision") or ("audio",)
    loaded = load_episode(episode, layer.space, senses)
    arrays = build(loaded, layer.space, senses, ends, layer.model.cfg.substeps)
    return np.concatenate([arrays[s][:, -1] for s in senses], axis=1)


def episode_folds(groups: np.ndarray, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(train, test)`` window indices of ``k`` folds that never split an episode."""
    episodes = np.unique(groups)
    k = min(k, len(episodes))
    folds = []
    for i in range(k):
        test_eps = episodes[i::k]
        test = np.isin(groups, test_eps)
        folds.append((np.flatnonzero(~test), np.flatnonzero(test)))
    return folds


def _prepare(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centre both on the training mean and divide by the training RMS (one global scale).

    Not standardised per column: a rarely active touch channel would get a huge weight.
    """
    mean = x_train.mean(0)
    scale = float(np.sqrt(((x_train - mean) ** 2).mean())) or 1.0
    return (x_train - mean) / scale, (x_test - mean) / scale


def ridge_path(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alphas: Sequence[float]
) -> np.ndarray:
    """Ridge predictions for several strengths from one SVD: ``[len(alphas), n_test, n_out]``."""
    a, b = _prepare(x_train, x_test)
    y_mean = y_train.mean(0)
    u, sv, vt = np.linalg.svd(a, full_matrices=False)
    uty = u.T @ (y_train - y_mean)
    bv = b @ vt.T
    return np.stack([bv @ ((sv / (sv**2 + alpha))[:, None] * uty) + y_mean for alpha in alphas])


def ridge_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float
) -> np.ndarray:
    """Ridge regression fitted on centred, globally scaled features; predictions for ``x_test``."""
    return ridge_path(x_train, y_train, x_test, [alpha])[0]


def choose_alphas(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, blocks: Sequence[slice]
) -> list[float]:
    """Per block of output columns, the ridge strength (``ALPHA_GRID`` x number of features)
    with the lowest error in an inner episode-grouped 3-fold."""
    d = x.shape[1]
    if len(np.unique(groups)) < 2:
        return [d] * len(blocks)
    err = np.zeros((len(ALPHA_GRID), len(blocks)))
    for tr, te in episode_folds(groups, INNER_FOLDS):
        paths = ridge_path(x[tr], y[tr], x[te], [m * d for m in ALPHA_GRID])
        for j, block in enumerate(blocks):
            err[:, j] += ((paths[:, :, block] - y[te][:, block]) ** 2).sum(axis=(1, 2))
    return [ALPHA_GRID[int(i)] * d for i in err.argmin(axis=0)]


def choose_alpha(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """The ridge strength for all output columns of ``y`` together (see ``choose_alphas``)."""
    return choose_alphas(x, y, groups, [slice(0, y.shape[1])])[0]


def cross_validated_blocks(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, blocks: Sequence[slice]
) -> np.ndarray:
    """Held-out predictions from leave-one-episode-out ridge probes, one strength per block."""
    pred = np.zeros_like(y, dtype=np.float64)
    for tr, te in episode_folds(groups, len(np.unique(groups))):
        alphas = choose_alphas(x[tr], y[tr], groups[tr], blocks)
        for block, alpha in zip(blocks, alphas, strict=True):
            pred[te, block] = ridge_predict(x[tr], y[tr][:, block], x[te], alpha)
    return pred


def cross_validated_predictions(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Held-out predictions from leave-one-episode-out ridge probes."""
    return cross_validated_blocks(x, y, groups, [slice(0, y.shape[1])])


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney, ties averaged); NaN if a class is missing."""
    pos = label > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    ranks = (ends - (counts - 1) / 2.0)[inverse.ravel()]  # 1-based ranks, ties averaged
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def r_squared(pred: np.ndarray, target: np.ndarray) -> float:
    """Pooled R^2 of held-out predictions (all output dimensions together)."""
    sse = float(((pred - target) ** 2).sum())
    sst = float(((target - target.mean(0)) ** 2).sum())
    return 1.0 - sse / sst if sst > 0 else float("nan")


@dataclass
class Predictions:
    """Out-of-fold probe predictions of every label, with what is needed to score them again.

    Attributes:
        pred, target: ``[N, 10]`` predictions and labels (the three binary labels, force, tip).
        blocks: Column slice of each label, in the order of ``names``.
        groups: ``[N]`` episode of each window.
        names: The labels (default the contact battery, ``BINARY + CONTINUOUS``).
        binary: The labels of ``names`` scored by AUROC; the others are scored by R^2.
    """

    pred: np.ndarray
    target: np.ndarray
    blocks: list[slice]
    groups: np.ndarray
    names: tuple[str, ...] = (*BINARY, *CONTINUOUS)
    binary: tuple[str, ...] = BINARY

    def subset(self, names: Sequence[str]) -> "Predictions":
        """The same predictions restricted to the labels ``names`` (so ``mean`` averages only
        those, e.g. the three contact AUROCs without log force).

        Raises:
            ValueError: if a name is not one of ``self.names``.
        """
        missing = set(names) - set(self.names)
        if missing:
            raise ValueError(f"labels {sorted(missing)} not in {self.names}")
        block = dict(zip(self.names, self.blocks, strict=True))
        cols = np.concatenate([np.arange(block[n].start, block[n].stop) for n in names])
        edges = np.cumsum([0, *(block[n].stop - block[n].start for n in names)])
        blocks = [slice(int(x), int(y)) for x, y in zip(edges[:-1], edges[1:], strict=True)]
        binary = tuple(n for n in names if n in self.binary)
        return Predictions(
            self.pred[:, cols], self.target[:, cols], blocks, self.groups, tuple(names), binary
        )

    def scores(self, rows: np.ndarray | None = None) -> dict[str, float]:
        """Scores of every label (and their ``mean``) on all windows or on the windows ``rows``."""
        rows = np.arange(len(self.groups)) if rows is None else rows
        scores: dict[str, float] = {}
        for name, block in zip(self.names, self.blocks, strict=True):
            p, y = self.pred[rows][:, block], self.target[rows][:, block]
            if name in self.binary:
                scores[f"{name}_auroc"] = auroc(p[:, 0], y[:, 0])
            else:
                scores[f"{name}_r2"] = r_squared(p, y)
        terms = [2 * (v - 0.5) if k.endswith("auroc") else v for k, v in scores.items()]
        finite = [t for t in terms if np.isfinite(t)]
        scores["mean"] = float(np.mean(finite)) if finite else float("nan")
        return scores

    def bootstrap(self, draws: int, rng: np.random.Generator) -> list[dict[str, float]]:
        """Scores on ``draws`` resamples of the episodes (with replacement), same pooled protocol.

        Each element of the list is the score dict of one resample; call it with the same ``rng``
        state for two encoders on the same windows to get a **paired** bootstrap.
        """
        episodes = np.unique(self.groups)
        rows_of = {e: np.flatnonzero(self.groups == e) for e in episodes}
        out = []
        for _ in range(draws):
            picked = rng.choice(episodes, len(episodes))
            out.append(self.scores(np.concatenate([rows_of[e] for e in picked])))
        return out


def predict_probes(
    data: ProbeData,
    feature: str,
    names: tuple[str, ...] = (*BINARY, *CONTINUOUS),
    binary: tuple[str, ...] = BINARY,
) -> Predictions:
    """Leave-one-episode-out ridge predictions of the labels ``names`` (default: the contact
    battery) from one feature matrix; ``binary`` are scored by AUROC."""
    x = data.features[feature]
    columns = [data.labels[n].reshape(len(x), -1) for n in names]
    y = np.concatenate(columns, axis=1)
    edges = np.cumsum([0, *(c.shape[1] for c in columns)])
    blocks = [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    pred = cross_validated_blocks(x, y, data.groups, blocks)
    return Predictions(pred, y, blocks, data.groups, names, binary)


def score_probes(data: ProbeData, feature: str) -> dict[str, float]:
    """Held-out scores of every label from one feature matrix, plus their ``mean``.

    ``mean`` averages the AUROC rescaled to ``2 * (AUROC - 0.5)`` and the R^2 values, ignoring
    NaNs, so 0 is chance for every term.
    """
    return predict_probes(data, feature).scores()


def predict_hidden(data: ProbeData, feature: str, min_windows: int = 30) -> dict[str, float]:
    """R^2 of leave-one-episode-out ridge probes of the ``HIDDEN`` labels, on contact windows.

    Only windows that have the labels (a hand or pincer touches an object) are used. NaN scores
    if fewer than ``min_windows`` windows or fewer than three episodes have them.

    Returns:
        ``{"<label>_r2": ..., "hidden_windows": n, "hidden_episodes": e}``.
    """
    y = np.stack([data.labels[n] for n in HIDDEN], axis=1)
    rows = np.flatnonzero(np.isfinite(y).all(axis=1))
    out = {
        "hidden_windows": float(len(rows)),
        "hidden_episodes": float(len(np.unique(data.groups[rows]))),
    }
    if len(rows) < min_windows or out["hidden_episodes"] < 3:
        return out | {f"{n}_r2": float("nan") for n in HIDDEN}
    blocks = [slice(i, i + 1) for i in range(len(HIDDEN))]
    pred = cross_validated_blocks(data.features[feature][rows], y[rows], data.groups[rows], blocks)
    return out | {
        f"{n}_r2": r_squared(pred[:, b], y[rows][:, b]) for n, b in zip(HIDDEN, blocks, strict=True)
    }


def run_probes(
    layer: SensoryLayer,
    paths: Sequence[Path],
    stride: int = 2,
    drop: Sequence[str] = (),
    raw: bool = True,
) -> dict[str, dict[str, float]]:
    """Probe scores of the layer's states and (``raw``) of the raw non-visual inputs.

    Returns:
        ``{"state": {...}, "raw": {...}}`` with the keys of ``score_probes`` plus ``windows``
        and ``positive_rate_*`` (the share of windows with each binary label).
    """
    data = collect(layer, paths, stride, drop, raw)
    out = {f: score_probes(data, f) for f in data.features}
    for scores in out.values():
        scores["windows"] = float(len(data.groups))
        for name in BINARY:
            scores[f"positive_rate_{name}"] = float(data.labels[name].mean())
    return out


OUTCOME_HORIZON = cc.WINDOW_STEPS
"""Steps ahead (1 s at 20 Hz) of the ``displaced`` and ``toppled`` outcome labels."""
SPEED_SCALE = 0.01
"""m/s: the unit of the ``log_speed`` outcome label."""


def _ahead(values: np.ndarray, horizon: int) -> np.ndarray:
    """``[T, ...]`` value at ``min(t + horizon, T - 1)`` for every step ``t``."""
    t = np.arange(len(values))
    return values[np.minimum(t + horizon, len(values) - 1)]


def _future_max(values: np.ndarray, horizon: int) -> np.ndarray:
    """``[T, n]`` maximum of ``values`` over ``(t, t + horizon]`` (``-inf`` at the last step)."""
    out = np.full(values.shape, -np.inf)
    for k in range(1, horizon + 1):
        out[:-k] = np.maximum(out[:-k], values[k:])
    return out


def outcome_steps(episode: Episode, horizon: int = OUTCOME_HORIZON) -> dict[str, np.ndarray]:
    """Per-step outcome labels of an episode (``[T]`` each), defined in
    ``sensorium.model.outcomes`` (which keeps this module the only reader of privileged state).

    Raises:
        KeyError: if the episode lacks ``privileged/object_pose`` or ``object_contact``.
    """
    pose = np.asarray(episode.privileged["object_pose"], dtype=float)
    touching = np.asarray(episode.privileged["object_contact"]) >= CONTACT_N
    names = episode.meta.get("object_names")
    round_objects = cc.spherical(names if isinstance(names, list) else ["obj"] * pose.shape[1])
    xy = pose[:, :, :2]
    moved = np.linalg.norm(_ahead(xy, horizon) - xy, axis=-1) > cc.DISPLACED_M
    kernel = np.ones(2 * horizon + 1)
    near = np.stack(
        [np.convolve(touching[:, i], kernel, mode="same") > 0 for i in range(touching.shape[1])],
        axis=1,
    )
    rest = pose[:3, :, 2].mean(axis=0)
    lifted = (pose[:, :, 2] - rest >= AIR_M) & touching
    tilt = tilt_degrees(pose[:, :, 3:7])
    tips = (tilt < cc.TOPPLE_DEG) & (_future_max(tilt, horizon) >= cc.TOPPLE_DEG)
    tips[:, round_objects] = False
    step = np.linalg.norm(np.diff(pose[:, :, :3], axis=0), axis=-1) / episode.dt
    speed = np.concatenate([np.zeros(1), step.max(axis=1)])
    return {
        "displaced": (moved & near).any(axis=1),
        "lifted": lifted.any(axis=1),
        "toppled": tips.any(axis=1),
        "log_speed": np.log1p(speed / SPEED_SCALE),
    }
