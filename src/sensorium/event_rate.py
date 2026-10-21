"""The event rate: how much of an episode has something happening in it (Track A, event-dense data).

Label-free: computed from what physically happens in the episode (privileged object poses, the
per-step peak contact forces), not from what a program meant to do, and never model input.
Per control step, onsets of these types are found:

* ``contact_start`` / ``contact_end``: an effector starts or stops touching an object (normal force
  over ``CONTACT_N``; bouts as in ``caused_change.contact_bouts``);
* ``grasp``: an object touched and off the table (``GRASP_M`` above its resting height) for 2 steps;
* ``lift``: an object touched and ``LIFT_M`` above its resting height (its onset; no hold needed);
* ``drop``: an untouched object ``FREE_FALL_M`` above its resting height and falling;
* ``fall``: an object goes below the table's surface (off the table);
* ``topple``: a non-spherical object's tilt crosses 45 degrees (either way);
* ``impact``: the peak normal force of a class rises over ``IMPACT_N`` (from under it the step
  before): ``effector_object`` (the hands or pads hit an object), ``limb_scene`` (arms, torso or
  legs hit an object or the table: "hurting") and ``self`` (the body hits itself). Effector-table
  contacts are left out: a hand pressing or sliding on the table re-crosses any threshold all the
  time.

and one state, ``motion``: some object faster than ``MOTION_MPS``.

A window (``WINDOW_S``) is *eventful* when it holds an onset of any type or ``MOTION_STEPS`` steps
of motion; the rest share is the share of windows that are not. ``event_rate`` gives per episode the
eventful and rest shares, the share of windows with an onset (without motion), and the onsets per
minute by type.

The resting height of an object is the 5th percentile of its height over the steps it spends on the
table (objects can be re-placed or fall, so the first steps do not define it).
"""

from dataclasses import dataclass

import numpy as np

from sensorium.babble_stats import BOUT_GAP, CONTACT_N, LIFT_M, runs, tilt_degrees
from sensorium.caused_change import FREE_FALL_M, TOPPLE_DEG, spherical
from sensorium.episode import Episode

WINDOW_S = 1.0  # s: the windows the event share is counted in
GRASP_M = 0.01  # m above rest with contact: the object is held, not resting
IMPACT_N = 20.0  # N: a contact force rising past this is an impact (about a 2 kg weight)
IMPACT_CLASSES = ("effector_object", "limb_scene", "self")
MOTION_MPS = 0.05  # m/s: a visibly moving object (MOVING_MPS, 2 cm/s, catches slow creep)
MOTION_STEPS = 3  # steps of motion that make a window eventful
EVENT_TYPES = (
    "contact_start",
    "contact_end",
    "grasp",
    "lift",
    "drop",
    "fall",
    "topple",
    "impact",
)


@dataclass(frozen=True)
class EventRate:
    """Event statistics of one episode.

    Attributes:
        minutes: episode length.
        eventful: share of windows with an onset or with motion.
        onset_windows: share of windows with an onset (motion not counted).
        per_minute: onsets per minute by type (``EVENT_TYPES``).
        windows: ``[W]`` number of onsets per window (motion not counted).
        motion: ``[W]`` steps with motion per window.
    """

    minutes: float
    eventful: float
    onset_windows: float
    per_minute: dict[str, float]
    windows: np.ndarray
    motion: np.ndarray

    @property
    def rest(self) -> float:
        """Share of windows with nothing happening."""
        return 1.0 - self.eventful


def onset(flags: np.ndarray, min_length: int = 1) -> np.ndarray:
    """``[T]`` True at the first step of each run of ``flags`` of at least ``min_length``."""
    out = np.zeros(flags.shape[0], dtype=bool)
    for start, _ in runs(flags, min_length):
        out[start] = True
    return out


def resting_heights(z: np.ndarray, table_top: float) -> np.ndarray:
    """``[n]`` per object, the 5th percentile of its height over its steps on the table."""
    on_table = z > table_top - 0.01
    rest = np.empty(z.shape[1])
    for i in range(z.shape[1]):
        heights = z[on_table[:, i], i]
        rest[i] = np.percentile(heights, 5) if len(heights) else z[0, i]
    return rest


def event_steps(episode: Episode) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """``({type: [T] onset counts}, [T] motion)`` of one seated episode.

    Raises:
        KeyError: if the episode lacks ``object_pose``, ``object_contact`` or ``peak_force``.
    """
    priv, meta = episode.privileged, episode.meta
    pose, force, peaks = priv["object_pose"], priv["object_contact"], priv["peak_force"]
    top = float(meta["table_top"])  # type: ignore[arg-type]
    steps, n = pose.shape[:2]
    touching = force >= CONTACT_N
    z = pose[:, :, 2]
    above = z - resting_heights(z, top)
    velocity = np.zeros((steps, n, 3))
    velocity[1:] = np.diff(pose[:, :, :3], axis=0) / episode.dt
    speed = np.linalg.norm(velocity, axis=-1)
    round_objects = spherical(meta.get("object_names", []))  # type: ignore[arg-type]
    tilt = tilt_degrees(pose[:, :, 3:7])
    counts = {t: np.zeros(steps, dtype=int) for t in EVENT_TYPES}
    for i in range(n):
        for start, stop in runs(touching[:, i], 2, BOUT_GAP):
            counts["contact_start"][start] += 1
            if stop < steps:
                counts["contact_end"][stop] += 1
        counts["grasp"] += onset(touching[:, i] & (above[:, i] >= GRASP_M), 2)
        counts["lift"] += onset(touching[:, i] & (above[:, i] >= LIFT_M))
        falling = velocity[:, i, 2] < -0.05
        counts["drop"] += onset(~touching[:, i] & (above[:, i] >= FREE_FALL_M) & falling, 2)
        counts["fall"] += onset(z[:, i] < top - 0.02)
        if not (len(round_objects) > i and round_objects[i]):
            crossed = np.diff((tilt[:, i] >= TOPPLE_DEG).astype(int)) != 0
            counts["topple"][1:] += crossed
    columns = list(meta["peak_force_columns"])  # type: ignore[call-overload]
    for name in IMPACT_CLASSES:
        over = peaks[:, columns.index(name)] >= IMPACT_N
        counts["impact"][1:] += over[1:] & ~over[:-1]
    motion = (speed > MOTION_MPS).any(axis=1)
    return counts, motion


def event_rate(episode: Episode, window_s: float = WINDOW_S) -> EventRate:
    """The event statistics of one seated episode (see the module docstring)."""
    counts, motion = event_steps(episode)
    size = max(1, round(window_s / episode.dt))
    steps = len(motion)
    whole = steps // size * size
    total = np.stack(list(counts.values())).sum(axis=0)[:whole].reshape(-1, size).sum(axis=1)
    moving = motion[:whole].reshape(-1, size).sum(axis=1)
    minutes = steps * episode.dt / 60.0
    per_minute = {t: float(counts[t].sum()) / minutes for t in EVENT_TYPES}
    eventful = (total > 0) | (moving >= MOTION_STEPS)
    return EventRate(
        minutes=minutes,
        eventful=float(eventful.mean()),
        onset_windows=float((total > 0).mean()),
        per_minute=per_minute,
        windows=total,
        motion=moving,
    )
