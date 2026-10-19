"""A1 "babbling quality" statistics of seated H1 episodes (``docs/tracks.md``, experiment A1).

Answers whether task-free experience engages objects instead of flailing or
slamming. All numbers come from the episode's ``privileged`` logs and ``meta``
(never model input), so a source can be judged without a model. Definitions
(rows are the A1 table in the D0.3 design memo):

* contact: normal force above ``CONTACT_N``; *effector* = hands or pincers, *limb* =
  every other robot part (``h1_forces``);
* row 1 ``object_contact_rate``: share of steps with an effector-object contact;
* row 2 ``bouts_per_minute``: effector-object contact onsets per minute (a bout
  ends after a gap of more than ``BOUT_GAP`` steps and lasts at least 2 steps);
* row 3 ``objects_touched``: distinct objects with a bout;
* row 4 ``object_motion_rate`` (and ``two_object_motion_rate``, ``in_air_rate``,
  ``flipped_rate``): shares of steps with >= 1 / >= 2 objects *moved by an effector*:
  moving faster than ``MOVING_MPS`` in a bout of motion that an effector contact on that
  object preceded by at most ``ATTRIBUTION_STEPS`` (so a ball that rolls on by itself does
  not count), an object ``AIR_M`` above its resting height, or a box tipped past
  ``TIPPED_DEG`` (the metrics of Sancaktar et al., NeurIPS 2022);
  ``unattributed_motion_rate`` is the share of steps with motion no effector caused;
* row 5 ``lifts``: an object held ``LIFT_M`` above its resting height for
  ``LIFT_STEPS`` steps with an effector touching it at some point of the window;
* row 6 ``free_space_rate``: share of tip-steps more than ``FREE_SPACE_M`` above the table;
* row 7 ``voxel_coverage``: share of ``VOXEL_M`` voxels of the tabletop workspace visited by a tip;
* row 8 ``tip_entropy_gap``: Kozachenko-Leonenko (``KNN_K`` neighbours) entropy of the tip
  positions minus the log volume of the workspace box (nats; 0 is uniform, more negative
  is more concentrated);
* row 9 ``object_cells``: ``CELL_M`` cells of the table an object rests in at some step;
* row 10 ``skin_regions``: distinct effector skin regions (a finger bone, a palm, a pad) with
  a dense-skin channel above ``SKIN_N`` at some step, out of ``skin_region_total``;
* row 12 ``limb_peak_n``, ``limb_slam_steps`` (> ``SLAM_N``): non-effector part vs table or object;
* row 13 ``self_peak_n``, ``self_rate`` (> ``SELF_N``): contact between different limbs;
* row 14 ``fell``, ``action_jerk``, ``action_saturation``.

* row 11 ``sound_pairs``: distinct (robot voice, surface voice) pairs, of ``SOUND_PAIRS``, that
  had a new contact above ``CONTACT_N`` at some step (``privileged/impact_pairs``): which
  pairings of the contact-sound voices an episode excites (arms and hands against the table
  and the four object materials: 10 pairs).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields

import numpy as np

from sensorium.bodies.h1_scene import edge_base, table_edges
from sensorium.episode import Episode

CONTACT_N = 0.5  # N: normal force that counts as contact
BOUT_GAP = 3  # steps without contact that end a bout
MOVING_MPS = 0.02  # m/s: an object faster than this is moving
AIR_M = 0.03  # m above the resting height: "in the air"
LIFT_M = 0.05  # m above the resting height held for LIFT_STEPS: a lift
LIFT_STEPS = 10  # control steps (0.5 s)
FREE_SPACE_M = 0.10  # m above the table: the tip is in free space
VOXEL_M = 0.05  # m, side of the tip-coverage voxels
CELL_M = 0.05  # m, side of the object-position cells
SLAM_N = 100.0  # N: a limb contact above this is a slam
SELF_N = 5.0  # N: a limb-limb contact above this is a self-collision
TIPPED_DEG = 60.0  # degrees from upright: a box tipped past this is flipped
ATTRIBUTION_STEPS = 20  # an effector contact this long before a motion bout (1 s) is its cause
KNN_K = 5  # neighbours of the tip-entropy estimate
SKIN_N = 0.1  # N: a dense skin channel above this is touched
ENTROPY_HEIGHT = 0.30  # m above the table: top of the tip-entropy box
SOUND_PAIRS = tuple(
    (part, surface)
    for part in ("arms", "hands")
    for surface in ("table", "wood", "plastic", "metal", "rubber")
)
WORKSPACE_Y = 0.38  # m: half width of the tip-coverage box
WORKSPACE_HEIGHT = 0.15  # m above the table: top of the tip-coverage box
TABLE_Y = 0.5  # m: half width of the table, for the object cells
PEAK_COLUMNS = {"effector_object": 0, "effector_table": 1, "limb_scene": 2, "self": 3}


@dataclass(frozen=True)
class A1Stats:
    """A1 statistics of one episode (see the module docstring for the definitions)."""

    steps: int
    object_contact_rate: float
    table_contact_rate: float
    bouts_per_minute: float
    objects_touched: int
    object_motion_rate: float
    two_object_motion_rate: float
    unattributed_motion_rate: float
    in_air_rate: float
    flipped_rate: float
    lifts: int
    free_space_rate: float
    voxel_coverage: float
    tip_entropy_gap: float
    skin_regions: frozenset[str]
    skin_region_total: int
    sound_pairs: frozenset[tuple[str, str]]
    object_cells: frozenset[int]
    table_cells: int
    object_peak_n: float
    limb_peak_n: float
    limb_slam_steps: int
    self_peak_n: float
    self_rate: float
    fell: bool
    wait_share: float
    plans_per_primitive: tuple[tuple[str, int], ...]
    action_jerk: float
    action_saturation: float


def runs(flags: np.ndarray, min_length: int = 1, max_gap: int = 0) -> list[tuple[int, int]]:
    """Maximal ``[start, stop)`` runs of True in ``flags``, bridging gaps of up to ``max_gap``.

    Runs shorter than ``min_length`` (after bridging) are dropped.
    """
    found: list[list[int]] = []
    for t in np.flatnonzero(flags):
        if found and t - found[-1][1] <= max_gap + 1:
            found[-1][1] = int(t) + 1
        else:
            found.append([int(t), int(t) + 1])
    return [(a, b) for a, b in found if b - a >= min_length]


def attributed_motion(moving: np.ndarray, touching: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split ``[T, n]`` object motion into motion an effector caused and motion nobody caused.

    A bout of motion (moving steps, gaps of up to ``BOUT_GAP`` bridged) is caused if the effector
    touched that object within ``ATTRIBUTION_STEPS`` before it started or while it lasted.

    Returns:
        ``(attributed, unattributed)``, both ``[T, n]`` bool, partitioning ``moving``.
    """
    attributed = np.zeros_like(moving)
    for i in range(moving.shape[1]):
        for start, stop in runs(moving[:, i], 1, BOUT_GAP):
            if touching[max(start - ATTRIBUTION_STEPS, 0) : stop, i].any():
                attributed[start:stop, i] = True
    return attributed & moving, moving & ~attributed


def knn_entropy(points: np.ndarray, k: int = KNN_K) -> float:
    """Kozachenko-Leonenko differential entropy (nats) of ``[N, d]`` points (``k`` neighbours).

    Coincident points are separated by a 0.1 mm jitter (a still tip repeats its position).
    """
    n, dim = points.shape
    jittered = points + np.random.default_rng(0).normal(0.0, 1e-4, points.shape)
    distances = np.linalg.norm(jittered[:, None] - jittered[None], axis=-1)
    radius = np.sort(distances, axis=1)[:, k]  # column 0 is the point itself
    unit_ball = math.pi ** (dim / 2) / math.gamma(dim / 2 + 1)
    digamma_n = math.log(n) - 0.5 / n  # psi(n) for large n
    digamma_k = -0.5772156649 + sum(1.0 / j for j in range(1, k))  # psi(k)
    return float(digamma_n - digamma_k + math.log(unit_ball) + dim * np.log(radius).mean())


def sound_pairs(episode: Episode) -> frozenset[tuple[str, str]]:
    """Voice pairs of ``SOUND_PAIRS`` with a new contact above ``CONTACT_N`` (none if unlogged)."""
    pairs = episode.privileged.get("impact_pairs")
    names = episode.meta.get("impact_voices")
    if pairs is None or not isinstance(names, list):
        return frozenset()
    strongest = pairs.max(axis=0)
    return frozenset(
        (part, surface)
        for part, surface in SOUND_PAIRS
        if strongest[names.index(part), names.index(surface)] >= CONTACT_N
    )


def skin_regions(names: list[str], touch: np.ndarray) -> tuple[frozenset[str], int]:
    """Effector skin regions touched at some step, and how many regions there are.

    A region is a dense-skin body (``hand/left/ffdistal``, ``pincer/left/left_pad``); it is
    touched if any of its channels exceeds ``SKIN_N`` at some step.
    """
    columns: dict[str, list[int]] = {}
    for column, name in enumerate(names):
        if name.startswith(("hand/", "pincer/")):
            columns.setdefault(name.rsplit("/", 1)[0], []).append(column)
    touched = {r for r, cols in columns.items() if (touch[:, cols] > SKIN_N).any()}
    return frozenset(touched), len(columns)


def moving_objects(pose: np.ndarray, dt: float) -> np.ndarray:
    """``[T, n]`` True where an object's speed exceeds ``MOVING_MPS`` (finite difference)."""
    speed = np.zeros(pose.shape[:2])
    speed[1:] = np.linalg.norm(np.diff(pose[:, :, :3], axis=0), axis=-1) / dt
    return speed > MOVING_MPS


def tilt_degrees(quat: np.ndarray) -> np.ndarray:
    """Angle (degrees) between the world z axis and the body z axis of wxyz quaternions."""
    w, x, y, z = (quat[..., i] for i in range(4))
    cos = 1.0 - 2.0 * (x * x + y * y)
    return np.asarray(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def count_lifts(height: np.ndarray, touched: np.ndarray) -> int:
    """Lifts of one object: runs of ``LIFT_STEPS`` steps at ``LIFT_M`` above rest, with contact.

    Args:
        height: ``[T]`` the object's height above its resting height.
        touched: ``[T]`` an effector touches the object.
    """
    count = 0
    for start, stop in runs(height >= LIFT_M, LIFT_STEPS):
        window = slice(max(start - 2, 0), stop + 2)
        count += int(touched[window].any())
    return count


def a1_stats(episode: Episode) -> A1Stats:
    """A1 statistics of a seated H1 episode.

    Raises:
        KeyError: if the episode lacks ``privileged/peak_force`` or ``object_contact``
            (recorded before D0.3: regenerate it) or is not a seated episode.
    """
    priv, meta = episode.privileged, episode.meta
    for key in ("peak_force", "object_contact", "object_pose", "tip_position"):
        if key not in priv:
            raise KeyError(f"episode has no privileged/{key}: regenerate it")
    table_top = meta.get("table_top")
    if table_top is None:
        raise KeyError("meta.table_top is missing: not a seated H1 episode")
    top = float(table_top)  # type: ignore[arg-type]
    steps, dt = episode.length, episode.dt
    peak, by_object, pose, tips = (
        priv["peak_force"],
        priv["object_contact"],
        priv["object_pose"],
        priv["tip_position"],
    )
    touching = by_object >= CONTACT_N  # [T, n_objects]
    contact = touching.any(axis=1)
    bouts = runs(contact, 2, BOUT_GAP)
    touched_objects = {i for a, b in bouts for i in np.flatnonzero(touching[a:b].any(axis=0))}

    moving, unattributed = attributed_motion(moving_objects(pose, dt), touching)
    rest = pose[:3, :, 2].mean(axis=0)  # resting height: the first steps, before babbling
    height = pose[:, :, 2] - rest
    names = meta.get("object_names")
    boxes = np.array(["box" in str(n) for n in names] if isinstance(names, list) else [True])
    tilt = tilt_degrees(pose[:, :, 3:7])
    flipped = (tilt > TIPPED_DEG) & boxes[None, : tilt.shape[1]]
    lifts = sum(count_lifts(height[:, i], touching[:, i]) for i in range(pose.shape[1]))

    clearance = tips[:, :, 2] - top
    settings = meta.get("settings")
    posture = str(settings.get("posture", "seated")) if isinstance(settings, dict) else "seated"
    edges = table_edges(top, edge_base(posture))
    low = np.array(meta.get("reach_x", (0.4, 0.68)), dtype=float)
    voxels = voxel_ids(tips, top, (float(low[0]), float(low[1])))
    n_voxels = (
        math.ceil((low[1] - low[0]) / VOXEL_M)
        * math.ceil(2 * WORKSPACE_Y / VOXEL_M)
        * math.ceil(WORKSPACE_HEIGHT / VOXEL_M)
    )

    plan_stats = meta.get("plan_stats")
    plan_stats = plan_stats if isinstance(plan_stats, dict) else {}
    wait_share = float(plan_stats.get("wait_steps", 0)) / (2 * steps)  # of the two arms' steps
    attempts = plan_stats.get("attempts")
    plans = (
        tuple(sorted((str(k), int(v)) for k, v in attempts.items()))
        if isinstance(attempts, dict)
        else ()
    )

    regions, region_total = skin_regions(episode.touch_names, episode.touch)
    action = episode.action.astype(float)
    jerk = np.abs(action[2:] - 2 * action[1:-1] + action[:-2]).mean() if steps > 2 else 0.0
    return A1Stats(
        steps=steps,
        object_contact_rate=float(contact.mean()),
        table_contact_rate=float((peak[:, PEAK_COLUMNS["effector_table"]] >= CONTACT_N).mean()),
        bouts_per_minute=len(bouts) / (steps * dt / 60.0),
        objects_touched=len(touched_objects),
        object_motion_rate=float((moving.sum(axis=1) >= 1).mean()),
        two_object_motion_rate=float((moving.sum(axis=1) >= 2).mean()),
        unattributed_motion_rate=float(unattributed.any(axis=1).mean()),
        in_air_rate=float((height >= AIR_M).any(axis=1).mean()),
        flipped_rate=float(flipped.any(axis=1).mean()),
        lifts=lifts,
        free_space_rate=float((clearance > FREE_SPACE_M).mean()),
        voxel_coverage=len(set(voxels.tolist())) / n_voxels,
        tip_entropy_gap=tip_entropy_gap(tips, top, (float(low[0]), float(low[1]))),
        skin_regions=regions,
        skin_region_total=region_total,
        sound_pairs=sound_pairs(episode),
        object_cells=frozenset(object_cell_ids(pose, edges).tolist()),
        table_cells=table_cell_count(edges),
        object_peak_n=float(peak[:, PEAK_COLUMNS["effector_object"]].max(initial=0.0)),
        limb_peak_n=float(peak[:, PEAK_COLUMNS["limb_scene"]].max(initial=0.0)),
        limb_slam_steps=int((peak[:, PEAK_COLUMNS["limb_scene"]] > SLAM_N).sum()),
        self_peak_n=float(peak[:, PEAK_COLUMNS["self"]].max(initial=0.0)),
        self_rate=float((peak[:, PEAK_COLUMNS["self"]] > SELF_N).mean()),
        fell=bool(meta.get("fell", False)),
        wait_share=wait_share,
        plans_per_primitive=plans,
        action_jerk=float(jerk),
        action_saturation=float((np.abs(action) >= 0.999).mean()),
    )


def tip_entropy_gap(tips: np.ndarray, top: float, reach_x: tuple[float, float]) -> float:
    """``knn_entropy`` of the tip positions minus the log volume of their workspace box."""
    box = np.array([reach_x[1] - reach_x[0], 2 * WORKSPACE_Y, ENTROPY_HEIGHT])
    return knn_entropy(tips.reshape(-1, 3)) - float(np.log(box.prod()))


def voxel_ids(tips: np.ndarray, top: float, reach_x: tuple[float, float]) -> np.ndarray:
    """Ids of the ``VOXEL_M`` voxels (of the tabletop workspace box) that the tips are in.

    Args:
        tips: ``[T, 2, 3]`` tip positions. Tips outside the box are dropped.
    """
    points = tips.reshape(-1, 3)
    origin = np.array([reach_x[0], -WORKSPACE_Y, top])
    size = np.array([reach_x[1], WORKSPACE_Y, top + WORKSPACE_HEIGHT]) - origin
    inside = ((points >= origin) & (points < origin + size)).all(axis=1)
    index = np.floor((points[inside] - origin) / VOXEL_M).astype(int)
    shape = np.ceil(size / VOXEL_M).astype(int)
    return np.asarray(np.ravel_multi_index(index.T, shape))


def table_cell_count(edges: tuple[float, float]) -> int:
    """Number of ``CELL_M`` cells of a table with the given near and far edge."""
    return math.ceil((edges[1] - edges[0]) / CELL_M) * math.ceil(2 * TABLE_Y / CELL_M)


def object_cell_ids(pose: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    """Ids of the ``CELL_M`` table cells that objects rest in, over all steps (unique)."""
    xy = pose[:, :, :2].reshape(-1, 2)
    low = np.array([edges[0], -TABLE_Y])
    shape = np.ceil((np.array([edges[1], TABLE_Y]) - low) / CELL_M).astype(int)
    index = np.floor((xy - low) / CELL_M).astype(int)
    inside = ((index >= 0) & (index < shape)).all(axis=1)
    return np.unique(np.ravel_multi_index(index[inside].T, shape))


def summarise(
    stats: Sequence[A1Stats], resamples: int = 1000, seed: int = 0
) -> dict[str, tuple[float, float, float]]:
    """Mean and 95% bootstrap interval over episodes of every numeric statistic.

    Returns:
        ``{name: (mean, low, high)}``; the cell coverage is of the union of cells over
        all episodes (as a share of the table's cells, the same for all), without an interval.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, tuple[float, float, float]] = {}
    for f in fields(A1Stats):
        if f.name in (
            "object_cells",
            "table_cells",
            "skin_regions",
            "skin_region_total",
            "sound_pairs",
            "plans_per_primitive",
        ):
            continue
        values = np.array([float(getattr(s, f.name)) for s in stats])
        means = [
            values[rng.integers(len(values), size=len(values))].mean() for _ in range(resamples)
        ]
        out[f.name] = (
            float(values.mean()),
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
        )
    cells = set().union(*(s.object_cells for s in stats))
    share = len(cells) / stats[0].table_cells
    out["object_cell_coverage"] = (share, share, share)
    regions = set().union(*(s.skin_regions for s in stats))
    pooled = len(regions) / max(stats[0].skin_region_total, 1)
    out["skin_region_coverage"] = (pooled, pooled, pooled)
    heard = set().union(*(s.sound_pairs for s in stats))
    out["sound_pair_coverage"] = (len(heard) / len(SOUND_PAIRS),) * 3
    return out
