"""An offline grasp library: which grasps of which object, as it lies, lift it (grasp database).

The 5-joint H1 arm cannot reach an arbitrary effector pose, so the library does not store poses to
reach. It stores *descriptors* of grasps the planner actually produced and executed, each with its
outcome, made in an isolated scene (``scripts/grasp_library.py``: the object alone on the table,
at random places, yaws and resting faces). At run time the planner draws several reachable grasp
candidates and keeps the one whose nearest library neighbours lifted most often; objects (as they
lie) with no reliable grasp are not grasp targets.

A descriptor, in the object's frame (absolute values: the boxes, cylinders and spheres are
symmetric under axis flips): the approach axis and the second frame axis (a pincer's closing axis,
a hand's palm normal), and the grasp point's offset from the object's centre divided by its half
sizes; then in the world: both axes' heights, the grasp point's height over the object's centre,
and the object's extent along each frame axis (what the effector must span). Its key is the
object's name and which of its axes is nearest to vertical (how it lies).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

WORLD_SCALE = 0.05  # m: heights and widths in the descriptor are in units of this
NEIGHBOURS = 7  # library grasps a candidate is scored by
PRIOR = 0.5, 1.0  # (successes, trials) added to the neighbours: a shrunk success rate


def grasp_key(model: mujoco.MjModel, data: mujoco.MjData, body: int) -> tuple[str, int]:
    """``(object name, index of its axis nearest to vertical)``: the object as it lies."""
    up = data.xmat[body].reshape(3, 3)[2]  # world z in the body's axes
    return str(model.body(body).name), int(np.argmax(np.abs(up)))


def grasp_features(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body: int,
    point: np.ndarray,
    approach: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    """The descriptor of a grasp at ``point`` with these world axes, on object ``body``."""
    rotation = data.xmat[body].reshape(3, 3)
    geom = int(model.body_geomadr[body])
    size = model.geom_size[geom]
    kind = int(model.geom_type[geom])
    if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        half = np.array(size[:3])
    elif kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        half = np.array([size[0], size[0], size[1]])
    else:
        half = np.full(3, size[0])
    offset = rotation.T @ (point - data.xpos[body]) / half
    across = np.cross(axis, approach)

    def width(direction: np.ndarray) -> float:  # the object's extent along a world direction
        local = np.abs(rotation.T @ direction)
        if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            return float(size[0])
        return float(local @ half)

    world = [
        approach[2],  # how steeply the approach axis points
        axis[2],  # how the second axis (closing axis, palm normal) is tilted
        (point[2] - data.xpos[body][2]) / WORLD_SCALE,  # grasp height over the centre
        width(approach) / WORLD_SCALE,  # the object's extent along the approach axis
        width(axis) / WORLD_SCALE,  # ... along the second axis
        width(across) / WORLD_SCALE,  # ... across both
    ]
    return np.concatenate(
        [np.abs(rotation.T @ approach), np.abs(rotation.T @ axis), np.abs(offset), world]
    ).astype(float)


@dataclass
class GraspLibrary:
    """Executed grasps per key: descriptors and outcomes (1 lifted, 0 not)."""

    features: dict[tuple[str, int], np.ndarray] = field(default_factory=dict)
    outcomes: dict[tuple[str, int], np.ndarray] = field(default_factory=dict)

    @classmethod
    def load(cls, paths: list[Path]) -> "GraspLibrary":
        """From JSON-lines files of ``{"object", "rest", "features", "lifted"}`` records."""
        rows: dict[tuple[str, int], list[tuple[list[float], bool]]] = {}
        for path in paths:
            for line in path.read_text().splitlines():
                record = json.loads(line)
                key = (str(record["object"]), int(record["rest"]))
                rows.setdefault(key, []).append((record["features"], bool(record["lifted"])))
        library = cls()
        for key, items in rows.items():
            library.features[key] = np.array([f for f, _ in items], dtype=float)
            library.outcomes[key] = np.array([o for _, o in items], dtype=float)
        return library

    def score(self, key: tuple[str, int], features: np.ndarray) -> float:
        """Shrunk success rate of the ``NEIGHBOURS`` nearest library grasps of this key."""
        if key not in self.features:
            return 0.0
        distance = np.linalg.norm(self.features[key] - features, axis=1)
        nearest = np.argsort(distance)[:NEIGHBOURS]
        hits = float(self.outcomes[key][nearest].sum())
        return (hits + PRIOR[0]) / (len(nearest) + PRIOR[1])

    def rate(self, key: tuple[str, int]) -> float:
        """The key's success rate over all its library grasps (0 if it has none)."""
        outcomes = self.outcomes.get(key)
        return 0.0 if outcomes is None or not len(outcomes) else float(outcomes.mean())
