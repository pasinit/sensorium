"""A scene director for event-dense table episodes (Track A): it keeps things happening.

The babbling arms make most events themselves, but objects drift out of reach, fall off the
table, or the arms work over empty table for a while. The director, a recorder setting
(``director``), intervenes in the scene, never in the robot:

* **stacks** (``stacks``, a probability per episode): at reset, one box or cylinder is set on
  another, for the arms to knock over;
* **re-placing** (``respawn``): an object that fell off the table (once it is on the floor, so that
  its fall is seen), or that stayed out of reach for ``OUT_OF_REACH_S``, is dropped back from
  ``drop_height`` over a free spot within reach;
* **dropping when idle** (``idle_s``): when no object has moved for that long, an object (the one
  least recently dropped) is dropped from ``drop_height``: onto a hand (``onto_hand``, a
  probability: the touch of an impact on the body) or onto a free spot on the table (seen falling
  and bouncing).

Every intervention is logged per step in ``privileged/intervention`` (``INTERVENTIONS`` codes, 0 for
none): never model input, there to tell a director's events from the robot's own.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

INTERVENTIONS = ("none", "respawn", "drop_table", "drop_hand")
REACH_Y = 0.32  # m: objects further out in |y| are out of reach (goals lie within 0.30)
REACH_MARGIN = 0.06  # m: ... and those this far beyond the reach in x
MOVING_MPS = 0.05  # m/s: an object this fast is moving (event_rate.MOTION_MPS)
OUT_OF_REACH_S = 2.0  # s an object may stay out of reach before it is re-placed
FLOOR_M = 0.30  # m under the table's surface: an object this low has fallen off
SPOT_TRIES = 30  # random spots tried for a free place to drop onto
SPOT_CLEAR = 0.09  # m: a free spot is this far from every other object


@dataclass(frozen=True)
class DirectorSettings:
    """The ``director`` recorder setting.

    Attributes:
        respawn: re-place objects that left the table or the reach.
        idle_s: seconds without a moving object before one is dropped (0: never).
        onto_hand: probability that an idle drop aims at a hand rather than the table.
        drop_height: range (m) above the target an object is dropped from.
        stacks: probability per episode of a stack at reset.
    """

    respawn: bool = True
    idle_s: float = 1.5
    onto_hand: float = 0.5
    drop_height: tuple[float, float] = (0.15, 0.30)
    stacks: float = 0.5

    @classmethod
    def from_setting(cls, setting: object) -> "DirectorSettings":
        """From the recorder setting (a table of these fields).

        Raises:
            ValueError: for an unknown field or a value out of range.
        """
        if not isinstance(setting, dict):
            raise ValueError(f"director must be a table of settings, got {setting!r}")
        unknown = set(setting) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown director settings {sorted(unknown)}")
        values = dict(setting)
        if "drop_height" in values:
            low, high = (float(v) for v in values["drop_height"])
            values["drop_height"] = (low, high)
        made = cls(**values)
        if not (0 <= made.onto_hand <= 1 and 0 <= made.stacks <= 1 and made.idle_s >= 0):
            raise ValueError(f"director settings out of range: {made}")
        return made


class SceneDirector:
    """Intervenes in a seated table scene to keep events coming (see the module docstring)."""

    def __init__(
        self,
        model: mujoco.MjModel,
        settings: DirectorSettings,
        objects: list[int],
        half_heights: list[float],
        table_top: float,
        reach_x: tuple[float, float],
        tips: list[tuple[int, np.ndarray]],
        dt: float,
    ) -> None:
        self.model, self.settings, self.objects = model, settings, objects
        self.half, self.top, self.reach_x, self.tips = half_heights, table_top, reach_x, tips
        self.idle_limit = round(settings.idle_s / dt) if settings.idle_s > 0 else 0
        self.away_limit = round(OUT_OF_REACH_S / dt)
        self.away = np.zeros(len(objects), dtype=int)  # steps each object has been out of reach
        geoms = [int(model.body_geomadr[b]) for b in objects]
        self.stackable = [
            i
            for i, g in enumerate(geoms)
            if int(model.geom_type[g])
            in (int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_CYLINDER))
        ]
        # the smaller horizontal half extent of each object: a stack puts the wider one below
        self.footprint = [float(min(model.geom_size[g][:2])) for g in geoms]
        self.rng = np.random.default_rng(0)
        self.idle = 0
        self.last_dropped = np.zeros(len(objects))
        self.moved: int | None = None  # the object the last intervention moved (an index)

    def reset(self, data: mujoco.MjData, rng: np.random.Generator) -> None:
        """Start an episode (after the objects are placed): maybe stack two of them."""
        self.rng, self.idle = rng, 0
        self.last_dropped[:] = 0.0
        self.away[:] = 0
        if len(self.stackable) >= 2 and self.rng.random() < self.settings.stacks:
            pair = [int(i) for i in self.rng.choice(self.stackable, 2, replace=False)]
            base, top = sorted(pair, key=lambda i: -self.footprint[i])  # the wider one below
            adr = self.model.jnt_qposadr[self.model.body_jntadr[self.objects[base]]]
            xy = data.qpos[adr : adr + 2]  # placed, not yet through the kinematics
            z = self.top + 2 * self.half[base] + self.half[top] + 0.002
            self.place(data, top, np.array([xy[0], xy[1], z]), np.zeros(3))

    def step(self, data: mujoco.MjData, tick: int) -> int:
        """Intervene if needed before this control step; returns the ``INTERVENTIONS`` code.

        ``moved`` is then the index of the object it moved (None without an intervention).
        """
        self.moved = None
        if self.settings.respawn:
            for i, body in enumerate(self.objects):
                position = data.xpos[body]
                self.away[i] = self.away[i] + 1 if self.out_of_reach(position) else 0
                if position[2] < self.top - FLOOR_M or self.away[i] >= self.away_limit:
                    self.away[i] = 0
                    self.drop(data, i, self.free_spot(data, i), tick)
                    return INTERVENTIONS.index("respawn")
        moving = any(self.speed(data, i) > MOVING_MPS for i in range(len(self.objects)))
        self.idle = 0 if moving else self.idle + 1
        if not self.idle_limit or self.idle < self.idle_limit:
            return 0
        self.idle = 0
        index = int(np.argmin(self.last_dropped))
        if self.rng.random() < self.settings.onto_hand:
            body, offset = self.tips[int(self.rng.integers(len(self.tips)))]
            point = data.xpos[body] + data.xmat[body].reshape(3, 3) @ offset
            if point[2] > self.top - 0.02:
                target = np.array([point[0], point[1], point[2] + 0.02])
                self.drop(data, index, target, tick)
                return INTERVENTIONS.index("drop_hand")
        self.drop(data, index, self.free_spot(data, index), tick)
        return INTERVENTIONS.index("drop_table")

    def out_of_reach(self, position: np.ndarray) -> bool:
        """True if an object at ``position`` is off the table or outside the reachable region."""
        low, high = self.reach_x
        return bool(
            position[2] < self.top - 0.02
            or abs(position[1]) > REACH_Y
            or not low - REACH_MARGIN <= position[0] <= high + REACH_MARGIN
        )

    def speed(self, data: mujoco.MjData, index: int) -> float:
        """Linear speed (m/s) of object ``index``."""
        dof = self.model.jnt_dofadr[self.model.body_jntadr[self.objects[index]]]
        return float(np.linalg.norm(data.qvel[dof : dof + 3]))

    def free_spot(self, data: mujoco.MjData, index: int) -> np.ndarray:
        """A table point within reach, clear of the other objects if one is found (surface z)."""
        others = [data.xpos[b][:2] for k, b in enumerate(self.objects) if k != index]
        point = np.zeros(2)
        for _ in range(SPOT_TRIES):
            sign = self.rng.choice([-1.0, 1.0])
            point = np.array(
                [self.rng.uniform(*self.reach_x), sign * self.rng.uniform(0.05, REACH_Y - 0.04)]
            )
            if all(np.linalg.norm(point - other) > SPOT_CLEAR for other in others):
                break
        return np.array([point[0], point[1], self.top + self.half[index]])

    def drop(self, data: mujoco.MjData, index: int, target: np.ndarray, tick: int) -> None:
        """Release object ``index`` at rest, ``drop_height`` over ``target``, randomly turned."""
        height = float(self.rng.uniform(*self.settings.drop_height))
        position = target + np.array([0.0, 0.0, height])
        self.place(data, index, position, np.zeros(3))
        self.last_dropped[index] = tick
        self.moved = index

    def place(
        self, data: mujoco.MjData, index: int, position: np.ndarray, velocity: np.ndarray
    ) -> None:
        """Put object ``index`` at ``position`` with a random yaw and this linear velocity."""
        joint = self.model.body_jntadr[self.objects[index]]
        adr, dof = self.model.jnt_qposadr[joint], self.model.jnt_dofadr[joint]
        yaw = float(self.rng.uniform(0, 2 * np.pi))
        data.qpos[adr : adr + 7] = (*position, np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
        data.qvel[dof : dof + 6] = (*velocity, 0.0, 0.0, 0.0)
