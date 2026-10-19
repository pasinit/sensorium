"""Which contacts are a problem: limb-vs-scene and cross-limb forces, per control step.

The babbling policy should touch things with its *effectors* (hands, pincers).
Contacts of any other robot part with the table or an object (an elbow slammed
on the table, a knee under it) and contacts between different limbs (a shoulder
ground into the torso, a hand on a thigh) are what ``docs/touch.md`` calls slams
and self-collisions; they corrupt the touch and sound channels and show the
policy's posture is bad. This module owns the geom classification and two uses
of it:

* ``PeakForces``: the strongest normal force per class over the physics
  sub-steps of one control step (single sub-step spikes are what slams are),
  stored as ``privileged/peak_force`` and read by ``babble_stats``;
* ``GeomClasses.limb_contacts``: which contacts of a state count as limb
  contacts, for posture checks and the withdrawal reflex of the policies.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from sensorium.bodies.h1_sense import body_voice
from sensorium.bodies.h1_skin import Contacts

PEAK_CLASSES = ("effector_object", "effector_table", "limb_scene", "self")
CORE, LEFT_ARM, RIGHT_ARM = 0, 1, 2  # limb groups of the robot's bodies
OTHER, TABLE, OBJECT = 0, 1, 2  # kinds of non-robot geoms (floor, walls and landmarks are OTHER)


@dataclass(frozen=True)
class GeomClasses:
    """Per-geom class arrays of a compiled scene.

    Attributes:
        robot: ``[G]`` the geom belongs to the robot.
        effector: ``[G]`` robot geom of a hand or pincer (the parts meant to touch things).
        kind: ``[G]`` for non-robot geoms ``OTHER`` / ``TABLE`` / ``OBJECT``.
        group: ``[G]`` for robot geoms ``CORE`` (pelvis, torso, legs) / ``LEFT_ARM`` /
            ``RIGHT_ARM`` (shoulder to fingertips); ``-1`` for the rest.
        object_index: ``[G]`` for object geoms their rank among the ``obj_*`` geoms in model
            order (the order of ``meta.object_names``); ``-1`` for the rest.
    """

    robot: np.ndarray
    effector: np.ndarray
    kind: np.ndarray
    group: np.ndarray
    object_index: np.ndarray

    @classmethod
    def from_model(cls, model: mujoco.MjModel, robot_bodies: set[int]) -> "GeomClasses":
        """Classify every geom by its body and name (``table*`` / ``obj_*``)."""
        arm_root = {
            LEFT_ARM: model.body("left_shoulder_pitch_link").id,
            RIGHT_ARM: model.body("right_shoulder_pitch_link").id,
        }
        count = model.ngeom
        robot = np.zeros(count, dtype=bool)
        effector = np.zeros(count, dtype=bool)
        kind = np.zeros(count, dtype=np.int8)
        group = np.full(count, -1, dtype=np.int8)
        object_index = np.full(count, -1, dtype=np.int8)
        objects = 0
        for geom in range(count):
            body = int(model.geom_bodyid[geom])
            if body in robot_bodies:
                robot[geom] = True
                name = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body))
                effector[geom] = body_voice(name)[0] == "hands"
                group[geom] = CORE
                for side, root in arm_root.items():
                    if is_descendant(model, body, root):
                        group[geom] = side
            else:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
                if name.startswith("table"):
                    kind[geom] = TABLE
                elif name.startswith("obj_"):
                    kind[geom], object_index[geom] = OBJECT, objects
                    objects += 1
        return cls(robot, effector, kind, group, object_index)

    def limb_contacts(self, geom: np.ndarray) -> np.ndarray:
        """``[n]`` mask of contacts that are limb contacts, given their ``[n, 2]`` geom ids.

        A limb contact is a non-effector robot geom touching the table or an
        object, or two robot geoms of different limb groups touching.
        """
        first, second = geom[:, 0], geom[:, 1]
        both_robot = self.robot[first] & self.robot[second]
        cross = both_robot & (self.group[first] != self.group[second])
        one_robot = self.robot[first] ^ self.robot[second]
        mine = np.where(self.robot[first], first, second)
        theirs = np.where(self.robot[first], second, first)
        scene = one_robot & ~self.effector[mine] & (self.kind[theirs] > OTHER)
        return np.asarray(cross | scene)


def is_descendant(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    """True if ``body`` is ``ancestor`` or lies below it in the kinematic tree."""
    while body > 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return False


class PeakForces:
    """Per control step, the peak normal force of each class in ``PEAK_CLASSES``.

    Call ``update`` with ``read_contacts`` after every physics sub-step and
    ``take`` once per control step.
    """

    def __init__(self, classes: GeomClasses) -> None:
        self.classes = classes
        self.peak = np.zeros(len(PEAK_CLASSES))
        self.by_object = np.zeros(int(classes.object_index.max()) + 1)

    def clear(self) -> None:
        """Forget the peaks so far."""
        self.peak[:] = 0.0
        self.by_object[:] = 0.0

    def update(self, contacts: Contacts) -> None:
        """Fold one physics state's contacts into the running peaks."""
        if not len(contacts.force):
            return
        c, geom, force = self.classes, contacts.geom, contacts.force
        first, second = geom[:, 0], geom[:, 1]
        one_robot = c.robot[first] ^ c.robot[second]
        mine = np.where(c.robot[first], first, second)
        theirs = np.where(c.robot[first], second, first)
        eff = one_robot & c.effector[mine]
        masks = (
            eff & (c.kind[theirs] == OBJECT),
            eff & (c.kind[theirs] == TABLE),
            one_robot & ~c.effector[mine] & (c.kind[theirs] > OTHER),
            c.robot[first] & c.robot[second] & (c.group[first] != c.group[second]),
        )
        for index, mask in enumerate(masks):
            if mask.any():
                self.peak[index] = max(self.peak[index], float(force[mask].max()))
        np.maximum.at(self.by_object, c.object_index[theirs[masks[0]]], force[masks[0]])

    def take(self) -> tuple[np.ndarray, np.ndarray]:
        """Peaks since the last call, then clear them.

        Returns:
            ``[len(PEAK_CLASSES)]`` float32 peaks per class, and ``[n_objects]`` float32
            peak effector force on each object.
        """
        peaks = self.peak.astype(np.float32), self.by_object.astype(np.float32)
        self.clear()
        return peaks
