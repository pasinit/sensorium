"""Seated babbling with a repertoire of contact primitives (D0.3 step 2).

``BabblePolicy`` repeats, for each hand: choose a *primitive* and an *object*, plan the
whole movement as joint-space waypoints, execute it, retreat. No task reward is
involved; what to try next is chosen by counts (what was tried least in this
episode), the parameters of each attempt are random.

Primitives (``PRIMITIVES``): ``hover`` (a point above the table), ``touch`` (press an
object's top or a table point lightly), ``poke`` (touch an object's side and push it
5 to 10 cm), ``slide`` (drag the tip over the table), and for pincers ``grasp`` or for
hands ``power``: an oriented approach, a closing reflex, a lift, a hold and a drop. The hand's
``power`` is a claw pinch from above (``HAND_PINCH``): it travels open, takes the pre-shape over
the object, and replans if the object was pushed away on the way.
Each attempt is checked before it starts: every waypoint and the paths between them
must keep ``CLEARANCE`` between the non-effector robot parts and the table, objects
and other limbs (``GeomClasses.limb_contacts``), and the tip must reach the goal.
Joint speeds are capped, and an arm withdraws on a limb contact above ``REFLEX_N``.

The grasp poses use a 5-scalar IK (tip position, the approach axis's height and the
closing or palm axis's height; the approach azimuth stays free) because the 5-joint
arm cannot meet a full 6-D pose (see ``.tmp/phd-a/lift_spike.md`` in the D0.3 PR).
"""

import copy
import math
from dataclasses import dataclass

import mujoco
import numpy as np

from sensorium.bodies.h1_forces import LEFT_ARM, RIGHT_ARM, GeomClasses
from sensorium.bodies.h1_policies import (
    ARM_JOINTS,
    ARM_READY,
    ROLL_YAW_SIGN,
    SIDES,
    ActionSpace,
    ArmIK,
    find_tips,
)
from sensorium.bodies.noise import SmoothNoise

PHASES = ("travel", "descent", "contact", "wait", "retreat")  # of an experiment attempt
PHASE_WAIT = PHASES.index("wait")
PRIMITIVES = {
    "hand": ("hover", "touch", "poke", "slide", "power"),
    "pincer": ("hover", "touch", "poke", "slide", "grasp"),
}
# The hand's grasp is a claw pinch from above (docs/touch.md, "Hand grasping"): the palm faces
# down over the object, the thumb is pre-placed under the palm on the near side, and the fingers,
# flexed at the knuckles, sweep the object against it. Hand actuator -> (pre-shape, closed) target
# (rad; the J0 tendons drive J1 + J2). The closed targets lie past where the fingers meet the
# object: the position actuators (kp 1 N m/rad) squeeze with the remaining error, 4-10 N per
# fingertip. The wrist is held flexed (palm towards the table) and deviated while grasping: with
# it straight the five arm joints reach a level, palm-down hand at a third of the goals, with this
# at all of them.
HAND_PINCH = {
    "FFJ3": (0.5, 1.57), "FFJ0": (0.0, 3.0), "MFJ3": (0.5, 1.57), "MFJ0": (0.0, 3.0),
    "RFJ3": (0.5, 1.57), "RFJ0": (0.0, 3.0), "LFJ3": (0.5, 1.57), "LFJ0": (0.0, 3.0),
    "THJ5": (-0.25, -0.25), "THJ4": (1.22, 1.22), "THJ3": (0.0, 0.0), "THJ2": (0.35, 0.35),
    "THJ1": (0.9, 1.5), "WRJ2": (-0.3, -0.3), "WRJ1": (-0.35, -0.35),
}  # fmt: skip
# Outside a grasp these follow the closure from 0 (open); the other hand actuators babble.
HAND_FLEXORS = (
    "FFJ3", "FFJ0", "MFJ3", "MFJ0", "RFJ3", "RFJ0", "LFJ3", "LFJ0", "THJ4", "THJ1", "THJ2",
)  # fmt: skip
GRASP_HEIGHTS = (-0.8, -0.9, -1.0)  # z-component of the approach axis: how steeply it points down
# Hands: the fingers point about level and the palm normal down at this z-component.
HAND_GRASP_HEIGHTS = (0.0, -0.1)
HAND_PALM_Z = -0.95  # the palm normal 18 degrees off vertical (-0.87: 57% -> 32% lifts per grasp)
PINCH_GAP = 0.075  # m along the fingers from the palm to the object's near face (the thumb's pad)
PINCH_DEPTH = 0.085  # m along the palm normal from the palm to the pinch (thumb and finger tips)


@dataclass
class Waypoint:
    """One leg of a movement: arm joint targets (5, wrist yaw last), speed cap, hold, closure."""

    q: np.ndarray
    speed: float  # largest joint step per control step (rad)
    hold: int = 0  # control steps to stay once there
    grip: float = 0.0  # effector closure to move towards: 0 open, 1 closed
    wait: bool = False  # the arm has no plan: parking and holding, counted as waiting
    preshape: bool = False  # hand: the grasp's pre-shape, closing towards the pinch with ``grip``
    watch: tuple[int, np.ndarray] | None = None  # (object body, planned position): see ``advance``
    phase: int = 0  # for the experiment programs: PHASES index (travel, descent, contact, ...)


@dataclass(frozen=True)
class GraspFrame:
    """The effector frame an oriented grasp is planned in.

    Attributes:
        body: MuJoCo body carrying the frame.
        offset: grasp point in the body frame.
        approach: local unit vector of the approach axis (fingers' pointing direction).
        axis: local unit vector kept horizontal: the pads' closing axis (pincer) or the
            palm normal (hand).
        advance: ``"approach"`` or ``"axis"``: the world direction the effector moves along
            when going onto the object (a pincer along its fingers, a hand along its palm normal).
        back_off: distance (m) of the pre-grasp point from the object along ``advance``.
        axis_z: the world height ``axis`` is held at (0: horizontal).
        fixed: (qpos address, value) pairs held while planning (the hand's wrist).
    """

    body: int
    offset: np.ndarray
    approach: np.ndarray
    axis: np.ndarray
    advance: str
    back_off: float
    axis_z: float = 0.0
    fixed: tuple[tuple[int, float], ...] = ()


class OrientedIK:
    """Damped-least-squares IK of an arm's 5 joints on 5 scalars.

    The scalars are the grasp point's position (3), the approach axis's world height and
    (if ``axis_weight`` > 0) the other axis's world height. The approach azimuth is left free.
    """

    def __init__(self, model: mujoco.MjModel, side: str, frame: GraspFrame) -> None:
        self.model, self.frame = model, frame
        self.scratch = mujoco.MjData(model)
        from_joints = [model.joint(f"{side}_{name}") for name in _ARM5]
        self.qpos_adr = np.array([int(j.qposadr[0]) for j in from_joints])
        self.dof_adr = np.array([int(j.dofadr[0]) for j in from_joints])
        self.low = model.jnt_range[[j.id for j in from_joints], 0]
        self.high = model.jnt_range[[j.id for j in from_joints], 1]

    def frame_at(
        self, qpos: np.ndarray, q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World grasp point, approach axis and other axis with the arm at ``q`` (5 joints)."""
        self.scratch.qpos[:] = qpos
        for adr, value in self.frame.fixed:
            self.scratch.qpos[adr] = value
        self.scratch.qpos[self.qpos_adr] = q
        mujoco.mj_kinematics(self.model, self.scratch)
        rotation = self.scratch.xmat[self.frame.body].reshape(3, 3)
        point = self.scratch.xpos[self.frame.body] + rotation @ self.frame.offset
        return point, rotation @ self.frame.approach, rotation @ self.frame.axis

    def solve(
        self,
        qpos: np.ndarray,
        start: np.ndarray,
        goal: np.ndarray,
        approach_z: float,
        axis_weight: float = 0.3,
        iterations: int = 300,
    ) -> tuple[np.ndarray, float, float]:
        """Joints that put the grasp point at ``goal`` with the approach axis's z at ``approach_z``.

        Returns:
            ``(q, position error in m, largest orientation-scalar error)``; the best found,
            joint limits respected, even if not reached.
        """
        scratch, model, frame = self.scratch, self.model, self.frame
        scratch.qpos[:] = qpos
        for adr, value in frame.fixed:
            scratch.qpos[adr] = value
        q = np.clip(start, self.low, self.high)
        jac_p, jac_r = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        pos_error, ori_error = 1.0, 1.0
        for _ in range(iterations):
            scratch.qpos[self.qpos_adr] = q
            mujoco.mj_kinematics(model, scratch)
            mujoco.mj_comPos(model, scratch)
            rotation = scratch.xmat[frame.body].reshape(3, 3)
            point = scratch.xpos[frame.body] + rotation @ frame.offset
            approach, axis = rotation @ frame.approach, rotation @ frame.axis
            error = goal - point
            e_approach = approach_z - approach[2]
            e_axis = (frame.axis_z - axis[2]) * (axis_weight > 0)
            pos_error, ori_error = float(np.linalg.norm(error)), max(abs(e_approach), abs(e_axis))
            if pos_error < 0.003 and ori_error < 0.02:
                break
            mujoco.mj_jac(model, scratch, jac_p, jac_r, point, frame.body)
            rotation_jac = jac_r[:, self.dof_adr]
            d_approach = (-_skew(approach) @ rotation_jac)[2]
            d_axis = (-_skew(axis) @ rotation_jac)[2]
            weight = 0.3
            jacobian = np.vstack(
                [jac_p[:, self.dof_adr], weight * d_approach, axis_weight * d_axis]
            )
            residual = np.concatenate([error, [weight * e_approach, axis_weight * e_axis]])
            residual *= min(1.0, 0.1 / max(float(np.linalg.norm(residual)), 1e-9))
            step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-4 * np.eye(5), residual)
            q = np.clip(q + step, self.low, self.high)
        return q, pos_error, ori_error


_ARM5 = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_yaw")


def _skew(v: np.ndarray) -> np.ndarray:
    """The matrix of ``a -> v x a``."""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def grasp_frame(model: mujoco.MjModel, effector: str, side: str) -> GraspFrame:
    """The grasp frame of one effector.

    Pincer: the pinch point, approach along the fingers, pads closing along the base's y.
    Hand: the claw pinch's near point (``PINCH_GAP`` along the fingers, ``PINCH_DEPTH`` along
    the palm normal: where the object's near face meets the thumb), approach along the fingers,
    the palm normal as the axis (found from how the fingers curl), held at ``HAND_PALM_Z``; the
    hand advances along the palm normal (down onto the object), its wrist at the pinch's angles.
    """
    tips = find_tips(model, "pincer" if effector == "pincer" else "hand")[side]
    if effector == "pincer":
        return GraspFrame(
            tips.body,
            tips.offset,
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
            "approach",
            0.06,
        )
    prefix = "lh_" if side == "left" else "rh_"
    palm = model.body(f"{prefix}palm").id
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_kinematics(model, data)
    to_local = data.xmat[palm].reshape(3, 3).T
    tip_open = to_local @ (data.xpos[model.body(f"{prefix}mfdistal").id] - data.xpos[palm])
    for joint, value in (("MFJ3", 1.4), ("MFJ2", 1.5), ("MFJ1", 1.0)):
        data.qpos[model.jnt_qposadr[model.joint(f"{prefix}{joint}").id]] = value
    mujoco.mj_kinematics(model, data)
    tip_closed = to_local @ (data.xpos[model.body(f"{prefix}mfdistal").id] - data.xpos[palm])
    fingers = tip_open / np.linalg.norm(tip_open)
    curl = tip_closed - tip_open
    normal = curl - fingers * float(curl @ fingers)
    normal /= np.linalg.norm(normal)
    wrist = tuple(
        (int(model.jnt_qposadr[model.joint(f"{prefix}{name}").id]), HAND_PINCH[name][0])
        for name in ("WRJ1", "WRJ2")
    )
    # Sideways, the pinch lies midway between the thumb and the index and middle fingers, closed.
    for name, (_, closed) in HAND_PINCH.items():
        joints = (name[:2] + "J1", name[:2] + "J2") if name.endswith("J0") else (name,)
        for joint in joints:
            adr = model.jnt_qposadr[model.joint(f"{prefix}{joint}").id]
            data.qpos[adr] = closed / len(joints)
    mujoco.mj_kinematics(model, data)
    distal = {
        f: to_local @ (data.xpos[model.body(f"{prefix}{f}distal").id] - data.xpos[palm])
        for f in ("th", "ff", "mf")
    }
    across = np.cross(normal, fingers)
    middle = float(across @ (distal["th"] + (distal["ff"] + distal["mf"]) / 2)) / 2
    offset = PINCH_GAP * fingers + PINCH_DEPTH * normal + middle * across
    return GraspFrame(palm, offset, fingers, normal, "axis", 0.10, HAND_PALM_Z, wrist)


class BabblePolicy:
    """Seated babbling from a repertoire of primitives with count-based goal selection.

    ``reflexes`` counts withdrawals, ``attempts`` the planned movements per primitive,
    ``planning_failures`` the goals dropped because no clear plan was found.
    """

    HOVER = 0.10  # m above the surface for the pre-contact and travel postures ...
    HOVER_MIN = 0.03  # ... but at least this far above a goal ...
    # ... and no higher than HOVER over the table (the head camera sees about 12 cm above it), so
    # hover, retreat and the grasp approach stay in view (camera framing, D0.3 step 4).
    # Joint step caps per control step (rad): travelling, approaching a contact (about 0.15 m/s
    # at the tip: ISO/TS 15066 style contact speeds), and carrying a grasped object.
    FAST, SLOW, LIFT = 0.10, 0.012, 0.03
    ARRIVED = 0.01  # rad: a waypoint is reached when all joints are this close
    REACH_TOLERANCE = 0.02  # m the position IK may miss its goal by
    CLEARANCE = 0.006  # m a non-effector part keeps from the table, objects and other limbs
    GRASP_CLEARANCE = -0.003  # ... for a pincer and during grasps: the forearm may graze the table
    REFLEX_N = 80.0  # N of limb contact that triggers a withdrawal
    PRESS_DEPTH = 0.004  # m below a surface the tip is sent to (a light press)
    OBJECT_GOAL = 0.7  # share of touches aimed at an object
    START_Y = 0.2  # m, how far to its side each arm starts
    START_HEIGHT = 0.10  # m above the table the arms start at
    APPROACH_MARGIN = 0.02  # m above the pre-grasp point the approach starts
    LIFT_HEIGHT = 0.10  # m a grasped object is lifted (8 cm gave 4-6 cm rises: under a 5 cm lift)
    TABLE_Y = 0.30  # m, half width of the goals: the head camera sees no further
    TRIES = 12  # goals sampled per plan before giving up and waiting
    RESTARTS = 10  # IK starts per posture
    CEILING, ENVELOPE_X, ENVELOPE_Y = 0.50, 0.05, 0.80  # m: the tip's envelope along a path
    REFLEX_COOLDOWN = 10  # control steps (0.5 s) before the next withdrawal of an arm
    LOOP_REFLEXES, LOOP_STEPS = 3, 30  # this many reflexes this close: retreat to the rest instead
    REST_Y = 0.28  # m, where the waiting arm parks (low, within the camera's view)
    ELBOW_CANDIDATES = 3  # clear near postures compared for the lowest elbow
    FLOOR = 0.12  # m: no tip further under the table's surface (fingertips hang under the palm)
    TRAVEL_STEP = 0.04  # m between the waypoints of a Cartesian travel
    PRESHAPE_HOLD = 8  # control steps a hand waits over the object while its thumb moves under
    # m under an object's top a hand pinches at most (0: at the centre, as before): the
    # pinch lies 5.7 cm under the palm's face, so a 15 cm tall block pinched at its centre met the
    # palm with its top and was pushed over
    PINCH_BELOW_TOP = 0.045
    # m: a pincer grasp is planned only if the object is narrower than this along the closing axis
    # (0: no check). The 2F-85 opens 85 mm; the 10 cm plastic box gripped along its length was
    # held by the pads' tips on its top and never rose.
    PINCER_SPAN = 0.0
    REAIMS = 2  # times a hand's grasp is aimed again at its moved object before it is dropped
    MOVED = 0.015  # m an object may have moved since its grasp was planned before it is replanned
    LINE_POS_TOL = 0.012  # m: an oriented waypoint of a line may miss its point by this much
    ENVELOPE_JUMP = 0.5  # rad: a joint move this big is checked against the tip envelope
    WIDE_FROM = 12  # oriented IK: from this restart on, seeds are drawn over the joint ranges
    NEAR = 1.0  # rad: a clear posture this close to the start is taken without looking further
    FAR_SPREAD = 0.7  # rad: far restarts draw their seed around the ready pose, not anywhere
    LINE_TRIES = 4  # IK starts per waypoint of a straight path
    GRIP_RATE = 0.08  # closure change per control step: 85 mm in 0.6 s = 140 mm/s (2F-85: 20-150)
    ARM_NOISE = 0.02  # rad of smooth noise on the arm targets, only while travelling

    def __init__(
        self,
        model: mujoco.MjModel,
        space: ActionSpace,
        effector: str,
        rng: np.random.Generator,
        table_top: float,
        object_bodies: list[int],
        reach_x: tuple[float, float],
        classes: GeomClasses,
        primitives: tuple[str, ...] | None = None,
    ) -> None:
        self.model, self.space, self.rng, self.table_top = model, space, rng, table_top
        self.reach_x, self.objects, self.classes = reach_x, object_bodies, classes
        self.effector = effector
        self.pincer = effector == "pincer"
        self.primitives = primitives or PRIMITIVES[effector]
        self.midline = 0.10 if self.pincer else 0.05  # m: the closest a goal gets to the middle
        self.tips = find_tips(model, effector)
        self.elbow = {s: model.body(f"{s}_elbow_link").id for s in SIDES}
        self.ik = {s: ArmIK(model, space, s, self.tips[s]) for s in SIDES}
        # Per arm, the ready pose (pitch, roll, yaw, elbow): far restarts start around it.
        self.ready = {
            s: np.array(
                [
                    ARM_READY[j]
                    * (ROLL_YAW_SIGN[s] if j in ("shoulder_roll", "shoulder_yaw") else 1)
                    for j in ARM_JOINTS[:4]
                ]
            )
            for s in SIDES
        }
        self.frames = {s: grasp_frame(model, effector, s) for s in SIDES}
        self.oriented = {s: OrientedIK(model, s, self.frames[s]) for s in SIDES}
        self.home_ctrl = space.home("seated")
        # A copy of the model whose contact margin reaches CLEARANCE, so that near misses show
        # up as contacts with a positive distance (checked with kinematics and collision only:
        # mj_forward would run the skin plugin, which disturbs the main scene's rendering).
        self.check_model = copy.deepcopy(model)
        self.check_model.geom_margin[:] = self.CLEARANCE + 0.004
        self.clearance = self.CLEARANCE  # the one posture_ok enforces now
        self.axis_weight = 0.0  # weight of the closing-axis scalar in the oriented IK now
        self.exempt = np.zeros(model.ngeom, dtype=bool)  # geoms the clearance check ignores now
        self.check = mujoco.MjData(self.check_model)
        self.arm_noise = {s: SmoothNoise(rng, 5, 1.0, theta=0.1) for s in SIDES}
        self.arm_noise_scale = self.ARM_NOISE / space.half_range
        self.effector_noise = {
            s: SmoothNoise(rng, len(space.effector[s]), 0.5, theta=0.12, mean=self.effector_mean(s))
            for s in SIDES
        }
        self.cmd_q = {s: self.home_ctrl[space.arm[s]].copy() for s in SIDES}
        self.hover_q = {s: self.cmd_q[s].copy() for s in SIDES}
        self.ticks = dict.fromkeys(SIDES, 0)  # control steps per arm, for the reflex loop guard
        self.cooldown_until = dict.fromkeys(SIDES, 0)  # no new withdrawal before this tick
        self.last_travel_q = {
            s: self.cmd_q[s].copy() for s in SIDES
        }  # last travel waypoint reached
        self.reflex_ticks: dict[str, list[int]] = {s: [] for s in SIDES}
        self.rest_q: dict[str, np.ndarray] = {}  # per arm, a low lateral posture to wait in
        self.queue: dict[str, list[Waypoint]] = {s: [] for s in SIDES}
        self.grip = dict.fromkeys(SIDES, 0.0)
        self.closure = {s: self.closure_actuators(s) for s in SIDES}
        self.flexors = {s: self.closure_actuators(s, HAND_FLEXORS) for s in SIDES}
        self.preshape = dict.fromkeys(SIDES, False)  # the hand holds the grasp's pre-shape
        self.primitive_counts = dict.fromkeys(self.primitives, 0)
        self.object_counts = np.zeros(len(object_bodies))
        self.reflexes = 0
        self.phase = dict.fromkeys(SIDES, 0)  # the phase of each arm's current waypoint
        self.wait_steps = 0  # arm-steps spent without a plan (both arms)
        self.planning_failures = 0
        self.failures = dict.fromkeys(self.primitives, 0)  # plans dropped, per primitive
        self.replans = 0  # grasps dropped at the descent because the object had moved
        self.reaims = 0  # ... and grasps aimed again at the moved object instead
        self.reaims_left = dict.fromkeys(SIDES, 0)  # re-aims still allowed in this grasp

    def effector_mean(self, side: str) -> np.ndarray:
        """Normalised home action of this side's effector actuators (open)."""
        return self.space.to_action(self.home_ctrl)[self.space.effector[side]]

    def closure_actuators(
        self, side: str, names: tuple[str, ...] | None = None
    ) -> dict[int, tuple[float, float]]:
        """Hand: effector-action index -> (pre-shape, closed) ctrl value of the grasp's actuators.

        ``names`` restricts it to those actuators (``HAND_PINCH`` keys).
        """
        if self.pincer:
            return {}
        prefix = "lh_A_" if side == "left" else "rh_A_"
        indices = self.space.effector[side]
        return {
            indices.index(self.space.index[prefix + name]): values
            for name, values in HAND_PINCH.items()
            if names is None or name in names
        }

    # ------------------------------------------------------------------ checks

    def posture_ok(
        self,
        data: mujoco.MjData,
        side: str,
        q: np.ndarray,
        goal: np.ndarray | None = None,
        tolerance: float = REACH_TOLERANCE,
    ) -> bool:
        """True if arm posture ``q`` (5 joints) has the clearance, and reaches ``goal`` if given.

        Only limb contacts of this arm count: the other arm's and the body's are not its doing.
        """
        ik = self.ik[side]
        if goal is not None and np.linalg.norm(ik.tip_at(data.qpos, q) - goal) > tolerance:
            return False
        scratch = self.check
        scratch.qpos[:] = data.qpos
        scratch.qpos[ik.qpos_adr5] = q
        mujoco.mj_kinematics(self.check_model, scratch)
        mujoco.mj_collision(self.check_model, scratch)
        n = scratch.ncon
        if n == 0:
            return True
        geom = np.asarray(scratch.contact.geom[:n])
        arm = (
            self.classes.group[geom] == (LEFT_ARM if side == "left" else RIGHT_ARM)
        ) & self.classes.robot[geom]
        limb = self.classes.limb_contacts(geom) & arm.any(axis=1) & ~self.exempt[geom].any(axis=1)
        return not bool((limb & (np.asarray(scratch.contact.dist[:n]) < self.clearance)).any())

    def in_envelope(self, data: mujoco.MjData, side: str, q: np.ndarray) -> bool:
        """True if the tip at ``q`` is in front of the body, over the table, under its ceiling."""
        tip = self.ik[side].tip_at(data.qpos, q)
        return bool(
            tip[0] >= self.ENVELOPE_X
            and abs(tip[1]) <= self.ENVELOPE_Y
            and self.table_top - self.FLOOR <= tip[2] <= self.table_top + self.CEILING
        )

    def path_ok(self, data: mujoco.MjData, side: str, q0: np.ndarray, q1: np.ndarray) -> bool:
        """True if the straight joint-space path from ``q0`` to ``q1`` keeps the clearance.

        The tip must also stay in the envelope all the way: a joint-space path between two clear
        postures can swing the arm overhead or out beside the table.
        """
        fractions = np.linspace(0.125, 0.875, 7)
        big = float(np.abs(q1[:4] - q0[:4]).max()) > self.ENVELOPE_JUMP  # small moves cannot swing
        if big and not all(self.in_envelope(data, side, q0 + f * (q1 - q0)) for f in fractions):
            return False
        return all(self.posture_ok(data, side, q0 + f * (q1 - q0)) for f in (0.25, 0.5, 0.75))

    def plan_ok(self, data: mujoco.MjData, side: str, queue: list[Waypoint]) -> bool:
        """True if every leg of the queue, from the arm's current command, is clear."""
        starts = [self.cmd_q[side], *(w.q for w in queue)]
        return all(self.path_ok(data, side, a, w.q) for a, w in zip(starts, queue, strict=False))

    # ------------------------------------------------------------------ inverse kinematics

    def with_wrist(self, side: str, q4: np.ndarray) -> np.ndarray:
        """A 5-joint posture from 4 joints, the wrist yaw at its home value."""
        return np.append(q4, self.home_ctrl[self.space.arm[side][4]])

    def solve_valid(
        self, data: mujoco.MjData, side: str, start: np.ndarray, goal: np.ndarray
    ) -> np.ndarray | None:
        """Position IK for ``goal`` from ``start``, then random restarts, until a clear posture.

        A clear posture within ``NEAR`` rad of ``start`` is taken at once; otherwise the nearest
        clear one found over all restarts, so that the arm does not swing between IK branches.
        """
        ik = self.ik[side]
        best: tuple[float, np.ndarray] | None = None
        near: list[tuple[float, np.ndarray]] = []  # (elbow height, posture) of clear, near ones
        for attempt in range(self.RESTARTS):
            seed = start[:4]
            if attempt >= 3:  # far restarts: other arm branches around the ready pose
                seed = np.clip(
                    self.ready[side] + self.rng.normal(0.0, self.FAR_SPREAD, 4), ik.low, ik.high
                )
            elif attempt:
                seed = np.clip(seed + self.rng.normal(0.0, 0.5, 4), ik.low, ik.high)
            q = self.with_wrist(side, ik.solve(data.qpos, seed, goal, 60))
            if self.posture_ok(data, side, q, goal):
                distance = float(np.abs(q[:4] - start[:4]).max())
                if distance <= self.NEAR:
                    near.append((float(self.check.xpos[self.elbow[side]][2]), q))
                    if len(near) >= self.ELBOW_CANDIDATES or attempt >= 3:
                        break
                elif best is None or distance < best[0]:
                    best = (distance, q)
        if near:  # of the near ones, the lowest elbow: it keeps the forearm out of the view
            return min(near, key=lambda c: c[0])[1]
        return None if best is None else best[1]

    def line(
        self,
        data: mujoco.MjData,
        side: str,
        q_start: np.ndarray,
        p0: np.ndarray,
        p1: np.ndarray,
        speed: float,
        step: float = 0.015,
        approach_z: float | None = None,
    ) -> list[Waypoint] | None:
        """Waypoints along the straight line ``p0 -> p1`` (the first point excluded), or None.

        Position IK, or the oriented IK if ``approach_z`` is given; each waypoint is warm-started
        from the last and must be reached and clear.
        """
        count = max(1, math.ceil(float(np.linalg.norm(p1 - p0)) / step))
        waypoints, q = [], q_start
        a0 = (
            0.0
            if approach_z is None
            else float(self.oriented[side].frame_at(data.qpos, q_start)[1][2])
        )
        for k in range(1, count + 1):
            point = p0 + (p1 - p0) * k / count
            # the orientation changes gradually along the line, reaching ``approach_z`` at its end
            goal_z = a0 if approach_z is None else a0 + (approach_z - a0) * k / count
            found = None
            for attempt in range(
                self.LINE_TRIES
            ):  # retry with a jittered start if the first is blocked
                start = q if attempt == 0 else q + self.rng.normal(0.0, 0.1, q.shape)
                if approach_z is None:
                    trial = self.with_wrist(
                        side, self.ik[side].solve(data.qpos, start[:4], point, 40)
                    )
                    clear = self.posture_ok(data, side, trial, point, tolerance=0.012)
                else:
                    trial, pos_error, ori_error = self.oriented[side].solve(
                        data.qpos, start, point, goal_z, self.axis_weight
                    )
                    clear = (
                        pos_error < self.LINE_POS_TOL
                        and ori_error < 0.05
                        and self.posture_ok(data, side, trial)
                    )
                if clear:
                    found = trial
                    break
            if found is None:
                return self.joint_move(data, side, q_start, p1, speed, approach_z)
            q = found
            waypoints.append(Waypoint(q.copy(), speed))
        return waypoints

    def joint_move(
        self,
        data: mujoco.MjData,
        side: str,
        q_start: np.ndarray,
        point: np.ndarray,
        speed: float,
        approach_z: float | None = None,
    ) -> list[Waypoint] | None:
        """One joint-space move to a clear posture at ``point`` (restarts), if its path is clear."""
        if approach_z is None:
            q = self.solve_valid(data, side, q_start, point)
        else:
            found = self.oriented_pose(data, side, point, approach_z)
            q = None if found is None else found[0]
        if q is None or not self.path_ok(data, side, q_start, q):
            return None
        return [Waypoint(q, speed)]

    def oriented_pose(
        self, data: mujoco.MjData, side: str, goal: np.ndarray, approach_z: float
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """A clear 5-joint posture with the grasp point at ``goal``, and the way to advance."""
        oriented = self.oriented[side]
        starts = [self.cmd_q[side], self.hover_q[side]]
        best: tuple[float, np.ndarray, np.ndarray] | None = None
        for attempt in range(20):
            if attempt < len(starts):
                start = starts[attempt]
            elif attempt >= self.WIDE_FROM:  # any branch: a vertical pincer needs an unusual arm
                start = self.rng.uniform(oriented.low, oriented.high)
            else:  # other arm branches around the ready pose
                ready = self.ready[side] + self.rng.normal(0.0, self.FAR_SPREAD, 4)
                start = np.append(
                    np.clip(ready, oriented.low[:4], oriented.high[:4]), self.cmd_q[side][4]
                )
            q, pos_error, ori_error = oriented.solve(
                data.qpos, start, goal, approach_z, self.axis_weight
            )
            if pos_error < 0.004 and ori_error < 0.03 and self.posture_ok(data, side, q):
                _, approach, axis = oriented.frame_at(data.qpos, q)
                advance = approach if self.frames[side].advance == "approach" else axis
                distance = float(np.abs(q[:4] - self.cmd_q[side][:4]).max())
                if distance <= self.NEAR:
                    return q, advance
                if best is None or distance < best[0]:
                    best = (distance, q, advance)
        return None if best is None else (best[1], best[2])

    # ------------------------------------------------------------------ choosing

    def plan_stats(self) -> dict[str, object]:
        """What the planner did: planned movements and dropped plans per primitive, waiting."""
        return {
            "attempts": dict(self.primitive_counts),
            "failures": dict(self.failures),
            "planning_failures": self.planning_failures,
            "replans": self.replans,
            "wait_steps": self.wait_steps,
        }

    def choose(self, counts: dict[str, int] | np.ndarray) -> int:
        """Index drawn with probability proportional to ``1 / (1 + count)``: fewest tries first."""
        values = list(counts.values()) if isinstance(counts, dict) else counts
        weights = 1.0 / (1.0 + np.asarray(values, dtype=float))
        return int(self.rng.choice(len(weights), p=weights / weights.sum()))

    def free_objects(self, data: mujoco.MjData, side: str) -> list[int]:
        """Indices of the objects on the table and on this hand's side of the middle."""
        sign = ROLL_YAW_SIGN[side]
        return [
            i
            for i, body in enumerate(self.objects)
            if data.xpos[body][2] > self.table_top - 0.01
            and sign * data.xpos[body][1] > self.midline
        ]

    def table_point(self, side: str) -> np.ndarray:
        """A random point on this hand's half of the table (at the surface)."""
        sign = ROLL_YAW_SIGN[side]
        x = float(self.rng.uniform(*self.reach_x))
        y = sign * float(self.rng.uniform(self.midline, self.TABLE_Y))
        return np.array([x, y, self.table_top])

    def object_geometry(self, data: mujoco.MjData, index: int) -> tuple[np.ndarray, float, float]:
        """An object's centre, its top height above the centre and its horizontal half extent."""
        body = self.objects[index]
        geom = int(self.model.body_geomadr[body])
        size, kind = self.model.geom_size[geom], self.model.geom_type[geom]
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            half_height, radius = float(size[2]), float(max(size[0], size[1]))
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            half_height, radius = float(size[1]), float(size[0])
        else:
            half_height, radius = float(size[0]), float(size[0])
        return np.array(data.xpos[body]), half_height, radius

    # ------------------------------------------------------------------ primitives

    def plan(self, side: str, data: mujoco.MjData) -> None:
        """Choose a primitive and an object by counts, plan it, and queue it (or wait)."""
        for _ in range(self.TRIES):
            primitive = self.primitives[self.choose(self.primitive_counts)]
            builder = getattr(self, f"plan_{primitive}")
            grasping = primitive in ("grasp", "power")
            roomy = grasping or self.pincer  # a pincer's forearm must graze the table to touch it
            self.clearance = self.GRASP_CLEARANCE if roomy else self.CLEARANCE
            self.axis_weight = 0.3 if grasping else 0.0
            queue = builder(side, data)
            if queue and self.plan_ok(data, side, queue):
                self.clearance, self.axis_weight = self.CLEARANCE, 0.0
                self.exempt[:] = False
                self.queue[side] = queue
                self.primitive_counts[primitive] += 1
                self.reaims_left[side] = self.REAIMS if grasping else 0
                return
            self.failures[primitive] += 1
        self.clearance, self.axis_weight = self.CLEARANCE, 0.0
        self.exempt[:] = False
        self.planning_failures += 1
        target = self.cmd_q[side].copy()
        rest = self.rest_q.get(side)
        if rest is not None and self.path_ok(data, side, target, rest):
            target = rest  # no plan: park low and to the side rather than freezing mid-air
        self.queue[side] = [Waypoint(target, self.FAST, hold=10, wait=True)]

    def pick_object(self, side: str, data: mujoco.MjData) -> int | None:
        """An object on this hand's side, fewest tried first; counts it as tried."""
        candidates = self.free_objects(data, side)
        if not candidates:
            return None
        index = candidates[self.choose(self.object_counts[candidates])]
        self.object_counts[index] += 1
        return index

    def tilt(self) -> float | None:
        """The approach height of a pincer's contact movements (its forearm stays clear)."""
        return float(self.rng.choice((-0.8, -0.9, -1.0))) if self.pincer else None

    def hover_point(self, point: np.ndarray) -> np.ndarray:
        """``point`` raised to ``HOVER`` over the table, but at least ``HOVER_MIN`` above itself."""
        z = max(point[2] + self.HOVER_MIN, self.table_top + self.HOVER)
        return np.array([point[0], point[1], z])

    def to_hover(
        self,
        data: mujoco.MjData,
        side: str,
        point: np.ndarray,
        tilt: float | None = None,
        above: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[Waypoint]] | None:
        """A clear posture hovering at ``point`` (see ``hover_point``) and the travel there.

        With ``tilt`` the posture is oriented (see ``tilt``), otherwise position-only; ``above``
        overrides the hover position.
        """
        above = self.hover_point(point) if above is None else above
        if tilt is None:
            q = self.solve_valid(data, side, self.hover_q[side], above)
        else:
            found = self.oriented_pose(data, side, above, tilt)
            q = None if found is None else found[0]
        if q is None:
            return None
        start = self.cmd_q[side]
        if self.path_ok(data, side, start, q):
            return q, [Waypoint(q, self.FAST)]
        # The direct joint path leaves the envelope (it arcs overhead): travel along the straight
        # line at hover height instead.
        tip = self.ik[side].tip_at(data.qpos, start)
        via = self.line(data, side, start, tip, above, self.FAST, self.TRAVEL_STEP, tilt)
        return None if via is None else (via[-1].q, via)

    def finish(self, side: str, queue: list[Waypoint], q_hover: np.ndarray) -> list[Waypoint]:
        """End a movement with a retreat to the hover posture ``q_hover``."""
        queue.append(Waypoint(q_hover, self.FAST))
        self.hover_q[side] = q_hover
        return queue

    def plan_hover(self, side: str, data: mujoco.MjData) -> list[Waypoint] | None:
        """Travel to a point hovering over this hand's half of the table."""
        point = self.table_point(side)
        above = point + np.array([0.0, 0.0, self.rng.uniform(self.HOVER_MIN, self.HOVER)])
        hover = self.to_hover(data, side, point, above=above)
        if hover is None:
            return None
        q, queue = hover
        self.hover_q[side] = q
        return queue

    def plan_touch(self, side: str, data: mujoco.MjData) -> list[Waypoint] | None:
        """Press an object's top (70%) or a table point lightly, stay a few steps, retreat."""
        index = self.pick_object(side, data) if self.rng.random() < self.OBJECT_GOAL else None
        if index is None:
            goal = self.table_point(side)
        else:
            centre, half_height, _ = self.object_geometry(data, index)
            goal = np.array([centre[0], centre[1], centre[2] + half_height])
        goal[2] -= self.PRESS_DEPTH
        tilt = self.tilt()
        hover = self.to_hover(data, side, goal, tilt)
        if hover is None:
            return None
        q_hover, queue = hover
        descent = self.line(
            data, side, q_hover, self.hover_point(goal), goal, self.SLOW, 0.01, tilt
        )
        if descent is None:
            return None
        close = self.pincer and self.rng.random() < 0.7
        queue += descent
        queue.append(
            Waypoint(descent[-1].q, self.SLOW, int(self.rng.integers(6, 21)), float(close))
        )
        return self.finish(side, queue, q_hover)

    def plan_poke(self, side: str, data: mujoco.MjData) -> list[Waypoint] | None:
        """Touch an object's side at mid-height and push it 5 to 10 cm in a random direction."""
        index = self.pick_object(side, data)
        if index is None:
            return None
        centre, half_height, radius = self.object_geometry(data, index)
        angle = float(self.rng.uniform(-1.2, 1.2))  # about +x, away from the body
        push = np.array([math.cos(angle), math.sin(angle), 0.0])
        start = centre - (radius + 0.03) * push
        start[2] = centre[2] - 0.2 * half_height
        end = centre + push * float(self.rng.uniform(0.05, 0.10))
        end[2] = start[2]
        tilt = self.tilt()
        hover = self.to_hover(data, side, start, tilt)
        if hover is None:
            return None
        q_hover, queue = hover
        down = self.line(data, side, q_hover, self.hover_point(start), start, self.SLOW, 0.01, tilt)
        if down is None:
            return None
        along = self.line(data, side, down[-1].q, start, end, self.SLOW, 0.01, tilt)
        if along is None:
            return None
        grip = float(self.pincer)  # a closed pincer pushes with its blunt side
        queue += down + [Waypoint(w.q, w.speed, 0, grip) for w in along]
        queue.append(Waypoint(along[-1].q, self.SLOW, 3, grip))
        up = self.line(data, side, along[-1].q, end, self.hover_point(end), self.FAST, 0.02, tilt)
        if up is None:
            return None
        return self.finish(side, queue + up, up[-1].q)

    def plan_slide(self, side: str, data: mujoco.MjData) -> list[Waypoint] | None:
        """Drag the tip over the table for 10 to 18 cm in a random direction."""
        start = self.table_point(side)
        angle = float(self.rng.uniform(-math.pi, math.pi))
        end = start + float(self.rng.uniform(0.10, 0.18)) * np.array(
            [math.cos(angle), math.sin(angle), 0.0]
        )
        sign = ROLL_YAW_SIGN[side]
        end[0] = float(np.clip(end[0], *self.reach_x))
        end[1] = sign * float(np.clip(sign * end[1], self.midline, self.TABLE_Y))
        start[2] = end[2] = self.table_top - self.PRESS_DEPTH
        tilt = self.tilt()
        hover = self.to_hover(data, side, start, tilt)
        if hover is None:
            return None
        q_hover, queue = hover
        down = self.line(data, side, q_hover, self.hover_point(start), start, self.SLOW, 0.01, tilt)
        if down is None:
            return None
        along = self.line(data, side, down[-1].q, start, end, self.SLOW, 0.01, tilt)
        if along is None:
            return None
        up = self.line(data, side, along[-1].q, end, self.hover_point(end), self.FAST, 0.02, tilt)
        if up is None:
            return None
        return self.finish(side, queue + down + along + up, up[-1].q)

    def plan_grasp(self, side: str, data: mujoco.MjData) -> list[Waypoint] | None:
        """Pincer: an oriented approach, a closing reflex, a lift, a hold and a drop."""
        return self.plan_oriented(side, data, lift=self.LIFT_HEIGHT)

    def plan_power(self, side: str, data: mujoco.MjData) -> list[Waypoint] | None:
        """Hand: the same sequence with a power closure of the fingers."""
        return self.plan_oriented(side, data, lift=self.LIFT_HEIGHT)

    def plan_oriented(
        self, side: str, data: mujoco.MjData, lift: float, index: int | None = None
    ) -> list[Waypoint] | None:
        """Approach an object oriented, close (1), lift, hold, release (0), retreat.

        ``index`` names the object (a re-aim, see ``advance``); by default one is picked.
        """
        index = self.pick_object(side, data) if index is None else index
        if index is None:
            return None
        centre = self.grasp_centre(data, index)
        goal = centre
        body = self.objects[index]  # the forearm may graze the object it grasps, nothing else
        self.exempt[:] = False
        self.exempt[
            self.model.body_geomadr[body] : self.model.body_geomadr[body]
            + self.model.body_geomnum[body]
        ] = True
        approach_z = float(self.rng.choice(GRASP_HEIGHTS if self.pincer else HAND_GRASP_HEIGHTS))
        found = self.oriented_pose(data, side, centre, approach_z)
        if found is not None and not self.pincer:
            # The hand's frame point is the object's near face: back it off along the fingers by
            # the object's half extent in their direction, and solve again from there.
            approach = self.oriented[side].frame_at(data.qpos, found[0])[1]
            goal = centre - self.half_extent(data, index, approach) * approach
            found = self.oriented_pose(data, side, goal, approach_z)
        if found is None or not self.fits(data, side, index, found[0]):
            return None
        q_grasp, advance = found
        pre = goal - self.frames[side].back_off * advance
        above = pre + np.array([0.0, 0.0, self.APPROACH_MARGIN])
        q_above, pos_error, ori_error = self.oriented[side].solve(
            data.qpos, q_grasp, above, approach_z, self.axis_weight
        )
        if pos_error > 0.01 or ori_error > 0.05 or not self.posture_ok(data, side, q_above):
            return None
        to_pre = self.line(data, side, q_above, above, pre, self.SLOW, 0.02, approach_z)
        onto = (
            self.line(data, side, to_pre[-1].q, pre, goal, self.SLOW, 0.012, approach_z)
            if to_pre
            else None
        )
        up = (
            self.line(
                data, side, onto[-1].q, goal, goal + [0, 0, lift], self.LIFT, 0.012, approach_z
            )
            if onto
            else None
        )
        if not (to_pre and onto and up):
            return None
        lead = [Waypoint(q_above, self.FAST)]
        if not self.path_ok(data, side, self.cmd_q[side], q_above):
            # The direct joint path arcs out of the envelope: travel along the line instead.
            start = self.cmd_q[side]
            tip = self.ik[side].tip_at(data.qpos, start)
            via = self.line(data, side, start, tip, above, self.FAST, self.TRAVEL_STEP, approach_z)
            if via is None:
                return None
            lead = via
        if not self.pincer:
            # The travel can brush the object aside (effector contacts are allowed on the way):
            # the hand's descent checks that it is still where the grasp was planned.
            to_pre[0].watch = (body, np.array(data.xpos[body]))
        # A hand travels open (its pre-shaped thumb hangs under the palm and would sweep the
        # table) and takes the pre-shape over the object, before the descent.
        shape = [] if self.pincer else [Waypoint(q_above, self.SLOW, self.PRESHAPE_HOLD)]
        queue = [*lead, *shape, *to_pre, *onto]
        queue.append(Waypoint(onto[-1].q, self.SLOW, 3, 0.0))
        queue.append(Waypoint(onto[-1].q, self.SLOW, 14, 1.0))  # close
        queue += [Waypoint(w.q, w.speed, 0, 1.0) for w in up]
        queue.append(Waypoint(up[-1].q, self.LIFT, 12, 1.0))  # hold
        queue.append(Waypoint(up[-1].q, self.LIFT, 6, 0.0))  # drop
        queue = self.finish(side, queue, q_above)
        for waypoint in queue[len(lead) :]:  # a hand keeps its pre-shape until the retreat
            waypoint.preshape = not self.pincer
        return queue

    def reaim(self, side: str, data: mujoco.MjData, index: int) -> list[Waypoint] | None:
        """The hand's grasp planned again for object ``index`` where it is now, if allowed.

        A grasp is re-aimed at most ``REAIMS`` times (2: hands' own lifts per event-dense episode
        0.60 -> 0.87; 0 was the behaviour before); after that, and for pincers (whose descent is not
        watched), the arm plans something new instead.
        """
        if self.pincer or self.reaims_left[side] <= 0:
            return None
        self.reaims_left[side] -= 1
        self.clearance, self.axis_weight = self.GRASP_CLEARANCE, 0.3  # as plan() for a grasp
        try:
            queue = self.plan_oriented(side, data, self.LIFT_HEIGHT, index)
            if queue and not self.plan_ok(data, side, queue):
                queue = None
        finally:
            self.clearance, self.axis_weight = self.CLEARANCE, 0.0
            self.exempt[:] = False
        return queue

    def grasp_centre(self, data: mujoco.MjData, index: int) -> np.ndarray:
        """Where object ``index`` is grasped: its centre, or for a hand at most
        ``PINCH_BELOW_TOP`` under its top (pinched at the centre of a tall object, the palm would
        land on its top)."""
        centre, _, _ = self.object_geometry(data, index)
        if self.pincer or not self.PINCH_BELOW_TOP:
            return centre
        top = centre[2] + self.vertical_half_extent(data, index)
        return np.array([centre[0], centre[1], max(centre[2], top - self.PINCH_BELOW_TOP)])

    def fits(self, data: mujoco.MjData, side: str, index: int, q: np.ndarray) -> bool:
        """For a pincer with ``PINCER_SPAN``: True if object ``index`` is narrower than it along
        the pads' closing axis at posture ``q``; always True otherwise."""
        if not (self.pincer and self.PINCER_SPAN):
            return True
        axis = self.oriented[side].frame_at(data.qpos, q)[2]
        return 2 * self.half_extent(data, index, axis) <= self.PINCER_SPAN

    def vertical_half_extent(self, data: mujoco.MjData, index: int) -> float:
        """Half the height of object ``index`` as it lies now (tipped boxes and cylinders too)."""
        body = self.objects[index]
        geom = int(self.model.body_geomadr[body])
        kind, size = int(self.model.geom_type[geom]), self.model.geom_size[geom]
        up = data.xmat[body].reshape(3, 3)[2]  # world z in the body's axes
        if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
            return float(sum(abs(up[k]) * size[k] for k in range(3)))
        if kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            along = abs(float(up[2]))
            return float(size[0] * math.sqrt(max(0.0, 1 - along * along)) + size[1] * along)
        return float(size[0])

    def half_extent(self, data: mujoco.MjData, index: int, direction: np.ndarray) -> float:
        """An object's horizontal half extent along ``direction`` (its radius if round)."""
        body = self.objects[index]
        geom = int(self.model.body_geomadr[body])
        if self.model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
            return float(self.model.geom_size[geom][0])
        flat = np.array([direction[0], direction[1], 0.0])
        flat /= max(float(np.linalg.norm(flat)), 1e-9)
        axes = data.xmat[body].reshape(3, 3)
        return float(sum(abs(flat @ axes[:, k]) * self.model.geom_size[geom][k] for k in range(3)))

    # ------------------------------------------------------------------ acting

    def reset(self, data: mujoco.MjData) -> np.ndarray:
        """Home targets with the arms already hovering over the table.

        The home pose has the arms hanging into the table, so each arm's start is checked with the
        other arm already at a plain IK start rather than at home.
        """
        ctrl = self.home_ctrl.copy()
        state = mujoco.MjData(self.model)
        state.qpos[:] = data.qpos
        plain = {}
        for side in SIDES:
            start = np.array([0.5, ROLL_YAW_SIGN[side] * self.START_Y, self.table_top + 0.2])
            home = ctrl[self.space.arm[side][:4]]
            plain[side] = self.with_wrist(side, self.ik[side].solve(data.qpos, home, start, 80))
            state.qpos[self.ik[side].qpos_adr5] = plain[side]
        for side in SIDES:
            q = plain[side]
            # Descend to the start height in small steps, so that the posture stays on the same IK
            # branch (a plain solve for a low goal can land on a folded arm).
            for height in (0.2, 0.16, 0.13, self.START_HEIGHT):
                start = np.array([0.5, ROLL_YAW_SIGN[side] * self.START_Y, self.table_top + height])
                lower = self.solve_valid(state, side, q, start)
                if lower is None:
                    break
                q = lower
            state.qpos[self.ik[side].qpos_adr5] = q
            ctrl[self.space.arm[side]] = q
            self.cmd_q[side], self.hover_q[side] = q.copy(), q.copy()
            self.last_travel_q[side] = q.copy()
        for side in SIDES:  # the waiting posture: low, to the side, in front of the body
            point = np.array([0.45, ROLL_YAW_SIGN[side] * self.REST_Y, self.table_top + 0.06])
            rest = self.solve_valid(state, side, self.cmd_q[side], point)
            if rest is not None:
                self.rest_q[side] = rest
        self.home_ctrl = ctrl
        return ctrl

    def limb_forces(self, data: mujoco.MjData) -> dict[str, float]:
        """Per arm, the strongest limb contact (N) it takes part in."""
        forces = dict.fromkeys(SIDES, 0.0)
        n = data.ncon
        if n == 0:
            return forces
        geom = np.asarray(data.contact.geom[:n])
        wrench = np.zeros(6)
        for i in np.flatnonzero(self.classes.limb_contacts(geom)):
            mujoco.mj_contactForce(self.model, data, int(i), wrench)
            for g in geom[i]:
                group = int(self.classes.group[g]) if self.classes.robot[g] else -1
                if group in (LEFT_ARM, RIGHT_ARM):
                    side = SIDES[group - LEFT_ARM]
                    forces[side] = max(forces[side], float(wrench[0]))
        return forces

    def advance(self, side: str, data: mujoco.MjData, force: float) -> tuple[bool, float]:
        """Move the side's command one step along its queue.

        Returns:
            ``(travelling freely, closure target)``.
        """
        self.ticks[side] += 1
        if force > self.REFLEX_N and self.ticks[side] >= self.cooldown_until[side]:
            self.reflexes += 1  # a withdrawal; a new one waits for the cooldown (no thrashing)
            self.cooldown_until[side] = self.ticks[side] + self.REFLEX_COOLDOWN
            recent = [t for t in self.reflex_ticks[side] if self.ticks[side] - t < self.LOOP_STEPS]
            self.reflex_ticks[side] = [*recent, self.ticks[side]]
            # Back out the way the arm came: the last travel posture it reached, which is clear of
            # the object (the older hover posture can lie behind it).
            retreat = self.last_travel_q[side]
            if len(recent) >= self.LOOP_REFLEXES - 1 and side in self.rest_q:
                retreat = self.rest_q[side]  # hit again and again: go to the rest
                self.reflex_ticks[side] = []
            self.queue[side] = [Waypoint(retreat.copy(), self.FAST, hold=5)]
        watch = self.queue[side][0].watch if self.queue[side] else None
        if watch is not None and np.linalg.norm(data.xpos[watch[0]] - watch[1]) > self.MOVED:
            # the object was pushed on the way: aim at it again from here, else plan anew
            queue = self.reaim(side, data, self.objects.index(watch[0]))
            self.replans += queue is None
            self.reaims += queue is not None
            self.queue[side] = queue or []
        if not self.queue[side]:
            self.plan(side, data)
        waypoint = self.queue[side][0]
        self.wait_steps += int(waypoint.wait)
        self.preshape[side] = waypoint.preshape
        self.phase[side] = PHASE_WAIT if waypoint.wait else waypoint.phase
        # One scale for all joints, so that the command follows the straight joint-space path that
        # ``path_ok`` checked (clipping each joint separately bends it, and swings the arm).
        delta = waypoint.q - self.cmd_q[side]
        self.cmd_q[side] += delta * min(1.0, waypoint.speed / max(float(np.abs(delta).max()), 1e-9))
        if np.abs(waypoint.q - self.cmd_q[side]).max() < self.ARRIVED:
            if waypoint.hold > 0:
                waypoint.hold -= 1
            else:
                self.queue[side].pop(0)
                if waypoint.speed == self.FAST:
                    self.last_travel_q[side] = waypoint.q.copy()
        return waypoint.speed == self.FAST, waypoint.grip

    def act(self, step: int, data: mujoco.MjData) -> np.ndarray:
        """Targets from the movement queues; arm noise only while travelling."""
        forces = self.limb_forces(data)
        ctrl = self.home_ctrl.copy()
        free, target = {}, {}
        for side in SIDES:
            free[side], target[side] = self.advance(side, data, forces[side])
            ctrl[self.space.arm[side]] = self.cmd_q[side]
            self.grip[side] += float(
                np.clip(target[side] - self.grip[side], -self.GRIP_RATE, self.GRIP_RATE)
            )
        action = self.space.to_action(ctrl)
        for side in SIDES:
            sample = self.arm_noise[side].sample()
            if free[side]:
                action[self.space.arm[side]] += self.arm_noise_scale[self.space.arm[side]] * sample
            noise = self.effector_noise[side].sample()
            grip = self.grip[side]
            if self.pincer:
                noise = np.clip(noise, -1, 1) * 0.2 + (-1.0 + 2.0 * grip)
            else:
                shaped = self.preshape[side]
                for j, (open_, closed) in (self.closure if shaped else self.flexors)[side].items():
                    actuator = self.space.effector[side][j]
                    value = open_ + grip * (closed - open_) if shaped else grip * closed
                    noise[j] = (
                        2
                        * (value - self.space.low[actuator])
                        / (self.space.high[actuator] - self.space.low[actuator])
                        - 1
                    )
            action[self.space.effector[side]] = noise
        return np.asarray(np.clip(action, -1.0, 1.0))
