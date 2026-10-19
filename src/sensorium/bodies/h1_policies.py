"""Scripted behaviour for the H1: pose-hold and seated babbling.

Control is MuJoCo position actuators: the model's PD controllers (HumanoidBench's
stock gains) track joint-angle targets, and a policy's action is that target
normalised to ``[-1, 1]`` over each actuator's control range. D0.3 adds
whole-body babbling through released reach policies; the two policies here are
only enough to produce a first review sample.

* ``hold``: stand (or sit) in the home pose, with a small smooth noise on the
  arm targets. H1's stock leg gains cannot hold a static stance (the standing
  body is an inverted pendulum whose gravity "stiffness", m g h ~ 470 N m/rad,
  exceeds the ankles' 2 x 40 N m/rad), so while standing the recorder
  multiplies the legs' gains by ``STANDING_LEG_GAIN_SCALE``.
* ``seated_babble``: pelvis welded to a seat in front of a table; each hand
  repeatedly reaches for a random point on or just above the table (often an
  object), solved by damped-least-squares IK, with smooth noise on the arm
  and finger targets. This is the D0.2 behaviour, kept as the baseline
  (``safe_reach = false``); ``h1_babble.BabblePolicy`` is the default.
"""

import math
from dataclasses import dataclass, field
from typing import Protocol

import mujoco
import numpy as np

from sensorium.bodies.h1_scene import REACH_X
from sensorium.bodies.noise import SmoothNoise

STANDING_LEG_GAIN_SCALE = 10.0
SIDES = ("left", "right")
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_yaw")
EFFECTOR_PREFIX = {
    "hand": {"left": "lh_", "right": "rh_"},
    "pincer": {"left": "lp_", "right": "rp_"},
}

LEG_HOME = {
    "standing": {"hip_yaw": 0.0, "hip_roll": 0.0, "hip_pitch": -0.4, "knee": 0.8, "ankle": -0.4},
    "seated": {
        "hip_yaw": 0.0,
        "hip_roll": 0.0,
        "hip_pitch": -math.pi / 2,
        "knee": math.pi / 2,
        "ankle": 0.0,
    },
}
# Arms bent in front of the chest ("ready"), mirrored for the right side.
ARM_READY = {
    "shoulder_pitch": -0.9,
    "shoulder_roll": 0.2,
    "shoulder_yaw": 0.0,
    "elbow": 1.4,
    "wrist_yaw": 0.0,
}
# Standing at rest: arms hang along the sides. On this H1 model elbow 0 points the forearm forward
# and horizontal (the published home keyframe, with every arm joint at 0, is arms out in front);
# the arm hangs straight at about +1.3 rad. 1.2 rad with 0.25 rad of shoulder roll puts the
# fingertips beside the thighs, 0.1-0.2 m in front of the pelvis and 0.2-0.4 m below it, clear of
# them (touch rate under 1%). Hanging arms move the centre of mass back, so ``HoldPolicy`` leans
# the hips forward by ``REST_HIP_PITCH`` (0 falls in 8 hand and 8 pincer episodes of 300 steps;
# -0.4 drops pincers, -0.6 drops both). Mirrored for the right side like ARM_READY.
ARM_REST = {
    "shoulder_pitch": 0.0,
    "shoulder_roll": 0.25,
    "shoulder_yaw": 0.0,
    "elbow": 1.2,
    "wrist_yaw": 0.0,
}
REST_HIP_PITCH = -0.5  # rad, both hips, while standing at rest (LEG_HOME has -0.4)
ROLL_YAW_SIGN = {"left": 1.0, "right": -1.0}
LEG_HOME["stand_supported"] = LEG_HOME["standing"]  # the pelvis is welded, the legs stand


class ActionSpace:
    """Normalised position targets for every actuator of a compiled model.

    ``action = 2 (ctrl - low) / (high - low) - 1`` over the actuator's control
    range, so actions of different joints are comparable and bounded; an
    action step of ``d`` is ``d * half_range`` radians (or ctrl units).
    """

    def __init__(self, model: mujoco.MjModel, effector: str) -> None:
        self.model = model
        self.names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(model.nu)
        ]
        self.low = model.actuator_ctrlrange[:, 0].copy()
        self.high = model.actuator_ctrlrange[:, 1].copy()
        self.half_range = (self.high - self.low) / 2
        if not model.actuator_ctrllimited.all():
            raise ValueError("every actuator needs a ctrlrange to be normalised")
        self.index = {name: i for i, name in enumerate(self.names)}
        self.arm = {s: [self.index[f"{s}_{j}"] for j in ARM_JOINTS] for s in SIDES}
        prefixes = EFFECTOR_PREFIX[effector]
        self.effector = {
            s: [i for i, n in enumerate(self.names) if n.startswith(prefixes[s])] for s in SIDES
        }
        # Joint (qpos address) driven by each actuator; -1 for tendon actuators.
        self.qpos_adr = np.full(model.nu, -1)
        for a in range(model.nu):
            if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
                self.qpos_adr[a] = model.jnt_qposadr[model.actuator_trnid[a, 0]]

    def to_action(self, ctrl: np.ndarray) -> np.ndarray:
        """Position targets -> normalised actions in ``[-1, 1]`` (clipped)."""
        return np.asarray(np.clip(2 * (ctrl - self.low) / (self.high - self.low) - 1, -1.0, 1.0))

    def to_ctrl(self, action: np.ndarray) -> np.ndarray:
        """Normalised actions -> position targets within the control range."""
        return np.asarray(
            self.low + (np.clip(action, -1.0, 1.0) + 1) * 0.5 * (self.high - self.low)
        )

    def home(self, posture: str) -> np.ndarray:
        """Home targets: legs per posture, arms at rest (standing) or ready, effector open (0)."""
        ctrl = np.clip(np.zeros(self.model.nu), self.low, self.high)
        for side in SIDES:
            for joint, value in LEG_HOME[posture].items():
                ctrl[self.index[f"{side}_{joint}"]] = value
            for joint, value in (ARM_REST if posture == "standing" else ARM_READY).items():
                sign = ROLL_YAW_SIGN[side] if joint in ("shoulder_roll", "shoulder_yaw") else 1.0
                ctrl[self.index[f"{side}_{joint}"]] = sign * value
        return np.clip(ctrl, self.low, self.high)


class Policy(Protocol):
    """Interface: ``reset`` once per episode, then ``act`` every step."""

    def reset(self, data: mujoco.MjData) -> np.ndarray:
        """Start an episode from the settled state in ``data``; return the home targets."""
        ...

    def act(self, step: int, data: mujoco.MjData) -> np.ndarray:
        """Normalised actions ``[A]`` for this step."""
        ...


class HoldPolicy:
    """Home pose plus a small Ornstein-Uhlenbeck noise on the arm targets."""

    def __init__(
        self,
        space: ActionSpace,
        posture: str,
        rng: np.random.Generator,
        noise_radians: float = 0.05,
    ) -> None:
        self.space, self.posture = space, posture
        arm = [i for side in SIDES for i in space.arm[side]]
        self.arm = np.array(arm)
        home = space.home(posture)
        if posture == "standing":  # arms hanging: lean forward to keep the balance
            for side in SIDES:
                home[space.index[f"{side}_hip_pitch"]] = REST_HIP_PITCH
        self.home = home
        self.home_action = space.to_action(home)
        self.noise = SmoothNoise(rng, len(arm), 1.0, theta=0.1)
        self.noise_scale = noise_radians / space.half_range[self.arm]

    def reset(self, data: mujoco.MjData) -> np.ndarray:
        """Return the home targets."""
        return self.home.copy()

    def act(self, step: int, data: mujoco.MjData) -> np.ndarray:
        """Home action with noise on the arms."""
        action = self.home_action.copy()
        action[self.arm] += self.noise_scale * self.noise.sample()
        return np.asarray(np.clip(action, -1.0, 1.0))


@dataclass
class Tip:
    """The point of an effector that touches things: a body-frame offset, or a site."""

    body: int
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def position(self, data: mujoco.MjData) -> np.ndarray:
        """World position of the tip."""
        rotation = data.xmat[self.body].reshape(3, 3)
        return np.asarray(data.xpos[self.body] + rotation @ self.offset)


def find_tips(model: mujoco.MjModel, effector: str) -> dict[str, Tip]:
    """Per side: the middle fingertip (hand) or the pinch point between the pads (pincer)."""
    tips = {}
    for side in SIDES:
        prefix = EFFECTOR_PREFIX[effector][side]
        if effector == "hand":
            body = model.body(f"{prefix}mfdistal").id
            tips[side] = Tip(body, np.array([0.0, 0.0, 0.026]))
        else:
            site = model.site(f"{prefix}pinch")
            tips[side] = Tip(int(site.bodyid[0]), np.asarray(site.pos))
    return tips


class ArmIK:
    """Damped-least-squares position IK of one arm (4 joints; wrist yaw is left alone)."""

    def __init__(self, model: mujoco.MjModel, space: ActionSpace, side: str, tip: Tip) -> None:
        self.model, self.tip = model, tip
        self.scratch = mujoco.MjData(model)
        joints = [model.joint(f"{side}_{name}") for name in ARM_JOINTS[:4]]
        self.qpos_adr = np.array([int(j.qposadr[0]) for j in joints])
        wrist = model.joint(f"{side}_{ARM_JOINTS[4]}")
        self.qpos_adr5 = np.append(self.qpos_adr, int(wrist.qposadr[0]))  # with the wrist yaw
        self.dof_adr = np.array([int(j.dofadr[0]) for j in joints])
        self.actuator = np.array(space.arm[side][:4])
        self.low, self.high = space.low[self.actuator], space.high[self.actuator]

    def tip_at(self, qpos: np.ndarray, arm_q: np.ndarray) -> np.ndarray:
        """World position of the tip with the arm at ``arm_q`` (4 joints, or 5 with the wrist yaw).

        The rest of the body is at ``qpos``.
        """
        self.scratch.qpos[:] = qpos
        self.scratch.qpos[self.qpos_adr5 if len(arm_q) == 5 else self.qpos_adr] = arm_q
        mujoco.mj_kinematics(self.model, self.scratch)
        return self.tip.position(self.scratch)

    def solve(
        self, qpos: np.ndarray, arm_ctrl: np.ndarray, target: np.ndarray, iterations: int = 40
    ) -> np.ndarray:
        """Joint targets (4) that bring the tip to ``target``, starting from ``arm_ctrl``.

        ``qpos`` is the full state to hold the rest of the body at. Returns the
        best configuration found (joint limits respected), even if not reached.
        """
        scratch, model = self.scratch, self.model
        scratch.qpos[:] = qpos
        q = arm_ctrl.copy()
        jacobian = np.zeros((3, model.nv))
        for _ in range(iterations):
            scratch.qpos[self.qpos_adr] = q
            mujoco.mj_kinematics(model, scratch)
            mujoco.mj_comPos(model, scratch)
            point = self.tip.position(scratch)
            error = target - point
            if np.linalg.norm(error) < 0.004:
                break
            mujoco.mj_jac(model, scratch, jacobian, None, point, self.tip.body)
            j = jacobian[:, self.dof_adr]
            error = error * min(1.0, 0.05 / max(float(np.linalg.norm(error)), 1e-9))
            step = j.T @ np.linalg.solve(j @ j.T + 1e-3 * np.eye(3), error)
            q = np.clip(q + step, self.low, self.high)
        return q


class SeatedBabblePolicy:
    """Both hands reach for random points on or above the table; fingers wiggle.

    Every 10 to 30 steps (0.5 to 1.5 s) each hand picks a new goal: an object's
    centre (70%) or a random point on its half of the table, pressed into the
    surface (65%) or hovering 10 cm above it. The goal is turned into arm
    targets by IK and approached with a first-order filter; smooth noise is
    added to the arm and effector actions on top. A pincer also closes its
    gripper halfway through most goals.
    """

    TABLE_Y = 0.42  # half width reachable by one hand
    PRESS_DEPTH = 0.006  # m below the surface the tip is sent to (so it presses)
    OBJECT_GOAL = 0.7  # share of reach goals aimed at an object's centre

    def __init__(
        self,
        model: mujoco.MjModel,
        space: ActionSpace,
        effector: str,
        rng: np.random.Generator,
        table_top: float,
        object_bodies: list[int],
        reach_x: tuple[float, float] = REACH_X,
    ) -> None:
        self.model, self.space, self.rng, self.table_top = model, space, rng, table_top
        self.reach_x = reach_x
        self.objects = object_bodies
        self.tips = find_tips(model, effector)
        self.ik = {s: ArmIK(model, space, s, self.tips[s]) for s in SIDES}
        self.home_ctrl = space.home("seated")
        self.arm_noise = {s: SmoothNoise(rng, 5, 1.0, theta=0.1) for s in SIDES}
        self.arm_noise_scale = {s: 0.04 / space.half_range[space.arm[s]] for s in SIDES}
        self.effector_noise = {
            s: SmoothNoise(rng, len(space.effector[s]), 0.5, theta=0.12, mean=self.effector_mean(s))
            for s in SIDES
        }
        self.goal_q = {s: self.home_ctrl[space.arm[s][:4]].copy() for s in SIDES}
        self.cmd_q = {s: self.goal_q[s].copy() for s in SIDES}
        self.next_goal = dict.fromkeys(SIDES, 0)
        self.close_at = dict.fromkeys(SIDES, -1)  # pincer: step at which the gripper closes
        self.pincer = effector == "pincer"
        self.targets = {s: np.zeros(3) for s in SIDES}

    def effector_mean(self, side: str) -> np.ndarray:
        """Normalised home action of this side's effector actuators (open)."""
        return self.space.to_action(self.home_ctrl)[self.space.effector[side]]

    def reset(self, data: mujoco.MjData) -> np.ndarray:
        """Home targets with the arms already over the table."""
        ctrl = self.home_ctrl.copy()
        for side in SIDES:
            sign = ROLL_YAW_SIGN[side]
            start = np.array([0.5, sign * 0.2, self.table_top + 0.15])
            q = self.ik[side].solve(data.qpos, ctrl[self.space.arm[side][:4]], start, 80)
            ctrl[self.space.arm[side][:4]] = q
            self.goal_q[side], self.cmd_q[side] = q.copy(), q.copy()
        self.home_ctrl = ctrl
        return ctrl

    def new_goal(self, side: str, data: mujoco.MjData) -> None:
        """Pick the next reach target for ``side`` and solve its IK."""
        sign = ROLL_YAW_SIGN[side]  # the left hand works the +y half of the table
        on_table = [b for b in self.objects if data.xpos[b][2] > self.table_top - 0.01]
        if on_table and self.rng.random() < self.OBJECT_GOAL:
            pick = data.xpos[on_table[int(self.rng.integers(len(on_table)))]]
            x, y, z = float(pick[0]), float(pick[1]), float(pick[2])  # the object's centre
            if y * sign < 0.0:  # on the other hand's side: reach to the middle
                y = 0.0
        else:
            x = float(self.rng.uniform(*self.reach_x))
            y = sign * float(self.rng.uniform(0.02, self.TABLE_Y))
            press = self.rng.random() < 0.65
            z = self.table_top - self.PRESS_DEPTH if press else self.table_top + 0.10
        self.targets[side] = np.array([x, y, z])
        self.goal_q[side] = self.ik[side].solve(data.qpos, self.cmd_q[side], self.targets[side])

    def act(self, step: int, data: mujoco.MjData) -> np.ndarray:
        """Targets from the current reach goals plus smooth noise."""
        ctrl = self.home_ctrl.copy()
        for side in SIDES:
            if step >= self.next_goal[side]:
                self.new_goal(side, data)
                dwell = int(self.rng.integers(10, 31))
                self.next_goal[side] = step + dwell
                # Pincers close on whatever is between the pads late in the dwell.
                self.close_at[side] = step + dwell // 2 if self.rng.random() < 0.7 else -1
            self.cmd_q[side] += 0.25 * (self.goal_q[side] - self.cmd_q[side])
            ctrl[self.space.arm[side][:4]] = self.cmd_q[side]
        action = self.space.to_action(ctrl)
        for side in SIDES:
            action[self.space.arm[side]] += (
                self.arm_noise_scale[side] * self.arm_noise[side].sample()
            )
            noise = self.effector_noise[side].sample()
            if self.pincer:
                closing = 0 <= self.close_at[side] <= step
                noise = np.clip(noise, -1, 1) * 0.2 + (1.0 if closing else -1.0)
            action[self.space.effector[side]] = noise
        return np.asarray(np.clip(action, -1.0, 1.0))
