"""Humanoid body: Unitree H1 with whole-body skin, head cameras and two end effectors.

The project's one body (see ``docs/thesis.md``): HumanoidBench's H1 with its
``h1touch`` skin, simulated in MuJoCo. Scene assembly is in ``h1_scene``, the
skin and contact bookkeeping in ``h1_sense``, behaviour in ``h1_policies``.

Settings (all optional): ``effector`` ``"hand"`` (Shadow hands, default) or
``"pincer"`` (Robotiq 2F-85); ``posture`` ``"standing"`` (default) or
``"seated"`` (pelvis welded to a seat at a table with objects); ``camera``
(MuJoCo camera name, default the left eye ``left_eye_camera``).

Control: 20 Hz. Each step the policy gives joint-angle targets for MuJoCo's
position actuators (the H1 XML's PD controllers), normalised to ``[-1, 1]`` over
each actuator's range; physics runs at 500 Hz (25 sub-steps).

Senses per step, in the one episode format:

* ``rgb``: ``IMAGE_SIZE`` (256) square frames from the left head camera
  (tilted down 45 degrees, 80 degree vertical field of view), in a textured room
  whose colours and lighting change every episode (``h1_looks``, recorded in
  ``meta["appearance"]``). v0 is monocular: the right eye is stored under
  ``privileged/rgb_right`` and never reaches the model.
* ``touch`` (see ``docs/touch.md``), three blocks of channels:

  1. the body's ``touch_grid`` taxels (HumanoidBench's coarse skin): normal
     force, e.g. ``body/left_hip_pitch/3``;
  2. ``DenseSkin`` points on the hands (about 1,000 per hand) or the pincer
     pads: normal force spread from each contact, e.g. ``hand/left/ffdistal/17``;
  3. ``ContactDynamics`` per skin region, from every physics step since the
     last observation: ``dynamics/<region>/onset``, ``.../slip`` (m/s) and
     ``.../vibration`` (N).

  ``privileged/touch_xyz`` keeps the 3-D forces of every ``touch_grid`` patch,
  hand patches included (HumanoidBench's 448-taxel layout), named by
  ``meta["touch_xyz_names"]``.
* ``proprio``: joint positions and velocities of the body and effectors, the
  pelvis IMU (``vest_gyro_*``, ``vest_acc_*``), pelvis height and the gravity
  direction in the pelvis frame.
* ``audio``: ``ContactSound`` driven by new contacts and sliding contacts.

Policies: ``hold`` (stand or sit still, smooth noise on the arms) and
``seated_babble`` (seated only; both hands choose primitives by counts: touch, poke,
slide, hover, and a grasp-lift-drop with a pincer or a power closure with a hand; see
``h1_babble``). Episodes end on the step budget, or when a standing robot falls.
"""

import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType

# Offscreen rendering backend: EGL on Linux GPU boxes, GLFW on macOS. Must be set
# before mujoco is first imported.
os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from sensorium.audio import ContactSound, logmel, to_int16  # noqa: E402
from sensorium.bodies.h1_babble import (
    PHASES,  # noqa: E402
    PRIMITIVES,
    BabblePolicy,
)
from sensorium.bodies.h1_director import DirectorSettings, SceneDirector  # noqa: E402
from sensorium.bodies.h1_experiments import (  # noqa: E402
    ExperimentPolicy,
    check_program,
    task_name,
)
from sensorium.bodies.h1_forces import (
    LEFT_ARM,
    PEAK_CLASSES,
    RIGHT_ARM,
    GeomClasses,
    PeakForces,
)  # noqa: E402
from sensorium.bodies.h1_grasp_library import GraspLibrary  # noqa: E402
from sensorium.bodies.h1_layout import skin_layout
from sensorium.bodies.h1_looks import Appearance, apply_appearance, draw_appearance  # noqa: E402
from sensorium.bodies.h1_policies import (  # noqa: E402
    SIDES,
    STANDING_LEG_GAIN_SCALE,
    ActionSpace,
    HoldPolicy,
    Policy,
    SeatedBabblePolicy,
    find_tips,
)
from sensorium.bodies.h1_props import (  # noqa: E402
    HIDDEN_COLUMNS,
    PROPS,
    STACK_MIN_GAP,
    STACK_X,
    STACK_Y,
    TARGET_X,
    TARGET_Y,
    draw_hidden,
    in_exclusion,
)
from sensorium.bodies.h1_reach import ReachFreePolicy  # noqa: E402
from sensorium.bodies.h1_scene import (  # noqa: E402
    EFFECTORS,
    OBJECT_HALF_WIDTH,
    POSTURES,
    REACH_X,
    TABLE_POSTURES,
    asset_paths,
    asset_revisions,
    build_scene,
    default_table_top,
    pelvis_height,
)
from sensorium.bodies.h1_sense import (  # noqa: E402
    MAX_CONTACT_ROWS,
    TOUCH_THRESHOLD_N,
    VOICES,
    ContactFrame,
    ContactMonitor,
    SkinLayout,
)
from sensorium.bodies.h1_skin import (  # noqa: E402
    ContactDynamics,
    DenseSkin,
    dense_bodies,
    read_contacts,
    region_name,
)
from sensorium.describe import describe_humanoid  # noqa: E402
from sensorium.episode import IMAGE_SIZE, SAMPLE_RATE, Episode  # noqa: E402

DT = 0.05
STANDING_PELVIS_Z = 0.98
FALL_HEIGHT = 0.55  # pelvis below this (m) = fallen (standing is ~0.97)
FALL_UPRIGHT = 0.5  # pelvis z-axis tilted past 60 degrees = fallen
SETTLE_STEPS = 10  # unrecorded control steps that let the pose and the objects settle
ARM_SOLREF = 0.04  # contact time constant (s) of the arms' and effectors' geoms, see the recorder
DEFAULT_CAMERA = "left_eye_camera"
# The task a posture records: standing is "stand", seated at the table is "table", and standing
# at a raised table with the pelvis supported is "stand_table" (its own source since split v2).
TASK_BY_POSTURE = {"standing": "stand", "seated": "table", "stand_supported": "stand_table"}
SPACINGS = (0.11, 0.09, 0.07)  # m between object centres: the usual spacing, then relaxed
PLACEMENT_RETRIES = 100
RIGHT_CAMERA = "right_eye_camera"
LEG_ACTUATORS = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")


@dataclass
class Log:
    """Per-step records of one episode (lists of arrays, stacked at the end)."""

    rgb: list[np.ndarray] = field(default_factory=list)
    rgb_right: list[np.ndarray] = field(default_factory=list)
    audio: list[np.ndarray] = field(default_factory=list)
    proprio: list[np.ndarray] = field(default_factory=list)
    touch: list[np.ndarray] = field(default_factory=list)
    touch_xyz: list[np.ndarray] = field(default_factory=list)
    action: list[np.ndarray] = field(default_factory=list)
    qpos: list[np.ndarray] = field(default_factory=list)
    qvel: list[np.ndarray] = field(default_factory=list)
    contacts: list[np.ndarray] = field(default_factory=list)
    peak_force: list[np.ndarray] = field(default_factory=list)
    object_contact: list[np.ndarray] = field(default_factory=list)
    impact_pairs: list[np.ndarray] = field(default_factory=list)
    objects: list[np.ndarray] = field(default_factory=list)
    tips: list[np.ndarray] = field(default_factory=list)
    prop_state: list[np.ndarray] = field(default_factory=list)
    phase: list[list[int]] = field(default_factory=list)
    intervention: list[int] = field(default_factory=list)
    fell: list[bool] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    skin_touched: list[bool] = field(default_factory=list)
    object_touched: list[bool] = field(default_factory=list)
    table_touched: list[bool] = field(default_factory=list)


class HumanoidRecorder:
    """Records H1 episodes for one ``(effector, posture)`` scene."""

    policies: tuple[str, ...] = ("hold", "seated_babble", "reach_free", "experiment")
    appearance: Appearance  # colours and lights of the episode being recorded (set by record)

    def __init__(self, settings: dict[str, object]) -> None:
        self.effector = str(settings.get("effector", "hand"))
        self.posture = str(settings.get("posture", "standing"))
        self.camera = str(settings.get("camera", DEFAULT_CAMERA))
        if self.effector not in EFFECTORS or self.posture not in POSTURES:
            raise ValueError(f"effector must be in {EFFECTORS} and posture in {POSTURES}")
        self.settings: dict[str, object] = {
            "effector": self.effector,
            "posture": self.posture,
            "camera": self.camera,
        }
        # Seated-scene knobs, recorded in the settings only when they apply: the table's height,
        # where objects are placed and hands aim, and the babbling policy's safety behaviours.
        self.has_table = self.posture in TABLE_POSTURES
        default_top = default_table_top(self.posture)
        self.table_top = float(settings.get("table_top", default_top))  # type: ignore[arg-type]
        reach = settings.get("reach_x", REACH_X)
        self.reach_x = (float(reach[0]), float(reach[1]))  # type: ignore[index]
        self.safe_reach = bool(settings.get("safe_reach", True))
        self.arm_solref = float(settings.get("arm_solref", ARM_SOLREF))  # type: ignore[arg-type]
        chosen_props = settings.get("props", [])
        self.props = tuple(sorted(str(p) for p in chosen_props)) if self.has_table else ()  # type: ignore[attr-defined]
        unknown = set(self.props) - set(PROPS)
        if unknown:
            raise ValueError(f"unknown props {sorted(unknown)}; known: {PROPS}")
        self.program = check_program(
            str(settings["program"]) if "program" in settings else None, self.effector
        )
        self.task = bool(settings.get("task", False))  # demonstrator mode of an experiment program
        self.hidden_properties = bool(settings.get("hidden_properties", False))
        self.effort = bool(settings.get("effort", False))
        # Event-dense scenes (h1_director): a scene director, and a cap on the babbling planner's
        # goals per plan (fewer retries: less wall-clock spent planning, more spent acting).
        director = settings.get("director")
        self.director_settings = (
            DirectorSettings.from_setting(director) if director and self.has_table else None
        )
        library = settings.get("grasp_library")
        self.grasp_library_paths = [str(p) for p in library] if isinstance(library, list) else None
        self.grasp_library = (
            GraspLibrary.load([Path(p) for p in self.grasp_library_paths])
            if self.grasp_library_paths
            else None
        )
        self.grasp_explore = bool(settings.get("grasp_explore", True))
        tries = settings.get("plan_tries")
        self.plan_tries = int(tries) if tries is not None else None  # type: ignore[call-overload]
        chosen = settings.get("primitives")
        self.primitives = tuple(str(p) for p in chosen) if isinstance(chosen, list) else None
        if self.has_table:
            self.settings |= self.table_settings(director)
        if self.effort:
            self.settings["effort"] = True

        scene = build_scene(self.effector, self.posture, table_top=self.table_top, props=self.props)
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.substeps = round(DT / self.model.opt.timestep)
        self.renderer = mujoco.Renderer(self.model, IMAGE_SIZE, IMAGE_SIZE)
        self.options = mujoco.MjvOption()
        self.space = ActionSpace(self.model, self.effector)
        self.skin = SkinLayout.from_model(self.model)
        self.dense = DenseSkin(self.model, dense_bodies(self.model, self.effector))
        dense = set(self.dense.regions_by_body)
        self.grid_kept = np.array([b not in dense for b in self.skin.bodies])
        skin_bodies = sorted({int(b) for b in self.skin.bodies[self.grid_kept]} | dense)
        self.dynamics = ContactDynamics(
            self.model, {b: region_name(self.model, b) for b in skin_bodies}
        )
        grid_names = [n for n, kept in zip(self.skin.names, self.grid_kept, strict=True) if kept]
        self.force_names = grid_names + self.dense.names
        self.touch_names = self.force_names + self.dynamics.names
        self.pelvis = self.model.body("pelvis").id
        robot = {b for b in range(self.model.nbody) if self.model.body_rootid[b] == self.pelvis}
        self.monitor = ContactMonitor(self.model, robot)
        self.classes = GeomClasses.from_model(self.model, robot)
        # Soft arms and effectors: the contact time constant (s) of their geoms is at least
        # ``arm_solref``. The Menagerie pads use 0.004 (the stiffest the 2 ms step allows) and the
        # H1 links the default 0.02; an impact at 0.3 m/s then gives 1-38 kN. Chosen by a sweep
        # (0.02 and 0.04, effectors only or all robot geoms; D0.3 step 3), not calibrated on a
        # measured stiffness. The legs and torso stay rigid so that the stance is unchanged.
        arms = self.classes.robot & np.isin(self.classes.group, (LEFT_ARM, RIGHT_ARM))
        soft = np.flatnonzero(arms)
        self.model.geom_solref[soft, 0] = np.maximum(
            self.model.geom_solref[soft, 0], self.arm_solref
        )
        self.peaks = PeakForces(self.classes)
        self.joints = [
            j
            for j in range(self.model.njnt)
            if self.model.jnt_bodyid[j] in robot
            and self.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
        ]
        self.proprio_names = (
            [f"q_{self.joint_name(j)}" for j in self.joints]
            + [f"dq_{self.joint_name(j)}" for j in self.joints]
            + [f"vest_gyro_{axis}" for axis in "xyz"]
            + [f"vest_acc_{axis}" for axis in "xyz"]
            + ["pelvis_height"]
            + [f"gravity_{axis}" for axis in "xyz"]
        )
        self.effort_names = [
            f"eff_{mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)}"
            for a in range(self.model.nu)
        ]
        if self.effort:
            self.proprio_names += self.effort_names
        self.effort_range = np.abs(self.model.actuator_forcerange).max(axis=1)
        self.gyro = int(self.model.sensor("pelvis_gyro").adr[0])
        self.acc = int(self.model.sensor("pelvis_acc").adr[0])
        self.objects = (
            [b for b in range(self.model.nbody) if self.model.body(b).name.startswith("obj_")]
            if self.has_table
            else []
        )
        self.object_names = [self.model.body(b).name for b in self.objects]
        self.object_geoms = [int(self.model.body_geomadr[b]) for b in self.objects]
        self.nominal = {  # restored at every reset: hidden properties change the shared model
            "mass": self.model.body_mass[self.objects].copy(),
            "inertia": self.model.body_inertia[self.objects].copy(),
            "friction": self.model.geom_friction[self.object_geoms, 0].copy(),
            "solref": self.model.geom_solref[self.object_geoms, 0].copy(),
        }
        self.prop_joints = [
            j
            for j in range(self.model.njnt)
            if (self.model.joint(j).name or "").startswith("prop_")
        ]
        self.prop_names = [self.model.joint(j).name for j in self.prop_joints]
        self.target_geom = (
            int(self.model.geom("target_disc").id) if "target_disc" in self.props else None
        )
        self.patch_geoms = [
            int(self.model.geom(f"table_patch{i}").id)
            for i in range(3)
            if "friction_patches" in self.props
        ]
        self.hidden = np.zeros((len(self.objects), len(HIDDEN_COLUMNS)))
        self.mass_dirty = False
        self.props_rng = np.random.default_rng(0)  # replaced per episode by ``record``
        self.effort_rng = np.random.default_rng(0)
        self.target = np.zeros(3)
        self.tips = find_tips(self.model, self.effector)
        self.director = (
            SceneDirector(
                self.model,
                self.director_settings,
                self.objects,
                [object_half_height(self.model, g) for g in self.object_geoms],
                self.table_top,
                self.reach_x,
                [(self.tips[s].body, np.asarray(self.tips[s].offset)) for s in SIDES],
                DT,
            )
            if self.director_settings is not None
            else None
        )
        self.director_rng = np.random.default_rng(0)  # replaced per episode by ``record``
        self.not_foot = np.array(["ankle" not in n for n in self.force_names])
        self.leg_actuators = [
            self.space.index[f"{side}_{joint}"] for side in SIDES for joint in LEG_ACTUATORS
        ]
        self.arm_actuators = [i for side in SIDES for i in self.space.arm[side]]
        self.stock_gains = (self.model.actuator_gainprm.copy(), self.model.actuator_biasprm.copy())
        # Render once so that the first episode of a recorder looks like all later ones: the very
        # first frame differs from the same frame of a repeat in a couple of pixels by one level.
        mujoco.mj_forward(self.model, self.data)
        self.render(self.camera)
        self.render(RIGHT_CAMERA)

    def table_settings(self, director: object) -> dict[str, object]:
        """The settings recorded for a table posture (only those that apply)."""
        settings: dict[str, object] = {
            "table_top": self.table_top,
            "reach_x": list(self.reach_x),
            "safe_reach": self.safe_reach,
            "arm_solref": self.arm_solref,
            "primitives": list(self.primitives or PRIMITIVES[self.effector]),
        }
        optional = {
            "props": list(self.props) if self.props else None,
            "hidden_properties": True if self.hidden_properties else None,
            "program": self.program,
            "task": True if self.task else None,
            "director": director if self.director_settings is not None else None,
            "plan_tries": self.plan_tries,
            "grasp_library": self.grasp_library_paths,
            "grasp_explore": None if self.grasp_explore else False,
        }
        return settings | {k: v for k, v in optional.items() if v is not None}

    def skin_layout(self) -> dict[str, object]:
        """Static positions of every touch channel (see ``h1_layout``)."""
        body_sensors = asset_paths()[0] / "h1" / "touch_sensors_body.xml"
        return skin_layout(
            self.model, self.skin, self.dense, self.grid_kept, str(body_sensors), self.effector
        )

    def joint_name(self, joint: int) -> str:
        """Name of a joint of the model."""
        return str(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint))

    def scale_leg_gains(self, scale: float) -> None:
        """Multiply the legs' PD gains (kp, kv) by ``scale`` relative to the stock model."""
        gain, bias = self.stock_gains
        self.model.actuator_gainprm[:] = gain
        self.model.actuator_biasprm[:] = bias
        for a in self.leg_actuators:
            self.model.actuator_gainprm[a, 0] = gain[a, 0] * scale
            self.model.actuator_biasprm[a, 1:3] = bias[a, 1:3] * scale

    def record(self, seed: int, policy: str, steps: int) -> Episode:
        """Run one episode of at most ``steps`` steps.

        Raises:
            ValueError: unknown policy, or ``seated_babble`` outside the seated posture.
        """
        self.check_policy(policy)
        # Later streams are new; the first ones are unchanged by adding them.
        streams = np.random.SeedSequence(seed).spawn(7)
        layout_rng, policy_rng, sound_rng, look_rng, self.props_rng, self.effort_rng = (
            np.random.default_rng(child) for child in streams[:6]
        )
        self.director_rng = np.random.default_rng(streams[6])
        self.appearance = draw_appearance(look_rng)
        apply_appearance(self.model, self.appearance)
        actor = self.make_policy(policy, policy_rng)
        stance = STANDING_LEG_GAIN_SCALE if self.posture == "standing" else 1.0
        self.scale_leg_gains(getattr(actor, "leg_gain_scale", stance))
        self.reset(actor, layout_rng)
        self.peaks.clear()
        sound = ContactSound(sound_rng, round(DT * SAMPLE_RATE))
        started = time.time()
        log = Log()
        end_reason = "steps"
        ticks = getattr(actor, "substeps_per_tick", 0)  # a policy that runs faster than 20 Hz
        for step in range(steps):
            if self.director is not None:
                log.intervention.append(self.director.step(self.data, step))
            mujoco.mj_forward(self.model, self.data)
            contacts = self.monitor.read(self.data)
            gravity = self.data.xmat[self.pelvis].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
            fell = self.posture == "standing" and (
                self.data.xpos[self.pelvis][2] < FALL_HEIGHT or -gravity[2] < FALL_UPRIGHT
            )
            action = actor.act(step, self.data)
            self.observe(log, contacts, gravity, bool(fell), action, sound)
            self.note_phase(log, actor)
            stop = self.stop_reason(actor, bool(fell))
            if stop:
                end_reason = stop
                break
            ctrl = self.space.to_ctrl(action)
            if getattr(actor, "gravity_compensation", True):
                ctrl = self.with_gravity_compensation(ctrl)
            self.data.ctrl[:] = ctrl
            for sub in range(self.substeps):
                if ticks and (step * self.substeps + sub) % ticks == 0:
                    self.data.ctrl[:] = actor.tick(self.data)  # type: ignore[attr-defined]
                mujoco.mj_step(self.model, self.data)
                contacts_now = read_contacts(self.model, self.data)
                self.dynamics.substep(self.data, contacts_now)
                self.peaks.update(contacts_now)
        plan_stats = getattr(actor, "plan_stats", None)
        return self.episode(
            log,
            seed,
            policy,
            end_reason,
            time.time() - started,
            getattr(actor, "reflexes", None),
            plan_stats() if plan_stats else None,
        )

    def check_policy(self, policy: str) -> None:
        """Raise ``ValueError`` if this recorder cannot run ``policy`` (see ``record``)."""
        if policy not in self.policies:
            raise ValueError(f"humanoid policy must be one of {self.policies}, got {policy!r}")
        if policy == "seated_babble" and not self.has_table:
            raise ValueError("seated_babble needs a table posture: 'seated' or 'stand_supported'")
        if policy == "experiment" and not (self.has_table and self.program):
            raise ValueError("experiment needs a table posture and a program setting")
        if policy == "reach_free" and self.posture != "standing":
            raise ValueError("reach_free needs posture = 'standing'")

    def with_gravity_compensation(self, ctrl: np.ndarray) -> np.ndarray:
        """Add each arm joint's gravity sag to its position target.

        A position actuator settles at ``target - torque / kp``; with the stock
        arm gains (kp 100) an extended arm sags by ~10 cm at the hand. The
        feed-forward ``qfrc_bias / kp`` (valid at the current state, after
        ``mj_forward``) cancels the sag, so the *action* stays "where the joint
        should be". The recorded action is the uncompensated target.
        """
        compensated = ctrl.copy()
        for a in self.arm_actuators:
            joint = self.model.actuator_trnid[a, 0]
            dof = self.model.jnt_dofadr[joint]
            compensated[a] += self.data.qfrc_bias[dof] / self.model.actuator_gainprm[a, 0]
        return compensated

    def make_policy(self, policy: str, rng: np.random.Generator) -> Policy:
        """The scripted policy ``policy`` for this scene."""
        if policy == "hold":
            return HoldPolicy(self.space, self.posture, rng)
        if policy == "reach_free":
            return ReachFreePolicy(self.model, self.space, rng)
        args = (self.model, self.space, self.effector, rng, self.table_top, self.objects)
        if policy == "experiment":
            assert self.program is not None
            return ExperimentPolicy(
                *args,
                reach_x=self.reach_x,
                classes=self.classes,
                program=self.program,
                props=self.props,
                task_mode=self.task,
            )
        if self.safe_reach:
            babble = BabblePolicy(
                *args,
                reach_x=self.reach_x,
                classes=self.classes,
                primitives=self.primitives,
            )
            if self.plan_tries is not None:
                babble.TRIES = self.plan_tries
            if self.grasp_library is not None:
                babble.library, babble.explore = self.grasp_library, self.grasp_explore
            return babble
        return SeatedBabblePolicy(*args, reach_x=self.reach_x)

    def reset(self, actor: Policy, rng: np.random.Generator) -> None:
        """Put the robot (and objects) in the start pose and let them settle."""
        model, data = self.model, self.data
        mujoco.mj_resetData(model, data)
        if self.has_table:
            self.apply_props()  # before the poses: it resets the model's constants
        data.qpos[:] = model.qpos0
        data.qpos[:3] = (0.0, 0.0, STANDING_PELVIS_Z)
        if self.has_table:
            data.qpos[2] = pelvis_height(self.posture)
            data.mocap_pos[0] = (0.0, 0.0, pelvis_height(self.posture))
            data.mocap_quat[0] = (1.0, 0.0, 0.0, 0.0)
            self.place_objects(rng)
            if self.director is not None:
                self.director.reset(data, self.director_rng)
        data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        self.apply_pose(self.space.home(self.posture))
        mujoco.mj_forward(model, data)
        home = actor.reset(data)
        self.apply_pose(home)
        for _ in range(SETTLE_STEPS):
            mujoco.mj_step(model, data, nstep=self.substeps)
        mujoco.mj_forward(model, data)
        self.monitor.reset(data)
        self.dynamics.reset(data)

    def apply_pose(self, ctrl: np.ndarray) -> None:
        """Set the position-controlled joints to ``ctrl`` and make it the target."""
        driven = self.space.qpos_adr >= 0
        self.data.qpos[self.space.qpos_adr[driven]] = ctrl[driven]
        self.data.ctrl[:] = ctrl
        self.data.qvel[:] = 0.0

    def place_objects(self, rng: np.random.Generator) -> None:
        """Scatter the table objects at random non-overlapping positions and yaws."""
        placed: list[np.ndarray] = []
        # the stack blocks share an arm's side (a draw only when they are there: old layouts hold)
        stack_side = float(rng.choice([-1.0, 1.0])) if "stack_blocks" in self.props else 1.0
        for body in self.objects:
            geom = int(self.model.body_geomadr[body])
            half_height = object_half_height(self.model, geom)
            stack = "stackbox" in self.model.body(body).name
            # push to target: the first object starts 17 cm or more from the disc, on its side
            far = self.program == "push_to_target" and not placed
            for attempt in range(200 + 2 * PLACEMENT_RETRIES):
                # 200 tries at a spacing of 11 cm; a crowded table (props) then relaxes it twice
                spacing = SPACINGS[0 if attempt < 200 else 1 + (attempt - 200) // PLACEMENT_RETRIES]
                if stack:
                    xy = np.array([rng.uniform(*STACK_X), stack_side * rng.uniform(*STACK_Y)])
                    spacing = STACK_MIN_GAP
                else:
                    y = rng.uniform(-OBJECT_HALF_WIDTH, OBJECT_HALF_WIDTH)
                    xy = np.array([rng.uniform(*self.reach_x), y])
                if far:
                    side = math.copysign(1.0, self.target[1])
                    xy = np.array([rng.uniform(0.38, 0.56), side * rng.uniform(0.12, 0.30)])
                    if np.linalg.norm(xy - self.target[:2]) < 0.17:
                        continue
                clear = all(np.linalg.norm(xy - other) > spacing for other in placed)
                if clear and not in_exclusion(xy, self.props):
                    break
            placed.append(xy)
            adr = int(self.model.jnt_qposadr[self.model.body_jntadr[body]])
            yaw = rng.uniform(0, 2 * np.pi)
            self.data.qpos[adr : adr + 7] = (
                xy[0], xy[1], self.table_top + half_height + 0.003,
                np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2),
            )  # fmt: skip

    def stop_reason(self, actor: Policy, fell: bool) -> str:
        """``"fall"``, ``"success"`` (a demonstrator whose task is done) or ``""`` to go on."""
        if fell:
            return "fall"
        done = getattr(actor, "done", None)
        return "success" if done is not None and done(self.data) else ""

    @staticmethod
    def note_phase(log: Log, actor: Policy) -> None:
        """Log each arm's attempt phase when the actor is an experiment program."""
        if isinstance(actor, ExperimentPolicy):
            log.phase.append([actor.phase[s] for s in SIDES])

    def apply_props(self) -> None:
        """Restore the nominal object properties, then draw this episode's props and hidden ones.

        The target disc and the patch colours come first and the hidden properties after, all
        from ``props_rng`` (so an episode is a function of its seed). Hidden properties
        (``hidden_properties`` setting) are mass, sliding friction and contact softness of every
        object, drawn independently of its look; they reach ``privileged/hidden_properties`` only.
        """
        model = self.model
        if self.target_geom is not None:
            x = self.props_rng.uniform(*TARGET_X)
            y = self.props_rng.choice([-1.0, 1.0]) * self.props_rng.uniform(*TARGET_Y)
            model.geom_pos[self.target_geom, :2] = (x, y)
            self.target = np.array([x, y, self.table_top])
        if self.patch_geoms:
            colours = np.array([[0.85, 0.35, 0.3, 1], [0.3, 0.55, 0.85, 1], [0.9, 0.8, 0.3, 1]])
            for geom, colour in zip(
                self.patch_geoms, self.props_rng.permutation(colours), strict=True
            ):
                model.geom_rgba[geom] = colour
        n = len(self.objects)
        model.body_mass[self.objects] = self.nominal["mass"]
        model.body_inertia[self.objects] = self.nominal["inertia"]
        model.geom_friction[self.object_geoms, 0] = self.nominal["friction"]
        model.geom_solref[self.object_geoms, 0] = self.nominal["solref"]
        if self.hidden_properties and n:
            self.hidden = draw_hidden(self.props_rng, n)
            scale = self.hidden[:, 0] / self.nominal["mass"]
            model.body_mass[self.objects] = self.hidden[:, 0]
            model.body_inertia[self.objects] = self.nominal["inertia"] * scale[:, None]
            model.geom_friction[self.object_geoms, 0] = self.hidden[:, 1]
            model.geom_solref[self.object_geoms, 0] = self.hidden[:, 2]
        if self.hidden_properties or self.mass_dirty:  # masses changed: refresh the constants
            mujoco.mj_setConst(model, self.data)
        self.mass_dirty = self.hidden_properties and n > 0

    def effort_reading(self) -> np.ndarray:
        """Actuator forces as a sensor would report them: Gaussian noise of 1% of the range,
        clipped to the range and quantised to 12 bits (the data card's noise model)."""
        rng = self.effort_range
        noisy = self.data.actuator_force + self.effort_rng.normal(0.0, 0.01 * rng)
        step = 2 * rng / 4095
        return np.asarray(np.round(np.clip(noisy, -rng, rng) / step) * step)

    def observe(
        self,
        log: Log,
        contacts: ContactFrame,
        gravity: np.ndarray,
        fell: bool,
        action: np.ndarray,
        sound: ContactSound,
    ) -> None:
        """Append every sense of the current step (before ``action`` is applied)."""
        model, data = self.model, self.data
        sensordata = data.sensordata
        force = np.concatenate(
            [
                self.skin.normal_force(sensordata)[self.grid_kept],
                self.dense.read(data, read_contacts(model, data)),
            ]
        )
        log.touch.append(np.concatenate([force, self.dynamics.features()]))
        self.dynamics.clear()
        log.touch_xyz.append(self.skin.force_xyz(sensordata))
        log.rgb.append(self.render(self.camera))
        log.rgb_right.append(self.render(RIGHT_CAMERA))
        qpos = [data.qpos[model.jnt_qposadr[j]] for j in self.joints]
        qvel = [data.qvel[model.jnt_dofadr[j]] for j in self.joints]
        log.proprio.append(
            np.concatenate(
                [
                    qpos,
                    qvel,
                    sensordata[self.gyro : self.gyro + 3],
                    sensordata[self.acc : self.acc + 3],
                    [data.xpos[self.pelvis][2]],
                    gravity,
                    self.effort_reading() if self.effort else [],
                ]
            ).astype(np.float32)
        )
        if self.prop_joints:
            travel = [
                (data.qpos[model.jnt_qposadr[j]] - model.jnt_range[j, 0])
                / (model.jnt_range[j, 1] - model.jnt_range[j, 0])
                for j in self.prop_joints
            ]
            log.prop_state.append(np.clip(travel, 0.0, 1.0).astype(np.float32))  # soft limits
        log.action.append(action.astype(np.float32))
        log.audio.append(sound.step(contacts.impact, contacts.slide))
        log.fell.append(fell)
        log.qpos.append(data.qpos.astype(np.float32))
        log.qvel.append(data.qvel.astype(np.float32))
        log.contacts.append(contacts.table)
        peak, by_object = self.peaks.take()
        log.peak_force.append(peak)
        log.object_contact.append(by_object)
        log.impact_pairs.append(contacts.impact_pairs.astype(np.float32))
        log.objects.append(
            np.array([[*data.xpos[b], *data.xquat[b]] for b in self.objects], dtype=np.float32)
            .reshape(len(self.objects), 7)
        )  # fmt: skip
        log.tips.append(np.array([self.tips[s].position(data) for s in SIDES], dtype=np.float32))
        touching = [
            (part, surface)
            for part, surface in contacts.touches
            if not (part.endswith("foot") and surface == "floor")
        ]
        log.text.append(describe_humanoid(self.posture.replace("_", " "), touching, fell))
        log.skin_touched.append(bool((force[self.not_foot] > TOUCH_THRESHOLD_N).any()))
        log.object_touched.append(contacts.object_contact)
        log.table_touched.append(contacts.table_contact)

    def render(self, camera: str) -> np.ndarray:
        """An ``IMAGE_SIZE`` square uint8 RGB frame from the named camera.

        Decorations are hidden: the ``touch_grid`` plugin draws a coloured patch
        for every active taxel (whatever the category mask), which would be a
        visible copy of the touch signal.
        """
        view = mujoco.MjvCamera()
        view.type = mujoco.mjtCamera.mjCAMERA_FIXED
        view.fixedcamid = self.model.camera(camera).id
        scene = self.renderer.scene
        mujoco.mjv_updateScene(
            self.model, self.data, self.options, None, view, mujoco.mjtCatBit.mjCAT_ALL, scene
        )
        for i in range(scene.ngeom):
            if scene.geoms[i].category == mujoco.mjtCatBit.mjCAT_DECOR:
                scene.geoms[i].rgba[3] = 0.0
        return np.asarray(self.renderer.render())

    def episode(
        self,
        log: Log,
        seed: int,
        policy: str,
        end_reason: str,
        wall_seconds: float,
        reflexes: int | None = None,
        plan_stats: dict[str, object] | None = None,
    ) -> Episode:
        """Assemble the arrays collected in ``log`` into an ``Episode``."""
        audio = to_int16(log.audio)
        steps = len(log.rgb)
        privileged = {
            "touch_xyz": np.stack(log.touch_xyz),
            "rgb_right": np.stack(log.rgb_right).astype(np.uint8),
            "qpos": np.stack(log.qpos),
            "qvel": np.stack(log.qvel),
            "contacts": np.stack(log.contacts),
            "peak_force": np.stack(log.peak_force),
            "object_contact": np.stack(log.object_contact),
            "impact_pairs": np.stack(log.impact_pairs),
            "fall": np.array(log.fell, dtype=bool),
            "object_pose": np.stack(log.objects),
            "tip_position": np.stack(log.tips),
        }
        if log.prop_state:
            privileged["prop_state"] = np.stack(log.prop_state)
        if log.phase:
            privileged["phase"] = np.array(log.phase, dtype=np.int8)
        if log.intervention:
            privileged["intervention"] = np.array(log.intervention, dtype=np.int8)
        if self.target_geom is not None:
            privileged["target"] = np.tile(self.target.astype(np.float32), (steps, 1))
        if self.hidden_properties and self.objects:
            privileged["hidden_properties"] = np.tile(self.hidden.astype(np.float32), (steps, 1, 1))
        meta = {
            "seed": seed,
            "policy": policy,
            "task": task_name(self.program or "")
            if policy == "experiment"
            else TASK_BY_POSTURE[self.posture],
            "program": self.program if policy == "experiment" else None,
            "success": end_reason == "success",
            "instance": f"layout-{seed}" if self.objects else "default",  # object layout
            "effector": self.effector,
            "settings": self.settings,
            "end_reason": end_reason,
            "taxels": len(self.force_names),
            "touch_channels": {
                "grid": int(self.grid_kept.sum()),
                "dense": len(self.dense.names),
                "dynamics": len(self.dynamics.names),
            },
            "appearance": self.appearance.as_meta(),
            "touch_rate": round(float(np.mean(log.skin_touched)), 4),
            "object_contact_rate": round(float(np.mean(log.object_touched)), 4),
            "table_contact_rate": round(float(np.mean(log.table_touched)), 4),
            "fell": bool(any(log.fell)),
            "touch_xyz": "[tangent1, tangent2, normal] in the patch's site frame (z outward)",
            "touch_xyz_names": self.skin.names,
            "body_names": [
                str(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b))
                for b in range(self.model.nbody)
            ],
            "table_top": self.table_top if self.objects else None,
            "reflexes": reflexes,
            "plan_stats": plan_stats,
            "peak_force_columns": list(PEAK_CLASSES),
            "impact_voices": list(VOICES),
            "impact_pairs_note": "[T, voice, voice]: strength (N) of new contacts between a robot "
            "part (first index) and a scene surface (second index), over the control step",
            "peak_force_note": "peak normal force (N) of each class over the physics sub-steps "
            "before this step; see h1_forces.py",
            "contact_columns": "body1, body2, normal force N, x, y, z (top "
            f"{MAX_CONTACT_ROWS} by force, padded with body -1)",
            "object_names": self.object_names,
            "prop_names": self.prop_names,
            "phase_names": list(PHASES) if log.phase else [],
            "hidden_property_columns": HIDDEN_COLUMNS if self.hidden_properties else [],
            "mujoco": mujoco.__version__,
            "assets": asset_revisions(),
            "wall_seconds": round(wall_seconds, 2),
            "steps_per_second": round(steps / max(wall_seconds, 1e-9), 1),
        }
        return Episode(
            body="humanoid",
            dt=DT,
            rgb=np.stack(log.rgb).astype(np.uint8),
            audio=audio,
            audio_logmel=logmel(audio),
            proprio=np.stack(log.proprio),
            touch=np.stack(log.touch),
            action=np.stack(log.action),
            proprio_names=self.proprio_names,
            touch_names=self.touch_names,
            action_names=self.space.names,
            privileged=privileged,
            text=log.text,
            meta=meta,
        )

    def close(self) -> None:
        """Release the renderer."""
        self.renderer.close()

    def __enter__(self) -> "HumanoidRecorder":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def object_half_height(model: mujoco.MjModel, geom: int) -> float:
    """Half the height of a box, sphere or cylinder geom, for resting it on a surface."""
    size = model.geom_size[geom]
    kind = model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return float(size[2])
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return float(size[1])
    return float(size[0])
