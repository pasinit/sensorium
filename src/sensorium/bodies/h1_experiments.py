"""Scripted experiment programs: the robot acts on the world and the world answers.

``ExperimentPolicy`` repeats one *program* per episode (the recorder's ``program`` setting), on
the babbling planner's machinery (clear-posture IK, speed caps, withdrawal reflex). Each program
is a way of making something happen whose outcome the senses carry; ``sensorium.caused_change``
counts the outcomes. Programs, with the props they use (``h1_props``):

* ``push``: side contact, then a 12 cm push (an object moves, or is too heavy or too grippy);
* ``topple``: a push at 25, 50 or 75% of a tall object's height (it tips, slides or stays);
* ``slide``: drag the tip 15 cm along a friction patch (slip, stick-slip, sliding sound);
* ``press``: press a spring button, or an object's top 1 cm deep, and hold (indentation, click);
* ``knock``: strike the table or an object from above at one of three speeds (impact sound);
* ``door``: push a hinged flap open.

The held-out programs (stack, drawer, push to target, turn-reach) are not here: they are the
demonstrators of the few-shot tasks and live with them. Programs are independent per arm; an arm
that cannot do the program of the moment (no flap on its side, no object it can reach) parks.
"""

import math

import mujoco
import numpy as np

from sensorium.bodies.h1_babble import GRASP_HEIGHTS, PHASES, BabblePolicy, Waypoint
from sensorium.bodies.h1_forces import GeomClasses
from sensorium.bodies.h1_policies import ROLL_YAW_SIGN, ActionSpace
from sensorium.bodies.h1_props import (
    BUTTON_POS,
    BUTTON_TRAVEL,
    FLAP_LENGTH,
    FLAP_POS,
    PATCH_X,
)

PROGRAMS = (
    "push",
    "topple",
    "slide",
    "press",
    "knock",
    "door",
    "lift_drop",
    "stack",
    "drawer_open",
    "push_to_target",
)
PINCER_PROGRAMS = ("stack", "drawer_open")  # hands lift (the claw pinch) but do not place or pull
# The demonstrators of the held-out few-shot tasks (T1 stack, T2 drawer, T3 push to the target):
# their episodes carry the held-out task as their task name, not ``exp_<program>``.
HELD_OUT_PROGRAMS = ("stack", "drawer_open", "push_to_target")
SUCCESS_STEPS = 20  # a task counts as done when its test has held this long (1 s)
RESTING_MPS = 0.02  # m/s: a stacked block must be slower than this (babble_stats.MOVING_MPS)
TARGET_NEAR_M = 0.03  # push to target: the object's centre this close to the disc's centre
TARGET_FAR_M = 0.15  # ... for an object that began this far away
PUSH_M = 0.12  # push distance of ``push``
TOPPLE_M = 0.10  # ... of ``topple``
TOPPLE_HEIGHTS = (0.25, 0.5, 0.75)  # fraction of the object's height at which it is pushed
SLIDE_M = (0.12, 0.27)  # a slide runs between these |y| on its side
PRESS_M = 0.010  # an object is pressed this far below its top
PRESS_HOLD = 15  # control steps held down
# Pincer approach heights (z of its approach axis: -1 straight down) per program; None keeps the
# babbling default (-0.8 to -1.0). A near-vertical pincer cannot reach past x of about 0.5 m.
TILTS: dict[str, tuple[float, ...] | None] = {
    "push": None,
    "topple": (-0.5, -0.7),
    "slide": None,
    "press": None,
    "knock": None,
    "door": (-0.5, -0.7),
    "lift_drop": None,
    "stack": None,
    "drawer_open": None,
    "push_to_target": None,
}
LIFT_HEIGHTS = (0.05, 0.10, 0.15)  # m a lift-drop program lifts to before releasing
STACK_GAP = 0.04  # m above the target's top a carried block travels
STACK_DWELL = 50  # control steps (2.5 s) the arm waits after stacking
PULL_M = 0.07  # a drawer is pulled or pushed this far
PINCER_REACH_X = 0.55  # a near-vertical pincer plans targets no further than this
HOME_BIAS = 0.6  # share of pushes aimed back towards the arm's reachable middle ground
HOME_X, HOME_Y = 0.47, 0.22
KNOCK_SPEEDS = (0.012, 0.03, 0.06)  # joint step caps (rad per step) of the strike
FLAP_PUSH_M = 0.05


class ExperimentPolicy(BabblePolicy):
    """One experiment program, repeated by both arms (see the module docstring)."""

    # Denser than the babbling defaults: a lower hover, a faster descent (no contact is made at
    # its end), and faster pushes (joint step caps, rad per control step).
    HOVER = 0.06
    DESCENT_SPEED = 0.03
    PUSH_SPEED = 0.02

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
        program: str,
        props: tuple[str, ...] = (),
        task_mode: bool = False,
    ) -> None:
        if program not in PROGRAMS:
            raise ValueError(f"program must be one of {PROGRAMS}, got {program!r}")
        super().__init__(
            model, space, effector, rng, table_top, object_bodies, reach_x, classes, (program,)
        )
        self.program = program
        self.props = props
        if program == "push_to_target":
            self.HOVER = (
                BabblePolicy.HOVER
            )  # clear of the objects' tops: a low hover sweeps them away
        self.task_mode = task_mode  # a demonstrator: one job, ends when its success test holds
        self.success_steps = 0
        self.far_at_start: np.ndarray | None = None  # push to target: objects that began far away
        if "drawer" in props:
            self.START_Y = 0.12  # the arms start clear of the drawers (|y| 0.24 and out)
        self.names = [model.body(b).name for b in object_bodies]
        self.patch_counts = np.zeros(len(PATCH_X))
        self.height_counts = np.zeros(len(LIFT_HEIGHTS))

    def tilt(self) -> float | None:
        """The pincer's approach height for this program (see ``TILTS``); None for hands."""
        choices = TILTS[self.program]
        if not self.pincer or choices is None:
            return super().tilt()
        return float(self.rng.choice(choices))

    # ------------------------------------------------------------------ the shared shape

    def contact_path(
        self,
        side: str,
        data: mujoco.MjData,
        start: np.ndarray,
        end: np.ndarray,
        *,
        along_speed: float | None = None,
        down_speed: float | None = None,
        hold: int = 0,
    ) -> list[Waypoint] | None:
        """Hover over ``start``, descend, move ``start`` to ``end``, hold, rise, retreat.

        The shape of every contact program. A pincer closes for the push (its blunt side) after
        the descent; speeds default to the contact speed ``SLOW``.
        """
        along_speed = self.SLOW if along_speed is None else along_speed
        down_speed = self.DESCENT_SPEED if down_speed is None else down_speed
        tilt = self.tilt()
        hover = self.to_hover(data, side, start, tilt)
        if hover is None:
            return None
        q_hover, queue = hover
        down = self.line(
            data, side, q_hover, self.hover_point(start), start, down_speed, 0.01, tilt
        )
        if down is None:
            return None
        along = self.line(data, side, down[-1].q, start, end, along_speed, 0.01, tilt)
        if along is None:
            return None
        up = self.line(data, side, along[-1].q, end, self.hover_point(end), self.FAST, 0.02, tilt)
        if up is None:
            return None
        grip = float(self.pincer)
        contact, descent, retreat = (PHASES.index(p) for p in ("contact", "descent", "retreat"))
        for w in down:
            w.phase = descent
        for w in up:
            w.phase = retreat
        queue += down + [Waypoint(w.q, w.speed, 0, grip, phase=contact) for w in along]
        queue.append(Waypoint(along[-1].q, along_speed, hold, grip, phase=contact))
        queue = self.finish(side, queue + up, up[-1].q)
        queue[-1].phase = retreat
        return queue

    def side_push(
        self, side: str, data: mujoco.MjData, index: int, height: float, distance: float
    ) -> list[Waypoint] | None:
        """Touch object ``index`` from the side at ``height`` over the table and push it."""
        centre, _, radius = self.object_geometry(data, index)
        angle = float(self.rng.uniform(-math.pi, math.pi))  # any direction: far objects need it
        if self.rng.random() < HOME_BIAS:  # keep the objects within reach: pushes drift them away
            home = np.array([HOME_X, math.copysign(HOME_Y, centre[1])])
            angle = math.atan2(home[1] - centre[1], home[0] - centre[0]) + float(
                self.rng.uniform(-0.5, 0.5)
            )
        push = np.array([math.cos(angle), math.sin(angle), 0.0])
        start = centre - (radius + 0.03) * push
        start[2] = self.table_top + height
        end = centre + push * distance
        end[2] = start[2]
        return self.contact_path(side, data, start, end, along_speed=self.PUSH_SPEED)

    # ------------------------------------------------------------------ the programs

    def plan_push(self, side: str, data: mujoco.MjData) -> list[Waypoint] | None:
        """Push an object (fewest tried first) 12 cm from the side, low."""
        index = self.pick_object(side, data)
        if index is None:
            return None
        _, half_height, _ = self.object_geometry(data, index)
        return self.side_push(side, data, index, 0.8 * half_height, PUSH_M)

    def plan_topple(self, side: str, data: mujoco.MjData) -> list[Waypoint] | None:
        """Push a tall object at a quarter, half or three quarters of its height."""
        tall = [i for i in self.free_objects(data, side) if "tall" in self.names[i]]
        if not tall:
            return self.plan_push(
                side, data
            )  # nothing tall on this side: still make something move
        index = tall[self.choose(self.object_counts[tall])]
        self.object_counts[index] += 1
        _, half_height, _ = self.object_geometry(data, index)
        fraction = float(self.rng.choice(TOPPLE_HEIGHTS))
        return self.side_push(side, data, index, fraction * 2 * half_height, TOPPLE_M)

    def plan_slide(self, side: str, data: mujoco.MjData) -> list[Waypoint] | None:
        """Drag the tip 15 cm along a friction patch (fewest tried first), either way."""
        reachable = [i for i, x in enumerate(PATCH_X) if not self.pincer or x <= PINCER_REACH_X]
        patch = reachable[self.choose(self.patch_counts[reachable])]
        self.patch_counts[patch] += 1
        sign = ROLL_YAW_SIGN[side]
        near, far = SLIDE_M
        if self.rng.random() < 0.5:
            near, far = far, near
        z = self.table_top - self.PRESS_DEPTH
        start = np.array([PATCH_X[patch], sign * near, z])
        end = np.array([PATCH_X[patch], sign * far, z])
        return self.contact_path(side, data, start, end)

    def plan_press(self, side: str, data: mujoco.MjData) -> list[Waypoint] | None:
        """Press this side's button, or an object's top 1 cm deep, and hold."""
        sign = ROLL_YAW_SIGN[side]
        if "button" in self.props and self.rng.random() < 0.5:
            top = self.table_top + 0.034  # the button's top at rest
            goal = np.array([BUTTON_POS[0], sign * BUTTON_POS[1], top - BUTTON_TRAVEL - 0.002])
        else:
            index = self.pick_object(side, data)
            if index is None:
                return None
            centre, half_height, _ = self.object_geometry(data, index)
            if self.pincer and centre[0] > PINCER_REACH_X:
                return None
            goal = np.array([centre[0], centre[1], centre[2] + half_height - PRESS_M])
        return self.contact_path(side, data, goal, goal.copy(), hold=PRESS_HOLD)

    def plan_knock(self, side: str, data: mujoco.MjData) -> list[Waypoint] | None:
        """Strike the table or an object's top from above at one of three speeds."""
        speed = float(self.rng.choice(KNOCK_SPEEDS))
        index = self.pick_object(side, data) if self.rng.random() < 0.5 else None
        if index is None:
            goal = self.table_point(side)
        else:
            centre, half_height, _ = self.object_geometry(data, index)
            goal = np.array([centre[0], centre[1], centre[2] + half_height])
        goal[2] -= self.PRESS_DEPTH
        return self.contact_path(side, data, goal, goal.copy(), down_speed=speed)

    def plan_door(self, side: str, data: mujoco.MjData) -> list[Waypoint] | None:
        """Push this side's hinged flap open from the robot's side."""
        if "hinged_flap" not in self.props:
            return None
        sign = ROLL_YAW_SIGN[side]
        y_flap = sign * (FLAP_POS[1] - FLAP_LENGTH / 2) + float(self.rng.uniform(-0.02, 0.02))
        start = np.array([FLAP_POS[0] - 0.04, y_flap, self.table_top + 0.04])
        end = start + np.array([FLAP_PUSH_M, 0.0, 0.0])
        return self.contact_path(side, data, start, end)

    # ------------------------------------------------------------------ tasks (demonstrators)

    def succeeded(self, data: mujoco.MjData) -> bool:
        """The success test of this program's task at this step (see ``done``)."""
        if self.program == "stack":
            blocks = [i for i, n in enumerate(self.names) if "stackbox" in n]
            state = self.stacked_on(data, blocks) if len(blocks) == 2 else None
            return state is not None and self.resting_on(data, *state)
        if self.program == "drawer_open":
            for letter in ("l", "r"):
                joint = self.model.joint(f"prop_drawer_{letter}")
                low, high = self.model.jnt_range[joint.id]
                if (data.qpos[joint.qposadr[0]] - low) / (high - low) > 0.5:
                    return True
            return False
        if self.program == "push_to_target":
            target = self.target_xy(data)
            near = [
                float(np.linalg.norm(data.xpos[b][:2] - target)) <= TARGET_NEAR_M
                for b in self.objects
            ]
            if self.far_at_start is None:
                self.far_at_start = np.array(
                    [
                        np.linalg.norm(data.xpos[b][:2] - target) >= TARGET_FAR_M
                        for b in self.objects
                    ]
                )
            return bool((np.array(near) & self.far_at_start).any())
        return False

    def done(self, data: mujoco.MjData) -> bool:
        """True once a demonstrator's success test has held for ``SUCCESS_STEPS`` steps."""
        if not self.task_mode:
            return False
        self.success_steps = self.success_steps + 1 if self.succeeded(data) else 0
        return self.success_steps >= SUCCESS_STEPS

    def target_xy(self, data: mujoco.MjData) -> np.ndarray:
        """The target disc's centre (x, y)."""
        return np.array(data.geom_xpos[self.model.geom("target_disc").id][:2])

    def plan_push_to_target(self, side: str, data: mujoco.MjData) -> list[Waypoint] | None:
        """Push an object towards the target disc, from behind it (several pushes may follow)."""
        if "target_disc" not in self.props:
            return None
        target = self.target_xy(data)
        if self.far_at_start is None:
            self.succeeded(data)  # sets the objects that began far from the target
        sign = ROLL_YAW_SIGN[side]
        if sign * target[1] < self.midline:
            return None  # the disc is on the other arm's side
        far = [
            i
            for i in self.free_objects(data, side)
            if self.far_at_start is not None and self.far_at_start[i]
        ]
        movable = [i for i in far if "tall" not in self.names[i]]
        if not movable:
            return None
        # the object nearest the disc that is not on it yet (boxes and cylinders: a sphere rolls on)
        steady = [i for i in movable if "sphere" not in self.names[i]] or movable
        index = min(
            steady, key=lambda i: float(np.linalg.norm(data.xpos[self.objects[i]][:2] - target))
        )
        centre, half_height, radius = self.object_geometry(data, index)
        gap = target - centre[:2]
        distance = float(np.linalg.norm(gap))
        if distance < TARGET_NEAR_M:
            return None
        direction = np.array([gap[0] / distance, gap[1] / distance, 0.0])
        start = centre - (radius + 0.03) * direction
        start[2] = self.table_top + 0.8 * half_height
        end = np.array([target[0], target[1], start[2]]) - (radius + 0.002) * direction
        return self.contact_path(side, data, start, end, along_speed=self.PUSH_SPEED)

    # ------------------------------------------------------------------ pincer programs

    def grasp_moves(
        self,
        side: str,
        data: mujoco.MjData,
        centre: np.ndarray,
        moves: list[tuple[np.ndarray, float]],
        *,
        grazed: list[int] | None = None,
        hold: int = 0,
        dwell: int = 0,
    ) -> list[Waypoint] | None:
        """Pinch at ``centre``, make each ``(target, speed)`` move while gripping, release, retreat.

        The geoms in ``grazed`` (the grasped object or knob) may be grazed by the forearm;
        ``hold`` steps are held gripping after the last move, and the arm stays ``dwell`` steps at
        its retreat (a stack needs time to be seen as one). Returns None if the grasp, a move or
        the retreat has no clear plan.
        """
        self.exempt[:] = False
        self.exempt[grazed or []] = True
        approach_z = float(self.rng.choice(GRASP_HEIGHTS))
        found = self.oriented_pose(data, side, centre, approach_z)
        if found is None:
            return None
        q_grasp, advance = found
        pre = centre - self.frames[side].back_off * advance
        above = pre + np.array([0.0, 0.0, self.APPROACH_MARGIN])
        q_above, pos_error, ori_error = self.oriented[side].solve(
            data.qpos, q_grasp, above, approach_z, self.axis_weight
        )
        if pos_error > 0.01 or ori_error > 0.05 or not self.posture_ok(data, side, q_above):
            return None
        to_pre = self.line(data, side, q_above, above, pre, self.SLOW, 0.02, approach_z)
        onto = (
            self.line(data, side, to_pre[-1].q, pre, centre, self.SLOW, 0.012, approach_z)
            if to_pre
            else None
        )
        if not (to_pre and onto):
            return None
        lead = [Waypoint(q_above, self.FAST)]
        if not self.path_ok(data, side, self.cmd_q[side], q_above):
            start = self.cmd_q[side]
            tip = self.ik[side].tip_at(data.qpos, start)
            via = self.line(data, side, start, tip, above, self.FAST, self.TRAVEL_STEP, approach_z)
            if via is None:
                return None
            lead = via
        travel, descent, contact, retreat = (
            PHASES.index(p) for p in ("travel", "descent", "contact", "retreat")
        )
        queue = [Waypoint(w.q, w.speed, w.hold, w.grip, phase=travel) for w in lead]
        queue += [Waypoint(w.q, w.speed, 0, 0.0, phase=descent) for w in [*to_pre, *onto]]
        queue.append(Waypoint(onto[-1].q, self.SLOW, 3, 0.0, phase=descent))
        queue.append(Waypoint(onto[-1].q, self.SLOW, 14, 1.0, phase=contact))  # close
        q, point = onto[-1].q, centre
        for target, speed in moves:
            leg = self.line(data, side, q, point, target, speed, 0.015, approach_z)
            if leg is None:
                return None
            queue += [Waypoint(w.q, w.speed, 0, 1.0, phase=contact) for w in leg]
            q, point = leg[-1].q, target
        queue.append(Waypoint(q, self.SLOW, hold, 1.0, phase=contact))
        queue.append(Waypoint(q, self.SLOW, 6, 0.0, phase=contact))  # release
        out = self.line(
            data, side, q, point, point + np.array([0.0, 0.0, 0.06]), self.LIFT, 0.02, approach_z
        )
        if out is None:
            return None
        queue += [Waypoint(w.q, w.speed, 0, 0.0, phase=retreat) for w in out]
        queue = self.finish(side, queue, out[-1].q)
        queue[-1].phase = retreat  # ``plan`` clears the exemption after checking the queue
        queue[-1].hold = dwell
        return queue

    def body_geoms(self, index: int) -> list[int]:
        """Geom ids of object ``index``."""
        body = self.objects[index]
        first = int(self.model.body_geomadr[body])
        return list(range(first, first + int(self.model.body_geomnum[body])))

    def plan_lift_drop(self, side: str, data: mujoco.MjData) -> list[Waypoint] | None:
        """Grasp an object, lift it 5, 10 or 15 cm (fewest tried first), hold, let it fall."""
        index = self.pick_object(side, data)
        if index is None:
            return None
        height = float(LIFT_HEIGHTS[self.choose(self.height_counts)])
        self.height_counts[LIFT_HEIGHTS.index(height)] += 1
        centre, _, _ = self.object_geometry(data, index)
        return self.grasp_moves(
            side,
            data,
            centre,
            [(centre + [0, 0, height], self.LIFT)],
            grazed=self.body_geoms(index),
            hold=12,
        )

    def stacked_on(self, data: mujoco.MjData, blocks: list[int]) -> tuple[int, int] | None:
        """``(top, base)`` if one of the two blocks rests on the other, else None."""
        for top in blocks:
            for base in blocks:
                if top == base:
                    continue
                above = data.xpos[self.objects[top]] - data.xpos[self.objects[base]]
                if np.linalg.norm(above[:2]) < 0.02 and above[2] > 0.02:
                    return top, base
        return None

    def resting_on(self, data: mujoco.MjData, top: int, base: int) -> bool:
        """True if block ``top`` rests on ``base``: still, on it, and touched by no robot part.

        A block held over the other, or still being set down, is not a stack.
        """
        dof = self.model.jnt_dofadr[self.model.body_jntadr[self.objects[top]]]
        if float(np.linalg.norm(data.qvel[dof : dof + 3])) >= RESTING_MPS:
            return False
        top_geoms, base_geoms = set(self.body_geoms(top)), set(self.body_geoms(base))
        on_base = False
        for i in range(data.ncon):
            pair = (int(data.contact[i].geom1), int(data.contact[i].geom2))
            for mine, other in (pair, pair[::-1]):
                if mine in top_geoms:
                    if self.classes.robot[other]:
                        return False  # still held or pushed
                    on_base |= other in base_geoms
        return on_base

    def spot_beside(self, base: np.ndarray, side: str, half_height: float) -> np.ndarray:
        """A table spot 9 cm from ``base`` that this arm can reach: in y, else in x."""
        sign = ROLL_YAW_SIGN[side]
        offsets = [(0.0, 0.09), (0.0, -0.09), (0.08, 0.0), (-0.08, 0.0)]
        for dx, dy in self.rng.permutation(offsets):
            x, y = base[0] + dx, base[1] + dy
            if sign * y >= self.midline + 0.02 and sign * y <= self.TABLE_Y - 0.02:
                if self.reach_x[0] <= x <= min(self.reach_x[1], PINCER_REACH_X):
                    return np.array([x, y, self.table_top + half_height + 0.004])
        return np.array([base[0], base[1] + sign * 0.09, self.table_top + half_height + 0.004])

    def plan_stack(self, side: str, data: mujoco.MjData) -> list[Waypoint] | None:
        """Stack one block on the other; if stacked, take the top one off and set it aside."""
        blocks = [i for i in self.free_objects(data, side) if "stackbox" in self.names[i]]
        if len(blocks) < 2:
            return None
        state = self.stacked_on(data, blocks)
        if state is not None and self.task_mode:
            return None  # the job is done: hold still
        if state is None:
            order = [int(i) for i in self.rng.permutation(blocks)]
            moved, base = order[0], order[1]
            c_moved, h_moved, _ = self.object_geometry(data, moved)
            c_base, h_base, _ = self.object_geometry(data, base)
            lift = c_base[2] + h_base + h_moved + STACK_GAP - c_moved[2]
            over = np.array([c_base[0], c_base[1], c_moved[2] + lift])
            place = np.array([c_base[0], c_base[1], c_base[2] + h_base + h_moved + 0.004])
            moves = [(c_moved + [0, 0, lift], self.LIFT), (over, self.LIFT), (place, self.SLOW)]
        else:
            moved, base = state
            c_moved, h_moved, _ = self.object_geometry(data, moved)
            c_base, _, _ = self.object_geometry(data, base)
            aside = self.spot_beside(c_base, side, h_moved)
            over = np.array([aside[0], aside[1], c_moved[2] + 0.02])
            moves = [(c_moved + [0, 0, 0.03], self.LIFT), (over, self.LIFT), (aside, self.SLOW)]
        dwell = 0 if state else STACK_DWELL  # a new stack stays as it is for a while
        return self.grasp_moves(
            side, data, c_moved, moves, grazed=self.body_geoms(moved), hold=4, dwell=dwell
        )

    def plan_drawer_open(self, side: str, data: mujoco.MjData) -> list[Waypoint] | None:
        """Pinch this side's drawer knob and pull the drawer open, or push it shut if it is open."""
        if "drawer" not in self.props:
            return None
        letter = "l" if ROLL_YAW_SIGN[side] > 0 else "r"
        knob = np.array(data.geom_xpos[self.model.geom(f"table_drawer_{letter}_knob").id])
        joint = self.model.joint(f"prop_drawer_{letter}")
        low, high = self.model.jnt_range[joint.id]
        opened = (data.qpos[joint.qposadr[0]] - low) / (high - low) > 0.5
        if opened and self.task_mode:
            return None  # the job is done
        direction = 1.0 if opened else -1.0  # push shut (+x) or pull open (-x)
        target = knob + np.array([direction * PULL_M, 0.0, 0.0])
        grazed = [int(self.model.geom(f"table_drawer_{letter}_knob").id)]
        return self.grasp_moves(side, data, knob, [(target, self.SLOW)], grazed=grazed, hold=3)


def check_program(program: str | None, effector: str) -> str | None:
    """Validate a recorder's ``program`` setting for its effector; returns it.

    Raises:
        ValueError: for an unknown program, or a pincer-only program with hands.
    """
    if program is not None and program not in PROGRAMS:
        raise ValueError(f"program must be one of {PROGRAMS}, got {program!r}")
    if program in PINCER_PROGRAMS and effector != "pincer":
        raise ValueError(f"program {program!r} needs the pincer")
    return program


def task_name(program: str) -> str:
    """The task name an episode of ``program`` carries (a held-out program is its own task)."""
    return program if program in HELD_OUT_PROGRAMS else f"exp_{program}"
