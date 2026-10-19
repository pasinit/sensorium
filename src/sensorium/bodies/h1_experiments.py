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

from sensorium.bodies.h1_babble import PHASES, BabblePolicy, Waypoint
from sensorium.bodies.h1_forces import GeomClasses
from sensorium.bodies.h1_policies import ROLL_YAW_SIGN, ActionSpace
from sensorium.bodies.h1_props import (
    BUTTON_POS,
    BUTTON_TRAVEL,
    FLAP_LENGTH,
    FLAP_POS,
    PATCH_X,
)

PROGRAMS = ("push", "topple", "slide", "press", "knock", "door")
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
}
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
    ) -> None:
        if program not in PROGRAMS:
            raise ValueError(f"program must be one of {PROGRAMS}, got {program!r}")
        super().__init__(
            model, space, effector, rng, table_top, object_bodies, reach_x, classes, (program,)
        )
        self.program = program
        self.props = props
        self.names = [model.body(b).name for b in object_bodies]
        self.patch_counts = np.zeros(len(PATCH_X))

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
