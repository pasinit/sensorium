"""Reachability of hand grasp poses on the table, with the Shadow wrist held or free.

Usage:
    uv run python scripts/hand_reach.py <object body name> <placements> [--first N] [--out FILE]

The scene is the event-dense hand block (``configs/event_dense.toml``) with the director off. Each
placement puts the object alone at a random place on a random arm's side, at a random yaw, upright
or (not spheres, half of the time) on a side face (as ``scripts/grasp_library.py``). For each grasp
type the planner's oriented IK looks for a posture from ``STARTS`` random starts:

* ``claw``: the claw pinch from above (``h1_babble.grasp_frame``): palm normal at ``HAND_PALM_Z``,
  fingers level or 0.1 down, the pinch point at the object's near face;
* ``wrap<z>``: a side power wrap: palm normal at height ``z`` (``PALM_TILTS``: level or turned
  down onto the object), the palm's face ``WRAP_GAP`` off the object, fingers level, 0.3 or 0.6
  down.

and three sets of joints: the 5 arm joints with the Shadow wrist held at the pinch's angles (as the
planner does), the wrist's flexion (``WRJ1``, pitch) free too, and its deviation (``WRJ2``) free as
well. Reached: the grasp point within 1 cm and the axes' heights within 0.05, the arm clear (as
``BabblePolicy.posture_ok`` for a grasp) and the hand not into the table. Appends one JSON line per
placement: ``{"object", "rest", "side", "reached": {"<type>/<joints>": status}}``, the status
``ok``, or what the best posture found ran into (``object``, ``table``, ``arm``), or ``ik``.
"""

import argparse
import dataclasses
import json
import tomllib
from collections.abc import Iterable
from pathlib import Path

import mujoco
import numpy as np
from grasp_library import place_alone  # scripts/ is on the path when run

from sensorium.bodies.h1_babble import (
    HAND_GRASP_HEIGHTS,
    BabblePolicy,
    GraspFrame,
    OrientedIK,
)
from sensorium.bodies.h1_grasp_library import grasp_key
from sensorium.bodies.humanoid import IMAGE_SIZE, HumanoidRecorder

STARTS = 16
WRAP_HEIGHTS = (0.0, -0.3, -0.6)  # z of the fingers' direction in a wrap
PALM_TILTS = (0.0, -0.4, -0.7)  # z of the wrap's palm normal: level, or turned down onto the object
WRAP_GAP = 0.005  # m between the palm's face and the object
WRAP_MIN_HEIGHT = 0.05  # m over the table the palm's centre stays at least (half the hand's width)
PALM_FACE = (0.045, 0.02)  # m along the fingers and the palm normal: the palm face's centre
JOINTS = {"arm": (), "+pitch": ("WRJ1",), "+pitch+deviation": ("WRJ1", "WRJ2")}


class FreerIK(OrientedIK):
    """``OrientedIK`` with some of the Shadow wrist's joints solved for instead of held."""

    def __init__(
        self, model: mujoco.MjModel, side: str, frame: GraspFrame, free: tuple[str, ...]
    ) -> None:
        prefix = "lh_" if side == "left" else "rh_"
        joints = [model.joint(prefix + name) for name in free]
        adrs = {int(j.qposadr[0]) for j in joints}
        super().__init__(
            model,
            side,
            dataclasses.replace(frame, fixed=tuple(f for f in frame.fixed if f[0] not in adrs)),
        )
        self.qpos_adr = np.append(self.qpos_adr, [int(j.qposadr[0]) for j in joints]).astype(int)
        self.dof_adr = np.append(self.dof_adr, [int(j.dofadr[0]) for j in joints]).astype(int)
        self.low = np.append(self.low, [model.jnt_range[j.id, 0] for j in joints])
        self.high = np.append(self.high, [model.jnt_range[j.id, 1] for j in joints])
        self.fixed = frame.fixed


def clash(policy: BabblePolicy, data: mujoco.MjData, side: str, ik: FreerIK, q: np.ndarray) -> str:
    """What posture ``q`` runs into: ``"arm"`` (the grasp clearance), ``"table"`` or
    ``"object"`` (the hand, by more than 5 mm), or ``""`` if nothing."""
    probe = mujoco.MjData(policy.model)
    probe.qpos[:] = data.qpos
    for adr, value in ik.fixed:
        probe.qpos[adr] = value
    probe.qpos[ik.qpos_adr] = q
    if not policy.posture_ok(probe, side, q[:5]):
        return "arm"
    mujoco.mj_kinematics(policy.model, probe)
    mujoco.mj_collision(policy.model, probe)
    classes = policy.classes
    for i in range(probe.ncon):
        g = (int(probe.contact[i].geom1), int(probe.contact[i].geom2))
        hand = [classes.effector[x] for x in g]
        if any(hand) and not all(hand) and probe.contact[i].dist < -0.005:
            other = g[1] if hand[0] else g[0]
            if not classes.robot[other]:
                return "object" if classes.object_index[other] >= 0 else "table"
    return ""


def reach(
    policy: BabblePolicy, data: mujoco.MjData, side: str, ik: FreerIK, goal_of, approach_z: float
) -> str:
    """``"ok"`` if some start gives a clear posture with the grasp point at ``goal_of(q)``;
    else what the last posture found ran into (``clash``), or ``"ik"`` if none was found."""
    rng = np.random.default_rng(0)
    status = "ik"
    for _ in range(STARTS):
        q = rng.uniform(ik.low, ik.high)
        goal = goal_of(None)
        q, _, _ = ik.solve(data.qpos, q, goal, approach_z, 0.3, iterations=150)
        goal = goal_of(q)  # the goal depends on which way the hand faces: solve again
        q, pos_error, ori_error = ik.solve(data.qpos, q, goal, approach_z, 0.3)
        if pos_error < 0.01 and ori_error < 0.05:
            status = clash(policy, data, side, ik, q) or "ok"
            if status == "ok":
                return status
    return status


def best(statuses: Iterable[str]) -> str:
    """The best of several ``reach`` results: ok, then a clash (object, table, arm), then ik."""
    found = set(statuses)
    return next(s for s in ("ok", "object", "table", "arm", "ik") if s in found)


def placement(rec: HumanoidRecorder, body: int, seed: int) -> dict[str, object]:
    """One placement of ``body`` and which grasp types and joint sets reach it."""
    rng = np.random.default_rng(seed)
    policy = rec.make_policy("seated_babble", rng)
    assert isinstance(policy, BabblePolicy)
    rec.reset(policy, rng)
    side = "left" if rng.random() < 0.5 else "right"
    place_alone(rec, body, side, rng)
    data, index = rec.data, rec.objects.index(body)
    policy.clearance = policy.GRASP_CLEARANCE
    policy.exempt[:] = False
    policy.exempt[
        rec.model.body_geomadr[body] : rec.model.body_geomadr[body] + rec.model.body_geomnum[body]
    ] = True
    claw = policy.frames[side]
    centre = policy.grasp_centre(data, index)
    side_centre = np.array(data.xpos[body])
    side_centre[2] = max(side_centre[2], policy.table_top + WRAP_MIN_HEIGHT)
    reached = {}
    for joints, free in JOINTS.items():
        claw_ik = FreerIK(rec.model, side, claw, free)

        def near_face(q: np.ndarray | None, ik: FreerIK = claw_ik) -> np.ndarray:
            if q is None:
                return centre
            fingers = ik.frame_at(data.qpos, q)[1]
            return centre - policy.half_extent(data, index, fingers) * fingers

        reached[f"claw/{joints}"] = best(
            reach(policy, data, side, claw_ik, near_face, z) for z in HAND_GRASP_HEIGHTS
        )
        for tilt in PALM_TILTS:
            wrap = dataclasses.replace(
                claw, offset=PALM_FACE[0] * claw.approach + PALM_FACE[1] * claw.axis, axis_z=tilt
            )
            wrap_ik = FreerIK(rec.model, side, wrap, free)

            def palm_side(q: np.ndarray | None, ik: FreerIK = wrap_ik) -> np.ndarray:
                if q is None:
                    return side_centre
                normal = ik.frame_at(data.qpos, q)[2]
                return side_centre - (policy.half_extent(data, index, normal) + WRAP_GAP) * normal

            reached[f"wrap{tilt:+.1f}/{joints}"] = best(
                reach(policy, data, side, wrap_ik, palm_side, z) for z in WRAP_HEIGHTS
            )
    key = grasp_key(rec.model, data, body)
    return {"object": key[0], "rest": key[1], "side": side, "reached": reached}


def main() -> None:
    """Run the placements and append the records."""
    parser = argparse.ArgumentParser()
    parser.add_argument("object")
    parser.add_argument("placements", type=int)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/hand_reach/reach.jsonl"))
    args = parser.parse_args()
    blocks = tomllib.loads(Path("configs/event_dense.toml").read_text())["humanoid"]
    spec = next(s for name, s in blocks.items() if name.startswith("hand"))
    settings = {k: v for k, v in dict(spec["settings"]).items() if k != "director"}
    HumanoidRecorder.render = lambda self, camera: np.zeros(  # type: ignore[method-assign]
        (IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8
    )
    rec = HumanoidRecorder(settings)
    body = rec.model.body(args.object).id
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as handle:
        for k in range(args.first, args.first + args.placements):
            handle.write(json.dumps(placement(rec, body, 1000 * k + 11)) + "\n")
            handle.flush()
    print(f"{args.object}: {args.placements} placements", flush=True)


if __name__ == "__main__":
    main()
