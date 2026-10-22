"""Make a grasp library (``bodies/h1_grasp_library.py``): execute varied grasps of one object alone.

Usage:
    uv run python scripts/grasp_library.py <hand|pincer> <object body name> <trials>
        [--first N] [--out FILE] [--config configs/event_dense.toml]

The scene is the config's first block for the effector (its props, hidden object properties), with
the scene director off. Each trial: reset (fresh hidden properties), the other objects on the floor,
the object put at a random place within reach on a random arm's side, at a random yaw, upright or
(not spheres, half of the time) on a side face, settled; one grasp planned by the babbling planner
in ``explore`` mode (random IK starts and postures, a jittered grasp point), executed with the
recorder's control loop. Lifted: the object 3 cm over its start for 0.5 s, touched (as
``scripts/grasp_success.py``). Appends one JSON line per planned grasp to ``--out``:
``{"object", "rest", "features", "lifted", "trial"}`` (unplanned trials are counted and printed).
"""

import argparse
import json
import math
import tomllib
from pathlib import Path

import mujoco
import numpy as np

from sensorium.babble_stats import runs
from sensorium.bodies.h1_babble import BabblePolicy, Waypoint
from sensorium.bodies.humanoid import IMAGE_SIZE, HumanoidRecorder, object_half_height

LIFT_M, LIFT_STEPS, MAX_STEPS = 0.03, 10, 300


def vertical_half(model: mujoco.MjModel, geom: int, quat: np.ndarray) -> float:
    """Half the height of a box, cylinder or sphere geom turned by ``quat``."""
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, quat)
    up = rotation.reshape(3, 3)[2]  # world z in the geom's axes
    size, kind = model.geom_size[geom], int(model.geom_type[geom])
    if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        return float(sum(abs(up[k]) * size[k] for k in range(3)))
    if kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        along = abs(float(up[2]))
        return float(size[0] * math.sqrt(max(0.0, 1 - along * along)) + size[1] * along)
    return object_half_height(model, geom)


def place_alone(rec: HumanoidRecorder, body: int, side: str, rng: np.random.Generator) -> None:
    """``body`` alone on the table: random place on ``side``, yaw and resting face."""
    m, d = rec.model, rec.data
    for k, b in enumerate(rec.objects):
        adr = m.jnt_qposadr[m.body_jntadr[b]]
        if b != body:
            d.qpos[adr : adr + 7] = (-1.0, -1.0 + 0.3 * k, 0.1, 1, 0, 0, 0)
    geom = int(m.body_geomadr[body])
    sign = 1.0 if side == "left" else -1.0
    x, y = rng.uniform(*rec.reach_x), sign * rng.uniform(0.08, 0.28)
    yaw = rng.uniform(0, 2 * np.pi)
    quat = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    sphere = int(m.geom_type[geom]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
    if not sphere and rng.random() < 0.5:  # on a side face: turned 90 degrees about x or y
        tilt = np.array([math.cos(math.pi / 4), 0.0, 0.0, 0.0])
        tilt[1 + int(rng.integers(2))] = math.sin(math.pi / 4)
        turned = np.zeros(4)
        mujoco.mju_mulQuat(turned, quat, tilt)
        quat = turned
    half = vertical_half(m, geom, quat)
    adr = m.jnt_qposadr[m.body_jntadr[body]]
    d.qpos[adr : adr + 7] = (x, y, rec.table_top + half + 0.01, *quat)
    d.qvel[:] = 0.0
    mujoco.mj_step(m, d, nstep=10 * rec.substeps)
    mujoco.mj_forward(m, d)


def trial(
    rec: HumanoidRecorder, body: int, seed: int, explore: bool = True
) -> dict[str, object] | None:
    """One explored grasp of ``body``; its library record, or None if none was planned."""
    m, d = rec.model, rec.data
    rng = np.random.default_rng(seed)
    rec.props_rng = np.random.default_rng(seed + 1)  # fresh hidden properties
    policy = rec.make_policy("seated_babble", rng)
    assert isinstance(policy, BabblePolicy)
    rec.reset(policy, rng)
    side = "left" if rng.random() < 0.5 else "right"
    place_alone(rec, body, side, rng)
    other = "right" if side == "left" else "left"
    policy.queue[other] = [Waypoint(policy.cmd_q[other].copy(), policy.FAST, hold=10**6)]
    policy.explore = explore
    policy.clearance, policy.axis_weight = policy.GRASP_CLEARANCE, 0.3
    index = rec.objects.index(body)
    queue = policy.plan_oriented(side, d, policy.LIFT_HEIGHT, index)
    ok = bool(queue) and policy.plan_ok(d, side, queue)
    policy.clearance, policy.axis_weight = policy.CLEARANCE, 0.0
    policy.exempt[:] = False
    if not ok:
        return None
    record = dict(policy.last_grasp)
    policy.reaims_left[side] = policy.REAIMS
    policy.queue[side] = queue
    z0 = float(d.xpos[body][2])
    rise, touched = [], []
    for step in range(MAX_STEPS):
        mujoco.mj_forward(m, d)
        d.ctrl[:] = rec.with_gravity_compensation(rec.space.to_ctrl(policy.act(step, d)))
        mujoco.mj_step(m, d, nstep=rec.substeps)
        rise.append(float(d.xpos[body][2]) - z0)
        touched.append(
            any(
                body
                in (int(m.geom_bodyid[d.contact[i].geom1]), int(m.geom_bodyid[d.contact[i].geom2]))
                and rec.classes.effector[int(d.contact[i].geom1)]
                | rec.classes.effector[int(d.contact[i].geom2)]
                for i in range(d.ncon)
            )
        )
        if not policy.queue[side]:
            break
    up, contact = np.array(rise) >= LIFT_M, np.array(touched)
    lifted = any(contact[max(a - 2, 0) : b + 2].any() for a, b in runs(up, LIFT_STEPS))
    key = record["key"]
    assert isinstance(key, tuple)
    return {"object": key[0], "rest": key[1], "features": record["features"], "lifted": lifted}


def main() -> None:
    """Run the trials and append the records."""
    parser = argparse.ArgumentParser()
    parser.add_argument("effector", choices=("hand", "pincer"))
    parser.add_argument("object")
    parser.add_argument("trials", type=int)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/grasp_library/library.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/event_dense.toml"))
    parser.add_argument(
        "--planner",
        action="store_true",
        help="the planner's own grasps (no random IK starts or jitter) instead of explored ones",
    )
    args = parser.parse_args()
    blocks = tomllib.loads(args.config.read_text())["humanoid"]
    spec = next(s for name, s in blocks.items() if name.startswith(args.effector))
    settings = {k: v for k, v in dict(spec["settings"]).items() if k != "director"}
    HumanoidRecorder.render = lambda self, camera: np.zeros(  # type: ignore[method-assign]
        (IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8
    )
    rec = HumanoidRecorder(settings)
    body = rec.model.body(args.object).id
    args.out.parent.mkdir(parents=True, exist_ok=True)
    planned = lifted = 0
    with args.out.open("a") as handle:
        for k in range(args.first, args.first + args.trials):
            record = trial(rec, body, 1000 * k + 7, explore=not args.planner)
            if record is None:
                continue
            planned += 1
            lifted += bool(record["lifted"])
            handle.write(json.dumps({**record, "trial": k}) + "\n")
            handle.flush()
    print(
        f"{args.effector} {args.object}: {args.trials} trials, {planned} planned, {lifted} lifted"
    )


if __name__ == "__main__":
    main()
