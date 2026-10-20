"""Grasp success per object: one planned grasp at a time through the babbling planner.

Usage: uv run python scripts/hand_grasp_bench.py [trials] [out.json] [effector]
       (defaults: 20 trials per object and hand, runs/hand_grasp/bench.json, hand)

For every table object, ``trials`` times per arm: the seated scene is reset, the object is put
alone on the table at a random place on that arm's half (not under the arm's start, which the
first travel would sweep), with a random yaw; the other objects go to the floor. The
``BabblePolicy`` plans its grasp primitive (``power`` for hands, ``grasp`` for pincers) as
``plan`` does, and the movement runs to its end with the recorder's control loop (gravity
compensation, 25 physics steps per control step) while the other arm holds still. An attempt is a
planned grasp; it is a success if the object counts a lift (``babble_stats.count_lifts``: 5 cm
above rest for 0.5 s, touched). Prints per object the planned attempts, successes, the rate with
its Wilson 95% interval and the median and maximum effector-object force (N, the peak over
physics steps of the summed normal force), and writes every attempt to ``out.json``.
"""

import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

from sensorium.babble_stats import count_lifts
from sensorium.bodies.h1_babble import BabblePolicy, Waypoint
from sensorium.bodies.humanoid import HumanoidRecorder

START = {"left": np.array([0.5, 0.2]), "right": np.array([0.5, -0.2])}  # the tips' start (x, y)
MAX_STEPS = 400  # control steps an attempt may take


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of a proportion (95% by default); (0, 1) for n = 0."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def place(rec: HumanoidRecorder, body: int, side: str, rng: np.random.Generator) -> None:
    """``body`` alone on ``side``'s half of the table, the other objects on the floor."""
    m, d = rec.model, rec.data
    sign = 1.0 if side == "left" else -1.0
    for k, b in enumerate(rec.objects):
        adr = m.jnt_qposadr[m.body_jntadr[b]]
        if b != body:
            d.qpos[adr : adr + 7] = (-1.0, -1.0 + 0.3 * k, 0.1, 1, 0, 0, 0)
            continue
        while True:
            xy = np.array([rng.uniform(*rec.reach_x), sign * rng.uniform(0.08, 0.28)])
            if np.linalg.norm(xy - START[side]) > 0.12:
                break
        geom = int(m.body_geomadr[b])
        size, kind = m.geom_size[geom], m.geom_type[geom]
        half = size[2] if kind == mujoco.mjtGeom.mjGEOM_BOX else size[1]
        half = size[0] if kind == mujoco.mjtGeom.mjGEOM_SPHERE else half
        yaw = rng.uniform(0, 2 * np.pi)
        top = rec.table_top + half + 0.002
        d.qpos[adr : adr + 7] = (*xy, top, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
    d.qvel[:] = 0.0
    mujoco.mj_step(m, d, nstep=5 * rec.substeps)
    mujoco.mj_forward(m, d)


def plan_grasp(policy: BabblePolicy, side: str, data: mujoco.MjData) -> list[Waypoint] | None:
    """The grasp primitive's plan, with ``BabblePolicy.plan``'s settings and retries."""
    primitive = "grasp" if policy.pincer else "power"
    for _ in range(policy.TRIES):
        policy.clearance, policy.axis_weight = policy.GRASP_CLEARANCE, 0.3
        queue = getattr(policy, f"plan_{primitive}")(side, data)
        ok = bool(queue) and policy.plan_ok(data, side, queue)
        policy.clearance, policy.axis_weight = policy.CLEARANCE, 0.0
        policy.exempt[:] = False
        if ok:
            return queue
    return None


def effector_force(rec: HumanoidRecorder, body: int) -> float:
    """Summed normal force (N) of the effector contacts on ``body``."""
    m, d = rec.model, rec.data
    total, wrench = 0.0, np.zeros(6)
    for i in range(d.ncon):
        pair = (int(d.contact[i].geom1), int(d.contact[i].geom2))
        for g, other in (pair, pair[::-1]):
            if m.geom_bodyid[other] == body and rec.classes.effector[g]:
                mujoco.mj_contactForce(m, d, i, wrench)
                total += float(wrench[0])
    return total


def attempt(rec: HumanoidRecorder, index: int, side: str, seed: int) -> dict[str, object]:
    """One planned grasp of object ``index`` by ``side``; its outcome."""
    m, d = rec.model, rec.data
    rng = np.random.default_rng(seed)
    policy = rec.make_policy("seated_babble", rng)
    assert isinstance(policy, BabblePolicy)
    rec.reset(policy, rng)
    body = rec.objects[index]
    place(rec, body, side, rng)
    rest = float(d.xpos[body][2])
    other = "left" if side == "right" else "right"
    policy.queue[other] = [Waypoint(policy.cmd_q[other].copy(), policy.FAST, hold=10**6)]
    queue = plan_grasp(policy, side, d)
    if queue is None:
        return {"planned": False}
    policy.queue[side] = queue
    heights, touched, peak = [], [], 0.0
    for step in range(MAX_STEPS):
        mujoco.mj_forward(m, d)
        action = policy.act(step, d)
        d.ctrl[:] = rec.with_gravity_compensation(rec.space.to_ctrl(action))
        force = 0.0
        for _ in range(rec.substeps):
            mujoco.mj_step(m, d)
            force = max(force, effector_force(rec, body))
        peak = max(peak, force)
        heights.append(float(d.xpos[body][2]) - rest)
        touched.append(force > 0.0)
        if not policy.queue[side]:
            break
    lifted = count_lifts(np.array(heights), np.array(touched)) > 0
    return {"planned": True, "lift": bool(lifted), "rise": max(heights), "peak": peak}


def main() -> None:
    """Run the attempts, print the per-object table and write every attempt as JSON."""
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "runs/hand_grasp/bench.json")
    effector = sys.argv[3] if len(sys.argv) > 3 else "hand"
    out.parent.mkdir(parents=True, exist_ok=True)
    primitive = "grasp" if effector == "pincer" else "power"
    rec = HumanoidRecorder({"effector": effector, "posture": "seated", "primitives": [primitive]})
    results: list[dict[str, object]] = []
    print("| object | attempts (planned / tried) | lifts | rate [95% CI] | peak N, median / max |")
    print("|---|---|---|---|---|")
    for index, name in enumerate(rec.object_names):
        rows = []
        for side in ("left", "right"):
            for k in range(trials):
                seed = 7000 + 100 * index + k + (50 if side == "left" else 0)
                row = {"object": name, "side": side, "seed": seed}
                row |= attempt(rec, index, side, seed)
                rows.append(row)
        results += rows
        planned = [r for r in rows if r["planned"]]
        lifts = sum(bool(r["lift"]) for r in planned)
        low, high = wilson(lifts, len(planned))
        peaks = [float(r["peak"]) for r in planned] or [0.0]  # type: ignore[arg-type]
        rate = lifts / max(len(planned), 1)
        print(
            f"| {name[4:]} | {len(planned)} / {len(rows)} | {lifts} | {rate:.0%} "
            f"[{low:.0%}, {high:.0%}] | {np.median(peaks):.0f} / {max(peaks):.0f} |",
            flush=True,
        )
    out.write_text(json.dumps(results))


if __name__ == "__main__":
    main()
