"""Do elbow-low (less occluding) arm postures exist for the babbling's hover goals?

Usage: python scripts/elbow_swivel.py <effector> <table_top> [goals per side] [restarts] [grasp]

With ``grasp`` the goals are grasp points at an object's height and the candidates are the
oriented grasp postures (``OrientedIK``, near-vertical approach, the grasp clearance rules).

The arm has one redundant degree of freedom (the elbow's swivel). For random hover goals
over the table this collects every clear posture the IK finds in ``restarts`` starts, in the
order the policy tries them (``BabblePolicy.solve_valid``: the hover start, jittered starts,
random starts), and compares the policy's pick (the first clear one) with the clear postures
that have the lowest elbow and the least image occlusion (segmentation render of the head
camera, the arm's links only). Measure only: nothing here changes the policy.
"""

import sys

import mujoco
import numpy as np

from sensorium.bodies.h1_babble import BabblePolicy
from sensorium.bodies.h1_forces import LEFT_ARM, RIGHT_ARM
from sensorium.bodies.h1_policies import SIDES
from sensorium.bodies.humanoid import DEFAULT_CAMERA, HumanoidRecorder
from sensorium.episode import IMAGE_SIZE


def candidates(
    policy: BabblePolicy, data: mujoco.MjData, side: str, goal: np.ndarray, restarts: int
) -> list[np.ndarray]:
    """Clear postures for ``goal`` in the order ``solve_valid`` would try the starts."""
    ik = policy.ik[side]
    start = policy.hover_q[side]
    found = []
    for attempt in range(restarts):
        seed = start[:4]
        if attempt >= 3:
            seed = policy.rng.uniform(ik.low, ik.high)
        elif attempt:
            seed = np.clip(seed + policy.rng.normal(0.0, 0.5, 4), ik.low, ik.high)
        q = policy.with_wrist(side, ik.solve(data.qpos, seed, goal, 60))
        if policy.posture_ok(data, side, q, goal):
            found.append(q)
    return found


def grasp_candidates(
    policy: BabblePolicy, data: mujoco.MjData, side: str, goal: np.ndarray, restarts: int
) -> list[np.ndarray]:
    """Clear oriented grasp postures for ``goal`` (the starts of ``oriented_pose``, in order)."""
    oriented = policy.oriented[side]
    policy.clearance, policy.axis_weight = policy.GRASP_CLEARANCE, 0.3
    found = []
    for attempt in range(restarts):
        starts = [policy.cmd_q[side], policy.hover_q[side]]
        start = starts[attempt] if attempt < 2 else policy.rng.uniform(oriented.low, oriented.high)
        z = float(policy.rng.choice((-0.8, -0.9)))
        q, pos_error, ori_error = oriented.solve(data.qpos, start, goal, z, policy.axis_weight)
        if pos_error < 0.004 and ori_error < 0.03 and policy.posture_ok(data, side, q):
            found.append(q)
    policy.clearance, policy.axis_weight = policy.CLEARANCE, 0.0
    return found


def main() -> None:
    """Print the comparison."""
    effector, top = sys.argv[1], float(sys.argv[2])
    goals = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    restarts = int(sys.argv[4]) if len(sys.argv) > 4 else 24
    grasp = len(sys.argv) > 5 and sys.argv[5] == "grasp"
    recorder = HumanoidRecorder({"effector": effector, "posture": "seated", "table_top": top})
    policy = recorder.make_policy("seated_babble", np.random.default_rng(0))
    assert isinstance(policy, BabblePolicy)
    recorder.reset(policy, np.random.default_rng(0))
    data, model = recorder.data, recorder.model
    renderer = mujoco.Renderer(model, IMAGE_SIZE, IMAGE_SIZE)
    renderer.enable_segmentation_rendering()
    camera = model.camera(DEFAULT_CAMERA).id
    classes = recorder.classes
    scratch = mujoco.MjData(model)

    def occlusion(side: str, q: np.ndarray) -> tuple[float, float]:
        """Image share covered by the arm's links, and the elbow height, with the arm at ``q``."""
        scratch.qpos[:] = data.qpos
        scratch.qpos[policy.ik[side].qpos_adr5] = q
        mujoco.mj_forward(model, scratch)
        renderer.update_scene(scratch, camera=camera)
        seg = renderer.render()
        group = LEFT_ARM if side == "left" else RIGHT_ARM
        links = classes.robot & (classes.group == group) & ~classes.effector
        ids = np.where(seg[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[:, :, 0], 0)
        covered = links[ids] & (seg[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
        return float(covered.mean()), float(scratch.xpos[model.body(f"{side}_elbow_link").id][2])

    rows = []
    for side in SIDES:
        for _ in range(goals):
            if grasp:
                goal = policy.table_point(side) + np.array([0.0, 0.0, 0.035])
                clear = grasp_candidates(policy, data, side, goal, restarts)
            else:
                goal = policy.table_point(side) + np.array([0.0, 0.0, policy.HOVER])
                clear = candidates(policy, data, side, goal, restarts)
            if len(clear) < 2:
                continue
            measured = np.array([occlusion(side, q) for q in clear])
            lowest = int(measured[:, 1].argmin())
            first, least = measured[0, 0], measured[:, 0].min()
            rows.append(
                (len(clear), first, least, measured[0, 1], measured[lowest, 1], measured[lowest, 0])
            )
    r = np.array(rows)
    pick, best, low_elbow = r[:, 1], r[:, 2], r[:, 5]
    sys.stdout.write(
        f"{effector}, table {top} m: {len(r)} hover goals with >= 2 clear postures among "
        f"{restarts} starts{' (grasp postures)' if grasp else ''} "
        f"(median {np.median(r[:, 0]):.0f} clear per goal)\n"
        f"  image share covered by the arm links: policy's pick {np.median(pick):.1%} "
        f"(mean {pick.mean():.1%}), least-occluding clear posture {np.median(best):.1%}, "
        f"lowest-elbow posture {np.median(low_elbow):.1%}\n"
        f"  elbow height: policy's pick {np.median(r[:, 3]):.2f} m, "
        f"lowest clear {np.median(r[:, 4]):.2f} m\n"
        f"  goals where the policy's pick covers > 15%: {np.mean(pick > 0.15):.0%}; "
        f"where a clear posture covers < 5%: {np.mean(best < 0.05):.0%}\n"
    )


if __name__ == "__main__":
    main()
