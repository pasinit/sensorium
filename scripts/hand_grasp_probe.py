"""Can the Shadow hand hold a table object at all? A scripted grasp, no planner (grasp stage 1).

Usage: uv run python scripts/hand_grasp_probe.py [out_dir]   (default runs/hand_grasp)

For each of the six table objects, alone on the table at x = 0.50 m, 0.20 m to the right: the
right hand is put by inverse kinematics (the arm's five joints and the wrist's two, full pose)
4 cm above the claw pinch of ``h1_babble.HAND_PINCH`` (palm down over the object, tilted 30
degrees, fingers pointing 30 degrees inwards, thumb pre-placed under the palm), descends onto it
along the palm normal in 0.5 s, closes for 0.8 s,
waits 0.3 s, lifts 10 cm in 1 s with the arm and holds 1 s. Each object is run with the friction
cone of the scene (elliptic, ``impratio`` 10) and with MuJoCo's pyramidal default, the hand
scene's setting before the hand grasp work. Prints per run the highest rise, the steps at 5 cm
or more (a lift is 10, as in ``babble_stats``) and the peak hand-object force; writes
``grasp_height.png`` (object height against time) and ``grasp_forces.png`` (per-finger normal
force on the object, elliptic cone) to the output directory.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from sensorium.babble_stats import LIFT_M, LIFT_STEPS  # noqa: E402
from sensorium.bodies.h1_babble import HAND_PINCH, PINCH_DEPTH, PINCH_GAP, grasp_frame  # noqa: E402
from sensorium.bodies.humanoid import DT, HumanoidRecorder  # noqa: E402

SPOT = np.array([0.50, -0.20])  # m, where the object stands
YAW, TILT = np.radians(30.0), np.radians(30.0)  # fingers inwards; palm normal off vertical
DESCENT, CLOSE, WAIT, LIFT, HOLD = 10, 16, 6, 20, 20  # control steps of each phase
ABOVE = 0.04  # m along the palm normal the hand starts above the grasp, pre-shaped
PARTS = ("ff", "mf", "rf", "lf", "th", "palm")
PREFIX = "rh_"


class Probe:
    """The right hand of a seated hand scene, driven joint by joint."""

    def __init__(self) -> None:
        self.rec = HumanoidRecorder({"effector": "hand", "posture": "seated"})
        self.model, self.data = self.rec.model, self.rec.data
        self.policy = self.rec.make_policy("seated_babble", np.random.default_rng(0))
        m = self.model
        self.frame = grasp_frame(m, "hand", "right")
        self.palm = self.frame.body
        names = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_yaw"]
        joints = [m.joint(f"right_{n}").id for n in names]
        joints += [m.joint(f"{PREFIX}WRJ2").id, m.joint(f"{PREFIX}WRJ1").id]
        self.qpos_adr = np.array([m.jnt_qposadr[j] for j in joints])
        self.dof_adr = np.array([m.jnt_dofadr[j] for j in joints])
        self.low, self.high = m.jnt_range[joints, 0], m.jnt_range[joints, 1]
        by_joint = {
            int(m.actuator_trnid[a, 0]): a for a in range(m.nu) if m.actuator_trntype[a] == 0
        }
        self.actuators = [by_joint[j] for j in joints]
        self.part = {}
        for g in range(m.ngeom):
            body = m.body(m.geom_bodyid[g]).name
            if body.startswith(PREFIX):
                self.part[g] = body[3:5] if body[3:5] in PARTS else "palm"
        self.scratch = mujoco.MjData(m)

    def ik(self, point: np.ndarray, rotation: np.ndarray, offset: np.ndarray) -> np.ndarray:
        """7 joints putting ``palm + R offset`` at ``point`` with the palm at ``rotation``."""
        m, s = self.model, self.scratch
        best: tuple[float, np.ndarray] | None = None
        rng = np.random.default_rng(0)
        for attempt in range(8):
            q = self.data.qpos[self.qpos_adr].copy()
            if attempt:
                q = np.clip(q + rng.normal(0.0, 0.6, 7), self.low, self.high)
            jac_p, jac_r = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            for _ in range(400):
                s.qpos[:] = self.data.qpos
                s.qpos[self.qpos_adr] = q
                mujoco.mj_kinematics(m, s)
                mujoco.mj_comPos(m, s)
                now = s.xmat[self.palm].reshape(3, 3)
                p = s.xpos[self.palm] + now @ offset
                quat, turn = np.zeros(4), np.zeros(3)
                mujoco.mju_mat2Quat(quat, (rotation @ now.T).flatten())
                mujoco.mju_quat2Vel(turn, quat, 1.0)
                error = np.concatenate([point - p, 0.3 * turn])
                if np.linalg.norm(error[:3]) < 1e-3 and np.linalg.norm(turn) < 0.01:
                    break
                mujoco.mj_jac(m, s, jac_p, jac_r, p, self.palm)
                jac = np.vstack([jac_p[:, self.dof_adr], 0.3 * jac_r[:, self.dof_adr]])
                error *= min(1.0, 0.1 / max(float(np.linalg.norm(error)), 1e-9))
                step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), error)
                q = np.clip(q + step, self.low, self.high)
            miss = float(np.linalg.norm(error))
            if best is None or miss < best[0]:
                best = (miss, q)
        assert best is not None
        return best[1]

    def place(self, body: int) -> None:
        """Reset; ``body`` alone at ``SPOT`` (unrotated), the other objects on the floor."""
        rec, m, d = self.rec, self.model, self.data
        rec.reset(self.policy, np.random.default_rng(0))
        for k, b in enumerate(rec.objects):
            adr = m.jnt_qposadr[m.body_jntadr[b]]
            _, half_height, _ = self.policy.object_geometry(d, rec.objects.index(b))
            spot = (*SPOT, rec.table_top + half_height + 0.001) if b == body else (-1, k * 0.3, 0.1)
            d.qpos[adr : adr + 7] = (*spot, 1, 0, 0, 0)
        d.qvel[:] = 0.0
        mujoco.mj_step(m, d, nstep=10 * rec.substeps)

    def hand_targets(self, grip: float) -> dict[int, float]:
        """Pinch actuator -> ctrl at closure ``grip`` (0 pre-shape, 1 closed)."""
        return {
            self.model.actuator(f"{PREFIX}A_{name}").id: start + grip * (end - start)
            for name, (start, end) in HAND_PINCH.items()
            if not name.startswith("WR")  # the wrist is posed by the IK here
        }

    def set_hand(self, grip: float) -> None:
        """Put the hand's joints at the pinch targets (the J0 tendons split over J1 and J2)."""
        m, d = self.model, self.data
        for name, (start, end) in HAND_PINCH.items():
            if name.startswith("WR"):
                continue
            value = start + grip * (end - start)
            joints = (name[:2] + "J1", name[:2] + "J2") if name.endswith("J0") else (name,)
            for joint in joints:
                d.qpos[m.jnt_qposadr[m.joint(PREFIX + joint).id]] = value / len(joints)

    def forces(self, geom: int) -> dict[str, float]:
        """Normal force (N) on the object geom per hand part."""
        out, wrench = dict.fromkeys(PARTS, 0.0), np.zeros(6)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            pair = (int(contact.geom1), int(contact.geom2))
            if geom in pair:
                other = pair[1] if pair[0] == geom else pair[0]
                if other in self.part:
                    mujoco.mj_contactForce(self.model, self.data, i, wrench)
                    out[self.part[other]] += float(wrench[0])
        return out

    def run(self, name: str, elliptic: bool) -> dict[str, np.ndarray]:
        """One scripted grasp of object ``name``: heights (m above rest) and per-part peaks."""
        m, d, rec = self.model, self.data, self.rec
        m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC if elliptic else mujoco.mjtCone.mjCONE_PYRAMIDAL
        m.opt.impratio = 10.0 if elliptic else 1.0
        body = m.body(name).id
        geom = int(m.body_geomadr[body])
        self.place(body)
        fingers = np.array([np.cos(YAW), np.sin(YAW), 0.0])
        normal = np.array([0, 0, -np.cos(TILT)]) + np.cross([0, 0, 1.0], fingers) * np.sin(TILT)
        world = np.stack([fingers, normal, np.cross(normal, fingers)], axis=1)
        f = self.frame
        local = np.stack([f.approach, f.axis, np.cross(f.axis, f.approach)], axis=1)
        rotation = world @ local.T
        size = m.geom_size[geom]
        box = m.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX
        half = abs(np.cos(YAW)) * size[0] + abs(np.sin(YAW)) * size[1] if box else size[0]
        centre = np.array(d.xpos[body])
        goal = centre - half * fingers  # the frame point is the object's near face
        start = goal - ABOVE * normal
        down = [
            self.ik(start + (goal - start) * k / DESCENT, rotation, f.offset)
            for k in range(1, DESCENT + 1)
        ]
        path = [
            self.ik(goal + [0, 0, 0.1 * k / LIFT], rotation, f.offset) for k in range(1, LIFT + 1)
        ]
        d.qpos[self.qpos_adr] = self.ik(start, rotation, f.offset)
        self.set_hand(0.0)
        d.qvel[:] = 0.0
        ctrl = d.ctrl.copy()
        z_rest = float(d.xpos[body][2])
        heights, peaks = [], []
        arm = down[0]
        for step in range(DESCENT + CLOSE + WAIT + LIFT + HOLD):
            if step < DESCENT:
                arm = down[step]
            elif DESCENT + CLOSE + WAIT <= step < DESCENT + CLOSE + WAIT + LIFT:
                arm = path[step - DESCENT - CLOSE - WAIT]
            grip = min(1.0, max(0, step + 1 - DESCENT) / CLOSE)
            for a, value in self.hand_targets(grip).items():
                ctrl[a] = value
            for k, a in enumerate(self.actuators):
                ctrl[a] = arm[k]
            mujoco.mj_forward(m, d)
            d.ctrl[:] = ctrl
            for k, a in enumerate(self.actuators[:5]):  # gravity compensation, as recorded
                d.ctrl[a] += d.qfrc_bias[self.dof_adr[k]] / m.actuator_gainprm[a, 0]
            peak = dict.fromkeys(PARTS, 0.0)
            for _ in range(rec.substeps):
                mujoco.mj_step(m, d)
                for part, value in self.forces(geom).items():
                    peak[part] = max(peak[part], value)
            heights.append(float(d.xpos[body][2]) - z_rest)
            peaks.append([peak[p] for p in PARTS])
        return {"height": np.array(heights), "force": np.array(peaks)}


def held_steps(height: np.ndarray) -> int:
    """Longest run of steps at ``LIFT_M`` or more above rest."""
    best = run = 0
    for up in height >= LIFT_M:
        run = run + 1 if up else 0
        best = max(best, run)
    return best


Runs = dict[tuple[str, bool], dict[str, np.ndarray]]


def run_all(probe: Probe) -> Runs:
    """Every object with both cones; prints a markdown table row per run."""
    print(f"claw pinch: gap {PINCH_GAP} m, depth {PINCH_DEPTH} m; a lift is {LIFT_STEPS} steps")
    print("| object | cone | rise max (cm) | steps >= 5 cm | lift | peak hand-object (N) |")
    print("|---|---|---|---|---|---|")
    runs: Runs = {}
    for name in probe.rec.object_names:
        for elliptic in (True, False):
            result = probe.run(name, elliptic)
            runs[name, elliptic] = result
            held = held_steps(result["height"])
            lifted = "yes" if held >= LIFT_STEPS else "no"
            print(
                f"| {name[4:]} | {'elliptic' if elliptic else 'pyramidal'} "
                f"| {result['height'].max() * 100:.1f} | {held} | {lifted} "
                f"| {result['force'].sum(axis=1).max():.0f} |",
                flush=True,
            )
    return runs


def plot_heights(runs: Runs, names: list[str], path: Path) -> None:
    """Object height against time, both cones, one panel per object."""
    t = np.arange(DESCENT + CLOSE + WAIT + LIFT + HOLD) * DT
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, names, strict=True):
        for elliptic, style in ((True, "-"), (False, "--")):
            label = "elliptic" if elliptic else "pyramidal"
            ax.plot(t, runs[name, elliptic]["height"] * 100, style, label=label)
        ax.axhline(LIFT_M * 100, color="grey", lw=0.8)
        for edge in np.cumsum([DESCENT, CLOSE, WAIT, LIFT]):
            ax.axvline(edge * DT, color="grey", lw=0.5, ls=":")
        ax.set_title(name[4:])
    axes[0, 0].legend()
    for ax in axes[:, 0]:
        ax.set_ylabel("object height above rest (cm)")
    for ax in axes[1]:
        ax.set_xlabel("time (s): descend | close | wait | lift | hold")
    fig.suptitle("Scripted claw pinch, one object at a time (5 cm line: a lift)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def plot_forces(runs: Runs, names: list[str], path: Path) -> None:
    """Per-finger normal force on the object against time, elliptic cone."""
    t = np.arange(DESCENT + CLOSE + WAIT + LIFT + HOLD) * DT
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for ax, name in zip(axes.flat, names, strict=True):
        force = runs[name, True]["force"]
        for k, part in enumerate(PARTS):
            if force[:, k].max() > 0.5:
                ax.plot(t, force[:, k], label=part)
        ax.set_title(name[4:])
        ax.legend(fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("normal force on object (N)")
    for ax in axes[1]:
        ax.set_xlabel("time (s)")
    fig.suptitle("Per-finger force, scripted claw pinch, elliptic cone (peak per 50 ms step)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main() -> None:
    """Run every object with both cones, print the table and write the two figures."""
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/hand_grasp")
    out.mkdir(parents=True, exist_ok=True)
    probe = Probe()
    runs = run_all(probe)
    names = probe.rec.object_names
    plot_heights(runs, names, out / "grasp_height.png")
    plot_forces(runs, names, out / "grasp_forces.png")
    print(f"wrote {out / 'grasp_height.png'} and {out / 'grasp_forces.png'}")


if __name__ == "__main__":
    main()
