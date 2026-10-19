"""How much of the head camera's image the robot's own arms and hands cover, per primitive.

Usage:
    python scripts/arm_occlusion.py run <effector> <table_top> <out.json> <seed>...
    python scripts/arm_occlusion.py report <out.json>...

``run`` records seated babbling episodes (the default policy) and, at every step, renders the
head camera with segmentation, counts the pixels covered by each arm's links and by its
effector, projects the tips (are they in the image?), and notes which primitive each arm is
executing. ``report`` prints the table per primitive. Nothing here changes the policy or the
recorded data: the policy's ``plan`` is wrapped to learn which primitive it chose.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np

from sensorium.bodies.h1_babble import BabblePolicy
from sensorium.bodies.h1_camera import outside_image
from sensorium.bodies.h1_forces import LEFT_ARM, RIGHT_ARM
from sensorium.bodies.h1_policies import SIDES
from sensorium.bodies.humanoid import DEFAULT_CAMERA, HumanoidRecorder
from sensorium.episode import IMAGE_SIZE


def geom_masks(recorder: HumanoidRecorder) -> dict[str, np.ndarray]:
    """Boolean per-geom masks: each arm's links and each arm's effector."""
    classes = recorder.classes
    masks = {}
    for side, group in zip(SIDES, (LEFT_ARM, RIGHT_ARM), strict=True):
        arm = classes.robot & (classes.group == group)
        masks[f"{side}_links"] = arm & ~classes.effector
        masks[f"{side}_effector"] = arm & classes.effector
    return masks


def run(effector: str, table_top: float, out: Path, seeds: list[int]) -> None:
    """Record the episodes and write one row per step."""
    recorder = HumanoidRecorder({"effector": effector, "posture": "seated", "table_top": table_top})
    model = recorder.model
    segmenter = mujoco.Renderer(model, IMAGE_SIZE, IMAGE_SIZE)
    segmenter.enable_segmentation_rendering()
    camera = model.camera(DEFAULT_CAMERA).id
    masks = geom_masks(recorder)
    current = dict.fromkeys(SIDES, "start")
    state = {"seed": 0}
    rows: list[dict[str, object]] = []
    plan = BabblePolicy.plan
    observe = recorder.observe

    def wrapped_plan(self: BabblePolicy, side: str, data: mujoco.MjData) -> None:
        before = dict(self.primitive_counts)
        plan(self, side, data)
        chosen = [name for name, count in self.primitive_counts.items() if count > before[name]]
        current[side] = chosen[0] if chosen else "wait"

    def wrapped_observe(*args: object, **kwargs: object) -> None:
        observe(*args, **kwargs)  # type: ignore[arg-type]
        data = recorder.data
        segmenter.update_scene(data, camera=camera)
        seg = segmenter.render()
        ids = np.where(seg[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[:, :, 0], -1)
        row: dict[str, object] = {"seed": state["seed"], "step": len(rows)}
        position, rotation = data.cam_xpos[camera], data.cam_xmat[camera].reshape(3, 3)
        for side in SIDES:
            tip = recorder.tips[side].position(data)[None]
            row[f"{side}_primitive"] = current[side]
            row[f"{side}_tip_outside"] = bool(outside_image(tip, position, rotation, 80.0)[0])
            elbow = recorder.model.body(f"{side}_elbow_link").id
            row[f"{side}_elbow_z"] = round(float(data.xpos[elbow][2]), 3)
            for kind in ("links", "effector"):
                inside = masks[f"{side}_{kind}"][np.clip(ids, 0, None)] & (ids >= 0)
                row[f"{side}_{kind}"] = round(float(inside.mean()), 4)
        rows.append(row)

    BabblePolicy.plan = wrapped_plan  # type: ignore[method-assign]
    recorder.observe = wrapped_observe  # type: ignore[method-assign]
    for seed in seeds:
        state["seed"] = seed
        current.update(dict.fromkeys(SIDES, "start"))
        recorder.record(seed, "seated_babble", 600)
    out.write_text(json.dumps({"effector": effector, "table_top": table_top, "rows": rows}))


def report(paths: list[Path]) -> None:
    """Per cell and primitive: image share covered by the arm's links and effector, tips outside."""
    for path in paths:
        data = json.loads(path.read_text())
        groups: dict[str, list[tuple[float, float, bool, float]]] = defaultdict(list)
        for row in data["rows"]:
            for side in SIDES:
                groups[row[f"{side}_primitive"]].append(
                    (
                        row[f"{side}_links"],
                        row[f"{side}_effector"],
                        row[f"{side}_tip_outside"],
                        row[f"{side}_elbow_z"],
                    )
                )
        every = [v for values in groups.values() for v in values]
        both = [
            (r["left_links"] + r["right_links"], r["left_effector"] + r["right_effector"])
            for r in data["rows"]
        ]
        links = np.array([a for a, _ in both])
        effectors = np.array([b for _, b in both])
        sys.stdout.write(
            f"\n{data['effector']} table {data['table_top']} m, {len(data['rows'])} steps: "
            f"both arms cover {links.mean():.0%} of the image with links "
            f"(p90 {np.percentile(links, 90):.0%}) and {effectors.mean():.1%} with effectors; "
            "steps with the links > 25%: "
            f"{(links > 0.25).mean():.0%}\n"
        )
        sys.stdout.write("  primitive     steps   links    effector  tip outside  elbow z (m)\n")
        for name, values in sorted(groups.items(), key=lambda kv: -len(kv[1])) + [("all", every)]:
            v = np.array(values, dtype=float)
            sys.stdout.write(
                f"  {name:<12} {len(v):>6} {v[:, 0].mean():>7.1%} {v[:, 1].mean():>9.1%} "
                f"{v[:, 2].mean():>11.0%} {v[:, 3].mean():>11.2f}\n"
            )


def main() -> None:
    """Dispatch ``run`` or ``report``."""
    if len(sys.argv) >= 6 and sys.argv[1] == "run":
        run(sys.argv[2], float(sys.argv[3]), Path(sys.argv[4]), [int(s) for s in sys.argv[5:]])
    elif len(sys.argv) >= 3 and sys.argv[1] == "report":
        report([Path(p) for p in sys.argv[2:]])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
