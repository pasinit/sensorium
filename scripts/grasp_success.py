"""The robot's own grasp success on the event-dense table, and why grasps fail.

Usage:
    uv run python scripts/grasp_success.py <hand|pincer> <episodes> [--first N] [--out FILE]
        [--config configs/event_dense.toml] [--set module.ATTR=json ...] [--render]

Runs the first block of ``--config`` for the effector (its settings: props, scene director,
hidden properties), without rendering unless ``--render``, and follows every grasp the babbling
planner queues (``power`` for hands, ``grasp`` for pincers) from its plan to the arm's next plan.
``--set`` overrides a module constant or class attribute for an ablation, e.g.
``sensorium.bodies.h1_babble.PINCH_GAP=0.07`` or
``sensorium.bodies.h1_babble.BabblePolicy.MOVED=0.03``.

A grasp is a **success** (an own lift) when its object stays ``LIFT_M`` above its height at the
plan for ``LIFT_STEPS`` steps, touched by an effector in that run (as
``babble_stats.count_lifts``). Otherwise one cause, checked in this order:

* ``replan``: dropped before the descent because the object moved (``BabblePolicy.MOVED``);
* ``reflex``: a limb contact withdrew the arm;
* ``director``: the scene director dropped or re-placed this very object during the attempt;
* ``no pinch``: in the first 15 steps at full closure the object is never held between a finger
  and the thumb (hands) or both pads (pincers);
* ``metal box``: the 1.17 kg box held but not lifted;
* ``slip``: held and rising at least 1 cm, but not for the lift;
* ``no rise``: held, under 1 cm.

Attempts cut by the episode's end are left out. Prints per effector the own lifts per episode,
the success per attempt with its Wilson 95% interval, success per executed grasp (replans and
reflexes left out), the cause table, and writes every attempt to ``--out`` as JSON.
"""

import argparse
import importlib
import json
import math
import tomllib
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from sensorium.babble_stats import runs
from sensorium.bodies.h1_babble import BabblePolicy
from sensorium.bodies.humanoid import IMAGE_SIZE, HumanoidRecorder

LIFT_M = 0.03  # m over the object's height at the plan: an own lift (the supervisor's 3 cm)
LIFT_STEPS = 10  # control steps (0.5 s) held there
STEPS = 600
FINGERS = ("ff", "mf", "rf", "lf")


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of a proportion; (0, 1) for n = 0."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def apply_overrides(items: list[str]) -> None:
    """Set ``module.ATTR=json`` or ``module.Class.ATTR=json`` for an ablation."""
    for item in items:
        target, value = item.split("=", 1)
        parts = target.split(".")
        for split in range(len(parts) - 1, 0, -1):
            try:
                owner: object = importlib.import_module(".".join(parts[:split]))
            except ModuleNotFoundError:
                continue
            for name in parts[split:-1]:
                owner = getattr(owner, name)
            setattr(owner, parts[-1], json.loads(value))
            break
        else:
            raise ValueError(f"cannot resolve {target}")


class Tracker:
    """Follows the grasps of one ``BabblePolicy`` (wraps its ``plan`` and ``advance``)."""

    def __init__(self, rec: HumanoidRecorder, policy: BabblePolicy, primitive: str) -> None:
        self.rec, self.policy, self.primitive = rec, policy, primitive
        self.open: dict[str, dict[str, object]] = {}
        self.done: list[dict[str, object]] = []
        self.replans = 0
        plan, advance, pick = policy.plan, policy.advance, policy.pick_object
        self.picked: int | None = None

        def tracked_pick(side: str, data: mujoco.MjData, *options: object) -> int | None:
            self.picked = pick(side, data, *options)  # type: ignore[arg-type]  # the last pick
            return self.picked

        def tracked_plan(side: str, data: mujoco.MjData) -> None:
            count = policy.primitive_counts[primitive]
            plan(side, data)
            if policy.primitive_counts[primitive] > count and self.picked is not None:
                body = policy.objects[self.picked]
                self.close(side, "replan" if policy.replans > self.replans else "done")
                self.open[side] = {
                    "side": side,
                    "object": rec.model.body(body).name,
                    "body": body,
                    "queue": policy.queue[side],
                    "z0": float(data.xpos[body][2]),
                    "xy0": np.array(data.xpos[body][:2]),
                    "rows": [],
                }

        def tracked_advance(side: str, data: mujoco.MjData, force: float) -> tuple[bool, float]:
            attempt, reflexes = self.open.get(side), policy.reflexes
            self.replans, reaims = policy.replans, getattr(policy, "reaims", 0)
            result = advance(side, data, force)
            if attempt is not None and self.open.get(side) is attempt:
                if getattr(policy, "reaims", 0) > reaims:  # the same grasp, aimed again
                    attempt["queue"] = policy.queue[side]
                elif policy.replans > self.replans:
                    self.close(side, "replan")
                elif policy.reflexes > reflexes:
                    self.close(side, "reflex")
                elif policy.queue[side] is not attempt["queue"]:
                    self.close(side, "done")
            return result

        policy.plan = tracked_plan  # type: ignore[method-assign]
        policy.pick_object = tracked_pick  # type: ignore[method-assign]
        policy.advance = tracked_advance  # type: ignore[method-assign]

    def close(self, side: str, end: str) -> None:
        """End the open attempt of ``side``, if any."""
        attempt = self.open.pop(side, None)
        if attempt is not None:
            attempt["end"] = end
            self.done.append(attempt)

    def record_step(self, data: mujoco.MjData, moved: int | None) -> None:
        """Log each open attempt's object rise, closure, contacts and the director's moves."""
        for side, attempt in self.open.items():
            body = int(attempt["body"])  # type: ignore[call-overload]
            rows = attempt["rows"]
            assert isinstance(rows, list)
            if "at_close" not in attempt and self.policy.grip[side] >= 0.99:
                attempt["at_close"] = self.geometry(data, side, body, attempt)
            rows.append(
                {
                    "rise": float(data.xpos[body][2]) - float(attempt["z0"]),  # type: ignore[arg-type]
                    "grip": self.policy.grip[side],
                    "parts": self.parts(data, side, body),
                    "intervened": moved == body,
                }
            )

    def geometry(
        self, data: mujoco.MjData, side: str, body: int, attempt: dict[str, object]
    ) -> dict[str, object]:
        """Where the object is at closure: in the grasp frame, moved since the plan, tilted."""
        policy, frame = self.policy, self.policy.frames[side]
        rotation = data.xmat[frame.body].reshape(3, 3)
        fingers, normal = rotation @ frame.approach, rotation @ frame.axis
        across = np.cross(normal, fingers)
        rel = data.xpos[body] - (data.xpos[frame.body] + rotation @ frame.offset)
        index = policy.objects.index(body)
        if not policy.pincer:  # the hand's frame point is the object's near face
            rel = rel - policy.half_extent(data, index, fingers) * fingers
        actual = data.qpos[policy.ik[side].qpos_adr5]
        tip_lag = policy.ik[side].tip_at(data.qpos, policy.cmd_q[side]) - policy.ik[side].tip_at(
            data.qpos, actual
        )
        xy0 = attempt["xy0"]
        assert isinstance(xy0, np.ndarray)
        return {
            "rel": [round(float(rel @ v), 3) for v in (fingers, normal, across)],
            "moved": round(float(np.linalg.norm(data.xpos[body][:2] - xy0)), 3),
            "tilt": round(float(np.degrees(np.arccos(np.clip(data.xmat[body][8], -1, 1)))), 0),
            "tip_lag": round(float(np.linalg.norm(tip_lag)), 3),
            # |cos| between the closing axis (pincer) / palm normal (hand) and the object's x axis
            "axis_cos_x": round(abs(float(normal @ data.xmat[body].reshape(3, 3)[:, 0])), 2),
        }

    def parts(self, data: mujoco.MjData, side: str, body: int) -> list[str]:
        """The effector parts of ``side`` touching ``body`` (fingers, thumb, palm or pads)."""
        m, found = self.rec.model, set()
        prefix = {"left": ("lh_", "lp_"), "right": ("rh_", "rp_")}[side]
        for i in range(data.ncon):
            pair = (int(data.contact[i].geom1), int(data.contact[i].geom2))
            for g, other in (pair, pair[::-1]):
                name = m.body(m.geom_bodyid[g]).name
                if m.geom_bodyid[other] == body and name.startswith(prefix):
                    part = name[3:]
                    found.add(part[:2] if part[:2] in (*FINGERS, "th") else part)
        return sorted(found)


def held(parts: list[str], pincer: bool) -> bool:
    """True if the contacts hold an object: both pads, or a finger and the thumb."""
    if pincer:
        return "left_pad" in parts and "right_pad" in parts
    return any(p in FINGERS for p in parts) and "th" in parts


def cause(attempt: dict[str, object], pincer: bool) -> str:
    """The attempt's outcome (see the module docstring)."""
    rows = attempt["rows"]
    assert isinstance(rows, list)
    rise = np.array([r["rise"] for r in rows] or [0.0])
    touched = np.array([bool(r["parts"]) for r in rows] or [False])
    for start, stop in runs(rise >= LIFT_M, LIFT_STEPS):
        if touched[max(start - 2, 0) : stop + 2].any():  # as babble_stats.count_lifts
            return "success"
    if attempt["end"] in ("replan", "reflex"):
        return str(attempt["end"])
    if any(r["intervened"] for r in rows):
        return "director"
    closing = [r for r in rows if r["grip"] >= 0.99][:15]  # the closed hold before the lift
    if not any(held(r["parts"], pincer) for r in closing):
        return "no pinch"
    if attempt["object"] == "obj_metal_box":
        return "metal box"
    return "slip" if rise.max() >= 0.01 else "no rise"


def run(
    effector: str,
    episodes: int,
    first: int,
    config: Path,
    render: bool,
    extra: dict[str, object] | None = None,
) -> list[dict]:
    """Classified grasp attempts of ``episodes`` episodes of the config's first block."""
    blocks = tomllib.loads(config.read_text())["humanoid"]
    spec = next(s for name, s in blocks.items() if name.startswith(effector))
    if not render:
        HumanoidRecorder.render = lambda self, camera: np.zeros(  # type: ignore[method-assign]
            (IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8
        )
    rec = HumanoidRecorder(dict(spec["settings"]) | (extra or {}))
    pincer, primitive = effector == "pincer", "grasp" if effector == "pincer" else "power"
    m, d = rec.model, rec.data
    attempts: list[dict] = []
    for seed in range(first, first + episodes):
        rng = np.random.default_rng(seed)
        policy = rec.make_policy("seated_babble", rng)
        assert isinstance(policy, BabblePolicy)
        rec.director_rng = np.random.default_rng(seed + 10_000)
        rec.reset(policy, rng)
        tracker = Tracker(rec, policy, primitive)
        for step in range(STEPS):
            moved = None
            if rec.director is not None and rec.director.step(d, step):
                assert rec.director.moved is not None
                moved = rec.objects[rec.director.moved]
            mujoco.mj_forward(m, d)
            d.ctrl[:] = rec.with_gravity_compensation(rec.space.to_ctrl(policy.act(step, d)))
            mujoco.mj_step(m, d, nstep=rec.substeps)
            tracker.record_step(d, moved)
        for attempt in tracker.done:
            attempts.append(
                {"effector": effector, "episode": seed, "object": attempt["object"]}
                | {"end": attempt["end"], "cause": cause(attempt, pincer)}
                | {"at_close": attempt.get("at_close")}
            )
        print(f"{effector} episode {seed}: {len(tracker.done)} grasps", flush=True)
    return attempts


def report(attempts: list[dict], episodes: int) -> None:
    """Print the success and cause tables."""
    counts, n = Counter(a["cause"] for a in attempts), len(attempts)
    ok = counts["success"]
    executed = n - counts["replan"] - counts["reflex"]
    low, high = wilson(ok, n)
    elow, ehigh = wilson(ok, executed)
    print(f"own lifts per episode {ok / episodes:.2f}; grasps per episode {n / episodes:.1f}")
    print(f"success per attempt {ok}/{n} = {ok / max(n, 1):.0%} [{low:.0%}, {high:.0%}]")
    rate = ok / max(executed, 1)
    print(f"success per executed grasp {ok}/{executed} = {rate:.0%} [{elow:.0%}, {ehigh:.0%}]")
    print("| cause | grasps | share [Wilson 95%] |\n|---|---|---|")
    for name, k in counts.most_common():
        lo, hi = wilson(k, n)
        print(f"| {name} | {k} | {k / n:.0%} [{lo:.0%}, {hi:.0%}] |")


def main() -> None:
    """Parse the arguments, run, report and save."""
    parser = argparse.ArgumentParser()
    parser.add_argument("effector", choices=("hand", "pincer"))
    parser.add_argument("episodes", type=int)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/grasp/attempts.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/event_dense.toml"))
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--setting",
        action="append",
        default=[],
        help="recorder setting key=json, e.g. grasp_library",
    )
    args = parser.parse_args()
    apply_overrides(args.overrides)
    extra = {k: json.loads(v) for k, v in (item.split("=", 1) for item in args.setting)}
    attempts = run(args.effector, args.episodes, args.first, args.config, args.render, extra)
    report(attempts, args.episodes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(attempts))


if __name__ == "__main__":
    main()
