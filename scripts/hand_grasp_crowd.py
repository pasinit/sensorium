"""Why hand grasps fail on the crowded babbling table: every power attempt, classified by cause.

Usage: uv run python scripts/hand_grasp_crowd.py <episodes> [first seed] [out.json]
       (defaults: first seed 0, runs/hand_grasp/crowd.json; 600 steps per episode)

Runs the seated hand babbling with its full repertoire and all six objects, without rendering
(the recorder's control loop otherwise: gravity compensation, 25 physics steps per control step),
and follows every ``power`` grasp the planner queues, from the plan to the next plan of that arm.
An attempt ends in one of these causes, checked in this order:

* ``success``: the object counts a lift (``babble_stats.count_lifts``);
* ``replan``: the object moved more than ``MOVED`` between the plan and the descent, so the grasp
  was dropped (``own hand`` if the grasping hand touched it before the descent);
* ``reflex``: a limb contact above ``REFLEX_N`` withdrew the arm;
* ``no pinch``: at full closure the fingers and the thumb were not both on the object;
* ``metal box``: pinched, but the 1.17 kg box did not come up or slipped;
* ``no rise`` / ``slipped``: pinched; the object rose under 1 cm / rose but fell short of a lift.

Attempts cut by the episode's end are left out. Prints the table with Wilson 95% intervals, lifts
per episode and the planner's dropped power plans per episode, and writes every attempt as JSON.
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from sensorium.babble_stats import count_lifts
from sensorium.bodies.h1_babble import BabblePolicy
from sensorium.bodies.humanoid import HumanoidRecorder

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


class Tracker:
    """Follows the power attempts of one ``BabblePolicy`` (wraps its ``plan`` and ``advance``)."""

    def __init__(self, rec: HumanoidRecorder, policy: BabblePolicy) -> None:
        self.rec, self.policy = rec, policy
        self.open: dict[str, dict[str, object]] = {}
        self.done: list[dict[str, object]] = []
        self.dropped = 0
        self.replans = 0  # the policy's replans when the current advance began
        plan, advance = policy.plan, policy.advance

        def tracked_plan(side: str, data: mujoco.MjData) -> None:
            before = policy.object_counts.copy()
            count, failures = policy.primitive_counts["power"], policy.failures["power"]
            plan(side, data)
            self.dropped += policy.failures["power"] - failures
            if policy.primitive_counts["power"] > count:
                picked = np.flatnonzero(policy.object_counts != before)
                body = policy.objects[int(picked[-1])]
                # a replan empties the queue and plans again within the same advance
                self.close(side, "replan" if policy.replans > self.replans else "done")
                self.open[side] = {
                    "side": side,
                    "object": rec.model.body(body).name,
                    "body": body,
                    "queue": policy.queue[side],
                    "start": np.array(data.xpos[body]),
                    "rows": [],
                }

        def tracked_advance(side: str, data: mujoco.MjData, force: float) -> tuple[bool, float]:
            attempt, replans, reflexes = self.open.get(side), policy.replans, policy.reflexes
            self.replans = replans
            result = advance(side, data, force)
            if attempt is not None and self.open.get(side) is attempt:
                if policy.replans > replans:
                    self.close(side, "replan")
                elif policy.reflexes > reflexes:
                    self.close(side, "reflex")
                elif policy.queue[side] is not attempt["queue"]:
                    self.close(side, "done")
            return result

        policy.plan = tracked_plan  # type: ignore[method-assign]
        policy.advance = tracked_advance  # type: ignore[method-assign]

    def close(self, side: str, end: str) -> None:
        """End the open attempt of ``side``, if any."""
        attempt = self.open.pop(side, None)
        if attempt is not None:
            attempt["end"] = end
            self.done.append(attempt)

    def record_step(self, data: mujoco.MjData) -> None:
        """Log each open attempt's object height, displacement, closure and hand contacts."""
        for side, attempt in self.open.items():
            body = int(attempt["body"])  # type: ignore[call-overload]
            start = attempt["start"]
            assert isinstance(start, np.ndarray)
            queue = self.policy.queue[side]
            parts = self.hand_parts(data, side, body)
            rows = attempt["rows"]
            assert isinstance(rows, list)
            rows.append(
                {
                    "rise": float(data.xpos[body][2] - start[2]),
                    "moved": float(np.linalg.norm(data.xpos[body][:2] - start[:2])),
                    "grip": self.policy.grip[side],
                    "descending": bool(queue and queue[0].watch is not None),
                    "parts": parts,
                }
            )

    def hand_parts(self, data: mujoco.MjData, side: str, body: int) -> list[str]:
        """The parts of ``side``'s hand (ff, mf, rf, lf, th, palm) touching ``body``."""
        m, prefix, parts = self.rec.model, "lh_" if side == "left" else "rh_", set()
        for i in range(data.ncon):
            pair = (int(data.contact[i].geom1), int(data.contact[i].geom2))
            for g, other in (pair, pair[::-1]):
                name = m.body(m.geom_bodyid[g]).name
                if m.geom_bodyid[other] == body and name.startswith(prefix):
                    parts.add(name[3:5] if name[3:5] in (*FINGERS, "th") else "palm")
        return sorted(parts)


def cause(attempt: dict[str, object]) -> str:
    """The attempt's outcome (see the module docstring)."""
    rows = attempt["rows"]
    assert isinstance(rows, list)
    heights = np.array([r["rise"] for r in rows] or [0.0])
    touched = np.array([bool(r["parts"]) for r in rows] or [False])
    if count_lifts(heights, touched) > 0:
        return "success"
    if attempt["end"] == "replan":
        descent = next((i for i, r in enumerate(rows) if r["descending"]), len(rows))
        own = any(r["parts"] for r in rows[:descent])
        return "replan: object moved before the descent" + (" (own hand)" if own else "")
    if attempt["end"] == "reflex":
        return "reflex (limb > 80 N)"
    closed = next((r for r in rows if r["grip"] >= 0.99), None)
    if closed is None:
        return "ended before closure"
    parts = closed["parts"]
    if not (any(p in parts for p in FINGERS) and "th" in parts):
        return "no pinch at closure"
    heavy = " (metal box)" if attempt["object"] == "obj_metal_box" else ""
    return (
        "pinched, slipped during the lift" if heights.max() >= 0.01 else "pinched, no rise"
    ) + heavy


def run(episodes: int, first: int) -> tuple[list[dict[str, object]], int]:
    """Classified attempts of ``episodes`` episodes, and the dropped power plans."""
    rec = HumanoidRecorder({"effector": "hand", "posture": "seated"})
    m, d = rec.model, rec.data
    attempts, dropped = [], 0
    for seed in range(first, first + episodes):
        rng = np.random.default_rng(seed)
        policy = rec.make_policy("seated_babble", rng)
        assert isinstance(policy, BabblePolicy)
        rec.reset(policy, rng)
        tracker = Tracker(rec, policy)
        for step in range(STEPS):
            mujoco.mj_forward(m, d)
            d.ctrl[:] = rec.with_gravity_compensation(rec.space.to_ctrl(policy.act(step, d)))
            mujoco.mj_step(m, d, nstep=rec.substeps)
            tracker.record_step(d)
        for attempt in tracker.done:
            attempts.append(
                {"episode": seed, "side": attempt["side"], "object": attempt["object"]}
                | {"cause": cause(attempt)}
            )
        dropped += tracker.dropped
        print(f"episode {seed}: {len(tracker.done)} finished power attempts", flush=True)
    return attempts, dropped


def main() -> None:
    """Run, print the cause table and write the attempts."""
    episodes = int(sys.argv[1])
    first = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out = Path(sys.argv[3] if len(sys.argv) > 3 else "runs/hand_grasp/crowd.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    attempts, dropped = run(episodes, first)
    counts, n = Counter(str(a["cause"]) for a in attempts), len(attempts)
    print(
        f"{episodes} episodes, {n} power attempts, {dropped / episodes:.1f} dropped plans/episode"
    )
    print(f"lifts per episode {counts['success'] / episodes:.2f}")
    print("| cause | attempts | share [Wilson 95%] |\n|---|---|---|")
    for name, k in counts.most_common():
        low, high = wilson(k, n)
        print(f"| {name} | {k} | {k / n:.0%} [{low:.0%}, {high:.0%}] |")
    out.write_text(json.dumps(attempts))


if __name__ == "__main__":
    main()
