"""Closed-loop evaluation of a learned controller on a held-out task (D0.6; docs/experiment-01.md).

The task's scripted program is the judge (``h1_experiments.ExperimentPolicy`` in task mode). It
sets the episode up exactly as for its demonstrations, from the same seeds: the props, the object
layout and the arms' start. It then tells when the task's success test has held for
``SUCCESS_STEPS``, which ends the episode as a success. It never moves the arms: a
``LearnedPolicy`` does.
- Every ``horizon`` steps it encodes the last ``window`` steps of what the robot sensed
  (``HumanoidRecorder.recent_episode``, model inputs only).
- It asks the head (``fewshot.BCHead``) for the next ``horizon`` actions and plays them.
- Until a full window has been sensed, it holds the start posture.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import cast

import mujoco
import numpy as np

from sensorium.bodies.h1_experiments import ExperimentPolicy
from sensorium.bodies.humanoid import HumanoidRecorder, Log
from sensorium.episode import Episode
from sensorium.fewshot import BCHead, FeatureFn


class LearnedPolicy:
    """A head on frozen features drives the arms; the task program ``judge`` only judges.

    Args:
        judge: The task's demonstrator, in task mode (its ``reset`` and ``done`` are used).
        recorder: The recorder running the episode (for ``recent_episode``).
        features: The frozen encoder (``fewshot.FeatureFn``).
        head: The trained head; its ``horizon`` actions are played before it is asked again.
        first: The first step a window can end at (``fewshot.first_end``).
        privileged: Hand the features the privileged state too (the oracle diagnostic,
            ``fewshot.state_features``); off for every encoder.
    """

    needs_observation = True

    def __init__(
        self,
        judge: ExperimentPolicy,
        recorder: HumanoidRecorder,
        features: FeatureFn,
        head: BCHead,
        first: int,
        privileged: bool = False,
    ) -> None:
        self.judge, self.recorder, self.features, self.head = judge, recorder, features, head
        self.first, self.window = first, first + 1
        self.privileged = privileged
        self.plan: list[np.ndarray] = []
        self.log: Log | None = None
        self.hold: np.ndarray | None = None
        self.queries = 0

    def reset(self, data: mujoco.MjData) -> np.ndarray:
        """The judge's start (props, layout, arms), as for the demonstrations."""
        home = self.judge.reset(data)
        self.hold = np.asarray(self.judge.space.to_action(home), dtype=np.float32)
        self.plan = []
        return home

    def see(self, log: Log) -> None:
        """The senses logged so far (the recorder calls this before ``act``)."""
        self.log = log

    def act(self, step: int, data: mujoco.MjData) -> np.ndarray:
        """The next action of the current chunk; a new chunk every ``horizon`` steps."""
        assert self.log is not None and self.hold is not None
        if step < self.first:
            return self.hold
        if not self.plan:
            recent = self.recorder.recent_episode(self.log, self.window, self.privileged)
            feats = self.features(recent, np.array([recent.length - 1]))
            self.plan = list(self.head.predict(feats)[0])
            self.queries += 1
        return np.asarray(self.plan.pop(0), dtype=np.float32)

    def done(self, data: mujoco.MjData) -> bool:
        """The judge's success test (held ``SUCCESS_STEPS``)."""
        return self.judge.done(data)

    @property
    def task_counts(self) -> dict[str, int]:
        """The judge's counts (its own pushes stay 0: it never acts)."""
        return {}

    @property
    def milestones(self) -> dict[str, bool]:
        """The judge's sub-goals of the task, as reached so far."""
        return self.judge.milestones


@dataclass
class Outcome:
    """One closed-loop episode: its seed, success, length, the head's queries, the task's
    sub-goals reached (``ExperimentPolicy.milestones``) and whether it ran into the step cap
    without succeeding (``capped``: a failure, counted as one)."""

    seed: int
    success: bool
    steps: int
    queries: int
    milestones: dict[str, bool] = field(default_factory=dict)
    capped: bool = False


class ReplayPolicy:
    """A null controller for the task's floor: plays ``actions`` open loop from the start, then
    holds the last one; with no actions it holds the start posture throughout. The task program
    ``judge`` sets the episode up and judges it, as for ``LearnedPolicy``."""

    queries = 0

    def __init__(self, judge: ExperimentPolicy, actions: np.ndarray | None = None) -> None:
        self.judge = judge
        self.actions = None if actions is None else np.asarray(actions, dtype=np.float32)
        self.hold: np.ndarray | None = None

    def reset(self, data: mujoco.MjData) -> np.ndarray:
        """The judge's start, as for the demonstrations."""
        home = self.judge.reset(data)
        self.hold = np.asarray(self.judge.space.to_action(home), dtype=np.float32)
        return home

    def act(self, step: int, data: mujoco.MjData) -> np.ndarray:
        """The replayed action of this step, or the held one."""
        assert self.hold is not None
        if self.actions is not None and len(self.actions):
            self.hold = self.actions[min(step, len(self.actions) - 1)]
        return self.hold

    def done(self, data: mujoco.MjData) -> bool:
        """The judge's success test."""
        return self.judge.done(data)

    @property
    def milestones(self) -> dict[str, bool]:
        """The judge's sub-goals reached so far."""
        return self.judge.milestones


Controller = LearnedPolicy | ReplayPolicy


def evaluate(
    recorder: HumanoidRecorder,
    seeds: Sequence[int],
    steps: int,
    features: FeatureFn,
    head: BCHead,
    first: int,
    on_episode: Callable[[Outcome], None] | None = None,
    privileged: bool = False,
) -> list[Outcome]:
    """Run the learned controller on the recorder's task from each seed (``evaluate_with``);
    ``privileged`` for the oracle features (``LearnedPolicy``)."""
    return evaluate_with(
        recorder,
        seeds,
        steps,
        lambda judge: LearnedPolicy(judge, recorder, features, head, first, privileged),
        on_episode,
    )


def evaluate_program(
    recorder: HumanoidRecorder,
    seeds: Sequence[int],
    steps: int,
    on_episode: Callable[[Outcome], None] | None = None,
) -> list[Outcome]:
    """Run the task's scripted program itself from each seed: the demonstrator on the closed-loop
    start states, judged by its own success test with the same step cap (the reachable ceiling of
    the task; ``queries`` is 0).

    Raises:
        ValueError: if the recorder has no task program.
    """
    if not recorder.program:
        raise ValueError("the expert needs a task program (the program setting)")
    out = []
    for seed in seeds:
        outcome = _outcome(seed, recorder.record(seed, "experiment", steps), 0)
        out.append(outcome)
        if on_episode is not None:
            on_episode(outcome)
    return out


DIVERGENCE_STEPS = (0, 1, 2, 5, 10, 25, 50, 100, 200, 400)
"""Steps at which ``divergence`` reports the gap (those within the compared length)."""


def divergence(
    recorded: Episode, replayed: Episode, tol_qpos: float = 1e-3, tol_pos: float = 1e-3
) -> dict[str, object]:
    """How far a replay's physics drifts from the recorded episode, step by step.

    ``qpos`` is the largest absolute joint-position gap (all of ``privileged["qpos"]``) and ``pos``
    the largest gap in the props' positions (metres) at each step, over the steps both have.
    ``first_step`` is the first step at which either exceeds its tolerance (``None``: never).
    """
    n = min(recorded.length, replayed.length)
    dq = np.abs(
        np.asarray(recorded.privileged["qpos"])[:n] - np.asarray(replayed.privileged["qpos"])[:n]
    )
    dq = dq.reshape(n, -1).max(1)
    a = np.asarray(recorded.privileged["object_pose"])[:n, :, :3]
    b = np.asarray(replayed.privileged["object_pose"])[:n, :, :3]
    dp = np.abs(a - b).reshape(n, -1).max(1) if a.size else np.zeros(n)
    over = np.flatnonzero((dq > tol_qpos) | (dp > tol_pos))
    first = int(over[0]) if len(over) else None
    return {
        "steps_compared": int(n),
        "first_step": first,
        "qpos_max": float(dq.max()) if n else 0.0,
        "pos_max": float(dp.max()) if n else 0.0,
        "at": {str(s): [float(dq[s]), float(dp[s])] for s in DIVERGENCE_STEPS if s < n},
        "at_first": None if first is None else [float(dq[first]), float(dp[first])],
    }


def self_replay(
    recorder: HumanoidRecorder, demo: Episode, steps: int
) -> tuple[Outcome, dict[str, object]]:
    """Replay a demonstration's own recorded actions open loop from its own seed, judged by the task
    program, and measure the physics' drift from the recording (``divergence``).

    With a deterministic simulator and actions logged at the step they are applied, the replay is
    the recording again: same success, zero drift. A drift points at the step where logging,
    timing or determinism breaks.
    """
    seed = int(str(demo.meta["seed"]))
    wrap = Wrap(lambda judge: ReplayPolicy(judge, demo.action))
    episode = recorder.record(seed, "experiment", steps, wrap=wrap)
    return _outcome(seed, episode, 0), divergence(demo, episode)


def _outcome(seed: int, episode: Episode, queries: int) -> Outcome:
    """An episode's outcome: success, length, sub-goals, and whether it ran into the cap."""
    milestones = cast(dict[str, bool], episode.meta.get("milestones", {}))
    success = bool(episode.meta["success"])
    capped = not success and episode.meta.get("end_reason") == "steps"
    return Outcome(seed, success, episode.length, queries, dict(milestones), capped)


def evaluate_with(
    recorder: HumanoidRecorder,
    seeds: Sequence[int],
    steps: int,
    make: Callable[[ExperimentPolicy], Controller],
    on_episode: Callable[[Outcome], None] | None = None,
) -> list[Outcome]:
    """Run the controller ``make(judge)`` on the recorder's task from each seed.

    ``on_episode`` is called with each outcome as soon as its episode ends (progress output).

    The recorder must be a task scene (``program`` and ``task`` settings, as the task's
    demonstrations).

    Raises:
        ValueError: if the recorder has no task program.
    """
    if not recorder.program:
        raise ValueError("closed-loop evaluation needs a task program (the program setting)")
    out = []
    for seed in seeds:
        wrap = Wrap(make)
        episode = recorder.record(seed, "experiment", steps, wrap=wrap)
        assert wrap.policy is not None
        outcome = _outcome(seed, episode, wrap.policy.queries)
        out.append(outcome)
        if on_episode is not None:
            on_episode(outcome)
    return out


class Wrap:
    """``HumanoidRecorder.record``'s ``wrap``: the task program becomes a controller's judge."""

    def __init__(self, make: Callable[[ExperimentPolicy], Controller]) -> None:
        self.make = make
        self.policy: Controller | None = None

    def __call__(self, judge: object) -> Controller:
        """Wrap the program the recorder made.

        Raises:
            TypeError: if it is not a task program.
        """
        if not isinstance(judge, ExperimentPolicy):
            raise TypeError(f"the judge must be an ExperimentPolicy, got {type(judge).__name__}")
        self.policy = self.make(judge)
        return self.policy
