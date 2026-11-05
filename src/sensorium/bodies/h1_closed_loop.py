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

from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

from sensorium.bodies.h1_experiments import ExperimentPolicy
from sensorium.bodies.humanoid import HumanoidRecorder, Log
from sensorium.fewshot import BCHead, FeatureFn


class LearnedPolicy:
    """A head on frozen features drives the arms; the task program ``judge`` only judges.

    Args:
        judge: The task's demonstrator, in task mode (its ``reset`` and ``done`` are used).
        recorder: The recorder running the episode (for ``recent_episode``).
        features: The frozen encoder (``fewshot.FeatureFn``).
        head: The trained head; its ``horizon`` actions are played before it is asked again.
        first: The first step a window can end at (``fewshot.first_end``).
    """

    needs_observation = True

    def __init__(
        self,
        judge: ExperimentPolicy,
        recorder: HumanoidRecorder,
        features: FeatureFn,
        head: BCHead,
        first: int,
    ) -> None:
        self.judge, self.recorder, self.features, self.head = judge, recorder, features, head
        self.first, self.window = first, first + 1
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
            recent = self.recorder.recent_episode(self.log, self.window)
            feats = self.features(recent, np.array([recent.length - 1]))
            self.plan = list(self.head.predict(feats)[0])
            self.queries += 1
        return np.asarray(self.plan.pop(0), dtype=np.float32)

    def done(self, data: mujoco.MjData) -> bool:
        """The judge's success test (held ``SUCCESS_STEPS``)."""
        return self.judge.done(data)


@dataclass
class Outcome:
    """One closed-loop episode: its seed, success, length and the head's queries."""

    seed: int
    success: bool
    steps: int
    queries: int


def evaluate(
    recorder: HumanoidRecorder,
    seeds: Sequence[int],
    steps: int,
    features: FeatureFn,
    head: BCHead,
    first: int,
) -> list[Outcome]:
    """Run the learned controller on the recorder's task from each seed.

    The recorder must be a task scene (``program`` and ``task`` settings, as the task's
    demonstrations).

    Raises:
        ValueError: if the recorder has no task program.
    """
    if not recorder.program:
        raise ValueError("closed-loop evaluation needs a task program (the program setting)")
    out = []
    for seed in seeds:
        wrap = Wrap(recorder, features, head, first)
        episode = recorder.record(seed, "experiment", steps, wrap=wrap)
        assert wrap.policy is not None
        out.append(
            Outcome(seed, bool(episode.meta["success"]), episode.length, wrap.policy.queries)
        )
    return out


class Wrap:
    """``HumanoidRecorder.record``'s ``wrap``: the task program becomes a ``LearnedPolicy``'s
    judge."""

    def __init__(
        self, recorder: HumanoidRecorder, features: FeatureFn, head: BCHead, first: int
    ) -> None:
        self.recorder, self.features, self.head, self.first = recorder, features, head, first
        self.policy: LearnedPolicy | None = None

    def __call__(self, judge: object) -> LearnedPolicy:
        """Wrap the program the recorder made.

        Raises:
            TypeError: if it is not a task program.
        """
        if not isinstance(judge, ExperimentPolicy):
            raise TypeError(f"the judge must be an ExperimentPolicy, got {type(judge).__name__}")
        self.policy = LearnedPolicy(judge, self.recorder, self.features, self.head, self.first)
        return self.policy
