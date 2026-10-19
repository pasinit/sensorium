"""Experiment programs: the recorder hooks, task names, and that each program plans (assets)."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="needs the simulator extra: make setup")

from sensorium.bodies.h1_experiments import PROGRAMS, ExperimentPolicy  # noqa: E402
from sensorium.bodies.h1_scene import assets_available  # noqa: E402
from sensorium.bodies.humanoid import HumanoidRecorder  # noqa: E402
from sensorium.registry import load_registry  # noqa: E402

needs_assets = pytest.mark.skipif(not assets_available(), reason="run `make h1-assets`")
ROOT = Path(__file__).resolve().parent.parent
PROPS = {"topple": ["tall_block"], "slide": ["friction_patches"], "press": ["button"]}


def settings(effector: str, program: str) -> dict[str, object]:
    """Recorder settings of one program, with its props, hidden properties and effort."""
    return {
        "effector": effector,
        "posture": "seated",
        "program": program,
        "props": PROPS.get(program, ["hinged_flap"] if program == "door" else []),
        "hidden_properties": True,
        "effort": True,
    }


def test_unknown_program_and_missing_program_are_rejected() -> None:
    with pytest.raises(ValueError, match="program"):
        HumanoidRecorder({"effector": "hand", "posture": "seated", "program": "juggle"})


@needs_assets
def test_experiment_policy_needs_a_program() -> None:
    with HumanoidRecorder({"effector": "hand", "posture": "seated"}) as plain:
        with pytest.raises(ValueError, match="program"):
            plain.record(1, "experiment", 3)


@needs_assets
@pytest.mark.parametrize("program", PROGRAMS)
def test_every_program_plans_for_the_hands(program: str) -> None:
    """Each program produces a plan for at least one arm within a few draws (hands reach all)."""
    with HumanoidRecorder(settings("hand", program)) as recorder:
        rng = np.random.default_rng(2)
        policy = recorder.make_policy("experiment", rng)
        assert isinstance(policy, ExperimentPolicy) and policy.program == program
        recorder.reset(policy, rng)
        planned = 0
        for _ in range(3):
            for side in ("left", "right"):
                policy.plan(side, recorder.data)
                planned += int(not policy.queue[side][0].wait)
        assert planned > 0, f"{program}: no arm found a plan: {policy.plan_stats()}"


@needs_assets
def test_episode_names_its_program_and_task_and_resolves_to_a_source() -> None:
    with HumanoidRecorder(settings("pincer", "push")) as recorder:
        episode = recorder.record(7, "experiment", 20)
    assert episode.meta["program"] == "push" and episode.meta["task"] == "exp_push"
    registry = load_registry(ROOT / "configs/sources.toml")
    source = registry.resolve("humanoid", str(episode.meta["task"]), "pincer")
    assert source.name == "humanoid/pincer/exp_push" and "push" in source.programs
    assert registry.kind_of(str(episode.meta["policy"])) == "experiment"
    assert "plan_stats" in episode.meta
