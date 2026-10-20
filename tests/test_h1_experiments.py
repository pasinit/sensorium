"""Experiment programs: the recorder hooks, task names, and that each program plans (assets)."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="needs the simulator extra: make setup")

import mujoco  # noqa: E402

from sensorium.bodies.h1_experiments import PINCER_PROGRAMS, PROGRAMS, ExperimentPolicy  # noqa: E402
from sensorium.bodies.h1_scene import assets_available  # noqa: E402
from sensorium.bodies.humanoid import HumanoidRecorder  # noqa: E402
from sensorium.registry import load_registry  # noqa: E402

needs_assets = pytest.mark.skipif(not assets_available(), reason="run `make h1-assets`")
ROOT = Path(__file__).resolve().parent.parent
PROPS = {
    "topple": ["tall_block"],
    "slide": ["friction_patches"],
    "press": ["button"],
    "stack": ["stack_blocks", "target_disc"],
    "drawer_open": ["drawer"],
    "push_to_target": ["target_disc"],
}


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
@pytest.mark.parametrize("program", [p for p in PROGRAMS if p not in PINCER_PROGRAMS])
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


def test_pincer_programs_are_refused_for_hands() -> None:
    for program in PINCER_PROGRAMS:
        with pytest.raises(ValueError, match="pincer"):
            HumanoidRecorder({"effector": "hand", "posture": "seated", "program": program})


@needs_assets
@pytest.mark.parametrize("program", PINCER_PROGRAMS)
def test_pincer_programs_plan_in_some_layout(program: str) -> None:
    """Lift-drop, stack and the drawer plan for a pincer within a few layouts."""
    with HumanoidRecorder(settings("pincer", program)) as recorder:
        for seed in range(4):
            rng = np.random.default_rng(seed)
            policy = recorder.make_policy("experiment", rng)
            assert isinstance(policy, ExperimentPolicy)
            recorder.reset(policy, rng)
            for side in ("left", "right"):
                policy.plan(side, recorder.data)
                if not policy.queue[side][0].wait:
                    return
    pytest.fail(f"{program}: no plan in four layouts")


@needs_assets
def test_stack_blocks_replace_the_objects_and_share_a_side() -> None:
    with HumanoidRecorder(settings("pincer", "stack")) as recorder:
        for seed in range(4):
            episode = recorder.record(seed, "experiment", 2)
            pose = episode.privileged["object_pose"][0]
            assert pose.shape[0] == 2  # two blocks, not the six objects
            assert np.sign(pose[0, 1]) == np.sign(pose[1, 1])  # one arm's side
            assert np.linalg.norm(pose[0, :2] - pose[1, :2]) > 0.09


@needs_assets
def test_a_stack_counts_only_when_the_top_block_rests_released_on_the_other() -> None:
    with HumanoidRecorder({**settings("pincer", "stack"), "task": True}) as recorder:
        policy = recorder.make_policy("experiment", np.random.default_rng(0))
        assert isinstance(policy, ExperimentPolicy)
        recorder.reset(policy, np.random.default_rng(0))
        model, data = recorder.model, recorder.data
        top, base = (model.jnt_qposadr[model.body_jntadr[b]] for b in recorder.objects)
        size = model.geom_size[int(model.body_geomadr[recorder.objects[0]])]
        data.qpos[base : base + 2] = (0.62, 0.0)  # clear of both arms' starts
        data.qpos[top : top + 3] = data.qpos[base : base + 3] + [0.0, 0.0, 2 * size[2] + 0.03]
        data.qpos[top + 3 : top + 7] = data.qpos[base + 3 : base + 7]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        assert not policy.succeeded(data)  # above the other block, not on it
        for _ in range(400):  # let it fall onto the base and settle
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        assert policy.succeeded(data)


@needs_assets
def test_a_demonstrator_ends_on_success_and_carries_the_held_out_task_name() -> None:
    task = {**settings("pincer", "drawer_open"), "task": True}
    with HumanoidRecorder(task) as recorder:
        episode = recorder.record(41, "experiment", 400)
    assert episode.meta["task"] == "drawer_open" and episode.meta["success"] is True
    assert episode.meta["end_reason"] == "success" and episode.length < 400
    registry = load_registry(ROOT / "configs/sources.toml")
    assert (
        registry.resolve("humanoid", "drawer_open", "pincer").name == "humanoid/pincer/drawer_open"
    )
