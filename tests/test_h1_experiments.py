"""Experiment programs: the recorder hooks, task names, and that each program plans (assets)."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="needs the simulator extra: make setup")

import mujoco  # noqa: E402

from sensorium.bodies.h1_experiments import (  # noqa: E402
    PINCER_PROGRAMS,
    PLACE_FAR_M,
    PROGRAMS,
    SUCCESS_STEPS,
    ExperimentPolicy,
)
from sensorium.bodies.h1_props import STACK_X, STACK_Y  # noqa: E402
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
    "place_on_target": ["target_disc", "place_block"],
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
        episode = recorder.record(2, "experiment", 400)  # seed 41 succeeded by a shove until fixed
    assert episode.meta["task"] == "drawer_open" and episode.meta["success"] is True
    assert episode.meta["end_reason"] == "success" and episode.length < 400
    assert episode.meta["milestones"] == {"pinched": True, "opened": True}  # pulled by the knob
    registry = load_registry(ROOT / "configs/sources.toml")
    assert (
        registry.resolve("humanoid", "drawer_open", "pincer").name == "humanoid/pincer/drawer_open"
    )


@needs_assets
def test_a_drawer_shoved_open_is_no_success_and_one_pulled_by_its_knob_is() -> None:
    with HumanoidRecorder({**settings("pincer", "drawer_open"), "task": True}) as recorder:
        policy = recorder.make_policy("experiment", np.random.default_rng(0))
        assert isinstance(policy, ExperimentPolicy)
        recorder.reset(policy, np.random.default_rng(0))
        model, data = recorder.model, recorder.data
        joint = model.joint("prop_drawer_l")
        low, high = model.jnt_range[joint.id]
        data.qpos[joint.qposadr[0]] = low + 0.9 * (high - low)  # open, with nothing on the knob
        mujoco.mj_forward(model, data)
        assert not policy.succeeded(data)
        assert policy.milestones == {"pinched": False, "opened": False}
        policy.pinched = lambda data, geoms: True  # type: ignore[method-assign]  # pads on the knob
        assert policy.succeeded(data)
        assert policy.milestones == {"pinched": True, "opened": True}


@needs_assets
def test_the_drawer_demonstrator_travels_clear_of_the_drawers() -> None:
    with HumanoidRecorder({**settings("pincer", "drawer_open"), "task": True}) as recorder:
        episode = recorder.record(41, "experiment", 40)
        names = cast(list[str], episode.meta["body_names"])
    contacts = episode.privileged["contacts"][:20]  # [T, rows, 6]: body1, body2, force, xyz
    drawer = {i for i, n in enumerate(names) if n.startswith("prop_drawer_")}
    pincer = {i for i, n in enumerate(names) if n.startswith(("lp_", "rp_"))}
    hits = [
        (t, names[int(a)], names[int(b)])
        for t, rows in enumerate(contacts)
        for a, b, *_ in rows
        if a >= 0 and {int(a), int(b)} & drawer and {int(a), int(b)} & pincer
    ]
    assert not hits  # the old travel shoved a drawer open within the first 10 steps
    assert float(episode.privileged["prop_state"][:20].max()) < 0.05
    assert episode.meta["milestones"] == {"pinched": False, "opened": False}


@needs_assets
def test_push_to_target_layout_is_split_v3_1_and_pushes_are_short_and_re_aimed() -> None:
    from sensorium.bodies.h1_experiments import PUSH_STEP_M
    from sensorium.bodies.h1_props import FAR_OBJECT_X, TARGET_X

    with HumanoidRecorder({**settings("hand", "push_to_target"), "task": True}) as recorder:
        for seed in range(3):
            policy = recorder.make_policy("experiment", np.random.default_rng(seed))
            assert isinstance(policy, ExperimentPolicy)
            recorder.reset(policy, np.random.default_rng(seed))
            data = recorder.data
            target = policy.target_xy(data)
            assert TARGET_X[0] <= target[0] <= TARGET_X[1]  # inside both effectors' reach
            first = data.xpos[recorder.objects[0]][:2]
            assert FAR_OBJECT_X[0] <= first[0] <= FAR_OBJECT_X[1]
            assert np.linalg.norm(first - target) >= 0.17
        assert isinstance(policy, ExperimentPolicy)
        side = "left" if target[1] > 0 else "right"
        legs: list[tuple[np.ndarray, np.ndarray]] = []

        def fake_path(
            side: str, data: mujoco.MjData, start: np.ndarray, end: np.ndarray, **kw: object
        ) -> list[object]:
            legs.append((start, end))
            return []

        policy.contact_path = fake_path  # type: ignore[method-assign,assignment]
        assert policy.plan_push_to_target(side, data) == []
        start, end = legs[0]
        assert float(np.linalg.norm(end - start)) <= 0.03 + PUSH_STEP_M + 1e-6  # one short push
        assert policy.task_counts == {"pushes": 1} and policy.pushed is not None
        pushed = policy.pushed
        policy.plan_push_to_target(side, data)  # the same object again, re-aimed from where it is
        assert policy.pushed == pushed and policy.task_counts == {"pushes": 2}
        assert policy.milestones == {"reached": False, "at_target": False}
        policy.effector_touches = lambda data, geoms: True  # type: ignore[method-assign]
        assert not policy.succeeded(data) and not policy.milestones["reached"]  # nothing moved
        body = recorder.objects[0]
        adr = recorder.model.jnt_qposadr[recorder.model.body_jntadr[body]]
        data.qpos[adr] += 0.02  # the far object pushed 2 cm
        mujoco.mj_forward(recorder.model, data)
        assert not policy.succeeded(data) and policy.milestones["reached"]


@needs_assets
def test_place_on_target_starts_t1s_block_on_the_disc_side_10_cm_or_more_away() -> None:
    with HumanoidRecorder({**settings("pincer", "place_on_target"), "task": True}) as recorder:
        assert [recorder.model.body(b).name for b in recorder.objects] == ["obj_wood_stackbox_a"]
        for seed in range(6):
            policy = recorder.make_policy("experiment", np.random.default_rng(seed))
            recorder.reset(policy, np.random.default_rng(seed))
            xy = recorder.data.xpos[recorder.objects[0]][:2]
            target = recorder.target[:2]
            assert np.sign(xy[1]) == np.sign(target[1])
            assert STACK_X[0] <= xy[0] <= STACK_X[1] and STACK_Y[0] <= abs(xy[1]) <= STACK_Y[1]
            assert np.linalg.norm(xy - target) >= PLACE_FAR_M


@needs_assets
def test_place_on_target_needs_a_lift_then_a_second_at_rest_on_the_disc() -> None:
    with HumanoidRecorder({**settings("pincer", "place_on_target"), "task": True}) as recorder:
        policy = recorder.make_policy("experiment", np.random.default_rng(0))
        assert isinstance(policy, ExperimentPolicy)
        recorder.reset(policy, np.random.default_rng(0))
        model, data = recorder.model, recorder.data
        assert not policy.succeeded(data)
        assert policy.milestones == dict.fromkeys(
            ("grasped", "lifted", "at_target", "released"), False
        )
        body = recorder.objects[0]
        adr = model.jnt_qposadr[model.body_jntadr[body]]
        start = data.qpos[adr : adr + 3].copy()
        target = policy.target_xy(data)
        policy.robot_touches = lambda data, geoms: False  # type: ignore[method-assign]

        def put(xy: np.ndarray, dz: float = 0.0) -> None:
            data.qpos[adr : adr + 3] = [xy[0], xy[1], start[2] + dz]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

        put(target)  # dragged onto the disc, never lifted: not at the target, not a success
        assert not policy.succeeded(data) and not policy.milestones["at_target"]
        policy.pinched = lambda data, geoms: True  # type: ignore[method-assign]
        put(start[:2], 0.03)  # risen 3 cm in the pads
        assert not policy.succeeded(data) and policy.milestones["lifted"]
        policy.pinched = lambda data, geoms: False  # type: ignore[method-assign]
        put(target)  # set down on the disc and let go
        assert policy.succeeded(data) and policy.milestones["at_target"]
        assert not policy.milestones["released"]  # only once the test has held 1 s
        held = [policy.done(data) for _ in range(SUCCESS_STEPS)]
        assert held[-1] and not any(held[:-1]) and policy.milestones["released"]
        put(start[:2])
        side = "left" if target[1] > 0 else "right"
        policy.milestones["released"] = False
        queue = policy.plan_place_on_target(side, data)
        assert queue is None or len(queue) > 3  # a plan, if this layout has one, is a full move
