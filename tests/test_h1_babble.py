"""The D0.3 safety layer: geom classes, peak forces, collision-checked reaches, the table setting.

Needs the assets from ``make h1-assets`` and the simulator extra (skipped otherwise).
"""

from collections.abc import Iterator

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="needs the simulator extra: make setup")

import mujoco  # noqa: E402

from sensorium.bodies.h1_babble import HAND_PINCH, BabblePolicy, Waypoint
from sensorium.bodies.h1_forces import (  # noqa: E402
    CORE,
    LEFT_ARM,
    OBJECT,
    PEAK_CLASSES,
    TABLE,
    PeakForces,
)
from sensorium.bodies.h1_policies import SeatedBabblePolicy  # noqa: E402
from sensorium.bodies.h1_scene import (  # noqa: E402
    TABLE_TOP_Z,
    TABLE_X_BASE_Z,
    assets_available,
    table_edges,
)
from sensorium.bodies.h1_skin import Contacts  # noqa: E402
from sensorium.bodies.humanoid import HumanoidRecorder  # noqa: E402
from sensorium.episode import Episode  # noqa: E402

needs_assets = pytest.mark.skipif(
    not assets_available(), reason="H1 assets missing: run `make h1-assets`"
)


@pytest.fixture(scope="module")
def pincer() -> Iterator[HumanoidRecorder]:
    recorder = HumanoidRecorder({"effector": "pincer", "posture": "seated"})
    yield recorder
    recorder.close()


def settings_of(episode: Episode) -> dict[str, object]:
    """The recorder settings stored in an episode's meta."""
    settings = episode.meta["settings"]
    assert isinstance(settings, dict)
    return settings


def geom_of(model: mujoco.MjModel, body: str) -> int:
    """First geom of the named body."""
    return int(model.body_geomadr[model.body(body).id])


def test_table_edges_move_out_with_height() -> None:
    near, far = table_edges(TABLE_X_BASE_Z)
    assert (near, far) == (0.24, 0.76)
    assert table_edges(TABLE_TOP_Z) == pytest.approx((near + 0.1, far))  # the default table
    assert table_edges(TABLE_X_BASE_Z - 0.1) == (near, far)


@needs_assets
def test_geom_classes_and_limb_contacts(pincer: HumanoidRecorder) -> None:
    model, classes = pincer.model, pincer.classes
    table = int(model.geom("table_top").id)
    box = int(model.geom("obj_wood_box").id)
    pad = geom_of(model, "lp_left_pad")
    elbow = geom_of(model, "left_elbow_link")
    right_elbow = geom_of(model, "right_elbow_link")
    torso = geom_of(model, "torso_link")
    assert classes.kind[table] == TABLE and classes.kind[box] == OBJECT
    assert classes.object_index[box] == 0 and classes.object_index[table] == -1
    assert classes.effector[pad] and not classes.effector[elbow]
    assert classes.group[elbow] == classes.group[pad] == LEFT_ARM
    assert classes.group[torso] == CORE
    pairs = np.array(
        [
            [elbow, table],  # an elbow on the table: a slam
            [pad, table],  # the effector on the table: what babbling is for
            [pad, box],
            [elbow, box],
            [elbow, torso],  # a shoulder or elbow into the torso
            [elbow, pad],  # same limb
            [elbow, right_elbow],  # one arm into the other
            [torso, table],  # the body on the table
        ]
    )
    got = classes.limb_contacts(pairs)
    assert got.tolist() == [True, False, False, True, True, False, True, True]


@needs_assets
def test_peak_forces_take_the_maximum_per_class(pincer: HumanoidRecorder) -> None:
    model, classes = pincer.model, pincer.classes
    table, box = int(model.geom("table_top").id), int(model.geom("obj_wood_box").id)
    pad, elbow = geom_of(model, "lp_left_pad"), geom_of(model, "left_elbow_link")

    def contacts(geoms: list[list[int]], forces: list[float]) -> Contacts:
        n = len(forces)
        geom = np.array(geoms)
        return Contacts(
            body=model.geom_bodyid[geom],
            geom=geom,
            pos=np.zeros((n, 3)),
            normal=np.zeros((n, 3)),
            force=np.array(forces),
            slip=np.zeros(n),
        )

    peaks = PeakForces(classes)
    peaks.update(contacts([[pad, box], [elbow, table], [pad, table]], [5.0, 120.0, 7.0]))
    peaks.update(contacts([[box, pad], [elbow, table]], [9.0, 60.0]))
    peak, by_object = peaks.take()
    assert dict(zip(PEAK_CLASSES, peak.tolist(), strict=True)) == {
        "effector_object": 9.0,
        "effector_table": 7.0,
        "limb_scene": 120.0,
        "self": 0.0,
    }
    assert by_object.shape == (6,) and by_object[0] == 9.0 and by_object[1:].sum() == 0
    assert peaks.take()[0].sum() == 0  # cleared by take


@needs_assets
def test_record_logs_peak_forces_and_table_setting(pincer: HumanoidRecorder) -> None:
    episode = pincer.record(5, "seated_babble", 12)
    assert episode.privileged["peak_force"].shape == (12, len(PEAK_CLASSES))
    assert episode.privileged["object_contact"].shape == (12, 6)
    assert episode.meta["table_top"] == TABLE_TOP_Z and episode.meta["reflexes"] == 0
    assert settings_of(episode)["safe_reach"] is True
    assert episode.meta["peak_force_columns"] == list(PEAK_CLASSES)
    with HumanoidRecorder({"effector": "pincer", "posture": "seated", "table_top": 0.72}) as low:
        short = low.record(5, "seated_babble", 3)
    assert short.meta["table_top"] == 0.72
    assert short.privileged["object_pose"][0, :, 2].min() > 0.72
    assert settings_of(short)["table_top"] == 0.72
    with HumanoidRecorder({"effector": "pincer", "posture": "standing"}) as standing:
        assert "table_top" not in standing.settings


@needs_assets
def test_hover_stays_in_the_cameras_view_and_goals_inside_the_table_width(
    pincer: HumanoidRecorder,
) -> None:
    policy = pincer.make_policy("seated_babble", np.random.default_rng(0))
    assert isinstance(policy, BabblePolicy)
    top = pincer.table_top
    at_table = policy.hover_point(np.array([0.5, 0.2, top - 0.004]))
    assert at_table[2] == pytest.approx(top + policy.HOVER)  # over the table, not goal + 10 cm
    on_object = policy.hover_point(np.array([0.5, 0.2, top + 0.07]))
    assert on_object[2] == pytest.approx(top + 0.07 + policy.HOVER_MIN)  # never closer than 3 cm
    assert policy.TABLE_Y <= 0.30 and policy.LIFT_HEIGHT == 0.10
    pincer.reset(policy, np.random.default_rng(0))
    assert all(abs(policy.table_point(side)[1]) <= policy.TABLE_Y for side in ("left", "right"))
    objects = pincer.record(2, "seated_babble", 2).privileged["object_pose"][0, :, :2]
    assert np.abs(objects[:, 1]).max() <= 0.30


@needs_assets
def test_commands_follow_the_straight_joint_path(pincer: HumanoidRecorder) -> None:
    policy = pincer.make_policy("seated_babble", np.random.default_rng(0))
    assert isinstance(policy, BabblePolicy)
    pincer.reset(policy, np.random.default_rng(0))
    start = policy.cmd_q["left"].copy()
    goal = start + np.array([0.5, -0.2, 0.1, 0.3, 0.0])
    policy.queue["left"] = [Waypoint(goal, 0.05)]
    policy.advance("left", pincer.data, 0.0)
    step = policy.cmd_q["left"] - start
    assert np.abs(step).max() == pytest.approx(0.05)  # the fastest joint moves at the cap ...
    assert step / np.abs(step).max() == pytest.approx((goal - start) / 0.5)  # ... all on one line


@needs_assets
def test_safe_reach_switch_selects_the_policy(pincer: HumanoidRecorder) -> None:
    rng = np.random.default_rng(0)
    assert isinstance(pincer.make_policy("seated_babble", rng), BabblePolicy)
    with HumanoidRecorder(
        {"effector": "pincer", "posture": "seated", "safe_reach": False}
    ) as legacy:
        assert isinstance(legacy.make_policy("seated_babble", rng), SeatedBabblePolicy)
        episode = legacy.record(1, "seated_babble", 5)
        assert episode.meta["reflexes"] is None and settings_of(episode)["safe_reach"] is False


@needs_assets
def test_posture_check_rejects_collisions_and_misses(pincer: HumanoidRecorder) -> None:
    policy = pincer.make_policy("seated_babble", np.random.default_rng(0))
    assert isinstance(policy, BabblePolicy)
    pincer.reset(policy, np.random.default_rng(0))
    data = pincer.data
    goal = np.array([0.5, 0.2, pincer.table_top + 0.3])
    q = policy.solve_valid(data, "left", policy.cmd_q["left"], goal)
    assert q is not None and policy.posture_ok(data, "left", q, goal)  # a clear hover posture
    assert not policy.posture_ok(data, "left", q, goal + [0.5, 0.0, 0.0])  # tip misses the goal
    home = pincer.space.home("seated")[pincer.space.arm["left"]]
    assert not policy.posture_ok(data, "left", home, policy.ik["left"].tip_at(data.qpos, home))


@needs_assets
def test_safe_policy_keeps_arms_clear_and_touches_objects(pincer: HumanoidRecorder) -> None:
    episode = pincer.record(11, "seated_babble", 150)
    peak = episode.privileged["peak_force"]
    assert (peak[:, PEAK_CLASSES.index("limb_scene")] > 100.0).mean() < 0.05  # slams are rare
    assert float(str(episode.meta["object_contact_rate"])) > 0.0
    assert (episode.privileged["object_contact"] > 0.5).any()


@needs_assets
def test_goal_choice_prefers_the_least_tried(pincer: HumanoidRecorder) -> None:
    policy = pincer.make_policy("seated_babble", np.random.default_rng(0))
    assert isinstance(policy, BabblePolicy)
    draws = [policy.choose(np.array([30.0, 0.0, 30.0])) for _ in range(200)]
    assert draws.count(1) > 150
    assert set(policy.primitive_counts) == {"hover", "touch", "poke", "slide", "grasp"}


@needs_assets
def test_primitives_setting_restricts_the_repertoire() -> None:
    settings: dict[str, object] = {
        "effector": "hand",
        "posture": "seated",
        "primitives": ["touch", "slide"],
    }
    with HumanoidRecorder(settings) as recorder:
        policy = recorder.make_policy("seated_babble", np.random.default_rng(0))
        assert isinstance(policy, BabblePolicy) and policy.primitives == ("touch", "slide")
        assert settings_of(recorder.record(2, "seated_babble", 4))["primitives"] == [
            "touch",
            "slide",
        ]
    with HumanoidRecorder({"effector": "hand", "posture": "seated"}) as recorder:
        assert recorder.settings["primitives"] == ["hover", "touch", "poke", "slide", "power"]


@needs_assets
def test_grasp_plan_closes_lifts_and_releases(pincer: HumanoidRecorder) -> None:
    queue: list[Waypoint] = []
    for seed in range(3, 9):  # a layout in which some grasp is plannable
        policy = pincer.make_policy("seated_babble", np.random.default_rng(seed))
        assert isinstance(policy, BabblePolicy)
        policy.primitives = ("grasp",)
        policy.primitive_counts = {"grasp": 0}
        pincer.reset(policy, np.random.default_rng(seed))
        for side in ("left", "right"):
            policy.plan(side, pincer.data)
            queue = policy.queue[side]
            if len(queue) > 10:
                break
        if len(queue) > 10:
            break
    assert len(queue) > 10, "no grasp was planned on either side in six layouts"
    assert isinstance(policy, BabblePolicy)
    assert all(len(w.q) == 5 for w in queue)  # four arm joints and the wrist yaw
    grips = [w.grip for w in queue]
    assert max(grips) == 1.0 and grips[-1] == 0.0 and grips.index(1.0) < len(grips) - 3
    rises = [policy.ik["left"].tip_at(pincer.data.qpos, w.q)[2] for w in queue]
    assert max(rises) - min(rises) > 0.08  # it goes down to the object and lifts it (10 cm)


@needs_assets
def test_hand_closure_reaches_the_actuators() -> None:
    with HumanoidRecorder({"effector": "hand", "posture": "seated"}) as recorder:
        policy = recorder.make_policy("seated_babble", np.random.default_rng(0))
        assert isinstance(policy, BabblePolicy)
        assert len(policy.closure["left"]) == len(HAND_PINCH)  # the pinch's actuators
        assert len(policy.flexors["left"]) == 11  # those that follow the closure outside a grasp
        recorder.reset(policy, np.random.default_rng(0))

        def action_with(grip: float, preshape: bool) -> np.ndarray:
            policy.grip["left"] = grip
            waypoint = Waypoint(policy.cmd_q["left"].copy(), 0.1, 5, grip, preshape=preshape)
            policy.queue["left"] = [waypoint]
            return policy.act(0, recorder.data)

        flexors = [recorder.space.effector["left"][j] for j in policy.flexors["left"]]
        opened, closed = action_with(0.0, False), action_with(1.0, False)
        assert (closed[flexors] > opened[flexors] + 0.3).all()
        closing = [
            recorder.space.effector["left"][j]
            for j, (start, end) in policy.closure["left"].items()
            if end > start
        ]
        shaped, pinched = action_with(0.0, True), action_with(1.0, True)
        assert (pinched[closing] > shaped[closing] + 0.1).all()
        thumb = recorder.space.index["lh_A_THJ4"]  # the thumb is under the palm from the start
        assert shaped[thumb] > opened[thumb] + 1.0


@needs_assets
def test_hand_grasp_preshapes_over_the_object_and_replans_if_it_moved() -> None:
    with HumanoidRecorder(
        {"effector": "hand", "posture": "seated", "primitives": ["power"]}
    ) as recorder:
        queue: list[Waypoint] = []
        for seed in range(6):  # a layout in which some grasp is plannable
            policy = recorder.make_policy("seated_babble", np.random.default_rng(seed))
            assert isinstance(policy, BabblePolicy)
            recorder.reset(policy, np.random.default_rng(seed))
            for side in ("left", "right"):
                policy.plan(side, recorder.data)
                queue = policy.queue[side]
                if len(queue) > 10:
                    break
            if len(queue) > 10:
                break
        assert len(queue) > 10, "no hand grasp was planned on either side in six layouts"
        assert isinstance(policy, BabblePolicy)
        shaped = [w.preshape for w in queue]
        first = shaped.index(True)
        assert first > 0 and all(shaped[first:])  # travels open, pre-shaped from over the object
        assert queue[first].hold == policy.PRESHAPE_HOLD and queue[first].grip == 0.0
        watched = [i for i, w in enumerate(queue) if w.watch is not None]
        assert watched == [first + 1]  # the descent's first waypoint checks the object
        grips = [w.grip for w in queue]
        assert max(grips) == 1.0 and grips[-1] == 0.0

        watch = queue[first + 1].watch
        assert watch is not None
        body = watch[0]
        adr = recorder.model.jnt_qposadr[recorder.model.body_jntadr[body]]
        recorder.data.qpos[adr] += 0.05  # the object was pushed on the way
        mujoco.mj_forward(recorder.model, recorder.data)
        policy.queue[side] = queue[first + 1 :]
        policy.advance(side, recorder.data, 0.0)
        assert policy.replans + policy.reaims == 1
        assert policy.queue[side][0] is not queue[first + 1]
        if policy.reaims:  # aimed again at the same object, where it is now
            again = [w.watch for w in policy.queue[side] if w.watch is not None]
            assert again and again[0][0] == body
            assert np.allclose(again[0][1], recorder.data.xpos[body])
        policy.reaims_left[side] = 0  # re-aims used up: the next move drops the grasp
        policy.queue[side] = queue[first + 1 :]
        policy.advance(side, recorder.data, 0.0)
        assert policy.replans + policy.reaims == 2 and policy.replans >= 1


@needs_assets
def test_a_hand_pinches_a_standing_tall_block_near_its_top() -> None:
    with HumanoidRecorder(
        {"effector": "hand", "posture": "seated", "props": ["tall_block"]}
    ) as recorder:
        policy = recorder.make_policy("seated_babble", np.random.default_rng(0))
        assert isinstance(policy, BabblePolicy)
        recorder.reset(policy, np.random.default_rng(0))
        names = [recorder.model.body(b).name for b in policy.objects]
        index = names.index("obj_wood_tallbox")
        centre, half_height, _ = policy.object_geometry(recorder.data, index)
        assert policy.vertical_half_extent(recorder.data, index) == pytest.approx(half_height)
        point = policy.grasp_centre(recorder.data, index)
        top = centre[2] + half_height
        assert point[2] == pytest.approx(top - policy.PINCH_BELOW_TOP)  # not at its centre


@needs_assets
def test_scene_director_replaces_lost_objects_and_drops_when_idle() -> None:
    settings: dict[str, object] = {
        "effector": "pincer",
        "posture": "seated",
        "director": {"idle_s": 0.5, "stacks": 1.0},
    }
    with HumanoidRecorder(settings) as recorder:
        assert recorder.settings["director"] == {"idle_s": 0.5, "stacks": 1.0}
        episode = recorder.record(3, "hold", 60)
    codes = episode.privileged["intervention"]
    assert codes.shape == (60,) and codes.max() >= 2  # the arms hold still: idle drops follow
    pose = episode.privileged["object_pose"]
    stacked = [  # stacks = 1: one object starts on top of another
        (i, j)
        for i in range(pose.shape[1])
        for j in range(pose.shape[1])
        if i != j and np.linalg.norm(pose[0, i, :2] - pose[0, j, :2]) < 0.02
    ]
    assert stacked
    with HumanoidRecorder(settings) as recorder:
        recorder.reset(
            recorder.make_policy("hold", np.random.default_rng(0)), np.random.default_rng(0)
        )
        director = recorder.director
        assert director is not None
        body = recorder.objects[0]
        adr = recorder.model.jnt_qposadr[recorder.model.body_jntadr[body]]
        recorder.data.qpos[adr : adr + 3] = (0.5, 0.0, 0.3)  # fallen off the table, on the floor
        mujoco.mj_forward(recorder.model, recorder.data)
        assert director.step(recorder.data, 0) == 1  # re-placed
        assert recorder.data.qpos[adr + 2] > recorder.table_top


@needs_assets
def test_the_pincer_span_check_refuses_objects_wider_than_the_opening(
    pincer: HumanoidRecorder,
) -> None:
    policy = pincer.make_policy("seated_babble", np.random.default_rng(0))
    assert isinstance(policy, BabblePolicy)
    pincer.reset(policy, np.random.default_rng(0))
    q = policy.cmd_q["left"]
    assert policy.fits(pincer.data, "left", 0, q)  # PINCER_SPAN 0: no check
    policy.PINCER_SPAN = 0.001  # narrower than any object
    assert not policy.fits(pincer.data, "left", 0, q)
    policy.PINCER_SPAN = 1.0
    assert policy.fits(pincer.data, "left", 0, q)
