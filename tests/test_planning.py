"""B-plan-0: the model-side planning pieces and the simulator restore (the latter needs the
assets; skipped otherwise)."""

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="needs the model extra: uv sync --extra model")

from test_model import LOADER, TINY, make_episode  # noqa: E402

from sensorium.episode import Episode  # noqa: E402
from sensorium.model import planning  # noqa: E402
from sensorium.model.data import build, header_of, load_episode  # noqa: E402
from sensorium.model.layout import ChannelSpace  # noqa: E402
from sensorium.model.paired import contrast_values  # noqa: E402
from sensorium.model.sensory_layer import SensoryLayer, SensoryModel  # noqa: E402

SENSES = ("vision", "proprio_body", "vestibular", "proprio_effector")


def arm_episode(seed: int, steps: int = 96) -> Episode:
    """A synthetic episode whose proprio has the eight arm joints of the success criterion."""
    ep = make_episode("hand", steps=steps, seed=seed)
    rng = np.random.default_rng(seed)
    arms = np.sin(np.linspace(0, 4, steps)[:, None] + rng.uniform(0, 1, (1, 8)))
    ep.proprio_names = [*ep.proprio_names, *planning.ARM_JOINTS]
    ep.proprio = np.concatenate([ep.proprio, arms], 1).astype(np.float32)
    return ep


@pytest.fixture(scope="module")
def tiny_layer() -> SensoryLayer:
    eps = [arm_episode(0), arm_episode(1)]
    space = ChannelSpace.from_headers(header_of(e) for e in eps)
    torch.manual_seed(0)
    return SensoryLayer(SensoryModel(replace(TINY, senses=SENSES), space), space, LOADER)


def test_states_and_chunks_match_the_encoder_and_the_loader(tiny_layer: SensoryLayer) -> None:
    ep = arm_episode(0)
    states, ends = tiny_layer.encode(ep, stride=4)
    assert np.allclose(planning.states_at(tiny_layer, ep, ends[:5]), states[:5], atol=1e-5)
    loaded = load_episode(ep, tiny_layer.space, tiny_layer.model.senses)
    actions = build(loaded, tiny_layer.space, tiny_layer.model.senses, ends[:5], 4)["action"]
    assert np.array_equal(planning.chunks_at(tiny_layer, ep, ends[:5]), actions)
    q = planning.arm_q(ep, np.array([3, 7]))
    cols = [list(ep.proprio_names).index(n) for n in planning.ARM_JOINTS]
    assert q.shape == (2, 8) and np.allclose(q, ep.proprio[[3, 7]][:, cols])


def test_encode_at_returns_the_state_and_the_summary_it_projects(tiny_layer: SensoryLayer) -> None:
    ep = arm_episode(0)
    _, ends = tiny_layer.encode(ep, stride=4)
    s, z = planning.encode_at(tiny_layer, ep, ends[:5])
    assert np.array_equal(s, planning.states_at(tiny_layer, ep, ends[:5]))
    model = tiny_layer.model.eval()
    with torch.no_grad():
        projected = model.encoder.project(torch.as_tensor(z, dtype=torch.float32)[:, None])
    assert np.allclose(projected[:, 0].numpy(), s, atol=1e-5)
    assert not model.encoder.projector._forward_pre_hooks  # the hook is removed


def test_encode_parts_returns_every_stage_of_a_sense(tiny_layer: SensoryLayer) -> None:
    ep = arm_episode(0)
    _, ends = tiny_layer.encode(ep, stride=4)
    parts = planning.encode_parts(tiny_layer, ep, ends[:5])
    s, z = planning.encode_at(tiny_layer, ep, ends[:5])
    assert np.array_equal(parts["s"], s) and np.array_equal(parts["z"], z)
    model, enc = tiny_layer.model, tiny_layer.model.encoder
    loaded = load_episode(ep, tiny_layer.space, model.senses)
    arrays = build(loaded, tiny_layer.space, model.senses, ends[:5], 4)
    x = arrays["proprio_body"]
    assert np.allclose(parts["input_last"], x[:, -1]) and parts["input_window"].shape == (
        5,
        x[0].size,
    )
    with torch.no_grad():
        stem = enc.stems["proprio_body"](torch.as_tensor(x)).numpy()
    assert np.allclose(parts["stem"], stem, atol=1e-5)  # the stem of the model input
    fused: list[Any] = []
    hook = enc.fusion.register_forward_hook(lambda m, a, o: fused.append(o.detach()))
    planning.encode_parts(tiny_layer, ep, ends[:5])
    hook.remove()
    slot = 1 + sum(enc.n_tokens[x] for x in enc.senses[: enc.senses.index("proprio_body")])
    assert np.allclose(parts["fused"], fused[0][:, slot].numpy(), atol=1e-6)
    every = fused[0][:, 1:].reshape(5, -1).numpy()  # every sense token, after the summary
    assert np.allclose(parts["fused_all"], every, atol=1e-6)
    assert parts["fused_all"].shape == (5, sum(enc.n_tokens.values()) * enc.cfg.dim)
    assert not enc.stems["proprio_body"]._forward_hooks  # every hook is removed
    with pytest.raises(ValueError, match="one-token"):
        planning.encode_parts(tiny_layer, ep, ends[:5], sense="vision")


def test_tokens_dyn_helpers_decode_per_effector_and_score_each_horizon() -> None:
    plan = load_plan_script()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 6))
    eff = np.where(np.arange(200) < 100, "hand", "pincer")
    w = {"hand": rng.standard_normal((6, 8)), "pincer": rng.standard_normal((6, 8))}
    y = np.stack([x[i] @ w[e] for i, e in enumerate(eff)])
    groups = np.arange(200) // 10
    maps = plan.effector_ridges(x, y, groups, eff)
    assert set(maps) == {"hand", "pincer", "*"}
    assert np.allclose(plan.decode(maps, x, eff), y, atol=0.05)  # one map per effector
    seq = np.stack([x, x], 1)  # [n, H, D]: rows keep their effector across horizons
    assert plan.decode(maps, seq, eff).shape == (200, 2, 8)
    other = plan.decode(maps, x[:3], np.array(["drone"] * 3))  # unseen: the pooled map
    assert np.allclose(other, maps["*"](x[:3]))
    q = rng.standard_normal((50, 3, 8))
    r2 = plan.horizon_r2(q[:, 0], q)  # a held copy: perfect at the first horizon only
    assert r2[0] == pytest.approx(1.0) and r2.shape == (3,) and r2[1] < 0.5
    assert np.allclose(plan.horizon_r2(q, q), 1.0)


def test_choose_alpha_takes_the_number_of_inner_folds() -> None:
    from sensorium.model import probes  # noqa: PLC0415

    rng = np.random.default_rng(1)
    x, y = rng.standard_normal((120, 5)), rng.standard_normal((120, 2))
    groups = np.arange(120) // 6
    assert probes.choose_alpha(x, y, groups) == probes.choose_alpha(x, y, groups, 3)
    assert probes.choose_alpha(x, y, groups, 5) / 5 in probes.ALPHA_GRID


def test_token_cosine_is_one_for_copies_and_zero_for_orthogonal_tokens() -> None:
    plan = load_plan_script()
    rng = np.random.default_rng(0)
    t = rng.standard_normal((10, 1, 6))
    assert plan.token_cosine(np.concatenate([t, 2 * t, t], 1).reshape(10, -1), 3) == pytest.approx(
        1
    )
    eye = np.tile(np.eye(3)[None], (4, 1, 1)).reshape(4, -1)  # three orthogonal tokens per row
    assert plan.token_cosine(eye, 3) == pytest.approx(0)


def test_own_rollout_feeds_its_predictions_back(tiny_layer: SensoryLayer) -> None:
    ep = arm_episode(0)
    ends = np.array([3, 7, 11, 15, 19])
    states = planning.states_at(tiny_layer, ep, ends)
    chunks = planning.chunks_at(tiny_layer, ep, ends[:-1])
    pred = planning.own_rollout(tiny_layer, states[None, 0], chunks[None])
    assert pred.shape == (1, 4, states.shape[1])
    model = tiny_layer.model.eval()
    with torch.no_grad():  # the first step is the predictor on the observed start alone
        s = torch.as_tensor(states[None, :1], dtype=torch.float32)
        cond = model.action(torch.as_tensor(chunks[:1], dtype=torch.float32)).view(1, 1, -1)
        first = model.predict(s, cond)[0, -1].numpy()
    assert np.allclose(pred[0, 0], first, atol=1e-5)


def test_ridge_dynamics_recovers_a_linear_system() -> None:
    rng = np.random.default_rng(0)
    a, b = 0.9 * np.eye(3), rng.normal(size=(2, 3))
    s = rng.normal(size=(400, 3))
    u = rng.normal(size=(400, 1, 2))
    nxt = s @ a + u[:, 0] @ b
    groups = np.repeat(np.arange(20), 20)
    dyn = planning.RidgeDynamics.fit(s, u, nxt, groups)
    roll = dyn.rollout(s[:5], np.stack([u[:5], u[5:10]], 1))
    step1 = s[:5] @ a + u[:5, 0] @ b
    assert np.allclose(roll[:, 0], step1, atol=0.05)
    assert np.allclose(roll[:, 1], step1 @ a + u[5:10, 0] @ b, atol=0.1)


def test_cem_finds_the_minimum_of_a_quadratic() -> None:
    target = np.array([[0.3, -0.2], [0.1, 0.4]])
    best = planning.cem(
        lambda c: ((c - target) ** 2).sum((1, 2)),
        np.zeros((2, 2)),
        np.ones((2, 2)),
        -1.0,
        1.0,
        np.random.default_rng(0),
        iterations=10,
    )
    assert np.allclose(best, target, atol=0.05)


def test_contrast_values_pairs_by_seed_and_bootstraps_episodes() -> None:
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(30), 4)
    a = (rng.random((3, 120)) < 0.7).astype(float)
    b = (rng.random((3, 120)) < 0.4).astype(float)
    c = contrast_values(a, b, groups, draws=50)
    assert c.point == pytest.approx(a.mean() - b.mean()) and c.resolved
    same = contrast_values(a, a, groups, draws=20)
    assert same.point == 0 and same.seed_se == 0 and same.episode_se == 0
    with pytest.raises(ValueError, match="same items"):
        contrast_values(a, b[:, :100], groups)


def test_sliding_ends_stays_inside_the_episode() -> None:
    ends = planning.sliding_ends(100, 3, [3, 50], 4, 8)
    assert ends.tolist()[1] == list(range(50, 83, 4))
    with pytest.raises(ValueError, match="too close"):
        planning.sliding_ends(100, 3, [80], 4, 8)


mujoco = pytest.importorskip("mujoco", reason="needs the simulator extra: make setup")

from sensorium.bodies.h1_scene import assets_available  # noqa: E402

needs_assets = pytest.mark.skipif(not assets_available(), reason="run `make h1-assets`")
SETTINGS: dict[str, object] = {"effector": "hand", "posture": "seated"}


@pytest.fixture(scope="module")
def recorded() -> Iterator[Episode]:
    from sensorium.bodies.humanoid import HumanoidRecorder  # noqa: PLC0415

    with HumanoidRecorder(SETTINGS) as rec:
        yield rec.record(seed=11, policy="seated_babble", steps=48)


@needs_assets
def test_restore_and_replay_reproduce_the_recording(recorded: Episode) -> None:
    from sensorium.bodies.restore import RestoredScene  # noqa: PLC0415

    scene = RestoredScene(SETTINGS)
    t0 = 20
    scene.restore(recorded, t0)
    assert np.allclose(scene.recorder.data.qpos, recorded.privileged["qpos"][t0])
    for t in range(t0, t0 + 8):
        scene.step(np.asarray(recorded.action[t]))
    observed = scene.observed()
    assert observed.length == 8
    q_end = planning.arm_q(recorded, np.array([t0 + 8]))[0]
    assert float(planning.rms(scene.arm_q(), q_end)) < 0.02  # the replay gate's premise
    frame = np.abs(observed.rgb[0].astype(float) - recorded.rgb[t0 + 1].astype(float)).mean()
    assert frame < 10.0  # the same scene: appearance, props and layout restored
    assert np.allclose(planning.arm_q(observed, np.array([7]))[0], scene.arm_q(), atol=1e-5)
    scene.close()


def load_plan_script() -> Any:
    """``scripts/plan_test.py`` as a module (it imports ``variety_test`` from ``scripts/``)."""
    import importlib.util  # noqa: PLC0415
    import sys  # noqa: PLC0415

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("plan_test_script", scripts / "plan_test.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(scripts))
    return module


def test_goals_are_sharded_by_block_and_resume_from_any_shard_file(tmp_path: Path) -> None:
    plan = load_plan_script()
    goals = [
        (test, e, Path(f"humanoid-{block}-seated_babble-00{e}.npz"), t0)
        for test in ("babble", "variety")
        for block in ("hand_a", "hand_b", "pincer_a")
        for e in range(2)
        for t0 in (10, 40)
    ]
    shards = [plan.shard_goals(goals, f"{i}/4") for i in range(4)]
    ids = [gid for shard in shards for gid, _ in shard]
    assert sorted(ids) == list(range(len(goals)))  # every goal once, ids unchanged
    blocks = [{plan.block_of(g) for _, g in shard} for shard in shards]
    assert all(not (a & b) for i, a in enumerate(blocks) for b in blocks[i + 1 :])
    for shard in shards:  # one block after the other: a process needs one scene at a time
        order = [plan.block_of(g) for _, g in shard]
        assert order == sorted(order)
    (tmp_path / "0-30.jsonl").write_text(
        json.dumps({"config": "own/A0/0", "goal": 3, "rms": 0.1}) + "\n" + '{"config": "own/A'
    )
    (tmp_path / "5-30.jsonl").write_text(json.dumps({"config": "bar/hold/0", "goal": 7}) + "\n")
    assert plan.done_keys(tmp_path) == {("own/A0/0", 3), ("bar/hold/0", 7)}
    assert plan.peak_rss_mb() > 0


def test_r2_contrast_is_the_paired_difference_of_r2() -> None:
    plan = load_plan_script()
    rng = np.random.default_rng(0)
    q = rng.standard_normal((60, 8))
    groups = np.repeat(np.arange(12), 5)
    a = [q + 0.3 * rng.standard_normal(q.shape) for _ in range(3)]
    b = [q + 0.6 * rng.standard_normal(q.shape) for _ in range(3)]
    c = plan.r2_contrast_values(a, b, q, groups)
    want = np.mean([plan.r2(x, q) - plan.r2(y, q) for x, y in zip(a, b, strict=True)])
    assert c.point == pytest.approx(want) and c.point > 0
    assert plan.r2_contrast_values(a, a, q, groups).point == 0.0
    assert plan.token_features({"s": q, "s10": q, "s2": q, "z": q, "q": q}) == ["s2", "s10"]


def test_gallery_helpers_tile_rank_and_measure_shrinkage() -> None:
    plan = load_plan_script()
    ep = arm_episode(0, steps=200)
    ends = plan.tiled_ends(ep)
    assert ends[0, 0] == plan.first_end(ep) and ends.max() <= ep.length - 1
    assert np.array_equal(ends[1:, 0], ends[:-1, -1])  # each segment starts where the last ended
    q = np.random.default_rng(0).standard_normal((50, 8)) * 4.0 ** np.arange(8)
    assert list(plan.top_variance_joints([q[:25], q[25:]], 3)) == [5, 6, 7]
    true = np.linspace(-1, 1, 21)
    assert plan.slope(0.25 * true + 3, true) == pytest.approx(0.25)


def synthetic_plan(
    plan: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arms: dict[str, dict[str, Any]],
) -> None:
    """Point ``plan`` (``scripts/plan_test.py``) at synthetic data under ``tmp_path``: 6 babble_v2
    and 6 variety episodes with the arm joints (effectors alternate hand and pincer), a split
    with 4 held out per set, and an untrained 10,000-step checkpoint per seed of every arm
    (``ModelConfig`` options on ``TINY``)."""
    from test_model import privileged_episode  # noqa: PLC0415

    from sensorium.model.pretrain import RunConfig, save_checkpoint  # noqa: PLC0415

    names: dict[str, list[str]] = {}
    for test, d in (("babble", "babble_v2"), ("variety", "variety")):
        root = tmp_path / "data" / d / "humanoid"
        root.mkdir(parents=True)
        names[test] = []
        for i in range(6):
            ep = privileged_episode(i + (0 if test == "babble" else 10), steps=96)
            arms_q = np.sin(np.linspace(0, 4, 96)[:, None] + i + np.arange(8)[None] / 8)
            base = [f"{n}_synthetic" if n in planning.ARM_JOINTS else n for n in ep.proprio_names]
            ep.proprio_names = [*base, *planning.ARM_JOINTS]  # each arm joint once
            ep.proprio = np.concatenate([ep.proprio, arms_q], 1).astype(np.float32)
            ep.action_names = [
                *ep.action_names,
                *(n.removeprefix("q_") for n in planning.ARM_JOINTS),
            ]
            ep.action = np.concatenate([ep.action, 0.9 * arms_q], 1).astype(np.float32)
            effector = ("hand", "pincer")[i % 2]
            ep.meta |= {"effector": effector, "seed": i, "policy": "seated_babble"}
            name = f"humanoid-hand_seated_0{i % 2}-seated_babble-00{i}.npz"
            ep.save(root / name)
            names[test].append(name)
    variety_root = tmp_path / "runs/variety"
    variety_root.mkdir(parents=True)
    split = {t: {"heldout": v[:4], "pretrain": v, "spare": []} for t, v in names.items()}
    (variety_root / "split.json").write_text(json.dumps(split))
    monkeypatch.setattr(plan.variety, "ROOT", variety_root)
    monkeypatch.setattr(plan.variety, "DATA", tmp_path / "data")
    eps = [Episode.load(tmp_path / "data/babble_v2/humanoid" / n) for n in names["babble"]]
    space = ChannelSpace.from_headers(header_of(e) for e in eps)
    encoders = {}
    for arm, options in arms.items():
        for seed in plan.SEEDS:
            torch.manual_seed(seed)
            cfg = replace(TINY, senses=SENSES, **options)
            run = tmp_path / "ck" / f"{arm}-s{seed}"
            run.mkdir(parents=True)
            save_checkpoint(
                run / "ckpt-10000.pt", SensoryModel(cfg, space), space, RunConfig(model=cfg), 0
            )
        encoders[arm] = (tmp_path / "ck", arm)
    monkeypatch.setattr(plan, "ENCODERS", encoders)
    monkeypatch.setattr(plan, "ROOT", tmp_path / "plan")


def test_plan_script_fit_openloop_rank_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = load_plan_script()
    sigreg = {"objective": "sigreg"}
    arms = {"A0": sigreg, "recon05": sigreg, "fix1": sigreg, "MAE": {"objective": "mae"}}
    synthetic_plan(plan, tmp_path, monkeypatch, arms)
    monkeypatch.setattr(plan, "CANDIDATES", 4)
    plan.fit()
    assert (tmp_path / "plan/fit/action_stats.json").exists()
    plan.decoder("0/1")
    raw_check = json.loads((tmp_path / "plan/decoder/raw-s0.json").read_text())
    assert raw_check["raw"]["pooled"]["train"] > 0.99  # the window holds the arm joints
    check = json.loads((tmp_path / "plan/decoder/fix1-s2.json").read_text())
    assert set(check) == {"z", "s"} and set(check["s"]) == {"pooled", "per effector"}
    assert {"train", "cv", "babble", "variety", "alpha"} <= set(check["z"]["pooled"])
    assert plan.gallery_goal_ids() == [0, 2, 8]  # first goal of babble 0, babble 1, variety 0
    plan.gallery("openloop")
    plan.gallery("scatter")
    for name in ("openloop", "scatter"):
        assert (tmp_path / f"plan/figures/plan_gallery_{name}.png").exists()
    capsys.readouterr()
    plan.decoder_report()
    assert "| raw proprio window | raw | pooled |" in capsys.readouterr().out
    plan.openloop("0/1")
    raw = np.load(tmp_path / "plan/openloop/own-fix1-s0.npz")
    assert raw["latent"].shape == (8 * 8, 8) and raw["decoded"].shape == (64, 8)
    assert len(plan.goal_list()) == 2 * 8  # two goals per held-out episode, both sets
    plan.rank("0/1")
    assert np.load(tmp_path / "plan/rank/ridge-MAE-s2.npz")["ranks"].max() <= 4
    goals_dir = tmp_path / "plan/goals"  # an incomplete goal stage: a few goals, every config
    goals_dir.mkdir(parents=True)
    lines = [
        json.dumps({"config": "/".join(map(str, c)), "goal": g, "set": "babble", "rms": 0.5})
        for c in plan.configs()
        for g in range(3)
    ]
    lines.append(json.dumps({"config": "bar/hold/0", "goal": 5, "set": "babble", "rms": 0.05}))
    (goals_dir / "0-30.jsonl").write_text("\n".join(lines) + "\n")
    capsys.readouterr()
    plan.report()
    out = capsys.readouterr().out
    assert "Primary (a)" in out and "proxy" in out
    assert "| copy (true arm at the start) |" in out and "| excess, horizon 8 |" in out
    copy, mean = plan.copy_arm()
    assert copy.shape == mean.shape == (64, 8) and np.all(copy >= 0)
    assert "| constant mean arm |" in out
    floor = raw["decoded_true"]  # the floor is the decoder on the true state, not the prediction
    assert floor.shape == raw["decoded"].shape and not np.allclose(floor, raw["decoded"])
    assert "Post hoc planner-failure diagnosis on the 3 goals" in out
    assert "Primary (b)" not in out  # never a primary on incomplete goals
    assert (tmp_path / "plan/figures/plan_openloop.png").exists()
    assert (tmp_path / "plan/figures/plan_floor.png").exists()
    layer = plan.layer("A0", 0)
    assert isinstance(layer, SensoryLayer)
    plan.layer.cache_clear()
    plan.heads.cache_clear()


@needs_assets
def test_rewind_and_the_skipped_right_eye_change_no_result(recorded: Episode) -> None:
    from sensorium.bodies.restore import RestoredScene  # noqa: PLC0415

    def replay(scene: RestoredScene, reuse: bool) -> tuple[np.ndarray, Episode]:
        scene.restore(recorded, 20, reuse=reuse)
        for t in range(20, 28):
            scene.step(np.asarray(recorded.action[t]))
        return scene.recorder.data.qpos.copy(), scene.observed()

    scene = RestoredScene(SETTINGS)
    q_full, ep_full = replay(scene, reuse=False)
    q_rewound, ep_rewound = replay(scene, reuse=True)  # the snapshot of the restore above
    q_again, ep_again = replay(scene, reuse=False)
    assert np.array_equal(q_full, q_rewound) and np.array_equal(q_full, q_again)
    for other in (ep_rewound, ep_again):
        assert np.array_equal(ep_full.rgb, other.rgb)
        assert np.array_equal(ep_full.proprio, other.proprio)
        assert np.array_equal(ep_full.audio, other.audio)
    assert "rgb_right" not in ep_full.privileged  # the second eye is not rendered
    scene.close()
    both = RestoredScene(SETTINGS)
    both.recorder.render_right = True
    q_both, ep_both = replay(both, reuse=False)
    assert np.array_equal(q_full, q_both)  # rendering never touches the physics
    assert ep_both.privileged["rgb_right"].shape == ep_both.rgb.shape
    assert np.abs(ep_both.rgb.astype(int) - ep_full.rgb.astype(int)).max() <= 1
    both.close()
