"""The few-shot harness (``sensorium.fewshot``, ``bodies.h1_closed_loop``) on synthetic data and one
short closed-loop episode in the simulator."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="needs the model extra: make setup")
pytest.importorskip("mujoco", reason="needs the simulator extra: make setup")

from sensorium import fewshot  # noqa: E402
from sensorium.episode import Episode  # noqa: E402
from sensorium.model.data import LoaderConfig, header_of  # noqa: E402
from sensorium.model.layout import ChannelSpace  # noqa: E402
from sensorium.model.sensory_layer import ModelConfig, SensoryLayer, SensoryModel  # noqa: E402

TINY = ModelConfig(
    dim=32,
    fusion_depth=1,
    predictor_depth=1,
    heads=2,
    vision_size=32,
    vision_patch=8,
    vision_dim=32,
    vision_depth=1,
    vision_heads=2,
    vision_tokens=2,
    touch_dim=32,
    touch_depth=1,
    touch_heads=2,
    touch_tokens=2,
    steps=4,
    substeps=4,
    sigreg_proj=64,
    recon_weight=0.0,
    senses=("vision", "proprio_body"),
)


def demo(seed: int, steps: int = 60, success: bool = True) -> Episode:
    """A synthetic demonstration whose actions are a fixed linear map of its proprio."""
    rng = np.random.default_rng(seed)
    proprio = rng.normal(size=(steps, 3)).astype(np.float32)
    action = np.tanh(proprio @ np.array([[0.5, -0.3], [0.2, 0.4], [-0.1, 0.3]])).astype(np.float32)
    return Episode(
        body="humanoid",
        dt=0.05,
        rgb=rng.integers(0, 255, (steps, 256, 256, 3)).astype(np.uint8),
        audio=np.zeros((steps, 800), dtype=np.int16),
        audio_logmel=np.zeros((steps, 32), dtype=np.float16),
        proprio=proprio,
        touch=np.zeros((steps, 0), dtype=np.float32),
        action=action,
        proprio_names=["q_left_elbow", "q_torso", "q_left_shoulder_yaw"],
        touch_names=[],
        action_names=["left_elbow", "torso"],
        text=[""] * steps,
        meta={"success": success},
    )


def proprio_features(episode: Episode, ends: np.ndarray) -> np.ndarray:
    """A toy encoder: the proprio vector at each window end."""
    return np.asarray(np.asarray(episode.proprio)[ends])


@pytest.fixture(scope="module")
def demo_paths(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    root = tmp_path_factory.mktemp("demos")
    paths = []
    for seed in range(6):
        path = root / f"demo-{seed}.npz"
        demo(seed, success=seed != 5).save(path)  # the last one failed
        paths.append(path)
    return paths


def test_the_cache_keeps_successful_demos_with_their_next_action_chunks(
    demo_paths: list[Path], tmp_path: Path
) -> None:
    cache = fewshot.build_cache(demo_paths, proprio_features, horizon=4, substeps=4)
    first = fewshot.first_end(demo(0), 4)
    assert first == 3  # (K - 1) observations, one env step apart at 20 Hz
    assert len(cache.sources) == 5 and cache.horizon == 4  # the failed demonstration is left out
    assert cache.features.shape == (5 * (60 - first), 3)
    ep0 = Episode.load(demo_paths[0])
    rows = np.flatnonzero(cache.episode == 0)
    np.testing.assert_allclose(cache.targets[rows[0]], ep0.action[first : first + 4], atol=1e-6)
    np.testing.assert_allclose(cache.targets[rows[-1]][-1], ep0.action[-1])  # padded at the end
    cache.save(tmp_path / "c.npz")
    back = fewshot.DemoCache.load(tmp_path / "c.npz")
    np.testing.assert_array_equal(back.features, cache.features)
    assert back.sources == cache.sources and back.action_names == ["left_elbow", "torso"]


def test_a_head_learns_the_demonstrated_mapping_and_scores_held_out_demos(
    demo_paths: list[Path], tmp_path: Path
) -> None:
    cache = fewshot.build_cache(demo_paths, proprio_features, horizon=1, substeps=4)
    demos = fewshot.choose_demos(len(cache.sources), 3, seed=0)
    assert demos == fewshot.choose_demos(len(cache.sources), 3, seed=0)
    held = [i for i in range(len(cache.sources)) if i not in demos]
    cfg = fewshot.BCConfig(hidden=64, epochs=150)
    head = fewshot.train_head(cache, demos, cfg, seed=0)
    error = fewshot.action_rmse(head, cache, held)
    spread = float(np.sqrt(np.mean(cache.targets**2)))
    assert error < 0.25 * spread  # the map is learnt and generalises to unseen demonstrations
    fewshot.save_head(head, cfg, tmp_path / "head.pt", {"encoder": "proprio", "k": 3})
    back = fewshot.load_head(tmp_path / "head.pt")
    np.testing.assert_allclose(back.predict(cache.features[:5]), head.predict(cache.features[:5]))
    with pytest.raises(ValueError):
        fewshot.choose_demos(5, 6, seed=0)


def test_a_frozen_layer_gives_features_at_window_ends() -> None:
    episodes = [demo(0), demo(1)]
    space = ChannelSpace.from_headers(header_of(e) for e in episodes)
    layer = SensoryLayer(
        SensoryModel(TINY, space), space, LoaderConfig(steps=4, substeps=4, view_size=32)
    )
    features = fewshot.layer_features(layer)
    ends = np.array([12, 30, 59])
    out = features(episodes[0], ends)
    assert out.shape[0] == 3 and np.isfinite(out).all()
    np.testing.assert_allclose(features(episodes[0], ends[1:]), out[1:], atol=1e-5)  # per window


def test_closed_loop_the_head_drives_the_arms_and_the_program_only_judges() -> None:
    from sensorium.bodies.h1_closed_loop import (
        LearnedPolicy,
        Outcome,
        ReplayPolicy,
        Wrap,
        evaluate,
        evaluate_with,
    )
    from sensorium.bodies.h1_scene import assets_available
    from sensorium.bodies.humanoid import HumanoidRecorder

    if not assets_available():
        pytest.skip("H1 assets missing: run `make h1-assets`")
    settings: dict[str, object] = {
        "effector": "hand",
        "posture": "seated",
        "program": "push_to_target",
        "props": ["target_disc"],
        "task": True,
    }
    steps, first = 30, 12
    with HumanoidRecorder(settings) as rec:
        actions = len(rec.space.names)
        head = fewshot.BCHead(dim=4, horizon=5, actions=actions, hidden=8)
        with torch.no_grad():
            for p in head.parameters():
                p.zero_()
            last = head.net[-1]
            assert isinstance(last, torch.nn.Linear)
            last.bias.fill_(0.1)  # every chunk: 0.1 on every action

        def features(episode: Episode, ends: np.ndarray) -> np.ndarray:
            assert episode.length == first + 1 and not episode.privileged  # model inputs only
            return np.zeros((len(ends), 4), dtype=np.float32)

        wrap = Wrap(lambda judge: LearnedPolicy(judge, rec, features, head, first))
        learned = rec.record(7, "experiment", steps, wrap=wrap)
        scripted = rec.record(7, "experiment", steps)
        outcomes = evaluate(rec, [7], steps, features, head, first)
    assert learned.length == steps and not learned.meta["success"]
    np.testing.assert_allclose(learned.action[first:], 0.1, atol=1e-6)  # the head's chunks
    assert not np.allclose(learned.action[first:], scripted.action[first:])
    assert np.array_equal(learned.action[:first], np.tile(learned.action[0], (first, 1)))  # held
    # the same seed, the same task set-up: objects and target where the demonstrator had them
    np.testing.assert_allclose(
        learned.privileged["object_pose"][0], scripted.privileged["object_pose"][0], atol=1e-6
    )
    assert wrap.policy is not None and wrap.policy.queries == -(-(steps - first) // 5)
    assert outcomes[0].seed == 7 and not outcomes[0].success
    assert outcomes[0].queries == wrap.policy.queries
    assert set(outcomes[0].milestones) == {"reached", "at_target"}  # recorded for split scoring
    assert not outcomes[0].milestones["at_target"]
    # the floors: holding still, and an unrelated demonstration's actions replayed open loop
    with HumanoidRecorder(settings) as rec:
        held = evaluate_with(rec, [7], steps, lambda judge: ReplayPolicy(judge))
        replay = evaluate_with(rec, [8], steps, lambda judge: ReplayPolicy(judge, scripted.action))
        still = rec.record(7, "experiment", steps, wrap=Wrap(lambda judge: ReplayPolicy(judge)))
        # without the cameras: black frames, flagged blind, the same physics and outcome
        rec.render_eyes = False
        seen: list[Outcome] = []
        blind_held = evaluate_with(rec, [7], steps, lambda judge: ReplayPolicy(judge), seen.append)
        blind = rec.record(7, "experiment", steps, wrap=Wrap(lambda judge: ReplayPolicy(judge)))
    assert not held[0].success and not replay[0].success
    assert np.array_equal(still.action, np.tile(still.action[0], (steps, 1)))
    assert seen == blind_held and blind_held[0].success == held[0].success
    assert blind.meta["blind"] and "blind" not in still.meta
    assert not blind.rgb.any() and still.rgb.any()
    np.testing.assert_array_equal(blind.privileged["object_pose"], still.privileged["object_pose"])
    np.testing.assert_array_equal(blind.privileged["qpos"], still.privileged["qpos"])


def test_the_sweep_command_scores_every_k_and_seed(demo_paths: list[Path], tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    cache = fewshot.build_cache(demo_paths, proprio_features, horizon=2, substeps=4)
    cache.save(tmp_path / "cache.npz")
    script = Path(__file__).resolve().parent.parent / "scripts" / "fewshot.py"
    cmd = [sys.executable, str(script), "sweep", str(tmp_path / "cache.npz"), str(tmp_path / "out")]
    subprocess.run(
        [*cmd, "--k", "1", "3", "5", "--seeds", "0", "1"], check=True, capture_output=True
    )
    result = json.loads((tmp_path / "out" / "results.json").read_text())
    assert [(r["k"], r["seed"]) for r in result["rows"]] == [(1, 0), (1, 1), (3, 0), (3, 1)]
    assert set(result["by_k"]) == {"1", "3"}  # k = 5 leaves no demonstration to score
    assert result["by_k"]["3"]["mean"] < result["by_k"]["1"]["mean"]
    assert (tmp_path / "out" / "head_k3_s1.pt").exists()


def test_recalibration_reestimates_batchnorm_on_clean_pool_sequences(
    demo_paths: list[Path], tmp_path: Path
) -> None:
    import importlib.util

    from torch import Tensor

    from sensorium.model.pretrain import RunConfig, save_checkpoint
    from sensorium.model.recalibrate import pool_batches, recalibrate, recalibrated

    episodes = [Episode.load(p) for p in demo_paths]
    space = ChannelSpace.from_headers(header_of(e) for e in episodes)
    torch.manual_seed(0)
    layer = SensoryLayer(
        SensoryModel(TINY, space), space, LoaderConfig(steps=4, substeps=4, view_size=32, crop=1.0)
    )
    batches = pool_batches(layer, demo_paths, batches=3, batch_size=4, seed=0)
    assert len(batches) == 3 and batches[0]["vision"].shape[:2] == (4, 4)
    again = pool_batches(layer, demo_paths, batches=3, batch_size=4, seed=0)
    assert all(torch.equal(a["vision"], b["vision"]) for a, b in zip(batches, again, strict=True))
    seen: list[Tensor] = []
    bn = next(m for m in layer.model.modules() if isinstance(m, torch.nn.BatchNorm1d))
    hook = bn.register_forward_pre_hook(lambda m, a: seen.append(a[0].detach()))
    layer.model.train()
    with torch.no_grad():
        for b in batches:
            layer.model.embed(b)
    hook.remove()
    before = {k: v.clone() for k, v in layer.model.state_dict().items()}
    net = recalibrate(layer.model, batches)
    new_bn = next(m for m in net.modules() if isinstance(m, torch.nn.BatchNorm1d))
    assert new_bn.running_mean is not None
    torch.testing.assert_close(new_bn.running_mean, torch.cat(seen).mean(0), atol=1e-5, rtol=1e-4)
    weights = {
        k: v for k, v in net.state_dict().items() if "running" not in k and "batches" not in k
    }
    assert all(torch.equal(v, before[k]) for k, v in weights.items())  # no weight changes
    assert all(torch.equal(v, layer.model.state_dict()[k]) for k, v in before.items())  # a copy
    assert not net.training and recalibrated(layer, demo_paths).space is space
    with pytest.raises(ValueError):
        pool_batches(layer, [])
    # the command line: --recal-bn on a layer, refused for DINOv2
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, layer.model, space, RunConfig(model=TINY), 0)
    script = Path(__file__).resolve().parent.parent / "scripts" / "fewshot.py"
    spec = importlib.util.spec_from_file_location("fewshot_script", script)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    glob = str(demo_paths[0].parent / "*.npz")
    features, substeps = cli.make_encoder(f"layer:{path}", True, [], glob)
    plain, _ = cli.make_encoder(f"layer:{path}", True, [])
    ends = np.array([12, 30])
    assert substeps == 4 and not np.allclose(features(episodes[0], ends), plain(episodes[0], ends))
    with pytest.raises(ValueError, match="recal-bn"):
        cli.make_encoder("dino:none.pth", True, [], glob)


def test_the_floor_command_runs_one_floor_blind_and_reports_each_episode(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    from sensorium.bodies.h1_scene import assets_available

    if not assets_available():
        pytest.skip("H1 assets not fetched (make h1-assets)")
    root = Path(__file__).resolve().parent.parent
    out = tmp_path / "floor.json"
    cmd = [sys.executable, str(root / "scripts" / "fewshot.py"), "floor", str(out)]
    cmd += ["--config", str(root / "configs" / "tasks_demos.toml"), "--block", "pincer_stack"]
    cmd += ["--demos", "unused", "--episodes", "2", "--steps", "3", "--kinds", "hold"]
    run = subprocess.run([*cmd, "--no-render"], check=True, capture_output=True, text=True)
    assert run.stdout.count("pincer_stack hold episode seed ") == 2  # one line per episode
    result = json.loads(out.read_text())
    assert result["rendered"] is False and "replay" not in result
    assert result["hold"]["n"] == 2


def test_closed_loop_speed_options_leave_the_model_inputs_and_the_outcome_unchanged() -> None:
    from sensorium.bodies.h1_closed_loop import LearnedPolicy, Wrap
    from sensorium.bodies.h1_scene import assets_available
    from sensorium.bodies.humanoid import HumanoidRecorder

    if not assets_available():
        pytest.skip("H1 assets missing: run `make h1-assets`")
    settings: dict[str, object] = {
        "effector": "hand",
        "posture": "seated",
        "program": "push_to_target",
        "props": ["target_disc"],
        "task": True,
    }
    steps, first = 24, 6

    def run(right: bool, skip: frozenset[str]) -> tuple[list[Episode], Episode]:
        seen: list[Episode] = []

        def features(episode: Episode, ends: np.ndarray) -> np.ndarray:
            seen.append(episode)
            return np.full((len(ends), 4), float(episode.proprio[-1].sum()), dtype=np.float32)

        with HumanoidRecorder(settings) as rec:
            rec.render_right, rec.skip_senses = right, skip
            torch.manual_seed(0)
            head = fewshot.BCHead(dim=4, horizon=3, actions=len(rec.space.names), hidden=8)
            wrap = Wrap(lambda judge: LearnedPolicy(judge, rec, features, head, first))
            return seen, rec.record(7, "experiment", steps, wrap=wrap)

    base_seen, base = run(True, frozenset())
    eye_seen, eye = run(False, frozenset())
    lean_seen, lean = run(False, frozenset({"touch", "audio"}))
    assert "rgb_right" in base.privileged and "rgb_right" not in eye.privileged
    assert len(base_seen) == len(eye_seen) == len(lean_seen) > 1
    for a, b, c in zip(base_seen, eye_seen, lean_seen, strict=True):
        assert np.array_equal(a.rgb, b.rgb) and np.array_equal(a.rgb, c.rgb)  # what V reads
        assert np.array_equal(a.proprio, b.proprio) and np.array_equal(a.proprio, c.proprio)
        assert np.array_equal(a.touch, b.touch) and np.array_equal(a.audio, b.audio)
        assert not c.touch.any() and not c.audio.any()  # skipped: zeros of the usual width
        assert c.touch.shape == a.touch.shape and c.audio.shape == a.audio.shape
    for other in (eye, lean):  # the same physics, actions and judging
        np.testing.assert_array_equal(other.action, base.action)
        np.testing.assert_array_equal(other.privileged["qpos"], base.privileged["qpos"])
        assert other.meta["success"] == base.meta["success"]
        assert other.meta.get("milestones") == base.meta.get("milestones")


def test_the_step_cap_comes_from_the_demonstrations_success_steps(tmp_path: Path) -> None:
    import importlib.util
    import json

    script = Path(__file__).resolve().parent.parent / "scripts" / "fewshot.py"
    spec = importlib.util.spec_from_file_location("fewshot_script_cap", script)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    rows = [{"file": f"e{i}.npz", "steps": 100 + i, "end_reason": "success"} for i in range(20)]
    rows += [{"file": "f.npz", "steps": 600, "end_reason": "steps"}]  # a failure: not counted
    rows += [{"file": "e0.npz", "steps": 300, "end_reason": "success"}]  # a later line wins
    part = tmp_path / "manifest.parts" / "block.jsonl"
    part.parent.mkdir()
    part.write_text("\n".join(json.dumps(r) for r in rows))
    steps = cli.manifest_success_steps([part])
    assert sorted(steps) == sorted([300, *range(101, 120)])
    want = int(np.ceil(1.5 * np.percentile(np.asarray(steps, dtype=float), 95)))
    assert cli.success_cap(steps) == want
    whole = tmp_path / "manifest.json"
    whole.write_text(json.dumps({"episodes": rows[:3]}))
    assert cli.manifest_success_steps([whole]) == [100, 101, 102]
    with pytest.raises(ValueError):
        cli.success_cap([])


def test_state_features_are_proprio_and_the_privileged_state_in_order() -> None:
    ep = demo(0, steps=20)
    rng = np.random.default_rng(1)
    ep.privileged = {
        "target": rng.normal(size=(20, 3)).astype(np.float32),
        "object_pose": rng.normal(size=(20, 2, 7)).astype(np.float32),
        "tip_position": rng.normal(size=(20, 2, 3)).astype(np.float32),
        "qpos": rng.normal(size=(20, 5)).astype(np.float32),  # not a state key: left out
    }
    ends = np.array([3, 11])
    out = fewshot.state_features(ep, ends)
    assert out.shape == (2, 3 + 14 + 6 + 3)
    want = np.concatenate(
        [ep.proprio[ends], ep.privileged["object_pose"][ends].reshape(2, -1)]
        + [ep.privileged["tip_position"][ends].reshape(2, -1), ep.privileged["target"][ends]],
        axis=1,
    )
    np.testing.assert_allclose(out, want)
    with pytest.raises(ValueError, match="privileged"):
        fewshot.state_features(demo(0), ends)


def test_the_oracle_sees_in_closed_loop_the_state_a_saved_episode_has() -> None:
    from sensorium.bodies.h1_closed_loop import (
        LearnedPolicy,
        Wrap,
        evaluate,
        evaluate_program,
    )
    from sensorium.bodies.h1_scene import assets_available
    from sensorium.bodies.humanoid import HumanoidRecorder

    if not assets_available():
        pytest.skip("H1 assets missing: run `make h1-assets`")
    settings: dict[str, object] = {
        "effector": "hand",
        "posture": "seated",
        "program": "push_to_target",
        "props": ["target_disc"],
        "task": True,
    }
    steps, first = 20, 6
    seen: list[np.ndarray] = []

    def features(episode: Episode, ends: np.ndarray) -> np.ndarray:
        out = fewshot.state_features(episode, ends)
        seen.append(out)
        return out

    with HumanoidRecorder(settings) as rec:
        dim = fewshot.state_features(rec.record(7, "experiment", 2), np.array([0])).shape[1]
        head = fewshot.BCHead(dim=dim, horizon=4, actions=len(rec.space.names), hidden=8)
        wrap = Wrap(lambda judge: LearnedPolicy(judge, rec, features, head, first, True))
        learned = rec.record(7, "experiment", steps, wrap=wrap)
        scripted = rec.record(7, "experiment", steps)
        oracle = evaluate(rec, [7], steps, features, head, first, privileged=True)
        program = evaluate_program(rec, [7], steps)
        rec.render_eyes = False  # the oracle reads no frame: blind, the same features and outcome
        seen_rendered = list(seen)
        blind = evaluate(rec, [7], steps, features, head, first, privileged=True)
    # the first query saw step ``first``: the same state as the saved episode at that step
    saved = fewshot.state_features(learned, np.array([first]))
    np.testing.assert_allclose(seen[0], saved, atol=1e-6)
    assert "target" in learned.privileged and seen[0].shape[1] == dim
    assert oracle[0].queries == -(-(steps - first) // 4)
    assert program[0].queries == 0 and program[0].steps == scripted.length
    assert program[0].success == bool(scripted.meta["success"])
    assert program[0].milestones == scripted.meta.get("milestones", {})
    assert blind == oracle
    np.testing.assert_allclose(seen[len(seen_rendered)], seen_rendered[-oracle[0].queries])


def test_the_expert_command_runs_the_program_blind_and_reports_each_episode(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    from sensorium.bodies.h1_scene import assets_available

    if not assets_available():
        pytest.skip("H1 assets not fetched (make h1-assets)")
    root = Path(__file__).resolve().parent.parent
    out = tmp_path / "expert.json"
    cmd = [sys.executable, str(root / "scripts" / "fewshot.py"), "expert", str(out)]
    cmd += ["--config", str(root / "configs" / "tasks_demos.toml"), "--block", "pincer_stack"]
    cmd += ["--episodes", "2", "--max-steps", "3", "--no-render"]
    run = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert run.stdout.count("pincer_stack expert episode seed ") == 2
    result = json.loads(out.read_text())
    assert result["rendered"] is False and result["n"] == 2 and result["max_steps"] == 3


def test_divergence_is_nil_for_an_episode_and_itself_and_finds_the_first_drift() -> None:
    from sensorium.bodies.h1_closed_loop import divergence

    ep = demo(0, steps=30)
    rng = np.random.default_rng(2)
    ep.privileged = {
        "qpos": rng.normal(size=(30, 6)).astype(np.float32),
        "object_pose": rng.normal(size=(30, 2, 7)).astype(np.float32),
    }
    same = divergence(ep, ep)
    assert same["first_step"] is None and same["qpos_max"] == 0.0 and same["pos_max"] == 0.0
    moved = demo(0, steps=30)
    moved.privileged = {k: v.copy() for k, v in ep.privileged.items()}
    moved.privileged["object_pose"][12:, 1, 0] += 0.01  # a prop slides from step 12
    d = divergence(ep, moved)
    assert d["first_step"] == 12 and d["pos_max"] == pytest.approx(0.01, abs=1e-6)
    at = d["at"]
    assert d["qpos_max"] == 0.0 and isinstance(at, dict) and at["10"] == [0.0, 0.0]


def test_a_demonstration_replays_its_own_actions_into_the_same_episode(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys
    import tomllib

    from sensorium.bodies.h1_closed_loop import self_replay
    from sensorium.bodies.h1_scene import assets_available
    from sensorium.bodies.humanoid import HumanoidRecorder

    if not assets_available():
        pytest.skip("H1 assets missing: run `make h1-assets`")
    root = Path(__file__).resolve().parent.parent
    config = root / "configs" / "tasks_demos.toml"
    settings = tomllib.loads(config.read_text())["humanoid"]["pincer_stack"]["settings"]
    steps = 12
    with HumanoidRecorder(dict(settings)) as rec:
        rec.render_eyes = False
        recorded = rec.record(11, "experiment", steps)
        outcome, drift = self_replay(rec, recorded, steps)
    assert outcome.seed == 11 and outcome.steps == recorded.length
    assert outcome.success == bool(recorded.meta["success"])
    assert drift["first_step"] is None, drift  # deterministic, actions logged at the applied step
    path = tmp_path / "demo-0.npz"
    recorded.save(path)
    out = tmp_path / "self.json"
    cmd = [sys.executable, str(root / "scripts" / "fewshot.py"), "floor", str(out)]
    cmd += ["--config", str(config), "--block", "pincer_stack", "--demos", str(path)]
    cmd += ["--steps", str(steps), "--kinds", "self", "--no-render"]
    run = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert run.stdout.count("pincer_stack self episode seed 11") == 1
    result = json.loads(out.read_text())["self"]
    assert result["n"] == 1 and result["replays"][0]["demo"] == "demo-0.npz"
    assert result["replays"][0]["first_step"] is None


def test_a_replay_of_an_episodes_own_actions_is_bit_exact() -> None:
    import tomllib

    from sensorium.bodies.h1_closed_loop import self_replay
    from sensorium.bodies.h1_scene import assets_available
    from sensorium.bodies.humanoid import HumanoidRecorder

    if not assets_available():
        pytest.skip("H1 assets missing: run `make h1-assets`")
    root = Path(__file__).resolve().parent.parent
    config = tomllib.loads((root / "configs" / "tasks_demos.toml").read_text())
    steps = 60
    with HumanoidRecorder(dict(config["humanoid"]["pincer_stack"]["settings"])) as rec:
        rec.render_eyes = False
        recorded = rec.record(11, "experiment", steps)
        _, drift = self_replay(rec, recorded, steps)
    # the applied action is the logged float32 one: no drift at all, not just below tolerance
    assert drift["qpos_max"] == 0.0 and drift["pos_max"] == 0.0, drift


class _CountingHead:
    """A stand-in head: query ``q`` (from 0) predicts ``q`` for every step of its chunk."""

    horizon = 4

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, features: np.ndarray) -> np.ndarray:
        out = np.full((1, self.horizon, 2), float(self.calls), dtype=np.float32)
        self.calls += 1
        return out


class _Recent:
    """A stand-in recorder whose recent episode is a one-step synthetic demonstration."""

    def recent_episode(self, log: object, steps: int, privileged: bool = False) -> Episode:
        return demo(0, steps=1)


def _policy(**options: object) -> Any:
    from sensorium.bodies.h1_closed_loop import LearnedPolicy

    policy = LearnedPolicy(
        judge=None,  # type: ignore[arg-type]
        recorder=_Recent(),  # type: ignore[arg-type]
        features=proprio_features,
        head=_CountingHead(),  # type: ignore[arg-type]
        first=2,
        **options,  # type: ignore[arg-type]
    )
    policy.log = []  # type: ignore[assignment]
    policy.hold = np.zeros(2, dtype=np.float32)
    return policy


def test_chunks_are_played_in_full_by_default_and_replaced_when_queried_more_often() -> None:
    chunked = _policy()
    played = [float(chunked.act(t, None)[0]) for t in range(2, 12)]
    assert played == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2] and chunked.queries == 3
    every = _policy(query_every=1)
    played = [float(every.act(t, None)[0]) for t in range(2, 8)]
    assert played == [0, 1, 2, 3, 4, 5] and every.queries == 6
    assert float(chunked.act(0, None)[0]) == 0.0  # held before the first window


def test_temporal_ensembling_averages_the_chunks_covering_each_step() -> None:
    from sensorium.bodies.h1_closed_loop import ENSEMBLE_M

    policy = _policy(query_every=1, ensemble=True)
    played = [float(policy.act(t, None)[0]) for t in range(2, 9)]
    for i, value in enumerate(played):
        live = np.arange(max(0, i - 3), i + 1)  # the queries whose 4-step chunks cover this step
        w = np.exp(-ENSEMBLE_M * np.arange(len(live)))  # oldest first, weighted most
        assert value == pytest.approx(float(w @ live / w.sum()), abs=1e-5)
    assert len(policy.chunks) == 4


def test_dart_noise_perturbs_the_executed_action_and_logs_the_clean_one(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys
    import tomllib

    from sensorium.bodies.h1_scene import assets_available
    from sensorium.bodies.humanoid import HumanoidRecorder

    if not assets_available():
        pytest.skip("H1 assets missing: run `make h1-assets`")
    root = Path(__file__).resolve().parent.parent
    config = root / "configs" / "tasks_demos.toml"
    settings = dict(tomllib.loads(config.read_text())["humanoid"]["pincer_stack"]["settings"])
    steps = 20
    episodes = []
    for noise in (0.0, 0.05, 0.05):
        with HumanoidRecorder({**settings, "action_noise": noise}) as rec:
            rec.render_eyes = False
            episodes.append(rec.record(11, "experiment", steps))
    clean, noisy, again = episodes
    assert "executed_action" not in clean.privileged
    executed = noisy.privileged["executed_action"]
    assert executed.shape == noisy.action.shape and np.abs(executed).max() <= 1.0
    gap = executed - noisy.action  # the logged label is the clean action; the noise was applied
    assert 0.02 < float(gap.std()) < 0.08
    np.testing.assert_array_equal(executed, again.privileged["executed_action"])  # seeded
    np.testing.assert_array_equal(noisy.privileged["qpos"], again.privileged["qpos"])
    assert not np.allclose(noisy.privileged["qpos"][-1], clean.privileged["qpos"][-1])
    out = tmp_path / "dart"
    cmd = [sys.executable, str(root / "scripts" / "fewshot.py"), "dart", str(out)]
    cmd += ["--config", str(config), "--block", "pincer_stack", "--noise", "0.03"]
    cmd += ["--first", "3", "--episodes", "2", "--steps", "5"]
    run = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert run.stdout.count("pincer_stack dart ") == 2
    saved = Episode.load(out / "pincer_stack-dart-00004.npz")
    assert saved.privileged["executed_action"].shape == saved.action.shape
    assert (
        json.loads(json.dumps(saved.meta))["seed"]
        != Episode.load(out / "pincer_stack-dart-00003.npz").meta["seed"]
    )
