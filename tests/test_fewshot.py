"""The few-shot harness (``sensorium.fewshot``, ``bodies.h1_closed_loop``) on synthetic data and one
short closed-loop episode in the simulator."""

from pathlib import Path

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
    assert not held[0].success and not replay[0].success
    assert np.array_equal(still.action, np.tile(still.action[0], (steps, 1)))


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
