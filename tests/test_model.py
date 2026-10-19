"""The sensory layer v0: channel space, loader, stems, losses and a few training steps.

Synthetic H1-like episodes (hand and pincer) are built in memory, so nothing here needs
generated data, the simulators or a GPU. A final test uses the generated H1 sample when
``data/sample_h1`` exists.
"""

import json
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="needs the model extra: uv sync --extra model")

from torch import Tensor, nn  # noqa: E402

import sensorium.model as package  # noqa: E402
from sensorium.episode import Episode  # noqa: E402
from sensorium.model import probes, subsets  # noqa: E402
from sensorium.model.data import (  # noqa: E402
    LoaderConfig,
    SequenceDataset,
    header_of,
    load_episode,
    sequence_from_episode,
    spacing,
)
from sensorium.model.diagnostics import heldout_loss, sense_reliance  # noqa: E402
from sensorium.model.gpu_smoke import (  # noqa: E402
    Sampler,
    parse_smi,
    plan_hours,
    smoke,
)
from sensorium.model.layout import (  # noqa: E402
    ChannelSpace,
    check_senses,
    read_header,
    region_path,
)
from sensorium.model.modules import SIGReg  # noqa: E402
from sensorium.model.pretrain import (  # noqa: E402
    RunConfig,
    TrainConfig,
    ema_at,
    load_layer,
    parse_config,
    pick_device,
    save_checkpoint,
    senses_for_pool,
    train,
)
from sensorium.model.sensory_layer import (  # noqa: E402
    ModelConfig,
    SensoryLayer,
    SensoryModel,
    collapse_metrics,
    count_parameters,
    surprise_weights,
    vicreg_terms,
)

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
)
LOADER = LoaderConfig(steps=4, substeps=4, view_size=32)


class Tripwire(dict[str, np.ndarray]):
    """A ``privileged`` dict that fails the test if anything reads it."""

    def __getitem__(self, key: str) -> np.ndarray:
        raise AssertionError("the loader read privileged state")

    def __iter__(self):  # type: ignore[no-untyped-def]
        raise AssertionError("the loader read privileged state")

    def items(self):  # type: ignore[no-untyped-def]
        raise AssertionError("the loader read privileged state")


class TextTripwire(list[str]):
    """A ``text`` list that fails the test if anything reads it."""

    def __getitem__(self, index):  # type: ignore[no-untyped-def]
        raise AssertionError("the loader read text")

    def __iter__(self):  # type: ignore[no-untyped-def]
        raise AssertionError("the loader read text")


def touch_block(prefix: str, taxels: int, dyn_path: str) -> list[str]:
    """Taxel names ``prefix/i`` and the 3 dynamics names of the region."""
    return [f"{prefix}/{i}" for i in range(taxels)] + [
        f"dynamics/{dyn_path}/{stat}" for stat in ("onset", "slip", "vibration")
    ]


def make_episode(
    effector: str = "hand", steps: int = 80, seed: int = 0, body: str = "humanoid"
) -> Episode:
    """A synthetic H1-like episode with smooth signals; ``effector`` is hand, pincer or none."""
    rng = np.random.default_rng(seed)
    body_proprio = ["q_left_elbow", "dq_left_elbow", "q_torso", "dq_torso"]
    vestibular = ["vest_acc_x", "gravity_z", "pelvis_height"]
    touch = touch_block("body/left_elbow", 4, "body/left_elbow_link") + touch_block(
        "body/torso", 2, "body/torso_link"
    )
    action = ["left_elbow", "torso"]
    eff_proprio: list[str] = []
    if effector == "hand":
        eff_proprio = ["q_lh_FFJ1", "dq_lh_FFJ1"]
        touch += touch_block("hand/left/palm", 6, "hand/left/palm")
        touch += touch_block("hand/left/ffdistal", 3, "hand/left/ffdistal")
        action += ["lh_A_FFJ3"]
    elif effector == "pincer":
        eff_proprio = ["q_lp_left_driver_joint", "dq_lp_left_driver_joint"]
        touch += touch_block("pincer/left/left_pad", 4, "pincer/left/left_pad")
        action += ["lp_fingers_actuator"]
    proprio_names = body_proprio + vestibular + eff_proprio
    phase = np.linspace(0, 6, steps)[:, None] + rng.uniform(0, 1, (1, len(proprio_names)))
    base = rng.integers(0, 255, (1, 256, 256, 3))
    rgb = np.clip(base + 10 * np.sin(phase[:, :1, None, None]), 0, 255).astype(np.uint8)
    contact = (rng.random((steps, len(touch))) < 0.1) * rng.random((steps, len(touch)))
    return Episode(
        body=body,
        dt=0.05,
        rgb=rgb,
        audio=np.zeros((steps, 800), dtype=np.int16),
        audio_logmel=rng.normal(size=(steps, 32)).astype(np.float16),
        proprio=np.sin(phase).astype(np.float32),
        touch=contact.astype(np.float32),
        action=np.sin(phase[:, : len(action)] * 1.3).astype(np.float32),
        proprio_names=proprio_names,
        touch_names=touch,
        action_names=action,
        privileged=Tripwire(),
        text=TextTripwire(),
    )


@pytest.fixture(scope="module")
def episodes() -> dict[str, Episode]:
    return {"hand": make_episode("hand", seed=1), "pincer": make_episode("pincer", seed=2)}


@pytest.fixture(scope="module")
def space(episodes: dict[str, Episode]) -> ChannelSpace:
    return ChannelSpace.from_headers(header_of(e) for e in episodes.values())


def batch_of(
    episodes: dict[str, Episode], space: ChannelSpace, senses: tuple[str, ...], seed: int = 0
) -> dict[str, Tensor]:
    rng = np.random.default_rng(seed)
    seqs = [
        sequence_from_episode(load_episode(e, space, senses), space, senses, LOADER, rng)
        for e in episodes.values()
    ]
    return {k: torch.as_tensor(np.stack([s[k] for s in seqs])) for k in seqs[0]}


def make_car(steps: int = 80, seed: int = 5) -> Episode:
    """A skinless, effector-less body at 10 Hz with its own channel names."""
    rng = np.random.default_rng(seed)
    return Episode(
        body="car",
        dt=0.1,
        rgb=rng.integers(0, 255, (steps, 256, 256, 3)).astype(np.uint8),
        audio=np.zeros((steps, 1600), dtype=np.int16),
        audio_logmel=rng.normal(size=(steps, 32)).astype(np.float16),
        proprio=rng.normal(size=(steps, 3)).astype(np.float32),
        touch=np.zeros((steps, 0), dtype=np.float32),
        action=rng.uniform(-1, 1, (steps, 2)).astype(np.float32),
        proprio_names=["speed", "heading", "yaw_rate"],
        touch_names=[],
        action_names=["steer", "throttle"],
    )


def test_region_path_joins_body_dynamics_to_their_taxels() -> None:
    assert region_path("body/left_hip_yaw/3") == region_path("dynamics/body/left_hip_yaw_link/slip")
    assert region_path("hand/left/palm/7") == ["hand", "left", "palm"]
    assert region_path("dynamics/pincer/left/left_pad/onset") == ["pincer", "left", "left_pad"]
    assert region_path("right_thumb") == ["right_thumb"]


def test_channel_space_shares_body_channels_between_hand_and_pincer(
    episodes: dict[str, Episode], space: ChannelSpace
) -> None:
    hand, pincer = (space.columns(header_of(e)) for e in episodes.values())
    assert set(space.regions) == {
        "body/left_elbow",
        "body/torso",
        "hand/left/palm",
        "hand/left/ffdistal",
        "pincer/left/left_pad",
    }
    assert hand.region_mask.sum() == 4 and pincer.region_mask.sum() == 3
    body_hand = set(hand.dst["touch"][: 4 + 3 + 2 + 3])  # elbow, torso taxels + dynamics
    body_pincer = set(pincer.dst["touch"][: 4 + 3 + 2 + 3])
    assert body_hand == body_pincer
    assert not set(hand.dst["proprio_effector"]) & set(pincer.dst["proprio_effector"])
    assert len(space.keys["vestibular"]) == 3


def test_batch_shapes_for_hands_and_pincers(
    episodes: dict[str, Episode], space: ChannelSpace
) -> None:
    senses = check_senses(TINY.senses)
    batch = batch_of(episodes, space, senses)
    assert batch["vision"].shape == (2, 4, 32, 32, 3) and batch["vision"].dtype == torch.uint8
    assert batch["touch"].shape == (2, 4, 4, space.size("touch"))
    assert batch["audio"].shape == (2, 4, 4, 32)
    assert batch["action"].shape == (2, 4, 4, space.size("action"))
    assert batch["present"].shape == (2, 6) and batch["region_mask"].shape == (
        2,
        len(space.regions),
    )
    assert batch["present"].all()
    out = SensoryModel(TINY, space)(batch)
    assert out["states"].shape == (2, 4, 32)
    assert all(torch.isfinite(out[k]) for k in ("loss", "pred", "reg"))


def test_absent_senses_create_no_tokens(episodes: dict[str, Episode], space: ChannelSpace) -> None:
    cfg = replace(TINY, senses=("vision", "audio"))
    model = SensoryModel(cfg, space)
    assert model.n_tokens == {"vision": 2, "audio": 1}
    assert model.encoder.slot.shape[1] == 1 + 2 + 1
    assert set(model.encoder.stems) == {"vision", "audio"}
    assert not any("touch" in name for name, _ in model.named_parameters())
    batch = batch_of(episodes, space, cfg.senses)
    assert "touch" not in batch and "proprio_body" not in batch
    assert model(batch)["states"].shape == (2, 4, 32)


def test_missing_sense_in_an_episode_is_masked(
    space: ChannelSpace, episodes: dict[str, Episode]
) -> None:
    senses = check_senses(TINY.senses)
    model = SensoryModel(TINY, space).eval()
    batch = batch_of(episodes, space, senses)
    batch["present"][:, senses.index("touch")] = False
    other = {
        **batch,
        "touch": torch.rand_like(batch["touch"]),
        "region_mask": ~batch["region_mask"],
    }
    assert torch.allclose(model.embed(batch), model.embed(other), atol=1e-5)


def test_loader_never_reads_privileged_state_or_text(
    episodes: dict[str, Episode], space: ChannelSpace
) -> None:
    ep = episodes["hand"]  # privileged and text are tripwires
    senses = check_senses(TINY.senses)
    loaded = load_episode(ep, space, senses)
    sequence_from_episode(loaded, space, senses, LOADER, np.random.default_rng(0))
    SensoryLayer(SensoryModel(TINY, space), space, LOADER).encode(ep, stride=8)
    assert set(loaded.inputs) <= {"rgb", "audio_logmel", "proprio", "touch", "action"}


def test_sequences_are_deterministic_and_use_the_window_convention(
    episodes: dict[str, Episode], space: ChannelSpace
) -> None:
    senses = check_senses(TINY.senses)
    loaded = load_episode(episodes["hand"], space, senses)
    a = sequence_from_episode(
        loaded, space, senses, replace(LOADER, touch_noise=False), np.random.default_rng(3)
    )
    b = sequence_from_episode(
        loaded, space, senses, replace(LOADER, touch_noise=False), np.random.default_rng(3)
    )
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert spacing(0.05) == 1 and spacing(0.025) == 2 and spacing(0.1) == 1


def test_sigreg_is_low_for_gaussian_and_high_for_collapsed_embeddings() -> None:
    torch.manual_seed(0)
    reg = SIGReg(num_proj=256)
    gaussian = torch.randn(128, 4, 16)
    collapsed = torch.zeros(128, 4, 16) + 0.01 * torch.randn(128, 4, 16)
    rank_one = torch.randn(128, 4, 1).expand(-1, -1, 16).contiguous()
    low, high, mid = reg(gaussian), reg(collapsed), reg(rank_one)
    # An isotropic Gaussian scores about 1 whatever the batch; collapse scores ~B/2.
    assert low < 1.5 and high > 20 * low and mid > 10 * low
    assert reg(gaussian, pool_time=True) < 1.5


def test_collapse_metrics_tell_isotropic_from_collapsed() -> None:
    torch.manual_seed(0)
    iso = collapse_metrics(torch.randn(8, 16, 16))
    flat = collapse_metrics(torch.randn(8, 16, 1).expand(-1, -1, 16).contiguous())
    assert iso["erank"] > 12 and flat["erank"] < 1.5


def test_sense_dropout_never_drops_every_sense(space: ChannelSpace) -> None:
    model = SensoryModel(replace(TINY, sense_dropout=0.9), space).train()
    keep = model.encoder.keep_mask(torch.ones(64, 6, dtype=torch.bool), True)
    assert keep.any(-1).all() and not keep.all()
    assert model.encoder.keep_mask(torch.ones(4, 6, dtype=torch.bool), False).all()


def test_same_seed_gives_the_same_training_loss(
    tmp_path: Path, episodes: dict[str, Episode], space: ChannelSpace
) -> None:
    losses = []
    for _ in range(2):
        torch.manual_seed(0)
        model = SensoryModel(TINY, space)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for step in range(3):
            out = model(batch_of(episodes, space, check_senses(TINY.senses), seed=step))
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
        losses.append(float(out["loss"].detach()))
    assert losses[0] == losses[1]


def test_a_few_training_steps_reduce_the_loss(
    episodes: dict[str, Episode], space: ChannelSpace
) -> None:
    torch.manual_seed(0)
    batch = batch_of(episodes, space, check_senses(TINY.senses))
    model = SensoryModel(replace(TINY, sigreg_pool_time=True), space)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(30):
        out = model(batch)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        losses.append(float(out["loss"].detach()))
    assert np.mean(losses[-3:]) < 0.5 * np.mean(losses[:3])


def test_encode_returns_one_state_per_window(
    episodes: dict[str, Episode], space: ChannelSpace
) -> None:
    layer = SensoryLayer(SensoryModel(TINY, space), space, LOADER)
    states, ends = layer.encode(episodes["pincer"], stride=4)
    assert states.shape == (len(ends), 32) and ends[0] == 3 and np.isfinite(states).all()
    with pytest.raises(ValueError, match="shorter than one window"):
        layer.encode(make_episode("hand", steps=3), stride=1)


def test_config_rejects_unknown_keys_and_pool_restricts_senses() -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        parse_config({"model": {"dimm": 3}})
    with pytest.raises(ValueError, match="distinct members"):
        parse_config({"model": {"senses": ["vision", "smell"]}})
    cfg = parse_config(
        {"name": "x", "model": {"senses": ["vision", "touch"]}, "train": {"steps": 5}}
    )
    assert cfg.model.senses == ("vision", "touch") and cfg.train.steps == 5
    assert senses_for_pool(("vision", "touch", "audio"), ("rgb", "audio_logmel", "action")) == (
        "vision",
        "audio",
    )


SAMPLE = Path("data/sample_h1/humanoid")


@pytest.mark.skipif(not SAMPLE.exists(), reason="needs the H1 sample: make sample-h1")
def test_cpu_run_on_the_h1_sample(tmp_path: Path) -> None:
    cfg = RunConfig(
        name="t",
        data=parse_config({"data": {"paths": [str(SAMPLE / "*.npz")]}}).data,
        model=replace(TINY, steps=8),
        train=TrainConfig(
            steps=3, batch_size=2, out_dir=str(tmp_path), log_every=1, save_steps=(2,)
        ),
    )
    metrics = train(cfg)
    assert np.isfinite(metrics["loss"]) and (tmp_path / "t" / "ckpt.pt").exists()
    assert (tmp_path / "t" / "ckpt-2.pt").exists()
    layer = load_layer(tmp_path / "t" / "ckpt.pt")
    states, _ = layer.encode(Episode.load(sorted(SAMPLE.glob("*pincer*.npz"))[0]), stride=16)
    assert states.shape[1] == TINY.dim and np.isfinite(states).all()


def test_a_body_without_skin_trains_in_the_same_batch_as_the_humanoid(
    episodes: dict[str, Episode],
) -> None:
    mixed = {"hand": episodes["hand"], "car": make_car()}
    space = ChannelSpace.from_headers(header_of(e) for e in mixed.values())
    senses = check_senses(TINY.senses)
    batch = batch_of(mixed, space, senses)
    assert batch["present"][1].tolist() == [True, False, True, True, False, False]
    assert not batch["region_mask"][1].any() and batch["region_mask"][0].any()
    out = SensoryModel(TINY, space)(batch)
    assert torch.isfinite(out["loss"]) and out["states"].shape == (2, 4, 32)
    out["loss"].backward()


def test_checkpoint_round_trip_reproduces_the_states(
    tmp_path: Path, episodes: dict[str, Episode], space: ChannelSpace
) -> None:
    torch.manual_seed(0)
    model = SensoryModel(TINY, space)
    save_checkpoint(tmp_path / "c.pt", model, space, RunConfig(model=TINY), 0)
    layer = load_layer(tmp_path / "c.pt")
    ref = SensoryLayer(model, space, LOADER)
    a, _ = ref.encode(episodes["hand"], stride=8)
    b, _ = layer.encode(episodes["hand"], stride=8)
    assert np.allclose(a, b, atol=1e-5)


EMA = replace(TINY, objective="ema_vicreg")


def test_vicreg_terms_separate_healthy_from_collapsed_states() -> None:
    torch.manual_seed(0)
    healthy = torch.randn(64, 4, 8)
    var, cov = vicreg_terms(healthy)
    assert var < 0.05 and cov < 0.05
    var, cov = vicreg_terms(0.1 * torch.randn(64, 4, 8))
    assert var > 0.8
    shared = torch.randn(64, 4, 1).expand(-1, -1, 8).contiguous() + 0.01 * torch.randn(64, 4, 8)
    assert vicreg_terms(shared)[1] > 0.5


def test_the_ema_target_is_a_frozen_copy_that_follows_the_encoder(
    episodes: dict[str, Episode], space: ChannelSpace
) -> None:
    model = SensoryModel(EMA, space)
    assert model.target is not None and not any(p.requires_grad for p in model.target.parameters())
    assert SensoryModel(TINY, space).target is None
    batch = batch_of(episodes, space, check_senses(EMA.senses))
    out = model(batch)
    assert {"loss", "pred", "reg", "reg_var", "reg_cov", "states"} <= set(out)
    out["loss"].backward()
    assert all(p.grad is None for p in model.target.parameters())
    assert any(p.grad is not None for p in model.encoder.parameters())
    before = [p.clone() for p in model.target.parameters()]
    model.update_target(1.0)
    assert all(torch.equal(a, b) for a, b in zip(before, model.target.parameters(), strict=True))
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(1.0)
    model.update_target(0.0)
    assert all(
        torch.equal(a, b)
        for a, b in zip(model.encoder.parameters(), model.target.parameters(), strict=True)
    )
    assert count_parameters(model)["total"] == count_parameters(SensoryModel(TINY, space))["total"]


def test_the_ema_objective_reduces_the_loss(
    episodes: dict[str, Episode], space: ChannelSpace
) -> None:
    torch.manual_seed(0)
    batch = batch_of(episodes, space, check_senses(EMA.senses))
    model = SensoryModel(EMA, space)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    losses = []
    for step in range(30):
        out = model(batch)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        model.update_target(ema_at(step + 1, EMA, 30))
        losses.append(float(out["loss"].detach()))
    assert np.mean(losses[-3:]) < 0.8 * np.mean(losses[:3])


def test_config_validation_and_schedules() -> None:
    with pytest.raises(ValueError, match="objective"):
        ModelConfig(objective="byol")
    assert ema_at(1, EMA, 100) == EMA.ema_momentum
    assert ema_at(100, EMA, 100) == pytest.approx(EMA.ema_momentum_end)
    assert pick_device("cpu").type == "cpu"
    with pytest.raises(ValueError, match="not available"):
        pick_device("tpu")


@pytest.mark.skipif(not SAMPLE.exists(), reason="needs the H1 sample: make sample-h1")
def test_ema_objective_and_bf16_autocast_run_on_the_sample(tmp_path: Path) -> None:
    cfg = RunConfig(
        name="e",
        data=parse_config({"data": {"paths": [str(SAMPLE / "*.npz")]}}).data,
        model=replace(EMA, steps=8),
        train=TrainConfig(
            steps=3,
            batch_size=2,
            out_dir=str(tmp_path),
            log_every=1,
            precision="bf16",
            eval_windows=16,
            probes=True,
        ),
    )
    metrics = train(cfg)
    assert np.isfinite(metrics["loss"]) and metrics["eval_erank"] >= 1.0
    assert np.isfinite(metrics["probe_mean"])
    assert load_layer(tmp_path / "e" / "ckpt.pt").model.target is not None


def test_nvidia_smi_lines_parse_and_the_sampler_averages_them() -> None:
    assert parse_smi("87, 41234, 612.5") == {
        "gpu_util_pct": 87.0,
        "gpu_mem_used_mib": 41234.0,
        "gpu_power_w": 612.5,
    }
    with pytest.raises(ValueError):
        parse_smi("[N/A], [N/A]")
    readings = iter(
        [
            {"gpu_util_pct": 100.0, "gpu_mem_used_mib": 10.0, "gpu_power_w": 400.0},
            {"gpu_util_pct": 50.0, "gpu_mem_used_mib": 30.0, "gpu_power_w": 200.0},
        ]
    )
    sampler = Sampler(lambda: next(readings, {}), period=0.01)
    with sampler:
        time.sleep(0.2)
    taken = sampler.take()
    assert taken == {"gpu_util_pct": 75.0, "gpu_power_w": 300.0, "gpu_mem_used_mib": 30.0}
    assert sampler.take() == {}


def test_plan_hours_follows_the_pool_arithmetic() -> None:
    # 20 epochs of a 5M-step pool = 1.56M sequences of 64 steps
    assert plan_hours(100.0, 20) == pytest.approx(20 * (5_000_000 // 64) / 100.0 / 3600)


@pytest.mark.skipif(not SAMPLE.exists(), reason="needs the H1 sample: make sample-h1")
def test_the_gpu_smoke_phases_run_on_a_cpu(tmp_path: Path) -> None:
    cfg = RunConfig(
        name="s",
        data=parse_config({"data": {"paths": [str(SAMPLE / "*.npz")]}}).data,
        model=replace(TINY, steps=8),
        train=TrainConfig(
            batch_size=2,
            out_dir=str(tmp_path),
            device="cpu",
            precision="bf16",
            log_every=1,
            steps=1000,
            eval_windows=8,
        ),
    )
    report = smoke(cfg, minutes=0.03, workers=[0], loader_seconds=0.5, compute_steps=1)
    assert set(report) >= {"hardware", "loader", "compute", "train", "plan_hours_per_run"}
    loader = report["loader"]
    assert isinstance(loader, dict) and loader["0"]["cold_seq_per_s"] > 0
    assert report["plan_hours_per_run"]["epochs_100"] > 0  # type: ignore[index]


def privileged_episode(seed: int, steps: int = 80, effector: str = "hand") -> Episode:
    """A synthetic episode with real privileged arrays, so it can be saved and probed."""
    ep = make_episode(effector, steps=steps, seed=seed)
    rng = np.random.default_rng(seed)
    names = ["world", "lh_palm", "rh_palm", "lp_left_pad", "obj_box", "torso_link"]
    contacts = np.full((steps, 16, 6), -1.0, dtype=np.float32)
    on = rng.random(steps) < 0.5
    contacts[on, 0, :3] = np.array([1, 4, 5.0])  # lh_palm - obj_box, 5 N
    contacts[~on & (rng.random(steps) < 0.5), 0, :3] = np.array([2, 0, 2.0])  # rh_palm - world
    ep.privileged = {
        "contacts": contacts,
        "tip_position": rng.normal(size=(steps, 2, 3)).astype(np.float32),
    }
    ep.text = [""] * steps
    ep.meta = {"body_names": names}
    return ep


def test_privileged_labels_follow_the_contact_table() -> None:
    ep = privileged_episode(0)
    lab = probes.privileged_labels(ep)
    contacts = ep.privileged["contacts"]
    left_on = contacts[:, 0, 0] == 1
    right_on = contacts[:, 0, 0] == 2
    assert np.array_equal(lab["contact_left"], left_on)
    assert np.array_equal(lab["contact_right"], right_on)
    assert np.array_equal(lab["contact_object"], left_on)  # the left contact is with obj_box
    assert np.allclose(lab["log_force"][left_on], np.log1p(5.0))
    assert lab["tip"].shape == (80, 6)
    window = probes.window_labels(lab, np.array([3, 7]), 4, 1)
    assert window["contact_left"][0] == left_on[0:4].any()
    assert window["log_force"][1] == pytest.approx(lab["log_force"][4:8].mean())
    assert np.array_equal(window["tip"][1], lab["tip"][7])


def test_auroc_and_r_squared_on_known_cases() -> None:
    label = np.array([0, 0, 1, 1.0])
    assert probes.auroc(np.array([0.1, 0.2, 0.8, 0.9]), label) == 1.0
    assert probes.auroc(np.array([0.9, 0.8, 0.2, 0.1]), label) == 0.0
    assert probes.auroc(np.zeros(4), label) == 0.5
    assert np.isnan(probes.auroc(np.ones(4), np.zeros(4)))
    y = np.arange(10.0)[:, None]
    assert probes.r_squared(y, y) == 1.0
    assert probes.r_squared(np.full_like(y, y.mean()), y) == pytest.approx(0.0)


def test_probe_folds_never_split_an_episode() -> None:
    groups = np.repeat(np.arange(6), [5, 9, 3, 7, 4, 8])
    seen = []
    for fit, test in probes.episode_folds(groups, 4):
        assert not set(groups[fit]) & set(groups[test])
        assert len(fit) + len(test) == len(groups)
        seen.extend(test.tolist())
    assert sorted(seen) == list(range(len(groups)))
    assert len(probes.episode_folds(groups, 99)) == 6


def test_ridge_probes_recover_a_linear_label_and_not_a_random_one() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(240, 12))
    groups = np.repeat(np.arange(6), 40)
    y = (x @ rng.normal(size=(12, 1))) + 0.05 * rng.normal(size=(240, 1))
    assert probes.r_squared(probes.cross_validated_predictions(x, y, groups), y) > 0.95
    noise = rng.normal(size=(240, 1))
    assert probes.r_squared(probes.cross_validated_predictions(x, noise, groups), noise) < 0.1


def test_probes_run_on_saved_episodes_and_score_every_label(tmp_path: Path) -> None:
    paths = []
    for i in range(5):
        path = tmp_path / f"e{i}.npz"
        privileged_episode(i, effector="hand" if i % 2 else "pincer").save(path)
        paths.append(path)
    headers = [read_header(p) for p in paths]
    space = ChannelSpace.from_headers(headers)
    layer = SensoryLayer(SensoryModel(TINY, space), space, LOADER)
    result = probes.run_probes(layer, paths, stride=4)
    assert set(result) == {"state", "raw"}
    for scores in result.values():
        assert np.isfinite(scores["mean"]) and scores["windows"] > 0
        assert {"contact_left_auroc", "contact_object_auroc", "log_force_r2", "tip_r2"} <= set(
            scores
        )


def test_the_training_path_never_loads_privileged_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = []
    for i in range(2):
        path = tmp_path / f"t{i}.npz"
        privileged_episode(i).save(path)
        paths.append(path)
    space = ChannelSpace.from_headers(read_header(p) for p in paths)
    calls: list[tuple[bool, int]] = []
    real_load = Episode.load.__func__  # type: ignore[attr-defined]

    def spy(cls, path, privileged=True):  # type: ignore[no-untyped-def]
        episode = real_load(cls, path, privileged)
        calls.append((privileged, len(episode.privileged)))
        return episode

    monkeypatch.setattr(Episode, "load", classmethod(spy))
    dataset = SequenceDataset(paths, space, check_senses(TINY.senses), LOADER, seed=0)
    batch = dataset.sample(np.random.default_rng(0))
    assert calls and all(flag is False and n == 0 for flag, n in calls)
    allowed = {*check_senses(TINY.senses), "action", "present", "region_mask"}
    assert set(batch) <= allowed
    assert "privileged" not in "".join(batch)


def test_only_the_probe_module_reads_privileged_arrays() -> None:
    reading = re.compile(r"\.privileged\b|['\"]privileged/")
    offenders = [
        f.name
        for f in Path(package.__file__ or "").parent.glob("*.py")
        if reading.search(f.read_text()) and f.name != "probes.py"
    ]
    assert offenders == []
    loads = re.findall(
        r"Episode\.load\([^)]*\)", (Path(package.__file__ or "").parent / "data.py").read_text()
    )
    assert loads and all("privileged=False" in c for c in loads)


def test_ridge_path_matches_the_closed_form_for_every_strength() -> None:
    rng = np.random.default_rng(0)
    x, xt = rng.normal(size=(30, 7)), rng.normal(size=(5, 7))
    y = rng.normal(size=(30, 3))
    path = probes.ridge_path(x, y, xt, [0.1, 5.0])
    mean = x.mean(0)
    scale = np.sqrt(((x - mean) ** 2).mean())
    a, b = (x - mean) / scale, (xt - mean) / scale
    for i, alpha in enumerate([0.1, 5.0]):
        w = np.linalg.solve(a.T @ a + alpha * np.eye(7), a.T @ (y - y.mean(0)))
        assert np.allclose(path[i], b @ w + y.mean(0))


def test_per_block_alphas_let_each_label_pick_its_own_strength() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(180, 10))
    groups = np.repeat(np.arange(6), 30)
    y = np.concatenate([x[:, :1] * 3.0, rng.normal(size=(180, 1))], axis=1)
    strong, weak = probes.choose_alphas(x, y, groups, [slice(0, 1), slice(1, 2)])
    assert strong < weak  # the noise label wants the strongest shrinkage


def test_dropping_a_sense_at_encode_time_changes_the_states_only_for_that_sense(
    episodes: dict[str, Episode], space: ChannelSpace
) -> None:
    torch.manual_seed(0)
    layer = SensoryLayer(SensoryModel(TINY, space), space, LOADER)
    ep = episodes["hand"]
    full, _ = layer.encode(ep, stride=8)
    same, _ = layer.encode(ep, stride=8, drop=())
    no_touch, _ = layer.encode(ep, stride=8, drop=("touch",))
    no_vision, _ = layer.encode(ep, stride=8, drop=("vision",))
    assert np.array_equal(full, same)
    assert not np.allclose(full, no_touch) and not np.allclose(full, no_vision)
    assert np.isfinite(layer.encode(ep, stride=8, drop=check_senses(TINY.senses))[0]).all()


def test_label_free_diagnostics_run_on_saved_episodes(tmp_path: Path) -> None:
    paths = []
    for i in range(3):
        path = tmp_path / f"d{i}.npz"
        privileged_episode(i, steps=120).save(path)
        paths.append(path)
    space = ChannelSpace.from_headers(read_header(p) for p in paths)
    layer = SensoryLayer(SensoryModel(replace(TINY, objective="ema_vicreg"), space), space, LOADER)
    loss = heldout_loss(layer, paths, batches=2, batch_size=2)
    assert set(loss) == {"heldout_pred", "heldout_reg"} and all(np.isfinite(list(loss.values())))
    reliance = sense_reliance(layer, paths, stride=16)
    assert set(reliance) == {f"reliance_{s}" for s in check_senses(TINY.senses)}
    assert all(v >= 0 for v in reliance.values())


NO_TOUCH = ("vision", "proprio_body", "vestibular", "proprio_effector")
TEACH = replace(TINY, senses=NO_TOUCH, teacher="touch", sigreg_pool_time=True)


def test_a_touch_teacher_must_be_a_known_kind() -> None:
    with pytest.raises(ValueError, match="teacher must be"):
        replace(TEACH, teacher="sound")


def test_touch_can_be_both_an_input_and_a_target(
    episodes: dict[str, Episode], space: ChannelSpace
) -> None:
    torch.manual_seed(0)
    both = replace(
        TEACH, senses=(*NO_TOUCH, "touch"), sense_dropout=0.5, sense_dropout_senses=("touch",)
    )
    model = SensoryModel(both, space).train()
    assert model.teacher is not None and "touch" in model.encoder.senses
    assert model.data_senses == check_senses(both.senses)
    batch = batch_of(episodes, space, model.data_senses)
    model.eval()
    scrambled = {**batch, "touch": torch.rand_like(batch["touch"]) * 50}
    assert not torch.equal(model.embed(batch), model.embed(scrambled))  # touch is an input
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    assert not encoder_ids & {id(p) for p in model.teacher.parameters()}
    model.train()
    out = model(batch)
    assert {"aux", "teacher_own"} <= set(out) and torch.isfinite(out["loss"])
    out["loss"].backward()
    assert any(p.grad is not None for p in model.encoder.stems["touch"].parameters())
    assert any(p.grad is not None for p in model.teacher.head_now.parameters())
    assert all(p.grad is None for p in model.teacher.target.parameters())


@pytest.mark.parametrize("kind", ["touch", "touch_placebo"])
def test_touch_never_enters_the_state_encoder_of_the_teacher_arm(
    episodes: dict[str, Episode], space: ChannelSpace, kind: str
) -> None:
    torch.manual_seed(0)
    model = SensoryModel(replace(TEACH, teacher=kind), space).eval()
    assert model.teacher is not None
    assert "touch" not in model.encoder.senses and "touch" in model.data_senses
    assert not any("touch" in name for name, _ in model.encoder.named_parameters())
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    assert not encoder_ids & {id(p) for p in model.teacher.parameters()}
    batch = batch_of(episodes, space, model.data_senses)
    assert batch["present"].shape[1] == len(model.data_senses) == 5
    scrambled = {
        **batch,
        "touch": torch.rand_like(batch["touch"]) * 50,
        "region_mask": ~batch["region_mask"],
    }
    assert torch.equal(model.embed(batch), model.embed(scrambled))
    out = model(batch)
    assert {"aux", "teacher_own"} <= set(out) and torch.isfinite(out["loss"])


def test_the_aux_loss_trains_the_main_model_but_not_the_teacher_encoder(
    episodes: dict[str, Episode], space: ChannelSpace
) -> None:
    torch.manual_seed(0)
    model = SensoryModel(TEACH, space).train()
    assert model.teacher is not None
    batch = batch_of(episodes, space, model.data_senses)
    states = model.embed(batch)
    cond = model.action(batch["action"].flatten(0, 1)).view_as(states)
    trunk = model.trunk(states[:, :-1], cond[:, :-1])
    has = batch["present"][:, model.data_senses.index("touch")].float()
    aux = model.teacher.aux_loss(batch, states, trunk, has)
    aux.backward()  # type: ignore[no-untyped-call]
    assert any(p.grad is not None for p in model.encoder.parameters())
    assert any(p.grad is not None for p in model.teacher.head_now.parameters())
    assert all(p.grad is None for p in model.teacher.encoder.parameters())
    assert all(p.grad is None for p in model.teacher.target.parameters())


def test_the_ema_touch_target_follows_the_touch_encoder_and_the_placebo_rolls_the_batch(
    episodes: dict[str, Episode], space: ChannelSpace
) -> None:
    torch.manual_seed(0)
    real = SensoryModel(TEACH, space).eval()
    placebo = SensoryModel(replace(TEACH, teacher="touch_placebo"), space).eval()
    assert real.teacher is not None and placebo.teacher is not None
    placebo.teacher.load_state_dict(real.teacher.state_dict())
    batch = batch_of(episodes, space, real.data_senses)
    rolled = {k: v.roll(1, 0) for k, v in batch.items() if k in ("touch", "region_mask")}
    t_real = real.teacher.targets(batch["touch"], batch["region_mask"])
    t_placebo = placebo.teacher.targets(batch["touch"], batch["region_mask"])
    assert torch.allclose(t_placebo, real.teacher.targets(rolled["touch"], rolled["region_mask"]))
    assert not torch.allclose(t_placebo, t_real)
    with torch.no_grad():
        for p in real.teacher.encoder.parameters():
            p.add_(1.0)
    real.teacher.momentum = 0.0
    real.update_target(0.0)
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            real.teacher.encoder.parameters(), real.teacher.target.parameters(), strict=True
        )
    )


def test_the_teacher_arm_trains_and_encodes_without_touch(
    episodes: dict[str, Episode], space: ChannelSpace
) -> None:
    torch.manual_seed(0)
    model = SensoryModel(TEACH, space)
    batch = batch_of(episodes, space, model.data_senses)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    losses = []
    for _ in range(30):
        out = model(batch)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        model.update_target(0.99)
        losses.append(float(out["loss"].detach()))
    assert np.mean(losses[-3:]) < 0.7 * np.mean(losses[:3])
    layer = SensoryLayer(model, space, LOADER)
    states, _ = layer.encode(episodes["pincer"], stride=8)
    assert states.shape[1] == 32 and np.isfinite(states).all()
    assert count_parameters(model)["teacher"] > 0


def test_dropout_can_be_restricted_to_chosen_senses(space: ChannelSpace) -> None:
    model = SensoryModel(
        replace(TINY, sense_dropout=0.9, sense_dropout_senses=("touch",)), space
    ).train()
    keep = model.encoder.keep_mask(torch.ones(256, 6, dtype=torch.bool), True)
    touch = model.senses.index("touch")
    others = [i for i in range(6) if i != touch]
    assert keep[:, others].all() and not keep[:, touch].all() and keep[:, touch].any()


def saved_privileged_episodes(tmp_path: Path, count: int = 5, steps: int = 120) -> list[Path]:
    paths = []
    for i in range(count):
        path = tmp_path / f"c{i}.npz"
        privileged_episode(i, steps=steps, effector="hand" if i % 2 else "pincer").save(path)
        paths.append(path)
    return paths


def test_bootstrap_scores_and_pairing(tmp_path: Path) -> None:
    paths = saved_privileged_episodes(tmp_path)
    space = ChannelSpace.from_headers(read_header(p) for p in paths)
    layer = SensoryLayer(SensoryModel(TINY, space), space, LOADER)
    data = probes.collect(layer, paths, stride=4, raw=False)
    pred = probes.predict_probes(data, "state")
    assert pred.scores() == probes.score_probes(data, "state")
    draws = pred.bootstrap(12, np.random.default_rng(0))
    assert len(draws) == 12 and all("mean" in d for d in draws)
    again = pred.bootstrap(12, np.random.default_rng(0))
    assert [d["mean"] for d in draws] == [d["mean"] for d in again]  # same rng, same episodes
    other = probes.Predictions(pred.pred * 0.5, pred.target, pred.blocks, pred.groups)
    a, b = pred.bootstrap(6, np.random.default_rng(1)), other.bootstrap(6, np.random.default_rng(1))
    assert all(np.isfinite(x["mean"]) for x in a + b)


def test_probe_data_for_arbitrary_features_matches_the_encoder_windows(tmp_path: Path) -> None:
    from sensorium.model import controls  # noqa: PLC0415

    paths = saved_privileged_episodes(tmp_path)
    space = ChannelSpace.from_headers(read_header(p) for p in paths)
    layer = SensoryLayer(SensoryModel(TINY, space), space, LOADER)
    ref = probes.collect(layer, paths, stride=4, raw=False)
    raw = controls.RawFeatures(space, TINY.substeps)
    data = controls.build_probe_data(paths, TINY.substeps, 4, lambda ep, ends: raw.blocks(ep, ends))
    assert np.array_equal(data.groups, ref.groups)
    for name in ref.labels:
        assert np.array_equal(data.labels[name], ref.labels[name], equal_nan=True)
    assert data.features["vision"].shape[1] == 16 * 16 * 3
    assert data.features["proprio"].shape[0] == len(ref.groups)


def test_pca_and_block_scaling() -> None:
    from sensorium.model import controls  # noqa: PLC0415

    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3)) @ rng.normal(size=(3, 10))
    mean, comp = controls.pca_fit(x, 3)
    recon = controls.pca_apply(x, (mean, comp)) @ comp.T + mean
    assert np.allclose(recon, x, atol=1e-8) and comp.shape == (10, 3)
    big, small = 10 * rng.normal(size=(50, 4)), rng.normal(size=(50, 2))
    both = controls.concat_scaled(
        [big, small], [controls.block_rms(big), controls.block_rms(small)]
    )
    assert both.shape == (50, 6) and abs(controls.block_rms(both[:, :4]) - 1) < 1e-9


def test_dino_features_use_the_evaluation_crop_and_normalisation() -> None:
    from sensorium.model import controls  # noqa: PLC0415

    class Stub(nn.Module):
        seen: list[Tensor] = []

        def forward_features(self, x: Tensor) -> dict[str, Tensor]:
            self.seen.append(x)
            n = x.size(0)
            return {"x_norm_clstoken": torch.zeros(n, 5), "x_norm_patchtokens": torch.ones(n, 7, 5)}

    frames = np.full((3, 256, 256, 3), 124, dtype=np.uint8)  # about ImageNet's mean
    out = controls.dino_features(Stub(), frames, batch=2)
    assert out.shape == (3, 10) and np.allclose(out[:, :5], 0) and np.allclose(out[:, 5:], 1)
    assert (
        Stub.seen[0].shape == (2, 3, 224, 224)
        and abs(Stub.seen[0][0, 0, 0, 0].item() - 0.0056) < 1e-3
    )


def test_the_supervised_control_trains_on_labels_and_probes_like_any_encoder(
    tmp_path: Path,
) -> None:
    from sensorium.model import controls  # noqa: PLC0415

    paths = saved_privileged_episodes(tmp_path)
    space = ChannelSpace.from_headers(read_header(p) for p in paths)
    ds = controls.LabelledSequences(paths, space, check_senses(NO_TOUCH), LOADER, seed=0)
    seq = ds.sample(np.random.default_rng(0))
    assert seq["y_contact_left"].shape == (4,) and seq["y_tip"].shape == (4, 6)
    assert set(seq["y_contact_left"].tolist()) <= {0.0, 1.0}
    cfg = controls.SupervisedConfig(
        model=replace(TINY, senses=NO_TOUCH), steps=3, batch_size=2, workers=0
    )
    ckpt = controls.train_supervised(cfg, paths, space, tmp_path / "sup", lambda _m: None)
    layer = load_layer(ckpt)
    assert layer.model.senses == check_senses(NO_TOUCH)
    assert np.isfinite(probes.run_probes(layer, paths, stride=4, raw=False)["state"]["mean"])


def test_the_same_thread_count_gives_the_same_run(tmp_path: Path) -> None:
    paths = saved_privileged_episodes(tmp_path, count=3, steps=100)
    losses = []
    for name in ("x", "y"):
        cfg = RunConfig(
            name=name,
            data=parse_config({"data": {"paths": [str(p) for p in paths]}}).data,
            model=replace(TINY, teacher="touch", senses=NO_TOUCH, sigreg_pool_time=True),
            train=TrainConfig(
                steps=8,
                batch_size=2,
                out_dir=str(tmp_path),
                log_every=2,
                threads=2,
                eval_windows=0,
                num_workers=0,
            ),
        )
        train(cfg)
        rows = [
            json.loads(line) for line in (tmp_path / name / "log.jsonl").read_text().splitlines()
        ]
        losses.append([r["loss"] for r in rows])
        assert json.loads((tmp_path / name / "run.json").read_text())["threads"] == 2
    assert losses[0] == losses[1] and len(losses[0]) == 4


def test_plot_runs_draws_losses_heldout_loss_probe_and_floor(tmp_path: Path) -> None:
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    runs, probes = tmp_path / "runs", tmp_path / "probes"
    probes.mkdir()
    for seed in (0, 1):
        run = runs / f"armx-s{seed}"
        run.mkdir(parents=True)
        (run / "log.jsonl").write_text(
            "\n".join(
                json.dumps({"step": s, "loss": 2.0 - s / 100 + seed / 10, "pred": 1.0, "reg": 0.5})
                for s in (50, 100, 150)
            )
        )
        for step in (100, 200):
            (probes / f"armx-s{seed}-n{step}.json").write_text(
                json.dumps(
                    {"vp": {"mean": 0.2 + step / 1000}, "heldout_pred": 0.4, "heldout_reg": 0.1}
                )
            )
        (probes / f"floor-s{seed}.json").write_text(json.dumps({"vp": {"mean": 0.22 + seed / 50}}))
    out = tmp_path / "fig.png"
    script = Path(__file__).resolve().parent.parent / "scripts" / "plot_runs.py"
    done = subprocess.run(
        [
            sys.executable,
            str(script),
            "--runs",
            str(runs),
            "--probes",
            str(probes),
            "--metric",
            "vp.mean",
            "--floor",
            "armx=floor",
            "--title",
            "test",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert out.exists() and out.stat().st_size > 5000


def consequence_episode(seed: int = 0, steps: int = 120) -> Episode:
    """A privileged episode where the left palm touches ``obj_box`` in steps 20-49 and the box moves
    from step 30 on (so steps 20-39 have a consequence 10 steps later) and ``rh_palm`` touches
    nothing."""
    ep = privileged_episode(seed, steps=steps)
    contacts = np.full((steps, 16, 6), -1.0, dtype=np.float32)
    contacts[20:50, 0, :3] = np.array([1, 4, 5.0])  # lh_palm - obj_box
    contacts[70:80, 0, :3] = np.array([2, 0, 2.0])  # rh_palm - world: not an object
    pose = np.zeros((steps, 1, 7), dtype=np.float32)
    pose[30:, 0, 0] = np.linspace(0.0, 0.3, steps - 30)  # moves 0.3 m after step 30
    ep.privileged = {**ep.privileged, "contacts": contacts, "object_pose": pose}
    ep.meta = {**ep.meta, "object_names": ["obj_box"]}
    return ep


def test_consequence_steps_need_contact_and_a_later_move() -> None:
    flags = probes.consequence_steps(consequence_episode(), horizon=10, min_move=0.01)
    contact, conseq = flags["hand_object_contact"], flags["consequence"]
    assert contact[20:50].all() and contact.sum() == 30  # the world contact does not count
    assert conseq[20:50].sum() > 0 and conseq[:20].sum() == 0 and conseq[50:].sum() == 0
    still = consequence_episode()
    still.privileged["object_pose"][:] = 0.0
    assert not probes.consequence_steps(still)["consequence"].any()


def test_index_selects_windows_and_the_sampler_draws_only_those(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = []
    for i in range(2):
        path = tmp_path / f"s{i}.npz"
        consequence_episode(i, steps=140).save(path)
        paths.append(path)
    span = LOADER.steps * LOADER.substeps
    index = subsets.build_index(paths, span, frac=0.25)
    chosen = index[paths[0].name]["conseq"]
    assert chosen and set(chosen) < set(index[paths[0].name]["all"])
    assert not set(chosen) & set(index[paths[0].name]["contact"])
    summary = subsets.summarise(index)
    assert (
        summary["conseq"]["episodes"] == 2
        and "event" in summary
        and summary["all"]["starts"] == 2 * (140 - span + 1)
    )
    space = ChannelSpace.from_headers(read_header(p) for p in paths)
    starts: list[int] = []
    real = subsets.sequence_with_ends  # type: ignore[attr-defined]

    def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        starts.append(kwargs["start"])
        return real(*args, **kwargs)

    monkeypatch.setattr(subsets, "sequence_with_ends", spy)
    dataset = subsets.IndexedSequences(
        paths, space, check_senses(TINY.senses), LOADER, index, "conseq", seed=0
    )
    rng = np.random.default_rng(0)
    for _ in range(20):
        dataset.sample(rng)
    assert starts and set(starts) <= set(chosen)
    subsets.save_index(index, tmp_path / "i.json")
    assert subsets.load_index(tmp_path / "i.json") == index
    with pytest.raises(ValueError, match="key must be"):
        subsets.IndexedSequences(paths, space, check_senses(TINY.senses), LOADER, index, "nope")
    empty = {name: {**v, "contact": []} for name, v in index.items()}
    with pytest.raises(ValueError, match="no sequence start"):
        subsets.IndexedSequences(paths, space, check_senses(TINY.senses), LOADER, empty, "contact")


def test_proprio_dropout_drops_all_proprio_groups_together(space: ChannelSpace) -> None:
    model = SensoryModel(replace(TINY, proprio_dropout=0.5), space).train()
    n = len(model.senses)
    keep = model.encoder.keep_mask(torch.ones(512, n, dtype=torch.bool), True)
    proprio = [i for i, s in enumerate(model.senses) if s.startswith(("proprio", "vestib"))]
    other = [i for i in range(n) if i not in proprio]
    assert proprio and keep[:, other].all()
    assert (keep[:, proprio] == keep[:, proprio[:1]]).all()  # all groups go together
    assert 0.35 < 1 - keep[:, proprio[0]].float().mean() < 0.65
    assert model.encoder.keep_mask(torch.ones(8, n, dtype=torch.bool), False).all()
    with pytest.raises(ValueError, match="proprio_dropout"):
        replace(TINY, proprio_dropout=1.0)


def test_surprise_weights_use_only_inputs_and_model_errors() -> None:
    err = torch.tensor([[0.1, 0.1, 0.8]])
    assert torch.equal(surprise_weights("none", err, {}), torch.ones_like(err))
    w = surprise_weights("error", err, {})
    assert w.mean() == pytest.approx(1.0) and w[0, 2] > w[0, 0]
    touch = torch.zeros(1, 4, 4, 3)  # [B, T, K, taxels]: touch changes only between steps 2 and 3
    touch[:, 3:] = 1.0
    w = surprise_weights("touch", err, {"touch": touch})
    assert w[0, 2] == w.max() and w.mean() == pytest.approx(1.0) and w.min() >= 0.2 / 5
    with pytest.raises(ValueError, match="surprise"):
        replace(TINY, surprise="privileged")
    with pytest.raises(ValueError, match="touch"):
        replace(TINY, senses=("vision", "proprio_body"), surprise="touch")


def test_a_surprise_weighted_step_runs_and_reports_the_unweighted_loss(
    episodes: dict[str, Episode], space: ChannelSpace
) -> None:
    torch.manual_seed(0)
    cfg = replace(TINY, surprise="error", proprio_dropout=0.5)
    model = SensoryModel(cfg, space).train()
    out = model(batch_of(episodes, space, check_senses(cfg.senses)))
    assert torch.isfinite(out["loss"]) and out["pred"] > 0
    out["loss"].backward()


def test_supervised_freeze_keeps_the_encoder_fixed_and_saves_intermediate_checkpoints(
    tmp_path: Path,
) -> None:
    from sensorium.model import controls  # noqa: PLC0415

    paths = saved_privileged_episodes(tmp_path)
    space = ChannelSpace.from_headers(read_header(p) for p in paths)
    cfg = controls.SupervisedConfig(
        model=replace(TINY, senses=NO_TOUCH),
        steps=4,
        batch_size=2,
        workers=0,
        freeze_steps=2,
        encoder_lr_scale=0.1,
        save_steps=(2,),
    )
    first = controls.train_supervised(cfg, paths, space, tmp_path / "ft", lambda _m: None)
    mid, last = tmp_path / "ft" / "ckpt-2.pt", first
    assert mid.exists() and last.exists()
    torch.manual_seed(cfg.seed)
    before = SensoryModel(cfg.model, space).state_dict()
    saved_mid = torch.load(mid, weights_only=True)["model"]
    saved_last = torch.load(last, weights_only=True)["model"]
    key = next(k for k in saved_mid if k.startswith("encoder."))
    # Frozen for steps 1-2 (lr 0, and AdamW weight decay scales with lr): the encoder at step 2 is
    # the initialisation of the same seed; it only moves afterwards.
    assert torch.equal(saved_mid[key], before[key])
    assert not torch.equal(saved_last[key], saved_mid[key])


def test_drop_prefix_leaves_effort_channels_out_of_the_space_and_the_batch() -> None:
    ep = make_episode("hand", seed=3)
    ep.proprio_names = [*ep.proprio_names, "eff_lh_A_FFJ3", "eff_torso"]
    ep.proprio = np.concatenate([ep.proprio, np.full((ep.proprio.shape[0], 2), 7.0, np.float32)], 1)
    with_eff = ChannelSpace.from_headers([header_of(ep)])
    without = ChannelSpace.from_headers([header_of(ep)], drop_prefix=("eff_",))
    assert "eff_torso" in with_eff.keys["proprio_body"]
    assert not any(k.startswith("eff_") for g in without.keys.values() for k in g)
    assert without.size("proprio_body") == with_eff.size("proprio_body") - 2
    cols = without.columns(header_of(ep))
    assert (
        set(cols.src["proprio_body"]) == set(range(4))
        and 7.0 not in ep.proprio[:, cols.src["proprio_body"]]
    )
    again = ChannelSpace.from_json(without.to_json())
    assert again.drop_prefix == ("eff_",) and again.keys == without.keys
    old = json.loads(without.to_json())
    del old["drop_prefix"]  # checkpoints written before the option existed
    assert ChannelSpace.from_json(json.dumps(old)).drop_prefix == ()


def event_episode(seed: int = 0, steps: int = 140, mass: float = 0.5) -> Episode:
    """``consequence_episode`` plus what Track A's events and the hidden-property labels read."""
    ep = consequence_episode(seed, steps=steps)
    pose = np.zeros((steps, 1, 7), dtype=np.float32)
    pose[:, 0, 3] = 1.0  # identity quaternion (w first or last: upright either way)
    pose[30:, 0, 0] = np.linspace(0.0, 0.3, steps - 30)
    touch = np.zeros((steps, 1), dtype=np.float32)
    touch[20:50] = 5.0
    hidden = np.tile(np.array([mass, 0.6, 0.02], dtype=np.float32), (steps, 1, 1))
    ep.privileged = {
        **ep.privileged,
        "object_pose": pose,
        "object_contact": touch,
        "hidden_properties": hidden,
    }
    return ep


def test_event_steps_follow_the_trackA_displaced_events() -> None:
    ep = event_episode()
    flags = probes.consequence_steps(ep)
    event = flags["event"]
    assert event[20:50].all() and event[50:70].all()  # the bout plus its attribution window
    assert not event[:20].any() and not event[71:].any()
    still = event_episode()
    still.privileged["object_pose"][:, 0, 0] = 0.0
    assert not probes.event_steps(still).any()


def test_hidden_labels_belong_to_the_touched_object_and_are_nan_without_contact() -> None:
    ep = event_episode(mass=0.5)
    ends = np.array([10, 30, 60])
    lab = probes.hidden_window_labels(ep, ends, substeps=4, gap=1)
    assert np.isnan(lab["hidden_log_mass"][0]) and np.isnan(lab["hidden_log_mass"][2])
    assert lab["hidden_log_mass"][1] == pytest.approx(np.log(0.5))
    assert lab["hidden_friction"][1] == pytest.approx(0.6)
    assert lab["hidden_softness"][1] == pytest.approx(np.log(0.02))
    no_hidden = event_episode()
    del no_hidden.privileged["hidden_properties"]
    assert np.isnan(probes.hidden_window_labels(no_hidden, ends, 4, 1)["hidden_log_mass"]).all()
    with pytest.raises(KeyError):
        probes.hidden_properties(no_hidden)


def test_the_hidden_probe_scores_a_decodable_label_and_is_nan_when_too_few_windows() -> None:
    rng = np.random.default_rng(0)
    episodes = 6
    groups = np.repeat(np.arange(episodes), 40)
    mass = np.repeat(rng.normal(size=episodes), 40)
    x = rng.normal(size=(len(groups), 5))
    x[:, 0] = mass + 0.1 * x[:, 0]  # the state carries the label
    labels = {n: mass + 0.0 * rng.normal(size=len(groups)) for n in probes.HIDDEN}
    data = probes.ProbeData({"state": x}, labels, groups)
    scores = probes.predict_hidden(data, "state")
    assert scores["hidden_log_mass_r2"] > 0.5 and scores["hidden_episodes"] == episodes
    labels["hidden_log_mass"] = np.full(len(groups), np.nan)
    assert np.isnan(probes.predict_hidden(data, "state")["hidden_log_mass_r2"])


def test_the_supervised_control_can_also_train_on_hidden_properties(tmp_path: Path) -> None:
    from sensorium.model import controls  # noqa: PLC0415

    paths = []
    for i in range(5):
        path = tmp_path / f"h{i}.npz"
        event_episode(i, steps=140, mass=0.2 + 0.3 * i).save(path)
        paths.append(path)
    space = ChannelSpace.from_headers(read_header(p) for p in paths)
    ds = controls.LabelledSequences(
        paths, space, check_senses(NO_TOUCH), LOADER, seed=0, hidden=True
    )
    seq = ds.sample(np.random.default_rng(0))
    assert set(probes.HIDDEN) <= {k.removeprefix("y_") for k in seq} and seq[
        "y_hidden_log_mass"
    ].shape == (4,)
    cfg = controls.SupervisedConfig(
        model=replace(TINY, senses=NO_TOUCH), steps=3, batch_size=2, workers=0, hidden=True
    )
    ckpt = controls.train_supervised(cfg, paths, space, tmp_path / "sup", lambda _m: None)
    layer = load_layer(ckpt)
    data = probes.collect(layer, paths, stride=4, raw=False)
    assert np.isfinite(data.features["state"]).all(), "NaN states"
    scores = probes.predict_hidden(data, "state", min_windows=5)
    assert scores["hidden_windows"] > 0 and "hidden_log_mass_r2" in scores


REF = replace(TINY, senses=NO_TOUCH)


def test_patchify_orders_pixels_by_patch_and_scales_them() -> None:
    from sensorium.model import reference  # noqa: PLC0415

    frames = torch.zeros(1, 1, 4, 4, 3, dtype=torch.uint8)
    frames[0, 0, 0:2, 2:4, :] = 255  # the top-right 2x2 patch is white
    patches = reference.patchify(frames, 2)
    assert patches.shape == (1, 1, 4, 12)
    assert torch.all(patches[0, 0, 1] == 1.0) and torch.all(patches[0, 0, 0] == -1.0)


def test_mae_masks_the_encoder_input_and_scores_only_masked_entries(
    episodes: dict[str, Episode], space: ChannelSpace
) -> None:
    from sensorium.model import reference  # noqa: PLC0415

    torch.manual_seed(0)
    batch = batch_of(episodes, space, check_senses(REF.senses))
    masked, hidden, channels = reference.mask_batch(batch, check_senses(REF.senses), 8, 0.75, 0.5)
    assert 0.55 < hidden.float().mean() < 0.95 and 0.2 < channels.float().mean() < 0.8
    side = 32 // 8
    pix = hidden.view(*hidden.shape[:2], side, side).repeat_interleave(8, 2).repeat_interleave(8, 3)
    assert torch.all(masked["vision"][pix] == reference.GRAY)
    assert torch.equal(masked["vision"][~pix], batch["vision"][~pix])  # visible pixels untouched
    gone = channels[:, :, None, :32]
    assert torch.all(
        masked["proprio_body"].masked_select(gone[..., : masked["proprio_body"].size(-1)]) == 0
    )


def test_the_input_space_objectives_train_the_encoder(
    episodes: dict[str, Episode], space: ChannelSpace
) -> None:
    for objective in ("mae", "nextobs"):
        torch.manual_seed(0)
        model = SensoryModel(replace(REF, objective=objective), space).train()
        batch = batch_of(episodes, space, check_senses(REF.senses))
        out = model(batch)
        assert torch.isfinite(out["loss"]) and out["loss"] > 0 and {"pix", "prop"} <= set(out)
        out["loss"].backward()
        assert all(p.grad is not None for p in model.decoder.parameters())  # type: ignore[union-attr]
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()
        )
        predictor_trained = any(p.grad is not None for p in model.predictor.parameters())
        assert predictor_trained == (objective == "nextobs")  # mae never touches the predictor


def test_the_input_space_objectives_support_vision_and_proprio_only() -> None:
    with pytest.raises(ValueError, match="vision and proprio"):
        replace(TINY, objective="mae")  # TINY has touch and audio
    with pytest.raises(ValueError, match="objective must be"):
        replace(REF, objective="mse")


def test_teacher_arms_start_from_the_same_encoder_as_the_plain_arm(space: ChannelSpace) -> None:
    """Paired-by-seed contrasts (B-teach-long) assume equal initial encoders for equal seeds."""
    weights = {}
    for kind in ("none", "touch", "touch_placebo"):
        torch.manual_seed(5)
        model = SensoryModel(replace(TEACH, teacher=kind), space)
        weights[kind] = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    for kind in ("touch", "touch_placebo"):
        assert all(torch.equal(weights[kind][k], weights["none"][k]) for k in weights["none"])


def load_objective_script() -> Any:
    """``scripts/objective_test.py`` as a module (scripts are not a package)."""
    import importlib.util  # noqa: PLC0415

    path = Path(__file__).resolve().parents[1] / "scripts" / "objective_test.py"
    spec = importlib.util.spec_from_file_location("objective_test_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_teacher_results(
    teach: float, placebo: float, step: int = 10000
) -> dict[str, dict[str, dict[str, float]]]:
    """Probe means of three seeds: jepa about 0.11, teach and placebo ``jepa + offset``."""
    jepa = [0.10, 0.12, 0.11]
    jitter = [0.0, 0.004, -0.004]
    res = {}
    for seed in range(3):
        for arm, offset in (("long-jepa", 0.0), ("long-teach", teach), ("long-placebo", placebo)):
            value = jepa[seed] + offset + (jitter[seed] if offset else 0.0)
            res[f"{arm}-s{seed}-n{step}"] = {"heldout": {"mean": value}}
    return res


@pytest.mark.parametrize(
    ("teach", "placebo", "verdict"),
    [
        (0.10, 0.02, "teacher prevents drift"),
        (0.08, 0.07, "extra heads only"),
        (-0.08, 0.0, "teacher harms"),
        (0.01, 0.0, "null"),
    ],
)
def test_the_preregistered_teacher_verdict_covers_each_outcome(
    teach: float, placebo: float, verdict: str
) -> None:
    script = load_objective_script()
    assert script.teacher_verdict(synthetic_teacher_results(teach, placebo)) == verdict


def test_the_teacher_verdict_needs_every_arm_and_is_null_when_noisy() -> None:
    script = load_objective_script()
    res = synthetic_teacher_results(0.10, 0.02)
    assert (
        script.teacher_verdict({k: v for k, v in res.items() if "placebo" not in k}) == "incomplete"
    )
    noisy = synthetic_teacher_results(0.08, 0.0)  # +0.08 on average but wildly different per seed
    for seed, shift in enumerate((-0.10, 0.0, 0.10)):
        noisy[f"long-teach-s{seed}-n10000"]["heldout"]["mean"] += shift
    assert script.teacher_verdict(noisy) == "null"  # above 0.032 but not 2.8 own standard errors
