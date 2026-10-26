"""``scripts/pusht_test.py`` and ``scripts/lewm_launch.py`` (B-pusht-0, training)."""

import importlib.util
import json
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch", reason="needs the model extra: uv sync --extra model")
pytest.importorskip("h5py", reason="needs the pusht extra")

from sensorium import pusht  # noqa: E402
from sensorium.episode import Episode  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name: str) -> Any:
    """A script of ``scripts/`` as a module."""
    spec = importlib.util.spec_from_file_location(f"{name}_script", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plan() -> Iterator[Any]:
    sys.path.insert(0, str(SCRIPTS))
    yield load("pusht_test")
    sys.path.remove(str(SCRIPTS))


def synthetic(index: int, steps: int, seed: int) -> Episode:
    """A Push-T-shaped episode: moving coloured squares at 224 px, agent proprio and actions."""
    rng = np.random.default_rng(seed)
    pixels = np.full((steps, 224, 224, 3), 230, dtype=np.uint8)
    xy = np.cumsum(rng.normal(0, 3, (steps, 2)), 0) + 100
    for t, (x, y) in enumerate(xy.astype(int).clip(10, 200)):
        pixels[t, y : y + 16, x : x + 16] = (40, 60, 200)
    arrays: dict[str, np.ndarray] = {
        "pixels": pixels,
        "action": rng.uniform(-1, 1, (steps, 2)),
        "proprio": np.concatenate([xy, rng.normal(0, 1, (steps, 2))], 1),
        "state": rng.normal(0, 1, (steps, 7)),
    }
    return pusht.to_episode(index, arrays, action_scale=1.0)


def test_our_sequences_end_on_lewms_clip_frames(plan: Any) -> None:
    assert plan.SUBSTEPS * plan.spacing(pusht.DT) == pusht.LEWM_FRAMESKIP  # one model step
    assert plan.SEQ_STEPS == pusht.LEWM_NUM_STEPS  # 3 context frames + 1 target
    assert plan.our_start(10) == 6 and plan.our_start(2) == -2
    w = plan.start_weights(["a.npz", "b.npz"], [30, 25], [(0, 10), (0, 2), (1, 5)])
    assert len(w["a.npz"]) == 30 - 20 + 1 and len(w["b.npz"]) == 25 - 20 + 1
    assert np.flatnonzero(w["a.npz"]).tolist() == [6]  # LeWM start 2 is before our first window
    assert np.flatnonzero(w["b.npz"]).tolist() == [1]


def test_the_subset_is_fixed_and_the_budget_is_as_registered(plan: Any) -> None:
    a, b = plan.subset_indices(18_685), plan.subset_indices(18_685)
    assert np.array_equal(a, b) and len(a) == 2000 and np.all(np.diff(a) > 0)
    assert plan.CLIPS == 2_048_000 and plan.BATCH == 128


def test_the_arms_are_matched_in_size_within_25_percent(plan: Any) -> None:
    lewm = plan.lewm_tiny_parameters()
    assert lewm["total"] == 1_427_054  # le-wm 8edfeb3's modules at LEWM_OVERRIDES, by hand
    counts = plan.params()
    for arm in plan.ARMS:
        ratio = counts[arm]["planning model"] / lewm["total"]
        assert 0.75 <= ratio <= 1.25, (arm, ratio)
    assert counts["fix1-pix"]["reconstruction decoder"] > 0  # counted apart
    assert counts["a0-pix"]["reconstruction decoder"] == 0


def test_our_64px_cache_gives_lewms_exact_pixels(plan: Any, tmp_path: Path) -> None:
    from sensorium.model.data import (  # noqa: PLC0415
        LoaderConfig,
        header_of,  # noqa: PLC0415
        sequence_with_ends,
    )
    from sensorium.model.frame_cache import build_cache, load_cached  # noqa: PLC0415
    from sensorium.model.layout import ChannelSpace  # noqa: PLC0415

    path = tmp_path / "pusht-00000.npz"
    synthetic(0, 40, 0).save(path)
    stored = Episode.load(path)  # decoded, as both formats are built from
    build_cache([path], tmp_path / "cache", plan.SIZE)
    space = ChannelSpace.from_headers([header_of(stored)])
    loaded = load_cached(path, tmp_path / "cache", space, ("vision",))
    cfg = LoaderConfig(steps=4, substeps=5, view_size=64, crop=1.0, frame_cache="x")
    for train in (True, False):
        arrays, ends = sequence_with_ends(
            loaded, space, ("vision",), cfg, np.random.default_rng(1), train=train, start=7
        )
        assert np.array_equal(arrays["vision"], pusht.lewm_frames(stored)[ends])


def test_train_runs_on_the_subset_and_lewms_clips(
    plan: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    for i in range(6):
        path = plan.episode_path(i)
        path.parent.mkdir(parents=True, exist_ok=True)
        synthetic(i, 40 + 3 * i, i).save(path)
    monkeypatch.setattr(plan, "N_SUBSET", 4)
    monkeypatch.setattr(plan, "STEPS", 4)
    monkeypatch.setattr(plan, "BATCH", 4)
    monkeypatch.setattr(plan, "CKPTS", (0, 4))
    plan.subset()
    chosen = json.loads((tmp_path / "runs/pusht/subset.json").read_text())
    assert chosen == plan.subset_indices(6).tolist()
    plan.weights()
    clips = json.loads((tmp_path / "runs/pusht/clips.json").read_text())
    lengths = [plan.episode_length(p) for p in plan.subset_paths()]
    assert clips == json.loads(json.dumps(pusht.clip_split(lengths)))  # LeWM's split, subset
    w = json.loads((tmp_path / "runs/pusht/start_weights.json").read_text())["weights"]
    assert sum(sum(v) for v in w.values()) == sum(1 for _, t in clips["train"] if t >= 4)
    plan.cache(1)
    for arm in plan.ARMS:
        cfg = plan.run_config(arm, 0)
        short = replace(cfg.train, save_steps=(0, 4), num_workers=0, log_every=1)
        cfg = replace(cfg, train=short)
        plan.train(cfg)
        assert (tmp_path / f"runs/pusht/runs/{arm}-s0/ckpt-4.pt").exists()


def test_lewm_command_keeps_lewms_split_seed_and_passes_the_run_seed() -> None:
    sys.path.insert(0, str(SCRIPTS))
    try:
        plan = load("pusht_test")
        launch = load("lewm_launch")
    finally:
        sys.path.remove(str(SCRIPTS))
    cmd = plan.lewm_command(2)
    assert cmd[:3] == ["python", "lewm_launch.py", "--run-seed=2"]
    assert not any(a.startswith("seed=") for a in cmd)  # cfg.seed stays 3072: LeWM's split
    assert "+trainer.max_steps=16000" in cmd and "loader.batch_size=128" in cmd
    assert f"data.dataset.name={plan.LEWM_DATASET}" in cmd
    seed, rest = launch.split_seed(cmd[2:])
    assert seed == 2 and rest == cmd[3:]
    with pytest.raises(ValueError, match="run-seed"):
        launch.split_seed(["img_size=64"])
