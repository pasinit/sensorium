"""``scripts/skip_test.py`` (B-skip-0): tiny end-to-end runs on synthetic episodes."""

import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch", reason="needs the model extra: uv sync --extra model")

from test_planning import synthetic_plan  # noqa: E402
from test_variety import save_episodes  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SIBLINGS = ("rep_test", "plan_test", "fix_test", "variety_test", "jepa_diagnostics")


@pytest.fixture
def skip() -> Iterator[Any]:
    """``scripts/skip_test.py`` with fresh copies of its sibling scripts (``configure`` rewrites
    their globals, so no other test may see them)."""
    sys.path.insert(0, str(SCRIPTS))
    saved = {n: sys.modules.pop(n) for n in SIBLINGS if n in sys.modules}
    spec = importlib.util.spec_from_file_location("skip_test_script", SCRIPTS / "skip_test.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    for name in SIBLINGS:
        sys.modules.pop(name, None)
    sys.modules.update(saved)
    sys.path.remove(str(SCRIPTS))


def test_configure_points_the_rep_machinery_at_the_skip_runs(skip: Any) -> None:
    skip.configure()
    rep = skip.rep
    assert rep.fix.ROOT == skip.ROOT and set(rep.fix.ARMS) == {skip.FIX1, *skip.ARMS}
    assert rep.plan.ENCODERS["skip-both"] == (skip.ROOT / "runs", "skip-both")
    assert rep.plan.LOCATE_ARMS == ("A0", skip.FIX1, "skip-z", "skip-proj", "skip-both")
    assert rep.plan.TOKENS_ARMS == (skip.FIX1, "skip-z", "skip-proj", "skip-both")
    assert skip.ARMS["skip-both"] == rep.FIX1_OPTIONS | {
        "skip_tokens": True,
        "skip_projector": True,
    }
    assert "fused_all" in rep.plan.LOCATE_FEATURES  # the ceiling of the gap


def test_skip_primary_locate_tokens_and_reliance(
    skip: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    skip.configure()
    rep, plan = skip.rep, skip.rep.plan
    arms = {"A0": {"recon_weight": 0.0}, skip.FIX1: rep.FIX1_OPTIONS, **skip.ARMS}
    synthetic_plan(plan, tmp_path, monkeypatch, arms)
    monkeypatch.setattr(rep, "ROOT", tmp_path / "skip")
    monkeypatch.setattr(skip, "ROOT", tmp_path / "skip")
    monkeypatch.setattr(skip, "RELIANCE_EPISODES", 2)
    plan.decoder("0/1")
    plan.locate("0/1")
    found = json.loads((tmp_path / "plan/locate/fix1-recon05-s0.json").read_text())
    assert "fused_all" in found
    plan.tokens_dyn("0/1")
    monkeypatch.setattr(plan, "arm_ctrl_range", lambda: (-np.ones(8), np.ones(8)))
    plan.tokens_base()
    skip.reliance("0/1")
    rel = json.loads((tmp_path / "skip/reliance/skip-z-s1.json").read_text())
    assert {"reliance_vision", "reliance_proprio_body"} <= set(rel)
    assert skip.fused_ceiling() == pytest.approx(
        np.mean(
            [
                json.loads((tmp_path / f"plan/locate/fix1-recon05-s{s}.json").read_text())[
                    "fused_all"
                ]["per effector"]["babble"]
                for s in plan.SEEDS
            ]
        )
    )
    capsys.readouterr()
    skip.report_primary()
    rep.r2_figure()
    plan.tokens_report(judged=False)
    skip.report_reliance()
    out = capsys.readouterr().out
    for arm in skip.ARMS:
        assert f"| {arm} |" in out
    assert "gap closed" in out and "keeps the arm" in out
    assert "Not judged here" in out and "Pre-registered bar" not in out
    assert "sense reliance" in out
    assert (tmp_path / "skip/figures/rep_r2.png").exists()


def test_skip_trains_the_arms_and_draws_their_curves(
    skip: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "tiny.toml"
    base.write_text(
        "[model]\ndim = 32\nfusion_depth = 1\npredictor_depth = 1\nheads = 2\nvision_size = 32\n"
        "vision_patch = 8\nvision_dim = 32\nvision_depth = 1\nvision_heads = 2\nvision_tokens = 2\n"
        "touch_dim = 32\ntouch_depth = 1\ntouch_heads = 2\ntouch_tokens = 2\nsteps = 4\n"
        "substeps = 4\nsigreg_proj = 64\nsigreg_pool_time = true\n[data]\nstats_episodes = 4\n"
    )
    variety = skip.rep.variety
    for name, value in {
        "BASE": base,
        "DATA": tmp_path / "data",
        "CACHE": tmp_path / "data",
        "ROOT": tmp_path / "runs/variety",
        "HELDOUT": 4,
        "SPACE_PER_SET": 2,
        "CKPTS": (0, 2),
        "STEPS": 2,
        "SEEDS": (0, 1),
    }.items():
        monkeypatch.setattr(variety, name, value)
    monkeypatch.setattr(skip, "ROOT", tmp_path / "runs/skip")
    skip.configure()
    save_episodes(
        tmp_path / "data/babble_v2/humanoid", ("hand_seated_00", "pincer_seated_00"), 5, 0
    )
    save_episodes(tmp_path / "data/variety/humanoid", ("hand_push_00", "pincer_knock_00"), 5, 50)
    monkeypatch.chdir(tmp_path)
    variety.write_split(from_config=False, dry_run=False)
    variety.write_cache(1, variety.SETS)
    for seed in (0, 1):
        variety.train_arm("jepa-babble", seed, 0, 0)
        for arm in skip.rep.fix.ARMS:
            skip.rep.fix.train_arm(arm, seed, 0, 0)
    both = skip.rep.fix.load_layer(tmp_path / "runs/skip/runs/skip-both-s0/ckpt-2.pt")
    cfg = both.model.cfg
    assert cfg.skip_tokens and cfg.skip_projector and cfg.recon == 0.5
    start = skip.rep.fix.load_layer(tmp_path / "runs/skip/runs/skip-z-s0/ckpt-0.pt")
    assert float(start.model.encoder.skip_weight.detach().abs().sum()) == 0.0  # starts as fix1
    trained = skip.rep.fix.load_layer(tmp_path / "runs/skip/runs/skip-z-s0/ckpt-2.pt")
    assert float(trained.model.encoder.skip_weight.detach().abs().sum()) > 0.0  # and learns
    skip.loss_figure()
    assert (tmp_path / "runs/skip/figures/skip_loss.png").exists()
