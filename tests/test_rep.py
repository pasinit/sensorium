"""``scripts/rep_test.py`` (B-rep-0): tiny end-to-end runs on synthetic episodes."""

import importlib.util
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
SIBLINGS = ("plan_test", "fix_test", "variety_test", "jepa_diagnostics")


@pytest.fixture
def rep() -> Iterator[Any]:
    """``scripts/rep_test.py`` with fresh copies of its sibling scripts (``configure`` rewrites
    their globals, so no other test may see them)."""
    sys.path.insert(0, str(SCRIPTS))
    saved = {n: sys.modules.pop(n) for n in SIBLINGS if n in sys.modules}
    spec = importlib.util.spec_from_file_location("rep_test_script", SCRIPTS / "rep_test.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    for name in SIBLINGS:
        sys.modules.pop(name, None)
    sys.modules.update(saved)
    sys.path.remove(str(SCRIPTS))


def test_configure_points_both_studies_at_the_rep_runs(rep: Any) -> None:
    rep.configure()
    assert rep.fix.ROOT == rep.ROOT and set(rep.fix.ARMS) == {rep.FIX1, *rep.ARMS}
    assert rep.fix.ARMS[rep.FIX1] == {"recon_weight": 0.5, "recon_input": "masked"}
    assert rep.plan.OWN == ("A0", rep.FIX1, *rep.ARMS) and rep.plan.RIDGE == ()
    assert rep.plan.ENCODERS["recon-s"] == (rep.ROOT / "runs", "recon-s")
    assert rep.ARMS["recon-s-multi-z"]["summary_tokens"] == 4
    assert rep.ARMS["recon-s-allprop"] == rep.ARMS["recon-s"] | {"recon_proprio": "all"}
    assert len(rep.ARMS) == 4
    assert all(v["recon_weight"] == 0.5 for v in rep.ARMS.values())


def test_rep_decoder_check_open_loop_and_primary(
    rep: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rep.configure()
    plan = rep.plan
    arms = {"A0": {"recon_weight": 0.0}, rep.FIX1: rep.FIX1_OPTIONS, **rep.ARMS}
    synthetic_plan(plan, tmp_path, monkeypatch, arms)
    monkeypatch.setattr(rep, "ROOT", tmp_path / "rep")
    plan.fit()
    assert set(plan.effector_decoders("multi-z", 0)) == {"hand", "pincer"}
    plan.decoder("0/1")
    raw = np.load(tmp_path / "plan/decoder/multi-z-s0.npz")
    assert {"s/per effector/babble", "s3/pooled/variety", "q/babble", "groups/babble"} <= set(
        raw.files
    )
    assert rep.best_token(rep.decoder_runs("multi-z")) in {"s0", "s1", "s2", "s3"}
    assert rep.best_token(rep.decoder_runs("recon-s")) is None  # one token: no control
    plan.openloop("0/1")
    run = np.load(tmp_path / "plan/openloop/own-recon-s-multi-z-s1.npz")
    assert run["decoded_eff"].shape == run["decoded"].shape == (64, 8)
    capsys.readouterr()
    rep.report_primary()
    rep.r2_figure()
    rep.report_openloop()
    out = capsys.readouterr().out
    for arm in rep.ARMS:
        assert f"| {arm} |" in out
    assert "Capacity against structure" in out and "| copy | - |" in out
    assert any(r in out for r in ("improved", "no difference", "worse"))
    for name in ("rep_r2.png", "rep_openloop.png"):
        assert (tmp_path / "rep/figures" / name).exists()


def test_rep_trains_and_probes_the_new_arms(
    rep: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    base = tmp_path / "tiny.toml"
    base.write_text(
        "[model]\ndim = 32\nfusion_depth = 1\npredictor_depth = 1\nheads = 2\nvision_size = 32\n"
        "vision_patch = 8\nvision_dim = 32\nvision_depth = 1\nvision_heads = 2\nvision_tokens = 2\n"
        "touch_dim = 32\ntouch_depth = 1\ntouch_heads = 2\ntouch_tokens = 2\nsteps = 4\n"
        "substeps = 4\nsigreg_proj = 64\nsigreg_pool_time = true\n[data]\nstats_episodes = 4\n"
    )
    variety = rep.variety
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
    monkeypatch.setattr(rep, "STEP", 2)
    monkeypatch.setattr(rep, "ROOT", tmp_path / "runs/rep")
    rep.configure()
    monkeypatch.setattr(rep.fix.diag, "FLOORS", (0, 1))
    save_episodes(
        tmp_path / "data/babble_v2/humanoid", ("hand_seated_00", "pincer_seated_00"), 5, 0
    )
    save_episodes(tmp_path / "data/variety/humanoid", ("hand_push_00", "pincer_knock_00"), 5, 50)
    monkeypatch.chdir(tmp_path)
    variety.write_split(from_config=False, dry_run=False)
    variety.write_cache(1, variety.SETS)
    for seed in (0, 1):
        variety.train_arm("jepa-babble", seed, 0, 0)
        for arm in rep.fix.ARMS:
            rep.fix.train_arm(arm, seed, 0, 0)
    multi = rep.fix.load_layer(tmp_path / "runs/rep/runs/multi-z-s0/ckpt-2.pt")
    assert multi.model.cfg.summary_tokens == 4 and multi.model.cfg.recon_source == "z"
    space = multi.space
    monkeypatch.setattr(variety, "shared_space", lambda: space)
    rep.fix.encode("0/1")
    held = np.load(tmp_path / "runs/rep/enc/held-out_babble/recon-s-multi-z-s1-n2.npz")
    assert held["pre"].shape[1] == held["post"].shape[1] == 4 * 32  # all four tokens
    rep.fix.probe("0/1")
    capsys.readouterr()
    rep.report_battery()
    out = capsys.readouterr().out
    assert "| recon-s | held-out babble | contact AUROCs | pre |" in out
    assert "| multi-z |" in out and "seq/s" in out
