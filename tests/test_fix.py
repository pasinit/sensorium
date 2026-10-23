"""``scripts/fix_test.py`` (B-fix-0): a tiny end-to-end run on synthetic episodes."""

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch", reason="needs the model extra: uv sync --extra model")

from test_variety import save_episodes  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def fix() -> Iterator[Any]:
    """``scripts/fix_test.py`` as a module (it imports its sibling scripts from ``scripts/``)."""
    import importlib.util  # noqa: PLC0415

    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("fix_test_script", SCRIPTS / "fix_test.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    module.predictions.cache_clear()
    module.variety.load_predictions.cache_clear()
    sys.path.remove(str(SCRIPTS))


def test_fix_end_to_end_tiny(
    fix: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    base = tmp_path / "tiny.toml"
    base.write_text(
        "[model]\ndim = 32\nfusion_depth = 1\npredictor_depth = 1\nheads = 2\nvision_size = 32\n"
        "vision_patch = 8\nvision_dim = 32\nvision_depth = 1\nvision_heads = 2\nvision_tokens = 2\n"
        "touch_dim = 32\ntouch_depth = 1\ntouch_heads = 2\ntouch_tokens = 2\nsteps = 4\n"
        "substeps = 4\nsigreg_proj = 64\nsigreg_pool_time = true\n[data]\nstats_episodes = 4\n"
    )
    variety = fix.variety
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
    monkeypatch.setattr(fix.diag, "FLOORS", (0, 1))
    monkeypatch.setattr(fix, "ROOT", tmp_path / "runs/fix")
    monkeypatch.setattr(
        fix,
        "ARMS",
        {
            "recon05": {"recon_weight": 0.5},
            "drop02": {"proprio_dropout": 0.2},
            "fix1-recon05": {"recon_weight": 0.5, "recon_input": "masked"},
        },
    )
    save_episodes(
        tmp_path / "data/babble_v2/humanoid", ("hand_seated_00", "pincer_seated_00"), 5, 0
    )
    save_episodes(tmp_path / "data/variety/humanoid", ("hand_push_00", "pincer_knock_00"), 5, 50)
    monkeypatch.chdir(tmp_path)
    variety.write_split(from_config=False, dry_run=False)
    variety.write_cache(1, variety.SETS)
    for seed in (0, 1):
        for arm in ("jepa-babble", "mae-babble"):
            variety.train_arm(arm, seed, 0, 0)
        for arm in fix.ARMS:
            fix.train_arm(arm, seed, 0, 0)
    rows = (tmp_path / "runs/fix/runs/recon05-s0/log.jsonl").read_text().splitlines()
    assert "recon" in json.loads(rows[-1])
    space = fix.load_layer(tmp_path / "runs/variety/runs/jepa-babble-s0/ckpt-2.pt").space
    monkeypatch.setattr(variety, "shared_space", lambda: space)
    names = fix.encoder_names()
    assert len(names) == 2 + (2 + 3) * 2 * 2  # floors, 2 references + 3 arms x 2 seeds x 2 steps
    fix.encode("0/2")
    fix.encode("1/2")
    fix.probe("0/1")
    held = tmp_path / "runs/fix/enc/held-out_babble"
    (held / "recon05-s0-n2.npz").unlink()  # probed already: a later encode must not redo it
    fix.encode("0/1")
    assert not (held / "recon05-s0-n2.npz").exists()
    for label in fix.EVAL_SETS:
        for score in fix.SCORES:
            pred = fix.predictions(label, score, "pre", "recon05-s1-n2")
            assert pred is not None and "mean" in pred.scores()
    capsys.readouterr()
    fix.report()
    out = capsys.readouterr().out
    assert "Pre-registered verdict" in out and "recon05" in out
    assert "B-fix-1" in out and "fix1-recon05 - recon05" in out
    assert (tmp_path / "runs/fix/figures/fix1_losses.png").exists()
    assert (tmp_path / "runs/fix/figures/fix_probe_held-out_babble.png").exists()
    assert (tmp_path / "runs/fix/figures/fix_curves.png").exists()
