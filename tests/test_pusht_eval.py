"""``scripts/pusht_eval.py`` (B-pusht-0, evaluation): the parts that run without
stable-worldmodel."""

import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch", reason="needs the model extra: uv sync --extra model")
pytest.importorskip("h5py", reason="needs the pusht extra")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"{name}_script", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ev() -> Iterator[Any]:
    sys.path.insert(0, str(SCRIPTS))
    yield load("pusht_eval")
    sys.path.remove(str(SCRIPTS))


def test_zscore_is_sklearns_standard_scaler_as_lewm_fits_it(ev: Any) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(3, 2, (50, 3))
    x[:, 2] = 7.0  # a constant column scales by 1
    x[4, 0] = np.nan  # rows with a NaN are dropped
    z = ev.ZScore.fit(x)
    kept = np.delete(x, 4, axis=0)
    assert np.allclose(z.mean, kept.mean(0)) and np.allclose(z.scale[:2], kept[:, :2].std(0))
    assert z.scale[2] == 1.0
    y = rng.normal(0, 1, (5, 3))
    assert np.allclose(z.inverse_transform(z.transform(y)), y)


def test_lewm_frame_takes_one_channels_first_frame_as_the_evaluator_gives_it(ev: Any) -> None:
    import torch

    from sensorium.pusht_plan import frames64

    rng = np.random.default_rng(0)
    hwc = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    out = ev.lewm_frame(torch.as_tensor(hwc).permute(2, 0, 1))  # as tv_tensors.Image(hwc)
    assert tuple(out.shape) == (3, 64, 64)
    want = torch.as_tensor(frames64(hwc[None])[0]).permute(2, 0, 1).float() / 255.0
    assert torch.equal(out, want)
    with pytest.raises(ValueError, match="one"):
        ev.lewm_frame(torch.as_tensor(hwc[None]).permute(0, 3, 1, 2))


def test_eval_pairs_follow_lewms_draw(ev: Any) -> None:
    lengths = [60, 30, 80, 45]
    episode_idx = np.concatenate([np.full(n, e) for e, n in enumerate(lengths)])
    step_idx = np.concatenate([np.arange(n) for n in lengths])
    rows, episodes, starts = ev.eval_pairs(episode_idx, step_idx, 20, 42, 25)
    again = ev.eval_pairs(episode_idx, step_idx, 20, 42, 25)
    assert all(np.array_equal(a, b) for a, b in zip((rows, episodes, starts), again, strict=True))
    assert len(rows) == 20 and np.all(np.diff(rows) > 0)  # sorted, distinct
    for e, s in zip(episodes, starts, strict=True):
        assert s + 25 <= lengths[e] - 1  # the goal is inside the episode
    other = ev.eval_pairs(episode_idx, step_idx, 20, 43, 25)[0]
    assert not np.array_equal(rows, other)
    with pytest.raises(ValueError, match="not enough"):
        ev.eval_pairs(episode_idx, step_idx, 1000, 42, 25)


def test_pose_targets_and_probe(ev: Any) -> None:
    rng = np.random.default_rng(1)
    state = rng.normal(0, 1, (200, 7))
    y = ev.pose_targets(state)
    assert np.allclose(y[:, 0], state[:, 2]) and np.allclose(y[:, 2], np.sin(state[:, 4]))
    feats = np.concatenate([y @ rng.normal(0, 1, (4, 6)), rng.normal(0, 1, (200, 3))], 1)
    r2 = ev.pose_r2(feats, y, np.arange(200) // 10)
    assert all(v > 0.99 for v in r2.values())  # linear in the features
    held = ev.pose_episodes(1000, list(range(0, 1000, 2)))
    assert len(held) == ev.POSE_EPISODES and all(i % 2 == 1 for i in held)


def test_report_reads_both_protocols_and_the_pose_probes(
    ev: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ev, "EVAL", tmp_path / "eval")
    monkeypatch.setattr(ev, "POSE", tmp_path / "pose")
    (tmp_path / "eval").mkdir()
    (tmp_path / "pose").mkdir()
    rng = np.random.default_rng(2)
    for pairs, seed in ev.PROTOCOLS.values():
        for name in ["lewm-lewm-tiny", *(f"ours-{a}" for a in ev.OURS)]:
            for s in ev.SEEDS:
                hits = (rng.random(pairs) < 0.5).astype(int).tolist()
                path = tmp_path / "eval" / f"{name}-s{s}-{pairs}-{seed}.json"
                path.write_text(json.dumps({"episode_successes": hits}))
    pose = {"model": "ours:a0-pix-s0"} | {
        layer: {"x": 0.9, "y": 0.8, "sin": 0.5, "cos": 0.4} for layer in ("state", "pre")
    }
    (tmp_path / "pose" / "ours-a0-pix-s0.json").write_text(json.dumps(pose))
    monkeypatch.setattr(ev, "ROOT", tmp_path / "pusht")
    monkeypatch.setattr(ev, "LEWM_LOGS", tmp_path / "lewm_logs")
    run = tmp_path / "pusht/runs/a0-pix-s0"
    run.mkdir(parents=True)
    rows = [{"step": k * 50, "pred": 1.0 / (k + 1), "reg": 0.5} for k in range(1, 6)]
    (run / "log.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    csv = tmp_path / "lewm_logs/lewm-tiny-s0/version_0/metrics.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("step,train/pred_loss,train/sigreg_loss\n49,0.9,3.0\n99,0.5,2.0\n")
    assert ev.lewm_curve(0)["reg"].tolist() == pytest.approx([0.27, 0.18])
    flat = tmp_path / "lewm_logs/lewm-tiny-s1/metrics.csv"  # the VM's copy: no version folder
    flat.parent.mkdir(parents=True)
    flat.write_text("epoch,train/sigreg_loss_step,global_step,train/pred_loss_step\n0,1.0,49,0.7\n")
    one = ev.lewm_curve(1)
    assert one["clips"].tolist() == [49 * 128] and one["pred"].tolist() == [0.7]
    ev.report()
    out = capsys.readouterr().out
    assert (tmp_path / "pusht/figures/pusht_curves.png").exists()
    assert "primary (200 pairs, rng 43)" in out and "published (50 pairs, rng 42)" in out
    assert "| a0-pix - lewm-tiny |" in out and "seeds unpaired" in out
    assert "| fix1-pix - a0-pix |" in out and "| ours:a0-pix-s0 | state |" in out


def test_the_launcher_names_the_csv_logs_after_the_run() -> None:
    launch = load("lewm_launch")
    assert launch.run_name(["img_size=64", "output_model_name=lewm-tiny-s1"]) == "lewm-tiny-s1"
    assert launch.run_name([]) == "lewm"


def test_the_launcher_runs_train_py_as_main_with_the_remaining_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch = load("lewm_launch")
    script = tmp_path / "train.py"
    script.write_text(
        "import sys, json\n"
        "if __name__ == '__main__':\n"
        "    open(sys.argv[0] + '.ran', 'w').write(json.dumps(sys.argv[1:]))\n"
    )
    monkeypatch.setattr(launch, "__file__", str(tmp_path / "lewm_launch.py"))
    monkeypatch.setattr(launch, "with_csv_logger", lambda name: None)
    monkeypatch.setattr(launch, "with_final_save", lambda: None)
    monkeypatch.setattr(sys, "argv", ["lewm_launch.py", "--run-seed=3", "img_size=64"])
    launch.main()
    assert json.loads((tmp_path / "train.py.ran").read_text()) == ["img_size=64"]
