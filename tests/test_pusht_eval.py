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
    flat.write_text(  # RegistryLogger's header, as the B-pusht-0 VM logged it
        "epoch,fit/loss,fit/pred_loss,fit/sigreg_loss,hardware/cpu_percent,hparams/lr_model_opt,step\n"
        "0,0.8,0.7,1.0,90,0.0001,49\n"
    )
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


def test_success_rules_split_pushts_rule_into_block_and_agent(ev: Any) -> None:
    goal = np.array([[100.0, 100.0, 200.0, 200.0, 0.1, 0.0, 0.0]] * 5)
    final = goal.copy()
    final[1, :2] += [30.0, 0.0]  # the agent far, the block on its goal
    final[2, 2:4] += [0.0, 25.0]  # the block far, the agent on its goal
    final[3, 4] = 2 * np.pi - 0.1  # the block turned by 0.2 rad across the wrap: within pi/9
    final[4, [0, 2]] += 15.0  # each half within 20, the 4-vector at 21.2: not a success
    rules = ev.success_rules(final, goal)
    assert rules["full"].tolist() == [True, False, False, True, False]
    assert rules["block"].tolist() == [True, True, False, True, True]
    assert rules["agent"].tolist() == [True, False, True, True, True]


def test_plan_names_keep_the_first_runs_files_and_mark_the_variants(ev: Any) -> None:
    assert ev.plan_name("ours:a0-pix-s0", 200, 43) == "ours-a0-pix-s0-200-43"
    assert (
        ev.plan_name("lewm224:quentinll/lewm-pusht", 50, 42) == "lewm224-quentinll_lewm-pusht-50-42"
    )
    assert ev.plan_name("ours:fix1-pix-s2", 200, 43, "recal") == "ours-fix1-pix-s2-recal-200-43"
    name = ev.plan_name("ours:a0-pix-agent-s1", 200, 43, "current")
    assert name == "ours-a0-pix-agent-s1-goalcurrent-200-43"


def test_val_sample_takes_only_clips_our_windows_reach(
    ev: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ev, "PRED_CLIPS", 5)
    lengths = [60, 60]
    val = [[0, 0], [0, 3], [0, 4], [0, 10], [0, 40], [0, 44], [0, 45], [1, 4], [1, 30]]
    got = ev.val_sample(val, lengths)  # starts 0-3 and > 44 cannot be taken (span 20)
    assert len(set(got)) == 5 and got == sorted(got)
    assert all(4 <= t <= 44 for _, t in got)
    assert ev.val_sample(val, lengths) == got  # a fixed draw
    monkeypatch.setattr(ev, "PRED_CLIPS", 7)
    with pytest.raises(ValueError, match="only 6"):
        ev.val_sample(val, lengths)


def test_pred_loss_with_the_per_frame_mask_is_fix1s_training_pred(ev: Any, tmp_path: Path) -> None:
    import torch

    from sensorium import pusht
    from sensorium.model.data import (
        LoaderConfig,
        compute_stats,
        header_of,
        load_episode,
        sequence_with_ends,
    )
    from sensorium.model.layout import ChannelSpace
    from sensorium.model.sensory_layer import ModelConfig, SensoryModel

    rng = np.random.default_rng(0)
    steps = 30
    ep = pusht.to_episode(
        0,
        {
            "pixels": rng.integers(0, 255, (steps, 224, 224, 3), dtype=np.uint8),
            "action": rng.uniform(-0.5, 0.5, (steps, 2)),
            "proprio": rng.normal(0, 1, (steps, 4)),
            "state": rng.normal(0, 1, (steps, 7)),
        },
    )
    space = ChannelSpace.from_headers([header_of(ep)])
    path = tmp_path / "pusht-00000.npz"
    ep.save(path)
    compute_stats([path], space, 1)
    cfg = ModelConfig(
        senses=("vision",),
        dim=32,
        fusion_depth=1,
        predictor_depth=1,
        heads=2,
        vision_size=64,
        vision_patch=8,
        vision_dim=32,
        vision_depth=1,
        vision_heads=2,
        vision_tokens=2,
        steps=4,
        substeps=5,
        sigreg_proj=64,
        recon_weight=0.5,
        recon_input="masked",
    )
    torch.manual_seed(0)
    model = SensoryModel(cfg, space)
    loader = LoaderConfig(steps=4, substeps=5, view_size=64, crop=1.0)
    loaded = load_episode(ep, space, model.data_senses)
    seqs = [
        sequence_with_ends(loaded, space, model.data_senses, loader, rng, False, start=s)[0]
        for s in (0, 3, 7, 10)
    ]
    batch = {k: torch.as_tensor(np.stack([q[k] for q in seqs])) for k in seqs[0]}
    model.train()
    torch.manual_seed(ev.PRED_SEED)
    with torch.no_grad():
        want = float(model(batch)["pred"])  # fix1's one masked pass, as in training
    state = {k: v.clone() for k, v in model.state_dict().items()}
    got = ev.pred_loss(model, [batch], "per-frame", "batch")
    assert got == pytest.approx(want, rel=1e-5)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())  # untouched
    assert model.recon is not None
    clean = ev.pred_loss(model, [batch], "clean", "eval")
    shared = ev.pred_loss(model, [batch], "shared", "eval")
    assert np.isfinite(clean) and np.isfinite(shared) and clean != shared


def test_check_report_scores_the_rules_and_contrasts(
    ev: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ev, "ROOT", tmp_path / "pusht")
    monkeypatch.setattr(ev, "EVAL", tmp_path / "pusht/eval")
    monkeypatch.setattr(ev, "CHECK", tmp_path / "pusht/check")
    for d in ("eval", "eval_first", "check"):
        (tmp_path / "pusht" / d).mkdir(parents=True)
    pairs, seed = ev.PROTOCOLS["primary"]
    rng = np.random.default_rng(3)
    goal = np.tile([100.0, 100.0, 200.0, 200.0, 0.1, 0.0, 0.0], (pairs, 1))
    for family, variant in ev.CHECK_ARMS.values():
        for path in ev.check_paths(tmp_path / "pusht/eval", family, variant):
            final = goal + rng.normal(0, 12, goal.shape) * [1, 1, 1, 1, 0.01, 1, 1]
            hits = ev.success_rules(final, goal)["full"].astype(int).tolist()
            record = {"episode_successes": hits, "final_states": final.tolist()}
            path.write_text(json.dumps(record | {"goal_states": goal.tolist()}))
            if variant == "goal":
                (tmp_path / "pusht/eval_first" / path.name).write_text(json.dumps(record))
    pred = {"model": "ours:fix1-pix-s0"} | {
        bn: dict(zip(ev.MASKS, (0.1, 0.7, 0.3), strict=True)) for bn in ("eval", "batch")
    }
    (tmp_path / "pusht/check/pred-fix1-pix-s0.json").write_text(json.dumps(pred))
    plans = ev.check_plans()
    assert set(plans) == set(ev.CHECK_ARMS)
    assert plans["a0-pix"]["full"].shape == (3, pairs)
    ev.check_report()
    out = capsys.readouterr().out
    assert (tmp_path / "pusht/figures/pusht_check_rules.png").exists()
    assert "| a0-pix-agent-cur | " in out and "| agent | a0-pix-agent - a0-pix |" in out
    assert "| lewm-tiny | 1.000 | 1.000 |" in out  # the full rule is the evaluator's
    assert "| ours:fix1-pix-s0 | eval | 0.1000 | 0.7000 | 0.3000 |" in out


def test_ratio_report_scores_fix1_at_each_ratio_against_a0_and_fix1(
    ev: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ev, "ROOT", tmp_path / "pusht")
    monkeypatch.setattr(ev, "EVAL", tmp_path / "pusht/eval")
    monkeypatch.setattr(ev, "CHECK", tmp_path / "pusht/check")
    monkeypatch.setattr(ev, "POSE", tmp_path / "pusht/pose")
    for d in ("eval", "check", "pose"):
        (tmp_path / "pusht" / d).mkdir(parents=True)
    pairs, _ = ev.PROTOCOLS["primary"]
    rng = np.random.default_rng(4)
    goal = np.tile([100.0, 100.0, 200.0, 200.0, 0.1, 0.0, 0.0], (pairs, 1))
    for family, variant in ev.RATIO_ARMS.values():
        for s, path in enumerate(ev.check_paths(tmp_path / "pusht/eval", family, variant)):
            final = goal + rng.normal(0, 12, goal.shape) * [1, 1, 1, 1, 0.01, 1, 1]
            hits = ev.success_rules(final, goal)["full"].astype(int).tolist()
            path.write_text(
                json.dumps(
                    {"episode_successes": hits, "final_states": final.tolist()}
                    | {"goal_states": goal.tolist()}
                )
            )
            run = tmp_path / f"pusht/runs/{family}-s{s}"
            run.mkdir(parents=True, exist_ok=True)
            rows = [{"step": k * 50, "pred": 0.5, "reg": 0.2, "recon": 0.01} for k in range(1, 4)]
            (run / "log.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            pose = {"model": f"ours:{family}-s{s}"} | {
                layer: {"x": 0.9, "y": 0.8, "sin": 0.3, "cos": 0.2} for layer in ("state", "pre")
            }
            (tmp_path / f"pusht/pose/ours-{family}-s{s}.json").write_text(json.dumps(pose))
    pred = {"model": "ours:fix1-pix-r50-s0"} | {
        bn: dict(zip(ev.MASKS, (0.1, 0.4, 0.3), strict=True)) for bn in ("eval", "batch")
    }
    pred["clean_var"] = {"eval": 0.5, "batch": 0.2}
    (tmp_path / "pusht/check/pred-fix1-pix-r50-s0.json").write_text(json.dumps(pred))
    other = pred | {"model": "ours:other-arm-s0"}  # not a follow-up run: left out
    (tmp_path / "pusht/check/pred-other-arm-s0.json").write_text(json.dumps(other))
    ev.ratio_report()
    out = capsys.readouterr().out
    assert (tmp_path / "pusht/figures/pusht_ratio_rules.png").exists()
    assert (tmp_path / "pusht/figures/pusht_ratio_curves.png").exists()
    assert (
        "| block | fix1-pix-r30 - fix1-pix |" in out and "| full | fix1-pix-r50 - a0-pix |" in out
    )
    assert "| ours:fix1-pix-r50-s0 | eval | 0.1000 | 0.4000 | 0.3000 |" in out
    assert "| ours:fix1-pix-r50-s0 | batch | 0.1000 | 0.2000 | 0.500 |" in out  # pred / var
    assert "other-arm" not in out and "| block | a0-pix-agent-imggoal - a0-pix |" in out
    assert (tmp_path / "pusht/figures/pusht_imggoal_curves.png").exists()
    assert "| ours:fix1-pix-r30-s2 | pre | 0.900 |" in out


def test_recalibrate_reestimates_only_the_batchnorm_statistics(ev: Any) -> None:
    import torch

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4))
    model.embed = model.forward  # type: ignore[assignment]  # the one call recalibrate makes
    batches = [torch.randn(16, 3) * 2 + 1 for _ in range(3)]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    net = ev.recalibrate(model, batches)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())  # a copy
    x = torch.cat([model[0](b) for b in batches]).detach()
    bn = net[1]
    assert torch.allclose(bn.running_mean, x.mean(0), atol=1e-5)
    assert torch.equal(net.state_dict()["0.weight"], before["0.weight"]) and not net.training


def test_state_variance_is_what_a_mean_predictor_would_score(ev: Any) -> None:
    import torch

    class Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = torch.nn.BatchNorm1d(3)

        def embed(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
            x = batch["x"]
            y: torch.Tensor = self.norm(x.flatten(0, 1))
            return y.view(x.shape)

    torch.manual_seed(0)
    model = Encoder()
    mean = model.norm.running_mean
    assert mean is not None
    mean.fill_(1.0)  # eval statistics differ from the batch's own
    batches = [{"x": torch.randn(4, 2, 3) * 3 + 2} for _ in range(2)]
    want = np.mean([float(b["x"].flatten(0, 1).var(0, unbiased=False).mean()) for b in batches])
    assert ev.state_variance(model, batches, "eval") == pytest.approx(want, rel=1e-4)  # BN eps
    assert ev.state_variance(model, batches, "batch") == pytest.approx(1.0, rel=1e-3)
    assert float(mean[0]) == 1.0 and model.training  # the model is untouched
