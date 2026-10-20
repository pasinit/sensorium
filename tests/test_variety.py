"""``scripts/variety_test.py`` (B-variety-0): splits, verdicts and a tiny end-to-end run."""

import json
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="needs the model extra: uv sync --extra model")

from test_model import privileged_episode  # noqa: E402

from sensorium.model.paired import Contrast  # noqa: E402
from sensorium.model.pretrain import load_layer  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def script() -> Iterator[Any]:
    """``scripts/variety_test.py`` as a module (it imports ``scale_test`` from ``scripts/``)."""
    import importlib.util  # noqa: PLC0415

    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(
            "variety_test_script", SCRIPTS / "variety_test.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPTS))
    yield module
    module.load_predictions.cache_clear()
    module.shared_space.cache_clear()


def write_config(path: Path, blocks: dict[str, tuple[str, int]]) -> Path:
    """A generator config with one policy per block: ``{block: (policy, episodes)}``."""
    lines = ['out_dir = "data/x"', "master_seed = 9"]
    for block, (policy, n) in blocks.items():
        effector = block.split("_")[0]
        lines += [
            f"[humanoid.{block}]",
            f'policies = ["{policy}"]',
            f"episodes_per_policy = {n}",
            "steps = 600",
            f'settings = {{ effector = "{effector}", posture = "seated" }}',
        ]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_variety_splits_are_stratified_disjoint_and_matched_in_size(
    script: Any, tmp_path: Path
) -> None:
    programs = ("push", "slide", "press", "knock", "door", "topple", "lift_drop")
    blocks = {f"{e}_{p}_00": ("experiment", 24) for e in ("hand", "pincer") for p in programs}
    blocks |= {f"{e}_{p}_01": ("experiment", 24) for e in ("hand", "pincer") for p in programs}
    blocks |= {f"{e}_babble_00": ("seated_babble", 24) for e in ("hand", "pincer")}
    variety = script.scale.expected_names(write_config(tmp_path / "variety.toml", blocks))
    seated = {
        f"{e}_seated_{i:02d}": ("seated_babble", 24) for e in ("hand", "pincer") for i in range(15)
    }
    babble = script.scale.expected_names(write_config(tmp_path / "babble.toml", seated))
    assert len(variety) == len(babble) == 720
    full = script.make_splits({"variety": variety, "babble": babble})
    assert {len(full[s]["pretrain"]) for s in script.SETS} == {620}
    # a time-capped run: one babble block stopped after 14 of its 24 episodes
    capped = [
        n
        for n in babble
        if not (script.scale.block_of(n) == "hand_seated_03" and n[-7:-4] >= "014")
    ]
    assert len(capped) == 710
    names = {"variety": variety, "babble": capped}
    splits = script.make_splits(names)
    assert splits == script.make_splits(names)
    for name in script.SETS:
        held, pre, spare = (splits[name][k] for k in ("heldout", "pretrain", "spare"))
        assert (len(held), len(pre)) == (100, 610)  # matched: the smaller set's 710 - 100
        assert len(set(held) | set(pre) | set(spare)) == len(names[name])
        effectors = Counter(script.scale.block_of(n).split("_")[0] for n in held)
        assert effectors == {"hand": 50, "pincer": 50}
    per_block = Counter(script.scale.block_of(n) for n in splits["variety"]["heldout"])
    assert len(per_block) == 30 and set(per_block.values()) <= {3, 4}  # every program held out
    assert script.program_of("hand_lift_drop_00") == "lift_drop"
    assert script.program_of("pincer_seated") == "seated"
    with pytest.raises(ValueError, match="more than 100"):
        script.make_splits({"variety": variety, "babble": babble[:90]})


@pytest.mark.parametrize(
    ("babble", "variety", "reading"),
    [
        (0.10, 0.10, "variety helps on both test sets"),
        (0.01, 0.10, "variety helps on its own distribution only"),
        (0.10, 0.01, "variety helps on babbling only"),
        (-0.10, 0.10, "mixed"),
        (-0.10, 0.01, "variety harms"),
        (0.01, -0.10, "variety harms"),
        (0.01, -0.01, "no effect"),
    ],
)
def test_variety_verdicts_cover_each_preregistered_outcome(
    script: Any, babble: float, variety: float, reading: str
) -> None:
    # SE 0.01: 0.10 is resolved (MDD 0.028), 0.01 is not
    vb = script.verdict(Contrast(babble, 0.01, 0.0, 3, 3))
    vv = script.verdict(Contrast(variety, 0.01, 0.0, 3, 3))
    assert script.reading(vb, vv) == reading
    assert script.verdict(None) == "incomplete" and script.reading("incomplete", vv) == "incomplete"
    assert script.level(Contrast(-0.1, 0.01, 0.0, 3, 5)) == "below"


def save_episodes(root: Path, blocks: tuple[str, ...], per_block: int, seed: int) -> None:
    """Probe-ready synthetic episodes with ``eff_`` proprio and hidden properties, named as the
    generator names them."""
    root.mkdir(parents=True)
    k = seed
    for block in blocks:
        effector = block.split("_")[0]
        for i in range(per_block):
            ep = privileged_episode(k, steps=96, effector=effector)
            rng = np.random.default_rng(k)
            ep.proprio_names = [*ep.proprio_names, "eff_torso"]
            ep.proprio = np.concatenate([ep.proprio, rng.normal(size=(96, 1))], 1).astype(
                np.float32
            )
            props = np.array([rng.uniform(0.2, 2.0), rng.uniform(0.3, 1.0), 0.02])
            ep.privileged["hidden_properties"] = np.tile(props, (96, 1, 1)).astype(np.float32)
            ep.meta |= {"object_names": ["obj_box"]}
            ep.save(root / f"humanoid-{block}-policy-{i:03d}.npz")
            k += 1


def test_variety_end_to_end_tiny(
    script: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    base = tmp_path / "tiny.toml"
    base.write_text(
        "[model]\ndim = 32\nfusion_depth = 1\npredictor_depth = 1\nheads = 2\nvision_size = 32\n"
        "vision_patch = 8\nvision_dim = 32\nvision_depth = 1\nvision_heads = 2\nvision_tokens = 2\n"
        "touch_dim = 32\ntouch_depth = 1\ntouch_heads = 2\ntouch_tokens = 2\nsteps = 4\n"
        "substeps = 4\nsigreg_proj = 64\nsigreg_pool_time = true\n[data]\nstats_episodes = 4\n"
    )
    for name, value in {
        "BASE": base,
        "DATA": tmp_path / "data",
        "CACHE": tmp_path / "data",
        "ROOT": tmp_path / "runs",
        "HELDOUT": 4,
        "SPACE_PER_SET": 2,
        "CKPTS": (0, 2),
        "STEPS": 2,
        "SUP_STEPS": 2,
        "SEEDS": (0, 1),
        "FLOOR_SEEDS": (0,),
    }.items():
        monkeypatch.setattr(script, name, value)
    save_episodes(
        tmp_path / "data/babble_v2/humanoid", ("hand_seated_00", "pincer_seated_00"), 4, 0
    )
    save_episodes(tmp_path / "data/variety/humanoid", ("hand_push_00", "pincer_knock_00"), 5, 50)
    monkeypatch.chdir(tmp_path)  # no configs/ here: the split warns about nothing
    script.write_split(from_config=False, dry_run=False)
    split = json.loads((tmp_path / "runs/split.json").read_text())
    assert {len(split[s]["heldout"]) for s in script.SETS} == {4}
    assert {len(split[s]["pretrain"]) for s in script.SETS} == {4}  # matched to babble's 8 - 4
    script.write_cache(1, script.SETS)
    for arm in ("jepa-babble", "jepa-variety", "sup-babble"):
        for seed in (0, 1):
            script.train_arm(arm, seed, 0, 0)
    layer = load_layer(tmp_path / "runs/runs/jepa-variety-s0/ckpt-2.pt")
    assert not any(k.startswith("eff_") for g in layer.space.keys.values() for k in g)
    script.probe_all(0, 1)
    jobs = script.probe_jobs()
    assert all(script.pred_path(t, n).exists() for t, n in jobs)
    assert len(jobs) == 2 * (
        1 + 1 + 2 * 2 * 2 + 2
    )  # floor, raw, 2 SSL arms x 2 seeds x 2 ckpts, sup
    for test in script.SETS:  # the same seed is the same initial encoder in every arm and the floor
        floor = script.load_predictions(test, "floor-vp-s0")
        for arm in ("jepa-babble", "jepa-variety"):
            assert np.allclose(script.load_predictions(test, f"{arm}-s0-n0").pred, floor.pred)
        assert "hidden_log_mass_r2" in script.probe_json(test, "floor-vp-s0")["hidden"]
    script.report()
    out = capsys.readouterr().out
    assert "Pre-registered outcomes" in out and "jepa" in out and "EXPLORATORY" in out
    assert (tmp_path / "runs/figures/variety_probe_babble.png").exists()
    assert (tmp_path / "runs/figures/variety_curves_variety.png").exists()
