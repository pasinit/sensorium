"""The E1 pilot's four per-arm training configs (``configs/model/e1-pilot-*.toml``)."""

from dataclasses import asdict
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="needs the model extra: make setup")

from sensorium.model.pretrain import load_config  # noqa: E402
from sensorium.model.sensory_layer import ModelConfig  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "model"
ARMS = ("v-expert", "v-babble", "all-expert", "all-babble")
V = {"vision", "proprio_body", "vestibular", "proprio_effector"}


def test_the_four_arms_differ_only_in_senses_and_pool() -> None:
    cfgs = {arm: load_config(CONFIGS / f"e1-pilot-{arm}.toml") for arm in ARMS}
    ref = cfgs["v-expert"]
    for arm, cfg in cfgs.items():
        assert cfg.name == f"e1-pilot-{arm}" and cfg.seed == 0
        assert cfg.train == ref.train and cfg.augment == ref.augment  # identical budgets
        model = asdict(cfg.model) | {"senses": None}
        assert model == asdict(ref.model) | {"senses": None}
        senses = set(cfg.model.senses)
        assert senses == (V if arm.startswith("v-") else V | {"touch", "audio"})
        kind = arm.split("-")[1]
        assert kind in cfg.data.pool and all(kind in p for p in cfg.data.heldout_paths)
    m, t = ref.model, ref.train
    assert m.recon == 0.0 and m.objective == "sigreg"  # A0
    assert (m.dim, m.fusion_depth, m.predictor_depth) == (
        ModelConfig().dim,
        ModelConfig().fusion_depth,
        ModelConfig().predictor_depth,
    )  # the v0 widths
    assert (t.mean_stop, t.mean_stop_step, t.heldout_every, t.resume_every) == (
        0.8,
        2000,
        2000,
        500,
    )
    assert t.save_steps[-1] == t.steps and t.batch_size == 128
