"""``sensorium.pusht_plan``: our layer as a stable-worldmodel cost model (B-pusht-0)."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="needs the model extra: uv sync --extra model")
pytest.importorskip("h5py", reason="needs the pusht extra")

from sensorium import pusht  # noqa: E402
from sensorium.episode import Episode  # noqa: E402
from sensorium.model.data import LoaderConfig, compute_stats, header_of  # noqa: E402
from sensorium.model.layout import ChannelSpace  # noqa: E402
from sensorium.model.planning import own_rollout, states_at  # noqa: E402
from sensorium.model.sensory_layer import ModelConfig, SensoryLayer, SensoryModel  # noqa: E402
from sensorium.pusht_plan import SensoryCostModel, frames64  # noqa: E402

CFG = ModelConfig(
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
    recon_weight=0.0,
)


class Affine:
    """A stand-in for sklearn's ``StandardScaler`` (``x * scale + mean`` back)."""

    def __init__(self, mean: float, scale: float) -> None:
        self.mean, self.scale = mean, scale

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x) * self.scale + self.mean


def source(steps: int = 30, seed: int = 0) -> dict[str, np.ndarray]:
    """Source arrays of a Push-T-shaped episode (224 px), with constant proprio."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 255, (steps, 224, 224, 3), dtype=np.uint8)
    return {
        "pixels": pixels,
        "action": rng.uniform(-0.5, 0.5, (steps, 2)),
        "proprio": np.tile([[200.0, 150.0, 3.0, -2.0]], (steps, 1)),
        "state": rng.normal(0, 1, (steps, 7)),
    }


def layer_for(senses: tuple[str, ...], ep: Episode, tmp_path: object) -> SensoryLayer:
    space = ChannelSpace.from_headers([header_of(ep)])
    path = Path(f"{tmp_path}") / "pusht-00000.npz"
    ep.save(path)
    compute_stats([path], space, 1)
    torch.manual_seed(0)
    model = SensoryModel(replace(CFG, senses=senses), space)
    return SensoryLayer(model, space, LoaderConfig(steps=4, substeps=5, view_size=64, crop=1.0))


def test_frames64_is_the_training_pipeline() -> None:
    arrays = source()
    ep = pusht.to_episode(0, arrays)
    assert np.array_equal(frames64(arrays["pixels"]), pusht.lewm_frames(ep))
    with pytest.raises(ValueError, match="224"):
        frames64(arrays["pixels"][:, :64])


@pytest.mark.parametrize("senses", [("vision",), ("vision", "proprio_effector")])
def test_encode_matches_the_trained_layers_states(
    senses: tuple[str, ...], tmp_path: object
) -> None:
    arrays = source()
    ep = pusht.to_episode(0, arrays)
    layer = layer_for(senses, ep, tmp_path)
    cost = SensoryCostModel(layer, header_of(ep), Affine(0.0, 1.0), Affine(0.0, 1.0))
    ends = np.array([4, 9, 17])
    got = cost.encode(arrays["pixels"][ends], arrays["proprio"][ends]).numpy()
    want = states_at(layer, ep, ends)  # the episode path (constant proprio: the same window)
    assert np.allclose(got, want, atol=1e-5)


def test_candidates_become_our_action_chunks_and_costs(tmp_path: object) -> None:
    arrays = source()
    ep = pusht.to_episode(0, arrays)
    layer = layer_for(("vision",), ep, tmp_path)
    scaler = Affine(0.1, 0.5)
    cost = SensoryCostModel(layer, header_of(ep), scaler)
    rng = np.random.default_rng(1)
    b, n, h = 2, 3, 5
    cand = torch.as_tensor(rng.normal(0, 1, (b, n, h, 10)), dtype=torch.float32)
    chunks = cost.chunks(cand.reshape(b * n, h, -1))
    assert chunks.shape == (b * n, h, 5, layer.space.size("action"))
    raw = cand.numpy().reshape(b * n, h, 5, 2) * 0.5 + 0.1
    cols = layer.space.columns(header_of(ep))
    assert np.allclose(chunks[..., cols.dst["action"]], np.clip(raw, -1, 1), atol=1e-6)
    frames = arrays["pixels"][[3, 8]]
    goals = arrays["pixels"][[20, 25]]
    info = {  # as the solver passes it: every entry repeated over the candidates, a history of 1
        "pixels": np.repeat(frames[:, None, None], n, axis=1),
        "goal": np.repeat(goals[:, None, None], n, axis=1),
    }
    costs = cost.get_cost(info, cand)
    assert costs.shape == (b, n)
    start = cost.encode(frames).numpy()
    goal = cost.encode(goals)
    final = own_rollout(layer, np.repeat(start, n, 0), chunks)[:, -1]
    want = ((torch.as_tensor(final, dtype=torch.float32).view(b, n, -1) - goal[:, None]) ** 2).sum(
        -1
    )
    assert torch.allclose(costs, want, atol=1e-4)


def test_a_proprio_layer_needs_the_evaluators_proprio_scaler(tmp_path: object) -> None:
    ep = pusht.to_episode(0, source())
    layer = layer_for(("vision", "proprio_effector"), ep, tmp_path)
    with pytest.raises(ValueError, match="proprio scaler"):
        SensoryCostModel(layer, header_of(ep), Affine(0.0, 1.0))
