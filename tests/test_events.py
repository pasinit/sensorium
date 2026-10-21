"""Event-weighted sampling (``sensorium.model.events``) and its use by the sequence samplers."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch", reason="needs the model extra: uv sync --extra model")

from test_model import LOADER, saved_privileged_episodes  # noqa: E402

from sensorium.model import events  # noqa: E402
from sensorium.model.data import SequenceDataset  # noqa: E402
from sensorium.model.frame_cache import build_cache  # noqa: E402
from sensorium.model.layout import ChannelSpace, check_senses, read_header  # noqa: E402


def test_step_signals_see_joint_effort_and_frame_changes() -> None:
    t = 100
    names = ["q_torso", "dq_torso", "dq_lh_FFJ1", "eff_torso"]
    proprio = np.zeros((t, 4))
    proprio[20:25, 1] = 3.0  # body joint moves
    proprio[40:45, 2] = 3.0  # finger moves
    proprio[60:, 3] = 5.0  # effort steps up at 60
    rgb = np.zeros((t, 16, 16, 3), dtype=np.uint8)
    rgb[80:] = 200  # the image changes at 80
    sig = events.step_signals(proprio, names, rgb, 0.05)
    assert set(sig) == set(events.SIGNALS)  # no effector actuator: no effector_effort
    assert np.flatnonzero(sig["body_speed"]).tolist() == list(range(20, 25))
    assert np.flatnonzero(sig["effector_speed"]).tolist() == list(range(40, 45))
    assert np.flatnonzero(sig["effort_change"]).tolist() == [60]
    assert np.flatnonzero(sig["frame_change"]).tolist() == [80]
    noise = {k: v + np.random.default_rng(0).uniform(0, 0.01, t) for k, v in sig.items()}
    flags = events.event_flags(noise, events.fit_scales([noise]))
    assert set(np.flatnonzero(flags)) == {*range(20, 25), *range(40, 45), 60, 80}
    no_effort = events.step_signals(proprio[:, :3], names[:3], np.zeros(0), 0.05)
    assert set(no_effort) == {"body_speed", "effector_speed"}


def test_weights_keep_a_rest_floor_and_a_cap() -> None:
    span = 8
    busy = np.ones(100, dtype=bool)  # every start is an event start
    busy[:30] = False
    quiet = np.zeros(100, dtype=bool)
    raw = {"a": events.raw_weights(busy, span), "b": events.raw_weights(quiet, span)}
    assert raw["b"].tolist() == [1.0] * 93 and raw["a"].max() == events.CAP
    w = events.finalise(raw)
    total = np.concatenate(list(w.values()))
    rest = np.concatenate([raw["a"], raw["b"]]) == 1.0
    assert total[rest].sum() / total.sum() >= events.REST_SHARE - 1e-9
    assert total.max() <= events.CAP * total.mean() * (1 + 1e-6)
    # mostly events: rest is scaled up to the floor
    raw = {"a": events.raw_weights(np.ones(200, dtype=bool), span), "b": raw["b"][:10]}
    w = events.finalise(raw)
    share = w["b"].sum() / (w["a"].sum() + w["b"].sum())
    assert share == pytest.approx(events.REST_SHARE, abs=1e-6)
    # no event anywhere: uniform
    w = events.finalise({"b": events.raw_weights(quiet, span)})
    assert np.allclose(w["b"], 1.0)


def test_weighted_dataset_draws_only_weighted_starts(tmp_path: Path) -> None:
    paths = saved_privileged_episodes(tmp_path, count=2)
    space = ChannelSpace.from_headers(read_header(p) for p in paths)
    senses = check_senses(("proprio_body",))
    n_starts = 120 - LOADER.steps * LOADER.substeps + 1  # dt 0.05: one env step per observation
    weights = {p.name: np.zeros(n_starts) for p in paths}
    weights[paths[1].name][50] = 1.0  # one start of one episode
    weighted = SequenceDataset(paths, space, senses, LOADER, seed=0, start_weights=weights)
    rng = np.random.default_rng(0)
    first = weighted.sample(rng)["proprio_body"]
    for _ in range(5):
        assert np.array_equal(weighted.sample(rng)["proprio_body"], first)
    with pytest.raises(ValueError, match="start weights"):
        SequenceDataset(
            paths, space, senses, LOADER, start_weights={p.name: np.ones(3) for p in paths}
        )


def test_event_index_is_the_same_from_episodes_and_from_the_frame_cache(tmp_path: Path) -> None:
    paths = saved_privileged_episodes(tmp_path, count=3)
    build_cache(paths, tmp_path / "cache", None)
    a = events.build_event_index(paths, 16)
    b = events.build_event_index(paths, 16, tmp_path / "cache")
    assert a["weights"] == b["weights"] and a["scales"] == b["scales"]
    out = tmp_path / "events.json"
    out.write_text(json.dumps(a))
    loaded = events.load_start_weights(out)
    assert (
        set(loaded) == {p.name for p in paths} and len(loaded[paths[0].name]) == 120 - 16 + 1
    )  # span 16 = 4 x 4


def test_groups_keep_their_uniform_share_of_draws() -> None:
    span = 8
    busy = events.raw_weights(np.ones(100, dtype=bool), span)
    quiet = events.raw_weights(np.zeros(100, dtype=bool), span)
    mixed = events.raw_weights(np.arange(100) % 20 < 5, span)
    w = events.finalise_by_group(
        {"p1": busy, "p2": quiet, "h1": mixed, "h2": quiet},
        {"p1": "pincer", "p2": "pincer", "h1": "hand", "h2": "hand"},
    )
    pincer = w["p1"].sum() + w["p2"].sum()
    hand = w["h1"].sum() + w["h2"].sum()
    assert pincer == pytest.approx(hand)  # equal start counts: equal draws
    assert w["h1"].mean() > w["h2"].mean()  # inside a group, events are drawn more
    total = np.concatenate(list(w.values()))
    assert total.max() <= events.CAP * total.mean() * (1 + 1e-6)
