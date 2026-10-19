"""Pool builder: nesting, ratios, tolerance, held-out exclusion, senses and determinism."""

import dataclasses
from pathlib import Path

import pytest
from world import World, make_world

from sensorium.episode import Episode, model_inputs
from sensorium.index import IndexRecord, build_index
from sensorium.pool import (
    Pool,
    PoolError,
    PoolSpec,
    Splits,
    build_pool,
    load_pool_spec,
    load_splits,
    read_pool,
    validate_pool,
    write_pool,
)
from sensorium.registry import load_registry

ALL_SENSES = ["rgb", "audio_logmel", "proprio", "touch", "action"]
THIRDS = {"expert": 1, "noisy": 1, "babbling": 1}


@pytest.fixture(scope="module")
def records_and_splits(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[list[IndexRecord], Splits]:
    world: World = make_world(tmp_path_factory.mktemp("world"))
    registry = load_registry(world.registry)
    return build_index([world.manifest], registry), load_splits(world.splits, registry)


def spec(**overrides: object) -> PoolSpec:
    raw: dict[str, object] = {
        "name": "t",
        "total_steps": 1200,
        "senses": ALL_SENSES,
        "kinds": THIRDS,
        "effector_ratio": {"alien": 1},
        "n_sources": "all",
        "seed": 0,
    }
    return PoolSpec.from_dict(raw | overrides)


def test_sources_are_nested(records_and_splits: tuple[list[IndexRecord], Splits]) -> None:
    records, splits = records_and_splits
    chosen = {
        n: build_pool(records, spec(n_sources=n), splits).chosen_sources for n in (2, 4, "all")
    }
    assert len(chosen[2]) == 2 and len(chosen[4]) == 4 and len(chosen["all"]) == 6
    assert set(chosen[2]) < set(chosen[4]) < set(chosen["all"])
    assert chosen[4][:2] == chosen[2]  # ranking order is the same prefix


def test_pool_only_uses_chosen_sources(
    records_and_splits: tuple[list[IndexRecord], Splits],
) -> None:
    records, splits = records_and_splits
    pool = build_pool(records, spec(n_sources=2), splits)
    assert {e.source for e in pool.episodes} == set(pool.chosen_sources)


def test_effector_and_kind_ratios_and_total(
    records_and_splits: tuple[list[IndexRecord], Splits],
) -> None:
    records, splits = records_and_splits
    pool = build_pool(
        records,
        spec(
            total_steps=2400,
            effector_ratio={"hand": 1, "alien": 3},
            kinds={"expert": 2, "babbling": 1},
        ),
        splits,
    )
    assert set(pool.strata) == {"hand/expert", "hand/babbling", "alien/expert", "alien/babbling"}
    assert abs(pool.total_steps - 2400) <= 0.02 * 2400
    for label, stratum in pool.strata.items():
        assert abs(stratum.actual_steps - stratum.target_steps) <= 0.02 * stratum.target_steps, (
            label
        )
    targets = {k: round(v.target_steps) for k, v in pool.strata.items()}
    assert targets == {
        "hand/expert": 400,
        "hand/babbling": 200,
        "alien/expert": 1200,
        "alien/babbling": 600,
    }
    hand = sum(e.steps for e in pool.episodes if e.effector == "hand")
    assert hand / pool.total_steps == pytest.approx(0.25, abs=0.02)
    kinds = {e.kind for e in pool.episodes}
    assert kinds == {"expert", "babbling"}


def test_alien_excluded_unless_ratio_is_positive(
    records_and_splits: tuple[list[IndexRecord], Splits],
) -> None:
    records, splits = records_and_splits
    pool = build_pool(records, spec(effector_ratio={"hand": 1, "alien": 0}), splits)
    assert {e.effector for e in pool.episodes} == {"hand"}
    pool = build_pool(records, spec(effector_ratio={"hand": 1}), splits)
    assert {e.effector for e in pool.episodes} == {"hand"}


def test_heldout_source_never_in_pool(records_and_splits: tuple[list[IndexRecord], Splits]) -> None:
    records, splits = records_and_splits
    assert any(r.source == "hand/h2" for r in records)  # it is in the index ...
    pool = build_pool(records, spec(effector_ratio={"hand": 1}, n_sources="all"), splits)
    assert "hand/h2" not in {e.source for e in pool.episodes}  # ... but never in a pool
    assert set(pool.chosen_sources) == {"hand/h0", "hand/h1"}

    leaked = next(r for r in records if r.source == "hand/h2")
    tampered = dataclasses.replace(pool, episodes=(*pool.episodes, leaked))
    with pytest.raises(PoolError, match="held-out"):
        validate_pool(tampered, splits)


def test_heldout_instances_never_in_pool(
    records_and_splits: tuple[list[IndexRecord], Splits],
) -> None:
    records, splits = records_and_splits
    assert any(r.task == "play0" and r.instance == "inst0" for r in records)
    pool = build_pool(records, spec(effector_ratio={"hand": 1}), splits)
    play0 = {e.instance for e in pool.episodes if e.task == "play0"}
    assert play0 and play0 <= {"inst2", "inst3"}  # the task stays, its held-out instances go
    assert {e.instance for e in pool.episodes if e.task == "play1"} == {
        f"inst{i}" for i in range(4)
    }
    assert pool.excluded_instances == {"play0": 2}

    leaked = next(r for r in records if r.task == "play0" and r.instance == "inst1")
    tampered = dataclasses.replace(pool, episodes=(*pool.episodes, leaked))
    with pytest.raises(PoolError, match="held-out instances"):
        validate_pool(tampered, splits)


def test_split_cannot_be_heldout(records_and_splits: tuple[list[IndexRecord], Splits]) -> None:
    with pytest.raises(PoolError, match="held-out"):
        spec(split="heldout")


def test_unfillable_stratum_fails_loudly(
    records_and_splits: tuple[list[IndexRecord], Splits],
) -> None:
    records, splits = records_and_splits
    with pytest.raises(PoolError, match=r"stratum alien/expert"):
        build_pool(records, spec(total_steps=10_000_000), splits)
    with pytest.raises(PoolError, match="n_sources=7"):
        build_pool(records, spec(n_sources=7), splits)
    no_babbling = [r for r in records if r.kind != "babbling"]
    with pytest.raises(PoolError, match="babbling"):
        build_pool(no_babbling, spec(), splits)


def test_deterministic(records_and_splits: tuple[list[IndexRecord], Splits]) -> None:
    records, splits = records_and_splits
    first = build_pool(records, spec(seed=3, n_sources=3), splits)
    second = build_pool(list(reversed(records)), spec(seed=3, n_sources=3), splits)
    assert first.content_hash == second.content_hash
    assert first.paths == second.paths
    other = build_pool(records, spec(seed=4, n_sources=3), splits)
    assert other.content_hash != first.content_hash


def test_senses_change_the_hash_and_are_normalised(
    records_and_splits: tuple[list[IndexRecord], Splits],
) -> None:
    records, splits = records_and_splits
    vision = build_pool(records, spec(senses=["rgb"]), splits)
    assert vision.senses == ("rgb", "action")
    assert vision.content_hash != build_pool(records, spec(), splits).content_hash
    with pytest.raises(PoolError, match="privileged"):
        spec(senses=["rgb", "privileged"])


def test_pool_senses_apply_at_load_time(
    records_and_splits: tuple[list[IndexRecord], Splits],
) -> None:
    records, splits = records_and_splits
    pool = build_pool(records, spec(senses=["rgb", "touch"]), splits)
    episode = Episode.load(pool.paths[0])
    assert set(model_inputs(episode, senses=pool.senses)) == {"rgb", "touch", "action"}


def test_pool_json_round_trip(
    records_and_splits: tuple[list[IndexRecord], Splits], tmp_path: Path
) -> None:
    records, splits = records_and_splits
    pool = build_pool(records, spec(effector_ratio={"hand": 1, "alien": 1}, n_sources=5), splits)
    write_pool(pool, tmp_path / "pool.json")
    loaded: Pool = read_pool(tmp_path / "pool.json")
    assert loaded.spec == pool.spec
    assert loaded.chosen_sources == pool.chosen_sources
    assert loaded.strata == pool.strata
    assert loaded.paths == pool.paths
    assert loaded.content_hash == pool.content_hash
    assert loaded.excluded_instances == pool.excluded_instances
    assert loaded.episodes[0].instance == pool.episodes[0].instance
    assert loaded.split_sha256 == splits.sha256


def test_spec_toml(tmp_path: Path) -> None:
    path = tmp_path / "spec.toml"
    path.write_text(
        'name = "x"\ntotal_steps = 100\nsenses = ["rgb"]\nseed = 1\nn_sources = 8\n'
        "[kinds]\nbabbling = 1\n[effector_ratio]\nhand = 1\n"
    )
    loaded = load_pool_spec(path)
    assert (loaded.n_sources, loaded.kinds, loaded.senses) == (
        8,
        {"babbling": 1.0},
        ("rgb", "action"),
    )
    path.write_text(path.read_text().replace("n_sources = 8", "n_sources = 0"))
    with pytest.raises(PoolError, match="n_sources"):
        load_pool_spec(path)


def test_splits_validation(tmp_path: Path) -> None:
    world = make_world(tmp_path, episodes_per_policy=1)
    registry = load_registry(world.registry)
    text = world.splits.read_text()
    cases = {
        "unknown source": text.replace('"hand/h1"', '"hand/zzz"'),
        "both": text.replace(
            "[heldout.tasks]\n",
            '[heldout.tasks]\n"hand/h0" = { reason = "x", tag = "contact_rich" }\n',
        ),
        "not assigned": text.replace('"hand/h0", "hand/h1"]', '"hand/h0"]'),
        "needs": text.replace('tag = "contact_rich"', 'tag = "other"'),
    }
    for message, broken in cases.items():
        world.splits.write_text(broken)
        with pytest.raises(PoolError, match=message):
            load_splits(world.splits, registry)


def test_instance_split_validation(tmp_path: Path) -> None:
    world = make_world(tmp_path, episodes_per_policy=1)
    registry = load_registry(world.registry)
    text = world.splits.read_text()
    cases = {
        "not a registered task": text.replace("heldout.instances.play0", "heldout.instances.nope"),
        "itself held out": text.replace("heldout.instances.play0", "heldout.instances.play2"),
        "no ids": text.replace('ids = ["inst0", "inst1"]', "ids = []"),
        "needs a reason": text.replace('reason = "test held-out instances"', 'reason = ""'),
    }
    for message, broken in cases.items():
        world.splits.write_text(broken)
        with pytest.raises(PoolError, match=message):
            load_splits(world.splits, registry)
