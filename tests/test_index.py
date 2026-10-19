"""Episode index: manifests -> parquet -> plain records."""

import json
from pathlib import Path

import pytest
from world import World, make_world

from sensorium.index import build_index, read_index, write_index
from sensorium.registry import load_registry


@pytest.fixture
def world(tmp_path: Path) -> World:
    return make_world(tmp_path, episodes_per_policy=3)


def test_round_trip(world: World, tmp_path: Path) -> None:
    registry = load_registry(world.registry)
    records = build_index([world.manifest], registry)
    assert len(records) == 9 * 3 * 3
    write_index(records, tmp_path / "out" / "index.parquet")
    assert read_index(tmp_path / "out" / "index.parquet") == records

    by_hash = {r.content_hash: r for r in records}
    record = by_hash["alien/a1-explore-2"]
    assert (record.body, record.effector, record.source) == ("a1", "alien", "alien/a1")
    assert (record.family, record.task, record.instance) == ("fam1", "t", "inst2")
    assert (record.policy, record.kind, record.seed, record.steps) == ("explore", "babbling", 2, 12)
    assert record.tags == ("vision_dominated",)
    assert Path(record.path).is_absolute() and Path(record.path).is_file()


def test_unknown_body_or_task_is_a_clear_error(world: World) -> None:
    registry = load_registry(world.registry)
    manifest = json.loads(world.manifest.read_text())
    manifest["episodes"][0]["body"] = "humanoid"
    world.manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r"body='humanoid'"):
        build_index([world.manifest], registry)


def test_unknown_policy_and_missing_file_and_duplicates(world: World) -> None:
    registry = load_registry(world.registry)
    good = json.loads(world.manifest.read_text())

    bad = json.loads(world.manifest.read_text())
    bad["episodes"][0]["policy"] = "dance"
    world.manifest.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="unknown policy"):
        build_index([world.manifest], registry)

    bad = json.loads(world.manifest.read_text())
    bad["episodes"][0]["policy"] = "expert"
    bad["episodes"][0]["file"] = "nope.npz"
    world.manifest.write_text(json.dumps(bad))
    with pytest.raises(FileNotFoundError):
        build_index([world.manifest], registry)

    world.manifest.write_text(json.dumps(good))
    with pytest.raises(ValueError, match="more than once"):
        build_index([world.manifest, world.manifest], registry)


def test_instance_defaults_when_manifest_has_no_field(tmp_path: Path) -> None:
    world = make_world(tmp_path, episodes_per_policy=2, with_instance=False)
    records = build_index([world.manifest], load_registry(world.registry))
    assert {r.instance for r in records} == {"default"}
    write_index(records, tmp_path / "index.parquet")
    assert {r.instance for r in read_index(tmp_path / "index.parquet")} == {"default"}
