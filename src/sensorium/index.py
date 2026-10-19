"""The episode index: one row per generated episode, stored as parquet.

Usage:
    sensorium-index data/sample/index.parquet data/sample/manifest.json [more manifests]

``build_index`` joins every ``manifest.json`` entry with the source registry
(``configs/sources.toml``) to add the effector, source, family, task, kind of
experience and tags. The pool builder reads only this index, never the manifests.

Manifest fields relied on (written by ``sensorium.generate``): ``file``
(relative to the manifest's directory), ``body``, ``policy``, ``seed``,
``steps``, ``content_hash``. ``task`` and ``instance`` are optional; when absent the
body's default task from the registry and the instance ``"default"`` are used.
"""

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sensorium.registry import DEFAULT_REGISTRY_PATH, Registry, load_registry

log = logging.getLogger("sensorium.index")

DEFAULT_INSTANCE = "default"  # for manifests that predate the ``instance`` field

SCHEMA = pa.schema(
    [
        ("path", pa.string()),
        ("body", pa.string()),
        ("effector", pa.string()),
        ("source", pa.string()),
        ("family", pa.string()),
        ("task", pa.string()),
        ("instance", pa.string()),
        ("policy", pa.string()),
        ("kind", pa.string()),
        ("seed", pa.int64()),
        ("steps", pa.int64()),
        ("content_hash", pa.string()),
        ("tags", pa.list_(pa.string())),
    ]
)


@dataclass(frozen=True)
class IndexRecord:
    """One episode in the index (one parquet row).

    Attributes:
        path: Absolute path of the episode ``.npz``.
        body, effector, source, family, task: From the registry (``source`` is its name).
        instance: The specific content of the task (song id, mocap clip, demo id, ...);
            ``"default"`` when the manifest has no ``instance`` field.
        policy: The recorded policy name; ``kind`` is its kind of experience.
        seed: The episode's seed.
        steps: Number of steps ``T``.
        content_hash: Hash of the episode's arrays (from the manifest).
        tags: The source's tags.
    """

    path: str
    body: str
    effector: str
    source: str
    family: str
    task: str
    instance: str
    policy: str
    kind: str
    seed: int
    steps: int
    content_hash: str
    tags: tuple[str, ...]


def entries_to_records(
    manifest_path: Path, entries: Sequence[dict[str, object]], registry: Registry
) -> list[IndexRecord]:
    """Turn the ``episodes`` list of one manifest into index records.

    Raises:
        ValueError: on an entry with a missing field, an unregistered ``(body, task)``
            or an unknown policy.
        FileNotFoundError: if an episode file listed in the manifest does not exist.
    """
    records = []
    for entry in entries:
        missing = {"file", "body", "policy", "seed", "steps", "content_hash"} - set(entry)
        if missing:
            raise ValueError(f"{manifest_path}: entry {entry} is missing {sorted(missing)}")
        body = str(entry["body"])
        task = entry.get("task")
        effector = entry.get("effector")
        source = registry.resolve(
            body, None if task is None else str(task), None if effector is None else str(effector)
        )
        path = (manifest_path.parent / str(entry["file"])).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{manifest_path}: episode file {path} does not exist")
        policy = str(entry["policy"])
        records.append(
            IndexRecord(
                path=str(path),
                body=body,
                effector=source.effector,
                source=source.name,
                family=source.family,
                task=source.task,
                instance=str(entry.get("instance", DEFAULT_INSTANCE)),
                policy=policy,
                kind=registry.kind_of(policy),
                seed=int(str(entry["seed"])),
                steps=int(str(entry["steps"])),
                content_hash=str(entry["content_hash"]),
                tags=source.tags,
            )
        )
    return records


def build_index(manifests: Sequence[Path], registry: Registry) -> list[IndexRecord]:
    """Read every manifest and return the records, sorted by path.

    Raises:
        ValueError: if an entry is invalid (see ``entries_to_records``) or the same
            episode file appears twice.
        FileNotFoundError: if a manifest or one of its episode files does not exist.
    """
    records: list[IndexRecord] = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        records += entries_to_records(manifest_path, manifest["episodes"], registry)
    records.sort(key=lambda r: r.path)
    paths = [r.path for r in records]
    duplicates = sorted({p for p in paths if paths.count(p) > 1})
    if duplicates:
        raise ValueError(f"episodes listed more than once: {duplicates}")
    return records


def write_index(records: Sequence[IndexRecord], path: Path) -> None:
    """Write the records to a parquet file (parent directories are created)."""
    columns = {f.name: [getattr(r, f.name) for r in records] for f in fields(IndexRecord)}
    columns["tags"] = [list(r.tags) for r in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns, schema=SCHEMA), path)


def read_index(path: Path) -> list[IndexRecord]:
    """Read a parquet index written by ``write_index`` into plain records.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if a column is missing.
    """
    table = pq.read_table(path)
    missing = {f.name for f in fields(IndexRecord)} - set(table.column_names)
    if missing:
        raise ValueError(f"{path}: index is missing columns {sorted(missing)}")
    return [IndexRecord(**{**row, "tags": tuple(row["tags"])}) for row in table.to_pylist()]


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point: ``sensorium-index <out.parquet> <manifest.json>...``."""
    parser = argparse.ArgumentParser(description="Build the episode index from manifests.")
    parser.add_argument("out", type=Path, help="parquet file to write")
    parser.add_argument("manifests", type=Path, nargs="+", help="manifest.json files")
    parser.add_argument("--sources", type=Path, default=DEFAULT_REGISTRY_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    records = build_index(args.manifests, load_registry(args.sources))
    write_index(records, args.out)
    log.info(
        "wrote %d episodes (%d steps) to %s", len(records), sum(r.steps for r in records), args.out
    )


if __name__ == "__main__":
    main()
