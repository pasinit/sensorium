"""The pool builder: equal-volume pretraining pools with controlled ingredients.

Usage:
    sensorium-pool data/sample/index.parquet configs/pools/example.toml data/sample/pool.json

A **pool** is an ordered list of whole episodes drawn from the episode index
(``sensorium.index``) such that

* it only uses sources the split file allows (``configs/splits/v3.toml``);
* its total number of steps is ``total_steps`` within ``tolerance`` (default 2%);
* every ``(effector, kind of experience)`` stratum holds its share of the steps
  (``effector_ratio`` x ``kinds``), each within ``tolerance`` of its target;
* it uses the first ``n_sources`` sources of a seeded ranking, so smaller source
  subsets are nested in larger ones (n=8 within n=16 within all);
* it carries the ``senses`` subset that a loader applies with
  ``sensorium.episode.model_inputs(episode, senses=pool.senses)``.

Everything is deterministic: the same index, spec and split file give the same
pool and the same pool hash. ``docs/pools.md`` has the full TOML spec.
"""

import argparse
import hashlib
import json
import logging
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from sensorium.episode import MODEL_INPUT_KEYS
from sensorium.index import IndexRecord, read_index
from sensorium.registry import (
    CHARACTER_TAGS,
    DEFAULT_REGISTRY_PATH,
    EFFECTORS,
    KINDS,
    Registry,
    load_registry,
)

log = logging.getLogger("sensorium.pool")

DEFAULT_SPLITS_PATH = Path("configs/splits/v3.toml")
HELDOUT = "heldout"


class PoolError(ValueError):
    """A pool or split cannot be built or violates a contract."""


# --------------------------------------------------------------------------- splits


@dataclass(frozen=True)
class Splits:
    """The frozen split file.

    Attributes:
        version: Free label of the split file, e.g. ``"v0"``.
        sections: Section name (e.g. ``"pretrain"``) -> source names a pool may draw from.
        heldout: Names of held-out sources (whole tasks) that no pool may contain.
        notes: Held-out source -> ``{"reason": ..., "tag": "contact_rich" | "vision_dominated"}``.
        sha256: Hash of the split file's bytes, recorded in every pool.
        heldout_instances: Task -> instance ids held out within that task (the task's other
            instances stay in the pretraining pool).
        instance_notes: Task -> ``{"reason": ..., "tag": ...}`` for its held-out instances.
    """

    version: str
    sections: dict[str, tuple[str, ...]]
    heldout: tuple[str, ...]
    notes: dict[str, dict[str, str]]
    sha256: str
    heldout_instances: dict[str, tuple[str, ...]] = field(default_factory=dict)
    instance_notes: dict[str, dict[str, str]] = field(default_factory=dict)

    def is_heldout_instance(self, task: str, instance: str) -> bool:
        """Whether ``instance`` of ``task`` is held out."""
        return instance in self.heldout_instances.get(task, ())


def load_splits(path: Path, registry: Registry) -> Splits:
    """Load and validate a split file against the registry.

    The file has a ``[pretrain]`` table with ``sources`` and a ``[heldout]`` table with

    * ``[heldout.tasks]``: held-out sources by name, each ``{reason, tag}``;
    * ``[heldout.instances.<task>]``: ``ids`` (instances of that task held out),
      ``reason`` and ``tag``.

    Raises:
        PoolError: if a source is unknown, appears in two sections, is missing from
            all sections, a note lacks a reason or valid tag, an instance list names an
            unregistered task, or it names a task that is itself held out.
    """
    raw_bytes = path.read_bytes()
    raw = tomllib.loads(raw_bytes.decode())
    heldout_table = raw.get(HELDOUT, {})
    notes = _notes(heldout_table.get("tasks", {}))
    instances = heldout_table.get("instances", {})
    sections = {
        name: tuple(table["sources"])
        for name, table in raw.items()
        if isinstance(table, dict) and name != HELDOUT and "sources" in table
    }
    splits = Splits(
        version=str(raw.get("version", "")),
        sections=sections,
        heldout=tuple(notes),
        notes=notes,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        heldout_instances={
            str(t): tuple(str(i) for i in v.get("ids", [])) for t, v in instances.items()
        },
        instance_notes=_notes(instances),
    )
    _validate_splits(splits, registry, path)
    return splits


def _notes(table: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, str]]:
    """Stringify a table of ``{reason, tag, ...}`` entries (``ids`` lists are dropped)."""
    return {
        str(k): {str(a): str(b) for a, b in v.items() if not isinstance(b, list)}
        for k, v in table.items()
    }


def _validate_no_leak(splits: Splits, registry: Registry, path: Path) -> None:
    """Raise ``PoolError`` if a held-out source's program or prop is used by a pretraining source.

    Held-out *tasks* must be unseen, so neither the scripted program that demonstrates one nor
    a prop configuration made for it may appear in any source that is not held out.
    """
    held = [registry.by_name(n) for n in splits.heldout]
    programs = {p: s.name for s in held for p in s.programs}
    props = {p: s.name for s in held for p in s.props}
    for section, names in splits.sections.items():
        for name in names:
            source = registry.by_name(name)
            for program in source.programs:
                if program in programs:
                    raise PoolError(
                        f"{path}: [{section}] source {name!r} runs program {program!r}, which is "
                        f"held out with {programs[program]!r}"
                    )
            for prop in source.props:
                if prop in props:
                    raise PoolError(
                        f"{path}: [{section}] source {name!r} uses prop {prop!r}, which is "
                        f"held out with {props[prop]!r}"
                    )


def _validate_splits(splits: Splits, registry: Registry, path: Path) -> None:
    """Raise ``PoolError`` unless the split is consistent with the registry (``load_splits``)."""
    known = {s.name for s in registry.sources}
    seen: dict[str, str] = {}
    for section, names in {**splits.sections, HELDOUT: splits.heldout}.items():
        for name in names:
            if name not in known:
                raise PoolError(f"{path}: [{section}] names unknown source {name!r}")
            if name in seen:
                raise PoolError(
                    f"{path}: source {name!r} is in both [{seen[name]}] and [{section}]"
                )
            seen[name] = section
    unassigned = sorted(known - set(seen))
    if unassigned:
        raise PoolError(f"{path}: sources not assigned to any section: {unassigned}")
    _validate_notes(splits.notes, "heldout.tasks", path)
    _validate_notes(splits.instance_notes, "heldout.instances", path)
    _validate_no_leak(splits, registry, path)
    tasks = {s.task for s in registry.sources}
    heldout_tasks = {registry.by_name(n).task for n in splits.heldout}
    for task, ids in splits.heldout_instances.items():
        if task not in tasks:
            raise PoolError(
                f"{path}: [heldout.instances.{task}] is not a registered task {sorted(tasks)}"
            )
        if task in heldout_tasks:
            raise PoolError(
                f"{path}: instances listed under task {task!r}, which is itself held out"
            )
        if not ids:
            raise PoolError(f"{path}: [heldout.instances.{task}] has no ids")


def _validate_notes(notes: Mapping[str, Mapping[str, str]], table: str, path: Path) -> None:
    """Raise ``PoolError`` unless every entry has a reason and a tag in ``CHARACTER_TAGS``."""
    for name, note in notes.items():
        if not note.get("reason") or note.get("tag") not in CHARACTER_TAGS:
            raise PoolError(
                f"{path}: [{table}] {name!r} needs a reason and a tag in {CHARACTER_TAGS}"
            )


# ---------------------------------------------------------------------------- spec


@dataclass(frozen=True)
class PoolSpec:
    """What to build. ``kinds`` and ``effector_ratio`` are relative weights (normalised).

    Attributes:
        name: Label of the pool.
        total_steps: Target number of steps.
        senses: Subset of ``MODEL_INPUT_KEYS`` kept at load time; ``action`` is always included.
        kinds: Kind of experience -> weight. Kinds with weight 0 are omitted.
        effector_ratio: Effector -> weight. Effectors with weight 0 (or absent) are excluded.
        n_sources: Number of sources (first n of the seeded ranking) or ``"all"``.
        seed: Seed of the source ranking and the episode order.
        split: Section of the split file the sources come from (e.g. ``"pretrain"``).
        tolerance: Allowed relative deviation of the total and of every stratum.
    """

    name: str
    total_steps: int
    senses: tuple[str, ...]
    kinds: dict[str, float]
    effector_ratio: dict[str, float]
    n_sources: int | str
    seed: int
    split: str = "pretrain"
    tolerance: float = 0.02

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "PoolSpec":
        """Build and validate a spec from a parsed TOML/JSON table.

        Raises:
            PoolError: on any invalid field (the message names it).
        """
        try:
            requested = {str(s) for s in raw["senses"]}  # type: ignore[attr-defined]
            spec = cls(
                name=str(raw.get("name", "pool")),
                total_steps=int(raw["total_steps"]),  # type: ignore[call-overload]
                senses=tuple(k for k in MODEL_INPUT_KEYS if k in requested or k == "action"),
                kinds={str(k): float(v) for k, v in raw["kinds"].items()},  # type: ignore[attr-defined]
                effector_ratio={str(k): float(v) for k, v in raw["effector_ratio"].items()},  # type: ignore[attr-defined]
                n_sources=raw["n_sources"],  # type: ignore[arg-type]
                seed=int(raw["seed"]),  # type: ignore[call-overload]
                split=str(raw.get("split", "pretrain")),
                tolerance=float(raw.get("tolerance", 0.02)),  # type: ignore[arg-type]
            )
        except KeyError as error:
            raise PoolError(f"pool spec: missing field {error}") from None
        unknown = requested - set(MODEL_INPUT_KEYS)
        if unknown:
            raise PoolError(f"senses {sorted(unknown)} not in {MODEL_INPUT_KEYS}")
        spec.validate()
        return spec

    def validate(self) -> None:
        """Raise ``PoolError`` unless the fields are usable."""
        if "action" not in self.senses or set(self.senses) - set(MODEL_INPUT_KEYS):
            raise PoolError(f"senses must be a subset of {MODEL_INPUT_KEYS} including action")
        if self.total_steps <= 0:
            raise PoolError(f"total_steps must be positive, got {self.total_steps}")
        for label, weights, allowed in (
            ("kinds", self.kinds, KINDS),
            ("effector_ratio", self.effector_ratio, EFFECTORS),
        ):
            if set(weights) - set(allowed):
                raise PoolError(f"{label}: keys must be in {allowed}, got {sorted(weights)}")
            if any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
                raise PoolError(f"{label}: weights must be >= 0 with a positive sum, got {weights}")
        valid_n = self.n_sources == "all" or (
            isinstance(self.n_sources, int) and self.n_sources >= 1
        )
        if not valid_n:
            raise PoolError(f'n_sources must be a positive int or "all", got {self.n_sources!r}')
        if not 0 < self.tolerance < 0.5:
            raise PoolError(f"tolerance must be in (0, 0.5), got {self.tolerance}")
        if self.split == HELDOUT:
            raise PoolError("a pool cannot be drawn from the held-out split")


def load_pool_spec(path: Path) -> PoolSpec:
    """Read a ``PoolSpec`` from a TOML file (see ``docs/pools.md``).

    Raises:
        PoolError: on an invalid spec.
    """
    return PoolSpec.from_dict(tomllib.loads(path.read_text()))


# ---------------------------------------------------------------------------- pool


@dataclass(frozen=True)
class Stratum:
    """One ``(effector, kind)`` cell of a pool.

    Attributes:
        target_steps: Steps the ratios ask for.
        actual_steps: Steps of the episodes chosen.
        n_episodes: Number of episodes chosen.
    """

    target_steps: float
    actual_steps: int
    n_episodes: int


@dataclass(frozen=True)
class Pool:
    """A built pool. ``episodes`` is the ordered list of whole episodes.

    ``content_hash`` covers the ordered episode content hashes and the senses, so it
    does not depend on where the data lives. ``excluded_instances`` is, per task, the
    number of held-out instances that were present in the index and left out.
    """

    spec: PoolSpec
    split_version: str
    split_sha256: str
    chosen_sources: tuple[str, ...]
    strata: dict[str, Stratum]
    episodes: tuple[IndexRecord, ...]
    content_hash: str = field(default="")
    excluded_instances: dict[str, int] = field(default_factory=dict)

    @property
    def paths(self) -> list[Path]:
        """Episode files in pool order."""
        return [Path(e.path) for e in self.episodes]

    @property
    def senses(self) -> tuple[str, ...]:
        """Senses to apply at load time (``model_inputs(episode, senses=pool.senses)``)."""
        return self.spec.senses

    @property
    def total_steps(self) -> int:
        """Summed steps of all episodes."""
        return sum(e.steps for e in self.episodes)


def source_rank(source: str, seed: int) -> str:
    """Sort key of a source in the seeded ranking: ``sha256(f"{seed}/{source}")``."""
    return hashlib.sha256(f"{seed}/{source}".encode()).hexdigest()


def choose_sources(sources: Sequence[str], n_sources: int | str, seed: int) -> tuple[str, ...]:
    """Return the first ``n_sources`` of ``sources`` ranked by ``source_rank`` (nested in n).

    Raises:
        PoolError: if fewer than ``n_sources`` sources are available.
    """
    ranked = tuple(sorted(set(sources), key=lambda s: source_rank(s, seed)))
    if n_sources == "all":
        return ranked
    assert isinstance(n_sources, int)
    if n_sources > len(ranked):
        raise PoolError(f"n_sources={n_sources} but only {len(ranked)} sources available: {ranked}")
    return ranked[:n_sources]


def normalise(weights: Mapping[str, float], order: Sequence[str]) -> dict[str, float]:
    """Positive weights as fractions summing to 1, in the canonical ``order``."""
    total = sum(weights.values())
    return {k: weights[k] / total for k in order if weights.get(k, 0.0) > 0}


def interleave(
    candidates: Sequence[IndexRecord], ranked_sources: Sequence[str], key: str, seed: int
) -> list[IndexRecord]:
    """Deterministic episode order for a stratum: round-robin over sources, shuffled within.

    Spreads a stratum across all its sources instead of exhausting one first.
    """
    per_source: list[list[IndexRecord]] = []
    for source in ranked_sources:
        own = [c for c in candidates if c.source == source]
        own.sort(
            key=lambda c: (
                hashlib.sha256(f"{seed}/{key}/{c.content_hash}".encode()).hexdigest(),
                c.path,
            )
        )
        if own:
            per_source.append(own)
    ordered: list[IndexRecord] = []
    for rank in range(max((len(p) for p in per_source), default=0)):
        ordered += [p[rank] for p in per_source if rank < len(p)]
    return ordered


def fill_stratum(
    ordered: Sequence[IndexRecord], target: float, tolerance: float
) -> list[IndexRecord]:
    """First-fit whole episodes, in order, until within ``tolerance * target`` of ``target``.

    An episode is taken when it does not overshoot ``target`` by more than the
    tolerance; larger ones are skipped. May return fewer steps than asked if the
    candidates run out; the caller checks.
    """
    slack = tolerance * target
    chosen: list[IndexRecord] = []
    total = 0
    for episode in ordered:
        if target - total <= slack:
            break
        if total + episode.steps <= target + slack:
            chosen.append(episode)
            total += episode.steps
    return chosen


def build_pool(records: Sequence[IndexRecord], spec: PoolSpec, splits: Splits) -> Pool:
    """Build the pool described by ``spec`` from index ``records``.

    Raises:
        PoolError: if the split section is unknown, too few sources exist, or any
            stratum (or the total) cannot be filled within ``spec.tolerance``.
    """
    if spec.split not in splits.sections:
        raise PoolError(
            f"split {spec.split!r} not in split file (sections: {sorted(splits.sections)})"
        )
    effectors = normalise(spec.effector_ratio, EFFECTORS)
    kinds = normalise(spec.kinds, KINDS)
    allowed = set(splits.sections[spec.split])
    in_split = [r for r in records if r.source in allowed]
    excluded = {r for r in in_split if splits.is_heldout_instance(r.task, r.instance)}
    eligible = [r for r in in_split if r not in excluded and r.effector in effectors]
    ranked = choose_sources([r.source for r in eligible], "all", spec.seed)
    chosen = choose_sources([r.source for r in eligible], spec.n_sources, spec.seed)
    strata: dict[str, Stratum] = {}
    episodes: list[IndexRecord] = []
    for effector, kind in ((e, k) for e in effectors for k in kinds):
        label = f"{effector}/{kind}"
        target = spec.total_steps * effectors[effector] * kinds[kind]
        candidates = [
            r for r in eligible if r.source in chosen and r.effector == effector and r.kind == kind
        ]
        picked = fill_stratum(
            interleave(candidates, [s for s in ranked if s in chosen], label, spec.seed),
            target,
            spec.tolerance,
        )
        strata[label] = Stratum(target, sum(e.steps for e in picked), len(picked))
        episodes += picked
        _check_stratum(label, strata[label], candidates, spec.tolerance)
    pool = Pool(
        spec,
        splits.version,
        splits.sha256,
        chosen,
        strata,
        tuple(episodes),
        excluded_instances=count_excluded_instances(excluded),
    )
    _check_total(pool)
    pool = replace(pool, content_hash=pool_hash(pool))
    validate_pool(pool, splits)
    log.info(
        "pool %s: %d episodes, %d steps, %d sources, hash %s",
        spec.name,
        len(episodes),
        pool.total_steps,
        len(chosen),
        pool.content_hash,
    )
    return pool


def count_excluded_instances(excluded: Iterable[IndexRecord]) -> dict[str, int]:
    """Per task, the number of distinct instances among the excluded records."""
    instances: dict[str, set[str]] = {}
    for record in excluded:
        instances.setdefault(record.task, set()).add(record.instance)
    return {task: len(ids) for task, ids in sorted(instances.items())}


def _check_stratum(
    label: str, stratum: Stratum, candidates: Sequence[IndexRecord], tolerance: float
) -> None:
    """Raise ``PoolError`` if a stratum misses its target by more than the tolerance."""
    if abs(stratum.actual_steps - stratum.target_steps) > tolerance * stratum.target_steps:
        available = sum(c.steps for c in candidates)
        raise PoolError(
            f"stratum {label}: target {stratum.target_steps:.0f} steps (+-{tolerance:.0%}), got "
            f"{stratum.actual_steps}; {len(candidates)} candidate episodes with {available} steps"
        )


def _check_total(pool: Pool) -> None:
    """Raise ``PoolError`` if the pool's total is not within tolerance of ``total_steps``."""
    spec = pool.spec
    if abs(pool.total_steps - spec.total_steps) > spec.tolerance * spec.total_steps:
        raise PoolError(
            f"pool total {pool.total_steps} steps misses {spec.total_steps} "
            f"by more than {spec.tolerance:.0%}"
        )


def pool_hash(pool: Pool) -> str:
    """SHA-256 (16 hex chars) over the senses and the ordered episode content hashes."""
    payload = json.dumps(
        {"senses": pool.senses, "episodes": [e.content_hash for e in pool.episodes]}
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def validate_pool(pool: Pool, splits: Splits) -> None:
    """Assert that no held-out task or instance, nor any source outside the split, is in the pool.

    Raises:
        PoolError: naming the offending sources.
    """
    used = {e.source for e in pool.episodes}
    leaked = sorted(used & set(splits.heldout))
    if leaked:
        raise PoolError(f"held-out sources in pool: {leaked}")
    leaked_instances = sorted(
        {
            (e.task, e.instance)
            for e in pool.episodes
            if splits.is_heldout_instance(e.task, e.instance)
        }
    )
    if leaked_instances:
        raise PoolError(f"held-out instances in pool: {leaked_instances}")
    outside = sorted(used - set(splits.sections.get(pool.spec.split, ())))
    if outside:
        raise PoolError(f"sources outside split {pool.spec.split!r} in pool: {outside}")


def pool_to_dict(pool: Pool) -> dict[str, object]:
    """JSON-ready form of a pool: spec, seed, chosen sources, per-stratum steps, episodes, hash."""
    return {
        "spec": asdict(pool.spec),
        "seed": pool.spec.seed,
        "split_version": pool.split_version,
        "split_sha256": pool.split_sha256,
        "chosen_sources": list(pool.chosen_sources),
        "strata": {k: asdict(v) for k, v in pool.strata.items()},
        "total_steps": pool.total_steps,
        "content_hash": pool.content_hash,
        "excluded_instances": pool.excluded_instances,
        "episodes": [
            {
                "path": e.path,
                "source": e.source,
                "task": e.task,
                "instance": e.instance,
                "kind": e.kind,
                "effector": e.effector,
                "steps": e.steps,
                "content_hash": e.content_hash,
            }
            for e in pool.episodes
        ],
    }


def write_pool(pool: Pool, path: Path) -> None:
    """Write the pool JSON (parent directories are created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pool_to_dict(pool), indent=2))


def read_pool(path: Path) -> Pool:
    """Read a pool JSON written by ``write_pool``.

    Only the fields a loader needs are restored on the episodes (path, source, task,
    instance, kind, effector, steps, content hash); other index columns are empty strings.

    Raises:
        PoolError: if the stored spec is invalid.
    """
    raw = json.loads(path.read_text())
    spec = PoolSpec.from_dict(raw["spec"])
    episodes = tuple(
        IndexRecord(
            path=e["path"],
            body="",
            effector=e["effector"],
            source=e["source"],
            family="",
            task=e["task"],
            instance=e["instance"],
            policy="",
            kind=e["kind"],
            seed=0,
            steps=e["steps"],
            content_hash=e["content_hash"],
            tags=(),
        )
        for e in raw["episodes"]
    )
    strata = {k: Stratum(**v) for k, v in raw["strata"].items()}
    return Pool(
        spec,
        raw["split_version"],
        raw["split_sha256"],
        tuple(raw["chosen_sources"]),
        strata,
        episodes,
        raw["content_hash"],
        raw["excluded_instances"],
    )


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point: ``sensorium-pool <index.parquet> <spec.toml> <out.json>``."""
    parser = argparse.ArgumentParser(description="Build a pretraining pool from the episode index.")
    parser.add_argument("index", type=Path, help="parquet index from sensorium-index")
    parser.add_argument("spec", type=Path, help="pool spec TOML")
    parser.add_argument("out", type=Path, help="pool JSON to write")
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--sources", type=Path, default=DEFAULT_REGISTRY_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    splits = load_splits(args.splits, load_registry(args.sources))
    pool = build_pool(read_index(args.index), load_pool_spec(args.spec), splits)
    write_pool(pool, args.out)
    for label, stratum in pool.strata.items():
        log.info(
            "  %-16s %9d steps (target %.0f)", label, stratum.actual_steps, stratum.target_steps
        )
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
