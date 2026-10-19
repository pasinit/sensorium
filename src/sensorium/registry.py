"""The experience-source registry.

An **experience source** is one ``(body, effector, task)`` triple, e.g. the
piano hands playing RP1M songs. The registry (``configs/sources.toml``) names
every source, says which end effector it uses, which family it belongs to and
whether it is contact-rich or vision-dominated, and maps the recorded
``policy`` name of an episode to its **kind of experience**.

Everything downstream (index, pools, splits) refers to sources by ``name``.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

EFFECTORS = ("hand", "pincer", "none", "alien")
KINDS = ("expert", "noisy", "babbling", "experiment")  # experiment: scripted programs with outcomes
CHARACTER_TAGS = ("contact_rich", "vision_dominated")  # every source has exactly one

DEFAULT_REGISTRY_PATH = Path("configs/sources.toml")


@dataclass(frozen=True)
class Source:
    """One experience source.

    Attributes:
        name: Unique identifier, e.g. ``"piano/rp1m"``.
        body: Body that records it (``"car"``, ``"drone"``, ``"piano"``, ...).
        effector: One of ``EFFECTORS``.
        task: Task or motion set within the body.
        family: Group of related sources (e.g. ``"piano"``).
        tags: Free labels; always contains exactly one of ``CHARACTER_TAGS``.
        programs: Names of the scripted experiment programs that produce it (``[]`` for others).
        props: Names of the prop configurations its scenes use (``[]`` for the plain scene).
            A split must not let a held-out source's program or prop appear in a pretraining
            source (``sensorium.pool.load_splits``).
    """

    name: str
    body: str
    effector: str
    task: str
    family: str
    tags: tuple[str, ...] = field(default_factory=tuple)
    programs: tuple[str, ...] = field(default_factory=tuple)
    props: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Registry:
    """All known sources, the policy -> kind table and the per-body default task."""

    sources: tuple[Source, ...]
    policy_kinds: dict[str, str]
    default_task: dict[str, str]

    def by_name(self, name: str) -> Source:
        """Return the source called ``name``.

        Raises:
            KeyError: if there is no such source (the message lists the known ones).
        """
        for source in self.sources:
            if source.name == name:
                return source
        raise KeyError(f"unknown source {name!r}; known: {sorted(s.name for s in self.sources)}")

    def resolve(self, body: str, task: str | None, effector: str | None = None) -> Source:
        """Return the source for a manifest entry's ``(body, task)``, or ``(body, task, effector)``.

        ``task=None`` means the manifest has no task field: the body's default task is used.
        ``effector`` is needed only when several sources share ``(body, task)`` (a body
        with end-effector variants, like the humanoid).

        Raises:
            ValueError: if nothing is registered for the key, or the key is ambiguous
                without an effector (the registered ones are listed either way).
        """
        if task is None:
            task = self.default_task.get(body)
        matches = [s for s in self.sources if s.body == body and s.task == task]
        if effector is not None:
            matches = [s for s in matches if s.effector == effector]
        if len(matches) == 1:
            return matches[0]
        known = sorted((s.body, s.task, s.effector) for s in self.sources)
        problem = "ambiguous without an effector" if matches else "no registered source"
        raise ValueError(
            f"{problem} for (body={body!r}, task={task!r}, effector={effector!r}); "
            f"add it to configs/sources.toml. Registered (body, task, effector): {known}"
        )

    def kind_of(self, policy: str) -> str:
        """Return the kind of experience (one of ``KINDS``) of a recorded policy name.

        Raises:
            ValueError: if the policy is not in the ``[policy_kinds]`` table.
        """
        try:
            return self.policy_kinds[policy]
        except KeyError:
            raise ValueError(
                f"unknown policy {policy!r}; add it to [policy_kinds] in configs/sources.toml "
                f"(known: {sorted(self.policy_kinds)})"
            ) from None


def parse_source(raw: dict[str, object]) -> Source:
    """Build a ``Source`` from one ``[[source]]`` table.

    Raises:
        ValueError: on missing fields, an unknown effector, or a tag set that does not
            contain exactly one of ``CHARACTER_TAGS``.
    """
    missing = {"name", "body", "effector", "task", "family", "tags"} - set(raw)
    if missing:
        raise ValueError(f"source {raw.get('name', raw)!r}: missing fields {sorted(missing)}")
    name = str(raw["name"])
    if raw["effector"] not in EFFECTORS:
        raise ValueError(
            f"source {name!r}: effector must be one of {EFFECTORS}, got {raw['effector']!r}"
        )
    tags = tuple(str(t) for t in raw["tags"])  # type: ignore[attr-defined]
    character = [t for t in tags if t in CHARACTER_TAGS]
    if len(character) != 1:
        raise ValueError(
            f"source {name!r}: tags must contain exactly one of {CHARACTER_TAGS}, got {tags}"
        )
    return Source(
        name=name,
        body=str(raw["body"]),
        effector=str(raw["effector"]),
        task=str(raw["task"]),
        family=str(raw["family"]),
        tags=tags,
        programs=tuple(str(x) for x in raw.get("programs", [])),  # type: ignore[attr-defined]
        props=tuple(str(x) for x in raw.get("props", [])),  # type: ignore[attr-defined]
    )


def load_registry(path: Path = DEFAULT_REGISTRY_PATH) -> Registry:
    """Load and validate the registry TOML.

    Raises:
        ValueError: on an invalid source, duplicate names or ``(body, task)`` pairs, a kind
            outside ``KINDS`` in ``[policy_kinds]``, or a default task that is not a source.
    """
    raw = tomllib.loads(path.read_text())
    sources = tuple(parse_source(table) for table in raw.get("source", []))
    names = [s.name for s in sources]
    pairs = [(s.body, s.task, s.effector) for s in sources]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate source names in {sorted(names)}")
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"{path}: duplicate (body, task, effector) triples in {sorted(pairs)}")
    policy_kinds = {str(k): str(v) for k, v in raw.get("policy_kinds", {}).items()}
    for policy, kind in policy_kinds.items():
        if kind not in KINDS:
            raise ValueError(f"{path}: policy {policy!r} has kind {kind!r}, want one of {KINDS}")
    default_task = {str(k): str(v) for k, v in raw.get("default_task", {}).items()}
    for body, task in default_task.items():
        if (body, task) not in {(b, t) for b, t, _ in pairs}:
            raise ValueError(
                f"{path}: default task ({body!r}, {task!r}) is not a registered source"
            )
    return Registry(sources=sources, policy_kinds=policy_kinds, default_task=default_task)
