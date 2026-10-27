"""Channel names -> model input layout.

Episodes of different bodies carry different channels (an H1 with hands has 2,775
touch channels, with pincers 946; a car has none). The model sees one fixed
**channel space**: for every group (touch, each proprio group, actions) the union of
the channel *names* of all episode kinds it was built from. A channel that an episode
does not have is zero in that episode and masked where it matters; a channel two kinds
share (the H1's body joints and body skin) is the same column, so it shares weights.

Names decide the groups, so ablations are configuration, not code:

* proprio: ``vestibular`` = ``vest_*``, ``gravity_*``, ``pelvis_height``;
  ``proprio_effector`` = joints of the hands or pincers (``q_lh_*``, ``dq_rp_*`` ...) and the
  controlled agent of a 2-D task (``agent_*``, Push-T);
  ``proprio_body`` = everything else.
* touch: one **region** per skin patch (``body/<link>``, ``hand/<side>/<segment>``,
  ``pincer/<side>/<pad>``), holding that patch's taxels and its 3 dynamics channels
  (``dynamics/.../onset|slip|vibration``). The body's dynamics are named after the link
  (``left_hip_yaw_link``) and its taxels after the joint (``left_hip_yaw``); the
  ``_link`` suffix is dropped to join them.

The H1 with hands and with pincers share one namespace (``humanoid``); every other body
gets its own, so names cannot collide across bodies.
"""

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SENSES = ("vision", "touch", "audio", "proprio_body", "vestibular", "proprio_effector")
"""The senses a model can be configured with; ``action`` is always present."""

INPUT_KEY = {
    "vision": "rgb",
    "touch": "touch",
    "audio": "audio_logmel",
    "proprio_body": "proprio",
    "vestibular": "proprio",
    "proprio_effector": "proprio",
}
"""Which ``MODEL_INPUT_KEYS`` array feeds each sense."""

PROPRIO_GROUPS = ("proprio_body", "vestibular", "proprio_effector")
GROUPS = ("touch", *PROPRIO_GROUPS, "action")
N_MELS = 32

EFFECTOR_JOINT = re.compile(r"^(?:d?q_)?(?:lh|rh|lp|rp)_")
VESTIBULAR_PREFIXES = ("vest_", "gravity_", "pelvis_height")
AGENT_PREFIX = "agent_"
"""Proprio of a 2-D task's controlled agent (Push-T: ``agent_x``, ``agent_y``, ``agent_vx``,
``agent_vy``), grouped as an effector."""
BLOCKS = ("body", "effector", "other")
SIDES = ("none", "left", "right")


def namespace(body: str) -> str:
    """Channel namespace of a body: both H1 variants share ``humanoid``."""
    return body


def proprio_group(name: str) -> str:
    """The proprio group a channel name belongs to (``agent_*``: the controlled agent of a 2-D
    task such as Push-T, an effector)."""
    if name.startswith(VESTIBULAR_PREFIXES):
        return "vestibular"
    if EFFECTOR_JOINT.match(name) or name.startswith(AGENT_PREFIX):
        return "proprio_effector"
    return "proprio_body"


def region_path(name: str) -> list[str]:
    """Path parts of the skin region of a touch channel name (see the module docstring)."""
    parts = name.split("/")
    if parts[0] == "dynamics":
        parts = parts[1:-1]
        if parts[0] == "body":
            parts[-1] = parts[-1].removesuffix("_link")
    elif len(parts) > 1 and parts[-1].isdigit():
        parts = parts[:-1]
    return parts


@dataclass(frozen=True)
class Header:
    """The names and timing of one episode, enough to place it in a channel space.

    Attributes:
        body: Body identifier (``humanoid``, ``car`` ...).
        dt: Control period in seconds.
        proprio_names, touch_names, action_names: One name per column.
    """

    body: str
    dt: float
    proprio_names: tuple[str, ...]
    touch_names: tuple[str, ...]
    action_names: tuple[str, ...]


def read_header(path: Path) -> Header:
    """Read an episode file's header without decoding any array.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        KeyError: if the file has no header (not an episode).
    """
    with np.load(path, allow_pickle=False) as data:
        raw = json.loads(str(data["header"]))
    return Header(
        body=raw["body"],
        dt=raw["dt"],
        proprio_names=tuple(raw["proprio_names"]),
        touch_names=tuple(raw["touch_names"]),
        action_names=tuple(raw["action_names"]),
    )


@dataclass(frozen=True)
class Columns:
    """Where one episode kind's channels sit in the union space.

    Attributes:
        src, dst: Per group, the episode columns and the union columns they map to.
        region_mask: ``[R]`` bool, which touch regions this kind has.
    """

    src: dict[str, np.ndarray]
    dst: dict[str, np.ndarray]
    region_mask: np.ndarray


@dataclass
class ChannelSpace:
    """The union of channel names over every episode kind in a dataset.

    Attributes:
        keys: Per group, the ordered union of channel keys (``namespace/name`` for
            bodies other than the humanoid).
        regions: Ordered union of touch region keys.
        region_of: Per touch union column, its region index.
        stats: Per group (proprio groups and ``audio``), ``{"mean": ..., "std": ...}`` per
            union column; empty until ``set_stats``.
        drop_prefix: Proprio channels whose name starts with one of these are not in the space
            (``("eff_",)``: proprio without the actuator-effort channels). Stored in checkpoints.
    """

    keys: dict[str, list[str]] = field(default_factory=lambda: {g: [] for g in GROUPS})
    regions: list[str] = field(default_factory=list)
    region_of: list[int] = field(default_factory=list)
    stats: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    drop_prefix: tuple[str, ...] = ()

    @classmethod
    def from_headers(
        cls, headers: Iterable[Header], drop_prefix: Sequence[str] = ()
    ) -> "ChannelSpace":
        """Build the union space of some episode headers (order-independent).

        Args:
            headers: The episode headers.
            drop_prefix: Proprio channel-name prefixes to leave out of the space.
        """
        drop = tuple(drop_prefix)
        found: dict[str, set[str]] = {g: set() for g in GROUPS}
        region_by_key: dict[str, str] = {}
        for h in headers:
            ns = namespace(h.body)
            for name in h.proprio_names:
                if not name.startswith(drop):
                    found[proprio_group(name)].add(_key(ns, name, h.body))
            found["action"].update(_key(ns, n, h.body) for n in h.action_names)
            for name in h.touch_names:
                key = _key(ns, name, h.body)
                found["touch"].add(key)
                region_by_key[key] = _region_key(name, ns, h.body)
        space = cls(drop_prefix=drop)
        regions = sorted(set(region_by_key.values()))
        space.regions = regions
        touch = sorted(found["touch"], key=lambda k: (regions.index(region_by_key[k]), k))
        space.keys = {g: sorted(found[g]) for g in GROUPS} | {"touch": touch}
        space.region_of = [regions.index(region_by_key[k]) for k in touch]
        return space

    def columns(self, header: Header) -> Columns:
        """Map an episode kind into this space.

        Raises:
            KeyError: if the header has a channel that the space does not contain.
        """
        ns = namespace(header.body)
        index = {g: {k: i for i, k in enumerate(self.keys[g])} for g in GROUPS}
        src: dict[str, list[int]] = {g: [] for g in GROUPS}
        dst: dict[str, list[int]] = {g: [] for g in GROUPS}
        for i, name in enumerate(header.proprio_names):
            if name.startswith(self.drop_prefix):
                continue
            g = proprio_group(name)
            src[g].append(i)
            dst[g].append(index[g][_key(ns, name, header.body)])
        for g, names in (("touch", header.touch_names), ("action", header.action_names)):
            for i, name in enumerate(names):
                src[g].append(i)
                dst[g].append(index[g][_key(ns, name, header.body)])
        mask = np.zeros(len(self.regions), dtype=bool)
        for d in dst["touch"]:
            mask[self.region_of[d]] = True
        return Columns(
            {g: np.array(v, dtype=np.int64) for g, v in src.items()},
            {g: np.array(v, dtype=np.int64) for g, v in dst.items()},
            mask,
        )

    def size(self, group: str) -> int:
        """Number of union channels of a group (``audio`` is ``N_MELS``)."""
        return N_MELS if group == "audio" else len(self.keys[group])

    def region_table(self) -> np.ndarray:
        """``[R, P]`` union touch columns per region, padded with ``-1``."""
        members: list[list[int]] = [[] for _ in self.regions]
        for col, region in enumerate(self.region_of):
            members[region].append(col)
        width = max((len(m) for m in members), default=1)
        table = np.full((len(self.regions), width), -1, dtype=np.int64)
        for r, cols in enumerate(members):
            table[r, : len(cols)] = cols
        return table

    def region_meta(self) -> list[tuple[int, int, str]]:
        """Per region ``(block, side, part)``: indices into ``BLOCKS`` and ``SIDES``, last name."""
        meta = []
        for key in self.regions:
            parts = key.split("/")
            block = parts[0] if parts[0] in ("body", "hand", "pincer") else "other"
            block = "effector" if block in ("hand", "pincer") else block
            side = next((s for s in ("left", "right") if s in parts[1:] or s in parts[-1]), "none")
            meta.append((BLOCKS.index(block), SIDES.index(side), parts[-1]))
        return meta

    def set_stats(self, group: str, mean: np.ndarray, std: np.ndarray) -> None:
        """Store the normalisation of a group (one value per union column)."""
        self.stats[group] = {"mean": mean.tolist(), "std": std.tolist()}

    def to_json(self) -> str:
        """Serialise (stored in checkpoints)."""
        return json.dumps(
            {
                "keys": self.keys,
                "regions": self.regions,
                "region_of": self.region_of,
                "stats": self.stats,
                "drop_prefix": list(self.drop_prefix),
            }
        )

    @classmethod
    def from_json(cls, text: str) -> "ChannelSpace":
        """Inverse of ``to_json``."""
        raw = json.loads(text)
        return cls(
            raw["keys"],
            raw["regions"],
            raw["region_of"],
            raw["stats"],
            tuple(raw.get("drop_prefix", ())),
        )


def _key(ns: str, name: str, body: str) -> str:
    """Union key of a channel: humanoid names stay as they are, other bodies are namespaced."""
    return name if body == "humanoid" else f"{ns}/{name}"


def _region_key(name: str, ns: str, body: str) -> str:
    path = "/".join(region_path(name))
    return path if body == "humanoid" else f"{ns}/{path}"


def check_senses(senses: Sequence[str]) -> tuple[str, ...]:
    """Validate a sense configuration.

    Raises:
        ValueError: on an unknown or repeated sense.
    """
    unknown = sorted(set(senses) - set(SENSES))
    if unknown or len(set(senses)) != len(senses):
        raise ValueError(f"senses must be distinct members of {SENSES}, got {list(senses)}")
    return tuple(s for s in SENSES if s in senses)
