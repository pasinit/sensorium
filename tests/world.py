"""A tiny synthetic world for registry/index/pool tests: sources, episodes, manifest, splits."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sensorium.episode import IMAGE_SIZE, SAMPLE_RATE, Episode

POLICIES = ("expert", "noisy", "explore")
LENGTHS = (8, 10, 12)  # episode lengths cycle through these

REGISTRY_TOML = (
    """
[policy_kinds]
expert = "expert"
noisy = "noisy"
explore = "babbling"

[default_task]
a0 = "t"
a1 = "t"
a2 = "t"
a3 = "t"
a4 = "t"
a5 = "t"
h0 = "play0"
h1 = "play1"
h2 = "play2"

"""
    + "".join(
        f"""
[[source]]
name = "alien/a{i}"
body = "a{i}"
effector = "alien"
task = "t"
family = "fam{i % 2}"
tags = ["vision_dominated"]
"""
        for i in range(6)
    )
    + "".join(
        f"""
[[source]]
name = "hand/h{i}"
body = "h{i}"
effector = "hand"
task = "play{i}"
family = "piano"
tags = ["contact_rich"]
"""
        for i in range(3)
    )
)

SPLITS_TOML = """
version = "test"

[pretrain]
sources = ["alien/a0", "alien/a1", "alien/a2", "alien/a3", "alien/a4", "alien/a5",
           "hand/h0", "hand/h1"]

[heldout.tasks]
"hand/h2" = { reason = "test held-out task", tag = "contact_rich" }

[heldout.instances.play0]
ids = ["inst0", "inst1"]
reason = "test held-out instances"
tag = "contact_rich"
"""


@dataclass(frozen=True)
class World:
    """Paths of a synthetic world written by ``make_world``."""

    registry: Path
    splits: Path
    manifest: Path


def make_episode(steps: int, dt: float = 0.1) -> Episode:
    """A valid all-zeros episode with ``steps`` steps."""
    return Episode(
        body="test",
        dt=dt,
        rgb=np.zeros((steps, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
        audio=np.zeros((steps, round(dt * SAMPLE_RATE)), dtype=np.int16),
        audio_logmel=np.zeros((steps, 4), dtype=np.float16),
        proprio=np.zeros((steps, 2), dtype=np.float32),
        touch=np.zeros((steps, 0), dtype=np.float32),
        action=np.zeros((steps, 3), dtype=np.float32),
        proprio_names=["a", "b"],
        touch_names=[],
        action_names=["x", "y", "z"],
        text=[""] * steps,
    )


def make_world(root: Path, episodes_per_policy: int = 40, with_instance: bool = True) -> World:
    """Write registry, split file, episodes and ``manifest.json`` under ``root``.

    Every source (6 alien, 3 hand) gets ``episodes_per_policy`` episodes per policy.
    Episode ``index`` has instance ``inst{index % 4}`` unless ``with_instance`` is False
    (a manifest that predates the field). Episodes are saved with ``Episode.save`` once per
    length and copied.
    """
    templates = {}
    for length in LENGTHS:
        templates[length] = root / "templates" / f"{length}.npz"
        make_episode(length).save(templates[length])
    registry = root / "sources.toml"
    registry.write_text(REGISTRY_TOML)
    splits = root / "splits.toml"
    splits.write_text(SPLITS_TOML)
    entries = []
    sources = [(f"a{i}", f"alien/a{i}") for i in range(6)] + [
        (f"h{i}", f"hand/h{i}") for i in range(3)
    ]
    for body, source in sources:
        for policy in POLICIES:
            for index in range(episodes_per_policy):
                length = LENGTHS[index % len(LENGTHS)]
                name = f"{body}/{body}-{policy}-{index:03d}.npz"
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(templates[length], root / name)
                entries.append(
                    {
                        "file": name,
                        "body": body,
                        "policy": policy,
                        "index": index,
                        "seed": index,
                        "steps": length,
                        "content_hash": f"{source}-{policy}-{index}",
                    }
                    | ({"instance": f"inst{index % 4}"} if with_instance else {})
                )
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"episodes": entries}))
    return World(registry, splits, manifest)
