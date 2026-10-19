"""Write ``manifest.json`` for a generation that was stopped before it wrote its own.

Usage: python scripts/manifest_from_dir.py <generated dir> <config.toml>

Reads every ``<dir>/humanoid/humanoid-<variant>-<policy>-<index>.npz``, and records the entries
``sensorium.generate`` would have (file, policy, variant, index, task, instance, effector, seed,
steps, end reason, content hash). The config is recorded as is; the provenance is that of the
checkout it runs in, not of the run that made the episodes (which the episodes' own meta keep).
"""

import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path

from sensorium.episode import Episode
from sensorium.generate import content_hash, provenance


def main() -> None:
    """Build and write the manifest."""
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    directory, config_path = Path(sys.argv[1]), Path(sys.argv[2])
    raw = config_path.read_bytes()
    entries = []
    for path in sorted((directory / "humanoid").glob("humanoid-*.npz")):
        match = re.fullmatch(r"humanoid-(.+)-([a-z_]+)-(\d+)\.npz", path.name)
        if match is None:
            continue
        variant, policy, index = match.group(1), match.group(2), int(match.group(3))
        e = Episode.load(path)
        entries.append(
            {
                "file": f"humanoid/{path.name}",
                "body": "humanoid",
                "policy": policy,
                "variant": variant,
                "index": index,
                "task": e.meta.get("task"),
                "instance": e.meta.get("instance", "default"),
                "effector": e.meta.get("effector"),
                "seed": e.meta.get("seed"),
                "steps": e.length,
                "end_reason": e.meta.get("end_reason"),
                "content_hash": content_hash(e),
            }
        )
    manifest = {
        "config_file": str(config_path),
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "config": tomllib.loads(raw.decode()),
        "provenance": provenance(),
        "note": "manifest rebuilt from the episodes after the generation was stopped",
        "episodes": entries,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    sys.stdout.write(f"{len(entries)} episodes -> {directory / 'manifest.json'}\n")


if __name__ == "__main__":
    main()
