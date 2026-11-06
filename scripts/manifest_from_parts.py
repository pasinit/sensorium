"""Write ``manifest.json`` for a generation run split over machines, or stopped before its own.

Usage: python scripts/manifest_from_parts.py <out dir> <config.toml> <generated dir> [...]

Merges the ``manifest.parts/*.jsonl`` of every generated directory (``generate.merged_parts``: one
entry per episode file; the same file in two parts must have the same content hash) into
``<out dir>/manifest.json``, where the episode files must all be (``humanoid/<name>.npz``). The
config is recorded as is; the provenance is that of the checkout it runs in, not of the runs that
made the episodes (which the episodes' own meta keep).
"""

import hashlib
import json
import sys
import tomllib
from pathlib import Path

from sensorium.generate import merged_parts, provenance


def main() -> None:
    """Merge the parts and write the manifest."""
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    out, config_path = Path(sys.argv[1]), Path(sys.argv[2])
    sources = [Path(p) for p in sys.argv[3:]]
    entries = merged_parts(sources)
    missing = [e["file"] for e in entries if not (out / str(e["file"])).is_file()]
    if missing:
        sys.exit(f"{len(missing)} episode files missing under {out}, e.g. {missing[0]}")
    raw = config_path.read_bytes()
    manifest = {
        "config_file": str(config_path),
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "config": tomllib.loads(raw.decode()),
        "provenance": provenance(),
        "note": f"merged from the manifest.parts of {len(sources)} generated directories",
        "episodes": entries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    sys.stdout.write(f"{len(entries)} episodes -> {out / 'manifest.json'}\n")


if __name__ == "__main__":
    main()
