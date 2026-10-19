"""Write the static touch-channel layout of the H1 (one JSON per effector).

Usage: python scripts/export_skin_layout.py [out_dir]   (default ``data/sample_h1``)

Writes ``skin_layout_hand.json`` and ``skin_layout_pincer.json``: for every touch channel
that has a position (body grid taxels, dense skin points), where it sits in its body's
frame, plus the bodies' poses in the pelvis frame. See ``sensorium.bodies.h1_layout``
and ``docs/touch.md``.
"""

import json
import sys
from pathlib import Path

from sensorium.bodies.h1_scene import EFFECTORS
from sensorium.bodies.humanoid import HumanoidRecorder


def main() -> None:
    """Export both layouts."""
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/sample_h1")
    out.mkdir(parents=True, exist_ok=True)
    for effector in EFFECTORS:
        with HumanoidRecorder({"effector": effector, "posture": "seated"}) as recorder:
            layout = recorder.skin_layout()
        path = out / f"skin_layout_{effector}.json"
        path.write_text(json.dumps(layout, separators=(",", ":")))
        sys.stdout.write(f"{path} ({path.stat().st_size / 1e3:.0f} kB)\n")


if __name__ == "__main__":
    main()
