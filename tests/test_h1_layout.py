"""The static touch-channel layout (needs the assets from ``make h1-assets``)."""

import json

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="needs the simulator extra: make setup")

from sensorium.bodies.h1_scene import assets_available  # noqa: E402
from sensorium.bodies.humanoid import HumanoidRecorder  # noqa: E402

needs_assets = pytest.mark.skipif(
    not assets_available(), reason="H1 assets missing: run `make h1-assets`"
)


@needs_assets
@pytest.mark.parametrize(("effector", "dense", "grid"), [("hand", 2429, 184), ("pincer", 690, 184)])
def test_layout_matches_the_touch_channels(effector: str, dense: int, grid: int) -> None:
    with HumanoidRecorder({"effector": effector, "posture": "seated"}) as recorder:
        layout = json.loads(json.dumps(recorder.skin_layout()))  # survives JSON
        names = recorder.touch_names
    assert layout["channels"] == {"grid": grid, "dense": dense}
    points = 0
    for region in layout["dense_regions"]:
        count = len(region["points"])
        first = region["first_channel"]
        assert names[first] == f"{region['name']}/0" and names[first + count - 1].startswith(
            region["name"]
        )
        norms = np.linalg.norm(np.array(region["normals"]), axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3) and np.array(region["points"]).shape == (count, 3)
        assert region["body"] in layout["body_pose"]
        points += count
    assert points == dense
    channels = sorted(c for patch in layout["grid_patches"] for c in patch["channels"])
    assert channels == list(range(grid))  # every kept grid taxel is in exactly one patch
    for patch in layout["grid_patches"]:
        columns, rows = patch["size"]
        assert len(patch["channels"]) == columns * rows
        assert [names[c] for c in patch["channels"]] == patch["taxel_names"]
