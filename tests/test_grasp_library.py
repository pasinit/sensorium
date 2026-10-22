"""The grasp library: descriptors, keys, scoring and loading."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="needs the simulator extra: make setup")

import mujoco  # noqa: E402

from sensorium.bodies.h1_grasp_library import (  # noqa: E402
    GraspLibrary,
    grasp_features,
    grasp_key,
)

SCENE = """<mujoco><worldbody>
  <body name="obj_box" pos="0.5 0 0.1" euler="90 0 0"><freejoint/>
    <geom type="box" size="0.05 0.03 0.02"/></body>
</worldbody></mujoco>"""


def test_a_box_on_its_side_has_its_key_and_symmetric_descriptor() -> None:
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body = model.body("obj_box").id
    name, rest = grasp_key(model, data, body)
    assert name == "obj_box" and rest == 1  # turned about x: its y axis is now vertical
    down, side = np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0])
    point = data.xpos[body] + [0.0, 0.0, 0.01]
    features = grasp_features(model, data, body, point, down, side)
    flipped = grasp_features(model, data, body, point, -down, -side)
    assert np.allclose(features[:9], flipped[:9])  # axis flips do not matter
    assert features[1] == pytest.approx(1.0)  # the approach runs along the box's y axis
    assert features[11] == pytest.approx(0.01 / 0.05)  # 1 cm over the centre, in units of 5 cm
    assert features[12] == pytest.approx(0.03 / 0.05)  # the box spans 3 cm along the approach


def test_scores_shrink_towards_the_neighbours_success(tmp_path: Path) -> None:
    path = tmp_path / "lib.jsonl"
    rows = [{"object": "a", "rest": 2, "features": [float(i)], "lifted": i < 10} for i in range(20)]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    library = GraspLibrary.load([path])
    assert library.score(("a", 2), np.array([0.0])) == pytest.approx(7.5 / 8)
    assert library.score(("a", 2), np.array([19.0])) == pytest.approx(0.5 / 8)
    assert library.score(("b", 2), np.array([0.0])) == 0.0
    assert library.rate(("a", 2)) == pytest.approx(0.5) and library.rate(("a", 0)) == 0.0


ROOT = Path(__file__).resolve().parent.parent


def test_the_shipped_pincer_library_covers_every_table_object() -> None:
    paths = sorted((ROOT / "configs/grasp_library").glob("pincer_*.jsonl"))
    library = GraspLibrary.load(paths)
    names = {name for name, _ in library.features}
    assert {"obj_wood_box", "obj_metal_box", "obj_wood_tallbox"} <= names
    assert all(len(f[0]) == 15 for f in library.features.values())  # the current descriptor
