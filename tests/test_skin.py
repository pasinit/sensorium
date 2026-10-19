"""Dense skin and contact dynamics on small hand-written scenes (no H1 assets needed)."""

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="needs the simulator extra: make setup")

import mujoco  # noqa: E402

from sensorium.bodies.h1_skin import (  # noqa: E402
    ContactDynamics,
    DenseSkin,
    SkinRegion,
    read_contacts,
    spread_linear,
    surface_points,
)

# A 4 x 4 x 2 cm box on a floor; a second, smaller box overlaps its top face.
SCENE = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="block" pos="0 0 {height}">
      <freejoint/>
      <geom name="block" type="box" size="0.02 0.02 0.01" mass="0.5"/>
      <geom name="lid" type="box" size="0.01 0.01 0.01" pos="0 0 0.005" mass="0.01"/>
    </body>
  </worldbody>
</mujoco>
"""


def scene(height: float = 0.0099) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(SCENE.format(height=height))
    return model, mujoco.MjData(model)


def flat_region(spacing: float = 0.004) -> SkinRegion:
    """Points on a 5 x 5 grid in z = 0 facing up, and the same grid facing down."""
    grid = np.array([[x, y, 0.0] for x in range(-2, 3) for y in range(-2, 3)]) * spacing
    up, down = np.tile([0.0, 0.0, 1.0], (25, 1)), np.tile([0.0, 0.0, -1.0], (25, 1))
    return SkinRegion(
        0, "test", spacing, np.concatenate([grid, grid]), np.concatenate([up, down]), 0
    )


def test_spread_linear_conserves_force_and_respects_facing() -> None:
    region = flat_region()
    weights = spread_linear(region, np.zeros(3), np.array([0.0, 0.0, 1.0]))
    assert weights.sum() == pytest.approx(1.0)
    assert (weights[25:] == 0).all()  # points facing away get nothing
    assert weights.argmax() == 12  # the centre point
    assert (weights[:25] > 0).sum() > 1  # the contact is shared, not binned to one point
    far = spread_linear(region, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]))
    assert far.sum() == pytest.approx(1.0) and far.max() == 1.0 and far.argmax() >= 25


def test_surface_points_cover_the_outer_surface() -> None:
    model, _ = scene()
    spacing = 0.004
    points, normals = surface_points(
        model, model.body("block").id, spacing, np.random.default_rng(0)
    )
    distance = np.linalg.norm(points[:, None] - points[None], axis=-1) + np.eye(len(points))
    assert distance.min() >= spacing
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    # every point is on the union's surface: none strictly inside the big box or the lid
    big = np.all(np.abs(points) < [0.02 - 1e-4, 0.02 - 1e-4, 0.01 - 1e-4], axis=1)
    lid = np.all(np.abs(points - [0, 0, 0.005]) < [0.01 - 1e-4, 0.01 - 1e-4, 0.01 - 1e-4], axis=1)
    assert not big.any() and not lid.any()
    assert len(points) > 50  # ~ 40 cm^2 of surface at 4 mm spacing


def test_dense_skin_conserves_the_contact_force() -> None:
    model, data = scene()
    for _ in range(200):  # let the block settle on the floor
        mujoco.mj_step(model, data)
    block = model.body("block").id
    skin = DenseSkin(model, {block: ("test/block", 0.004)})
    contacts = read_contacts(model, data)
    force = skin.read(data, contacts)
    assert force.sum() == pytest.approx(contacts.force.sum(), rel=1e-5)
    assert force.sum() == pytest.approx(0.51 * 9.81, rel=0.05)  # the block's weight
    bottom = np.array([p[2] < -0.0099 for p in skin.regions[0].points])
    assert force[~bottom].sum() == pytest.approx(0.0, abs=1e-6)  # only the face on the floor
    assert (force[bottom] > 0).sum() > 4  # spread over many points


def test_contacts_report_force_and_sliding_speed() -> None:
    model, data = scene()
    for _ in range(200):
        mujoco.mj_step(model, data)
    data.qvel[0] = 0.5  # shove the block along x
    mujoco.mj_step(model, data)
    contacts = read_contacts(model, data)
    wrench, total = np.zeros(6), 0.0
    for i in range(data.ncon):
        mujoco.mj_contactForce(model, data, i, wrench)
        total += wrench[0] if wrench[0] >= 0.05 else 0.0
    assert contacts.force.sum() == pytest.approx(total)
    assert contacts.slip.max() == pytest.approx(0.5, abs=0.1)


def test_dynamics_onset_slip_and_vibration() -> None:
    model, data = scene(height=0.05)  # starts in the air
    block = model.body("block").id
    dynamics = ContactDynamics(model, {block: "test/block"})
    assert dynamics.names == [f"dynamics/test/block/{f}" for f in ("onset", "slip", "vibration")]
    mujoco.mj_forward(model, data)
    dynamics.reset(data)
    for _ in range(100):  # falls ~4 cm and lands
        mujoco.mj_step(model, data)
        dynamics.substep(data)
    onset, slip, vibration = dynamics.features()
    assert onset == 1.0 and vibration > 0.0 and slip == pytest.approx(0.0, abs=0.01)
    dynamics.clear()
    for _ in range(100):  # resting: no new onset, little vibration
        mujoco.mj_step(model, data)
        dynamics.substep(data)
    onset, slip, rest_vibration = dynamics.features()
    assert onset == 0.0 and rest_vibration < vibration
    dynamics.clear()
    data.qvel[0] = 0.5
    for _ in range(10):
        mujoco.mj_step(model, data)
        dynamics.substep(data)
    assert dynamics.features()[1] > 0.1  # sliding
