"""Static layout of the H1's touch channels: where every skin point sits on the body.

The model fixes the layout (it is a function of the scene, seed 0 for the dense points), so
it is metadata, not privileged state: a loader may feed positions to a model as input
(for example as taxel coordinates). ``skin_layout`` returns it as a JSON-serialisable
dict; ``scripts/export_skin_layout.py`` writes one file per effector.

Channels (``touch`` columns, in this order; see ``docs/touch.md``): the body's ``touch_grid``
patches (normal force per taxel), then the dense skin points of the hands or pincer pads,
then three dynamics channels per skin region (no position of their own: they belong to the
region's body). Positions are in the frame of the MuJoCo body that carries the patch or the
region, in metres; ``body_pose`` gives each such body's pose in the pelvis frame at the
model's default pose (``qpos0``: standing, arms down) for a coarse body map.
"""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sensorium.bodies.h1_sense import SkinLayout
from sensorium.bodies.h1_skin import DenseSkin

LAYOUT_VERSION = 1
DECIMALS = 5


def rounded(array: np.ndarray) -> list[list[float]]:
    """A float array as nested lists rounded to ``DECIMALS`` decimals."""
    return [[round(float(v), DECIMALS) for v in row] for row in np.atleast_2d(array)]


def grid_shapes(body_sensors: str) -> dict[str, tuple[tuple[int, int], tuple[float, float]]]:
    """Grid size and field of view of every ``touch_grid`` sensor, by sensor name, from its XML."""
    shapes = {}
    for plugin in ET.parse(body_sensors).getroot().iter("plugin"):
        config = {c.get("key"): c.get("value", "") for c in plugin.iter("config")}
        columns, rows = (int(v) for v in config["size"].split())
        fov = tuple(float(v) for v in config["fov"].split())
        shapes[str(plugin.get("name"))] = ((columns, rows), (fov[0], fov[1]))
    return shapes


def body_pose(model: mujoco.MjModel, bodies: set[int]) -> dict[str, dict[str, list[float]]]:
    """Pose of each body in the pelvis frame at the default pose: position and 3x3 rotation."""
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_kinematics(model, data)
    pelvis = model.body("pelvis").id
    to_pelvis = data.xmat[pelvis].reshape(3, 3).T
    out = {}
    for body in sorted(bodies):
        position = to_pelvis @ (data.xpos[body] - data.xpos[pelvis])
        rotation = to_pelvis @ data.xmat[body].reshape(3, 3)
        out[str(model.body(body).name)] = {
            "position": [round(float(v), DECIMALS) for v in position],
            "rotation": [round(float(v), DECIMALS) for v in rotation.reshape(-1)],
        }
    return out


def skin_layout(
    model: mujoco.MjModel,
    skin: SkinLayout,
    dense: DenseSkin,
    grid_kept: np.ndarray,
    body_sensors: str,
    effector: str,
) -> dict[str, object]:
    """The layout of one effector variant's ``touch`` channels.

    Args:
        model, skin, dense: the compiled scene and its grid and dense skin layouts.
        grid_kept: ``[grid taxels]`` which grid taxels are touch channels (the hands' own
            grid patches are replaced by dense points).
        body_sensors: path of the XML that declares the body's ``touch_grid`` sensors.
        effector: ``"hand"`` or ``"pincer"``, recorded in the file.
    """
    shapes = grid_shapes(body_sensors)
    channel_of_grid = np.cumsum(grid_kept) - 1  # touch column of each kept grid taxel
    patches: list[dict[str, object]] = []
    names = np.array(skin.names)
    for prefix in dict.fromkeys(n.rsplit("/", 1)[0] for n in skin.names):
        taxels = np.flatnonzero([n.rsplit("/", 1)[0] == prefix for n in skin.names])
        if not grid_kept[taxels].all():
            continue
        body = int(skin.bodies[taxels[0]])
        sensor = prefix.split("/", 1)[1] if prefix.startswith("body/") else prefix
        site_name = _site_name(model, body, sensor)
        site = model.site(site_name)
        (columns, rows), fov = shapes[site_name]
        patches.append(
            {
                "name": prefix,
                "body": str(model.body(body).name),
                "site_pos": [round(float(v), DECIMALS) for v in site.pos],
                "site_quat": [round(float(v), DECIMALS) for v in site.quat],
                "size": [columns, rows],
                "fov_deg": list(fov),
                "channels": [int(channel_of_grid[t]) for t in taxels],
                "taxel_names": names[taxels].tolist(),
            }
        )
    first = int(grid_kept.sum())
    regions = [
        {
            "name": region.name,
            "body": str(model.body(region.body).name),
            "spacing": region.spacing,
            "first_channel": first + region.start,
            "points": rounded(region.points),
            "normals": rounded(region.normals),
        }
        for region in dense.regions
    ]
    bodies = {model.body(str(p["body"])).id for p in patches} | {r.body for r in dense.regions}
    return {
        "version": LAYOUT_VERSION,
        "effector": effector,
        "units": "m",
        "frame": "each patch or region is in the frame of its body (MuJoCo body frame)",
        "channels": {"grid": first, "dense": len(dense.names)},
        "grid_taxel_note": "all taxels of a patch sit at the patch's site; a taxel is a "
        "direction bin of the site's outward z axis (size = columns x rows over fov_deg, "
        "row-major as in taxel_names), bin centres assumed uniform in angle (plugin source "
        "not checked)",
        "grid_patches": patches,
        "dense_regions": regions,
        "dynamics_note": "three channels per region (onset, slip, vibration) follow the dense "
        "channels, in the order of touch_names",
        "body_pose": body_pose(model, bodies),
    }


def _site_name(model: mujoco.MjModel, body: int, sensor: str) -> str:
    """Name of the ``touch_grid`` site of ``sensor`` (a patch name without ``body/``)."""
    name = f"{sensor}_touch"
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) < 0:
        raise ValueError(f"no touch site {name!r} on body {model.body(body).name}")
    return name
