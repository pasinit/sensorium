"""Per-episode appearance of the H1 scene: room colours, object shades, lighting.

Every episode draws new colours for the floor, walls, table and seat, a new
key-light direction, strength and colour, and a new headlight level, so that
pixels vary across episodes in ways that carry no information about the task.
The table objects keep their material's colour family (wood brown, plastic
blue, metal grey, rubber red) and only change shade, because what an object
looks like should still predict how it sounds and feels.

Only material colours and light parameters change, which MuJoCo reads at
every render, so the renderer and its uploaded textures stay valid. Ranges are
our choice, not taken from a paper; ``docs/vision.md`` lists them.
"""

import colorsys
from dataclasses import dataclass

import mujoco
import numpy as np

from sensorium.bodies.h1_scene import MATERIALS, ROOM_MATERIALS

# (hue range, saturation range, value range) of each room material's tint.
ROOM_TINTS = {
    "floor": ((0.0, 1.0), (0.0, 0.35), (0.35, 0.9)),
    "wall": ((0.0, 1.0), (0.0, 0.25), (0.55, 1.0)),
    "table_wood": ((0.0, 1.0), (0.1, 0.6), (0.3, 0.8)),
    "seat": ((0.0, 1.0), (0.0, 0.5), (0.15, 0.6)),
}
OBJECT_HUE_SHIFT = 0.03  # objects: hue +- this, value scaled by OBJECT_VALUE_SCALE
OBJECT_VALUE_SCALE = (0.8, 1.15)
LIGHT_ELEVATION_DEG = (45.0, 90.0)  # the key light's angle above the horizon
LIGHT_DIFFUSE = (0.3, 0.7)
LIGHT_WARMTH = 0.08  # red vs blue imbalance of the key light, +- this
HEADLIGHT_AMBIENT = (0.15, 0.35)
HEADLIGHT_DIFFUSE = (0.25, 0.55)
# The key light sits LIGHT_DISTANCE back along its direction from LIGHT_TARGET,
# so the robot and the table stay inside its shadow map whatever the angle.
LIGHT_TARGET = np.array([0.4, 0.0, 0.8])
LIGHT_DISTANCE = 4.0


@dataclass(frozen=True)
class Appearance:
    """One episode's colours and lights (stored in ``meta["appearance"]``).

    Attributes:
        tints: Material name -> RGB in ``[0, 1]`` (room materials and objects).
        light_dir: Unit vector the key light points along (world frame).
        light_diffuse: RGB of the key light.
        headlight_ambient, headlight_diffuse: Grey levels of the camera headlight.
    """

    tints: dict[str, list[float]]
    light_dir: list[float]
    light_diffuse: list[float]
    headlight_ambient: float
    headlight_diffuse: float

    def as_meta(self) -> dict[str, object]:
        """A JSON-serialisable copy, values rounded to 4 decimals."""
        return {
            "tints": {name: rounded(rgb) for name, rgb in self.tints.items()},
            "light_dir": rounded(self.light_dir),
            "light_diffuse": rounded(self.light_diffuse),
            "headlight_ambient": round(self.headlight_ambient, 4),
            "headlight_diffuse": round(self.headlight_diffuse, 4),
        }


def rounded(values: list[float]) -> list[float]:
    """``values`` rounded to 4 decimals."""
    return [round(float(v), 4) for v in values]


def hsv(hue: float, saturation: float, value: float) -> list[float]:
    """RGB in ``[0, 1]`` of an HSV colour (hue wraps around)."""
    return list(colorsys.hsv_to_rgb(hue % 1.0, saturation, value))


def draw_appearance(rng: np.random.Generator) -> Appearance:
    """A random appearance; the same ``rng`` state gives the same appearance."""
    tints = {
        name: hsv(rng.uniform(*hue), rng.uniform(*saturation), rng.uniform(*value))
        for name, (hue, saturation, value) in ROOM_TINTS.items()
    }
    for name, (rgba, *_) in MATERIALS.items():
        r, g, b = (float(c) for c in rgba.split()[:3])
        hue, saturation, value = colorsys.rgb_to_hsv(r, g, b)
        hue += rng.uniform(-OBJECT_HUE_SHIFT, OBJECT_HUE_SHIFT)
        value = min(1.0, value * rng.uniform(*OBJECT_VALUE_SCALE))
        tints[name] = hsv(hue, saturation, value)
    elevation = np.radians(rng.uniform(*LIGHT_ELEVATION_DEG))
    azimuth = rng.uniform(0.0, 2 * np.pi)
    toward_light = [
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ]
    strength, warmth = rng.uniform(*LIGHT_DIFFUSE), rng.uniform(-LIGHT_WARMTH, LIGHT_WARMTH)
    return Appearance(
        tints=tints,
        light_dir=[-float(c) for c in toward_light],
        light_diffuse=[strength * (1 + warmth), strength, strength * (1 - warmth)],
        headlight_ambient=float(rng.uniform(*HEADLIGHT_AMBIENT)),
        headlight_diffuse=float(rng.uniform(*HEADLIGHT_DIFFUSE)),
    )


def apply_appearance(model: mujoco.MjModel, appearance: Appearance) -> None:
    """Set ``model``'s material colours and lights; call ``mj_forward`` afterwards.

    Raises:
        KeyError: if a tinted material or the ``key`` light is not in the model.
    """
    for name in (*ROOM_MATERIALS, *MATERIALS):
        model.mat_rgba[model.material(name).id, :3] = appearance.tints[name]
    light = model.light("key").id
    model.light_dir[light] = appearance.light_dir
    model.light_pos[light] = LIGHT_TARGET - LIGHT_DISTANCE * np.asarray(appearance.light_dir)
    model.light_diffuse[light] = appearance.light_diffuse
    model.vis.headlight.ambient[:] = appearance.headlight_ambient
    model.vis.headlight.diffuse[:] = appearance.headlight_diffuse
