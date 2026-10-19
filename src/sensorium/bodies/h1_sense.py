"""What the H1 feels: skin layout, proprioception names and the contact monitor.

* ``SkinLayout`` indexes the ``touch_grid`` plugin sensors of a compiled model.
  Each sensor is a grid of taxels with 3 channels laid out channel-major in
  ``sensordata``: channel 0 is the normal force (>= 0), channels 1 and 2 the two
  tangential forces, all in the sensor site's frame (site z = outward normal).
* ``ContactMonitor`` reads MuJoCo's contact list once per control step and turns
  it into the inputs of ``ContactSound`` (new contacts, slides), the privileged
  contact table, and the templated-text slots.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from sensorium.audio import ContactSound

VOICES = list(ContactSound.VOICES)
MIN_CONTACT_N = 0.05  # contacts weaker than this make no sound and no text
SLIDE_DEADBAND = 0.02  # m/s: slower tangential creep is static friction, not a slide
MAX_CONTACT_ROWS = 16  # strongest contacts kept in the privileged table
TOUCH_THRESHOLD_N = 0.1  # a taxel counts as "touched" above this normal force


def taxel_prefix(sensor_name: str) -> str:
    """Stable name prefix of one skin patch.

    E.g. ``"lh_ffdistal_touch"`` -> ``"hand/left/ffdistal"``.

    Body patches keep H1's link names (``body/left_hip_pitch`` is the thigh link,
    ``body/left_knee`` the shank, ``body/left_ankle`` the foot).
    """
    name = sensor_name.removesuffix("_touch")
    for prefix, kind, side in (
        ("lh_", "hand", "left"),
        ("rh_", "hand", "right"),
        ("lp_", "pincer", "left"),
        ("rp_", "pincer", "right"),
    ):
        if name.startswith(prefix):
            return f"{kind}/{side}/{name.removeprefix(prefix)}"
    return f"body/{name}"


@dataclass
class SkinLayout:
    """Where every taxel's channels live in ``sensordata``.

    Attributes:
        names: One name per taxel, ``"<patch>/<k>"`` with ``k`` the row-major
            index in the patch's grid (e.g. ``"body/left_hip_pitch/3"``).
        bodies: ``[C]`` MuJoCo body carrying each taxel's patch.
        normal, tangent1, tangent2: ``[C]`` indices into ``sensordata``.
    """

    names: list[str]
    bodies: np.ndarray
    normal: np.ndarray
    tangent1: np.ndarray
    tangent2: np.ndarray

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> "SkinLayout":
        """Collect every plugin sensor, in model order (the scenes only use ``touch_grid``).

        Raises:
            ValueError: if the model has no touch_grid sensor.
        """
        names: list[str] = []
        bodies: list[int] = []
        channels: list[list[int]] = [[], [], []]
        for sensor in range(model.nsensor):
            if model.sensor_type[sensor] != mujoco.mjtSensor.mjSENS_PLUGIN:
                continue
            adr, taxels = int(model.sensor_adr[sensor]), int(model.sensor_dim[sensor]) // 3
            prefix = taxel_prefix(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor))
            names += [f"{prefix}/{k}" for k in range(taxels)]
            bodies += [int(model.site_bodyid[model.sensor_objid[sensor]])] * taxels
            for channel in range(3):
                channels[channel] += list(
                    range(adr + channel * taxels, adr + (channel + 1) * taxels)
                )
        if not names:
            raise ValueError("model has no plugin (touch_grid) sensors")
        normal, tangent1, tangent2 = (np.array(c) for c in channels)
        return cls(names, np.array(bodies), normal, tangent1, tangent2)

    def normal_force(self, sensordata: np.ndarray) -> np.ndarray:
        """``[C]`` float32 normal force per taxel (never negative)."""
        return np.asarray(np.maximum(sensordata[self.normal], 0.0), dtype=np.float32)

    def force_xyz(self, sensordata: np.ndarray) -> np.ndarray:
        """``[C, 3]`` float32 taxel forces in HumanoidBench's x, y, z order (z = normal)."""
        return np.stack(
            [sensordata[self.tangent1], sensordata[self.tangent2], sensordata[self.normal]],
            axis=-1,
        ).astype(np.float32)


def body_voice(name: str) -> tuple[str, str]:
    """``(voice, label)`` of a robot body from its name: ``("hands", "left hand")``.

    The label is what the templated text says. Bodies not matched by a keyword
    count as torso.
    """
    side = ""
    if name.startswith(("left", "lh_", "lp_")):
        side = "left "
    elif name.startswith(("right", "rh_", "rp_")):
        side = "right "
    if name.startswith(("lh_", "rh_", "lp_", "rp_")) or name.endswith("-hand"):
        return "hands", f"{side}hand"
    if "ankle" in name:
        return "feet", f"{side}foot"
    if "hip" in name or "knee" in name:
        return "legs", f"{side}leg"
    if "shoulder" in name or "elbow" in name:
        return "arms", f"{side}arm"
    return "torso", "torso"


def surface_voice(geom_name: str) -> tuple[str, str]:
    """``(voice, label)`` of a scene geom: ``"obj_metal_box"`` -> ``("metal", "metal object")``."""
    if geom_name.startswith("obj_"):
        material = geom_name.split("_")[1]
        return material, f"{material} object"
    if geom_name.startswith("landmark_"):
        material = geom_name.split("_")[1]
        return material, f"{material} block"
    if geom_name.startswith("table"):
        return "table", "table"
    if geom_name.startswith("wall"):
        return "floor", "wall"  # sounds like the floor: the same hard, heavy surface
    return "floor", "floor"


@dataclass
class ContactFrame:
    """One control step's contacts.

    Attributes:
        impact, slide: ``[len(VOICES)]`` inputs of ``ContactSound.step``.
        impact_pairs: ``[len(VOICES), len(VOICES)]`` strength (N) of the *new* contacts between a
            robot part (first index, its voice) and a scene surface (second index): which pairing
            of voices each impact excites.
        table: ``[MAX_CONTACT_ROWS, 6]`` strongest contacts: body1, body2, normal
            force (N), x, y, z; unused rows are ``-1`` / 0.
        touches: sorted distinct ``(label, surface)`` per robot-scene contact
            (feet on the floor are included; callers filter).
        object_contact: a robot body touches an ``obj_*`` geom.
        table_contact: a robot body touches the table.
    """

    impact: np.ndarray
    slide: np.ndarray
    table: np.ndarray
    touches: list[tuple[str, str]]
    object_contact: bool
    table_contact: bool
    impact_pairs: np.ndarray


class ContactMonitor:
    """Reads the contact list of a model and remembers the previous step's pairs.

    The many mesh contacts between two bodies are summed into one *pair* (two
    bodies, plus which scene surface when a body touches the static world), so a
    foot resting on the floor is one persistent pair, not a flicker of mesh
    contacts. A pair that was absent at the previous step is a *new contact*.
    Each pair's two sides are classified by voice: robot bodies by
    ``body_voice``, scene geoms by ``surface_voice``. Sliding speed is the
    tangential relative velocity at the contact point, force-weighted over the
    pair's contacts.
    """

    def __init__(self, model: mujoco.MjModel, robot_bodies: set[int]) -> None:
        self.model = model
        self.previous: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        voice_ids = {name: i for i, name in enumerate(VOICES)}
        self.geom_voice: list[int] = []  # per geom: voice id
        self.geom_label: list[str] = []  # per geom: what the text calls it
        self.is_robot: list[bool] = []
        self.is_object: list[bool] = []
        self.is_table: list[bool] = []
        for geom in range(model.ngeom):
            body = int(model.geom_bodyid[geom])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
            robot = body in robot_bodies
            if robot:
                voice, label = body_voice(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body))
            else:
                voice, label = surface_voice(name)
            self.geom_voice.append(voice_ids[voice])
            self.geom_label.append(label)
            self.is_robot.append(robot)
            self.is_object.append(name.startswith("obj_"))
            self.is_table.append(name.startswith("table"))

    def reset(self, data: mujoco.MjData) -> None:
        """Forget history; the contacts present now count as old (nothing is 'new')."""
        self.previous = set(self.pairs(data))

    def key(self, geom1: int, geom2: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """Order-independent key of a pair: ``((body, surface voice or -1), ...)``."""
        sides = []
        for geom in (geom1, geom2):
            body = int(self.model.geom_bodyid[geom])
            sides.append((body, -1 if self.is_robot[geom] or body != 0 else self.geom_voice[geom]))
        return (min(sides), max(sides))

    def pairs(self, data: mujoco.MjData) -> dict[tuple[tuple[int, int], tuple[int, int]], int]:
        """Pair key -> index of its first contact, for every active contact."""
        found: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
        for i in range(data.ncon):
            contact = data.contact[i]
            found.setdefault(self.key(int(contact.geom1), int(contact.geom2)), i)
        return found

    def point_velocity(self, data: mujoco.MjData, body: int, point: np.ndarray) -> np.ndarray:
        """World-frame velocity of ``point`` rigidly attached to ``body``."""
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
        return np.asarray(velocity[3:] + np.cross(velocity[:3], point - data.xipos[body]))

    def read(self, data: mujoco.MjData) -> ContactFrame:
        """Aggregate the current contacts into pairs; update the 'previous' set."""
        model = self.model
        force = np.zeros(6)
        # pair -> [sum normal force, sum normal * sliding speed, first contact index]
        acc: dict[tuple[tuple[int, int], tuple[int, int]], list[float]] = {}
        rows: list[tuple[float, int, int, np.ndarray]] = []
        for i in range(data.ncon):
            contact = data.contact[i]
            mujoco.mj_contactForce(model, data, i, force)
            normal = float(force[0])
            if normal < MIN_CONTACT_N:
                continue
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            rows.append((normal, b1, b2, np.asarray(contact.pos)))
            relative = self.point_velocity(data, b1, contact.pos) - self.point_velocity(
                data, b2, contact.pos
            )
            n = np.asarray(contact.frame[:3])
            speed = float(np.linalg.norm(relative - np.dot(relative, n) * n))
            entry = acc.setdefault(self.key(g1, g2), [0.0, 0.0, float(i)])
            entry[0] += normal
            entry[1] += normal * speed
        impact, slide = np.zeros(len(VOICES)), np.zeros(len(VOICES))
        impact_pairs = np.zeros((len(VOICES), len(VOICES)))
        touches: list[tuple[float, str, str]] = []
        object_contact = table_contact = False
        for key, (normal, weighted_speed, first) in acc.items():
            contact = data.contact[int(first)]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            voices = (self.geom_voice[g1], self.geom_voice[g2])
            if key not in self.previous:
                impact[list(voices)] += normal
            sliding = max(weighted_speed / normal - SLIDE_DEADBAND, 0.0)
            slide[list(voices)] += sliding * np.sqrt(normal)
            if self.is_robot[g1] != self.is_robot[g2]:
                robot, other = (g1, g2) if self.is_robot[g1] else (g2, g1)
                if key not in self.previous:
                    impact_pairs[self.geom_voice[robot], self.geom_voice[other]] += normal
                touches.append((normal, self.geom_label[robot], self.geom_label[other]))
                object_contact |= self.is_object[other]
                table_contact |= self.is_table[other]
        self.previous = set(self.pairs(data))
        rows.sort(key=lambda row: -row[0])
        table = np.zeros((MAX_CONTACT_ROWS, 6), dtype=np.float32)
        table[:, :2] = -1
        for r, (normal, b1, b2, pos) in enumerate(rows[:MAX_CONTACT_ROWS]):
            table[r] = [b1, b2, normal, *pos]
        pairs_touched = sorted({(part, surface) for _, part, surface in touches})
        return ContactFrame(
            impact, slide, table, pairs_touched, object_contact, table_contact, impact_pairs
        )
