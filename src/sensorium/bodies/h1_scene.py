"""Scene assembly for the Unitree H1 humanoid (HumanoidBench ``h1touch`` + an end effector).

We do not depend on the ``humanoid-bench`` Python package (it pins gymnasium and
dm_control versions). Instead ``make h1-assets`` fetches its MJCF assets at a
pinned commit into ``data/external/humanoid-bench/``, and MuJoCo Menagerie's
``robotiq_2f85`` at a pinned commit into ``data/external/mujoco_menagerie/``.
This module turns those files into one scene XML per ``(effector, posture)``:

* **body**: HumanoidBench's ``robots/h1touch_pos.xml``: H1 with position
  actuators and 19 ``touch_grid`` skin patches (``h1/touch_sensors_body.xml``);
* **effector** ``"hand"``: the ``h1touch`` Shadow hands as shipped, with their 34
  skin patches (``shadow_hand_submodel/touch_sensors.xml``);
  ``"pincer"``: the Shadow hands are removed and a Robotiq 2F-85 is attached at
  each wrist (``<attach>``, prefixes ``lp_`` / ``rp_``), its pads' contact faces
  rebuilt from small cubes (the pads' skin is ``h1_skin.DenseSkin``);
* **extras**: a pelvis IMU site with a gyroscope and an accelerometer, head
  cameras tilted down so the hands and the table are in view, a textured room
  (floor, four walls, one key light; ``h1_looks`` recolours it per episode), and
  either a few static landmarks (``"standing"``) or a seat-welded pelvis in front
  of a table with free objects of four materials (``"seated"``).

Generated files are written next to HumanoidBench's own scene files, in
``<humanoid_bench>/assets/sensorium/``, because the assets reference each other
with paths relative to a sibling directory (``../h1/assets/...``). They are
rewritten identically every time (atomically), so concurrent writers are harmless.
"""

import math
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from sensorium.bodies.h1_props import STACK_OBJECTS, TALL_OBJECTS, prop_xml

EFFECTORS = ("hand", "pincer")
POSTURES = ("standing", "seated", "stand_supported")
TABLE_POSTURES = ("seated", "stand_supported")  # the postures with a table and objects

# `make h1-assets` pins these; see the Makefile. Revisions actually used are
# read back from the checkouts (see ``asset_revisions``) and stored in `meta`.
HUMANOID_BENCH_DIR = "humanoid-bench"
MENAGERIE_DIR = "mujoco_menagerie"

# Head cameras: HumanoidBench's eyes look horizontally with MuJoCo's default 45
# degree field of view, which sees mostly horizon. We tilt them down and widen
# them so the hands and the table are in view. The H1 has no neck, so one fixed
# tilt serves both postures: seated, the table fills the frame; standing, the
# landmarks ahead stay in view.
CAMERA_PITCH_DEG = 45.0
CAMERA_FOVY_DEG = 80.0

# The room: a floor and four walls around the robot, so the cameras see a
# place rather than a grey void. Textures are near-white greys; their colour
# comes from the material tint, which ``h1_looks`` redraws every episode.
ROOM_HALF = 3.0  # m, half the room's side; the robot stands at the centre
WALL_HEIGHT = 2.6
ROOM_MATERIALS = ("floor", "wall", "table_wood", "seat")
# MuJoCo scales the cameras' near clip (1% of it) and directional shadows (a box
# of +- shadowclip times it) by the model's extent, which the room would push
# to 12.5 m: a 12.5 cm near clip and coarse, ragged shadows. Pin it instead:
# a 4 cm near clip, and shadows sharp over +- 2 m around the robot.
SCENE_EXTENT = 4.0
SHADOW_CLIP = 0.5

# Seated pose (hip pitch -pi/2, knee +pi/2, ankle 0): thighs horizontal, shins
# vertical. The pelvis hangs 0.1742 m above the hip joints, a thigh and a shin
# are 0.4 m each and the ankle sits 0.07 m above the sole.
SEATED_PELVIS_Z = 0.64
SUPPORTED_PELVIS_Z = 0.98  # standing height: the pelvis of ``stand_supported`` is welded here
TABLE_TOP_Z = 0.82  # default table height (m); ``table_top`` in the settings overrides it
TABLE_X_BASE_Z = (
    0.72  # the seated table height at which TABLE_X holds (a higher table's edge moves out)
)
TABLE_X = (
    0.24,
    0.76,
)  # the table's near and far edge in front of the pelvis (m), at TABLE_X_BASE_Z
OBJECT_HALF_WIDTH = 0.30  # objects are placed within |y| <= this (the camera sees no further)
REACH_X = (0.40, 0.62)  # where objects are placed and hands aim: within the arms' reach (was 0.68)

PINCER_PREFIX = {"left": "lp_", "right": "rp_"}
PINCER_FILE = "2f85_cube_pads.xml"  # derived from Menagerie's 2f85.xml, written beside it
# Menagerie's pad: two boxes, together 22 x 8 x 37.5 mm, centred at y = -2.6 mm in
# the pad frame; -y points at the other pad. Its contact face becomes cubes.
PAD_CENTRE_Y = -0.0026
PAD_HALF_WIDTH, PAD_HALF_THICKNESS, PAD_HALF_HEIGHT = 0.011, 0.004, 0.01875
PAD_COLUMNS, PAD_ROWS = 5, 8
PAD_CUBE_THICKNESS = 0.001
PAD_CROWN = 0.0005  # extra cube depth at the pad centre (m), falling to 0 at the edges

PINCER_BODIES = ("base_mount", "base") + tuple(
    f"{side}_{part}"
    for side in ("left", "right")
    for part in ("driver", "coupler", "spring_link", "follower", "pad")
)
MATERIALS = {
    # name: (rgba, friction, density kg/m^3, solref) -- plausible, not measured
    "wood": ("0.62 0.42 0.22 1", "0.6 0.005 0.0005", 600, "0.004 1"),
    "plastic": ("0.2 0.45 0.85 1", "0.4 0.005 0.0005", 400, "0.004 1"),
    "metal": ("0.62 0.64 0.68 1", "0.3 0.005 0.0005", 7800, "0.002 1"),
    "rubber": ("0.85 0.25 0.2 1", "1.2 0.005 0.0005", 1100, "0.01 1"),
}
# (body name, material, geom type, size). Each is a free body on the table.
OBJECTS = (
    ("obj_wood_box", "wood", "box", "0.035 0.035 0.035"),
    ("obj_plastic_sphere", "plastic", "sphere", "0.035"),
    ("obj_metal_cylinder", "metal", "cylinder", "0.025 0.025"),
    ("obj_rubber_sphere", "rubber", "sphere", "0.03"),
    ("obj_plastic_box", "plastic", "box", "0.05 0.03 0.02"),
    ("obj_metal_box", "metal", "box", "0.025 0.025 0.03"),
)


def external_dir() -> Path:
    """Directory holding fetched assets: ``$SENSORIUM_EXTERNAL`` or ``data/external``."""
    return Path(os.environ.get("SENSORIUM_EXTERNAL", "data/external"))


def asset_paths(external: Path | None = None) -> tuple[Path, Path]:
    """``(HumanoidBench assets dir, Robotiq 2F-85 dir)``, as absolute paths.

    Raises:
        FileNotFoundError: if either is missing; the message says to run
            ``make h1-assets``.
    """
    external = (external or external_dir()).resolve()
    humanoid = external / HUMANOID_BENCH_DIR / "humanoid_bench" / "assets"
    robotiq = external / MENAGERIE_DIR / "robotiq_2f85"
    for path in (humanoid / "robots" / "h1touch_pos.xml", robotiq / "2f85.xml"):
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}; run `make h1-assets` first")
    return humanoid, robotiq


def assets_available() -> bool:
    """True if ``make h1-assets`` has been run (so scenes can be built)."""
    try:
        asset_paths()
    except FileNotFoundError:
        return False
    return True


def asset_revisions() -> dict[str, str]:
    """Git commit of each fetched asset checkout (``"unknown"`` if not a git checkout)."""
    external = external_dir().resolve()
    revisions = {}
    for name in (HUMANOID_BENCH_DIR, MENAGERIE_DIR):
        result = subprocess.run(
            ["git", "-C", str(external / name), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        revisions[name] = result.stdout.strip() or "unknown"
    return revisions


def default_table_top(posture: str) -> float:
    """Table height of a posture: the seated table, raised with the pelvis when standing.

    ``stand_supported`` keeps the arms' geometry relative to the table, so its table is
    ``TABLE_TOP_Z`` plus the height the pelvis has gained.
    """
    if posture == "stand_supported":
        return TABLE_TOP_Z + SUPPORTED_PELVIS_Z - SEATED_PELVIS_Z
    return TABLE_TOP_Z


def pelvis_height(posture: str) -> float:
    """Height (m) the pelvis is welded to in a table posture."""
    return SUPPORTED_PELVIS_Z if posture == "stand_supported" else SEATED_PELVIS_Z


def build_scene(
    effector: str,
    posture: str,
    external: Path | None = None,
    table_top: float | None = None,
    props: tuple[str, ...] = (),
) -> Path:
    """Write the scene XML for ``(effector, posture)`` and return its path.

    ``table_top`` (m) is the height of the seated table's surface; other heights get
    their own file, so scenes of different heights can be built side by side. ``props`` (names
    from ``h1_props.PROPS``) add experiment props to a table posture, in a file of their own.

    Raises:
        ValueError: for an unknown effector or posture.
        FileNotFoundError: if the assets have not been fetched.
    """
    if effector not in EFFECTORS:
        raise ValueError(f"effector must be one of {EFFECTORS}, got {effector!r}")
    if posture not in POSTURES:
        raise ValueError(f"posture must be one of {POSTURES}, got {posture!r}")
    humanoid, robotiq = asset_paths(external)
    out = humanoid / "sensorium"
    out.mkdir(exist_ok=True)

    robot_file = out / f"h1_{effector}.xml"
    robot = ET.parse(humanoid / "robots" / "h1touch_pos.xml")
    add_pelvis_imu(robot.getroot())
    tilt_head_cameras(robot.getroot())
    if effector == "pincer":
        robot_root = robot.getroot()
        remove_shadow_hands(robot_root)
        attach_pincers(robot_root, robotiq / PINCER_FILE)
        write_xml(robot_root, robot_file)
        write_xml(pincer_with_cube_pads(robotiq), robotiq / PINCER_FILE)
    else:
        write_xml(robot.getroot(), robot_file)

    default = default_table_top(posture)
    top = default if table_top is None else table_top
    suffix = "" if top == default or posture not in TABLE_POSTURES else f"_{round(top * 1000)}"
    props = tuple(sorted(props)) if posture in TABLE_POSTURES else ()
    suffix += "".join(f"_{p}" for p in props)
    scene_file = out / f"scene_{effector}_{posture}{suffix}.xml"
    write_atomic(scene_file, scene_xml(robot_file.name, effector, posture, top, props))
    return scene_file


def write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so a concurrent reader never sees a partial file."""
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def write_xml(root: ET.Element, path: Path) -> None:
    """Pretty-print ``root`` to ``path`` (same bytes every time)."""
    ET.indent(root)
    write_atomic(path, ET.tostring(root, encoding="unicode"))


def add_pelvis_imu(root: ET.Element) -> None:
    """Add a ``pelvis_imu`` site to the pelvis (the vestibular sense is read there)."""
    pelvis = find_body(root, "pelvis")
    ET.SubElement(pelvis, "site", name="pelvis_imu", pos="0 0 0", size="0.01", group="4")


def tilt_head_cameras(root: ET.Element) -> None:
    """Pitch both eye cameras down by ``CAMERA_PITCH_DEG`` and set their field of view."""
    pitch = math.radians(CAMERA_PITCH_DEG)
    # Camera x axis stays "to the left" (-y); the up axis tilts to follow the gaze.
    xyaxes = f"0 -1 0 {math.sin(pitch):.5f} 0 {math.cos(pitch):.5f}"
    for camera in root.iter("camera"):
        if camera.get("name") in ("left_eye_camera", "right_eye_camera"):
            camera.set("xyaxes", xyaxes)
            camera.set("fovy", str(CAMERA_FOVY_DEG))


def find_body(root: ET.Element, name: str) -> ET.Element:
    """The ``<body>`` called ``name``.

    Raises:
        KeyError: if there is none.
    """
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    raise KeyError(f"no body {name!r} in the robot XML")


def remove_shadow_hands(root: ET.Element) -> None:
    """Drop everything that belongs to the Shadow hands from the ``h1touch`` robot XML.

    Removes the hand sub-models (also the ones inside the wrist bodies), their
    meshes, tendons, actuators, contact excludes and skin patches, and keeps the
    21 H1 actuators (wrist yaw included).
    """
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag == "include" and "shadow_hand_submodel" in (child.get("file") or ""):
                parent.remove(child)
            elif child.tag == "mesh" and "shadow_hand_submodel" in (child.get("file") or ""):
                parent.remove(child)
    for child in list(root):
        names = [grandchild.get("name") or grandchild.get("body1") or "" for grandchild in child]
        is_hand = bool(names) and all(name.startswith(("lh_", "rh_")) for name in names)
        if child.tag in ("tendon", "actuator", "contact") and is_hand:
            root.remove(child)


def attach_pincers(root: ET.Element, pincer_file: Path) -> None:
    """Attach the Robotiq 2F-85 model in ``pincer_file`` to each wrist body, prefixed per side."""
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "model", name="robotiq", file=str(pincer_file))
    contact = ET.SubElement(root, "contact")
    for side, body_name in (("left", "left-hand"), ("right", "right-hand")):
        prefix = PINCER_PREFIX[side]
        ET.SubElement(
            find_body(root, body_name), "attach", model="robotiq", body="base_mount", prefix=prefix
        )
        # The gripper's rear linkage overlaps the elbow link's collision mesh, and
        # the pair is more than one joint apart, so MuJoCo would not skip it.
        for part in PINCER_BODIES:
            ET.SubElement(contact, "exclude", body1=f"{side}_elbow_link", body2=f"{prefix}{part}")


def pincer_with_cube_pads(robotiq: Path) -> ET.Element:
    """Menagerie's 2F-85 with each pad's contact face made of small cubes.

    MuJoCo resolves a pad pressing on an object into one or two contact
    points, which would light a single skin point. Following Han, Joshi and
    Deng 2025 (arXiv:2505.02915), the inner 1 mm of each pad becomes a
    ``PAD_COLUMNS`` x ``PAD_ROWS`` grid of cubes (about 4.4 x 4.7 mm), each its
    own contact, with a gentle crown (``PAD_CROWN`` at the centre) so the contact
    patch grows with force. The two pad boxes are recessed by the cube
    thickness, so the pad keeps its shape. The cubes inherit the pad boxes'
    friction, softness and collision group (hidden from the cameras). Written
    next to ``2f85.xml`` (meshes resolve through its ``meshdir``).
    """
    root = ET.parse(robotiq / "2f85.xml").getroot()
    for default in root.iter("default"):
        if default.get("class") in ("pad_box1", "pad_box2"):
            geom = default.find("geom")
            assert geom is not None
            x, _, z = geom.get("pos", "").split()
            sx, _, sz = geom.get("size", "").split()
            half = PAD_HALF_THICKNESS - PAD_CUBE_THICKNESS / 2
            geom.set("pos", f"{x} {PAD_CENTRE_Y + PAD_CUBE_THICKNESS / 2:.6f} {z}")
            geom.set("size", f"{sx} {half:.6f} {sz}")
    for side in ("left", "right"):
        pad = find_body(root, f"{side}_pad")
        for index, cube in enumerate(pad_cubes()):
            ET.SubElement(pad, "geom", {"name": f"{side}_pad_cube{index}", **cube})
    return root


def pad_cubes() -> list[dict[str, str]]:
    """Geom attributes of the cubes on one pad's contact face (pad frame, -y faces the object)."""
    width, height = 2 * PAD_HALF_WIDTH, 2 * PAD_HALF_HEIGHT
    hx, hz = width / (2 * PAD_COLUMNS), height / (2 * PAD_ROWS)
    back = PAD_CENTRE_Y - PAD_HALF_THICKNESS + PAD_CUBE_THICKNESS  # the recessed pad face
    cubes = []
    for row in range(PAD_ROWS):
        z = hz * (2 * row + 1)
        for column in range(PAD_COLUMNS):
            x = -PAD_HALF_WIDTH + hx * (2 * column + 1)
            u, v = x / PAD_HALF_WIDTH, (z - PAD_HALF_HEIGHT) / PAD_HALF_HEIGHT
            hy = (PAD_CUBE_THICKNESS + PAD_CROWN * (1 - u**2) * (1 - v**2)) / 2
            cubes.append(
                {
                    "class": "pad_box1" if z > PAD_HALF_HEIGHT else "pad_box2",
                    "pos": f"{x:.6f} {back - hy:.6f} {z:.6f}",
                    "size": f"{hx:.6f} {hy:.6f} {hz:.6f}",
                }
            )
    return cubes


def scene_xml(
    robot_file: str,
    effector: str,
    posture: str,
    table_top: float | None = None,
    props: tuple[str, ...] = (),
) -> str:
    """The full scene: options, room, robot, posture-specific props, IMU sensors.

    Both effectors use Menagerie's recommended elliptic friction cones with ``impratio`` 10 (they
    keep the gripper linkage and its grasps stable). The hand scene had HumanoidBench's pyramidal
    default until the hand grasp work: with it, spheres rolled out of a closed pinch.
    """
    cone = ' cone="elliptic" impratio="10"'
    materials = "\n".join(
        f'    <material name="{name}" rgba="{rgba}"/>' for name, (rgba, *_) in MATERIALS.items()
    )
    top = default_table_top(posture) if table_top is None else table_top
    body = seated_props(top, posture, props) if posture in TABLE_POSTURES else standing_props()
    weld = (
        '  <equality>\n    <weld body1="seat_anchor" body2="pelvis" relpose="0 0 0 1 0 0 0"/>\n'
        "  </equality>\n"
        if posture in TABLE_POSTURES
        else ""
    )
    return f"""<mujoco model="sensorium_h1_{posture}">
  <option timestep="0.002" iterations="100" ls_iterations="50" solver="Newton"{cone}>
    <flag eulerdamp="enable"/>
  </option>
  <statistic extent="{SCENE_EXTENT}" center="0.4 0 0.8"/>
  <visual>
    <headlight diffuse="0.45 0.45 0.45" ambient="0.25 0.25 0.25" specular="0 0 0"/>
    <map shadowclip="{SHADOW_CLIP}"/>
    <quality shadowsize="2048"/>
  </visual>
  <include file="{robot_file}"/>
  <asset>
{materials}
{room_assets()}
  </asset>
  <worldbody>
{room_body()}{body}  </worldbody>
{weld}  <sensor>
    <gyro name="pelvis_gyro" site="pelvis_imu"/>
    <accelerometer name="pelvis_acc" site="pelvis_imu"/>
  </sensor>
</mujoco>
"""


def room_assets() -> str:
    """Sky, textures and the materials of the room, the table and the seat.

    Textures are speckled near-white greys (MuJoCo's built-in ``checker`` and
    ``flat`` with random marks), so surfaces have grain the cameras can follow
    and take their colour from the material, which ``h1_looks`` retints.
    """
    return """\
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.55 0.65 0.8" rgb2="0.9 0.92 0.95"
      width="256" height="256"/>
    <texture name="floor" type="2d" builtin="checker" rgb1="0.95 0.95 0.95" rgb2="0.78 0.78 0.78"
      mark="random" random="0.08" markrgb="0.6 0.6 0.6" width="512" height="512"/>
    <texture name="wall" type="2d" builtin="flat" rgb1="0.95 0.95 0.95"
      mark="random" random="0.05" markrgb="0.75 0.75 0.75" width="256" height="256"/>
    <texture name="grain" type="2d" builtin="flat" rgb1="0.95 0.95 0.95"
      mark="random" random="0.15" markrgb="0.7 0.7 0.7" width="256" height="256"/>
    <material name="floor" texture="floor" texrepeat="6 6" texuniform="false" reflectance="0.05"
      rgba="0.6 0.55 0.5 1"/>
    <material name="wall" texture="wall" texrepeat="3 1" rgba="0.85 0.85 0.82 1"/>
    <material name="table_wood" texture="grain" texrepeat="4 4" rgba="0.55 0.38 0.2 1"/>
    <material name="seat" texture="grain" rgba="0.25 0.25 0.3 1"/>"""


def room_body() -> str:
    """A key light, the floor (``floor``) and four walls (``wall_*``), all static."""
    half, height = ROOM_HALF, WALL_HEIGHT
    lines = [
        '    <light name="key" pos="0 0 4" dir="0 0 -1" directional="true" castshadow="true" '
        'diffuse="0.5 0.5 0.5" specular="0.2 0.2 0.2"/>\n',
        f'    <geom name="floor" type="plane" size="{half} {half} 0.1" material="floor"/>\n',
    ]
    for i, (x, y, half_x, half_y) in enumerate(
        [
            (half, 0, 0.05, half),
            (-half, 0, 0.05, half),
            (0, half, half, 0.05),
            (0, -half, half, 0.05),
        ]
    ):
        lines.append(
            f'    <geom name="wall{i}" type="box" pos="{x} {y} {height / 2}" '
            f'size="{half_x} {half_y} {height / 2}" material="wall"/>\n'
        )
    return "".join(lines)


def standing_props() -> str:
    """A few static boxes ahead of the robot, so the head camera has something to see."""
    boxes = [(1.6, 0.0, 0.25, "wood"), (2.2, -0.9, 0.4, "metal"), (1.9, 1.0, 0.15, "plastic")]
    return "".join(
        f'    <geom name="landmark_{m}_{i}" type="box" pos="{x} {y} {h}" size="0.2 0.2 {h}" '
        f'material="{m}"/>\n'
        for i, (x, y, h, m) in enumerate(boxes)
    )


def edge_base(posture: str) -> float:
    """The table height at which ``TABLE_X`` holds in a posture (``TABLE_X_BASE_Z`` seated)."""
    return TABLE_X_BASE_Z + pelvis_height(posture) - SEATED_PELVIS_Z


def table_edges(table_top: float, base: float = TABLE_X_BASE_Z) -> tuple[float, float]:
    """Near and far edge (m in front of the pelvis) of a table with its surface at ``table_top``.

    A higher table is at elbow height, so its near edge moves out by the height
    gained (1 cm per cm above ``base``, see ``edge_base``) to leave room for the upper arms.
    """
    near, far = TABLE_X
    return near + max(table_top - base, 0.0), far


def seated_props(
    table_top: float = TABLE_TOP_Z, posture: str = "seated", props: tuple[str, ...] = ()
) -> str:
    """Seat (seated only, visual), table, a mocap anchor for the pelvis weld, and free objects."""
    top = table_top
    near, far = table_edges(table_top, edge_base(posture))
    pelvis_z = pelvis_height(posture)
    cx, hx = (near + far) / 2, (far - near) / 2
    lines = [f'    <body name="seat_anchor" mocap="true" pos="0 0 {pelvis_z}"/>\n']
    if posture == "seated":
        lines.append(
            f'    <geom name="seat" type="box" pos="-0.05 0 {pelvis_z - 0.17}" '
            'size="0.22 0.2 0.02" material="seat" contype="0" conaffinity="0"/>\n'
        )
    lines += [
        f'    <geom name="table_top" type="box" pos="{cx} 0 {top - 0.02}" size="{hx} 0.5 0.02" '
        'material="table_wood" friction="0.6 0.005 0.0005"/>\n',
    ]
    rim = 0.025  # a low lip keeps rolling objects on the table
    for i, (x, y, half_x, half_y) in enumerate(
        [(near, 0.0, 0.01, 0.5), (far, 0.0, 0.01, 0.5), (cx, -0.5, hx, 0.01), (cx, 0.5, hx, 0.01)]
    ):
        lines.append(
            f'    <geom name="table_rim{i}" type="box" pos="{x} {y} {top + rim / 2}" '
            f'size="{half_x} {half_y} {rim / 2}" material="table_wood"/>\n'
        )
    for i, (x, y) in enumerate(
        [(near + 0.03, -0.45), (near + 0.03, 0.45), (far - 0.03, -0.45), (far - 0.03, 0.45)]
    ):
        lines.append(
            f'    <geom name="table_leg{i}" type="box" pos="{x} {y} {(top - 0.04) / 2}" '
            f'size="0.02 0.02 {(top - 0.04) / 2}" material="table_wood"/>\n'
        )
    extra = TALL_OBJECTS if "tall_block" in props else ()
    base = STACK_OBJECTS if "stack_blocks" in props else OBJECTS  # two cubes instead of the six
    for name, material, shape, size in (*base, *extra):
        rgba, friction, density, solref = MATERIALS[material]
        lines.append(
            f'    <body name="{name}" pos="0.55 0 {top + 0.05}">\n'
            f'      <freejoint name="{name}"/>\n'
            f'      <geom name="{name}" type="{shape}" size="{size}" material="{material}" '
            f'friction="{friction}" density="{density}" solref="{solref}"/>\n    </body>\n'
        )
    lines.append(prop_xml(props, top))
    return "".join(lines)
