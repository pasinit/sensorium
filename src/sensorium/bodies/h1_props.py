"""Props for the experiments with consequences (docs/experiments.md), and hidden properties.

A scene gets any subset of ``PROPS`` through the recorder's ``props`` setting. Free bodies (the
tall block and bottle) are ``obj_*`` bodies like the six table objects, so every object statistic
sees them. Articulated props have joints named ``prop_*`` whose travel (0 closed or up, 1 open or
down) is logged as ``privileged/prop_state``; their geoms are named ``table_*`` so that contacts
with them count as table contacts and sound like the table. The target disc is visual only.

Hidden properties (per object and episode): mass, sliding friction and contact softness, drawn
from the ranges below and recorded in ``privileged/hidden_properties`` only. The look of an object
does not depend on them.
"""

import numpy as np

PROPS = (
    "tall_block",
    "friction_patches",
    "button",
    "hinged_flap",
    "drawer",
    "target_disc",
    "stack_blocks",
)

# The two blocks of ``stack_blocks`` (4 cm cubes), which replace the six standard objects.
STACK_OBJECTS = (
    ("obj_wood_stackbox_a", "wood", "box", "0.02 0.02 0.02"),
    ("obj_plastic_stackbox_b", "plastic", "box", "0.02 0.02 0.02"),
)
STACK_X = (0.42, 0.52)  # both blocks go on one arm's side, in this x range and |y| range
STACK_Y = (0.12, 0.26)
STACK_MIN_GAP = 0.10

# (body name, material, geom type, size): free bodies added by ``tall_block``.
TALL_OBJECTS = (
    ("obj_wood_tallbox", "wood", "box", "0.015 0.015 0.075"),  # 3 x 3 x 15 cm
    ("obj_plastic_tallcylinder", "plastic", "cylinder", "0.02 0.075"),  # a bottle: 4 cm, 15 cm
)

# Where each prop stands (m, in front of and to the side of the pelvis; the table's surface is z).
DRAWER_POS = (0.45, 0.30)  # the left drawer's front (the right one mirrors it); it opens towards
DRAWER_TRAVEL = 0.08  # the robot (-x) by this much: its knob stays within a pincer's reach
FLAP_POS = (0.50, 0.34)  # the left hinge; the right one mirrors it. The flap extends towards the
FLAP_LENGTH = (
    0.06  # m from the hinge to the free end: it stays clear of an open pincer at its start
)
FLAP_RANGE = 1.6  # rad  ... middle (-y on the left) and swings away from the robot when pushed
BUTTON_POS = (0.47, 0.12)  # the left button; the right one mirrors it
BUTTON_TRAVEL = 0.008
TARGET_X = (0.50, 0.56)  # the target disc is drawn in this x range (a pincer's reach edge) and |y|
TARGET_Y = (0.12, 0.17)  # in this one, so that an object can start 17 cm or more away on its side
TARGET_RADIUS = 0.04
PATCH_FRICTION = (0.1, 0.5, 1.2)  # sliding friction of the three patches
PATCH_X = (0.435, 0.515, 0.595)  # three strips across x, each 7 cm wide and 60 cm long in y
PATCH_HALF = (0.035, 0.30)

# Axis-aligned zones (x0, x1, y0, y1) in which objects are not placed when the prop is present.
EXCLUSION = {
    "drawer": (0.34, 0.62, 0.20, 0.42),  # and its mirror image
    "hinged_flap": (0.42, 0.62, 0.20, 0.38),  # and its mirror image (see ``in_exclusion``)
    "button": (0.42, 0.53, 0.05, 0.19),
}

# Hidden property ranges, fixed 2026-10-15 (docs/experiments.md; memo section 7).
MASS_KG = (0.1, 1.5)  # log-uniform
FRICTION = (0.2, 1.0)  # uniform, sliding
SOLREF_S = (0.01, 0.06)  # log-uniform, contact time constant
HIDDEN_COLUMNS = ["mass_kg", "friction", "solref_s"]


def draw_hidden(rng: np.random.Generator, count: int) -> np.ndarray:
    """``[count, 3]`` hidden properties (``HIDDEN_COLUMNS``) of ``count`` objects."""
    mass = np.exp(rng.uniform(np.log(MASS_KG[0]), np.log(MASS_KG[1]), count))
    friction = rng.uniform(*FRICTION, count)
    solref = np.exp(rng.uniform(np.log(SOLREF_S[0]), np.log(SOLREF_S[1]), count))
    return np.stack([mass, friction, solref], axis=1)


def in_exclusion(xy: np.ndarray, props: tuple[str, ...]) -> bool:
    """True if ``xy`` lies in the exclusion zone of any prop that is present."""
    for p in props:
        if p not in EXCLUSION:
            continue
        x0, x1, y0, y1 = EXCLUSION[p]
        left = x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1
        mirrored = x0 <= xy[0] <= x1 and y0 <= -xy[1] <= y1
        if left or mirrored:
            return True
    return False


def prop_xml(props: tuple[str, ...], top: float) -> str:
    """MJCF for the articulated and visual props in ``props``, on a table surface at height ``top``.

    Raises:
        ValueError: for a name that is not in ``PROPS``.
    """
    unknown = set(props) - set(PROPS)
    if unknown:
        raise ValueError(f"unknown props {sorted(unknown)}; known: {PROPS}")
    if "drawer" in props and "hinged_flap" in props:
        raise ValueError("the drawer and the hinged flaps overlap: use one or the other")
    parts = []
    if "friction_patches" in props:
        parts.append(patches_xml(top))
    if "drawer" in props:
        parts.append(drawer_xml(top))
    if "hinged_flap" in props:
        parts.append(flap_xml(top))
    if "button" in props:
        parts.append(button_xml(top))
    if "target_disc" in props:
        parts.append(
            f'    <geom name="target_disc" type="cylinder" size="{TARGET_RADIUS} 0.0005" '
            f'pos="0.5 0.18 {top + 0.0006}" rgba="0.95 0.85 0.1 1" contype="0" conaffinity="0"/>\n'
        )
    return "".join(parts)


def patches_xml(top: float) -> str:
    """Three flush strips across x whose friction overrides the table's and the touching geom's."""
    return "".join(
        f'    <geom name="table_patch{i}" type="box" pos="{x} 0 {top - 0.0004}" '
        f'size="{PATCH_HALF[0]} {PATCH_HALF[1]} 0.0005" friction="{mu} 0.005 0.0005" priority="1" '
        f'rgba="0.3 0.6 0.4 1"/>\n'
        for i, (x, mu) in enumerate(zip(PATCH_X, PATCH_FRICTION, strict=True))
    )


def drawer_xml(top: float) -> str:
    """Two cabinets (left and right), each a drawer on a slide joint with a pinchable knob."""
    depth, half_w, high = 0.12, 0.08, 0.05
    parts = []
    for side, sign in (("l", 1.0), ("r", -1.0)):
        x, y = DRAWER_POS[0], sign * DRAWER_POS[1]
        cx = x + depth / 2  # the drawer's centre when closed
        wall_z = top + 0.010 + high / 2
        parts.append(
            f'    <geom name="table_drawer_{side}_base" type="box" pos="{cx} {y} {top + 0.005}" '
            f'size="{depth / 2 + 0.01} {half_w + 0.012} 0.005" material="table_wood"/>\n'
            f'    <geom name="table_drawer_{side}_wall_a" type="box" '
            f'pos="{cx} {y + half_w + 0.007} {wall_z}" size="{depth / 2} 0.005 {high / 2}" '
            'material="table_wood"/>\n'
            f'    <geom name="table_drawer_{side}_wall_b" type="box" '
            f'pos="{cx} {y - half_w - 0.007} {wall_z}" size="{depth / 2} 0.005 {high / 2}" '
            'material="table_wood"/>\n'
            f'    <geom name="table_drawer_{side}_back" type="box" '
            f'pos="{x + depth + 0.005} {y} {wall_z}" size="0.005 {half_w + 0.012} {high / 2}" '
            'material="table_wood"/>\n'
            f'    <body name="prop_drawer_{side}" pos="{cx} {y} {top + 0.012 + high / 2}">\n'
            f'      <joint name="prop_drawer_{side}" type="slide" axis="-1 0 0" '
            f'range="0 {DRAWER_TRAVEL}" damping="3" frictionloss="0.5" solreflimit="0.004 1" '
            'solimplimit="0.95 0.99 0.001"/>\n'
            f'      <geom name="table_drawer_{side}_box" type="box" '
            f'size="{depth / 2 - 0.003} {half_w} {high / 2 - 0.002}" mass="0.3" '
            'material="metal"/>\n'
            f'      <geom name="table_drawer_{side}_knob" type="cylinder" '
            f'pos="{-depth / 2 + 0.02} 0 {high / 2 + 0.018}" size="0.012 0.02" mass="0.02" '
            'material="metal"/>\n'
            "    </body>\n"
        )
    return "".join(parts)


def flap_xml(top: float) -> str:
    """Two hinged flaps (left and right) that swing away when pushed from the robot's side."""
    parts = []
    for side, sign in (("l", 1.0), ("r", -1.0)):
        x, y = FLAP_POS[0], sign * FLAP_POS[1]
        parts.append(
            f'    <geom name="table_flap_post_{side}" type="cylinder" '
            f'pos="{x} {y + sign * 0.008} {top + 0.035}" size="0.006 0.035" material="metal"/>\n'
            f'    <body name="prop_flap_{side}" pos="{x} {y} {top + 0.035}">\n'
            f'      <joint name="prop_flap_{side}" type="hinge" axis="0 0 {sign}" '
            f'range="0 {FLAP_RANGE}" damping="0.02" frictionloss="0.004" solreflimit="0.004 1" '
            'solimplimit="0.95 0.99 0.001"/>\n'
            f'      <geom name="table_flap_{side}" type="box" pos="0 {-sign * FLAP_LENGTH / 2} 0" '
            f'size="0.005 {FLAP_LENGTH / 2} 0.03" mass="0.05" material="table_wood"/>\n'
            "    </body>\n"
        )
    return "".join(parts)


def button_xml(top: float) -> str:
    """Two spring-loaded push buttons (left and right): 8 mm of travel, about 5 N at full press."""
    parts = []
    for side, sign in (("l", 1.0), ("r", -1.0)):
        x, y = BUTTON_POS[0], sign * BUTTON_POS[1]
        parts.append(
            f'    <geom name="table_button_base_{side}" type="cylinder" '
            f'pos="{x} {y} {top + 0.005}" size="0.03 0.005" material="metal" solref="0.004 1"/>\n'
            f'    <body name="prop_button_{side}" pos="{x} {y} {top + 0.018 + BUTTON_TRAVEL}">\n'
            f'      <joint name="prop_button_{side}" type="slide" axis="0 0 -1" '
            f'range="0 {BUTTON_TRAVEL}" stiffness="600" damping="5" solreflimit="0.004 1" '
            'solimplimit="0.95 0.99 0.001"/>\n'
            f'      <geom name="table_button_{side}" type="cylinder" size="0.02 0.008" '
            'mass="0.05" material="plastic" solref="0.004 1"/>\n'
            "    </body>\n"
        )
    return "".join(parts)
