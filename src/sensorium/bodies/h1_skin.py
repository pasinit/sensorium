"""Dense skin and contact dynamics for the H1: the touch post-processing of ``docs/touch.md``.

HumanoidBench's ``touch_grid`` patches bin a body's contacts by direction into a
4 x 2 grid, and MuJoCo reports only a few contact points per touch, so a
touched patch usually lights one taxel of eight. Two additions fix the hands
and the pincer pads (the body keeps its coarse grid):

* ``DenseSkin``: sensor points spread over the collision surface of each
  hand / pad body at a fixed spacing, and every contact's normal force shared
  among the points near it with MIMo's linear spread (weight falls linearly to
  0 at twice the spacing, total force conserved). Only points whose outward
  normal faces the contact receive force, so a press on the finger pad does not
  leak to the nail side.
* ``ContactDynamics``: per skin region, three features computed from every
  physics sub-step between two observations: ``onset`` (a contact started),
  ``slip`` (mean tangential sliding speed under contact, m/s) and
  ``vibration`` (RMS of the sub-step change of normal force, N). Touch is
  observed at 20 Hz, but slip and texture live at hundreds of Hz.

References: MIMo's touch module (``spread_linear``); Han, Joshi and Deng 2025
(arXiv:2505.02915) for the small-cube pads (in ``h1_scene``); Veiga et al.
IROS 2015 and VibeAct 2026 for onset / slip / vibration features.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from sensorium.bodies.h1_sense import taxel_prefix

MIN_CONTACT_N = 0.05  # contacts weaker than this are ignored (same as the contact monitor)
SLIDE_DEADBAND = 0.02  # m/s: slower tangential creep is static friction, not slip
SAMPLING_SEED = 0  # the point layout is a fixed function of the model
CANDIDATES_PER_POINT = 30  # surface samples drawn per accepted point (Poisson-disc)
INSIDE_TOLERANCE = 2e-4  # m: a sample this close to another collision geom is hidden
DYNAMICS_FEATURES = ("onset", "slip", "vibration")
FACING_COS = 0.1  # a point feels a contact if its normal is within ~84 degrees of it

# Sensor spacing (m) per hand part: densest at the fingertips, as in human skin,
# and about 1,000 points per hand in all. Human fingertips have ~1 mm spacing;
# this resolves where on a phalanx a contact is, not texture (that is
# ``vibration``'s job).
HAND_SPACING = {
    "distal": 0.004,
    "middle": 0.006,
    "proximal": 0.006,
    "metacarpal": 0.008,
    "palm": 0.008,
}
FINGER_PARTS = tuple(
    f"{finger}{part}"
    for finger in ("ff", "mf", "rf", "lf")
    for part in ("proximal", "middle", "distal")
)
HAND_PARTS = ("palm", "lfmetacarpal", "thproximal", "thmiddle", "thdistal", *FINGER_PARTS)
PAD_SPACING = 0.003  # the pincer pads' contact faces are 22 x 37.5 mm


@dataclass(frozen=True)
class Contacts:
    """The active contacts of one physics state, as arrays.

    Attributes:
        body: ``[n, 2]`` body ids of the two sides (geom1, geom2).
        geom: ``[n, 2]`` geom ids of the two sides.
        pos: ``[n, 3]`` world contact position.
        normal: ``[n, 3]`` contact normal, pointing from side 0 to side 1.
        force: ``[n]`` normal force (N), ``>= 0``.
        slip: ``[n]`` tangential relative speed of the two sides at the contact (m/s).
    """

    body: np.ndarray
    geom: np.ndarray
    pos: np.ndarray
    normal: np.ndarray
    force: np.ndarray
    slip: np.ndarray


def read_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> Contacts:
    """Contacts of the current state with at least ``MIN_CONTACT_N`` normal force.

    Velocities come from ``data.cvel`` (spatial velocity about the subtree centre
    of mass of each body's root), which ``mj_step`` leaves consistent with the
    contacts it solved.
    """
    n = data.ncon
    force = np.zeros(n)
    wrench = np.zeros(6)
    for i in range(n):
        mujoco.mj_contactForce(model, data, i, wrench)
        force[i] = wrench[0]
    keep = force >= MIN_CONTACT_N
    geom = np.asarray(data.contact.geom[:n])[keep]
    pos = np.asarray(data.contact.pos[:n])[keep]
    normal = np.asarray(data.contact.frame[:n, :3])[keep]
    body = model.geom_bodyid[geom].reshape(-1, 2)
    velocity = point_velocity(model, data, body[:, 0], pos) - point_velocity(
        model, data, body[:, 1], pos
    )
    tangential = velocity - np.sum(velocity * normal, axis=1, keepdims=True) * normal
    return Contacts(body, geom, pos, normal, force[keep], np.linalg.norm(tangential, axis=1))


def point_velocity(
    model: mujoco.MjModel, data: mujoco.MjData, body: np.ndarray, point: np.ndarray
) -> np.ndarray:
    """``[n, 3]`` world velocity of ``point[i]`` moving rigidly with ``body[i]``."""
    cvel = data.cvel[body]
    centre = data.subtree_com[model.body_rootid[body]]
    return np.asarray(cvel[:, 3:] + np.cross(cvel[:, :3], point - centre))


@dataclass
class SkinRegion:
    """The sensor points of one body.

    Attributes:
        body: MuJoCo body id.
        name: Channel prefix, e.g. ``"hand/left/ffdistal"``.
        spacing: Distance between neighbouring points (m); the spread radius is twice this.
        points, normals: ``[P, 3]`` positions and outward unit normals in the body frame.
        start: Index of the region's first channel in ``DenseSkin.names``.
    """

    body: int
    name: str
    spacing: float
    points: np.ndarray
    normals: np.ndarray
    start: int


class DenseSkin:
    """Sensor points over the collision surface of chosen bodies; contacts spread over them."""

    def __init__(self, model: mujoco.MjModel, bodies: dict[int, tuple[str, float]]) -> None:
        """Lay out the points.

        Args:
            model: The compiled scene.
            bodies: body id -> (channel prefix, spacing in m).

        Raises:
            ValueError: if a body has no collision geom or gets no point.
        """
        self.regions: list[SkinRegion] = []
        self.names: list[str] = []
        self.region_of_body = np.full(model.nbody, -1)
        self.regions_by_body: dict[int, SkinRegion] = {}
        rng = np.random.default_rng(SAMPLING_SEED)
        for body, (name, spacing) in bodies.items():
            points, normals = surface_points(model, body, spacing, rng)
            if not len(points):
                raise ValueError(f"no skin points on body {name!r}")
            self.region_of_body[body] = len(self.regions)
            region = SkinRegion(body, name, spacing, points, normals, len(self.names))
            self.regions.append(region)
            self.regions_by_body[body] = region
            self.names += [f"{name}/{k}" for k in range(len(points))]

    def read(self, data: mujoco.MjData, contacts: Contacts) -> np.ndarray:
        """``[C]`` float32 normal force per sensor point."""
        out = np.zeros(len(self.names), dtype=np.float32)
        for side, outward in ((0, 1.0), (1, -1.0)):
            for i in np.flatnonzero(self.region_of_body[contacts.body[:, side]] >= 0):
                region = self.regions[self.region_of_body[contacts.body[i, side]]]
                rotation = data.xmat[region.body].reshape(3, 3)
                local = rotation.T @ (contacts.pos[i] - data.xpos[region.body])
                facing = rotation.T @ (outward * contacts.normal[i])
                weights = spread_linear(region, local, facing)
                end = region.start + len(region.points)
                out[region.start : end] += contacts.force[i] * weights
        return out


def spread_linear(region: SkinRegion, point: np.ndarray, facing: np.ndarray) -> np.ndarray:
    """MIMo's linear spread of one contact over a region's points, summing to 1.

    Weight ``max(0, 1 - d / (2 * spacing))`` for points whose outward normal faces
    the contact (``normal . facing > FACING_COS``); if none is in reach, the nearest facing
    point (or, failing that, the nearest point) takes everything.
    """
    distance = np.linalg.norm(region.points - point, axis=1)
    faces = region.normals @ facing > FACING_COS
    weights = np.maximum(1.0 - distance / (2.0 * region.spacing), 0.0) * faces
    total = weights.sum()
    if total > 0.0:
        return np.asarray(weights / total)
    nearest = np.argmin(np.where(faces, distance, np.inf) if faces.any() else distance)
    weights[:] = 0.0
    weights[nearest] = 1.0
    return np.asarray(weights)


class ContactDynamics:
    """Onset, slip and vibration per skin region, from every physics sub-step."""

    def __init__(self, model: mujoco.MjModel, regions: dict[int, str]) -> None:
        """Args: regions: body id -> region name (e.g. ``"hand/left/ffdistal"``)."""
        self.model = model
        self.region_of_body = np.full(model.nbody, -1)
        for index, body in enumerate(regions):
            self.region_of_body[body] = index
        self.names = [
            f"dynamics/{name}/{feature}"
            for name in regions.values()
            for feature in DYNAMICS_FEATURES
        ]
        size = len(regions)
        self.last_force = np.zeros(size)
        self.onset = np.zeros(size)
        self.slip = np.zeros(size)
        self.squared_change = np.zeros(size)
        self.substeps = 0

    def region_totals(self, contacts: Contacts) -> tuple[np.ndarray, np.ndarray]:
        """Per region: summed normal force and force-weighted slip speed (beyond the deadband)."""
        size = len(self.last_force)
        force, slip = np.zeros(size), np.zeros(size)
        sliding = np.maximum(contacts.slip - SLIDE_DEADBAND, 0.0)
        for side in (0, 1):
            region = self.region_of_body[contacts.body[:, side]]
            mine = region >= 0
            np.add.at(force, region[mine], contacts.force[mine])
            np.add.at(slip, region[mine], contacts.force[mine] * sliding[mine])
        return force, np.divide(slip, force, out=np.zeros(size), where=force > 0)

    def reset(self, data: mujoco.MjData) -> None:
        """Start from the current state: its contacts are not onsets."""
        self.last_force, _ = self.region_totals(read_contacts(self.model, data))
        self.clear()

    def clear(self) -> None:
        """Forget the accumulated sub-steps (call after ``features``)."""
        self.onset[:] = 0.0
        self.slip[:] = 0.0
        self.squared_change[:] = 0.0
        self.substeps = 0

    def substep(self, data: mujoco.MjData, contacts: Contacts | None = None) -> None:
        """Accumulate one physics step (call after each ``mj_step``).

        Pass ``contacts`` (``read_contacts`` of this state) if the caller already has them.
        """
        force, slip = self.region_totals(contacts or read_contacts(self.model, data))
        touching = force >= MIN_CONTACT_N
        self.onset = np.maximum(self.onset, touching & (self.last_force < MIN_CONTACT_N))
        self.slip += slip
        self.squared_change += (force - self.last_force) ** 2
        self.last_force = force
        self.substeps += 1

    def features(self) -> np.ndarray:
        """``[3 R]`` float32: onset, slip, vibration per region, over the sub-steps so far."""
        n = max(self.substeps, 1)
        values = np.stack([self.onset, self.slip / n, np.sqrt(self.squared_change / n)], axis=1)
        return values.reshape(-1).astype(np.float32)


def region_name(model: mujoco.MjModel, body: int) -> str:
    """Channel prefix of a body, e.g. ``lh_ffdistal`` -> ``hand/left/ffdistal``."""
    return taxel_prefix(str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)))


def dense_bodies(model: mujoco.MjModel, effector: str) -> dict[int, tuple[str, float]]:
    """Bodies that get ``DenseSkin``, with their channel prefix and spacing.

    The Shadow hands' palm, little-finger metacarpal and every phalanx (the
    bodies that carry ``h1touch`` patches), or the four pincer pads.

    Raises:
        ValueError: for an unknown effector.
    """
    if effector == "hand":
        names = [f"{side}_{part}" for side in ("lh", "rh") for part in HAND_PARTS]
        spacing = [next(v for k, v in HAND_SPACING.items() if name.endswith(k)) for name in names]
    elif effector == "pincer":
        names = [f"{side}_{pad}_pad" for side in ("lp", "rp") for pad in ("left", "right")]
        spacing = [PAD_SPACING] * len(names)
    else:
        raise ValueError(f"unknown effector {effector!r}")
    bodies = [model.body(name).id for name in names]
    return {b: (region_name(model, b), s) for b, s in zip(bodies, spacing, strict=True)}


# --- surface sampling ---------------------------------------------------------------


@dataclass(frozen=True)
class Solid:
    """One collision geom in its body's frame: triangles and a signed-distance test."""

    triangles: np.ndarray  # [F, 3, 3] in the body frame, outward winding
    planes: np.ndarray  # [F', 4] outward unit normal and offset: n . x - d <= 0 inside

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box, grown by ``INSIDE_TOLERANCE``."""
        corners = self.triangles.reshape(-1, 3)
        return corners.min(axis=0) - INSIDE_TOLERANCE, corners.max(axis=0) + INSIDE_TOLERANCE

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """``[N]`` max plane distance: negative inside, ~0 on the surface (convex solids)."""
        return np.asarray((points @ self.planes[:, :3].T - self.planes[:, 3]).max(axis=1))


def body_solids(model: mujoco.MjModel, body: int) -> list[Solid]:
    """The distinct colliding mesh and box geoms of ``body`` (exact duplicates dropped).

    Raises:
        ValueError: for a colliding geom that is not a mesh or a box.
    """
    solids: list[Solid] = []
    seen: set[bytes] = set()
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != body:
            continue
        if model.geom_contype[geom] == 0 and model.geom_conaffinity[geom] == 0:
            continue
        kind = model.geom_type[geom]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            vertices, faces = mesh_arrays(model, int(model.geom_dataid[geom]))
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            vertices, faces = box_arrays(model.geom_size[geom])
        else:
            raise ValueError(f"skin needs mesh or box geoms, body {body} has type {kind}")
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
        vertices = vertices @ rotation.reshape(3, 3).T + model.geom_pos[geom]
        key = np.round(np.sort(vertices, axis=0), 5).tobytes()
        if key in seen:
            continue
        seen.add(key)
        solids.append(convex_solid(vertices, faces))
    if not solids:
        raise ValueError(f"body {body} has no collision geom")
    return solids


def mesh_arrays(model: mujoco.MjModel, mesh: int) -> tuple[np.ndarray, np.ndarray]:
    """Vertices ``[V, 3]`` and triangles ``[F, 3]`` of a mesh asset (mesh frame)."""
    v0, nv = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    f0, nf = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
    return model.mesh_vert[v0 : v0 + nv].astype(float), model.mesh_face[f0 : f0 + nf]


def box_arrays(half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and triangles of a box with half-sizes ``half`` (centred at the origin)."""
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    faces = np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])
    return corners * half[:3], faces


def convex_solid(vertices: np.ndarray, faces: np.ndarray) -> Solid:
    """Triangles wound outward (relative to the centroid) and their face planes."""
    triangles = vertices[faces]
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area2 = np.linalg.norm(normal, axis=1)
    triangles, normal, area2 = triangles[area2 > 1e-12], normal[area2 > 1e-12], area2[area2 > 1e-12]
    normal /= area2[:, None]
    inward = np.sum(normal * (vertices.mean(axis=0) - triangles[:, 0]), axis=1) > 0
    normal[inward] *= -1
    triangles[inward] = triangles[inward][:, ::-1]
    planes = np.concatenate([normal, np.sum(normal * triangles[:, 0], axis=1)[:, None]], axis=1)
    return Solid(triangles, planes)


def surface_points(
    model: mujoco.MjModel, body: int, spacing: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Points about ``spacing`` apart on the outer surface of a body's collision geoms.

    Samples each triangle in proportion to its area, drops samples hidden inside
    (or on) another geom of the body, then keeps a Poisson-disc subset: a sample
    is accepted if no accepted point is closer than ``spacing``.

    Returns:
        ``(points, normals)``, ``[P, 3]`` each, in the body frame.
    """
    solids = body_solids(model, body)
    samples, normals = [], []
    area = sum(triangle_areas(s.triangles).sum() for s in solids)
    total = max(int(CANDIDATES_PER_POINT * area / spacing**2), 1)
    for index, solid in enumerate(solids):
        areas = triangle_areas(solid.triangles)
        count = int(round(total * areas.sum() / area))
        if count == 0:
            continue
        pick = rng.choice(len(areas), size=count, p=areas / areas.sum())
        u, v = rng.random(count), rng.random(count)
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        tri = solid.triangles[pick]
        points = (
            tri[:, 0] + u[:, None] * (tri[:, 1] - tri[:, 0]) + v[:, None] * (tri[:, 2] - tri[:, 0])
        )
        visible = np.ones(count, dtype=bool)
        for other, solid_other in enumerate(solids):
            if other == index:
                continue
            low, high = solid_other.bounds()
            near = np.flatnonzero(np.all((points > low) & (points < high), axis=1))
            hidden = solid_other.signed_distance(points[near]) <= INSIDE_TOLERANCE
            visible[near[hidden]] = False
        samples.append(points[visible])
        normals.append(solid.planes[pick, :3][visible])
    return poisson_subset(np.concatenate(samples), np.concatenate(normals), spacing)


def triangle_areas(triangles: np.ndarray) -> np.ndarray:
    """``[F]`` areas."""
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return np.asarray(0.5 * np.linalg.norm(cross, axis=1))


def poisson_subset(
    points: np.ndarray, normals: np.ndarray, spacing: float
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy Poisson-disc subset, in the given (already random) order, via a voxel hash."""
    cells: dict[tuple[int, int, int], list[int]] = {}
    keep: list[int] = []
    for i, point in enumerate(points):
        cell = tuple(int(c) for c in np.floor(point / spacing))
        near = (
            j
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            for j in cells.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), ())
        )
        if any(np.sum((points[j] - point) ** 2) < spacing**2 for j in near):
            continue
        cells.setdefault(cell, []).append(i)  # type: ignore[arg-type]
        keep.append(i)
    return points[keep], normals[keep]
