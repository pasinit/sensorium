"""The head camera's view of the effector tips: where the camera is and which points it cannot see.

Used by ``scripts/tip_visibility.py`` (tables) and ``sensorium.a1_gate`` (pass/fail). The camera is
fixed in the world while the pelvis is, so one pose per episode serves every step.
"""

import math

import mujoco
import numpy as np

from sensorium.episode import IMAGE_SIZE, Episode

CAMERA = "left_eye_camera"
FOVY_DEG = 80.0  # the scene's head camera (docs/vision.md)


def camera_pose(model: mujoco.MjModel, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World position and rotation (columns: right, up, backwards) of the head camera."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)
    camera = model.camera(CAMERA).id
    return data.cam_xpos[camera].copy(), data.cam_xmat[camera].reshape(3, 3).copy()


def outside_image(
    points: np.ndarray, position: np.ndarray, rotation: np.ndarray, fovy_deg: float
) -> np.ndarray:
    """``[N]`` True where a world point projects outside the image or behind the camera."""
    local = (points - position) @ rotation  # camera frame: x right, y up, z backwards
    focal = (IMAGE_SIZE / 2) / math.tan(math.radians(fovy_deg) / 2)
    depth = -local[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = IMAGE_SIZE / 2 + focal * local[:, 0] / depth
        v = IMAGE_SIZE / 2 - focal * local[:, 1] / depth
    inside = (depth > 0) & (u >= 0) & (u < IMAGE_SIZE) & (v >= 0) & (v < IMAGE_SIZE)
    return np.asarray(~inside)


def pitched(rotation: np.ndarray, extra_deg: float) -> np.ndarray:
    """The camera rotated about its own x axis (positive tilts it further down)."""
    angle = math.radians(-extra_deg)
    about_x = np.array(
        [[1, 0, 0], [0, math.cos(angle), -math.sin(angle)], [0, math.sin(angle), math.cos(angle)]]
    )
    return np.asarray(rotation @ about_x)


def tip_outside_shares(episode: Episode, model: mujoco.MjModel) -> np.ndarray:
    """``[2]`` share of steps in which the (left, right) effector tip is outside the image.

    Args:
        episode: a seated or supported H1 episode with ``privileged/qpos`` and ``tip_position``.
        model: the scene's model (``build_scene`` with the episode's table height).
    """
    position, rotation = camera_pose(model, episode.privileged["qpos"][0].astype(float))
    tips = episode.privileged["tip_position"].astype(float)
    return np.array(
        [outside_image(tips[:, side], position, rotation, FOVY_DEG).mean() for side in (0, 1)]
    )
