"""One recorder per body. Each recorder turns ``(seed, policy, steps)`` into an Episode.

All recorders share this shape:

    with CarRecorder(settings) as recorder:
        episode = recorder.record(seed=3, policy="expert", steps=300)

``make_recorder`` imports a body's module only when that body is requested, so
each worker process loads exactly one simulator. Simulators conflict when
loaded together (MuJoCo's renderer fails to load after MetaDrive).
"""

import os
import sys
from collections.abc import Mapping
from typing import Protocol

from sensorium.episode import Episode


def gl_defaults(platform: str, environ: Mapping[str, str]) -> dict[str, str]:
    """The offscreen-rendering variables to set when the caller has not chosen them.

    MuJoCo picks its OpenGL backend from ``MUJOCO_GL`` when it is first imported. macOS renders
    with GLFW; a Linux host without a display (a rented VM) has no X server for GLFW and must use
    EGL (with ``PYOPENGL_PLATFORM`` to match); a Linux desktop keeps MuJoCo's own default.
    Variables already set are never overridden.
    """
    out: dict[str, str] = {}
    if "MUJOCO_GL" not in environ:
        if platform == "darwin":
            out["MUJOCO_GL"] = "glfw"
        elif not environ.get("DISPLAY") and not environ.get("WAYLAND_DISPLAY"):
            out["MUJOCO_GL"] = "egl"
    if out.get("MUJOCO_GL", environ.get("MUJOCO_GL")) == "egl":
        out.setdefault("PYOPENGL_PLATFORM", environ.get("PYOPENGL_PLATFORM", "egl"))
    return {k: v for k, v in out.items() if k not in environ}


# Before any body module imports mujoco: every recorder and ``restore`` is imported through this
# package, so the backend is chosen here once (a script that imports mujoco itself first must set
# MUJOCO_GL itself).
os.environ.update(gl_defaults(sys.platform, os.environ))


class Recorder(Protocol):
    """Interface every body recorder implements."""

    policies: tuple[str, ...]

    def record(self, seed: int, policy: str, steps: int) -> Episode:
        """Run one episode of at most ``steps`` steps and return it."""
        ...

    def close(self) -> None:
        """Release the simulator."""
        ...


def make_recorder(body: str, settings: dict[str, object]) -> Recorder:
    """Build the recorder for ``body`` (``"car" | "drone" | "piano" | "humanoid"``).

    Raises:
        ValueError: for an unknown body name.
    """
    if body == "car":
        from sensorium.bodies.car import CarRecorder  # noqa: PLC0415 (one simulator per process)

        return CarRecorder(settings)
    if body == "drone":
        from sensorium.bodies.drone import DroneRecorder  # noqa: PLC0415 (one simulator per process)

        return DroneRecorder(settings)
    if body == "piano":
        from sensorium.bodies.piano import PianoRecorder  # noqa: PLC0415 (one simulator per process)

        return PianoRecorder(settings)
    if body == "humanoid":
        from sensorium.bodies.humanoid import HumanoidRecorder  # noqa: PLC0415 (one simulator per process)

        return HumanoidRecorder(settings)
    raise ValueError(f"unknown body {body!r}; expected car, drone, piano or humanoid")
