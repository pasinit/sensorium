"""One recorder per body. Each recorder turns ``(seed, policy, steps)`` into an Episode.

All recorders share this shape:

    with CarRecorder(settings) as recorder:
        episode = recorder.record(seed=3, policy="expert", steps=300)

``make_recorder`` imports a body's module only when that body is requested, so
each worker process loads exactly one simulator. Simulators conflict when
loaded together (MuJoCo's renderer fails to load after MetaDrive).
"""

from typing import Protocol

from sensorium.episode import Episode


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
