"""Templated state descriptions: labels for the language probe, never model input.

Each function turns one step of privileged state into a short sentence. They
are deliberately plain and slot-based, so the probe can be scored per slot
(direction, speed band, which finger, which key, ...).
"""

import numpy as np

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FINGERS = ("thumb", "index", "middle", "ring", "little")


def speed_band(speed: float, slow: float, fast: float) -> str:
    """Map a speed to ``"stopped" | "slowly" | "steadily" | "fast"``."""
    if speed < 0.2 * slow:
        return "stopped"
    if speed < slow:
        return "slowly"
    return "steadily" if speed < fast else "fast"


def describe_car(speed: float, yaw_rate: float, accel: float, crashed: bool) -> str:
    """E.g. ``"car moving steadily at 9 m/s, turning left, braking"``."""
    if crashed:
        return f"car collided at {speed:.0f} m/s"
    band = speed_band(speed, slow=5.0, fast=15.0)
    if band == "stopped":
        return "car stopped"
    parts = [f"car moving {band} at {speed:.0f} m/s"]
    if abs(yaw_rate) > 0.15:
        parts.append("turning left" if yaw_rate > 0 else "turning right")
    if accel > 1.0:
        parts.append("accelerating")
    elif accel < -1.0:
        parts.append("braking")
    return ", ".join(parts)


def describe_drone(velocity: np.ndarray, roll: float, altitude: float, crashed: bool) -> str:
    """E.g. ``"drone at 1.2 m, climbing, banking right, moving slowly"``."""
    if crashed:
        return "drone crashed"
    parts = [f"drone at {altitude:.1f} m"]
    if velocity[2] > 0.15:
        parts.append("climbing")
    elif velocity[2] < -0.15:
        parts.append("descending")
    if abs(roll) > 0.1:
        parts.append("banking right" if roll > 0 else "banking left")
    horizontal = float(np.linalg.norm(velocity[:2]))
    parts.append("hovering" if horizontal < 0.1 else f"moving {speed_band(horizontal, 0.5, 1.5)}")
    return ", ".join(parts)


def key_name(key_index: int) -> str:
    """Piano key 0..87 (A0..C8) -> scientific pitch name, e.g. 39 -> ``"C4"``."""
    midi = key_index + 21
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def describe_piano(pressed: list[tuple[str, str, int]]) -> str:
    """``pressed`` = ``[(hand, finger, key_index), ...]`` -> e.g.
    ``"right index presses C4; left thumb presses E3"`` or ``"no keys pressed"``."""
    if not pressed:
        return "no keys pressed"
    return "; ".join(f"{hand} {finger} presses {key_name(key)}" for hand, finger, key in pressed)


def describe_humanoid(posture: str, touches: list[tuple[str, str]], falling: bool) -> str:
    """``touches`` = ``[(body part, surface), ...]`` -> e.g.
    ``"seated; left hand touching metal object, right arm touching table"``,
    ``"standing"`` or ``"falling"``. Feet on the floor are not listed."""
    if falling:
        return "falling"
    if not touches:
        return posture
    return f"{posture}; " + ", ".join(f"{part} touching {surface}" for part, surface in touches)
