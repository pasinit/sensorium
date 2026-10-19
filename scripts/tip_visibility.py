"""How often the effector tips fall outside the head camera's image, and where.

Usage: python scripts/tip_visibility.py <generated dir> [extra_pitch_deg fovy_deg]

Reads ``<dir>/humanoid/humanoid-*.npz`` (seated episodes), projects each step's two tip
positions (``privileged/tip_position``) through the head camera of the scene, and prints per
variant the share of tip-steps outside the 256 x 256 image (or behind the camera), and which
table regions and heights those tips are over. The optional extra pitch (degrees, positive
tilts the camera further down) and field of view test a different camera on the same tip
positions (the camera is fixed in the world while seated).
"""

import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from sensorium.bodies.h1_camera import camera_pose, outside_image, pitched
from sensorium.bodies.h1_scene import build_scene
from sensorium.episode import Episode

Y_BINS = (-0.5, -0.3, -0.15, 0.0, 0.15, 0.3, 0.5)
X_BINS = (0.2, 0.4, 0.5, 0.6, 0.8)
HEIGHT_BINS = ((-1.0, 0.04), (0.04, 0.12), (0.12, 9.0))  # tip height above the table (m)
SAFE_X, SAFE_Y, SAFE_HEIGHT = 0.40, 0.30, 0.12  # the region proposed for goals


@dataclass
class Tally:
    """Counts of tip-steps outside the image, overall and by region."""

    steps: int = 0
    outside: np.ndarray = field(default_factory=lambda: np.zeros(2))  # left and right tip
    by_xy: np.ndarray = field(
        default_factory=lambda: np.zeros((len(X_BINS) - 1, len(Y_BINS) - 1, 2))
    )
    by_height: np.ndarray = field(default_factory=lambda: np.zeros((len(HEIGHT_BINS), 2)))
    safe: np.ndarray = field(default_factory=lambda: np.zeros(3))  # outside, all there, all outside
    causes: np.ndarray = field(default_factory=lambda: np.zeros(4))  # high, lateral, near, other

    def add(self, tip: np.ndarray, out: np.ndarray, side: int, top: float) -> None:
        """Fold one tip's ``[T, 3]`` positions and its outside flags in."""
        self.outside[side] += out.sum()
        x_bin = np.digitize(tip[:, 0], X_BINS) - 1
        y_bin = np.digitize(tip[:, 1], Y_BINS) - 1
        for xb, yb, flag in zip(x_bin, y_bin, out, strict=True):
            if 0 <= xb < self.by_xy.shape[0] and 0 <= yb < self.by_xy.shape[1]:
                self.by_xy[xb, yb] += (flag, 1)
        height = tip[:, 2] - top
        for i, (low, high) in enumerate(HEIGHT_BINS):
            chosen = (height >= low) & (height < high)
            self.by_height[i] += ((out & chosen).sum(), chosen.sum())
        safe = (tip[:, 0] >= SAFE_X) & (np.abs(tip[:, 1]) <= SAFE_Y) & (height <= SAFE_HEIGHT)
        self.safe += ((out & safe).sum(), safe.sum(), out.sum())
        high = height > SAFE_HEIGHT
        lateral = ~high & (np.abs(tip[:, 1]) > SAFE_Y)
        near = ~high & ~lateral & (tip[:, 0] < SAFE_X)
        other = out & ~high & ~lateral & ~near
        self.causes += ((out & high).sum(), (out & lateral).sum(), (out & near).sum(), other.sum())

    def report(self, name: str, episodes: int, pitch: float, fovy: float) -> str:
        """The tables for one variant as text."""
        n = self.steps
        lines = [
            f"\n{name}: {episodes} episodes, tip-steps outside the image: left "
            f"{self.outside[0] / n:.1%}, right {self.outside[1] / n:.1%} "
            f"(extra pitch {pitch:g} deg, fovy {fovy:g})",
            f"  region x >= {SAFE_X}, |y| <= {SAFE_Y}, <= {SAFE_HEIGHT * 100:.0f} cm up: "
            f"{self.safe[1] / (2 * n):.0%} of tip-steps, {self.safe[0] / max(self.safe[1], 1):.1%} "
            f"of them outside the image; {1 - self.safe[0] / max(self.safe[2], 1):.0%} of all "
            "outside steps lie elsewhere",
            "  outside steps by first cause: "
            + ", ".join(
                f"{label} {count / max(self.causes.sum(), 1):.0%}"
                for label, count in zip(
                    (
                        f"> {SAFE_HEIGHT * 100:.0f} cm up",
                        f"|y| > {SAFE_Y}",
                        f"x < {SAFE_X}",
                        "elsewhere",
                    ),
                    self.causes,
                    strict=True,
                )
            ),
            "  outside, by tip height above the table: "
            + ", ".join(
                f"{label} {self.by_height[i, 0] / max(self.by_height[i, 1], 1):.0%}"
                f" (of {self.by_height[i, 1] / (2 * n):.0%} of steps)"
                for i, label in enumerate(("< 4 cm", "4-12 cm", "> 12 cm"))
            ),
            "  outside, by tip x (rows, m) and y (columns, m):",
            "    x \\ y   " + " ".join(f"{Y_BINS[i]:+.2f}.." for i in range(len(Y_BINS) - 1)),
        ]
        for xb in range(self.by_xy.shape[0]):
            cells = " ".join(
                f"{self.by_xy[xb, yb, 0] / self.by_xy[xb, yb, 1]:>7.0%}"
                if self.by_xy[xb, yb, 1] > 20
                else "      -"
                for yb in range(self.by_xy.shape[1])
            )
            lines.append(f"    {X_BINS[xb]:.1f}-{X_BINS[xb + 1]:.1f} {cells}")
        return "\n".join(lines) + "\n"


def tally_variant(
    paths: list[Path], model: mujoco.MjModel, top: float, pitch: float, fovy: float
) -> Tally:
    """Tally every episode of one variant."""
    tally = Tally()
    for path in paths:
        episode = Episode.load(path)
        position, rotation = camera_pose(model, episode.privileged["qpos"][0].astype(float))
        rotation = pitched(rotation, pitch)
        tips = episode.privileged["tip_position"].astype(float)
        for side in (0, 1):
            tally.add(
                tips[:, side], outside_image(tips[:, side], position, rotation, fovy), side, top
            )
        tally.steps += len(tips)
    return tally


def main() -> None:
    """Print the visibility tables."""
    directory = Path(sys.argv[1])
    pitch = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    fovy = float(sys.argv[3]) if len(sys.argv) > 3 else 80.0
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted((directory / "humanoid").glob("humanoid-*.npz")):
        match = re.fullmatch(r"humanoid-(.+)-seated_babble-\d+\.npz", path.name)
        if match:
            groups[match.group(1)].append(path)
    for variant, paths in groups.items():
        meta = Episode.load(paths[0]).meta
        top = float(meta["table_top"])  # type: ignore[arg-type]
        scene = build_scene(str(meta["effector"]), "seated", table_top=top)
        model = mujoco.MjModel.from_xml_path(str(scene))
        tally = tally_variant(paths, model, top, pitch, fovy)
        sys.stdout.write(tally.report(variant, len(paths), pitch, fovy))


if __name__ == "__main__":
    main()
