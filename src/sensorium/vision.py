"""Load-time image views: a random crop for training, a centre crop for evaluation.

Episodes store ``IMAGE_SIZE`` (256) square frames; the training loader turns a
clip into the model's input size here, so the stored data never changes and
the resolution can differ per model (our own encoder, VC-1 or DINOv2 at 224,
a cheap pilot at 128). This follows the convention of the robot-learning
literature (see ``docs/vision.md``): store or render larger than the model
input, take one random crop per training sample, and take the centre crop at
evaluation. Robomimic and Diffusion Policy crop 76 of 84 px (90% of the side);
Octo crops, resizes to 256 and applies colour jitter; V-JEPA 2 uses random
resized crops, the same crop for every frame of a clip.

All frames of one clip share one crop and one colour jitter, so motion
between frames is not confused with augmentation. Pixels stay ``uint8``;
scaling to ``[0, 1]``, ``[-1, 1]`` or ImageNet statistics belongs to the model.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class View:
    """How to turn stored frames into model input.

    Attributes:
        size: Side of the output frames (px).
        crop: Side of the crop as a fraction of the stored side, in ``(0, 1]``.
        brightness, contrast, saturation: Colour jitter strength: each factor is
            drawn from ``U(1 - x, 1 + x)`` per clip, as in torchvision's
            ``ColorJitter``. 0 turns it off (the default: the H1 scene already
            changes colours and lighting every episode).
    """

    size: int = 224
    crop: float = 0.9
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0

    def __post_init__(self) -> None:
        if self.size < 1 or not 0.0 < self.crop <= 1.0:
            raise ValueError(f"need size >= 1 and crop in (0, 1], got {self.size}, {self.crop}")
        if min(self.brightness, self.contrast, self.saturation) < 0.0:
            raise ValueError("brightness, contrast and saturation must be >= 0")


def train_view(
    rgb: np.ndarray, rng: np.random.Generator, view: View, source_side: int | None = None
) -> np.ndarray:
    """``[T, H, W, 3]`` uint8 frames -> ``[T, size, size, 3]``: one random crop and jitter.

    ``source_side``: the frames are a square downscale of frames of this side (a frame cache,
    ``sensorium.model.frame_cache``). The crop is then drawn in the source's pixel grid with the
    same random calls as for the source frames, and taken from the downscale at the matching
    (fractional) position, so a cached stream follows the same random path as an uncached one.
    """
    if source_side is not None and source_side != rgb.shape[1]:
        side = max(1, round(source_side * view.crop))
        top = int(rng.integers(0, source_side - side + 1))
        left = int(rng.integers(0, source_side - side + 1))
        frames = resize_box(rgb, (left, top, side), source_side, view.size)
    else:
        side = crop_side(rgb, view)
        top = int(rng.integers(0, rgb.shape[1] - side + 1))
        left = int(rng.integers(0, rgb.shape[2] - side + 1))
        frames = resize(rgb[:, top : top + side, left : left + side], view.size)
    factors = [rng.uniform(1 - x, 1 + x) for x in (view.brightness, view.contrast, view.saturation)]
    return jitter(frames, *factors)


def eval_view(rgb: np.ndarray, view: View, source_side: int | None = None) -> np.ndarray:
    """``[T, H, W, 3]`` uint8 frames -> ``[T, size, size, 3]``: the centre crop, no jitter.

    ``source_side`` as in ``train_view``.
    """
    if source_side is not None and source_side != rgb.shape[1]:
        side = max(1, round(source_side * view.crop))
        corner = (source_side - side) // 2
        return resize_box(rgb, (corner, corner, side), source_side, view.size)
    side = crop_side(rgb, view)
    top, left = (rgb.shape[1] - side) // 2, (rgb.shape[2] - side) // 2
    return resize(rgb[:, top : top + side, left : left + side], view.size)


def resize_box(
    frames: np.ndarray, crop: tuple[int, int, int], source_side: int, size: int
) -> np.ndarray:
    """Resize the square ``crop = (left, top, side)``, given in the pixel grid of a
    ``source_side`` frame, of square downscaled ``frames`` to ``size`` x ``size`` (bilinear,
    antialiased; the box is fractional in the downscale's grid)."""
    left, top, side = crop
    scale = frames.shape[1] / source_side
    box = (left * scale, top * scale, (left + side) * scale, (top + side) * scale)
    return np.stack(
        [
            np.asarray(Image.fromarray(f).resize((size, size), Image.Resampling.BILINEAR, box))
            for f in frames
        ]
    )


def crop_side(rgb: np.ndarray, view: View) -> int:
    """Side of the square crop (px) for these frames.

    Raises:
        ValueError: if ``rgb`` is not ``[T, H, W, 3]`` uint8.
    """
    if rgb.ndim != 4 or rgb.shape[3] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"want [T, H, W, 3] uint8 frames, got {rgb.shape} {rgb.dtype}")
    return max(1, round(int(min(rgb.shape[1:3])) * view.crop))


def resize(frames: np.ndarray, size: int) -> np.ndarray:
    """Resize each frame to ``size`` x ``size`` (bilinear, antialiased when shrinking)."""
    if frames.shape[1:3] == (size, size):
        return np.ascontiguousarray(frames)
    return np.stack(
        [
            np.asarray(Image.fromarray(f).resize((size, size), Image.Resampling.BILINEAR))
            for f in frames
        ]
    )


def jitter(frames: np.ndarray, brightness: float, contrast: float, saturation: float) -> np.ndarray:
    """Apply brightness, contrast and saturation factors (1 = unchanged) to uint8 frames."""
    if brightness == contrast == saturation == 1.0:
        return frames
    out = frames.astype(np.float32) * brightness
    grey = out @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    out = grey.mean(axis=(1, 2), keepdims=True)[..., None] * (1 - contrast) + out * contrast
    grey = out @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    out = grey[..., None] * (1 - saturation) + out * saturation
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
