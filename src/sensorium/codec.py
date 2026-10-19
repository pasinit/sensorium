"""Camera frames stored as H.264 video inside the episode file.

256 x 256 frames of a textured scene compress to ~50 KB each with the lossless
zlib of ``np.savez_compressed`` (lossless WebP: ~44 KB), which would make the
~20M-step dataset ~2 TB per eye. Robot datasets store frames lossily instead:
LeRobot as AV1 video, Open X-Embodiment as JPEG. Measured on an H1 episode
(see ``docs/vision.md``), H.264 at CRF 18 with full-resolution colour
(``yuv444p``) takes ~2.5 KB per frame at a mean error of 1.6 / 255, the same
error as JPEG quality 95 at 20 KB; LeRobot's AV1 default (CRF 30, ``yuv420p``)
was larger or less accurate here and 5x slower to decode.

Encoding runs single-threaded, so the same frames give the same bytes with the
same ffmpeg (the bundled ``imageio-ffmpeg`` binary). Decoding H.264 is exact by
specification. A keyframe every ``KEYFRAME_STEPS`` frames keeps seeking cheap.
"""

import subprocess

import imageio_ffmpeg
import numpy as np

CRF = 18
PIXEL_FORMAT = "yuv444p"
KEYFRAME_STEPS = 20


def encode_frames(frames: np.ndarray, fps: float) -> np.ndarray:
    """``[T, H, W, 3]`` uint8 frames -> MP4 (H.264) bytes as a 1-D uint8 array.

    Raises:
        ValueError: if ``frames`` is not a non-empty ``[T, H, W, 3]`` uint8 array.
        subprocess.CalledProcessError: if ffmpeg fails.
    """
    if frames.ndim != 4 or frames.shape[3] != 3 or frames.dtype != np.uint8 or not len(frames):
        raise ValueError(f"want non-empty [T, H, W, 3] uint8 frames, got {frames.shape}")
    _, height, width, _ = frames.shape
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps:g}",
        "-i", "-",
        "-c:v", "libx264", "-crf", str(CRF), "-pix_fmt", PIXEL_FORMAT,
        "-g", str(KEYFRAME_STEPS), "-threads", "1",
        "-f", "mp4", "-movflags", "frag_keyframe+empty_moov", "-",
    ]  # fmt: skip
    result = subprocess.run(command, input=frames.tobytes(), capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=np.uint8)


def decode_frames(video: np.ndarray, shape: tuple[int, int, int, int]) -> np.ndarray:
    """MP4 bytes from ``encode_frames`` -> ``[T, H, W, 3]`` uint8 frames of ``shape``.

    Raises:
        ValueError: if the video does not hold exactly ``shape`` frames.
        subprocess.CalledProcessError: if ffmpeg fails.
    """
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-i", "-",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]  # fmt: skip
    result = subprocess.run(command, input=video.tobytes(), capture_output=True, check=True)
    raw = np.frombuffer(result.stdout, dtype=np.uint8)
    if raw.size != int(np.prod(shape)):
        raise ValueError(f"video holds {raw.size} bytes of frames, want {shape}")
    return raw.reshape(shape).copy()
