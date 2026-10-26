"""LeWorldModel's Push-T data as sensorium episodes, and a 64 px copy for LeWM's own loader.

Source: the Hugging Face dataset ``quentinll/lewm-pusht`` (MIT), ``pusht_expert_train.h5`` after
zstd, the data of LeWorldModel (arXiv 2603.19312, Push-T after Diffusion Policy). One HDF5 file in
stable-worldmodel's layout (read from the file's header, 2026-10-26: 18,685 episodes, 2,336,736
steps; the paper says 20,000 episodes of about 196 steps):
- per-step columns ``pixels`` (N, 224, 224, 3) uint8, ``action`` (N, 2), ``proprio`` (N, 4: agent
  x, y, vx, vy) and ``state`` (N, 7: agent x, y, block x, y, block angle, agent vx, vy);
- the episode index ``ep_len`` and ``ep_offset``.
``pixels`` is compressed with a plugin filter (``hdf5plugin`` registers it); the file is 46.3 GB,
13.1 GB as zstd. Needs the ``pusht``
extra (``h5py``, ``hdf5plugin``) and, for ``clip_split``, the ``model`` extra (torch).

An episode of body ``pusht`` (10 Hz, ``DT``) holds:
- ``rgb``: the frames upscaled to 256 px (``IMAGE_SIZE``);
- ``proprio``: the agent's position and velocity (``PROPRIO_NAMES``);
- ``action``: the commanded velocity divided by ``action_scale``, so that it lies in [-1, 1];
- no touch and silent audio, so a loader sees vision and proprio only;
- ``privileged/state``: the full state (``STATE_NAMES``), for goal states in planning; never model
  input.

LeWM reads the same frames at 64 px (``lewm_frames``: the full stored frame resized as
``sensorium.vision.resize`` does), so with a full-frame view both models see identical pixels.
``clip_split`` reproduces LeWM's train/validation split: a seeded random split of clips, not
episodes (``le-wm/train.py``).
"""

import itertools
import math
from collections.abc import Iterator, Sequence
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers the Blosc filters the source may use)
import numpy as np
import torch
from PIL import Image

from sensorium.episode import IMAGE_SIZE, SAMPLE_RATE, Episode
from sensorium.vision import resize

SOURCE = "huggingface.co/datasets/quentinll/lewm-pusht (pusht_expert_train.h5, MIT)"
BODY = "pusht"
DT = 0.1  # s: Push-T's 10 Hz control
PROPRIO_NAMES = ["agent_x", "agent_y", "agent_vx", "agent_vy"]  # workspace px and px/s
ACTION_NAMES = ["agent_vx_cmd", "agent_vy_cmd"]
STATE_NAMES = ["agent_x", "agent_y", "block_x", "block_y", "block_angle", "agent_vx", "agent_vy"]
SOURCE_SIZE = 224
LEWM_SIZE = 64
# LeWM's training clips (config/train/lewm.yaml, data/pusht.yaml): history 3 + 1 prediction,
# frame-skip 5; split 90/10 by a torch generator seeded 3072.
LEWM_NUM_STEPS, LEWM_FRAMESKIP, LEWM_TRAIN, LEWM_SEED = 4, 5, 0.9, 3072
COLUMNS = ("pixels", "action", "proprio", "state")


def episode_index(handle: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    """``(lengths, offsets)`` of the episodes of an open stable-worldmodel HDF5 file."""
    return np.asarray(handle["ep_len"][:], dtype=np.int64), np.asarray(
        handle["ep_offset"][:], dtype=np.int64
    )


def read_episode(handle: h5py.File, index: int) -> dict[str, np.ndarray]:
    """The ``COLUMNS`` arrays of episode ``index``.

    Raises:
        IndexError: if there is no such episode.
    """
    lengths, offsets = episode_index(handle)
    if not 0 <= index < len(lengths):
        raise IndexError(f"episode {index} of {len(lengths)}")
    start, stop = int(offsets[index]), int(offsets[index] + lengths[index])
    return {name: np.asarray(handle[name][start:stop]) for name in COLUMNS}


def upscale(pixels: np.ndarray) -> np.ndarray:
    """``[T, 224, 224, 3]`` -> ``[T, IMAGE_SIZE, IMAGE_SIZE, 3]`` uint8 (bilinear)."""
    return np.stack(
        [
            np.asarray(
                Image.fromarray(f).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
            )
            for f in pixels
        ]
    )


def to_episode(
    index: int, arrays: dict[str, np.ndarray], action_scale: float, revision: str = ""
) -> Episode:
    """One source episode as a sensorium ``Episode``.

    Raises:
        ValueError: if the scaled actions leave [-1, 1] or a column is malformed.
    """
    pixels = arrays["pixels"]
    t = len(pixels)
    if pixels.shape[1:] != (SOURCE_SIZE, SOURCE_SIZE, 3):
        raise ValueError(f"pixels: want [T, {SOURCE_SIZE}, {SOURCE_SIZE}, 3], got {pixels.shape}")
    action = np.asarray(arrays["action"], dtype=np.float32).reshape(t, -1) / np.float32(
        action_scale
    )
    if action.shape[1] != len(ACTION_NAMES) or np.abs(action).max(initial=0.0) > 1.0 + 1e-6:
        raise ValueError(f"action: {action.shape}, max |a| {np.abs(action).max():.3f} after scale")
    episode = Episode(
        body=BODY,
        dt=DT,
        rgb=upscale(pixels),
        audio=np.zeros((t, round(DT * SAMPLE_RATE)), dtype=np.int16),
        audio_logmel=np.zeros((t, 0), dtype=np.float16),
        proprio=np.asarray(arrays["proprio"], dtype=np.float32).reshape(t, len(PROPRIO_NAMES)),
        touch=np.zeros((t, 0), dtype=np.float32),
        action=np.clip(action, -1.0, 1.0),
        proprio_names=list(PROPRIO_NAMES),
        touch_names=[],
        action_names=list(ACTION_NAMES),
        privileged={
            "state": np.asarray(arrays["state"], dtype=np.float32).reshape(t, len(STATE_NAMES))
        },
        text=[""] * t,
        meta={
            "source": SOURCE,
            "source_revision": revision,
            "source_episode": index,
            "licence": "MIT",
            "action_scale": action_scale,
            "state_names": STATE_NAMES,
            "senses": ["vision", "proprio"],
        },
    )
    episode.validate()
    return episode


def lewm_frames(episode: Episode) -> np.ndarray:
    """``[T, 64, 64, 3]``: the stored frames as LeWM sees them (full frame, ``vision.resize``)."""
    return resize(episode.rgb, LEWM_SIZE)


def split_lengths(n: int, fractions: Sequence[float]) -> list[int]:
    """Subset sizes of ``n`` items for ``fractions``, as ``torch.utils.data.random_split``."""
    sizes = [int(math.floor(n * f)) for f in fractions]
    for i in range(n - sum(sizes)):
        sizes[i % len(sizes)] += 1
    return sizes


def clips(lengths: Sequence[int], num_steps: int, frameskip: int) -> list[tuple[int, int]]:
    """``(episode, start)`` of every clip, in stable-worldmodel's order
    (``Dataset.clip_indices``)."""
    span = num_steps * frameskip
    return [
        (ep, start)
        for ep, length in enumerate(lengths)
        if length >= span
        for start in range(length - span + 1)
    ]


def clip_split(
    lengths: Sequence[int],
    num_steps: int = LEWM_NUM_STEPS,
    frameskip: int = LEWM_FRAMESKIP,
    train: float = LEWM_TRAIN,
    seed: int = LEWM_SEED,
) -> dict[str, list[tuple[int, int]]]:
    """LeWM's train/validation clips: a random permutation from a torch generator seeded
    ``seed``, cut into ``[train, 1 - train]`` (``stable_pretraining.data.random_split``)."""
    every = clips(lengths, num_steps, frameskip)
    sizes = split_lengths(len(every), [train, 1.0 - train])
    order = torch.randperm(len(every), generator=torch.Generator().manual_seed(seed)).tolist()
    parts = [
        order[end - size : end]
        for end, size in zip(itertools.accumulate(sizes), sizes, strict=True)
    ]
    return {
        name: [every[i] for i in part] for name, part in zip(("train", "val"), parts, strict=True)
    }


def write_lewm_h5(path: Path, episodes: Iterator[dict[str, np.ndarray]]) -> int:
    """Write ``episodes`` (``COLUMNS`` arrays, pixels at 64 px) in stable-worldmodel's HDF5 layout.

    Returns:
        The number of episodes written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count, offset = 0, 0
    with h5py.File(path, "w") as out:
        for arrays in episodes:
            t = len(arrays["pixels"])
            for name in COLUMNS:
                data = np.asarray(arrays[name])
                if name not in out:
                    out.create_dataset(
                        name,
                        shape=(0, *data.shape[1:]),
                        maxshape=(None, *data.shape[1:]),
                        dtype=data.dtype,
                        chunks=(min(100, max(t, 1)), *data.shape[1:]),  # as the source
                    )
                out[name].resize(offset + t, axis=0)
                out[name][offset : offset + t] = data
            for name, value, dtype in (("ep_len", t, np.int32), ("ep_offset", offset, np.int64)):
                if name not in out:
                    out.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype)
                out[name].resize(count + 1, axis=0)
                out[name][count] = value
            count, offset = count + 1, offset + t
    return count
