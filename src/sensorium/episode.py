"""The episode format shared by every body.

One episode = one compressed ``.npz`` file. Arrays indexed by time share the same
length ``T``. A body that lacks a sense stores an empty array (shape ``[T, 0]``)
rather than omitting the key, so every file has the same keys. Camera frames
(``rgb`` and any ``[T, H, W, 3]`` uint8 privileged array, such as a second eye)
are stored as H.264 video under ``<key>.mp4`` (``codec.py``): lossy, mean error
~1.6 / 255, about 20x smaller than lossless. ``load`` decodes them, so callers
see arrays either way; files written before this change load unchanged.

Two groups of fields must never reach the world model's input:

* ``privileged/*``: ground-truth simulator state, used only for metrics,
  probe labels and audio synthesis.
* ``text``: templated descriptions, used only by the language probe.

``MODEL_INPUT_KEYS`` is the allow-list a training loader must use.

Timing convention (all bodies): at step ``t`` every sense (``rgb``, ``proprio``,
``touch``, ``privileged``, ``text``) is observed *before* ``action[t]`` is
applied. ``audio[t]`` is the sound of the interval that led up to step ``t``.
So ``(senses[t], action[t]) -> senses[t + 1]`` is one world-model transition.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sensorium.codec import decode_frames, encode_frames

SAMPLE_RATE = 16_000  # audio samples per second, all bodies
IMAGE_SIZE = 256  # RGB frames are IMAGE_SIZE x IMAGE_SIZE; crop and resize at load time (vision.py)

# Only these fields may be fed to the world model.
MODEL_INPUT_KEYS = ("rgb", "audio_logmel", "proprio", "touch", "action")


@dataclass
class Episode:
    """One recorded episode for one body.

    Attributes:
        body: Body identifier, e.g. ``"car"``, ``"drone"``, ``"piano"``.
        dt: Control period in seconds (one step = ``dt`` seconds).
        rgb: ``[T, IMAGE_SIZE, IMAGE_SIZE, 3]`` uint8 frames; the model sees a crop (``vision.py``).
        audio: ``[T, dt * SAMPLE_RATE]`` int16 waveform, one chunk per step.
        audio_logmel: ``[T, n_mels]`` float16 log-mel features of each chunk.
        proprio: ``[T, P]`` float32 body state the body itself can sense.
        touch: ``[T, C]`` float32 contact forces; ``[T, 0]`` if no touch.
        action: ``[T, A]`` float32 actions in ``[-1, 1]``.
        proprio_names, touch_names, action_names: One name per column.
        privileged: Ground-truth arrays (``[T, ...]``), never model input.
        text: ``[T]`` descriptions (``""`` where none), never model input.
        meta: Provenance: seed, policy, simulator settings, source data.
    """

    body: str
    dt: float
    rgb: np.ndarray
    audio: np.ndarray
    audio_logmel: np.ndarray
    proprio: np.ndarray
    touch: np.ndarray
    action: np.ndarray
    proprio_names: list[str]
    touch_names: list[str]
    action_names: list[str]
    privileged: dict[str, np.ndarray] = field(default_factory=dict)
    text: list[str] = field(default_factory=list)
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Number of steps ``T``."""
        return int(self.rgb.shape[0])

    def validate(self) -> None:
        """Check shapes, dtypes and value ranges.

        Raises:
            ValueError: on the first violated invariant, naming the field.
        """
        t = self.length
        expected = {
            "rgb": ((t, IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8),
            "audio": ((t, round(self.dt * SAMPLE_RATE)), np.int16),
            "audio_logmel": ((t, self.audio_logmel.shape[1]), np.float16),
            "proprio": ((t, len(self.proprio_names)), np.float32),
            "touch": ((t, len(self.touch_names)), np.float32),
            "action": ((t, len(self.action_names)), np.float32),
        }
        for name, (shape, dtype) in expected.items():
            array = getattr(self, name)
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(f"{name}: got {array.shape} {array.dtype}, want {shape} {dtype}")
        if len(self.text) != t:
            raise ValueError(f"text: got {len(self.text)} entries, want {t}")
        for name, array in self.privileged.items():
            if array.shape[0] != t:
                raise ValueError(f"privileged/{name}: got {array.shape[0]} steps, want {t}")
        if np.abs(self.action).max(initial=0.0) > 1.0 + 1e-6:
            raise ValueError("action: values outside [-1, 1]")
        for name in ("proprio", "touch", "action", "audio_logmel"):
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"{name}: contains NaN or inf")

    def save(self, path: Path) -> None:
        """Validate, then write the episode to ``path`` (a ``.npz`` file).

        Raises:
            ValueError: if the episode is invalid (nothing is written).
        """
        self.validate()
        header = {
            "body": self.body,
            "dt": self.dt,
            "proprio_names": self.proprio_names,
            "touch_names": self.touch_names,
            "action_names": self.action_names,
            "meta": self.meta,
        }
        arrays = {name: getattr(self, name) for name in MODEL_INPUT_KEYS}
        arrays["audio"] = self.audio
        arrays |= {f"privileged/{k}": v for k, v in self.privileged.items()}
        frames = {name: list(a.shape) for name, a in arrays.items() if is_frames(a)}
        for name in frames:
            arrays[f"{name}.mp4"] = encode_frames(arrays.pop(name), 1.0 / self.dt)
        header["frames"] = frames
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            header=np.array(json.dumps(header)),
            text=np.array(self.text, dtype=np.str_),
            **arrays,
        )

    @classmethod
    def load(cls, path: Path, privileged: bool = True) -> "Episode":
        """Read an episode written by ``save``.

        ``privileged=False`` skips the privileged arrays (``privileged`` is then
        empty), which a training loader never uses; it also skips decoding a
        second camera stored there, which is most of the load time.

        Raises:
            FileNotFoundError: if ``path`` does not exist.
            ValueError: if the stored episode fails validation.
        """
        with np.load(path, allow_pickle=False) as data:
            header = json.loads(str(data["header"]))
            frames = header.get("frames", {})

            def read(key: str) -> np.ndarray:
                if key in frames:
                    return decode_frames(data[f"{key}.mp4"], tuple(frames[key]))
                return np.asarray(data[key])

            stored = [k.removesuffix(".mp4") for k in data.files]
            episode = cls(
                body=header["body"],
                dt=header["dt"],
                rgb=read("rgb"),
                audio=data["audio"],
                audio_logmel=data["audio_logmel"],
                proprio=data["proprio"],
                touch=data["touch"],
                action=data["action"],
                proprio_names=header["proprio_names"],
                touch_names=header["touch_names"],
                action_names=header["action_names"],
                privileged={
                    k.removeprefix("privileged/"): read(k)
                    for k in stored
                    if privileged and k.startswith("privileged/")
                },
                text=[str(s) for s in data["text"]],
                meta=header["meta"],
            )
        episode.validate()
        return episode


def model_inputs(episode: Episode, senses: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    """Return only the fields a world model may see (``MODEL_INPUT_KEYS``).

    Training code must build batches through this function, so privileged
    state and text can never leak into the model's input.

    Args:
        episode: The episode to read.
        senses: Optional subset of ``MODEL_INPUT_KEYS`` to keep (a pool's ``senses``);
            other keys are dropped. ``action`` is always kept. ``None`` keeps everything.

    Raises:
        ValueError: if ``senses`` names a key that is not in ``MODEL_INPUT_KEYS``.
    """
    if senses is None:
        keep = set(MODEL_INPUT_KEYS)
    else:
        unknown = set(senses) - set(MODEL_INPUT_KEYS)
        if unknown:
            raise ValueError(f"senses {sorted(unknown)} not in {MODEL_INPUT_KEYS}")
        keep = set(senses) | {"action"}
    return {name: getattr(episode, name) for name in MODEL_INPUT_KEYS if name in keep}


def is_frames(array: np.ndarray) -> bool:
    """True for non-empty ``[T, H, W, 3]`` uint8 arrays, which are stored as video."""
    return array.ndim == 4 and array.shape[3] == 3 and array.dtype == np.uint8 and len(array) > 0
