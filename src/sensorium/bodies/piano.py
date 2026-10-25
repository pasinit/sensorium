"""Piano body: two Shadow hands at an 88-key piano (RoboPianist, MuJoCo).

This is the E1 *target* body. Expert data comes from RP1M (CC BY 4.0,
doi:10.17617/3.XCE8NX): for each song, 500 expert trajectories x 550 steps of
actions, hand joint angles, fingertip positions and key states. RP1M has no
images, sound or touch, so we *re-render* them.

Re-rendering is kinematic: each step we set the recorded hand joints and key
positions, then let MuJoCo compute geometry and contacts. Replaying the
recorded actions through the dynamics with our settings does not reproduce
the recorded key presses (RP1M's paper lists its env settings but not its
simulator version; matching them is still to do), while the recorded poses
are ground truth.

RP1M trajectories place the hands at different base positions along the
keyboard (offsets of a few cm from current RoboPianist, varying by trajectory).
For every episode we calibrate each hand's base offset from the first frame's
recorded fingertips, then require every frame's fingertips to match the
recording within ``MAX_FINGERTIP_ERROR``.

Senses: RGB looking down at hands and keys, synthesised piano sound (from key states), fingertip
touch sensors (10), proprioception (46 joint angles).
Actions: RP1M's 39-D actions (RoboPianist reduced action space), scaled to [-1, 1].
Control: 20 Hz.

Policies:
    expert   one RP1M trajectory, chosen by the seed.
"""

import sys
import time
import types
from pathlib import Path
from types import TracebackType

import numpy as np
import zarr

from sensorium.audio import PianoSound, logmel, to_int16
from sensorium.describe import FINGERS, describe_piano
from sensorium.episode import IMAGE_SIZE, SAMPLE_RATE, Episode

DT = 0.05
STAGE_SONG = "RoboPianist-debug-TwinkleTwinkleRousseau-v0"  # any song: only the scene is used
# Top-down shows both hands *and* the keys at 64 px; the egocentric view hides
# the keys behind the forearms. Overridable with the ``camera`` setting.
DEFAULT_CAMERA = "piano/topdown"
KEY_DOWN_RADIANS = 0.00872665  # RoboPianist: a key sounds within 0.5 degrees of fully down
MAX_FINGERTIP_ERROR = 0.002  # m, replay must match RP1M's recorded fingertips


class PianoRecorder:
    """Re-renders RP1M expert trajectories as multisensory episodes.

    Settings: ``rp1m_path`` (str, required) path to an RP1M ``.zarr`` directory;
    ``songs`` (list[str], optional) RP1M song groups to sample from, default all;
    ``camera`` (str, optional) MuJoCo camera name, default ``DEFAULT_CAMERA``.
    """

    policies: tuple[str, ...] = ("expert",)

    def __init__(self, settings: dict[str, object]) -> None:
        # robopianist imports pyaudio at module level for live playback, which we
        # never use (sound is synthesised offline). An empty stand-in avoids
        # requiring the system PortAudio library.
        sys.modules.setdefault("pyaudio", types.ModuleType("pyaudio"))
        # The offscreen rendering backend (MUJOCO_GL) is set by sensorium.bodies on import.
        # Imported here, after the stand-in and backend are set up, on purpose.
        from robopianist import suite  # noqa: PLC0415

        path = Path(str(settings["rp1m_path"]))
        self.rp1m = zarr.open_group(str(path), mode="r")
        all_songs = sorted(self.rp1m.group_keys())
        self.songs = list(settings.get("songs", all_songs))  # type: ignore[call-overload]
        self.camera = str(settings.get("camera", DEFAULT_CAMERA))
        self.settings = {"rp1m_path": str(path), "songs": self.songs, "camera": self.camera}

        self.env = suite.load(
            STAGE_SONG,
            seed=0,
            task_kwargs={"control_timestep": DT, "reduced_action_space": True},
        )
        self.env.reset()
        self.physics = self.env.physics
        task = self.env.task
        self.hands = (task.right_hand, task.left_hand)  # RP1M order: right, then left
        self.piano = task.piano
        spec = self.env.action_spec()
        self.action_low, self.action_high = spec.minimum, spec.maximum
        self.action_names = [clean_name(n) for n in spec.name.split("\t")]
        self.joint_names = [
            clean_name(j.full_identifier) for hand in self.hands for j in hand.joints
        ]
        self.touch_names = [
            f"{side}_{finger}" for side in ("right", "left") for finger in FINGERS
        ]  # RoboPianist fingertip order: thumb, index, middle, ring, little
        self.key_max = self.physics.bind(self.piano.joints).range[:, 1]
        self.default_bases = [self.physics.bind(h.root_body).pos.copy() for h in self.hands]

    def record(self, seed: int, policy: str, steps: int) -> Episode:
        """Re-render one RP1M trajectory, chosen by ``seed``, for ``steps`` steps.

        Raises:
            ValueError: unknown policy, or replayed fingertips deviate from the
                recording by more than ``MAX_FINGERTIP_ERROR``.
        """
        if policy not in self.policies:
            raise ValueError(f"piano policy must be one of {self.policies}, got {policy!r}")
        rng = np.random.default_rng(seed)
        song = self.songs[int(rng.integers(len(self.songs)))]
        group = self.rp1m[song]
        if not isinstance(group, zarr.Group):
            raise ValueError(f"RP1M entry {song!r} is not a group")
        n_trajectories, n_steps = rp1m_shape(group, "actions")[:2]
        trajectory = int(rng.integers(n_trajectories))
        steps = min(steps, n_steps)
        joints = read_rp1m(group, "hand_joints", trajectory, steps)
        tips_recorded = read_rp1m(group, "hand_fingertips", trajectory, steps)
        key_states = read_rp1m(group, "piano_states", trajectory, steps)[:, :88]  # drop sustain
        raw_actions = read_rp1m(group, "actions", trajectory, steps)
        base_offsets = self.calibrate_hand_bases(joints[0], tips_recorded[0])

        sound = PianoSound(rng, n_samples=round(DT * SAMPLE_RATE))
        key_down = key_states >= 1.0 - KEY_DOWN_RADIANS / self.key_max
        frames, chunks, touch, texts, tips = [], [], [], [], []
        started = time.time()
        for t in range(steps):
            self.set_pose(joints[t], key_states[t])
            tips.append(self.fingertips())
            frames.append(self.physics.render(IMAGE_SIZE, IMAGE_SIZE, camera_id=self.camera))
            chunks.append(sound.step(key_down[t - 1] if t > 0 else np.zeros(88, bool)))
            touch.append(self.touch())
            texts.append(describe_piano(self.who_presses(key_down[t], tips_recorded[t])))

        error = np.abs(np.asarray(tips) - tips_recorded).max()
        if error > MAX_FINGERTIP_ERROR:
            raise ValueError(f"{song}#{trajectory}: fingertip replay error {error:.4f} m")
        alignment = self.key_alignment_cm(key_down, tips_recorded)
        actions = 2 * (raw_actions - self.action_low) / (self.action_high - self.action_low) - 1
        audio = to_int16(chunks)
        return Episode(
            body="piano",
            dt=DT,
            rgb=np.stack(frames).astype(np.uint8),
            audio=audio,
            audio_logmel=logmel(audio),
            proprio=joints.astype(np.float32),
            touch=np.asarray(touch, dtype=np.float32),
            action=np.clip(actions, -1, 1).astype(np.float32),
            proprio_names=self.joint_names,
            touch_names=self.touch_names,
            action_names=self.action_names,
            privileged={
                "key_state": key_states.astype(np.float32),
                "key_down": key_down,
                "fingertips": tips_recorded.astype(np.float32),
            },
            text=texts,
            meta={
                "seed": seed,
                "policy": policy,
                "task": "rp1m",
                "instance": str(song),  # splits hold out RP1M songs by this id
                "settings": self.settings,
                "source": {"dataset": "RP1M", "song": song, "trajectory": trajectory},
                "fingertip_replay_error_m": round(float(error), 6),
                "hand_base_offsets_m": base_offsets.round(4).tolist(),  # [right, left] x (x, y, z)
                "pressed_key_to_fingertip_cm": alignment,
                "end_reason": "steps",
                "wall_seconds": round(time.time() - started, 2),
            },
        )

    def calibrate_hand_bases(self, joints: np.ndarray, tips_recorded: np.ndarray) -> np.ndarray:
        """Move each hand's base so its fingertips match the recording.

        Starts from RoboPianist's default base each time, so episodes are
        independent. Returns the ``[2, 3]`` offsets applied (right, left).
        """
        for hand, default in zip(self.hands, self.default_bases, strict=True):
            self.physics.bind(hand.root_body).pos = default
        self.set_pose(joints, np.zeros(88))
        offsets = (tips_recorded.reshape(2, 5, 3) - self.fingertips().reshape(2, 5, 3)).mean(axis=1)
        for hand, offset in zip(self.hands, offsets, strict=True):
            self.physics.bind(hand.root_body).pos += offset
        return np.asarray(offsets)

    def set_pose(self, joints: np.ndarray, key_state: np.ndarray) -> None:
        """Place both hands (46 joint angles) and all keys (0..1 depression)."""
        self.physics.bind(self.hands[0].joints).qpos = joints[:23]
        self.physics.bind(self.hands[1].joints).qpos = joints[23:]
        self.physics.bind(self.piano.joints).qpos = key_state * self.key_max
        self.physics.forward()

    def fingertips(self) -> np.ndarray:
        """``[30]`` fingertip positions: right hand then left, 5 tips x (x, y, z)."""
        return np.concatenate(
            [self.physics.bind(hand.fingertip_sites).xpos.ravel() for hand in self.hands]
        )

    def touch(self) -> np.ndarray:
        """``[10]`` fingertip touch-sensor readings (contact normal force)."""
        return np.concatenate(
            [self.physics.bind(hand.fingertip_touch_sensors).sensordata for hand in self.hands]
        )

    def key_alignment_cm(self, key_down: np.ndarray, tips: np.ndarray) -> float:
        """Median distance along the keyboard from each pressed key to the nearest
        fingertip, over the episode. Near 0 means fingers sit on the pressed keys."""
        key_y = self.physics.bind(self.piano.keys).xpos[:, 1]
        distances = [
            np.abs(tips[t].reshape(10, 3)[:, 1] - key_y[key]).min()
            for t, key in zip(*np.nonzero(key_down), strict=True)
        ]
        return round(100 * float(np.median(distances)), 2) if distances else float("nan")

    def who_presses(self, down: np.ndarray, tips: np.ndarray) -> list[tuple[str, str, int]]:
        """For each pressed key, the fingertip nearest to it along the keyboard."""
        key_y = self.physics.bind(self.piano.keys).xpos[:, 1]
        tip_y = tips.reshape(10, 3)[:, 1]
        pressed = []
        for key in np.flatnonzero(down):
            tip = int(np.argmin(np.abs(tip_y - key_y[key])))
            pressed.append(("right" if tip < 5 else "left", FINGERS[tip % 5], int(key)))
        return pressed

    def close(self) -> None:
        """Release the MuJoCo environment."""
        self.env.close()

    def __enter__(self) -> "PianoRecorder":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def rp1m_shape(group: zarr.Group, name: str) -> tuple[int, ...]:
    """Shape of one RP1M array: ``(trajectories, steps, dims)``.

    Raises:
        ValueError: if ``name`` is missing or not an array.
    """
    array = group.get(name)
    if not isinstance(array, zarr.Array):
        raise ValueError(f"RP1M group has no array {name!r}")
    return tuple(array.shape)


def read_rp1m(group: zarr.Group, name: str, trajectory: int, steps: int) -> np.ndarray:
    """First ``steps`` steps of one trajectory of one RP1M array, as ``[steps, dims]``.

    Raises:
        ValueError: if ``name`` is missing or not an array.
    """
    array = group.get(name)
    if not isinstance(array, zarr.Array):
        raise ValueError(f"RP1M group has no array {name!r}")
    return np.asarray(array[trajectory, :steps])


def clean_name(name: str) -> str:
    """Shorten RoboPianist names.

    ``"rh_shadow_handrh_A_WRJ2"`` -> ``"rh_A_WRJ2"``,
    ``"lh_shadow_handforearm_ty"`` -> ``"lh_forearm_ty"``.
    """
    for side in ("rh", "lh"):
        prefix = f"{side}_shadow_hand"
        if name.startswith(prefix):
            rest = name.removeprefix(prefix).lstrip("/")
            return rest if rest.startswith(f"{side}_") else f"{side}_{rest}"
    return name
