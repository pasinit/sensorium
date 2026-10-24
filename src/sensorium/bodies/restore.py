"""Restore a recorded H1 episode's scene and physical state, then step it with new actions.

Planning tests (``docs/model.md``, B-plan-0) start the simulator at a moment ``t0`` of a recorded
episode and let a planner act from there. ``RestoredScene`` rebuilds the episode's scene from its
generator settings and seed exactly as ``HumanoidRecorder.record`` does (appearance, props, hidden
object properties, object layout), then overwrites the physical state with the recorded
``qpos`` / ``qvel`` at ``t0`` and the joint targets of the action taken just before it. ``step``
applies one action like the recorder's loop and returns the observation that follows it, so
consecutive steps form an ``Episode`` the model can encode.

Restoring ``qpos`` / ``qvel`` does not restore the solver's warm start or the touch sensors'
history, so a restored run may drift from the recording; ``replay`` measures by how much (the
validity gate of B-plan-0).
"""

import time
from pathlib import Path

import mujoco
import numpy as np

from sensorium.audio import ContactSound
from sensorium.bodies.h1_looks import apply_appearance, draw_appearance
from sensorium.bodies.h1_skin import read_contacts
from sensorium.bodies.humanoid import DT, HumanoidRecorder, Log
from sensorium.episode import SAMPLE_RATE, Episode
from sensorium.model.planning import ARM_JOINTS


class RestoredScene:
    """One H1 scene (an effector and posture), restorable at any moment of its episodes.

    Args:
        settings: The generator settings of the episode's block (``settings`` of the config).
    """

    def __init__(self, settings: dict[str, object]) -> None:
        self.recorder = HumanoidRecorder(settings)
        self.log = Log()
        self.sound: ContactSound | None = None
        self.policy = ""
        self.seed = 0
        self.hold = np.zeros(0)
        self.t = 0

    def restore(self, episode: Episode, t0: int) -> None:
        """Rebuild ``episode``'s scene and put the simulator in its recorded state at ``t0``.

        Raises:
            KeyError: if the episode lacks ``meta["seed"]`` or ``privileged/qpos`` / ``qvel``.
            ValueError: if ``t0`` is 0 (there is no previous action to hold) or out of range.
        """
        if not 0 < t0 < episode.length:
            raise ValueError(f"t0 must be in [1, {episode.length - 1}], got {t0}")
        rec = self.recorder
        self.seed = int(episode.meta["seed"])  # type: ignore[call-overload]
        self.policy = str(episode.meta["policy"])
        streams = np.random.SeedSequence(self.seed).spawn(7)  # as HumanoidRecorder.record
        layout_rng, policy_rng, sound_rng, look_rng, rec.props_rng, rec.effort_rng = (
            np.random.default_rng(child) for child in streams[:6]
        )
        rec.director_rng = np.random.default_rng(streams[6])
        rec.appearance = draw_appearance(look_rng)
        apply_appearance(rec.model, rec.appearance)
        actor = rec.make_policy(self.policy, policy_rng)
        rec.reset(actor, layout_rng)
        rec.peaks.clear()
        self.sound = ContactSound(sound_rng, round(DT * SAMPLE_RATE))
        data = rec.data
        data.qpos[:] = episode.privileged["qpos"][t0]
        data.qvel[:] = episode.privileged["qvel"][t0]
        self.hold = np.asarray(episode.action[t0 - 1], dtype=np.float64)
        self.t = t0
        mujoco.mj_forward(rec.model, data)  # gravity compensation reads the restored state
        self._set_targets(self.hold)
        rec.monitor.reset(data)
        rec.dynamics.reset(data)
        self.log = Log()

    def _set_targets(self, action: np.ndarray) -> None:
        rec = self.recorder
        ctrl = rec.space.to_ctrl(action)
        rec.data.ctrl[:] = rec.with_gravity_compensation(ctrl)

    def step(self, action: np.ndarray) -> None:
        """Apply one action for one control step, then observe (appended to the log)."""
        rec, data = self.recorder, self.recorder.data
        if rec.director is not None:  # as the recorder's loop (scenes with a director only)
            rec.director.step(data, self.t)
        mujoco.mj_forward(rec.model, data)
        self._set_targets(np.asarray(action, dtype=np.float64))
        for _ in range(rec.substeps):
            mujoco.mj_step(rec.model, data)
            contacts_now = read_contacts(rec.model, data)
            rec.dynamics.substep(data, contacts_now)
            rec.peaks.update(contacts_now)
        mujoco.mj_forward(rec.model, data)
        contacts = rec.monitor.read(data)
        gravity = data.xmat[rec.pelvis].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        assert self.sound is not None
        rec.observe(self.log, contacts, gravity, False, np.asarray(action), self.sound)
        self.t += 1

    def observed(self) -> Episode:
        """The steps observed since ``restore`` (or the last ``observed``), as an episode; the
        log is then cleared."""
        rec = self.recorder
        episode = rec.episode(self.log, self.seed, self.policy, "steps", time.time())
        self.log = Log()
        return episode

    def arm_q(self) -> np.ndarray:
        """Current arm-joint positions (``ARM_JOINTS``, radians)."""
        rec = self.recorder
        names = [f"q_{rec.joint_name(j)}" for j in rec.joints]
        joints = [rec.joints[names.index(n)] for n in ARM_JOINTS]
        return np.array([rec.data.qpos[rec.model.jnt_qposadr[j]] for j in joints])


def block_settings(config: Path, block: str) -> dict[str, object]:
    """The generator settings of one block of a config (``[humanoid.<block>] settings``)."""
    import tomllib  # noqa: PLC0415

    raw = tomllib.loads(Path(config).read_text())
    return dict(raw["humanoid"][block]["settings"])
