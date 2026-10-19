"""Drone body: gym-pybullet-drones Crazyflie (CF2X) among obstacles.

Senses: forward-facing onboard RGB camera, synthesised rotor sound,
proprioception like an IMU + barometer (roll/pitch/yaw, angular velocity,
velocity, altitude). No touch.
Actions: four motor speeds, normalised to [-1, 1] (-1 = 0 RPM, 1 = max RPM).
Control: 40 Hz (240 Hz physics, 6 substeps).

Policies (all fly a random waypoint course by tracking a smooth reference):
    expert   PID tracks the reference at 0.4-1.0 m/s, facing where it goes.
    noisy    expert + smooth random motor perturbations.
    explore  faster (1-2 m/s), random yawing, light motor noise: erratic flight.
"""

import time
from types import TracebackType

import numpy as np
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel

from sensorium.audio import DroneSound, logmel, to_int16
from sensorium.bodies.noise import SmoothNoise
from sensorium.describe import describe_drone
from sensorium.episode import IMAGE_SIZE, SAMPLE_RATE, Episode

PHYSICS_HZ, CONTROL_HZ = 240, 40
DT = 1.0 / CONTROL_HZ
ARENA_LOW = np.array([-1.5, -2.0, 0.4])  # waypoint box (m)
ARENA_HIGH = np.array([2.5, 2.0, 1.8])


class DroneRecorder:
    """Records drone episodes. Reuses one PyBullet simulation.

    Settings (all optional): ``waypoints`` (int, 6) per course.
    """

    policies: tuple[str, ...] = ("expert", "noisy", "explore")

    def __init__(self, settings: dict[str, object]) -> None:
        self.env = CtrlAviary(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=np.array([[0.0, 0.0, 0.3]]),
            pyb_freq=PHYSICS_HZ,
            ctrl_freq=CONTROL_HZ,
            gui=False,
            obstacles=True,
            user_debug_gui=False,
        )
        self.env.IMG_RES = np.array([IMAGE_SIZE, IMAGE_SIZE])  # enables _getDroneImages
        self.controller = DSLPIDControl(drone_model=DroneModel.CF2X)
        self.max_rpm = float(self.env.MAX_RPM)
        self.n_waypoints = int(settings.get("waypoints", 6))  # type: ignore[call-overload]
        self.settings = {"waypoints": self.n_waypoints, **settings}

    def record(self, seed: int, policy: str, steps: int) -> Episode:
        """Fly one episode; stops early if the drone crashes."""
        if policy not in self.policies:
            raise ValueError(f"drone policy must be one of {self.policies}, got {policy!r}")
        rng = np.random.default_rng(seed)
        waypoints = rng.uniform(ARENA_LOW, ARENA_HIGH, size=(self.n_waypoints, 3))
        explore = policy == "explore"
        cruise_speed = rng.uniform(1.0, 2.0) if explore else rng.uniform(0.4, 1.0)
        motor_noise = SmoothNoise(rng, dim=4, scale=0.02 if policy == "noisy" else 0.01, theta=0.3)
        yaw_noise = SmoothNoise(rng, dim=1, scale=1.5, theta=0.02)
        sound = DroneSound(rng, n_samples=round(DT * SAMPLE_RATE))

        obs, _ = self.env.reset(seed=seed)
        self.controller.reset()
        state = obs[0]
        reference, target_index = state[:3].copy(), 0
        frames, chunks, proprio, actions, texts = [], [], [], [], []
        positions, quats, references, crashes = [], [], [], []
        last_rpm = np.zeros(4)
        started, crashed = time.time(), False
        for step in range(steps):
            # 1. Sense: everything observed *before* this step's action.
            pos, quat, rpy, vel, ang_vel = (
                state[0:3],
                state[3:7],
                state[7:10],
                state[10:13],
                state[13:16],
            )
            crashed = step > 10 and (pos[2] < 0.05 or max(abs(rpy[0]), abs(rpy[1])) > 1.5)
            rgb, _, _ = self.env._getDroneImages(0, segmentation=False)
            frames.append(np.asarray(rgb[..., :3], dtype=np.uint8))
            chunks.append(sound.step(last_rpm))
            proprio.append(np.concatenate([rpy, ang_vel, vel, pos[2:3]]))
            texts.append(describe_drone(vel, float(rpy[0]), float(pos[2]), crashed))
            positions.append(pos.copy())
            quats.append(quat.copy())
            references.append(reference.copy())
            crashes.append(crashed)

            # 2. Act: advance the smooth reference along the course, then track it.
            to_go = waypoints[target_index] - reference
            distance = float(np.linalg.norm(to_go))
            if distance < 0.05:
                target_index = (target_index + 1) % len(waypoints)
            else:
                reference = reference + to_go / distance * min(distance, cruise_speed * DT)
            yaw = np.arctan2(to_go[1], to_go[0]) + (yaw_noise.sample()[0] if explore else 0.0)
            rpm, _, _ = self.controller.computeControlFromState(
                control_timestep=DT,
                state=state,
                target_pos=reference,
                target_rpy=np.array([0.0, 0.0, yaw]),
            )
            action = 2.0 * np.asarray(rpm) / self.max_rpm - 1.0
            if policy != "expert":
                action = action + motor_noise.sample()
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
            actions.append(action)
            last_rpm = (action + 1.0) / 2.0 * self.max_rpm
            obs, _, _, _, _ = self.env.step(last_rpm.reshape(1, 4))
            state = obs[0]
            if crashed:
                break

        audio = to_int16(chunks)
        return Episode(
            body="drone",
            dt=DT,
            rgb=np.stack(frames),
            audio=audio,
            audio_logmel=logmel(audio),
            proprio=np.asarray(proprio, dtype=np.float32),
            touch=np.zeros((len(frames), 0), dtype=np.float32),
            action=np.stack(actions),
            proprio_names=[
                "roll",
                "pitch",
                "yaw",
                "roll_rate",
                "pitch_rate",
                "yaw_rate",
                "vel_x",
                "vel_y",
                "vel_z",
                "altitude",
            ],  # fmt: skip
            touch_names=[],
            action_names=["motor_0", "motor_1", "motor_2", "motor_3"],
            privileged={
                "position": np.asarray(positions, dtype=np.float32),
                "quaternion": np.asarray(quats, dtype=np.float32),
                "reference": np.asarray(references, dtype=np.float32),
                "crashed": np.asarray(crashes),
            },
            text=texts,
            meta={
                "seed": seed,
                "policy": policy,
                "task": "waypoints",
                "instance": f"course-{seed}",  # the waypoint course is a function of the seed
                "settings": self.settings,
                "cruise_speed": round(float(cruise_speed), 3),
                "end_reason": "crash" if crashed else "steps",
                "wall_seconds": round(time.time() - started, 2),
            },
        )

    def close(self) -> None:
        """Disconnect the PyBullet simulation."""
        self.env.close()

    def __enter__(self) -> "DroneRecorder":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
