"""Car body: MetaDrive, procedural roads with traffic.

Senses: ego RGB camera, synthesised engine/wind/tyre sound, proprioception
(speed, yaw rate, acceleration, lateral slip). No touch.
Actions: ``[steering, throttle_brake]`` in [-1, 1].
Control: 10 Hz (MetaDrive: 0.02 s physics x 5 repeats).

Policies:
    expert   MetaDrive's bundled PPO driving expert.
    noisy    expert + smooth random perturbations (sigma 0.15).
    explore  smooth random steering and throttle, no expert.
"""

import time
from types import TracebackType

import numpy as np
from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.examples import expert

from sensorium.audio import CarSound, logmel, to_int16
from sensorium.bodies.noise import SmoothNoise
from sensorium.describe import describe_car
from sensorium.episode import IMAGE_SIZE, SAMPLE_RATE, Episode

DT = 0.1


class CarRecorder:
    """Records car episodes. Reuses one MetaDrive engine (one per process).

    Settings (all optional): ``num_scenarios`` (int, 1000) distinct road maps,
    ``traffic_density`` (float, 0.1).
    """

    policies: tuple[str, ...] = ("expert", "noisy", "explore")

    def __init__(self, settings: dict[str, object]) -> None:
        self.num_scenarios = int(settings.get("num_scenarios", 1000))  # type: ignore[call-overload]
        self.env = MetaDriveEnv(
            {
                "num_scenarios": self.num_scenarios,
                "start_seed": 0,
                "traffic_density": float(settings.get("traffic_density", 0.1)),  # type: ignore[arg-type]
                "image_observation": True,
                "norm_pixel": False,
                "stack_size": 1,
                "vehicle_config": {"image_source": "rgb_camera"},
                "sensors": {"rgb_camera": (RGBCamera, IMAGE_SIZE, IMAGE_SIZE)},
                "use_render": False,
                "show_interface": False,
                "show_logo": False,
                "show_fps": False,
                "log_level": 50,
                # Keep driving after mistakes: the world model must see consequences.
                "out_of_road_done": False,
                "crash_vehicle_done": False,
                "crash_object_done": False,
            }
        )
        self.settings = {"num_scenarios": self.num_scenarios, **settings}

    def record(self, seed: int, policy: str, steps: int) -> Episode:
        """Drive one episode of ``steps`` steps; stops early only on arrival.

        Crashes and leaving the road do not end the episode.
        """
        if policy not in self.policies:
            raise ValueError(f"car policy must be one of {self.policies}, got {policy!r}")
        rng = np.random.default_rng(seed)
        noise = SmoothNoise(rng, dim=2, scale=0.15)
        explore = SmoothNoise(rng, dim=2, scale=0.5, theta=0.05, mean=np.array([0.0, 0.3]))
        sound = CarSound(rng, n_samples=round(DT * SAMPLE_RATE))

        obs, info = self.env.reset(seed=seed % self.num_scenarios)
        car = self.env.agent
        frames, chunks, proprio, actions, texts = [], [], [], [], []
        positions, headings, crashes = [], [], []
        prev_speed, prev_heading, last_throttle = car.speed, car.heading_theta, 0.0
        started = time.time()
        for _ in range(steps):
            # 1. Sense: everything observed *before* this step's action.
            speed = float(car.speed)  # m/s
            yaw_rate = wrap_angle(car.heading_theta - prev_heading) / DT
            accel = (speed - prev_speed) / DT
            vx, vy = car.velocity
            # Sideways speed: velocity component perpendicular to the heading (slip).
            lateral = float(np.cos(car.heading_theta) * vy - np.sin(car.heading_theta) * vx)
            crashed = bool(info.get("crash_vehicle") or info.get("crash_object"))
            prev_speed, prev_heading = speed, car.heading_theta

            # Last frame of the 1-frame stack. MetaDrive returns BGR: flip to RGB.
            frames.append(obs["image"][..., ::-1, -1])
            chunks.append(sound.step(speed, last_throttle, slip=min(abs(lateral) / 3, 1.0)))
            proprio.append([speed, yaw_rate, accel, lateral])
            texts.append(describe_car(speed, yaw_rate, accel, crashed))
            positions.append(np.asarray(car.position, dtype=np.float32))
            headings.append(car.heading_theta)
            crashes.append(crashed)

            # 2. Act.
            if policy == "explore":
                action = explore.sample()
            else:
                action = np.asarray(expert(car, deterministic=True))
                if policy == "noisy":
                    action = action + noise.sample()
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
            actions.append(action)
            last_throttle = float(action[1])
            obs, _, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                break

        audio = to_int16(chunks)
        return Episode(
            body="car",
            dt=DT,
            rgb=np.stack(frames).astype(np.uint8),
            audio=audio,
            audio_logmel=logmel(audio),
            proprio=np.asarray(proprio, dtype=np.float32),
            touch=np.zeros((len(frames), 0), dtype=np.float32),
            action=np.stack(actions),
            proprio_names=["speed_mps", "yaw_rate", "accel", "lateral_speed"],
            touch_names=[],
            action_names=["steering", "throttle_brake"],
            privileged={
                "position": np.stack(positions),
                "heading": np.asarray(headings, dtype=np.float32),
                "crashed": np.asarray(crashes),
            },
            text=texts,
            meta={
                "seed": seed,
                "policy": policy,
                "task": "drive",
                "instance": f"map-{seed}",  # the procedural map is a function of the seed
                "settings": self.settings,
                "end_reason": end_reason(info, len(frames), steps),
                "wall_seconds": round(time.time() - started, 2),
            },
        )

    def close(self) -> None:
        """Shut down the MetaDrive engine."""
        self.env.close()

    def __enter__(self) -> "CarRecorder":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def wrap_angle(angle: float) -> float:
    """Wrap an angle difference into [-pi, pi)."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def end_reason(info: dict[str, object], length: int, steps: int) -> str:
    """Why the episode ended: ``"steps" | "arrived" | "crash" | "off_road"``."""
    if length >= steps:
        return "steps"
    if info.get("arrive_dest"):
        return "arrived"
    if info.get("out_of_road"):
        return "off_road"
    return "crash"
