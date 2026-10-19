"""Free-space whole-body reaching for the standing H1: HumanoidBench's released policy in numpy.

``reach_free`` runs HumanoidBench's ``reach_two_hands`` policy (Sferrazza et al. 2024, MIT):
a 61 -> 256 -> 256 -> 19 tanh MLP, trained with PPO in MJX on a hands-free H1 in an empty
world, that drives the legs, torso and arms (19 position targets) so both hands reach two 3-D
targets. We reuse it unchanged, as a source of whole-body motion with near-zero contact:

* it must tick at its training rate of 50 Hz (physics 2 ms x 10), while episodes record at 20 Hz,
  so the recorder calls ``tick`` every ``substeps_per_tick`` physics steps and ``act`` only
  reports the target in force (the recorded action);
* it was trained on the stock gains and without the arm gravity feed-forward, so the recorder
  leaves both alone (``leg_gain_scale``, ``gravity_compensation``);
* wrists are fixed at 1.57 rad as in HumanoidBench's wrapper; the fingers are ours (smooth
  noise around open) because the policy does not control them;
* measured in our scene (``configs/reach_free.toml``, 20 episodes of 15 s per effector) it falls
  in 15-20% of them, the pelvis drifts a median 0.44 m and crouches to ~0.67 m; near a table it
  is unusable, hence free space only.

The weights are converted once by ``scripts/convert_reach_policy.py`` (no torch needed at
generation time); ``scripts/verify_reach_policy.py`` checks them against torch.
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from sensorium.bodies.h1_policies import SIDES, ActionSpace
from sensorium.bodies.h1_scene import external_dir
from sensorium.bodies.noise import SmoothNoise

TICK_SUBSTEPS = 10  # physics steps per policy tick: 50 Hz at the 2 ms timestep
WRIST_YAW = 1.57  # rad, fixed as in HumanoidBench's wrapper
SUCCESS_M = 0.1  # both hands this close to their targets: the next targets
TARGET_TICKS = 150  # ticks (3 s) before unreached targets are replaced
# Target box in front of the start position (m), inside the arms' range (the paper restricts targets
# to the robot workspace for its high-level policy).
TARGET_X, TARGET_Y, TARGET_Z = (0.25, 0.6), (0.05, 0.35), (0.8, 1.3)


@dataclass(frozen=True)
class ReachWeights:
    """The converted checkpoint: three dense layers and the observation statistics."""

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray
    mean: np.ndarray
    var: np.ndarray

    @classmethod
    def load(cls, path: Path | None = None) -> "ReachWeights":
        """Load ``reach_two_hands.npz`` (made by ``make h1-assets``).

        Raises:
            FileNotFoundError: if it is missing; the message says how to make it.
        """
        path = path or external_dir() / "humanoid-bench" / "reach_two_hands.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}; run `make h1-assets`")
        with np.load(path) as archive:
            return cls(**{name: archive[name] for name in cls.__dataclass_fields__})

    def action(self, observation: np.ndarray) -> np.ndarray:
        """The policy's 19 actions in ``[-1, 1]`` for a 61-dimensional observation."""
        x = (observation - self.mean) / np.sqrt(self.var + 1e-8)
        x = np.tanh(self.w1 @ x + self.b1)
        x = np.tanh(self.w2 @ x + self.b2)
        return np.asarray(np.clip(self.w3 @ x + self.b3, -1.0, 1.0))


def weights_available() -> bool:
    """True if the converted reach policy exists (``make h1-assets`` has been run)."""
    return (external_dir() / "humanoid-bench" / "reach_two_hands.npz").is_file()


class ReachFreePolicy:
    """Both hands reach random 3-D targets in front of the standing robot, whole body.

    ``targets_reached`` counts the target pairs reached (both hands within ``SUCCESS_M``).
    """

    substeps_per_tick = TICK_SUBSTEPS
    leg_gain_scale = 1.0  # the policy was trained on the stock gains
    gravity_compensation = False  # ... and without the arm feed-forward

    def __init__(
        self,
        model: mujoco.MjModel,
        space: ActionSpace,
        rng: np.random.Generator,
        weights: ReachWeights | None = None,
    ) -> None:
        self.model, self.space, self.rng = model, space, rng
        self.weights = weights or ReachWeights.load()
        self.body_q, self.body_v = self.state_indices(model)
        self.driven = list(range(15)) + list(range(16, 20))  # the policy's 19 actuators
        self.wrist = [space.index[f"{side}_wrist_yaw"] for side in SIDES]
        self.sites = [model.site(f"{side}_hand").id for side in SIDES]
        self.fingers = np.array([i for side in SIDES for i in space.effector[side]])
        self.home = space.home("standing")
        open_fingers = space.to_action(self.home)[self.fingers]  # fingers open: the noise's mean
        self.finger_noise = SmoothNoise(rng, len(self.fingers), 0.3, theta=0.03, mean=open_fingers)
        self.ctrl = self.home.copy()
        self.targets = np.zeros((2, 3))
        self.ticks_since_target = 0
        self.targets_reached = 0
        self.origin = np.zeros(3)

    @staticmethod
    def state_indices(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
        """Indices of the body state in ``qpos`` and ``qvel``: the base and every non-hand joint.

        As in HumanoidBench: the free base (7 and 6), then each joint that is not a finger or
        wrist joint.
        """
        body_q: list[int] = []
        body_v: list[int] = []
        for joint in range(model.njnt):
            name = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint))
            q, v = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
            if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
                if name == "free_base":
                    body_q += list(range(q, q + 7))
                    body_v += list(range(v, v + 6))
            elif not name.startswith(("lh_", "rh_", "lp_", "rp_")) and "wrist" not in name:
                body_q.append(q)
                body_v.append(v)
        return np.array(body_q), np.array(body_v)

    def reset(self, data: mujoco.MjData) -> np.ndarray:
        """Home targets (arms down, as in HumanoidBench's reset) and the first targets."""
        self.home = self.space.home("standing")
        for side in SIDES:  # the policy starts from straight arms, not our "ready" pose
            for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow"):
                self.home[self.space.index[f"{side}_{joint}"]] = 0.0
            self.home[self.space.index[f"{side}_wrist_yaw"]] = WRIST_YAW
        self.ctrl = np.clip(self.home, self.space.low, self.space.high)
        self.targets_reached = 0
        self.origin = data.xpos[self.model.body("pelvis").id].copy()
        self.new_targets(data)
        return self.ctrl.copy()

    def new_targets(self, data: mujoco.MjData) -> None:
        """Draw a target for each hand in front of the start position (left on +y, right on -y)."""
        origin = self.origin  # fixed in the world, so the robot stays near where it started
        for i, sign in enumerate((1.0, -1.0)):
            offset = np.array(
                [
                    self.rng.uniform(*TARGET_X),
                    sign * self.rng.uniform(*TARGET_Y),
                    0.0,
                ]
            )
            self.targets[i] = [
                origin[0] + offset[0],
                origin[1] + offset[1],
                self.rng.uniform(*TARGET_Z),
            ]
        self.ticks_since_target = 0

    def observation(self, data: mujoco.MjData) -> np.ndarray:
        """HumanoidBench's 61 observations: body state, hands and targets (pelvis-relative)."""
        offset = np.array([data.qpos[0], data.qpos[1], 0.0])
        position = data.qpos[self.body_q].copy()
        position[:3] -= offset
        hands = [data.site_xpos[s] - offset for s in self.sites]
        targets = [t - offset for t in self.targets]
        return np.concatenate([position[2:], data.qvel[self.body_v], *hands, *targets])

    def tick(self, data: mujoco.MjData) -> np.ndarray:
        """One 50 Hz policy step: the new actuator targets for the whole body."""
        errors = [
            np.linalg.norm(data.site_xpos[s] - t)
            for s, t in zip(self.sites, self.targets, strict=True)
        ]
        self.ticks_since_target += 1
        if max(errors) < SUCCESS_M or self.ticks_since_target > TARGET_TICKS:
            self.targets_reached += int(max(errors) < SUCCESS_M)
            self.new_targets(data)
        action = self.weights.action(self.observation(data))
        low, high = self.space.low[self.driven], self.space.high[self.driven]
        ctrl = self.ctrl.copy()
        ctrl[self.driven] = (action + 1) / 2 * (high - low) + low
        for wrist in self.wrist:
            ctrl[wrist] = WRIST_YAW
        action_fingers = np.clip(self.finger_noise.sample(), -1.0, 1.0)
        low, high = self.space.low[self.fingers], self.space.high[self.fingers]
        ctrl[self.fingers] = low + (action_fingers + 1) / 2 * (high - low)
        self.ctrl = np.clip(ctrl, self.space.low, self.space.high)
        return self.ctrl.copy()

    def act(self, step: int, data: mujoco.MjData) -> np.ndarray:
        """The targets in force (the recorded action); the ticks update them inside the step."""
        return self.space.to_action(self.ctrl)
