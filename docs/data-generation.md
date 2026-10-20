# Data generation

What we generate, how, and what has to be done by hand.

## Run it

```sh
make setup        # uv env, Python 3.13, simulators pinned to exact commits/versions
make rp1m-toy     # RP1M toy subset (351 MB), SHA-256 verified
make smoke        # 1 tiny episode per body, end to end (~1 min)
make sample       # review sample from configs/sample.toml (~1 min on an M1 laptop)
make browse       # data/sample/index.html: watch, listen, inspect every episode
make h1-assets    # H1 humanoid MJCF assets at pinned commits (HumanoidBench ~110 MB, Menagerie 7 MB)
make sample-h1    # 8 humanoid episodes (hands and pincers) + data/sample_h1/index.html (~45 s)
```

Everything runs on CPU. No GPU is involved until training.

## Episode format (`src/sensorium/episode.py`)

One body, one episode, one compressed `.npz`. Arrays indexed by time share `T`.

| Key | Shape / type | Model input? | Notes |
|---|---|---|---|
| `rgb` | `[T, 256, 256, 3]` uint8 | ✅ | stored as H.264 video inside the file (`rgb.mp4`, lossy, ~2.5 KB/frame); cropped and resized to the model's size at load time, see [vision.md](vision.md) |
| `audio_logmel` | `[T, 32]` float16 | ✅ | precomputed on CPU, one vector per step |
| `proprio` | `[T, P]` float32 + names | ✅ | car 4, drone 10, piano 46, humanoid 148 (hands) / 84 (pincers) |
| `touch` | `[T, C]` float32 + names | ✅ | piano 10 fingertips; humanoid 2,775 (hands) / 946 (pincers): body grid + dense skin + dynamics, see [touch.md](touch.md); `[T, 0]` for car and drone |
| `action` | `[T, A]` float32 + names | ✅ | in [-1, 1]; car 2, drone 4, piano 39, humanoid 61 (hands) / 23 (pincers) |
| `audio` | `[T, dt·16000]` int16 | ❌ (review only) | raw waveform, for listening |
| `privileged/*` | `[T, …]` | ❌ **never** | ground truth: positions, key states, crash flags, … |
| `text` | `[T]` str | ❌ probe only | templated description per step |
| header | JSON | — | body, `dt`, channel names, `meta` (seed, policy, settings, source, quality checks) |

- **Timing:** at step `t` every sense is observed *before* `action[t]`.
  `audio[t]` is the sound of the interval leading up to `t`. So
  `(senses[t], action[t]) → senses[t+1]` is one transition.
- **No leaks:** `MODEL_INPUT_KEYS` is the allow-list, and `model_inputs()` is
  the only way to build a batch. A test enforces this.
- **Validation:** every save and load checks shapes, dtypes, action range
  and finiteness.

## Reproducibility

- **Seeds:** each episode's seed is `sha256(master_seed/body/policy/index)`
  (plus the task name when it is not the default), so any episode can be
  regenerated on its own.
- **Manifest:** `manifest.json` stores the config and its hash, git commit +
  dirty flag, Python, platform and package versions. Per episode it stores the
  seed, length, end reason and a **content hash** of all arrays (independent of
  zip metadata).
- **Verified:** regenerating `configs/smoke.toml` gives identical content
  hashes for all three bodies, and `configs/sample_h1.toml` (8 humanoid
  episodes, fresh processes) for all 8, on the same machine. Rendering can differ across
  GPU drivers and platforms, so compare hashes within one platform.
- **Pinned inputs:** simulators are pinned to git SHAs/versions in
  `pyproject.toml` + `uv.lock`, and RP1M is SHA-256 checked.

## Bodies

The humanoid (the H1, `bodies/humanoid.py`, D0.2) is the project's body. The
car and drone stay as the alien-body ablation, and the piano hands move onto
the H1 arms in D0.5.

| | **Humanoid H1** (the body) | Car (ablation) | Drone (ablation) | Piano hands (hand side) |
|---|---|---|---|---|
| Simulator | MuJoCo 3.15.0; HumanoidBench @cb11890 and Menagerie robotiq_2f85 @0059d43 (MJCF assets only) | MetaDrive @325d9e2 | gym-pybullet-drones @7ebad1e (PyBullet 3.2.7) | RoboPianist 1.0.10 (MuJoCo 3.15) |
| Control | 20 Hz; PD position targets on MuJoCo position actuators (physics 500 Hz) | 10 Hz | 40 Hz | 20 Hz |
| Camera | left head camera, tilted 45° down, 80° field of view, in a room recoloured and relit every episode; right eye in `privileged/rgb_right` | ego, forward | onboard, forward | top-down over hands and keys |
| Policies | `hold` (stand or sit, smooth noise on the arms), `reach_free` (standing only: HumanoidBench's two-hand reach policy for whole-body free-space reaching, 50 Hz ticks, stock gains, soft arms like the others; episodes end on a fall), `seated_babble` (seated or `stand_supported`; each hand repeats: pick a primitive by counts, fewest tried first: hover, touch, poke an object's side, slide over the table, grasp-lift-drop (pincer) or power closure (hand); plan it as clear, speed-capped waypoints; withdraw on a limb contact above 80 N; `primitives`, `table_top` (default 0.82 m seated, 1.16 m supported), `reach_x`, `arm_solref` (contact softness of the arms, 0.04 s) and `safe_reach = false` for the D0.2 policy are settings) | `expert` (bundled PPO), `noisy` (+σ 0.15), `explore` (smooth random) | `expert` (PID on smooth waypoint course), `noisy` (+motor noise), `explore` (1–2 m/s, random yaw) | `expert` (RP1M) |
| Episode end | step budget, or a fall (standing) | step budget (crashes/off-road continue) | step budget, or crash | step budget |
| Generation speed (M1 Pro, 1 core) | ~21 steps/s seated with hands (was ~27 before the room; two 256² renders cost ~26 ms per step, video encoding ~8 ms), see [vision.md](vision.md#costs) | ~35 steps/s at 256 px (~50 at 64) | ~22 steps/s at 256 px (~60 at 64; PyBullet renders in software) | ~55 steps/s at 256 px (~70 at 64) |

### H1 body (`bodies/humanoid.py`)

One Unitree H1 (HumanoidBench's `h1touch` model, MIT) with two end effectors
and two postures. `make h1-assets` fetches only the MJCF and meshes at pinned
commits into `data/external/` (no `humanoid-bench` Python package, so no
gymnasium/dm_control pins); `bodies/h1_scene.py` assembles one scene XML per
(effector, posture) from them. The `touch_grid` sensor plugin is built into
the MuJoCo wheel and loads on its own (no `mj_loadPluginLibrary` call);
MuJoCo is pinned to 3.15.0 in `pyproject.toml`.

- **Skin** (full design in [touch.md](touch.md)). The body keeps
  HumanoidBench's `touch_grid` patches: 20 patches, 184 taxels (19 × 8 and a
  32-taxel torso), normal force per taxel, named `body/<link>/<k>` (H1's link
  names: `left_hip_pitch` is the thigh, `left_knee` the shank, `left_ankle`
  the foot). The hands (17 bodies each) and the pincer pads get **dense
  skin**: sensor points over the collision surface, each contact's force
  spread to nearby points (MIMo's linear spread), 2,429 points with hands
  and 690 with pincers, named `hand/left/ffdistal/<k>` and
  `pincer/left/left_pad/<k>`. Each pincer pad's face is a grid of small cubes
  so it reports several contacts. Per skin region, three **dynamics**
  channels come from every physics step: `dynamics/<region>/onset`, `slip`,
  `vibration`. `privileged/touch_xyz` keeps the 3-D forces of every
  `touch_grid` patch (HumanoidBench's 448-taxel layout with hands, 184 with
  pincers), named by `meta.touch_xyz_names`. The two variants share the body
  skin and differ only at the wrists.
- **Eyes** (full design in [vision.md](vision.md)). HumanoidBench's two eye
  cameras look horizontally with a 45° field of view, mostly horizon. We tilt
  both down 45° and widen them to 80° so the hands and the table are in view,
  and render 256 × 256. The scene is a room (checkered floor, four walls, one
  key light) whose colours and lighting are redrawn every episode
  (`h1_looks.py`, stored in `meta.appearance`). **v0 is monocular:** the left eye is `rgb`; the right eye is stored as
  `privileged/rgb_right` and never reaches the model (stereo is a later
  ablation). The `touch_grid` plugin draws a coloured patch on every active
  taxel; these decorations are hidden so the pixels do not copy the touch signal.
- **Proprioception** (148 / 84 channels): joint angles and velocities of every
  body and effector joint (`q_*`, `dq_*`), the pelvis IMU (`vest_gyro_x..z`,
  `vest_acc_x..z`), `pelvis_height` and the gravity direction in the pelvis
  frame (`gravity_x..z`).
- **Actions.** Position targets for all actuators (61 with hands: 21 body,
  2 × 20 hand; 23 with pincers: 21 body, one gripper each), normalised to
  [-1, 1] over each actuator's control range. MuJoCo's position actuators are the
  PD controllers (HumanoidBench's gains). Two controller details, both in
  `h1_policies.py` and `humanoid.py`: the arms get a gravity feed-forward
  (`qfrc_bias / kp` added to the target; with the stock `kp = 100` an extended
  arm otherwise sags ~10 cm), and while *standing* the legs' gains are ×10
  because the stock gains cannot hold a static stance (the stance is an
  inverted pendulum with gravity stiffness ≈ 470 N m/rad against two ankles of
  40 N m/rad). The released HumanoidBench reach policy (D0.3) runs on the stock
  model and must tick at its training rate of 50 Hz (at 20 Hz it fell within
  3 s in our scene).
- **Postures.** *Standing*: a floor with three static blocks; at rest the arms hang along the sides (`ARM_REST`: elbow 1.2 rad, shoulder roll 0.25 rad; on this H1 model elbow 0 points the forearm forward, so the published all-zero keyframe is arms out in front, and `reach_free` starts from it) and `HoldPolicy` leans the hips forward by 0.1 rad to keep the balance (`REST_HIP_PITCH`): the fingertips are beside the thighs, 0.1-0.2 m in front of the pelvis and 0.2-0.4 m below it, at the bottom edge of the image, touching nothing (0 falls in 8 hand and 8 pincer episodes of 300 steps). *Supported standing*
  (`stand_supported`): the seated scene with the pelvis welded at standing height
  (0.98 m), the legs straight and the table raised to 1.16 m, so every arm movement of the
  seated babbling transfers; no seat; falls cannot happen; recorded as its own source,
  `stand_table` (split v2). *Seated*: the
  pelvis is welded to a mocap anchor at seat height, the legs are held in a
  sitting pose, and a table with a rim holds six free objects (wood, plastic,
  metal, rubber; boxes, spheres, a cylinder) placed at random per episode, within 0.30 m of the
  middle so that the babbling stays in the head camera's view ([touch.md](touch.md#the-a1-gate-and-the-head-camera-framing-d03-step-4)).
  Both scenes use elliptic friction cones with `impratio` 10, as
  Menagerie recommends for the 2F-85's linkage; the hand scene too since the hand grasp work
  (with the pyramidal default, spheres rolled out of a closed pinch; [touch.md](touch.md#hand-grasping-the-claw-pinch)).
- **Sound** (`ContactSound`): per step, every pair of bodies in contact is
  classified by two *voices*: the robot part (feet, legs, torso, arms, hands)
  and the surface (floor, table, wood, plastic, metal, rubber). A pair that was
  not in contact at the previous step is an *impact*: a three-mode damped
  ring on both voices, loudness rising with the contact force. Sliding
  (tangential speed above 2 cm/s, force-weighted) adds band-limited noise
  scaled by speed and √force. Mode frequencies, decay times and gains per
  voice are drawn from the episode seed, plus a faint room tone, so the same
  hand sounds different on wood and on metal. Mesh contacts are summed per
  body pair, so a foot at rest is silent (standing episodes have RMS ≈ 25 of
  32 767; seated babbling ≈ 1 000–1 500).
- **Text** (`describe_humanoid`): posture, then which parts touch what
  (feet on the floor are left out), or `falling`:
  *"seated; left hand touching metal object, right arm touching table"*.
- **Privileged:** full `qpos`/`qvel`, the 16 strongest contacts per step (body
  ids, force, position; `meta.body_names` names the ids), the fall flag,
  object poses, fingertip (or pinch-point) positions, `touch_xyz`, `rgb_right`.
- **Meta:** seed, policy, settings, `taxels` (force channels),
  `touch_channels` (grid / dense / dynamics), `touch_xyz_names`, `touch_rate`
  (a non-foot force channel above 0.1 N), `object_contact_rate`, `table_contact_rate`, `fell`, MuJoCo
  version, asset commits, wall time.
- **Policies.** `hold`: home pose, arms with a smooth ±3° noise; 300-step
  episodes stand with the pelvis between 0.978 and 0.981 m. `seated_babble`:
  each hand picks a goal every 0.5–1.5 s (70% an object's centre, else a point
  on the table pressed into it or 10 cm above), damped-least-squares IK turns it
  into arm targets, smooth noise on arms and fingers; a pincer closes on
  whatever is between its pads halfway through most goals. In the sample, hands
  touch an object in 37–53% of steps and pincers in 27–60% (table: 94–98% and
  37–55%).

### Piano: RP1M re-rendering
RP1M stores poses and key states but no images, sound or touch. Replaying its
*actions* through RoboPianist with our settings does not reproduce its key
presses (F1 0.06; RP1M's own settings, listed in its App. B, are not yet
matched), so we replay *kinematically*. Each step we set the recorded
46 joint angles and 88 key depressions, then MuJoCo computes geometry and
contacts, and we render.

RP1M trajectories place the hands at different base positions (−5.5 to
+8.6 cm along the keyboard), so each episode calibrates the hand bases from
frame 0. Two checks are stored in every episode's `meta`:

- `fingertip_replay_error_m`: max deviation from RP1M's recorded fingertips.
  It is 0.0 mm, and anything above 2 mm aborts the episode.
- `pressed_key_to_fingertip_cm`: median distance from each pressed key to
  the nearest fingertip, 0.2–0.4 cm in the sample.

### Synthetic audio (`src/sensorium/audio.py`)
None of the simulators make sound. It is synthesised from privileged state with
a per-episode random timbre, so it is reproducible from the seed.

| Body | Model |
|---|---|
| Car | 4 engine harmonics, f₀ = idle + k·speed, gain ∝ throttle load; wind noise ∝ speed^1.5; tyre noise ∝ speed + lateral slip |
| Drone | per-motor blade-pass tone (RPM/60 × 2 blades) + 2nd harmonic, motor imbalance, broadband hiss ∝ thrust |
| Piano | 3 partials per key, onset on key-down, slow decay while held, fast decay on release |
| Humanoid | contact "ticks" and slides: new body-pair contacts ring three damped modes on a body-part voice and a surface voice (floor, table, four materials); sliding adds noise ∝ speed·√force; per-episode modes, decay and room tone (see *H1 body*) |

### Templated language (`src/sensorium/describe.py`), probe labels only
- Car: *"car moving steadily at 9 m/s, turning left, braking"*
- Drone: *"drone at 1.2 m, climbing, banking right, moving slowly"*
- Piano: *"right index presses C4; left thumb presses E3"*
- Humanoid: *"seated; left hand touching metal object, right arm touching table"*

## Must be done manually

| Item | Size | Why manual | Effort |
|---|---|---|---|
| **Gold description set** (E3) | ~400 descriptions (~80 per source family), free-form, written while viewing the clip with sound | Checks that the language probe isn't only learning templates (H4). Templates can't evaluate templates. | ~5–6 h. Built with a small labelling script (`scripts/label_gold.py`, E3) |
| **Audio listening check** | ~20 clips per body | Synthetic audio must be plausible and time-aligned. This is hard to assert automatically. | ~30 min per synth revision |
| **Policy/behaviour spot-check** | ~20 rollouts per source family | Catch broken experts (e.g. drone crashes logged as expert), babbling that never touches anything, falls | ~1 h |
| **BiGym demo filter review** | ~20 replayed demos | Confirm the automatic success filter agrees with a human eye | ~30 min |
| **Held-out task choice** | 8 of BiGym's 40 tasks; RP1M's OOD song split adopted as is | Balanced contact-rich / vision-dominated; no near-duplicate tasks across splits | ~1 h |
| *(E3, optional)* own craft recordings | ~1 h of hammering/sawing/screwing (or playing): phone video + audio only (no hardware purchases) | Real reference for the craft-bench simulator's audio and contact models | later |

## Quality gates before pretraining (D0 exit)
- Per-source stats report: steps, episode-length histogram, action
  distribution, experience-kind and effector proportions (must match the
  fixed pool ratio), contact rate, fall rate, audio RMS / spectral centroid
  distributions.
- Round-trip test: write → read → identical arrays. A test also asserts that
  `privileged` and `text` never enter the model batch.
- 5 random episodes per body exported as MP4 + audio for manual review
  (**done:** `make browse`; see [review-sample.md](review-sample.md)).
