# Touch on the H1: how contacts become channels

Updated 2026-10-12. Code: `bodies/h1_skin.py` (dense skin, dynamics),
`bodies/h1_scene.py` (pincer pads), `touch_noise.py` (load-time noise).

## Why the skin changed

HumanoidBench's `h1touch` skin is a set of `touch_grid` patches, mostly
4 × 2 taxels per body part. MuJoCo resolves a fingertip touching a box into
one or two contact points, and the plugin bins each point into one cell. In
the first humanoid sample, a touched hand patch lit **one taxel of eight 83%
of the time** (pincer pads: 75%). A finer grid would only add empty cells.
The literature fixes this in post-processing, not with finer grids:

| What | Who (verified 2026-10-12) | Used here |
|---|---|---|
| Sensor points over a body's surface; each contact's force spread to nearby points, falling linearly to 0 at twice the spacing, total force conserved | MIMo touch module (`spread_linear`) | ✅ hands and pincer pads |
| A pad built from small cubes (4.6 × 4.6 × 1 mm), because one block "will result in a sparse contact force due to the underlying collision detection"; MuJoCo `touch_grid` | Han, Joshi, Deng 2025, arXiv:2505.02915 | ✅ pincer pads (5 × 8 cubes, crowned) |
| Soft-fingertip model: ~15k surface points, penetration-based stress summed per taxel (PyBullet) | Kasolowsky & Bäuml, IROS 2024, arXiv:2409.12735 | — (the spread kernel plays this role) |
| Per-taxel gain N(1, 0.25) with 20% dropout, per-axis gains, a Poisson-effect convolution | Han et al. 2025 | ✅ gain + dropout, at load time |
| Per-taxel offset and per-step Gaussian noise, drawn per episode | Kasolowsky & Bäuml 2024 | ✅ at load time |
| Binary touch / no touch (16 FSRs, threshold) for sim-to-real | Yin et al., RSS 2023, arXiv:2303.10880 | ✅ as an ablation switch only |
| Slip from high-frequency signal energy (30–200 Hz on BioTac) | Veiga et al., IROS 2015 | ✅ as `vibration` |
| Per-fingertip contact onset, slip yes/no, slip magnitude | VibeAct 2026, arXiv:2606.27344 (abstract only) | ✅ as `onset`, `slip` |
| Normal + shear force fields from signed-distance shapes (Isaac) | TacSL, arXiv:2408.06506 | — (needs SDF objects; E4) |

MuJoCo 3.15's own `tactile` sensor reads penetration and slip at every vertex
of a mesh, but only contacts with SDF geoms count. Every scene object would
have to become an SDF, so it is not used.

## What `touch` holds now

Three blocks of channels, in this order (`meta.touch_channels`):

| Block | Hands | Pincers | Names | Unit |
|---|---|---|---|---|
| Body `touch_grid` taxels (unchanged) | 184 | 184 | `body/left_hip_pitch/3` | N, normal force |
| Dense skin points | **2,429** | **690** | `hand/left/ffdistal/17`, `pincer/left/left_pad/5` | N, normal force |
| Dynamics, 3 per skin region | 162 (54 regions) | 72 (24 regions) | `dynamics/hand/left/ffdistal/slip` | see below |

- **Dense skin.** Points are sampled over the union of each body's collision
  geoms (the hands' convex mesh pieces, the pads' boxes and cubes), points
  hidden inside another piece are dropped, and a Poisson-disc subset is kept
  at the body's spacing: fingertips 4 mm, other phalanges 6 mm, palm and
  little-finger metacarpal 8 mm, pincer pads 3 mm. The layout is a fixed
  function of the model (seed 0), so channel names are stable. Each contact's
  normal force goes to the points within twice the spacing whose outward
  normal faces it, so a press on the finger pad does not leak to the nail.
  In the seated sample, a touched hand region now lights **5–15 points**
  in most steps, instead of one.
- **Dynamics**, per skin region (every body with a grid patch or dense skin),
  over the 25 physics steps (500 Hz) since the previous observation:
  `onset` = 1 if a contact started; `slip` = mean tangential sliding speed
  under contact, beyond a 2 cm/s deadband (m/s); `vibration` = RMS of the
  step-to-step change of the region's normal force (N). The 20 Hz
  observation alone cannot see slip or impacts.
- **Privileged.** `touch_xyz` keeps the 3-D forces of every `touch_grid`
  patch, hand patches included: HumanoidBench's 448-taxel layout with hands,
  184 with pincers (`meta.touch_xyz_names`). The hand patches stay in the
  model for comparability; they are not model input.

The pincer pads: each pad's inner 1 mm is a 5 × 8 grid of ~4.4 × 4.7 mm
cubes, 0.5 mm proud at the centre and flush at the edges, so the contact patch
grows with force. The two Menagerie pad boxes are recessed by 1 mm so the pad
keeps its shape; the cubes inherit the boxes' friction and softness and are
in the collision group the cameras do not render.

## Static layout: where every channel sits (`h1_layout.py`)

The channels' positions are fixed by the model (the dense points are a function of the
scene with seed 0), so they are metadata, not privileged state, and a loader may feed them
to a model as taxel coordinates. `make sample-h1` writes one file per effector next to the
sample: `data/sample_h1/skin_layout_hand.json` (146 kB) and `skin_layout_pincer.json`
(40 kB); `uv run python scripts/export_skin_layout.py <dir>` writes them anywhere, and
`HumanoidRecorder.skin_layout()` returns the dict. Contents:

- `channels`: `{"grid": 184, "dense": 2429}` (hand) or `{"grid": 184, "dense": 690}`
  (pincer): the first channels of `touch`, in `meta.touch_names` order; the three dynamics
  channels per region follow and have no position of their own (they belong to the region's body).
- `dense_regions`: per skin region (a finger bone, the palm, a pad): `name`
  (`hand/left/ffdistal`), `body`, `spacing`, `first_channel` (its points are
  `first_channel ...  first_channel + len(points) - 1`), `points` and `normals`
  (`[P, 3]`, in the body's frame, metres; the normals point out of the surface).
- `grid_patches`: per `touch_grid` patch of the body: `name` (`body/left_hip_pitch`), `body`,
  `site_pos` and `site_quat` of its sensor site in the body frame (the site's z axis points
  out), `size` (columns, rows) and `fov_deg`, `channels` (touch columns, row-major) and
  `taxel_names`. All taxels of a patch sit at the site: a taxel is a direction bin of the
  site's outward z axis. Bin centres are assumed uniform in angle over the field of view
  *(the plugin's source was not checked)*.
- `body_pose`: each such body's position and 3 x 3 rotation (row-major) in the pelvis frame at
  the model's default pose (`qpos0`: standing, arms down), for a coarse body map. Combine with
  `proprio` joint angles for the pose at a step.

## Contact forces: ranges and what keeps them there (D0.3 step 3)

The first primitive babbling produced effector-object peaks of 1-38 kN, which would put
garbage into the touch, vibration and sound channels. The fix is physical, not a clamp
on the sensor output.

| What | Value | Source |
|---|---|---|
| Robotiq 2F-85 grip force, adjustable | 20 to 235 N | Robotiq product sheet, May 2025 |
| 2F-85 closing speed, adjustable | 20 to 150 mm/s | same |
| The model's own static grip (`fingers_actuator`, `forcerange` 5, control 255) | 80 N in all, 40 N per pad | our measurement: a fixed box between the pads, 1.5 s |
| Shadow Dexterous Hand E fingertip force | not given in Shadow's specification (2024 sheet: tendon force sensors, resolution ~30 mN; "comparable force output ... to the human hand"); about 10 N per fingertip in press coverage of the vendor's claim | Shadow E technical specification (Dec 2024); New Atlas article *(verify, secondary)* |
| What an H1 arm can push with, statically | about 50-70 N: shoulder and elbow `forcerange` 40 / 18 / 18 N m over a 0.35-0.6 m lever | the H1 model's actuators |

The 235 N and the arm bound are the ranges used below. Neither fixes the *transient* of an
impact, which is what the peaks were: a pad moving at 0.36 m/s met a 0.77 kg metal cylinder
that was pinned against it (a 3 ms event, up to 22 mm of penetration and 33 kN in one
physics step) because the contact was stiff (the Menagerie pads use `solref` 0.004, the
stiffest a 2 ms step allows) and the approach was fast. What changed:

1. **Soft arms and effectors**: every arm and effector geom has contact time constant
   `solref` >= 0.04 s (`arm_solref` setting; the legs and torso stay rigid so the stance is
   unchanged). Chosen by a sweep of 0.02 and 0.04 on effectors only and on all robot geoms
   (5 seeds per cell): the 0.04 cells kept the lifts and gave the smallest peaks. It is
   not calibrated on a measured pad or skin stiffness.
2. **Slower approach**: contact approaches move at about 0.15 m/s at the tip (joint step cap
   0.012 rad per step; 0.03 rad for carrying a grasped object), and the gripper closes at
   0.08 per step, 85 mm in 0.6 s = 140 mm/s, inside the 2F-85's 20-150 mm/s.
3. The gripper actuator needed no change: its static force is already in the published range.

Per-episode maximum normal force, effector-object, N (median / 90th percentile / maximum
over 10 episodes of 600 steps), step 2 -> step 3:

| Cell | step 2 | step 3 |
|---|---|---|
| pincers, table 0.72 m | 207 / 1,363 / 8,521 | 63 / 128 / 314 |
| pincers, table 0.82 m | 233 / 349 / 459 | 46 / 106 / 178 |
| hands, table 0.72 m | 414 / 1,069 / 1,469 | 70 / 172 / 304 |
| hands, table 0.82 m | 421 / 1,865 / 1,913 | 43 / 128 / 292 |

Pincers now sit inside 2 x 235 N in every episode (the maximum is 1.3 x). Hands: the
median is inside the arm's static bound (70 N), the 90th percentile within 2.5 x, the
maximum 4 x (transients of a palm or forearm hitting an object, not the fingers; no published
range applies to them). Self-contact (limb-limb) maxima per episode fell from 177 / 585 / 774
(pincers, 0.72 m) to 82 / 176 / 253 and from 54 / 186 / 248 (hands) to 1 / 136 / 284; no published
range exists for self-contact, so the arm bound is the yardstick, and about one episode in ten
per cell still exceeds 2 x it.

## The A1 gate and the head-camera framing (D0.3 step 4)

`make a1-gate` (after `uv run sensorium-generate configs/a1.toml`; `A1_DIR=` for another run) prints
one PASS / FAIL / INFO line per check and effector and exits 1 on a FAIL. Run it on every
regeneration of the babbling data. The thresholds are fixed, each with its reason, in the
docstring of `sensorium.a1_gate`, which also holds the changelog: change one only with a line
there, never after looking at a result.

| Check | Pincers | Hands | Reason |
|---|---|---|---|
| forces: effector-object peak per episode | every episode <= 2 x 235 N | median <= 70 N, p90 <= 175 N, max <= 280 N | 235 N is the 2F-85's maximum grip (Robotiq); Shadow publishes no fingertip force, so the yardstick is the arm's static push, 70 N (x 2.5 and x 4 for the tail) |
| contact | object contact >= 0.28 and >= 10 bouts / min | >= 0.23 and >= 10 | step 3's means at the 0.82 m table (the review: not below step 3); bouts: step 3's lowest |
| lifts | >= 1 per episode (mean) | INFO only | the step-3 review's bar; hands lift since the claw pinch (0.9, see below), not gated yet |
| visibility | each tip outside the head camera's image <= 10% of steps | same | the step-3 review |
| waiting | an arm without a plan <= 15% of its steps | same | the step-4 review |
| repertoire | every primitive planned >= 1 per episode (mean) | same | count-based choice must reach all of them |

**The framing fix** (decision 1 of the step-3 review): the default table is 0.82 m (was 0.72); hover
and retreat heights are capped at +10 cm over the table (the camera sees up to about 12 cm),
grasped objects are lifted 10 cm, the arms start +10 cm over the table, and goals and objects
lie within |y| <= 0.30 m. Differences from the proposal, each with its reason: the cap is +10 cm,
not +8 (at +8 cm the hands travel 1 cm above 7 cm-tall objects at the fast speed: effector-object
peaks went to 600-850 N); the lift is 10 cm, not 8 (8 cm lifts reached 4-6 cm, under the 5 cm that
counts as a lift); a pincer's pre-grasp point is 6 cm from the grasp point (was 8) and its approach
starts 2 cm above it. Planner repairs that came with it, the first one mattering most:
- the command moves all joints by one common scale, so that it follows the straight joint-space path
  that was checked (the old per-joint clipping bent the path, swung an arm up to 1.5 m and left it
  wedged beside the table);
- the tip must stay in an envelope along every path: in front of the body, within 0.60 m of the
  middle... 
- IK restarts are seeded around the ready pose, the clear posture nearest the current one is taken,
  and of several near ones the one with the lowest elbow (the elbow-low preference of the
  occlusion memo), 12 goals are tried per plan (was 6);
- with no plan found the arm parks low and to the side (0.28 m off the middle, 6 cm over the table)
  instead of freezing mid-air, and an arm that is hit three times in 30 steps retreats there too;
- `TABLE_X` now belongs to a 0.72 m seated table (`TABLE_X_BASE_Z`) so that raising the default to
  0.82 m does not move the table's edge into the arms. Slides stay at x >= 0.40 (the goal region);
  lifts could not be carried lower (a lift counts from 5 cm and 8 cm commands gave 4-6 cm).

A1 at the 0.82 m table, 10 episodes of 600 steps per effector, mean [95% bootstrap interval],
step 3 -> step 4:

| | hands | pincers |
|---|---|---|
| tip outside the image, left / right | 18 / 20% -> 4.1 / 8.4% | 10 / 17% -> 8.0 / 21.0% |
| object contact | 0.23 [0.18, 0.28] -> 0.26 [0.21, 0.33] | 0.28 [0.23, 0.32] -> 0.38 [0.30, 0.46] |
| bouts per minute | 24 [21, 28] -> 27 [23, 32] | 10 [8, 13] -> 19 [16, 21] |
| lifts per episode | 0.0 -> 0.0 | 1.5 [0.8, 2.3] -> 1.1 [0.6, 1.6] |
| arm steps waiting for a plan | 2% -> 6% [2, 10] | 11% -> 7% [3, 11] |
| free-space share | 0.28 [0.24, 0.33] -> 0.20 [0.18, 0.22] | 0.58 [0.56, 0.61] -> 0.43 [0.37, 0.48] |
| tip entropy gap (nats, more negative = more concentrated) | -1.8 [-2.2, -1.5] -> -3.3 [-4.0, -2.5] | -3.8 [-4.0, -3.6] -> -4.4 [-4.7, -4.0] |
| voxel coverage | 0.42 -> 0.41 | 0.21 -> 0.27 |
| effector-object peak per episode, mean N | 73 [35, 128] -> 133 [74, 202] | 64 [41, 93] -> 94 [48, 168] |
| limb slam steps per episode | 0.3 -> 0.4 | 0.2 -> 0.3 |
| reflexes per episode | - -> 0 | - -> 0-2 |
| own-arm occlusion, both arms' links, mean / p90 / steps over 25% | 6% / 19% / 3% -> 5% / 12% / 3% | 11% / 24% / 9% -> 14% / 23% / 4% |

(The waiting step-3 figures are from the 5-episode occlusion run; the step-4 ones are from the 10
episodes' `plan_stats`. Hands plan 5-8 movements per primitive per episode; the pincers' `poke`
0.9.)

Gate result of the final run: hands pass contact, bouts, visibility, waiting and repertoire, and
fail the force line (median 78 N, p90 267 N, max 338 N: the power closure, which now plans, hits
objects harder than the other primitives); pincers pass forces (max 409 N), contact, bouts, lifts
(1.1) and waiting, and fail visibility (the right tip, 21%) and the repertoire (`poke`: 0.9 per
episode; it plans rarely since the envelope and straight-path rules, which is not yet understood).
Hands still do not lift (0.0): the power closure has been the open problem since step 3.

The price of a visible workspace is the concentration of the tips (the entropy gap) and the
free-space share; voxel coverage did not fall (the voxel box is fixed). Wait share: a first
attempt at the envelope had the arms waiting in 35-47% of steps; the Cartesian travel (for hovers
and oriented approaches), the larger plan budget and the parking repaired that. A reflex now
starts a 10-step cooldown and backs the arm out along its last travel posture (the older hover
posture can lie behind the object, which pinned a cylinder at 260 N for 80 steps); loops are gone.

## Hand grasping: the claw pinch

Until this change the hands never lifted anything (0.0 lifts per episode at step 4; pincers 1.1).
The question was whether the hand's physics could hold an object at all, or whether the poses were
wrong. Answer: mostly the poses, plus one physics setting.

**Stage 1, physics or poses** (`scripts/hand_grasp_probe.py`, no planner). The old `power` grasp
could not work by construction: its grasp point lay 3.5 cm off the palm along the palm normal, but
the palm's face is already 2.8 cm out, so for every object (radius 2.5-5 cm) the palm landed inside
the object and pushed it away; and its closure (fingers and thumb flexed together) made a fist
with no thumb opposition. Put by script into a good pose, the hand holds: a claw pinch from above
(the palm faces the table over the object, the thumb is pre-placed under the palm on the near side,
the fingers, flexed at the knuckles, sweep the object against it), descend 4 cm, close 0.8 s, lift
10 cm in 1 s, hold 1 s, the right hand, one object alone at (0.50, -0.20) m:

| object (mass) | elliptic cone: rise max, steps >= 5 cm | lift | pyramidal cone (old hand scene) | lift | peak hand-object N (elliptic) |
|---|---|---|---|---|---|
| wood box (0.21 kg) | 12.2 cm, 32 | yes | 11.3 cm, 31 | yes | 28 |
| plastic sphere (0.07 kg) | 13.6 cm, 35 | yes | 5.6 cm, 3 | no | 23 |
| metal cylinder (0.77 kg) | 7.2 cm, 14 | yes | 6.8 cm, 9 | no | 129 |
| rubber sphere (0.12 kg) | 12.7 cm, 34 | yes | 7.8 cm, 8 | no | 17 |
| plastic box (0.10 kg, 4 cm tall) | 0.1 cm | no | 0.0 cm | no | 7 |
| metal box (1.17 kg) | 1.1 cm | no | 0.9 cm | no | 15 |

(A lift is 10 steps at 5 cm or more, as in `babble_stats`.) Figures: `runs/hand_grasp/grasp_height.png`
(object height against time, both cones) and `grasp_forces.png` (per-finger force), written by the
script. Two findings:

- **Physics: the friction cone.** The hand scene had MuJoCo's pyramidal default, the pincer scene
  Menagerie's `cone="elliptic" impratio="10"` (the option line of the pinned `2f85.xml`). With the
  pyramidal cone the spheres and the cylinder creep out of a closed pinch although the fingers'
  friction (1.0; MuJoCo takes the larger of the two geoms') would hold them: 1 of 6 objects lifts,
  4 of 6 with the elliptic cone. Both scenes now use the elliptic cone (`h1_scene.scene_xml`). Also
  tried with the pyramidal cone, scripted, on three or four objects x three poses: torsional
  friction (`condim` 4 or 6 on the hand: no gain), `noslip_iterations` 10 (no gain), finger gains
  x 3 (no gain; a gain change would also change every other primitive).
- **Strength: the metal box** (1.17 kg, metal friction 0.3) lifts with nothing tried. Only the
  index finger and the thumb carry the load, 4-10 N each: the HumanoidBench hand's finger actuators
  have kp 1 N m/rad and `forcerange` 1 N m, about 10 N at a fingertip at most, and the pinch reaches
  part of it. The published figure is about 10 N per fingertip, from press coverage *(verify)*.
  Left as it is: a heavy object that does not come up is also experience.

**Stage 2, the grasp in the planner** (`h1_babble`, `HAND_PINCH`). No grasp synthesis library was
needed: the pinch is analytic (a pre-shape, then closure targets past the contact, so that the
position actuators squeeze with the remaining error). DexGraspNet was not used. What the
`power` primitive does now, each part with the reason it is there:

1. Frame: the grasp point is the pinch's near point, 7.5 cm along the fingers and 8.5 cm along the
   palm normal, sideways midway between the thumb and the index and middle fingers. It is put on the
   object's near face (the centre minus the object's half extent along the fingers), with the
   fingers level or 6 degrees down and the palm normal 18 degrees off vertical (`HAND_PALM_Z` -0.95;
   with -0.87 the per-grasp rate was 32% against 57%, 96 attempts each).
2. Wrist: held flexed and deviated (WRJ1 -0.35, WRJ2 -0.3 rad) through the grasp. With the wrist
   straight, the five arm joints reach a level palm-down hand at a third of the goals; with this,
   at all (30 goals per setting; the planner keeps its 5-joint IK).
3. Travel open, pre-shape over the object (0.4 s hold), then descend: pre-shaped while travelling,
   the thumb hangs 10 cm under the palm and swept objects off the table.
4. Replan if the object moved more than 1.5 cm between planning and the descent (`replans` in
   `plan_stats`): the travel may brush objects (effector contacts are allowed on the way), and a
   grasp planned for where a sphere was closes on air.
5. Closure: fingers to J3 1.57 and J1 + J2 3.0, thumb tip joint to 1.5 (the first version, 2.0 and
   1.2, squeezed less: 26% against 19% per grasp over 96 attempts, before the other changes).

Measured with `scripts/hand_grasp_bench.py` (one planned grasp at a time through the planner, the
object alone at a random place and yaw on that arm's half, 20 placements per object and arm, Mac
CPU): a grasp counts once it is planned; success is a lift.

| object | planned / tried | lifts | success [Wilson 95%] | effector-object peak N, median / max |
|---|---|---|---|---|
| wood box | 27 / 40 | 16 | 59% [41, 75] | 28 / 759 |
| plastic sphere | 30 / 40 | 19 | 63% [46, 78] | 15 / 27 |
| metal cylinder | 26 / 40 | 14 | 54% [35, 71] | 303 / 1,850 |
| rubber sphere | 26 / 40 | 19 | 73% [54, 86] | 13 / 27 |
| plastic box | 29 / 40 | 14 | 48% [31, 66] | 20 / 63 |
| metal box | 27 / 40 | 0 | 0% [0, 12] | 21 / 538 |
| all | 165 / 240 | 82 | 50% [42, 57] | |
| all but the metal box | 138 / 200 | 82 | 59% [51, 67] | |

In the babbling itself (full repertoire, A1 settings, 10 episodes of 600 steps, `make a1-gate` on
them) hands lift **0.9 per episode [bootstrap 95%: 0.4, 1.4]** (step 4: 0.0; pincers 1.1). Of the
power grasps planned in four instrumented episodes, 6 of 26 lifted (23%): the bench's 59% does not
carry over to a crowded table yet. Gate lines, step 4 -> now: forces median 78 -> 72 N, p90 267 ->
158, max 338 -> 190 (the median still fails its 70 N line by 2 N; p90 and max now pass); object
contact 0.26 -> 0.41; visibility 4.1 / 8.4% -> 5.2 / 6.8%; waiting 6% -> 8.6%; repertoire passes
(power 7.2 plans per episode). The pincer scene and the pincers' planner are unchanged.

Not good yet:
- 50-59% per grasp alone, about 23% on the crowded table, against the 70% target; 0.9 lifts per
  episode against the pincers' 1.1. The difference between the bench and the babbling is not
  understood yet (candidates: neighbouring objects in the fingers' sweep, objects at the rim or
  tipped by earlier pokes, replanned attempts counted as attempts).
- The metal cylinder's grasps peak at 300 N in the median and up to 1.9 kN: one-step spikes on the
  index fingertip against the stiff metal contact (`solref` 0.002, see the step-3 section). A
  softer mix on the hand (`solmix` 10) made them larger.
- The metal box does not lift (strength, above); the 4 cm plastic box is the weakest of the rest.

## Load-time noise (`touch_noise.perturb_touch`)

Stored episodes are clean. The training loader draws, per episode:
per-channel gain N(1, 0.25) with 20% of channels dead, an optional per-channel
offset (default 0) and per-step Gaussian noise σ = 0.01 N on force channels,
then clips forces at 0. `dynamics/*` channels get gain and dropout only.
`binarize=<N>` turns forces into touch / no touch for the binarisation
ablation. Training will also compress forces with `log1p` (heavy tails).

## Hand strength: Shadow's effort limits (supervisor decision, 2026-10-20)

The scene now gives the Shadow hands' finger and thumb actuators Shadow's own effort limits
(`h1_scene.strengthen_shadow_hands`); the wrist is unchanged. Sources: Shadow's robot description
`sr_common` (`sr_description/hand/xacro`, the joints' `<limit effort=...>`, checked at commit
172c3d9 of January 2025) and Shadow's specification (*Shadow Dexterous Hand - Technical
Specification*, September 2025, 2.4: "The Hand, while in a power grasp, can hold up to 4 kg").
Shadow publishes no fingertip force. Menagerie's +-1 N m cap, which HumanoidBench's hands inherit,
is not documented.

| actuator | old `forcerange` (N m), `kp` (N m/rad) | new `forcerange`, `kp` | sr_common joint effort |
|---|---|---|---|
| fingers J3 (proximal), J4 (knuckle), LFJ5 | +-1, 1 | +-2, 2.86 | 2 |
| fingers J0 (tendon on J1 + J2) | +-1, 0.5 | +-2, 2.86 | 2 (J1 and J2 each) |
| THJ5 (base) | +-3, 0.4 | +-5, 7.14 | 5 |
| THJ4 (proximal) | +-2, 1 | +-3, 4.29 | 3 |
| THJ3 (hub) | +-1, 0.5 | +-2, 2.86 | 2 |
| THJ2 (middle) | +-1, 1.5 | +-2, 2.86 | 2 |
| THJ1 (distal) | +-1, 1 | +-1, 1.43 | 1 |
| WRJ2, WRJ1 (wrist) | +-10, 10 and +-5, 8 | unchanged | 10 and 30 |

`kp` = limit / 0.7 rad (`HAND_FULL_FORCE_ERROR`): an actuator pushes with its full limit once its
joint lags its target by 0.7 rad, which the claw pinch's closure targets reach (the proximal joints
run at 50-90% of the limit while holding). With 0.4 or 0.2 rad (stiffer) the scripted pinch lost
most objects: the fingers struck and pushed them out before the thumb held them.

Before (Menagerie's limits) -> after, same code otherwise:

| measure | before | after |
|---|---|---|
| bench: lifts per planned grasp, all objects (165 grasps each) | 50% [Wilson: 42, 57] | 46% [39, 54] |
| ... wood box / plastic sphere / metal cylinder | 59 / 63 / 54% | 52 / 63 / 60% |
| ... rubber sphere / plastic box | 73 / 48% | 56 / 38% |
| ... metal box (1.17 kg) | 0 of 27, 0% [0, 12] | 3 of 29, 10% [4, 26] |
| bench: largest hand-object contact, median N: wood box / plastic sphere / metal box | 7 / 5 / 6 | 20 / 15 / 20 |
| bench: largest contact, metal cylinder, median / max N | 58 / 154 | 81 / 182 |
| A1, 30 episodes: lifts per episode [bootstrap 95%] | 1.33 [0.97, 1.70] | 0.93 [0.63, 1.23] |
| A1, 30 episodes: effector-object peak per episode, median / p90 / max N | 68 / 144 / 268 | 81 / 157 / 412 |
| A1, 30 episodes: object contact share / bouts per minute | 0.35 / 21.1 | 0.36 / 24.1 |
| A1, 30 episodes: effector-table peak median, limb slam steps per episode | 404 N, 0.13 | 328 N, 0.07 |
| A1, 30 episodes: tips outside the image, left / right | 4.2 / 7.8% | 4.1 / 9.5% |
| `make a1-gate`, the 10 A1 seeds | forces FAIL (median 72), visibility pass (5.2 / 6.8%), lifts 0.9 | forces FAIL (median 121, p90 171, max 195), visibility FAIL (3.0 / 10.7%), lifts 1.1 |

(The 30-episode rows use seeds 5000-5029, the A1 gate the generator's own seeds; the bench is
`scripts/hand_grasp_bench.py`, 20 placements per object and hand.) The other primitives keep their
contact statistics (contact share, bouts and waiting within their intervals); the hands now press
harder on what they touch. Scripted pinches (`hand_grasp_probe.py`, elliptic cone): before 4 of 6
objects lift, after 3 of 6 (the cylinder now rises 5.5 cm for 2 steps); the metal box rises 4.3 cm
(1.1 before), not yet a lift.

Verdict: the stronger hand lifts the metal box sometimes and squeezes harder, but it lifts no more
in babbling (fewer lifts per episode, within the intervals), and it pushes the force line further
off (median and max up). The 10-episode visibility failure is within the noise of that check (30
episodes: 9.5% [6.0, 13.6] for the right tip). Open choices: keep the limits but reach them only at
a larger lag (with 1.0 rad and the thumb's old gains the scripted pinch lifted 4 of 6, the
cylinder included), keep the thumb's old gains (a stiff thumb works as a wall that pinched objects squirt off), or close more
slowly.

## Costs

| | Hands | Pincers |
|---|---|---|
| Recorder start-up (point layout) | ~2.6 s | ~0.7 s |
| Recording, seated, M1 Pro core | ~27 steps/s | ~30 steps/s |
| Sample (8 episodes × 300 steps) | 45 s, 22 MB (was 42 s, 21 MB) | |

The per-physics-step contact pass costs ~15% of recording time; rendering
still dominates.

## Known issues and open questions

- **The D0.2 seated babbling policy slammed the elbows on the table and
  ground the shoulder into the torso.** `vibration` reported it faithfully; it
  came from the policy, not the skin. D0.3 step 1 (`SafeBabblePolicy`, A1 rows
  12–13 in `babble_stats.py`, 10 episodes of 600 steps per variant): steps
  with a non-effector contact above 100 N fell from 108 to 1.4 per episode
  (hands, table 0.72 m) and 3.0 to 1.0 (pincers); steps with limb-limb
  contact above 5 N from 29% to 0.5% (hands) and 18% to 1.1% (pincers).
  Remaining: single-step spikes of 150–500 N (a moving object struck by an
  elbow, the two pincers meeting at the midline). Table height did not change
  the slam statistics beyond the intervals; it matters for lifts (spike).
- **Dense skin is far more than the ~200 taxels Tactile Genesis found
  sufficient** for its tasks. Whether the density helps is an E1 question:
  a touch-resolution ablation (body grid only vs dense) is cheap to add,
  since senses are dropped at load time and channels can be too.
- **Normal force only** on dense points; shear is summarised by `slip`.
  Per-point shear would triple the channels.
- The spread radius is fixed per body. Han et al. randomise pad depth and
  kernel width per episode; we do not yet.
