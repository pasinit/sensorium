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
- ~~The metal cylinder's grasps peak at 300 N in the median and up to 1.9 kN~~: corrected below
  ("Forces"); those were sums over the hand's many contact points, not the gate's single-contact
  peak, which stays under 160 N.
- The metal box does not lift (strength, above); the 4 cm plastic box is the weakest of the rest.

### Follow-up: the crowded table, forces, and the real hand's strength

**Why grasps fail on the crowded table** (`scripts/hand_grasp_crowd.py`: the full babbling
repertoire with all six objects, no rendering, every planned `power` grasp followed to the arm's
next plan and given one cause; 32 episodes of 600 steps, seeds 0-31; attempts cut by the episode's
end left out). On main:

| cause | attempts | share [Wilson 95%] |
|---|---|---|
| object moved more than 1.5 cm before the descent: replanned | 75 | 39% [32, 46] |
| ... of them touched by the grasping hand itself first | 24 | 12% [9, 18] |
| no pinch at closure (fingers and thumb not both on the object) | 59 | 31% [25, 38] |
| success (a lift) | 21 | 11% [7, 16] |
| metal box (1.17 kg): pinched, no lift | 23 | 12% [8, 17] |
| other objects: pinched, no rise or slipped in the lift | 14 | 7% [4, 12] |

(192 attempts; the planner also dropped 8.1 power plans per episode before queueing them.) The
bench's 50-59% does not carry over for three reasons: two of five grasps are abandoned before the
descent because the object moved (rolling spheres were planned on where they were; the hand's
travel and earlier primitives push objects), the metal box, which the hand cannot lift, draws a third of all
attempts (64 of 192: it stays on the table while light objects get pushed away or roll), and
closures miss more often (candidates, from the closures' geometry: neighbouring objects in the
fingers' way, tipped boxes and lying cylinders).

**What was tried against it** (branch `feat/hand-grasp-crowded-planner`, not proposed for merge):
grasp only objects slower than 2 cm/s; travel with the wrist and fingers already pre-shaped and the
thumb open (with the wrist straight the fingertips hung 6 cm lower and swept objects); rise
straight up first when the hand is low; travel along the straight line instead of a joint-space
move. On the same 32 episodes the replans fell from 39% to 18% and the share of successes rose from
11% [7, 16] to 21% [15, 27] (169 attempts; `runs/hand_grasp/failure_causes.png`), but the misses
grew (31% -> 41%), and what the A1 gate counts did not move: lifts per episode in recorded babbling
1.20 [bootstrap 95%: 0.95, 1.42] on main (60 episodes) against 1.14 [0.86, 1.44] (50 episodes), and
success per grasp with the object alone fell from 50% [42, 57] to 38% [31, 46] (162 attempts). Not
worth merging as it is. Also tried and dropped: a thumb that swings in from further back (24% vs
24% per attempt over 165-169 attempts), smaller grasp gaps, slower travel near the table (below).

**Forces.** Two corrections first:

- The 1.9 kN "cylinder spikes" of the bench were sums over all the hand's contact points on the
  object (a fingertip mesh touches with several). The gate counts the largest single contact. For
  the metal cylinder, 22 planned grasps on main: largest single contact median 62 N, max 157 N;
  summed median 302 N, max 1,866 N. `hand_grasp_bench.py` now prints both.
- The 72 N median of the 10-episode A1 run was a draw: over 60 episodes, main's per-episode
  effector-object peak has median 76 N [bootstrap 95%: 62, 96], p90 177 N, max 529 N, so main does
  not meet the 70 N median line (nor the 280 N max) on a larger sample either; 10-episode runs of
  the same or nearly the same code gave medians from 54 to 123 N. Visibility is as noisy: over 30 episodes main has 4.2% /
  7.8% of tip-steps outside the image (left / right), but single 10-episode runs gave up to 12-13%
  for the right tip with the experimental branch, which over 30 episodes has 7.5% / 6.1%.

Where the peaks come from (16 episodes, each episode's largest effector-object contact attributed
to the primitive and phase): hands travelling into the metal box or cylinder during `slide`, `poke`,
`hover` and the parking wait, at the fast travel speed; once the closing of a grasp. Not the grasps.
Two remedies tried, neither reduced the peaks measurably over 32 episodes: a withdrawal when a
travelling hand meets an object with more than 30 N (the peak happens in the impact, before any
reaction; median 69 N, max 468), and slower travel (x 0.4) below 8 cm over the table (median 57 N,
p90 174, max 468, but fewer grasp attempts and lifts per episode). Keeping objects from being placed
under a hand's start removed some first-step impacts, but those fall in the settle steps that the
gate does not count. Left as is; the force line needs either a gate decision (it is not met by main
on 60 episodes) or slower babbling overall.

**The real hand's strength** (for the supervisor's decision; the model is unchanged):

| quantity | real Shadow Dexterous Hand | the model (HumanoidBench = Menagerie `shadow_hand`) |
|---|---|---|
| grasp payload | "The Hand, while in a power grasp, can hold up to 4 kg." | not tested in a power grasp; the claw pinch holds 0.07-0.77 kg, not 1.17 kg |
| fingertip force | not published; joint force/torque control "is not available" (force control protects the hand or makes joints back-drivable); tendon force sensors of "about 30mN" resolution | finger flexion `kp` 1 N m/rad, `forcerange` +-1 N m (MCP and the coupled J1 + J2 tendon): at most about 10 N at the fingertip over the finger's ~0.1 m; 4-10 N measured in the pinch |
| other figures | about 10 N per fingertip (New Atlas, press, secondary, version not stated); "at least 8 N force at fingertip" is the newer DEX-EE hand, a different product | |

Sources: Shadow Robot Company, *Shadow Dexterous Hand - Technical Specification*, September 2025,
sections 2.4 "Weight and Payload" (also: hand and forearm 4.3 kg), 4.2 "Force" and the note to
section 1 (shadowrobot.com, `shadow_dexterous_hand_e_technical_specification.pdf`); MuJoCo
Menagerie `shadow_hand/right_hand.xml` (class `right_hand`: `<position forcerange="-1 1"/>`, the
J1 + J2 class `kp="0.5"`, and `<option cone="elliptic" impratio="10"/>`). The specification gives a
4 kg power-grasp payload and no fingertip figure, so it does not say whether the model's fingertips
are too weak for a pinch; it does say the real hand holds far more than 1.17 kg when it wraps an
object.

### Own grasp success on the event-dense table (2026-10-22)

The supervisor's priority after the event-dense data (docs/event-dense.md): more lifts the robot
makes itself. Measured with `scripts/grasp_success.py` on the `configs/event_dense.toml` scene
(props and scene director), Mac, no rendering. It follows every planned grasp (hands `power`,
pincers `grasp`). An own lift is the object held 3 cm over its height at the plan for 0.5 s,
touched. A failed grasp gets one cause: replanned (the object moved before the descent), reflex,
director (the director moved this object), no pinch (never held between a finger and the thumb,
or between both pads, in the first 0.75 s closed), metal box, slip (rose 1 cm or more), no rise.

Baseline, 30 episodes per effector (seeds 0-29):

| | grasps/ep | own lifts/ep | success per grasp [Wilson 95%] | main causes |
|---|---|---|---|---|
| hands | 6.4 | 0.60 | 9% [6, 14] | replanned 39%, no pinch 27%, no rise 9%, slip 7%, metal box 6% |
| pincers | 5.6 | 1.97 | 35% [28, 43] | no rise 23%, no pinch 19%, director 11%, metal box 4% |

Each change alone, 30 episodes (hands with re-aiming from the second row on):

| change | effector | own lifts/ep | success per grasp | kept |
|---|---|---|---|---|
| re-aim a moved object up to 2 times (`REAIMS`) instead of planning something new | hands | 0.60 -> 0.87 | 9% -> 16% [11, 22] | yes |
| raise the move threshold 1.5 -> 3 or 5 cm instead | hands | 0.67-0.73 (15 ep) | 11-13% | no: replans became missed pinches |
| pinch at most 4.5 cm under the object's top (`PINCH_BELOW_TOP`); at the centre of a standing 15 cm block the palm met its top and pushed it over | hands | 0.87 -> 1.03 | 16% -> 20% [14, 27] | yes |
| grip across the object's narrow side (IK restarts over the wrist yaw) | pincers / hands | 1.97 -> 1.41 / 0.87 (15 ep) | 35% -> 28% / 16% | no: the H1 wrist yaw rarely reaches an aligned grip |
| pinch gap 7.5 -> 6.5 cm | hands | 0.87 -> 0.47 | 16% -> 9% | no |
| close at half the rate | hands | 0.73 | 12% | no |
| the director leaves objects a hand is on or near alone | both | no change | pincer "director" cause 11% -> 8% | no |
| plan a pincer grasp only if the object is narrower than 7.5 cm along the closing axis (`PINCER_SPAN`) | pincers | 1.97 -> 1.97 | 35% -> 42% [34, 51] | off by default: fewer attempts on the wide boxes, not more lifts |

Verification of the two kept changes (now the defaults) on 30 fresh episodes (seeds 100-129), and
pooled with the ablation runs (60 episodes):

| hands | own lifts/ep | success per grasp | replanned | no pinch |
|---|---|---|---|---|
| before, fresh seeds | 0.67 | 13% [8, 19] | 40% | 27% |
| after, fresh seeds | 0.90 | 17% [12, 23] | 17% | 37% |
| before, 60 episodes | 0.63 | 10% [8, 15] | 39% | 27% |
| after, 60 episodes | 0.97 | 18% [14, 23] | 18% | 37% |

Pincers are unchanged by both changes: 1.60 own lifts/ep and 29% [23, 36] on the fresh seeds
(1.97 and 35% on seeds 0-29).

Frame strips: `runs/grasp/strip_success_hand.png` (a hand lifts the wood box) and
`runs/grasp/strip_no_pinch_hand.png` (the main remaining failure: the hand closes along the 10 cm
plastic box and its fingers pass beside it).

Not good yet: hands still miss the pinch in 37% of grasps (the wide plastic box, spheres that roll
in the sweep, lying cylinders). Pincers fail to raise the plastic box they grip along its length
(23-25% "no rise"). Gripping across the narrow side needs an arm that can turn the gripper; the
H1's wrist yaw does not reach far enough, and the span check only avoids those grasps.

### A grasp library (2026-10-22)

The target after the checkpoint above: most grasp attempts succeed. The approach is a grasp database
(as in GraspIt-style pipelines). The 5-joint H1 arm cannot reach an arbitrary effector pose, so the
library (`bodies/h1_grasp_library.py`) does not store poses to reach. It stores descriptors of grasps
the planner actually made and executed, each with its outcome.
- **Descriptor**, in the object's frame with absolute values (the objects are symmetric under axis
  flips): the approach axis, the second axis (pad closing axis or palm normal) and the grasp point's
  offset from the centre over the half sizes. Then, in the world: both axes' heights, the grasp
  height over the centre, and the object's extent along each frame axis.
- **Key**: the object, and which of its axes is up (how it lies).

`scripts/grasp_library.py` makes a library in an isolated scene. Each trial puts the object alone at
a random place within reach, at a random yaw, upright or (not spheres, half of the time) on a side
face, and redraws the hidden properties. The planner then plans one grasp in `explore` mode (random
IK starts and postures, a jittered grasp point) and executes it. Lifted means 3 cm for 0.5 s,
touched. 300 trials per object and effector took 19 minutes on the Mac.

Coverage: planned grasps and the share that lifted.

| object | hand: planned (of 300) | hand: lifted | pincer: planned (of 300) | pincer: lifted |
|---|---|---|---|---|
| wood box | 136 | 34% | 64 | 22% |
| plastic sphere | 112 | 17% | 53 | 17% |
| metal cylinder | 131 | 37% | 74 | 58% |
| rubber sphere | 119 | 23% | 56 | 43% |
| plastic box | 144 | 26% | 59 | 22% |
| metal box | 141 | 32% | 67 | 70% |
| wood tall box | 108 | 14% | 62 | 48% |
| plastic tall cylinder | 77 | 42% | 60 | 32% |
| all | 968 | 28% | 495 | 40% |

At run time (`grasp_library` recorder setting) the planner draws `GRASP_CANDIDATES` (8) reachable
grasps, scores each by its 7 nearest library grasps of the same key (success rate shrunk by half a
success in one trial), and makes the best if it reaches `LIBRARY_MIN_SCORE` (0.6). Objects whose key
lifts less than `LIBRARY_MIN_RATE` (20%) in the library are not grasp targets. Re-aiming stays.
Leave-one-out on the library: pincer grasps scored 0.6 or more lift 72% (24% of grasps qualify),
0.7 or more 81% (14%); for hands only 42% (0.5) and 54% (0.6).

Pincers on the event-dense table (`configs/event_dense.toml` now gives the pincer blocks the library
in `configs/grasp_library/`), every planned grasp counted, aborted ones included:

| pincers | episodes | grasps | success per grasp [Wilson 95%] | own lifts/ep | distinct objects lifted/ep |
|---|---|---|---|---|---|
| planner alone (seeds 0-29 and 100-129) | 60 | 326 | 32% [27, 37] | 1.75 | 1.27 |
| library, minimum 0.6, seeds 0-29 | 30 | 116 | 70% [61, 77] | 3.00 | |
| library, minimum 0.6, seeds 100-129 | 30 | 96 | 56% [46, 66] | 2.00 | |
| library, minimum 0.6, pooled | 60 | 212 | 63% [57, 70] | 2.25 | 1.35 |
| library, minimum 0.5 (seeds 0-29) | 30 | 122 | 63% [54, 71] | 2.75 | |
| library, minimum 0.7 (seeds 0-29) | 30 | 92 | 61% [51, 70] | 2.33 | |

(Runs of the same setting differ by about 10 points: the pooled 60 episodes are the figure to use.)
Remaining pincer causes, pooled: no pinch 17%, metal box 7%, reflex 5%, no rise 4%, director 2%. The
lifts concentrate on the metal box (64 of 135) and the metal cylinder (32). No grasp of the plastic
sphere or the wood box reached 0.6 in these runs, so the pincers no longer lift them.

Hands: the library does not get them there.
- With the explored library: 22-25% per grasp, but 1.7-2.2 grasps per episode, so 0.39-0.54 own
  lifts/ep against 0.60 without it.
- A planner-mode library (the planner's own grasps, 200 isolated trials per object, used as a filter
  by object as it lies): 21% [14, 30], 0.66 own lifts/ep.
- Per object as it lies, the claw pinch lifts at most 40-60% even where it fits best (plastic box on
  its long side, standing tall cylinder, metal box on end), and 4-19% on spheres and the lying tall
  block.

Hands stay without a library. The next steps change how they grasp: touch-triggered closed-loop
closing, then a side power wrap for the objects where the pinch is weak.

**Variety and reflexes of the library pincers (2026-10-22).** Event-dense seeds 0-24, lifts per 25
episodes:

| pincers with the library | per grasp [Wilson 95%] | own lifts/ep | distinct objects/ep | reflex |
|---|---|---|---|---|
| as merged | 74% [65, 82] | 2.80 | 1.40 | 9% |
| each own lift counts as 3 extra tries when objects are picked | 77% | 2.88 | 1.36 | |
| ... as 8 extra tries | 74% | 2.68 | 1.36 | |
| bar 0.4 for objects not lifted yet in the episode (0.6 after) | 61% [52, 70] | 2.76 | 1.40 | |
| grasp clearance +4 mm (from -3 mm) | 73% [62, 81] | 2.36 | 1.16 | 1% |

- Picking is not what concentrates the lifts on the metal box (28 of 70) and metal cylinder (24).
  The library is: few grasps of the plastic sphere, wood box and plastic box reach the bar.
- The lower bar for unlifted objects spreads the lifts a little: the metal box is 23 of 69, and the
  plastic box is lifted 6 times instead of once. But it adds failing grasps, and distinct objects per
  episode do not change.
- The reflexes are the forearm pressed into the table or the grasped object, both of which grasp
  plans allow by a few millimetres. Planning them clear removes the reflexes but also a sixth of
  the plans.
- None was kept.

### A side wrap, the wrist and touch-closed grasps (2026-10-22)

Hands after the library: they still lift about 20% of their grasps on the event-dense table, and
mostly fail as "no pinch". The fingertips land on the object's top 3-6 cm before the descent ends,
and the hand stays on it. Three checks follow.

**Touch-closed grasps (removed).** The skin was used to close the loop: the effector's normal force
against the world, read from the contacts.
- An observe-only pass on 25 event-dense episodes logged what the skin felt at each closure. Every
  closure that lifted touched on both sides (thumb and fingers, or both pads): 70 of 70 for the
  pincers, 27 of 27 for the hands. Of the misses that ended as "no pinch", 0 of 10 and 8 of 52 did.
  That makes the closure's touch a near-perfect check of whether a grasp holds anything.
- The first touch during the descent is no such check: it comes early in grasps that lift too (15
  of 70 pincer successes, 18 of 29 hand successes). Stopping or closing on it made both effectors
  worse:
  - pincers 74% -> 50% per grasp, and 37% when closing at the first touch;
  - hands 21% -> 11%, and 7%.
- Re-aiming after a closure that touched on one side only did not help either:
  - hands 20% [15, 25] -> 22% [17, 27] pooled over 55 episodes, the same 18% on fresh seeds 100-129;
  - pincers 74% -> 67% on the same seeds.

  A failed claw grasp is wrong in its geometry, and aimed again it is the same grasp. The loop was
  taken out.

**The wrist.** `scripts/hand_reach.py` asks whether a hand pose is reachable at all.
- Setup: 480 placements (each object 60 times, alone, at a random place, yaw and resting face). The
  planner's oriented IK runs from 16 random starts.
- Reached: the grasp point within 1 cm, the arm clear, and the hand no more than 5 mm into the table
  or the object.
- Joint sets: the 5 arm joints, which include the forearm's roll (`wrist_yaw`); then with the Shadow
  wrist's flexion (`WRJ1`, pitch) solved for instead of held at the pinch's angle; then with its
  deviation (`WRJ2`) as well.

| hand pose | arm (as planned) | + wrist pitch | + pitch and deviation |
|---|---|---|---|
| claw pinch from above | 100% | 100% | 100% |
| side wrap, palm level (vertical palm) | 12% | 12% | 44% |
| side wrap, palm normal at z -0.4 | 55% | 60% | 64% |
| side wrap, palm normal at z -0.7 | 81% | 84% | 84% |

- Nearly every miss is the hand meeting the table (417 of 422 for the level wrap), not the IK.
- The claw's failures are not a matter of wrist range: all its poses are reachable.
- A more mobile wrist would buy a level side wrap, not more claw lifts. The body stays as it is.

**A side wrap (parked).** `HAND_WRAP` and `wrap_frame` give the hand a second grasp:
- The palm is turned down onto the object's side (normal at z -0.7, the reachable tilt above), and
  the fingers point level, 0.3 or 0.6 down.
- The oriented IK leaves the approach's azimuth free. So the frame point is the object's centre, its
  extent in front of the palm's face, and the palm faces the object (`wrap_pose`).
- On closing, the fingers flex fully round the far side and the opposed thumb closes over the top.
- `WRAP_SHARE` (default 0) sets the share of a hand's grasps that are wraps. `scripts/grasp_library.py
  --wrap` makes isolated wraps.

Wraps of the table objects, 80 isolated placements each:

| object | wraps planned | wraps lifted | claw lifted (explored library) |
|---|---|---|---|
| wood box | 36 | 3 (8%) | 34% |
| plastic sphere | 27 | 0 | 17% |
| metal cylinder | 20 | 1 (5%) | 37% |
| rubber sphere | 20 | 0 | 23% |
| plastic box | 26 | 2 (8%) | 26% |
| metal box | 25 | 0 | 32% |
| wood tall box | 28 | 3 (11%) | 14% |
| plastic tall cylinder | 31 | 6 (19%) | 42% |
| all | 213 | 15 (7%) [4, 11] | 28% |

Films show the hand arriving over the object's side with the object inside it. The closing fingers
then sweep over it, and it stays on the table as the hand rises.

The table objects are small for a Shadow hand (3-10 cm). `--hand-sized` puts three temporary props
in place of the tall block and bottle, with hidden properties as for the others:
- a can: 7 cm across, 16 cm tall;
- a bottle: 6 cm across, 18 cm tall;
- a box: 11 x 7 x 6 cm.

40 placements each:

| object | claw: planned | claw: lifted | wrap: planned | wrap: lifted |
|---|---|---|---|---|
| can | 8 | 5 | 19 | 3 |
| bottle | 13 | 7 | 18 | 3 |
| box | 21 | 7 | 17 | 2 |
| all | 42 | 19 (45%) [31, 60] | 54 | 8 (15%) [8, 27] |

- The wrap does not work on hand-sized objects either, so it is parked.
- The claw lifts 45% of its grasps of them, against 28% of the table objects; the samples are small.
- The claw plans only a third of the placements, so the share of placements lifted is 16% (claw) and
  7% (wrap).

## Load-time noise (`touch_noise.perturb_touch`)

Stored episodes are clean. The training loader draws, per episode:
per-channel gain N(1, 0.25) with 20% of channels dead, an optional per-channel
offset (default 0) and per-step Gaussian noise σ = 0.01 N on force channels,
then clips forces at 0. `dynamics/*` channels get gain and dropout only.
`binarize=<N>` turns forces into touch / no touch for the binarisation
ablation. Training will also compress forces with `log1p` (heavy tails).

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
