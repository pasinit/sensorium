# Datasets & simulators: inventory and gaps

Legend: ✅ available natively · 🟡 partial / coarse · 🔧 we must generate ·
❌ not available · *(verify)* = claim not yet checked against the primary source.

Updated 2026-10-11 for the one-humanoid plan ([thesis.md](thesis.md)).
Facts marked **verified** were checked against the primary sources that day.

## 1. The body and its sources (E1)

| Source | What it gives | RGB | Audio | Touch | Proprio | Actions | Licence | Use |
|---|---|---|---|---|---|---|---|---|
| **HumanoidBench** (`h1touch`) | Unitree H1 + 2 Menagerie Shadow hands, 448 `touch_grid` taxels, 2 head cameras, 27 tasks, released whole-body *reach* policies only (**verified**) | ✅ ego | ❌ → 🔧 | ✅ 448 taxels | ✅ | ✅ | MIT | **the body**; reach policies drive goal babbling; tasks are probes, not few-shot targets |
| **BiGym** | same H1 with Robotiq 2F-85 pincers, floating base; 40 home tasks × 50 human VR demos, replayable joint-position actions, head + 2 wrist RGB-D (**verified**) | ✅ | ❌ → 🔧 | ❌ → ✅ cube pads + dense skin (built) | ✅ | ✅ | Apache-2.0; some assets CC BY-NC | pincer-side expert source; 8 tasks held out as the main few-shot suite |
| **LocoMuJoCo** v1 | H1 mocap: default + LAFAN1 (>22k samples per humanoid), AMASS separately; kinematic only, no actions (**verified**) | 🔧 render | ❌ → 🔧 | 🔧 skin | ✅ | 🔧 position targets | code MIT; data CC BY-NC-ND | locomotion source (kinematic replay); held-out motions |
| **RoboPianist + RP1M** | 2 Shadow hands + 88-key piano; RP1M ~1M expert trajectories, ~2k pieces (**verified**, see §2) | ✅ | ✅ notes | 🟡 → 🔧 whole-hand taxels | ✅ | ✅ | Apache-2.0 / CC BY 4.0 | hand-side expert source; OOD songs held out; **built** (hands only, to mount on the H1) |
| Playground scene (ours, parked) | tabletop objects, 4 materials, scripted reach/push/grasp/sort/stack/roll, contact sound | ✅ | 🔧 contact | ✅ taxels | ✅ | ✅ | ours | hand-side play source; *sort*, *roll* held out |
| **MetaDrive**, **gym-pybullet-drones** | car, quadrotor (**built**) | ✅ | 🔧 synth | n/a | ✅ | ✅ | Apache-2.0, MIT | alien-body ablation only |

Not used in E1 (other bodies or no data): Meta-World, DM Control, TD-MPC2
checkpoints, Adroit/Minari, MyoSuite/MyoDM, Bi-DexHands (Isaac Gym),
ManiSkill3, MuJoCo Playground, Isaac Lab.

## 2. Later bodies and datasets

| Item | Content | Senses | Use |
|---|---|---|---|
| **MIMo v2** (**verified**) | infant 0–24 months: binocular 256² foveated eyes, full-body skin (count unpublished), gyro + accel vestibular, sensorimotor delays, Adroit-based hands; envs reach, standup, selfbody, catch, roll_over; no released policies; MIT | V, T, vestibular, proprio | **E2** (H3, development) |
| **RP1M** (CoRL 2024; RP1M++ 2025) | CC BY 4.0 on Edmond, doi:10.17617/3.XCE8NX: toy 0.35 GB, 300-song 36 GB, all 252 GB. Per song: `[500 traj, 550 steps]` of actions (39), hand joints (46), fingertips (30), key states (89). Re-rendered kinematically (`bodies/piano.py`). | sim state | E1 (above) |
| PIG | piano fingering MIDI | symbolic | RoboPianist songs (download separately) |
| **ObjectFolder / ObjectFolder Real** | 1,000 neural objects (vision/impact audio/touch); 100 real objects | V, A, T | E3: objects with sensory signatures for C5 (reading → senses) |
| TVL, Octopi/PhysiCLEAR | touch–vision–language pairs | V, T, text | E3 |
| Ego-Exo4D, EPIC-Kitchens | egocentric human video, audio, IMU | video, audio, IMU | E4 human data |
| Sparsh-skin / Sparsh-X data, VTouch++, RH20T | real tactile and audio robot data | T, A | E4 real tactile targets |
| New York Smells; SmellNet / See & Sniff; COLIP-2 | e-nose readings ↔ images/text | smell, V | E4 |
| Sonicverse, SoundSpaces 2.0 | physics-based audio in scenes | A | later: physics-based contact audio |
| TACTO / Taxim | tactile-image rendering (DIGIT/GelSight) | tactile images | later, if taxel forces prove too coarse |

## 2b. External benchmark data

### Push-T, as LeWorldModel uses it (2026-10-26)

The comparison with LeWorldModel (LeWM, arXiv 2603.19312) at matched tiny size, on the CPU. The
source and its terms, all **verified** that day:

| Item | Value |
|---|---|
| Source | Hugging Face dataset `quentinll/lewm-pusht`: `pusht_expert_train.h5.zst`, 13.1 GB compressed. It is the data of LeWM's code (github.com/lucas-maes/le-wm, MIT), and the same expert trajectories DINO-WM uses (arXiv 2411.04983) |
| Licence | MIT (dataset card), code MIT. Push-T comes from Diffusion Policy (arXiv 2303.04137) |
| Content | 18,685 expert episodes, 2,336,736 steps (125 on average), 10 Hz, from the file's header; the paper says 20,000 episodes of about 196 steps. Decompressed 46.3 GB, `pixels` behind a plugin compression filter. Stable-worldmodel's HDF5 layout: `pixels` (224 x 224 RGB), `action` (2-D agent velocity command), `proprio` (agent x, y, vx, vy), `state` (agent position, block position and angle, agent velocity), `ep_len`, `ep_offset` |
| Converter | `scripts/pusht_convert.py`, `sensorium.pusht`; needs `--extra pusht` (h5py, hdf5plugin) |
| Our episodes | body `pusht`, dt 0.1 s (the env's 10 Hz step; the model's step is 5 of them), one per source episode: 256 px frames (bilinear from 224), proprio `agent_x`, `agent_y`, `agent_vx`, `agent_vy`, actions `act_vx`, `act_vy` (relative velocity, Box(-1, 1), stored as they are; any outside [-1, 1] clipped and counted in `meta`), no touch, silent audio, `privileged/state` (as in the source) and `privileged/block_pose`. `meta` records the source episode, the code commit and the frame paths |
| LeWM's copy | the same frames at 64 px, made from our stored episodes with our loader's resize (`pusht.lewm_frames`), in LeWM's HDF5 layout with its own unscaled columns. A full-frame view (`View(size=64, crop=1.0)`) gives our model the identical pixels |
| Split | LeWM's split is over clips, not episodes: 4 observations frame-skip 5 apart, 90/10, torch generator seed 3072 (`le-wm/train.py`). `split` writes it to JSON as `[episode, start]` clips: observations at `start + 5k` (k < 4), actions `start` to `start + 19`. Our model uses the same clips |
| Training subset | `subset` draws a fixed list of source episodes (`np.sort(default_rng(0).choice(n, 2000))` for the 2,000 of the B-pusht-0 pre-registration); `select` writes LeWM's 64 px file of those episodes in that order, and `split --episodes` the clip split over them, as LeWM's `train.py` computes it on that file |

**What the converted data shows (2026-10-26, converter at 01cbb93, all 18,685 episodes):**

| Item | Value |
|---|---|
| Episodes, steps | 18,685 episodes, 2,336,736 steps at 10 Hz; length min 49, median 123, mean 125.1, max 246 |
| LeWM clips | all episodes: 1,783,549 train / 198,172 validation; the 2,000-episode subset: 190,550 / 21,172 |
| Actions outside [-1, 1] | 2,909 of 4,673,472 values (0.062%), up to 2.04 in magnitude: clipped in our episodes (`meta["actions_clipped"]`), raw in LeWM's copy. 98% of values lie within about ±0.55 |
| Agent position | workspace pixels, nominally 0-512; 1st to 99th percentile 38-482, extremes -153 to 724 |

- **185 demonstrations, each replayed about 101 times.**
  - Read in file order, the episode lengths form 183 runs of 101 consecutive episodes of identical length; two runs are 202 long (two demonstrations of equal length that happen to be adjacent). That makes 185 x 101 = 18,685 exactly.
  - This fits DINO-WM's description of the data: released expert trajectories replayed with noise.
  - Consequences:
    - Every validation clip, under LeWM's clip split or any episode-level split, has near-duplicate siblings in training. Validation loss measures fit to the 185 demonstrations, not generalisation to new ones.
    - LeWM's planning evaluation draws its start-goal pairs from the same file, so its starts and goals lie on training demonstrations: the evaluation is in-distribution, for every model alike.
    - The 2,000-episode training subset covers all 183 runs (about 11 replays of each demonstration), so it keeps the full set's behavioural diversity.
- **The T's angle stays readable at 64 px.** Frames of three episodes were compared, 8 per episode, at 224 px and at the 64 px both models see. The T's orientation, its stem against its bar, the goal T and the agent's dot are visible in every 64 px frame. The T is about 4 px thick and 25-30 px long there, enough to resolve angles to a few degrees against the success rule's 20 degrees. Only edges soften, such as the joint of stem and bar.

The data and the converted episodes stay in cloud storage and under `data/`; none of it goes into
this repository.

## 3. Coverage matrix for E1 (one H1 body)

| Part | RGB | Audio | Touch | Vestibular | Proprio | Actions | Expert behaviour | Babbling |
|---|---|---|---|---|---|---|---|---|
| Body (legs, torso, arms, head) | ✅ 2 head cameras | 🔧 contact synth | ✅ `h1touch` body grids | ✅ pelvis IMU | ✅ | ✅ | ✅ mocap (kinematic) | ✅ reach policies (standing) |
| Shadow hands | (same) | 🔧 contact + ✅ piano notes | ✅ hand grids | — | ✅ | ✅ | ✅ RP1M; scripted play | ✅ seated + goal babbling |
| Robotiq pincers | (same) | 🔧 contact | ✅ dense pad skin | — | ✅ | ✅ | ✅ BiGym demos | 🔧 pincer babbling |
| Language labels | | | | | | | 🔧 templated from privileged state | **manual gold set (E3)** |

## 4. What's missing (and the plan for each)

1. **No unified interface across the sources.** HumanoidBench, BiGym,
   LocoMuJoCo and RoboPianist each have their own observation/action spec and
   their own copy of the H1/hand models. → One H1 model in our repo (from
   HumanoidBench's, with `h1touch`), one episode format
   ([data-generation.md](data-generation.md#episode-format)), and one adapter
   per source that *replays* its data onto our model.
2. **No simulator renders sound.** → Contact sound synthesised from taxel
   impacts and sliding (built); piano notes from key state (built). Known
   weakness: deterministic function of state (E1 risks). Physics-based
   audio (modal synthesis) is a later step.
3. **BiGym has no touch and a different end effector.** → Taxels added to the
   pincer pads with the same `touch_grid` plugin; the effector controls in
   [experiment-01.md](experiment-01.md#the-two-end-effectors).
4. **Locomotion data has no actions.** → Kinematic replay with
   position-target actions, the same caveat as piano.
5. **No demos for HumanoidBench's tasks.** → Used as privileged-state probes
   only; few-shot targets come from BiGym, RP1M and our scripted hand tasks.
6. **Mocap licences are non-commercial / no-derivatives.** → Locomotion
   episodes are internal in v0; the public release ships the generator.
7. **No language descriptions of sensorimotor state.** → Templated
   descriptions from *privileged* state (never fed to the model), plus a
   manually written gold set (E3).
8. **Craft, flight, driving as tasks for this body; smell.** → No simulators;
   out of scope until one exists (E4 for smell with real e-nose data).
