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
