# A learned hand grasp: scope and price (2026-10-22, proposal, nothing run)

**Why.** The planned claw pinch lifts about 20% of the hand's grasps on the event-dense table. The
reasons are in `docs/touch.md`:
- The failures come from the grasp's geometry, not from sensing or wrist range: every claw pose is
  reachable, and a touch-closed loop did not help.
- A hand-designed side wrap lifted 7%.

A grasp learned with reinforcement learning in simulation is the remaining route. This page prices
it for the supervisor's decision. The policy only generates data: like the planner, it may use
privileged state, while the physics stays honest (no welds, real contact forces).

## What published work does

| work | hand, arm | simulator | observations | compute |
|---|---|---|---|---|
| MuJoCo Playground, LEAP cube reorientation [1] | LEAP hand (16 DoF), fixed base | MJX (JAX, GPU) | asymmetric actor-critic: noisy proprio and cube pose for the actor, true state, fingertips and forces for the critic | 200M + 100M steps; about 35 min on one RTX 4090, 11 min on 8 H100; sim-to-real |
| MuJoCo Playground, Franka PickCube [1] | parallel gripper, 7-DoF arm | MJX | state, or 64x64 RGB | 10 min on one RTX 4090 (pixels); 140k env steps/s on an A100 (state) |
| DexPBT [2] | Allegro (16 DoF) on a 7-DoF Kuka; the policy drives arm and fingers | Isaac Gym | asymmetric: the critic gets everything, the actor a subset | 8192 envs per V100, 50k samples/s; 5e9 transitions = 30 h per agent; population of 8 agents (240 GPU-h) |
| UniDexGrasp / UniDexGrasp++ [3, 4] | Shadow hand, floating (no arm) | Isaac Gym | state-based teacher with object curriculum, distilled into a point-cloud student (DAgger) | not given in the abstracts *(verify)*; 85% train / 78% test success over thousands of objects |
| OpenAI, in-hand manipulation [5] | Shadow hand, fixed | MuJoCo (CPU) | asymmetric; vision student | thousands of CPU cores and tens of GPU-hours a day for days *(verify)*: not our scale |

There is no Shadow-hand task in MuJoCo Playground [1]. The MuJoCo Menagerie has a Shadow hand
model; whether it is MJX-ready, with collision proxies, is *(verify)*.

## Recommended setup: our arm in the loop

- **Floating hand, arm doing IK to the learned pre-grasp (UniDexGrasp style).** Rejected: the
  learned wrist poses would need the 5-joint H1 arm to reach them. `scripts/hand_reach.py` shows how
  much that limits: a level side wrap is reachable at 12% of placements. The policy would learn
  grasps the body cannot make.
- **Arm in the loop (DexPBT style).** Recommended. The planner brings the hand over the object, as
  now (IK, privileged object pose). The policy then drives the 5 arm joints (as residuals on the
  planned posture) and the Shadow hand's actuators, for the last 10 cm, the closure and the lift.
  It only learns poses the body can reach.
- **The training scene is cut down for MJX:**
  - one H1 arm, the Shadow hand and the table, with one of the 8 table objects;
  - hidden properties randomised in the dataset's ranges (mass 0.1-1.5 kg, friction 0.2-1.0,
    softness);
  - the skin plugin left out (MJX cannot run it, and the policy does not need it).

  The trained policy then runs in the full MuJoCo C scene during data generation.
- **Observations.**
  - Teacher: arm and hand joints and velocities, the object's pose, size and velocity, fingertip
    positions, contact forces, and the hidden properties for the critic only.
  - Distillation to proprio plus touch plus an estimated object pose is optional: data generation
    may use privileged state. It is only worth doing if the policy is ever to run on a real robot.
- **Reward:** reach the object, finger-object contact, object height, hold for 0.5 s, and penalties
  on penetration and on forces near the reflex limit. Success is our own-lift criterion.

## Compute and cost (one L4)

**Throughput.** It is not measured, so ranges follow.
- LEAP cube reorientation runs about 300M steps in 35 min on an RTX 4090 [1]: about 140k steps/s.
- An L4 has about a third of a 4090's FP32 throughput, so about 30-50k steps/s on that task.
- A 24-DoF Shadow hand, an arm and more contacts make each step 3-6 times dearer: **5-15k steps/s
  on one L4**.

**Steps.** The comparisons above:
- floating-hand grasping with a curriculum: of order 1e9 steps *(verify against [3, 4])*;
- arm-and-hand tasks: 5e9 [2], harder than grasp-and-lift.

So 1-2e9 steps per full run, and about 2e8 for a single-object pilot.

**Price.** L4 at $0.85/h on demand (GCP g2-standard-8, us-central1), Spot $0.34-0.71/h [6].

| stage | GPU-hours on one L4 | on demand | Spot |
|---|---|---|---|
| throughput benchmark of the MJX scene (collapses the ranges below) | 1 | $1 | under $1 |
| pilot: one object, PPO sanity, reward shaping | 4-11 | $3-10 | $2-8 |
| full run: 8 objects, randomised properties, 1-2e9 steps | 20-110 | $17-95 | $7-80 |
| programme: pilot, 3 full runs (seeds, reward fixes), optional distillation (+20%) | 80-400 | $70-340 | $30-280 |

**Engineering** (Mac, no spend): 1-2 weeks for the MJX scene, collision proxies, reward, PPO
(Brax or the Playground's trainer), and running the policy in the C scene.

## Risks

- **MJX contact cost.** The Shadow hand's mesh collisions are slow in MJX. Capsule proxies are
  needed, and they shift contact geometry from the C scene the data comes from.
- **Transfer from MJX to the MuJoCo C scene.** Solver and contact settings differ (our pads and
  fingertips use solref 0.004), and the full scene has props and a second arm. A policy can lose
  success when moved; it should be checked with the isolated trials of `scripts/grasp_library.py`.
- **Strength.** At the Shadow effort limits the hand cannot lift every mass up to 1.5 kg. A learned
  grasp helps the geometry, not the actuators.
- **Exploration and reward hacking.** Pushing objects up with penetration, or flinging them, can
  pass a rise criterion. Honest physics checks and the touch condition of our lift criterion guard
  against it. DexPBT needed population-based training for its harder tasks.
- **Range of the estimate.** Throughput and step counts are each uncertain by 3x: the 1-hour
  benchmark should come before any larger spend.
- **Data style.** A learned policy's movements differ from the babbling's (smoother or jerkier, other
  contact sequences). The dataset's statistics should be compared before mixing.

## Pre-registration: scripted grasp success with hand-sized objects (cloud)

**Question.** Does adding the `hand_sized` objects (`docs/event-dense.md`) raise the scripted hand's
grasp success on the event-dense table above its current level (20% [15, 25] per grasp, 1.04 own
lifts per episode, pooled over 55 episodes in `docs/touch.md`)?

**Runs.** One CPU machine; all runs use the same code and seeds, and count every planned grasp.
- **A, without the new objects:** `scripts/grasp_success.py hand 60 --first 0` with
  `--setting 'props=["button", "friction_patches", "hinged_flap", "tall_block"]'`.
- **B, with them:** `scripts/grasp_success.py hand 60 --first 0`, the config's hand block as
  committed, with `--strips` (frame strips of the first two grasps per object and cause).
- **C, isolated claw grasps:** `scripts/grasp_library.py hand <object> 100 --planner` for each of
  the four objects.

Both table runs cover 60 episodes (seeds 0-59), split over parallel processes. A 1-episode smoke
run of B checks the rendering before the rest.

**Measures.**
- Primary: success per grasp in B, overall and on the hand-sized objects, with Wilson 95%
  intervals.
- Secondary:
  - own lifts per episode, distinct objects lifted per episode, per-object lift and grasp counts,
    and the failure causes, in A and B;
  - the event rate of B (`sensorium.event_rate`, from 10 recorded episodes);
  - the lift share per object in C.
- Figures: the frame strips (successes and each failure cause per object) and a bar chart of
  per-object success in A and B.

**Decision, fixed now.**
- Keep the objects in generation if B's success per grasp has a lower Wilson bound above 20%, and
  B's own lifts per episode exceed A's.
- If not, revert the config change; the prop stays available.

About 300 grasps per run give intervals of about ±5 points. A 10-point difference is about 2.8
standard errors.

**Cost.** The Mac ran about 3.4 min per episode per process (hand, no rendering). Allowing 30%
slower cores on the cloud:
- 120 table episodes and 400 isolated trials are about 10 core-hours;
- on a 32-vCPU machine with 30 processes, about 25 minutes of runs plus about 15 of set-up: under
  1 hour.

## Pilot of a learned grasp

**Method.** As above:
- the planner brings the hand over the object;
- a PPO policy (MuJoCo Playground's Brax trainer) drives the 5 arm joints as residuals and the hand
  actuators, from privileged state;
- training runs in an MJX scene of one arm, the hand, the table and one object, with hidden
  properties randomised.

**Steps.**
1. Throughput benchmark, 1 GPU-hour. It fixes the step budget, and stops the pilot if the scene
   runs under 3k steps/s.
2. Pilot on one object, the hand-sized can (the claw's best case, so the comparison is fair):
   2e8 steps, at most 2 reward iterations.
3. Check in the full scene: the trained policy runs 100 isolated grasps of the can in the MuJoCo C
   scene (as `scripts/grasp_library.py`, honest physics, the same lift criterion), with frame
   strips.

**Budget.** At most 15 GPU-hours on one L4 (about $13 on demand, $5-11 Spot), plus under 1 CPU-hour
for the check.

**Success criterion.** In the C scene the policy lifts at least 60% of the 100 can grasps, and at
least 20 points more than the scripted claw's isolated rate on the can (run C above).

**Go / no-go for the full programme** (8 objects, 80-400 GPU-hours):
- Go if the criterion is met within the budget.
- No-go if the C-scene rate is under 30% after the two reward iterations, or if MJX-to-C transfer
  loses more than half of the MJX success. The hands then stay with the scripted claw on hand-sized
  objects.
- In between, report and decide.

## Sources

1. MuJoCo Playground, Zakka et al., 2025. https://arxiv.org/abs/2502.08844, Sec. IV-C, App. C,
   Table IX.
2. DexPBT, Petrenko et al., RSS 2023. https://arxiv.org/abs/2305.12127, Sec. IV.
3. UniDexGrasp, Xu et al., CVPR 2023. https://arxiv.org/abs/2303.00938
4. UniDexGrasp++, Wan et al., ICCV 2023. https://arxiv.org/abs/2304.00464
5. Learning Dexterous In-Hand Manipulation, OpenAI, 2018. https://arxiv.org/abs/1808.00177
6. GCP g2-standard-8 pricing (third-party listings, October 2026; check the console before
   spending): https://gcloud-compute.com/g2-standard-8.html
