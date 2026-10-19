# D0: Experientia, a whole-body multisensory experience dataset on one humanoid, with controlled experience kinds

**Purpose:** the data to build, and fairly evaluate, a general *sensory
layer* for one humanoid body. Its defining features are that
**the senses, the kind of experience (expert vs task-free) and the number of
sources are dials we control**, at fixed volume, which no real-world dataset
can offer, and that whole-body touch *and* contact sound are present in every episode.

## Should we build it? Recommendation: yes (v0 internal, v1 public)

- **E1 needs it anyway.** Testing H1 (senses) and H2 (kind of experience)
  requires a pool where senses can be dropped, experience kinds swapped and
  sources counted while data volume stays fixed, plus a held-out few-shot
  benchmark. Nothing public offers that (see the table below).
- **It's cheap.** Everything is simulated and generated on CPU. 20M
  whole-body steps cost ~$15–45 of rented CPU, or a few days on the Mac.
  Storage is a few dollars a month.
- **It's a contribution by itself**, even if the hypotheses fail: a
  regenerable, whole-body multisensory interaction benchmark on a humanoid.
- **Risks:** sim-only data; sound is synthesised from contacts rather than
  physically rendered; maintenance burden after release. Mitigations: real
  datasets (ARIO, RH20T) as E4 transfer targets; physics-based contact audio
  is on the roadmap (E3).

## How it differs from what exists

| Dataset | Real/sim | Senses | Bodies | Diversity controlled? | Few-shot benchmark? | Regenerable? |
|---|---|---|---|---|---|---|
| ARIO (2024) | **real + sim mix** (77% converted, 23% sim, 0.12% own real) | vision, 3D, proprio, text; audio in ~0.04% of episodes, touch only in the RH20T subset | many robots | ❌ | ❌ | ❌ |
| RH20T (2023) | real | vision, force, audio, action | several arms | ❌ | one-shot skills | ❌ |
| VTouch++ (2026) | real | RGB-D, proprio, visuo-tactile end effectors; no audio | dual-arm fixed/wheeled platforms, UMI handheld grippers | ❌ | ❌ | ❌ |
| VinT-6D (Dec 2024) | sim + real | vision, touch, proprio | hands | ❌ | ❌ | partly |
| Open X-Embodiment / DROID | real | vision, proprio (mostly) | 20+ robots / 1 robot | ❌ | ❌ | ❌ |
| MMBench (Newt, 2025) | sim | vision, proprio, language | many | ❌ | finetuning on held-out tasks | ✅ |
| CortexBench (VC-1, 2023) | sim | vision only | 7 benchmarks | ❌ | ✅ (frozen-encoder eval) | ✅ |
| MyoDM (MyoDex, 2023) | sim | proprio + object state | one muscle hand | ❌ | ❌ (fine-tuning, ~8M samples per task) | env code only, no checkpoints |
| Tactile Genesis (2026) | sim | vision, whole-hand touch, contact audio, temperature | one hand | ❌ | ❌ (per-task RL) | ✅ |
| HumanoidBench (2024) | sim | vision, 448-taxel touch, proprio | one humanoid | ❌ | ❌ (RL, no demos) | ✅ |
| BiGym (2024) | sim | RGB-D, proprio; no touch | one humanoid, pincers | ❌ | 50 VR demos / task | ✅ |
| **Experientia** | sim | **vision, sound, whole-body touch, vestibular, proprio, action** | **one humanoid → infant** | ✅ **senses, experience kind and sources, at equal volume** | ✅ **held-out songs, home tasks, motions, play tasks** | ✅ seeds, pinned sims, content hashes |

## Contents (v0 target: ~80 experience sources, ~20M steps)

An **experience source** is one (effector, task or motion set) pair on the
H1 body, recorded under **three kinds of experience**: *expert* (released
demonstrations or mocap, replayed), *noisy expert* (expert + Ornstein–Uhlenbeck
noise, built) and *babbling* (task-free, "reflex-and-primitive" babbling with no
task reward: seated reaches, touches and grasp reflexes aimed at objects;
whole-body free-space reaching through HumanoidBench's released reach policy;
with contact-quality and fall monitoring, see [A1 in tracks.md](tracks.md)). Every source uses the same episode
format (`src/sensorium/episode.py`): 256² egocentric RGB from the head
cameras, sound with log-mel features, 448-taxel skin, pelvis IMU,
proprioception and actions, plus privileged state and templated text that
never reach the model.

**One sensor suite for the body.** The skin is HumanoidBench's `h1touch`
layout (MuJoCo `touch_grid`, 3-D force per taxel, dense on the hands); the
pincer variant gets dense skin on cube-built pads ([touch.md](touch.md)). Contact sound
is synthesised from taxel impacts and sliding (built, `audio.py`); piano adds
note audio.

| Family | Effector / tasks | Expert behaviour from | Kind(s) | Status |
|---|---|---|---|---|
| Piano | Shadow hands, seated at the piano; RP1M songs grouped into ~30 song-cluster sources; arm IK places the hands | RP1M (CC BY 4.0), re-rendered | expert, noisy | ✅ hands built; mount on H1 + arm IK to do |
| Free-space reaching (`reach_free`) | Shadow hands, standing; random 3-D reach targets, whole-body motion, contact near 0 by design | HumanoidBench `reach_two_hands` policy (MIT), a 61 → 19 MLP ported to numpy, ticked at 50 Hz inside the 20 Hz steps; falls in 15-20% of 15 s episodes in our scene (20 per effector; median drift 0.44 m, the pelvis crouches to 0.67 m), so such episodes end early; `configs/reach_free.toml` | babbling | ✅ |
| Seated babbling | Shadow hands or pincers, pelvis fixed, table at 0.82 m with six objects (within |y| ≤ 0.30 m, goals and hovers kept in the head camera's view); count-based choice of touch, poke, slide, hover, grasp-lift-drop (pincer) or power closure (hand); collision-checked, withdrawal reflex | — | babbling | ✅ (pincers lift ~1.7 times per 30 s; hands do not lift yet) |
| Supported standing (`stand_supported`) | pelvis welded at standing height (0.98 m), table at 1.16 m (the seated arm geometry shifted up), same primitives as seated; its own source `stand_table` (split v2) | — | babbling | ✅ |
| Home tasks | **Pincers**, floating base; 32 of BiGym's 40 tasks | BiGym human VR demos (Apache-2.0; 123 MB), replayed from actions, successes kept | expert, noisy, + pincer babbling | 🔧 adapter |
| Locomotion | hands idle; walk, run, turns, LAFAN1 motions | LocoMuJoCo H1 mocap (code MIT; datasets CC BY-NC-ND), kinematic replay | expert, noisy | 🔧 adapter |
| Tabletop play | Shadow hands on the parked playground scene: reach, push, grasp, sort, stack, roll | scripted + babbling | all | 🟡 parked branch (gantry gripper): swap in the H1 arm + hand |
| Alien bodies *(ablation only)* | car (MetaDrive), quadrotor (PyBullet) | bundled PPO expert, PID | expert, noisy, explore | ✅ built |

Not in E1: HumanoidBench's 27 tasks as *few-shot targets* (no demos exist;
they serve as privileged-state probes), MIMo (E2), Meta-World, DM Control,
Adroit, MyoDM (other bodies).

Experts are reused, never retrained, wherever someone has released them.

## Splits (fixed before any training)

- **Pretraining pool:** all sources except the held-out ones below.
- **Held-out benchmark** (never seen in pretraining), tagged *contact-rich*
  or *vision-dominated* for H1, and reported per side:
  1. **Pincer side:** 8 BiGym tasks held out entirely, with their 50 human
     demos as the k-shot pool (e.g. contact-rich: drawer, dishwasher
     loading; vision-dominated: reach-and-place).
  2. **Hand side:** RP1M's out-of-distribution song split (contact-rich);
     tabletop *sort* and *roll*; 2–3 scripted hand tasks on the standing
     body (lift box, push cube) with demos we generate.
  3. **Body:** held-out mocap motions, scored by tracking error after
     few-shot adaptation (not success).
- **Controlled pools** for E1, all with the **same number of steps**:
  - **Senses:** the same episodes with touch, sound or proprio dropped at
    load time (one dataset, four views).
  - **Kind of experience:** expert · babbling · mix (⅓ each), drawn from the
    same sources.
  - **Diversity:** 8 sources vs all, nested, defined by a seed.
  - **End effector:** hands-only · pincers-only · both, at equal steps; the
    hand : pincer : body-only ratio is fixed in every other pool.
  - **Alien bodies:** the pool with and without car + drone.

## Few-shot evaluation protocol (the benchmark)

- **Input:** any encoder that implements `encode(model_inputs) -> latent`.
- **Two adaptations:** (a) encoder frozen, small head (behaviour cloning,
  2-layer MLP); (b) LoRA (rank 8) on the trunk, stems trainable, same head.
- **Per held-out task:** k ∈ {1, 5, 10, 50} demonstrations, with 3 seeds.
- **Metrics:**
  - task success or normalised score
  - RP1M's key-press F1 for piano
  - linear-probe accuracy on privileged state (contacts, object pose, key
    state): a cheap check of representation quality
- **Baselines run on the same protocol:**
  - from scratch (same head plus a trainable encoder)
  - VC-1, DINOv2, V-JEPA 2 (vision only)
  - HPT pretrained on the same pool

## Format and release

- **Storage:**
  - Episodes are `.npz` files in the format that already exists.
  - Training reads them packed into tar shards (WebDataset-style).
  - A parquet index lists every episode: source, body, task, policy, seed,
    length and content hash.
- **Reproducibility:**
  - Every source is a config file plus a generator.
  - Simulators are pinned.
  - The manifest records versions and content hashes, so the data can be
    regenerated on the same platform.
- **Release (v1):** Hugging Face Hub plus the generators. Public storage on
  a free account is "best-effort" beyond the first few GB; a 2 TB v1 needs a
  PRO account ($9/month, up to 10 TB public) or a storage grant. Limits: 500 GB
  per file, <10k files per folder, so shard. Licences (verified 2026-10-11):
  RP1M CC BY 4.0 (Edmond record; needs attribution), RoboPianist Apache-2.0,
  HumanoidBench MIT (Menagerie hand assets Apache-2.0), BiGym Apache-2.0
  (3D assets CC0 / CC BY 4.0 / **CC BY-NC 4.0**: check per asset before a
  public release), LocoMuJoCo code MIT but its mocap datasets are
  **CC BY-NC-ND 4.0** and LAFAN1/AMASS carry their own non-commercial terms,
  so rendered locomotion episodes may be **v0-internal only**: v1 ships the
  generator and the user downloads the mocap themselves. MetaDrive Apache-2.0
  (code; asset licences unchecked), gym-pybullet-drones MIT, MIMo MIT, HPT
  MIT. Unstated: HPT weights.

## Size and cost

| | v0 (internal) | v1 (public) |
|---|---|---|
| Sources | ~80 (one humanoid) | 150+ (+ MIMo) |
| Steps | ~20M (~280 h of experience at mixed rates) | 100M+ |
| Disk (256² RGB as H.264, ~7 KB per step with both eyes) | ~150 GB | ~1 TB |
| CPU to generate | ~200–600 core-hours (whole body, 448 taxels, 2 cameras; ~21 steps/s per process on the Mac) → **~$15–45** rented, or a few days on the Mac. Needs GPU rendering (EGL): software rendering would take ~2,900 core-hours | ~$80–200 |
| Storage (Cloudflare R2) | ~$5–8 / month | ~$30 / month, or HF Hub PRO ($9 / month) once public |

## Milestones

| M | Deliverable | Effort |
|---|---|---|
| D0.1 | Source registry, pool builder (senses / experience kind / sources / effector ratio at equal steps), split file, parquet index | ~1 week |
| D0.2 | H1 body with `h1touch` skin + head cameras in our episode format; contact sound from the skin; pincer taxels | ~1 week |
| D0.3 | Seated babbling: collision-checked reaches and A1 statistics ✅, primitives (poke, slide, grasp-lift-drop) ✅; `reach_free` and `stand_supported` (own source `stand_table`, split v2) ✅; A1 gate (`make a1-gate`) ✅; fall/contact monitoring | ~1 week |
| D0.4 | BiGym adapter (replay, success filter) + LocoMuJoCo adapter (kinematic replay) | ~1–2 weeks |
| D0.5 | Piano on the H1 (arm IK, hand mounting); playground with the H1 arm | ~1 week |
| D0.6 | Benchmark harness (encoder API, frozen + LoRA heads, probes incl. variant probe) + vision-only baselines; BiGym retargeting side check | ~1 week |
| D0.7 | Per-source review pages and stats; **your review** before E1 training | ~2 days |

**Total: ~7–8 weeks part-time.** That is most of the engineering E1 needs anyway.
