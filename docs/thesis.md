# Research thesis

## Goal

> **Can we build a general *sensory layer* from a plethora of different
> experiences, one that turns task-specific learning into few-shot learning?**

A **sensory layer** is a pretrained encoder that maps raw multisensory streams
(vision, sound, touch, proprioception) together with the actions taken into
one latent state. It is learned without rewards, labels or text, by predicting
how the senses change when the body acts. A new task is then learned from a
handful of examples, by training only a small head on the frozen layer, or by
a light LoRA adaptation of it.

The bet: what makes such a layer general is the **variety of sensory
contact** it was built from, more than the number of skills it was shown.
Children don't learn to grasp a cup from a million cup grasps, nor from a
curriculum of expert demonstrations; they learn from a huge variety of little
interactions, most of them aimless.

Language is not the substrate of this knowledge. It is a **readout**: a way
to inspect the latent state, and later a way to extend it.

## Body: one humanoid, whole body, all senses

The programme is built on **one body**: the Unitree H1 humanoid as shipped
by HumanoidBench, with two egocentric head cameras, **skin over the whole
body** (HumanoidBench's 184-taxel `touch_grid` on the body, ~1,200 spread
sensor points per hand, onset / slip / vibration per region; see
[touch.md](touch.md)), a pelvis IMU as
the vestibular sense, proprioception of every joint, and contact sound
synthesised from the skin. Later steps grow the body backwards in time (the
MIMo infant: smaller, weaker, foveated eyes, full-body skin), with the senses
keeping the same meaning.

**Two end effectors, one body.** From the wrists inward the body is always
the same. At the wrists it carries either **five-finger Shadow hands** (the
same Menagerie model RoboPianist uses) or **two-finger Robotiq pincers**.
The pincer variant exists for one reason: BiGym's 40 home tasks with 50
human VR demonstrations each were recorded on it, and that is the only
large set of human demonstrations for this body. The two variants are used
with explicit controls (see [experiment-01.md](experiment-01.md#the-two-end-effectors)):
the mix is held constant across every condition, a probe measures whether
the latent separates them, hand-side and pincer-side results are reported
separately, and cross-effector transfer is measured as its own ablation.

Why one body, not many (decided 2026-10-11, see
[literature-review.md](literature-review.md)):
- A human does all of driving, flying and piano with one body; the car and the
  plane are tools. The thesis is about one agent's accumulated experience.
- Cross-body transfer tracks *appearance* more than kinematics (XEWorld),
  so a pool of differently rendered bodies confounds any body-transfer result.
- One sensor suite means touch and sound mean the same thing in every source,
  with one set of input stems.
- The real-data path (E4: egocentric human video, real tactile datasets) is a
  human body.

Why the whole body and not just hands (decided 2026-10-11): the agent is the
body; hands are where most of the touch happens, not the unit of
experience. Locomotion, posture, balance and reaching are experience too, and
the skin covers all of it.

The car and drone sources that already exist remain as an **"alien body"
ablation**: does adding non-human bodies to the pool help or hurt?

## Hypotheses

| # | Hypothesis | Tested in |
|---|---|---|
| **H1** | **Senses beyond vision.** At matched data and compute, a layer pretrained with touch and contact sound beats a vision-only layer on contact-rich held-out tasks at k ≤ 10 demonstrations, and matches or beats frozen vision foundation encoders (VC-1, DINOv2, V-JEPA 2) that saw far more data. | E1 |
| **H2** | **Kind of experience.** At equal volume, a layer pretrained on *task-free* interaction (babbling, noisy behaviour) is a usable sensory layer: it beats training from scratch and the vision-only encoders on few-shot tasks, and a mix of task-free and expert experience beats either alone. | E1 |
| **H3** | **Development.** The layer transfers between the adult body and the infant body (adult → MIMo tasks; MIMo babbling → adult tasks) with new stems and a LoRA adaptation, faster than from scratch. | E2 |
| H4 | *(diagnostic)* **Language reads out the latent.** A tiny decoder on the frozen layer describes held-out states above a template-only baseline, and tells us which senses the latent encodes. | E3 probe |
| C5 | *(later)* **Reading extends grounded knowledge.** Text about an unseen object lets the layer predict its touch and sound signature before first contact. | E3 |

**Diversity** (number of distinct sources) stays in E1 as a *secondary*
factor with two levels, to check that H1 and H2 are not artefacts of one
pool. It is no longer the headline: multi-task pretraining on a hand is
established (MyoDex, below), and a rising diversity curve would surprise
nobody.

## Evidence per hypothesis: what is known, and why each is still open

Numbers below were checked against the primary sources on 2026-10-11
(tables and figures named). Thread-by-thread notes:
[literature-review.md](literature-review.md).

### H1: senses beyond vision

**Closest evidence**

| Work | Setting | Finding |
|---|---|---|
| **See, Hear, and Feel** (CoRL 2022; [2212.03858](https://arxiv.org/abs/2212.03858)) | one Franka Panda; camera + contact mic + GelSight; 2 tasks; 10 demos per scenario; trained per task | dense packing success (Table 1): vision 0.25 → +audio 0.60 → +touch 0.75 → **all three 1.00**; pouring error 40.9 g → 1.4 g (Table 3) |
| **SPOTS** ([2304.11193](https://arxiv.org/abs/2304.11193)) | joint vision–touch prediction | gains concentrate in "physically ambiguous interaction regimes" |
| **Visuo-Tactile World Models** ([2602.06001](https://arxiv.org/abs/2602.06001)) | one Panda + Allegro hand with 4 Digit 360; 8 tasks, 124 demos; frozen Cosmos tokenizer + fine-tuned Sparsh-X | real-robot planning gains of +10 to +35 points over vision-only (Fig. 7); a new task reaches 77% with 20 extra demos |
| **Sparsh-X** ([2506.14754](https://arxiv.org/abs/2506.14754)) | ~1M contacts (18.6 h) from one sensor model, Digit 360, **real** data; encoder frozen | +48% on physical-property estimation vs end-to-end; 90% plug insertion with ACT on 100 demos; baselines are end-to-end tactile models, **no vision foundation encoder** |
| **Tactile Genesis** ([2606.22332](https://arxiv.org/abs/2606.22332)) | simulated whole-hand taxels, contact audio, temperature; 20k parallel envs | whole-hand coverage ≫ fingertips, ~200 taxels suffice; **per-task teacher–student**, no pretrained encoder |
| **Datasets** ([dataset-proposal.md](dataset-proposal.md)) | ARIO: audio in ~0.04% of episodes; VTouch++: no audio; RH20T: audio + force on arms | no multisensory pool with touch *and* sound on a human-like body |

**Why H1 is not yet tested**

1. **No simulated pretraining pool has ever included touch and contact
   sound.** Tactile encoders exist only from real single-sensor data
   (Sparsh-skin, Sparsh-X, T-DEX); the one simulator with whole-hand touch and
   audio (Tactile Genesis) trains per task and learns no encoder.
2. **No matched comparison against vision foundation encoders.** Sparsh-X
   compares against end-to-end tactile models; VT-WM against its own
   vision-only ablation. Whether a multisensory layer beats VC-1, DINOv2 or
   V-JEPA 2 at the same few-shot budget is unmeasured.
3. **One body, one sensor model, every time.** Sparsh-X's own limitations
   say its tactile images have "the lowest diversity in terms of number of
   different devices used". Which senses matter *across many tasks* on one
   body has never been separated (touch vs sound vs vestibular vs proprio).
4. **Caveat on our side:** prior results use real sensors; our sound is
   synthesised from contacts. Audio and touch are ablated separately, and
   physics-based audio is on the roadmap.

### H2: kind of experience (task-free vs expert)

**Closest evidence**

| Work | Setting | Finding |
|---|---|---|
| **MyoDex** (ICML 2023; [2309.03130](https://arxiv.org/abs/2309.03130)) | MyoHand (23 joints, 39 muscles); **proprio + object state only**; multi-task PPO on 14 expert-trained tasks | 37/43 unseen tasks solved vs 22 from scratch, ~3× fewer iterations (1.9k vs 5.7k × 4,096 samples ≈ 8M samples per new task); zero-shot ≤ 10%; task count 6/14/18 → 72/95/87% (Table 2); diverse > homogeneous (Sec. 6.4.3); full fine-tuning; no checkpoints |
| **Arnold** ([2508.18066](https://arxiv.org/abs/2508.18066)) | 14 tasks across MyoElbow/Hand/Arm/Leg; proprio only | pretrain on 10, fine-tune on 4 held-out |
| **Data Scaling Laws** (Lin et al., ICLR 2025; [2410.18647](https://arxiv.org/abs/2410.18647)) | one UMI gripper, 4 tasks, >40k **expert** demos | power law in environments and objects; frozen DINOv2 0.00 vs LoRA 0.72 vs full fine-tune 0.90 (Table 2) |
| **Curious play** (Sancaktar et al., NeurIPS 2022, [2206.11403](https://arxiv.org/abs/2206.11403); Plan2Explore; Walker et al., ICML 2023) | reward-free exploration with a world model | zero-shot stacking, flipping, pick-and-place **in the same domain**; which components transfer is studied, but no frozen few-shot representation, no senses |
| **MIMo / BabyBench** | infant simulator; self-touch, hand regard from intrinsic rewards | single behaviours; no pretraining study |

**Why H2 is not yet tested**

1. **Every pretraining pool is made of expert behaviour.** MyoDex, Arnold,
   Lin et al., HPT, RP1M, Newt: all pretrain on demonstrations or trained
   experts. Nobody has asked whether *aimless* interaction at the same volume
   builds the same layer, nor whether mixing it in helps.
2. **Curiosity work never evaluates a frozen few-shot representation.** It
   trains a world model and plans zero-shot in the domain it explored; it
   does not measure demonstrations-to-competence on held-out tasks, and uses
   no touch or sound.
3. **Volume is confounded with task count** in every multi-task study (see
   the diversity notes below), so "more skills" and "more contact" have never
   been told apart.
4. **The few-shot regime is empty.** The smallest adaptation budgets are
   ~8M samples (MyoDex), 25 demos (VC-1), ~50 demos per pair (Lin et al.).

*Diversity, kept as a secondary factor:* diversity helps in every study (Lin
et al.'s power law; HPT's dataset-count scaling, Fig. 5b; embodiment scaling
in locomotion, [2505.05753](https://arxiv.org/abs/2505.05753); MyoDex Table
2), always with volume growing alongside. Two equal-volume levels in E1
reproduce the effect cheaply and anchor H1/H2 against it.

### H3: development (infant ↔ adult body)

**Closest evidence**

| Work | Bodies | Finding |
|---|---|---|
| **HPT** ([2409.20537](https://arxiv.org/abs/2409.20537)), **CrossFormer** ([2408.11812](https://arxiv.org/abs/2408.11812)) | 52 datasets, mostly arms; 20 embodiments | >20% after fine-tuning; vision + proprio only |
| **XEWorld** ([2608.05799](https://arxiv.org/abs/2608.05799)) | 5 bimanual arm setups, sim | 1-shot adaptation to Franka raises error on the seen UR5 by 69%; transfer follows **appearance** (r = 0.81), not kinematics (r = 0.55) |
| **H-Zero** ([2512.00971](https://arxiv.org/abs/2512.00971)), **Embodiment Scaling Laws** | humanoids; ~800 legged robots | near transfer, proprio only |
| **Arnold** | Myo bodies | muscle-level proprio only |
| **RP1M++** ([2511.02504](https://arxiv.org/abs/2511.02504)) | Shadow, Allegro, ORCA hands | nothing transferred between hands |
| **MIMo** ([2312.04318](https://arxiv.org/abs/2312.04318), v2 [2509.09805](https://arxiv.org/abs/2509.09805)) | infant: full-body skin, binocular, vestibular | no pretraining or transfer study exists |

**Why H3 is not yet tested:** non-visual senses have never crossed bodies
(every touch study is single-device); no adult-body layer has been
evaluated on an infant body or the reverse; and no task-free infant
experience has been tested as a pretraining source for adult skills.

### H4 (diagnostic) and C5: language

- **V-JEPA 2** ([2506.09985](https://arxiv.org/abs/2506.09985)): a video
  encoder aligned post hoc with an 8B LLM reaches state-of-the-art video QA
  (frozen 52.3 vs fine-tuned 54.4 average, Tables 6–7). No readout of robot
  states.
- **Grounded World Model** ([2604.11751](https://arxiv.org/abs/2604.11751)):
  a **frozen** language readout over a vision-only, vision-language-aligned
  latent; 87% of 288 WISER tasks vs 22% for ten fine-tuned VLAs.
- **Propositional probes** (ICLR 2025), **Dynalang** (ICML 2024).
- **Reading → senses:** ObjectFolder, TVL and MultiPLY pair objects with
  sound, touch and text, but none tests whether text about an unseen object
  lets a sensorimotor layer predict its sensory signature.

**Why it stays a diagnostic.** The large-scale readout is done for video and
vision-language latents. What no one has shown is a readout of a
**multisensory** latent from a layer trained with **no text**, nor a layer
whose touch and sound predictions improve from reading. The first is a probe
of what the layer encodes; the second is C5, the new part of E3.

**What is new, in one line:** a *multisensory* layer learned from *simulated
whole-body interaction on one humanoid* (locomotion, reaching, home
manipulation, piano, babbling), with the **senses** and the **kind of
experience** (task-free vs expert) as the controlled variables, evaluated
*frozen and LoRA-adapted, few-shot* (k ≤ 10) on held-out tasks and then on
the infant body.

## Falsifiable predictions

1. **H1:** on contact-rich held-out tasks (hand side and pincer side alike), the all-senses layer beats the
   vision-only layer at k ≤ 10 with 95% CIs over seeds excluding zero; the gap
   is smaller on vision-dominated tasks. If the gap is zero everywhere, our
   synthesised touch and sound carry no information the pixels lack.
2. **H1 vs baselines:** the all-senses layer, pretrained on < 300 h of
   simulated whole-body experience, matches or beats frozen VC-1 / DINOv2 / V-JEPA 2
   on the suite average at k ≤ 10.
3. **H2:** the babbling-only layer beats scratch and the vision-only encoders
   on the suite average; the expert + noisy + babbling mix beats expert-only
   at equal volume. If babbling-only is no better than scratch, task-free
   experience does not build a sensory layer at this scale.
4. **H3:** adult → MIMo adaptation is faster than scratch; adding MIMo
   babbling to the pool does not hurt adult few-shot. Zero gain in either
   direction is acceptable and publishable.
5. **Adaptation:** the gap between LoRA and training from scratch shrinks as
   k grows; the layer's value is at small k.

## Out of scope (for now)
- **Many bodies / embodiment distance:** car and drone are an ablation only.
- **Smell:** no simulator renders odour for agents. Revisit with real e-nose
  data in E4.
- **Fluent text:** text quality is not a goal, only a readout.
- **Real hardware:** rent-only policy. Real data comes from public datasets.

## Research roadmap

| Exp | Question | Status |
|---|---|---|
| **D0** | Build **Experientia**, a whole-body multisensory experience dataset on one humanoid with controlled senses, experience kinds and sources, plus a held-out few-shot benchmark | **Proposed:** [dataset-proposal.md](dataset-proposal.md) |
| **E1** | Senses × kind of experience on one humanoid (H1, H2; diversity and end-effector ablations; alien-body ablation) | **Planned:** [experiment-01.md](experiment-01.md) |
| E2 | Development: MIMo infant ↔ adult H1 (H3) | Sketch |
| E3 | Language: readout probe (H4) and reading → sensory prediction (C5) | Sketch |
| E4 | Real-world grounding from public data: egocentric human video (Ego-Exo4D, EPIC-Kitchens), real tactile datasets, e-nose data | Later |

Status, next steps and the decision log: [roadmap.md](roadmap.md).

Craft skills (carpentry bench), flight and driving as *tasks* for a
human-like body need simulators that don't exist; they return as held-out
tasks when one does.

## Closest work
- **Representations for embodied control:** VC-1 / CortexBench (vision only,
  17 tasks, no universal winner); R3M, MVP, VIP; HPT (heterogeneous stems with
  a shared trunk).
- **Humanoid suites:** HumanoidBench (H1 + Shadow hands, `h1touch` skin),
  BiGym (H1 + pincers, VR demos), LocoMuJoCo (mocap); Puppeteer and
  MaskedMimic (motion priors, no senses).
- **Hand pretraining:** MyoDex, SAR, Arnold (proprio only); Sparsh-skin,
  Sparsh-X, T-DEX (real tactile SSL); Tactile Genesis (sim touch + audio,
  per task).
- **World models:** DreamerV3; TD-MPC2; Newt; V-JEPA 2; DINO-WM;
  Visuo-Tactile World Models.
- **Play and curiosity:** Plan2Explore, Sancaktar et al., Walker et al.,
  MIMo / BabyBench.
- **Multisensory:** See-Hear-Feel, SPOTS, ObjectFolder, MultiPLY.
- **Datasets with several senses:** ARIO, RH20T, VTouch++, VinT-6D. None has
  touch *and* sound on a human-like body with controlled experience kinds. See
  [dataset-proposal.md](dataset-proposal.md).
- **Position papers:** Bisk et al. 2020, *Experience Grounds Language*;
  LeCun 2022, *A Path Towards Autonomous Machine Intelligence*; Silver &
  Sutton 2025, *Welcome to the Era of Experience*.
