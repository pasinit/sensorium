# Literature review: what's already done (checked 2026-10-11; citations verified against primary sources 2026-10-11)

This review was run before committing to E1's hypotheses, to avoid repeating
published experiments. Claims come from paper abstracts and summaries. The
**(verify)** tags mark numbers to check in the full paper before we cite them.

## Verdict per hypothesis

The hypothesis numbers follow [thesis.md](thesis.md). The per-hypothesis
evidence tables, with the figures and tables each number comes from, are in
[thesis.md § Evidence per hypothesis](thesis.md#evidence-per-hypothesis-what-is-known-and-why-each-is-still-open).

| Hyp. | Our claim | Status in the literature | Keep? |
|---|---|---|---|
| **H1** | Senses beyond vision | **Established within one body and one sensor** (MULSA, SPOTS, VT-WM, Sparsh-X), always from *real* sensors. **Open:** a simulated pretraining pool with touch *and* contact sound; a matched comparison against vision foundation encoders at few-shot budget; which senses matter across many tasks on one hand. | ✅ **Core**, with audio and touch ablated separately |
| **H2** | Kind of experience: task-free vs expert | **Multi-task pretraining on a hand is done** (MyoDex: 14 expert tasks → 37/43 unseen, ~8M samples per new task; Arnold). **Diversity helps, always with volume confounded** (Lin et al., HPT, embodiment scaling). **Open:** whether task-free babbling at equal volume builds a usable layer, whether a mix beats expert-only, and the k ≤ 10 regime. Curiosity work (Plan2Explore, Sancaktar) never evaluates a frozen few-shot representation. | ✅ **Core.** Diversity demoted to a 2-level check. |
| **H3** | Development (infant ↔ adult body) | **Near transfer between similar robots is established** (HPT, CrossFormer, H-Zero), vision + proprio only; XEWorld shows transfer follows appearance. **Open:** non-visual senses across bodies, hand → whole body, infant babbling as a pretraining source. Multi-body "embodiment distance" dropped: car and drone are an ablation only. | ✅ **E2** |
| **H4** | Language is a readout of a frozen grounded latent | **Partly done.** Post-hoc LLM alignment of a video world model (V-JEPA 2), vision-language-aligned world-model latents with a frozen language readout (GWM 2026), and probes of LM world states. Not done for multisensory latents from a text-free layer. | ⚠️ **Diagnostic only** (E3), with reading → senses (C5) as the new part |
| *(dropped)* | Action-conditioned pretraining beats passive video | **Largely answered.** Passive-video pretraining plus a little action data works very well (V-JEPA 2-AC, DINO-WM). Action-aware pretraining from unlabelled video is its own line (APV, PreLAR, latent actions). | ❌ as a hypothesis. Keep V-JEPA 2 / DINO-WM as **strong baselines** |
| Play | Basic play (reach, push, grasp) is a foundation | **Partly done.** Curious, play-like exploration gives zero-shot object manipulation in the same domain (Sancaktar 2022, Plan2Explore, Walker 2023). Infant simulators exist (MIMo, BabyBench). **Open:** play → skill few-shot, and multisensory play. | ✅ became **H2** (babbling as an experience kind) |

### Body choice (2026-10-11): one humanoid, whole body

Surveyed human-like bodies for "multisensory few-shot pretraining" novelty:

| Body | Ecosystem | Prior pretraining | Verdict |
|---|---|---|---|
| Isolated hand (Shadow, Adroit, LEAP) | RP1M, Minari Adroit, MyoDM env code, Bi-DexHands (Isaac), ManiSkill3 touch, Tactile Genesis (touch + audio in sim) | real tactile SSL (Sparsh-skin/X, T-DEX); proprio-only multi-task priors (MyoDex, Arnold) | gap open, tooling best; **folded into the humanoid's hands** |
| Infant (MIMo v2, pyCub) | reach, standup, selfbody, catch, roll_over; binocular 256² foveated eyes, full-body skin (count unpublished), gyro + accel vestibular, growth 0–24 months, v2 Adroit-based hands; no released policies (MIT) | none | **fully open** → E2 |
| Full humanoid (HumanoidBench `h1touch`, BiGym, LocoMuJoCo) | HumanoidBench: H1 + Menagerie Shadow hands, 448 `touch_grid` taxels, 2 head cameras, only reach policies released, RL fails most tasks (MIT). BiGym: H1 + **Robotiq pincers**, 40 tasks × 50 VR demos replayable from joint-position actions, floating base, no touch (Apache-2.0). LocoMuJoCo v1: H1 mocap, kinematic only (code MIT, data CC BY-NC-ND) | mocap priors (Puppeteer, MaskedMimic), vision VLAs; no whole-body touch/audio pretraining | **fully open; chosen for E1** (user: the agent is the whole body) |
| Arm + hand tabletop | DexArt, ManiSkill3, DexMimicGen | crowded by VLAs/LBMs | medium |
| Musculoskeletal (MyoSuite) | ~60 envs, HF baselines; **no touch observations** | MyoDex, SAR, Arnold | multi-task done, proprio only |

No human-like-body simulator renders audio natively except Tactile Genesis;
contact sound stays synthesised from taxel events.

## Evidence by thread

### 1. Cross-embodiment pretraining and transfer (H3, and the body axis of H1)
- **HPT** (Wang, Chen, Zhao, He; NeurIPS 2024). Per-embodiment "stems" tokenise
  proprioception and vision into a fixed number of tokens, and a shared trunk is
  pretrained on 52 datasets. It gives **>20% gains on unseen tasks** after
  fine-tuning. *This is our proposed tokeniser design, already built and
  open-sourced:* reuse it rather than reinvent it.
  [arXiv 2409.20537](https://arxiv.org/pdf/2409.20537)
- **CrossFormer** (2024, arXiv preprint): one policy for 20 embodiments
  (~900K trajectories), including quadcopters and quadrupeds. **Pushing the
  Limits of Cross-Embodiment Learning for Manipulation and Navigation** (2024,
  arXiv preprint): co-training with navigation and driving data gave a 20%
  higher success rate on manipulation (19% in its Table IX). These are
  policies, not world models, and the bodies overlap more than ours.
  [CrossFormer](https://arxiv.org/abs/2408.11812) · [Pushing the Limits](https://arxiv.org/abs/2402.19432)
- **TD-MPC2** (ICLR 2024): one 317M-parameter world model on 80 tasks across
  domains and embodiments. A 19M agent pretrained on 70 DMControl/Meta-World
  tasks and fine-tuned on 10 held-out tasks: the gain is "somewhat
  task-dependent"; the authors call the results "exploratory" but "promising"
  and fine-tuning "very much an open research problem".
  [arXiv 2310.16828](https://arxiv.org/abs/2310.16828)
- **Newt / MMBench** (Hansen et al., ICLR 2026): a language-conditioned
  multitask world model on **200 training tasks** across domains and
  embodiments (MMBench has 220 incl. 20 held-out). It is pretrained on
  demonstrations, then trained with online RL. On the 20 held-out
  tasks/embodiments it scores 0.192 zero-shot vs 0.013 from scratch, and
  0.868 vs 0.480 after 100k fine-tuning steps: a clear average gain.
  **Environments, demos, code and 200+ checkpoints are released.**
  [arXiv 2511.19584](https://arxiv.org/abs/2511.19584)
- **Video and world-model cross-embodiment, 2025–26:**
  - **TraceGen:** 67.5% vs 0% from scratch with 5 phone videos.
  - **CLAP:** "Cross-Embodiment Video World Models are Zero-Shot Physical
    Simulators": trained on human + robot video with actions mapped to a
    shared space; claims zero-shot deployment to real-world tasks.
  - **XEWorld:** adapting to an unseen robot *causes forgetting* of seen ones;
    models generalise by visual, not kinematic, similarity.
  - **DreamZero** (NVIDIA): few-shot transfer to a new embodiment with 30 min
    of play data. **AnyWorld** (XPeng Robotics): human-video pretraining, then
    mixed-embodiment fine-tuning, to turn one human video into robot rollouts.

  Sources: [TraceGen](https://arxiv.org/html/2511.21690),
  [CLAP](https://arxiv.org/pdf/2608.27406), [XEWorld](https://arxiv.org/pdf/2608.05799),
  [AnyWorld](https://arxiv.org/html/2608.29242), [DreamZero](https://arxiv.org/pdf/2602.15922).
- **H-Zero** (2025): cross-humanoid locomotion pretraining gives few-shot
  transfer to new humanoids. [arXiv 2512.00971](https://arxiv.org/pdf/2512.00971)
- **Pretraining doesn't always help world models:** ContextWM reports that
  a *vanilla world model "seldom benefits from pre-training"* on Meta-World
  (its Meta-World gain is "primarily due to the intrinsic exploration bonus"),
  while on DMC Remastered the vanilla model *does* benefit. ContextWM's own
  gain comes from separating context from dynamics.
  [ContextWM](https://arxiv.org/html/2305.18499v1)

**Gap:** every positive result transfers between bodies that share structure
(arms, hands, humanoids) or between video domains. Nobody has measured
**how transfer falls off with embodiment distance**, from near (other hands)
to far (car, drone), for a single world model, with or without non-visual
senses. The closest results are mixed: task-dependent in TD-MPC2, mixed by
domain in ContextWM, clearly positive within Newt's benchmark. Mapping
*where it stops* is itself a contribution.

### 2. Passive video vs action-conditioned pretraining (dropped hypothesis): answered
- **APV** (ICML 2022) and **IPV / ContextWM** (NeurIPS 2023): action-free
  video pretraining, then an action-conditioned model on top.
- **PreLAR** (ECCV 2024): infers latent actions from frame pairs so
  pretraining is action-aware without labels (Meta-World only). **FlowWAM**
  (2026) uses optical flow as the action representation.
  [PreLAR](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03363.pdf) · [FlowWAM](https://arxiv.org/abs/2607.13017)
- **V-JEPA 2** (2025): 1M+ hours of passive video, then **<62 h** of
  unlabelled robot video for action-conditioning, giving zero-shot
  pick-and-place. **DINO-WM** (ICML 2025): a world model on frozen DINOv2
  features plans zero-shot in 6 environments ("zero-shot" = test-time planning
  without demos or rewards, not transfer to new bodies).
  [V-JEPA 2](https://arxiv.org/abs/2506.09985) · [DINO-WM](https://proceedings.mlr.press/v267/zhou25t.html)

**Implication:** testing "acting beats watching" again would duplicate work.
V-JEPA 2 and DINO-WM become the **baselines to beat**.

### 3. Multisensory world models and representations (H2)
- **See, Hear, and Feel** (MULSA, CoRL 2022): camera + contact microphone +
  tactile sensor; audio catches key moments vision misses.
  [paper](https://jiajunwu.com/papers/seehearfeel_corl.pdf)
- **SPOTS** (2023; *Robotics and Autonomous Systems*): joint vision–touch
  prediction "provides the greatest benefits in physically ambiguous
  interaction regimes". [arXiv 2304.11193](https://arxiv.org/abs/2304.11193)
- **Visuo-Tactile World Models** (2026): touch curbs world-model
  hallucinations and improves zero-shot planning in contact-rich tasks.
  [arXiv 2602.06001](https://arxiv.org/html/2602.06001v1)
- **Sparsh-X** (2025): self-supervised touch backbone (tactile image, audio,
  motion/IMU, pressure) pretrained on ~1M Digit 360 contacts. Frozen features
  beat end-to-end training by 48% on physical-property estimation; the +63%
  policy gain is against an end-to-end model using tactile images only.
  [arXiv 2506.14754](https://arxiv.org/abs/2506.14754)
- **Audio-VLA** (2025), **MuseVLA** (2026): contact audio and extra sensors
  added to VLAs. [Audio-VLA](https://arxiv.org/pdf/2511.09958) · [MuseVLA](https://arxiv.org/pdf/2606.17598)

**Gap:** all of this is single-body. Whether non-visual senses learned on one
body help another body is open. Our synthetic audio is a weakness here; real
contact sound (ObjectFolder-style) would be needed for a strong claim.

### 4. Play, curiosity and infant-like learning
- **Sancaktar et al.** (NeurIPS 2022): curious free play with a structured
  world model gives **zero-shot** stacking, flipping, pick-and-place and
  throwing. [arXiv 2206.11403](https://arxiv.org/abs/2206.11403)
- **Plan2Explore** (2020); **Walker et al.** (ICML 2023): which components of
  reward-free pretraining transfer. [Walker](https://proceedings.mlr.press/v202/walker23a/walker23a.pdf)
- **InDRiVE** (2025): reward-free world-model pretraining for driving.
  [arXiv 2512.18850](https://arxiv.org/pdf/2512.18850)
- **MIMo** (IEEE TCDS 16(4), 2024; code MIT; "MIMo grows", ICDL 2025):
  simulated 18-month-old with binocular vision, proprioception, a vestibular
  system and full-body touch. **Baby Sophia** (2025), an agent in the
  **BabyBench** environment: self-touch and hand regard learned from intrinsic
  rewards. [MIMo](https://arxiv.org/abs/2312.04318) ·
  [Baby Sophia](https://arxiv.org/abs/2511.09727)

### 4b. Task-free babbling for the H1 (checked 2026-10-14)

Why the babbling is goal-directed and scripted, and what HumanoidBench's
reach policy can and cannot do; design in the D0.3 memo, statistics in
`babble_stats.py`. Verified against the primary sources:

- **Goal babbling**: exploring in task space beats exploring in actuator space
  for redundant robots, and competence-progress goals beat random goals
  (Baranes and Oudeyer, SAGG-RIAC, RAS 2013, arXiv 1301.4862, abstract); Rolf et
  al. (IEEE TAMD 2010) *(abstract only; full text unreachable)*. IMGEP
  (Forestier et al., JMLR 2022, arXiv 1708.02190, Sec. 4.2-4.3): object-centred
  goal spaces chosen by learning progress; exploration = share of reached cells
  of a discretised outcome space (100 bins per dimension in 2-D, 20 in 3-D);
  tool exploration 62% vs 37% against random space choice (2-D simulation).
- **Interaction metrics** (Sancaktar et al., NeurIPS 2022, Figs. 4-5): relative
  time moving one object, two or more, flipping, objects in the air, agent in
  free space; the RL curiosity baselines (Disagreement, RND, ICM) rarely
  interact (RND at least 10% of steps). Code MIT but Python 3.8 and mujoco-py.
- **Scripted object-directed primitives in task-free data**: random pokes on
  objects, 100K pokes in 400 h, because "most pokes are executed in free space"
  (Agrawal et al. 2016, arXiv 1606.07419); random grasps in object regions, 50K
  tries (Pinto and Gupta 2016, arXiv 1509.06825); random actions plus a palmar
  grasp reflex, "about 20% of trajectories included some sort of grasp" (Ebert
  et al. 2018, arXiv 1812.00568, Sec. 7). Body-part coverage as a metric: 32 of 34
  parts touched (Baby Sophia, arXiv 2511.09727, Table I).
- **HumanoidBench's released reach policy** (paper Sec. V-D, repo `data/`):
  `reach_one_hand` (obs 55) and `reach_two_hands` (obs 61), 256 x 256 tanh MLPs
  with 19 outputs (legs, torso, arms), PPO in MJX on a hands-free H1 with only
  foot collisions, 2 and 4 billion steps, 50 Hz. In our scene it fell within 3 s
  at 20 Hz, and 14 of 15 times with a table in reach (PhD A's test).
- **Trained play** (Play2Perfect, arXiv 2606.26428): goal-conditioned RL on a
  7-DoF arm and 22-DoF hand, 7 GPU-days on one A6000; no contact metrics.

### 4c. Contact forces of the end effectors (checked 2026-10-14)

Why the babbling's contact forces are held to published ranges (`docs/touch.md`, "Contact forces").

- **Robotiq 2F-85** (product sheet, May 2025): grip force adjustable 20 to 235 N, closing speed
  20 to 150 mm/s, stroke 85 mm, payload 5 kg. Its manual (6.2.1) defines the payload
  from an actuation-force model with charts we did not extract.
- **Shadow Dexterous Hand E** (technical specification, Dec 2024): tendon force sensors of
  about 30 mN resolution, "comparable force output ... to the human hand"; **no fingertip force
  figure**. About 10 N per fingertip appears in press coverage of the vendor's claim (New Atlas)
  *(verify)*; the Shadow DEX-EE sheet gives at least 8 N (a different hand).
- **DexGraspNet** (Wang et al., ICRA 2023, arXiv 2210.02697; code and data CC BY-NC 4.0): 1.32 M
  Shadow-hand grasps of 5,355 objects; the repo holds the grasps of 5 objects (1,141 grasps in
  OpenAI's joint naming, median finger flexion 0.2-0.7 rad, thumb opposition 1.1 rad). Tried as the
  closure of our power grasp: no lift (PhD A, `.tmp/phd-a/lift_spike.md`).

**Implication** *(written for the multi-body plan; superseded on 2026-10-11:
the body is the H1 and the playground scene is reused with its arm and
hand)*: a custom playground would re-create tabletop tasks that
exist in Meta-World, robosuite and MMBench. What those suites lack is **touch
and contact sound**. Better: add touch sites and contact audio to existing
MuJoCo suites (MMBench/Meta-World), and use MIMo/BabyBench for body-schema
play. The paused `feat/playground` branch (scene, contact sound, scripts) is
still useful: its touch and contact-sound code ports onto those suites.

### 5. Language readout of world-model latents (H4)
- **V-JEPA 2**: LLM alignment added *after* world-model pretraining,
  reaching state-of-the-art video QA at 8B.
- **Grounded World Model** (2026): plans in a vision-language-aligned latent;
  a *frozen readout* maps the imagined future and the instruction into one
  embedding space and their cosine similarity is the planning cost. Solves
  87% of 288 WISER tasks vs 22% for ten fine-tuned VLAs.
  [arXiv 2604.11751](https://arxiv.org/abs/2604.11751)
- **Propositional probes** (ICLR 2025): decode faithful world states from LM
  activations even when outputs are unfaithful.
  [paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3132d0405fabe24b2a7b6cd7ba9de6b5-Abstract-Conference.html)
- **Dynalang** (ICML 2024): language as a predicted modality.

### 6. RP1M / RoboPianist (our target)
- **RP1M** (CoRL 2024): ~1M expert trajectories (500 per song, 550 steps
  each) over ~2k pieces. Multi-task baselines in the **v2** paper (Nov 2024):
  BC-MLP, BeT, DP-U, DP-T, on 12 in-distribution and 20 OOD songs with
  12/25/50/100/150 training songs (Table 2). (v1 had BC-RNN and easy/hard-OOD
  splits; superseded.) *Use the v2 splits and metrics, and compare our
  from-scratch baseline against Table 2.* Env settings in App. B: gravity
  compensation on, stretch factor 1.25, trim silence, lookahead 10, 39-D
  action; **no RoboPianist/MuJoCo version is stated**.
  [arXiv 2408.11048](https://arxiv.org/abs/2408.11048)
- **RP1M++** (2025): policy representations and data scaling (12 vs 300
  training songs; 1,053 songs / 2,091 clips; >2,000 agents). Allegro and ORCA
  hands only **validate the fingering method with per-song RL**: nothing is
  transferred between hands, so near transfer for piano is *not* yet studied.
  [arXiv 2511.02504](https://arxiv.org/abs/2511.02504)
- **CANTABILE** (2026, expressive dynamics), **PANDORA** (diffusion policy),
  **real-world piano** (2025).
- **Not found:** any world model for piano, or any pretraining from *other
  domains* evaluated on RP1M. Our far-transfer test is new.

### 7. Representations for few-shot control, and diversity (H1)
- **VC-1 / CortexBench** (Majumdar et al., 2023): the largest study of
  pretrained *visual* representations for embodied AI, frozen and evaluated on
  17 tasks from 7 benchmarks. **No representation dominates everywhere.** More diverse
  pretraining data helps *on average, not uniformly*. Task-specific adaptation
  is still needed for top results. → our evaluation protocol mirrors this,
  extended to non-visual senses. [arXiv 2303.18240](https://www.arxiv.org/pdf/2303.18240)
- **Data Scaling Laws in Imitation Learning** (Lin et al., ICLR 2025 oral;
  40k+ demos, 15k+ real rollouts):
  generalisation follows a **power law in the number of environments and
  objects**, while more demonstrations of the same setting plateau quickly.
  This is single-body and vision-based, and it supports our diversity bet.
  [arXiv 2410.18647](https://arxiv.org/pdf/2410.18647)

### 7b. Simulated touch post-processing (checked 2026-10-12)

Why `touch_grid` alone is coarse and what others do about it; design in
[touch.md](touch.md). Verified against the primary sources:

- **MIMo** touch: sensor points spread over each geom at a set spacing;
  `spread_linear` shares a contact's force with nearby points, dropping to 0
  at twice the spacing, total force conserved; `nearest` puts it all on one.
- **Han, Joshi, Deng 2025** (arXiv:2505.02915): MuJoCo `touch_grid` on a pad
  built from 4.6 × 4.6 × 1 mm cubes, because a single block gives sparse
  contact forces; randomised cube depth, Poisson-effect convolution
  (scale U(0.1, 0.3)), per-axis gain U(0.5, 1.5), per-taxel gain N(1, 0.25)
  with 20% dropout; 2 × 2 sum pooling; no binarisation, no calibration.
- **Kasolowsky & Bäuml, IROS 2024** (arXiv:2409.12735): PyBullet; 14,884
  surface points (0.25 mm); local penetration with a linear stress model,
  scaled so stresses sum to the simulator's normal force, summed per taxel;
  per-episode taxel offsets and noise ranges, per-step Gaussian noise.
- **Yin et al., RSS 2023** (arXiv:2303.10880): 16 FSRs on an Allegro hand,
  binarised at a threshold to shrink the sim-to-real gap; a continuous-signal
  baseline (CT-Sensor) and a low-sensitivity one (LS-Sensor) as ablations.
- **TacSL** (arXiv:2408.06506): penalty-based normal and shear force fields
  from SDF penetration, plus GelSight-style images; objects fixed before the
  simulation starts.
- **Slip**: Veiga et al. (IROS 2015) flag slip when 30–200 Hz signal energy
  passes a threshold (BioTac); VibeAct (2026 preprint, abstract only) feeds
  per-fingertip onset, slip and slip magnitude to the policy.
- **MuJoCo 3.15 `tactile` sensor**: per-vertex penetration and sliding
  velocity, but only contacts with SDF geoms contribute.

### 7c. Image resolution, preprocessing and storage (checked 2026-10-12)

Why the H1 frames went from 64 to 256 px and to video storage; design in
[vision.md](vision.md). Verified against the primary sources:

- **DrQ-v2** (arXiv:2107.09645): 84 × 84, 3 stacked frames, action repeat 2;
  random shift = pad 4 px, random 84 crop, bilinear smoothing.
- **robomimic** (arXiv:2108.03298): 84 × 84 images, random 76 × 76 crops;
  ResNet-18 + spatial softmax; removing the crop augmentation hurts.
- **Diffusion Policy** (arXiv:2303.04137, Table 7): 84 → 76, 96 → 84,
  240 → 216 in simulation, 320 × 240 → 288 × 216 real; random crop in
  training, centre crop at inference.
- **Octo** (arXiv:2405.12213): main camera 256, wrist 128; random crop,
  resize, colour jitter; pixels to [-1, 1]; 16 px patches (256 + 64 tokens).
- **V-JEPA 2** (arXiv:2506.09985): 256 px (384 for ViT-g), 2 × 16 × 16
  tubelets; the action-conditioned model trains on 4 fps clips at 256 px with
  random resized crops (aspect 0.75–1.35).
- **DINO-WM** (arXiv:2411.04983): 224 px observations, DINOv2 patch features
  (196 tokens of 14 px at 196 px).
- **VC-1** (model card): 224 × 224 input, 16 px patches.
- **HumanoidBench** (code, `env.py`): eye cameras rendered at 256 × 256.
- **DreamerV3** (arXiv:2301.04104): the paper gives no image size (64 × 64
  is folklore from the code).
- **LeRobot** (Hugging Face blog, *Scaling robotics datasets with video
  encoding*): frames stored as AV1 video (`yuv420p`, keyframe every 2
  frames, CRF 30); policies trained on it (Diffusion on PushT, ACT on ALOHA)
  were "on par with those trained on the image versions". Open
  X-Embodiment stores JPEG frames.

### 8. Existing datasets with several senses
- **ARIO** (2024): 3.03M episodes in a unified cross-embodiment format, but
  a **real + sim mix**: ~77% converted open-source data, ~23% simulated by the
  authors, 0.12% real data they collected. Modalities: images, 3D, sound,
  text, tactile, but audio is in only ~0.04% of episodes (ManiWAV) and touch
  only in the RH20T subset. **RH20T** (ICRA 2024): 110k real contact-rich
  sequences with vision, force, audio and actions; 147 tasks; fingertip touch
  in one of seven robot configurations. **VTouch++** (2026): ~120k real
  bimanual episodes, 380+ tasks, RGB-D + proprio + visuo-tactile end
  effectors, on dual-arm fixed/wheeled platforms and UMI handheld grippers;
  no audio. **VinT-6D** (Dec 2024): 2M simulated + 0.1M real in-hand samples
  with vision, touch and proprio. **AVW-4k**: simulated audio-visual
  navigation (SoundSpaces 2.0), no touch.
  [ARIO](https://arxiv.org/abs/2408.10899) · [RH20T](https://rh20t.github.io) ·
  [VTouch++](https://arxiv.org/abs/2604.20444) · [VinT-6D](https://arxiv.org/abs/2501.00510) ·
  [AVW-4k](https://arxiv.org/abs/2512.00883)
- **None lets you control diversity at fixed volume, and none comes with a
  frozen-encoder few-shot benchmark.** A 2026 tactile survey also calls for
  temporally aligned, action-conditioned visuo-tactile trajectories rather
  than static pairs. [survey](https://arxiv.org/html/2605.17336v1)
  → motivates [Experientia](dataset-proposal.md).

## Earlier proposed revision (superseded)

**Superseded on 2026-10-11** by the sensory-layer framing (goal: a sensory
layer from many experiences that makes task learning few-shot). The adopted
version is in [thesis.md](thesis.md), [experiment-01.md](experiment-01.md)
and [dataset-proposal.md](dataset-proposal.md). Kept below for the record.

1. **One headline question:** *how does transfer to dexterous piano fall off
   with embodiment distance?* Pretraining sources, from near to far:
   - other hands: RP1M++ Allegro/Orca, MMBench hand tasks
   - generic tabletop play with touch: MMBench/Meta-World + our touch sites
   - locomotion
   - car and drone

   Measure data efficiency on RP1M's song splits. A clear fall-off curve is
   publishable whether far transfer is positive or zero.
2. **Second question (open):** do touch and contact sound learned on one
   body transfer to another body's sensors?
3. **Don't build what exists:**
   - Use **HPT-style stems** as the tokeniser.
   - Build on **TD-MPC2/Newt** (open code and checkpoints) as the world model
     rather than a new JEPA.
   - Baselines: **V-JEPA 2 / DINO-WM** frozen encoders, and RP1M's own
     BC-RNN / Diffusion Policy.
4. **Drop H2 as a hypothesis** (answered). **Keep H4 as a diagnostic.**
5. **Playground:** don't finish a custom simulator. Port its touch and
   contact-sound code onto existing suites.

Effect on what's built: the car/drone/piano pipeline and review sample stay
as they are. They are the "far" end of the curve and the target. Costs change
little; most of the compute is still the transfer sweep.
