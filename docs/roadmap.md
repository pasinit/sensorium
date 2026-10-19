# Roadmap

One page: where we are, what's next, and why the plan looks the way it does.
Details live in the linked docs. Updated 2026-10-12.

## The programme in one paragraph

One humanoid body (Unitree H1: whole-body skin, head cameras, vestibular
sense, Shadow hands or Robotiq pincers at the wrists) accumulates simulated
experience of every kind: locomotion, reaching, home tasks, piano, and
aimless babbling. A **sensory layer** is pretrained on that experience with
no rewards, labels or text, then adapted to new tasks from a handful of
demonstrations. E1 asks which **senses** and which **kind of experience**
make that few-shot. E2 grows the body backwards to an infant. E3 reads the
layer out in language and tests whether reading extends it. E4 grounds it
in real data.

## Team

You supervise; Claude is the postdoc; two PhD agents each own a research
track with its own paper: **Track A, Experientia** (dataset and benchmark,
D0.3–D0.7) and **Track B, the sensory layer** (model and E1 study). Proposed
in [tracks.md](tracks.md), awaiting your confirmation.

## Ladder

| Step | Question | Hypotheses | Status | Doc |
|---|---|---|---|---|
| M1 | Episode format, audio synthesis, car/drone/piano pipeline, browsable sample | — | ✅ built (PR #1) | [data-generation.md](data-generation.md), [review-sample.md](review-sample.md) |
| **D0** | Experientia v0: the H1 body in our format, four sources, three experience kinds, pools at equal volume, held-out suite, benchmark harness | — | **in progress**: D0.1 ✅ (PR #4), D0.2 ✅ (PR #5), manifest task/instance ✅, dense skin ✅, 256 px vision ✅; next D0.3 | [dataset-proposal.md](dataset-proposal.md) |
| **E1** | Senses × kind of experience → few-shot, on one body | H1, H2 (+ diversity, effector, alien-body ablations) | planned; pilot ≈ $70–150, full ≈ $370–950 | [experiment-01.md](experiment-01.md) |
| E2 | Development: MIMo infant ↔ adult H1 | H3 | sketch | [thesis.md](thesis.md#research-roadmap) |
| E3 | Language: readout probe; reading → sensory prediction | H4, C5 | sketch | [thesis.md](thesis.md#h4-diagnostic-and-c5-language) |
| E4 | Real-world grounding (egocentric video, real tactile data, e-nose): real data as *training* input. Real data for *comparison* is now in Track A | — | later | [datasets.md](datasets.md#2-later-bodies-and-datasets) |

## Next steps, in order

1. ✅ **D0.1 pool builder**: source registry; pools at equal steps
   with fixed effector ratio; senses dropped at load time; split file;
   parquet index. Pure Python over the existing episode format.
2. ✅ **D0.2 the body** (first humanoid sample: `make h1-assets && make sample-h1`): HumanoidBench's H1 + `h1touch` + head cameras
   in our episode format; contact sound from the skin; pincer taxels; **first
   humanoid review sample** for you.
   *Built:* `make h1-assets && make sample-h1` (see
   [data-generation.md](data-generation.md#h1-body-bodieshumanoidpy)).
   *Then:* dense skin on the hands and cube-built pincer pads, onset / slip /
   vibration per region, load-time touch noise ([touch.md](touch.md)); 256 px
   frames stored as H.264, a room recoloured and relit every episode, camera
   tilt 45°, load-time crop and resize ([vision.md](vision.md)).
   Regenerate the sample to see it.
3. **D0.3 babbling → experiments with outcomes** (decided 2026-10-15: scripted experiments, arms and upper body first; random-babbling tuning parked) (~1 week): *steps 1-4 done* (step 3: contact forces inside the published ranges, `reach_free`, `stand_supported`; step 4: table 0.82 m, goals kept in the head camera's view, `make a1-gate` (hands pass all but the force line; pincers fail on the right tip's visibility and `poke` planning), `stand_table` source in split v2; hands still do not lift): collision-checked seated
   reaches and a withdrawal reflex (elbow-slam steps ~108 → ~1 per 600 steps,
   hands), then a repertoire chosen by counts: touch, poke, slide, hover, and a
   grasp-lift-drop (pincers: about 1.7 lifts per 30 s episode)
   or power closure (hands: no lifts yet); A1 statistics in `babble_stats.py`;
   static taxel coordinates for Track B (`docs/touch.md`). *Next:* hand lifts (literature only until the supervisor decides: no NC grasp data may seed the generator). Decided 2026-10-14: these replace "goal babbling
   through the released policies".
4. **D0.4 adapters** (~1–2 weeks): BiGym replay + success filter; LocoMuJoCo
   kinematic replay.
5. **D0.5 piano and play on the H1** (~1 week): arm IK for the RP1M hands;
   playground scene with the H1 arm. Also retry the RP1M replay with its
   App. B settings.
6. **D0.6 benchmark harness** (~1 week): encoder API; frozen + LoRA heads;
   probes including the effector probe; vision-only baselines (VC-1, DINOv2,
   V-JEPA 2); BiGym retargeting side check.
7. **D0.7 review** (~2 days): per-source pages and stats; your sign-off.
8. **M3 model** (~2–3 weeks): stems, trunk, JEPA-style pretraining loop;
   smoke test on Modal's free tier.
9. **Pilot** (~1 week, ≤ $150 on Vast.ai): 4 runs, frozen heads; go/no-go.
10. **Full E1** (~2 weeks, ≤ $1,000): ~30 runs, both adaptations, ablations;
    `docs/results-01.md`.

What you decide along the way: the first humanoid sample (step 2), the
held-out task choice (step 7), go/no-go after the pilot (step 9).

## Standing rules

- No hardware; rent the cheapest provider (Vast.ai spot A100 by default,
  Modal for smoke tests, R2 for storage). Caps: pilot $150, full E1 $1,000.
- `privileged` state and `text` never enter the model batch.
- Every claim with a number is checked against its primary source before it
  is relied on; `literature-review.md` records what was verified.
- Readable code over frameworks; seeds, pinned simulators and content hashes
  for every dataset.

## Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-10-10 | Thesis: knowledge from embodied multisensory experience; text as readout only | Your framing: language is a means to describe the world, not the substrate |
| 2026-10-11 | E1 = car + drone → piano hands (far transfer) | Simulators with experts available; piano as the contact-rich craft target |
| 2026-10-11 | Literature check before committing; drop "acting beats watching" (answered by V-JEPA 2 / DINO-WM) | Your rule: don't redo published experiments |
| 2026-10-11 | Framing: a *sensory layer* from diverse experience → few-shot; E1 = diversity sweep; D0 = Experientia | Your stated ultimate goal |
| 2026-10-11 | Citation audit: ~75 claims confirmed, 4 wrong, ~10 overstated; all corrected | Numbers in the plan must be true |
| 2026-10-11 | LoRA added next to the frozen head | Your suggestion; Lin et al. Table 2 (frozen DINOv2 0.00 vs LoRA 0.72) |
| 2026-10-11 | Diversity demoted; headline = senses × kind of experience | MyoDex and Lin et al. already show diversity helps; task-free experience and senses are untested |
| 2026-10-11 | One body, not many; whole body, not hands | Your call: a unified model for the whole body; cross-body transfer tracks appearance (XEWorld) |
| 2026-10-11 | H1 humanoid with two end effectors (Shadow hands, Robotiq pincers) under explicit controls | BiGym's 2,000 human demos exist only on the pincers; controls make the compromise measurable |
| 2026-10-12 | Dense skin (MIMo spread) on hands and pads, cube pads, onset / slip / vibration, load-time noise; body grid unchanged | A touched patch lit 1 taxel of 8 in 83% of steps; coarse touch could make E1 undervalue touch |
| 2026-10-14 | Track A includes real data and a sim-vs-real study of each sense (public datasets or cheap self-recording; no hardware) | You: "we should include real data and make a study of the differences" |
| 2026-10-14 | Team: postdoc plus two PhD tracks, A = Experientia (dataset and benchmark), B = sensory layer (model and E1) (proposed) | You: act as postdoc; each PhD's contribution specific, different and publishable on its own |
| 2026-10-12 | Frames 256 px for every body, stored as H.264 (CRF 18, 4:4:4); crop/resize at load time; H1 in a room with per-episode colours and lights, camera tilt 45° | You: samples were very low quality. At 64 px fingers were 1–2 px; VC-1/DINOv2/V-JEPA 2 read 224–256 px; HumanoidBench renders 256. Lossless 256 px would be ~2 TB; video is ~150 GB |
| 2026-10-14 | D0.3: HumanoidBench's reach policy is free-space only (falls near a table, needs 50 Hz); standing contact babbling uses a supported stand instead; seated episodes are 600 steps; numpy port, no torch | PhD A's tests in our scene (D0.3 memo): 14 of 15 table runs fell; 20 Hz ticks fell within 3 s |
| 2026-10-14 | Table 0.82 m by default; babbling goals, hovers and lifts kept inside the head camera's view (goals within 0.30 m of the middle, hover <= +10 cm); coverage of free space is the price | Postdoc, decision 1 of the step-3 review: tips were outside the image in 18-28% of steps |
| 2026-10-14 | `stand_supported` is its own source (`stand_table`), split v2 | Postdoc, decision 2: cheapest before data exists at scale |
| 2026-10-14 | DexGraspNet and every other CC BY-NC grasp set stay out of `HAND_CLOSURE` and the generator; hand lifts are literature-only (PhD A's list of Shadow-hand grasp sets with licences) | Postdoc, decision 3, from supervisor decision 6 |
| 2026-10-14 | Senses axis of E1: V, V+T, V+S, V+T+S; proprioception and vestibular sense in every condition | Supervisor approved (decision 1): isolates each extra sense against vision |
| 2026-10-14 | Track A experiments A1–A6 approved, plus PhD A's comparison of babbling kinds | Supervisor approved (decision 3) |
| 2026-10-14 | Dense vs coarse touch split: Track A measures signal information (A3), Track B optionally the effect on few-shot learning | Supervisor approved (decision 4): keeps the two papers apart |
| 2026-10-14 | Datasets licensed NC, ND or without redistribution are used for evaluation only, never redistributed | Supervisor approved (decision 6) |
| 2026-10-14 | H1 framing "which senses help which tasks" rejected as obvious (touch helps grasping); reframe pending a literature check (candidate: senses as teachers, i.e. pretrained with a sense, tested without it) | Supervisor: the question must be meaningful |
| 2026-10-14 | On hold: self-recorded "Mini-Hits" sounds; GPU smoke test on Modal | Supervisor: wait (decisions 5 and 7) |
| 2026-10-15 | H1 reframe: a sense is both an input and a teacher (prediction target); the default model uses every sense in both roles, and H1 measures what each role contributes (arms: input + target, target only, none, placebo) | Supervisor: "a sense is an input AND a teacher" |
| 2026-10-15 | Experience must have consequences: the robot acts on the world on purpose. Order: (A) scripted experiment programs with outcomes, (B) BiGym demo replay moved forward, (C) a curious learned explorer after a literature check; tuning of random babbling (pincer view, pokes, hand force, hand lifts) parked | Supervisor: "if the robot does nothing, does not experiment with the world around it, there is nothing to be learnt". In data/probe_test an object is lifted in 0.5% of steps |
| 2026-10-15 | Arms, hands and upper body first (reach, lean, turn, bend; seated and supported standing); walking later | Supervisor decision |
| 2026-10-15 | Waist work skipped for now: no torso-turn experiment (E10) and no turn-reach held-out task (T4, kept reserved in split v3); lean and bend deferred to the legs phase (the H1 waist is yaw only). Few-shot tasks for now: stack, drawer-open, push-to-target | Supervisor decision |
| 2026-10-15 | A1 headline becomes "caused change" (objects displaced, lifted, toppled, stacked, opened after robot contact; own posture change); contact share, forces, falls and view stay as constraints | Follows the experience decision; thresholds fixed in PhD A's design memo before build |
| 2026-10-14 | Every results report carries tables (mean ± spread, n) and loss curves; PhDs produce them | Supervisor's standing rule (PR #22) |
| 2026-10-14 | Track B: no tiny SSL encoder beats an untrained one (probe mean 0.22–0.25 vs floor 0.22); the probe battery is sound (supervised ceiling 0.64, gap +0.42); SSL init gives no fine-tune gain at 4 or 16 labelled episodes | B-ctrl-0 (PR #23). Next on CPU: proprio dropout, scale curve (25/50/100 episodes, 128 px, width 256); differences under 0.08 (6 seeds) are not claimed |


## Open questions

- Will the trunk unify hand and pincer episodes? (effector probe, D0.6)
- Do RP1M's App. B settings fix the piano action replay? (D0.5)
- Can BiGym demos be retargeted to the hands with > 50% success? (D0.6)
- Does whole-body babbling produce enough contact to matter? (D0.3 stats)
- Does dense skin beat the coarse grid, given Tactile Genesis found ~200
  taxels enough? (cheap E1 ablation: drop the dense channels at load time)
- Does EGL rendering on the rented GPU box keep generation near the Mac's
  ~21 steps/s per process? (check on the first rental, before D0 at scale)
