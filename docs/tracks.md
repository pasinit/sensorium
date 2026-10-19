# Team and research tracks

**Status: proposed 2026-10-14 by the postdoc; awaiting the supervisor's
confirmation** (research direction is the supervisor's call).

| Role | Who | Responsibility |
|---|---|---|
| Supervisor | the user | Final say on major decisions, compute spending and research direction |
| Postdoc | Claude (`postdoc` skill) | The research is sound and novel, and experiments are sound and relevant; plans, reviews, integrates, merges, reports |
| PhD A | `phd-experientia` agent | Track A below: executes it and proposes ideas |
| PhD B | `phd-sensory-layer` agent | Track B below: executes it and proposes ideas |

Each PhD owns one track with its own contribution, publishable on its own.
The tracks split the programme along the line that already exists between
the dataset (D0) and the first experiment (E1).

## Track A: Experientia, a multisensory humanoid experience dataset and benchmark

- **Question.** Can simulation give a whole-body humanoid the senses and the
  breadth of experience that real multisensory datasets lack, at a volume and
  diversity we control, and how do its simulated senses differ from real
  ones?
- **Contributions.**
  1. Simulated senses for the H1, each validated against the literature
     and checked by eye: dense skin with onset, slip and vibration
     ([touch.md](touch.md)); contact sound; head cameras in varied rooms
     ([vision.md](vision.md)); vestibular sense and proprioception.
  2. An experience generator: expert sources (BiGym, RP1M piano,
     HumanoidBench reach), noisy experts, and **goal-directed task-free
     babbling** with contact-quality statistics, with volume and diversity
     controlled independently ([dataset-proposal.md](dataset-proposal.md)).
  3. A held-out few-shot benchmark and harness, with results for
     off-the-shelf encoders (VC-1, DINOv2, V-JEPA 2, HPT) and training from
     scratch.
  4. **Real data and a sim-vs-real study** (supervisor, 2026-10-14): real
     recordings for each sense, from public datasets or cheap
     self-recording (no hardware purchases), and a measured study of how the
     simulated senses differ from real ones, and which differences the
     sensor models can close.
- **Novel because** no existing dataset combines whole-body touch, contact
  sound, vision and actions with controllable diversity at fixed volume and
  a frozen-encoder few-shot benchmark ([literature-review.md §8](literature-review.md#8-existing-datasets-with-several-senses)).
- **Roadmap steps:** D0.3 (babbling), D0.4 (adapters), D0.5 (piano and play),
  D0.6 (benchmark harness and baselines), D0.7 (review).
- **Likely venue:** a datasets-and-benchmarks track (e.g. NeurIPS D&B).

## Track B: the sensory layer, and which senses and experience make it few-shot

- **Question.** Which senses (H1) and which kind of experience (H2) make a
  sensory layer, pretrained with no rewards, labels or text, adapt to new
  tasks from a handful of demonstrations?
- **Contributions.**
  1. A multisensory sensory-layer model: per-sense stems, a shared trunk,
     JEPA-style predictive pretraining (M3, [experiment-01.md](experiment-01.md)).
  2. A controlled study at matched volume and compute: senses (V, V+T,
     V+S, V+T+S) × experience (expert, task-free, mix), with frozen
     and LoRA few-shot adaptation, against vision foundation encoders.
  3. Analyses: the H1 question (being reframed; see the roadmap decision log,
     2026-10-14), the end-effector probe,
     the alien-body ablation, dense vs coarse touch, stereo.
- **Novel because** the benefit of senses beyond vision is established only
  with real sensors within one body, and task-free experience has not been
  tested as pretraining at equal volume in the few-shot regime
  ([literature-review.md](literature-review.md#verdict-per-hypothesis)).
- **Roadmap steps:** M3 (model), pilot, full E1.
- **Likely venue:** a robot-learning or ML conference (e.g. CoRL, ICLR).

## Experiments

### Track A

| # | Experiment | Shows | Compute | Status |
|---|---|---|---|---|
| A1 | Experience quality: experiments with outcomes vs random babbling vs expert demos; headline "caused change" per minute (objects displaced, lifted, toppled, stacked, opened; own posture), plus contact, coverage and skin regions (redirected 2026-10-15) | Task-free experience can be contact-rich; Track B's H2 depends on it | CPU | **approved** (2026-10-14) |
| A2 | Information per sense: probes from raw signals to known facts (material from sound, contact location and slip from skin, object pose from vision) | Each sense carries information the others lack | CPU, < 1 GPU-hour | **approved** (2026-10-14) |
| A3 | Sensor-model choices (dense vs grid skin, cube vs block pads, noise) judged by A2's probes | Whether the sensor design matters, at signal level | CPU | **approved** (2026-10-14) |
| A4 | Benchmark validity: VC-1, DINOv2, V-JEPA 2, HPT (frozen and LoRA) and scratch at k = 1, 5, 10, 50, 3 seeds | Tasks neither saturated nor impossible; methods separate; contact-rich tag justified | GPU, from E1's baseline budget | **approved** (2026-10-14) |
| A5 | Faithfulness: reproduce RP1M and BiGym published numbers on the H1 port | The conversion did not break the tasks | small GPU | **approved** (2026-10-14) |
| A6 | Generator properties: speed, storage, bit-identical regeneration, volume and diversity set independently | A reproducible tool | CPU | **approved** (2026-10-14) |
| A7+ | Sim vs real: signal statistics, information content and sim → real transfer of probes, per sense, on real data | How simulated senses differ from real ones, and what the sensor models can close | to be sized | **decided** (supervisor); design in progress |

### Track B (E1, [experiment-01.md](experiment-01.md))

| # | Experiment | Design |
|---|---|---|
| B0 | Pilot, go/no-go | vision vs all senses × expert vs babbling, 1 seed, frozen heads; 4 runs, ≈ $70–150 |
| B1 × B2 | H1 × H2 | senses (V, V+T, V+S, V+T+S; proprioception and vestibular sense always; approved 2026-10-14) × experience (expert, babbling, mix) at equal volume, 2 seeds; 24 runs |
| B3 | Comparisons | scratch; frozen and LoRA VC-1, DINOv2, V-JEPA 2; HPT on the same pool; RP1M and BiGym sanity numbers |
| B4 | Controls and ablations | diversity (8 vs all sources), effector (hands, pincers, both), alien bodies, body-only control, effector probe; ~6 runs |
| B5 | Analyses | which sense helps which task, demonstrations saved, probes |

Split of the dense-vs-coarse touch question (approved 2026-10-14): Track A measures the
information in the signal (A3); Track B, optionally (+2–4 runs), the effect on
few-shot learning. Stereo and binarised touch (≈ +2 runs each) are decided
after the pilot.

## Boundaries, so the two papers do not overlap

- Track A never trains our sensory layer; its benchmark reports
  off-the-shelf encoders and training from scratch only.
- Track B never builds data sources; it asks Track A for them, and reports on
  Track A's benchmark, citing it.
- Shared code (episode format, pools, loaders) has one owner per module,
  named in the PR; both may propose changes.
- Track B can start now: model code and a smoke-scale pretraining run on the
  existing samples, while Track A builds the sources.

## Open for the supervisor

- Confirm or change the two tracks.
- Authorship: proposed PhD first author on their own track; postdoc and
  supervisor on both.
- Later tracks (E2 development with MIMo, E3 language readout) once a PhD is
  free.
