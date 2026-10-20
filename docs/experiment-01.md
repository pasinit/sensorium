# Experiment 1: Senses × kind of experience, on one humanoid body

**Question.** Which *senses* in pretraining, and which *kind of experience*
(expert demonstrations vs task-free interaction), make new whole-body
skills learnable from a handful of examples?

**The result we want:** two main effects with confidence intervals on a
held-out suite of humanoid tasks at k = 1…50 demonstrations:
1. few-shot score vs **senses in pretraining** (vision → +touch → +sound → all)
2. few-shot score vs **kind of experience** (expert · babbling · mix), at equal
   volume

each compared against training from scratch and the best frozen vision
foundation encoders, with diversity (few vs many sources) as a check.

Data: [Experientia](dataset-proposal.md) (D0). Hypotheses H1–H2:
[thesis.md](thesis.md).

---

## 1. Design at a glance

```
 Experientia, one humanoid (H1) ──► pretraining pools, all the SAME number of steps
   hands:   piano (RP1M), goal babbling      senses:     vision | +touch | +sound | all
            (HumanoidBench reach policies),  experience: expert | babbling | mix
            seated babbling                  diversity:  8 sources | all
   pincers: BiGym home tasks (VR demos),     effector:   hands only | pincers only | both
            noisy replays, babbling
   body:    LocoMuJoCo mocap (walk, run, …)
                                                   │ pretrain (no reward, no text)
                                                   ▼
                                     sensory layer (stems + trunk)
                                                   │
                     ┌─────────────────────────────┼──────────────────────────────┐
                     ▼                             ▼                              ▼
          pincer side: 8 held-out BiGym    hand side: unseen songs,       body: held-out mocap
          tasks (human demos, k ≤ 50)      scripted hand tasks             motions (tracking)
                     └── frozen + head, and LoRA + head; k ∈ {1, 5, 10, 50}; 3 seeds ──┘
```

## 2. Hypotheses → measurements

| | Measurement | Supported if |
|---|---|---|
| **H1** senses beyond vision | Score vs senses condition, split by contact-rich vs vision-dominated held-out tasks | all > vision-only on contact-rich tasks at k ≤ 10, CIs exclude 0; gap smaller on vision-dominated tasks; all ≥ best frozen foundation encoder on the suite average |
| **H2** kind of experience | Score vs experience condition at equal steps | babbling > scratch and > vision-only encoders; mix ≥ expert |
| diversity *(secondary)* | 8 sources vs all, within the all-senses/mix cell | positive, consistent with MyoDex and Lin et al.; H1/H2 effects hold at both levels |
| end effector *(ablation)* | hands-only · pincers-only · both pools, equal steps, within the all-senses/mix cell | hand experience helps pincer few-shot and vice versa; reported either way |
| alien bodies *(ablation)* | all-senses/mix/all-sources pool, with and without car + drone | reported either way |

**Not tested** (already answered in the literature): passive vs
action-conditioned pretraining (V-JEPA 2 and DINO-WM are baselines instead);
multi-task pretraining helps on a hand (MyoDex); diversity as a scaling law
(Lin et al.).

## 3. Model: reuse first

- **Tokenisers:** HPT-style **stems**. Body stems (head cameras, body
  skin, body proprio + IMU, body actions) are shared by every episode.
  End-effector stems (effector proprio, effector touch, effector actions)
  come in two versions, hand and pincer; each episode uses the one it was
  recorded with, and the trunk is not told which. A sense that is absent
  from a condition is simply not fed (its stem is dropped), so all
  conditions share the trunk design.
- **Trunk and objective:**
  - A causal transformer over time, ~25–50M parameters in v0.
  - Objective: action-conditioned **latent prediction** of the next step's
    sense tokens, JEPA-style with an EMA target and an anti-collapse
    regulariser.
  - No reward, no text.
  - Fallback: the TD-MPC2/Newt world model (open code) with our stems.
- **The "sensory layer"** = stems + trunk. `encode(model_inputs) → latent`
  is the only interface the benchmark sees.

## 4. Conditions

| Axis | Values |
|---|---|
| Senses in pretraining | vision · vision + touch · vision + touch + sound · all (+ proprio) |
| Kind of experience, at equal steps | expert · babbling only · mix (expert ⅓, noisy expert ⅓, babbling ⅓) |
| Diversity (sources, equal steps) | 8 · all (~80 in v0), within the all-senses × mix cell only |
| End effector (equal steps) | hands only · pincers only · both (default), within the all-senses × mix cell only |
| Seeds | 2 for pretraining, 3 for few-shot heads |

Runs: 4 senses × 3 experience × 2 seeds = 24, plus diversity (2 more), the
end-effector ablation (2 more) and the alien-body ablation (2 more):
**~30 pretraining runs**. Pilot: 4 runs
(vision vs all senses) × (expert vs babbling), all sources, 1 seed.

Baselines on the same few-shot protocol:
- from scratch (same head plus a trainable encoder)
- frozen VC-1 / DINOv2 / V-JEPA 2, with and without LoRA
- HPT pretrained on the same pool (vision + proprio, its native senses)
- RP1M's published BC/BeT/Diffusion Policy numbers, as a sanity check on piano
- BiGym's published ACT / Diffusion Policy numbers on its demos, as a sanity
  check on the pincer side

## 5. Few-shot protocol

The Experientia benchmark, specified in [dataset-proposal.md](dataset-proposal.md#few-shot-evaluation-protocol-the-benchmark):
- two adaptations per encoder: **(a) frozen + 2-layer BC head**, **(b) LoRA**
  (rank 8, trunk only, stems trainable) + head
- k ∈ {1, 5, 10, 50} demonstrations per held-out task, 3 seeds
- metric: task success or normalised score (key-press F1 for piano)
- also linear probes of privileged state (contacts, object pose, key state)

H1 and H2 are reported on (a) as the cleaner measurement of the
representation, and on (b) as the practical number. Lin et al. (Table 2)
found a frozen DINOv2 scoring 0.00 where LoRA scored 0.72, so (b) is not
optional.

**Headline numbers:** the all-senses − vision-only gap at k = 10 on
contact-rich tasks (H1), and the babbling − scratch gap at k = 10 (H2), each
reported on the pincer side (BiGym) and the hand side separately. A
secondary measure: **demonstrations saved**, i.e. how many fewer
demonstrations than scratch are needed for the same score.

### The two end effectors

The body is one H1; only what is at the wrists differs (five-finger Shadow
hands, or two-finger Robotiq pincers for BiGym's human demos). Controls,
applied throughout:

1. **Constant mix.** Every pretraining pool, in every condition, has the
   same hand : pincer : body-only step ratio (fixed in the pool builder), so
   no senses or experience effect can be a change in effector mix.
2. **Variant probe.** A linear probe predicts the effector from the latent
   of body-only tokens. High accuracy means the trunk keeps two worlds; it
   is reported next to every result.
3. **Separate reporting.** Hand-side and pincer-side held-out tasks are
   never averaged together; the suite average is reported as a third number.
4. **Cross-effector ablation.** Hands-only vs pincers-only vs both pools at
   equal steps, evaluated on both sides: does hand experience help a cruder
   tool, and the reverse?
5. **Body-only control.** One few-shot run per layer with the effector
   tokens masked, to see how much of the few-shot score comes from the
   shared body senses.
6. **Retargeting side check (D0).** One BiGym task retargeted to the hands
   (same arm trajectories; pincer close → scripted five-finger grasp),
   replayed, success rate reported. If > 50% of demos survive, moving to a
   single effector later is a data change, not a model change.

## 6. Why this keeps GPU time low

- Each few-shot run on a frozen layer trains only a small head on cached
  latents, in minutes; LoRA runs add a backward pass through a 25–50M model
  on ≤ 50 demos, still minutes.
- Pretraining runs are small (25–50M parameters, ~5M steps each). The sweep
  is wide, not deep.
- Data is generated on the rented box at 256 px (frames stored as H.264,
  ~7 KB per step with both eyes) and cropped and resized to the model's input
  at load time ([vision.md](vision.md)); audio features are precomputed.
  Babbling data costs only simulation time.
- A **pilot first**: 4 runs, 1 seed, frozen heads only.
- Whole-body simulation is slower per step than a bare hand (≈ 70 DoF, ~2,800
  touch channels, two cameras), so data generation is a few days rather than a
  weekend. Rendering needs the box's GPU (EGL): in software it is ~260 ms per
  frame ([vision.md](vision.md#costs)).

## 7. Compute (rough, H100-equivalent hours)

| | Pilot | Full |
|---|---|---|
| Pretraining runs | 4 runs × 4–8 h | 30 runs × 4–8 h |
| Few-shot heads (frozen) | ~10 h | ~60–100 h |
| Few-shot LoRA | — | ~60–100 h |
| Debug / failed (+50%) | +50% | +50% |
| **Total** | **~40–65 h** | **~360–660 h** |

On rented spot A100s that is roughly
**$40–85 for the pilot and $340–890 for the full sweep**.

## 8. Milestones

| M | Deliverable | Done when |
|---|---|---|
| M0 | Thesis, plan, literature review | ✅ |
| M1 | Data pipeline + review sample (car, drone, piano) | ✅ |
| **D0** | Experientia v0: humanoid sources (piano, reach babbling, mocap, BiGym), three experience kinds, splits, benchmark harness, vision-only baselines | Your review of per-source pages |
| M3 | Stems + trunk + pretraining loop | Pretraining loss falls; no collapse |
| **Pilot** | 4 runs → few-shot suite | Go/no-go on the full sweep |
| M4 | Full sweep + baselines + ablations | Both main effects with CIs |
| M5 | Write-up (+ dataset release decision) | `docs/results-01.md` |

## 9. Risks

| Risk | Mitigation |
|---|---|
| Synthesised touch and sound carry no information beyond pixels (H1 null) | Touch and sound ablated separately; privileged-state probes show whether contacts are encoded at all; physics-based audio is a later step |
| Babbling never touches anything interesting (H2 null for a dull reason) | Babbling policies are goal-directed random reaches toward objects, with contact-rate monitoring; report contact statistics per pool |
| Equal-volume pools at low diversity overfit a single source | Matched training steps and early stopping on a held-out *pretraining* split; report pretraining loss too |
| Vision-only foundation encoders win outright | Possible: they saw far more data. Report a matched-data comparison (our vision-only layer) next to the off-the-shelf ones. |
| The trunk keeps hand and pincer episodes as two worlds | Variant probe reported with every result; cross-effector ablation; body-only control. If the probe is near-perfect and cross-effector transfer is zero, drop the pincer and retarget BiGym (D0 side check) |
| Whole-body babbling falls over | Free-space whole-body reaching uses HumanoidBench's released reach policy at 50 Hz, with falls monitored (about 1 in 5 episodes of 20 s fell in our scene); contact-rich babbling is seated or supported-standing with the pelvis fixed; contact and fall rates monitored per pool |
| BiGym demos don't all succeed (its README warns so) | Replay every demo, keep successes only, report the kept fraction per task |
| Mocap replay is kinematic, actions are position targets | Same caveat as piano; fine for BC heads; flagged in review pages |
| Piano action→motion pairs come from RP1M's simulator | First retry the replay with RP1M's App. B settings; fine for BC heads either way (review-sample.md) |
