# The sensory layer v0 (M3)

Code: `src/sensorium/model/`. Config: `configs/model/`. Design memo with the
literature check: `.tmp/phd-b/track_b_novelty_and_model.md` (not in git). Status: v0
built and tested on CPU; no GPU run yet (the supervisor decides compute).

```sh
make pretrain-smoke    # 300 steps of the tiny model on data/sample_h1, ~40 s on an M-series CPU
make measure           # parameters, tokens, FLOPs, CPU speed, loader throughput of configs/model/v0.toml
uv run sensorium-pretrain configs/model/v0.toml   # full size; needs a pool (data.pool) and a GPU
```

## What it is

```
 per model step (K = 4 observations, 0.2 s on the 20 Hz H1)
 vision  last frame, 128 px crop ─ ViT-Tiny (192 wide, 12 layers) ─ pool ─► 8 tokens ┐
 touch   54 skin regions (hands) ─ per-region linear ─ 2-layer transformer ─ pool ─► 6 ┤
 audio   4 x 32 log-mel ─ MLP ─► 1                                                     ├─► fusion encoder ─► state s_t (512)
 proprio body / vestibular / effector, 4 sub-steps each ─ MLP ─► 1 each                ┘     (4 layers, + summary token)
                                                                                             │
 action chunk (4 actions) ─ MLP ─► conditioning c_t ──► causal predictor (AdaLN-zero) ──► ŝ_{t+1}
 objective "sigreg":      MSE(ŝ_{t+1}, s_{t+1}) + 0.1 · SIGReg(s)         no EMA, no stop-gradient
 objective "ema_vicreg":  MSE(ŝ_{t+1}, EMA-encoder s_{t+1}) + var(s) + 0.04 · cov(s)   (see below)
```

* **Two objectives**, `model.objective`. `sigreg` is LeWorldModel's (above). `ema_vicreg`
  has the predictor regress the states of an **EMA copy of the whole state encoder** (stems,
  fusion, projector; stop-gradient; momentum `ema_momentum` to `ema_momentum_end`, linear over
  the run, default 0.99 to 0.999), with VICReg's **variance** (hinge on each dimension's std over
  the B x T online states, target 1) and **covariance** (squared off-diagonal covariance / D)
  terms on the online states as the anti-collapse part, weighted 1 : 0.04 against the prediction
  error (VICReg's 25 : 25 : 1; arXiv:2105.04906 Sec. 4.1-4.2, App. D.4). The target sees all
  senses (no sense dropout) and uses batch statistics in its BatchNorm. `pred` is measured
  against different targets in the two objectives, so it is not comparable across them.
* **One state per step.** `s_t` depends only on the window of observations ending at
  `e_t`; the predictor sees the sequence of states and the action chunks. So
  `SensoryLayer.encode(episode, stride)` is cheap and any stride is allowed.
* **Timing.** The window of `s_t` ends at `e_t`; the action chunk `a_t` is the K actions from
  `e_t` on, so `(s_t, a_t) -> s_{t+1}` is one transition (`episode.py` convention). A body with
  another control rate gets the observation spacing nearest 0.05 s (`data.spacing`: the car at
  10 Hz gets 0.4 s steps, the drone at 40 Hz subsamples by 2).
* **Senses are configuration.** `model.senses` is any subset of `vision, touch, audio,
  proprio_body, vestibular, proprio_effector`; `action` is always present. A sense that is
  not configured has no stem and no token (`SensoryModel.n_tokens`); a sense an episode does
  not have (a car has no skin) is masked in the fusion. `model.sense_dropout` drops each
  configured sense for a whole sequence at random, so a layer trained with all senses can run
  with fewer. A pool's `senses` (input keys) restrict the model senses (`senses_for_pool`).
* **Channel space** (`layout.py`). Proprio groups are chosen by name (`vest_*`, `gravity_*`,
  `pelvis_height` = vestibular; `q_lh_*`, `dq_rp_*` ... = effector; the rest body). Touch is cut
  into 54 regions (hands) / 24 (pincers) by name: taxels `hand/left/palm/7` and the dynamics
  `dynamics/hand/left/palm/slip` fall in one region (the body's `_link` suffix is dropped to
  join them). The model keeps the **union** of channel names over its dataset, so the hand and
  pincer episodes share the body channels' weights and the effector channels are separate.
  The touch stem's weights are per channel; its tokens carry an embedding of the region's
  block / side / segment, not of "hand" vs "pincer", so the trunk is not told the effector.
* **Loader** (`data.py`). Only `episode.model_inputs` reaches a batch; the tests plant a
  tripwire on `privileged` and `text`. Touch gets `touch_noise.perturb_touch` (once per
  sequence, not per episode) then `log1p`; frames get `vision.train_view` (one crop per
  sequence, centre crop in `encode`); proprio and audio are standardised with statistics fitted
  on the first 64 episodes (stored in the checkpoint). Each episode pick yields `reuse = 8`
  sequences, because decoding an episode costs more than cutting a sequence.

## Sources and what is verified

| Piece | Source | Status |
|---|---|---|
| Objective: next-state MSE + SIGReg, no EMA, AdaLN-zero predictor, projector with BatchNorm, frame-skip as action chunk | LeWorldModel, arXiv:2603.19312, github.com/lucas-maes/le-wm (MIT) | loss and SIGReg read in `train.py` / `module.py` (verified); paper numbers via a summarising fetch tool. `SIGReg` is copied from LeWM's statistic (attribution in `modules.py`) |
| Learned-query pooling to a fixed number of tokens | HPT, arXiv:2409.20537, github.com/liruiw/HPT (MIT) | Sec. 3.1 / App. A.2 via a summarising fetch tool |
| EMA target + VICReg terms | VICReg (Bardes et al. 2022, arXiv:2105.04906); V-JEPA 2 EMA 0.99925 or a ramp 0.999 to 1.0 (App. A.1, Tables 9-10) | VICReg term definitions and weights read via a fetch tool; our momentum 0.99 to 0.999 is shorter-run, not tuned |
| Fusion of sense tokens with a summary token | standard; Sparsh-X (arXiv:2506.14754) fuses touch modalities with bottleneck tokens | not copied |
| Region tokens from a skin layout | Tactile-JEPA (arXiv:2609.24385) uses a token per taxel and a taxel graph | idea only; our per-region linear is simpler |
| Frame stride / action chunks | V-JEPA 2-AC (4 fps, 16 frames), LeWM (frame-skip 5) | Sec. 3.1 of V-JEPA 2 via a fetch tool |

**Unverified or deviating** (also listed in the PR):

* LeWM's config uses `sigreg.weight` 0.09 while its paper says 0.1; we use 0.1.
* **SIGReg needs a large batch.** Its statistic compares the B samples of a time step with a
  Gaussian; it scores about 1 for any isotropic Gaussian and about B/2 for a collapsed
  embedding (measured with `tests/test_model.py::test_sigreg_*`), so at B = 8 it hardly tells them
  apart. LeWM trains with B = 128. `model.sigreg_pool_time` compares all B x T states at once;
  the tiny CPU config uses it. Whether the pooled statistic gives the same representation at
  B = 128 is untested.
* Applying SIGReg to a fused 512-d state is an extension of LeWM, which regularises one
  `[CLS]` embedding per frame.
* The predictor has 4 layers (the memo said 6): the AdaLN-zero modulation adds 1.6M parameters
  per layer, which put the model at 55M.
* Touch is stacked over K = 4 sub-steps, so the touch stem has K x 3.7k x 256 weights.
* The EMA objective's weights (1 : 1 : 0.04) and momentum are VICReg's / a guess, untuned.
  Checkpoints written before the EMA objective (state-dict keys without the `encoder.` prefix)
  do not load; none was kept.
* `train.save_steps`, `model.teacher`, `model.sense_dropout_senses` are used by the teacher test;
  the loader sees `model.data_senses` (the state encoder's senses plus touch for a teacher).
* Probe ridge uses one SVD per fold for all strengths and labels (same numbers as the closed form,
  tested); `train.save_steps` keeps numbered checkpoints.
* The DINOv2 vision-stem option of the memo is **not implemented** (no runs are planned yet).
* Not tested at scale: sense dropout, mixed bodies in one batch (car, drone, piano are supported
  by the channel space and tested only through the layout), `num_workers > 0`.

## Measurements (this Mac, 8 CPU threads, `make measure`)

| | v0 default (128 px) | v0 at 224 px | tiny CPU config |
|---|---|---|---|
| Parameters | 45.8M (vision 5.7, touch 5.6, fusion 12.6, predictor 18.9, rest 2.9) | same | 2.4M |
| Tokens per step (fusion) | 19 (8 + 6 + 1 + 3 + summary) | 19 | 12 |
| Training FLOPs per sequence (16 steps) | 69 GFLOPs | 139 GFLOPs | not measured |
| CPU training speed | 3.9 to 4.9 sequences/s (batch 2); 6.9 in a batch-8 run | 3.1 (batch 2) | ~65 (batch 8) |
| Loader, one process, stream (8 sequences per decoded episode), after PR #12 | 90 sequences/s (54 before) | not re-measured | |
| Loader, one process, episodes cached | 150 sequences/s | 122 | |

The memo estimated 74 / 159 GFLOPs; the counted values are within 15%.

### CPU runs on the H1 sample (8 episodes, 2,400 steps)

Collapse is read from two numbers of the states in a batch: the mean per-dimension std (about 1
for a healthy SIGReg embedding) and the effective rank (`exp` of the entropy of the normalised
singular values; the width at best, 1 when collapsed).

| Run | Batch | SIGReg | Steps | Loss first -> last | std | Effective rank (of 128) |
|---|---|---|---|---|---|---|
| tiny, 2.4M | 8 | per step (LeWM) | 300 | 0.35 -> 0.22 | 0.44 | 42.8 at step 25 -> **7.5** |
| tiny, 2.4M | 32 | per step | 300 | 0.76 -> 0.37 | 0.76 | 41.6 -> 22.3 |
| tiny, 2.4M (`cpu_tiny.toml`) | 8 | pooled over time | 300 | 1.43 -> 0.62 | 0.81 | 32.8 -> 22.1 |
| v0 size, 45.8M (width 512) | 8 | pooled | 60 | 2.46 -> 1.23 | 0.82 | 47.7 (of 512) -> 29.1 |
| tiny, H1 + car + drone + piano (24 episodes) | 8 | pooled | 40 | 1.98 -> 1.58 | 0.44 | 25.8 -> 19.5 (of 64) |

The loss falls in every run. The per-step statistic at batch 8 lets the rank drop to 7.5 while
the next-state error looks best (0.019): a partial collapse the loss does not punish. Pooling
the time steps (64 samples instead of 8) or a batch of 32 keeps the rank near 22. The sample
is 8 episodes of seated babbling and holding, so its own dimensionality is low and these
ranks cannot say what a healthy value is at scale; the GPU smoke test at batch 128 does.

**GPU estimate** (not measured). Time per run = sequences x GFLOPs / (989 TFLOPS dense bf16 x
utilisation), with the pool of 5M env steps = 78k non-overlapping sequences per epoch. At 128
px and 20 epochs: 1.6M sequences x 69 GF = 1.1e17 FLOPs = 0.12 to 0.30 H100-hours at 10 to 25%
utilisation; at 100 epochs 0.6 to 1.5 h. At 224 px, double. Utilisation and the loader are the
unknowns: decoding one 256 px H1 episode takes 0.089 s here with `Episode.load(privileged=False)`
(0.166 s before PR #12, which skipped the privileged right eye the model never uses).

## SIGReg vs EMA + VICReg at CPU scale

`scripts/compare_objectives.py`: the tiny model (2.4M, width 128) on the 8-episode H1 sample,
1,000 steps, batch 8 and 32, 2 seeds each (the two seeds agree within about 10% except per-step SIGReg at batch 32: eval rank 22.4 and 17.9; per-run summaries are in `runs/compare/`, git-ignored). *In-batch* values are the last four logged steps; *eval* values are 512
windows of the data encoded in eval mode after training, the same sample for every run, so they
do not depend on the batch size. The effective rank is at most 128 (the width) and, in-batch, at
most B x T - 1 = 63 (B = 8) or 255 (B = 32).

| Objective | Batch | In-batch std | In-batch erank | Eval std | **Eval erank** (of 128) |
|---|---|---|---|---|---|
| SIGReg per step (LeWM) | 8 | 0.60 | 3.7 | 0.67 | **5.7** |
| SIGReg per step | 32 | 0.92 | 17.5 | 1.00 | **20.1** |
| SIGReg pooled over time | 8 | 0.91 | 20.2 | 0.94 | **32.4** |
| SIGReg pooled over time | 32 | 0.97 | 42.4 | 0.99 | **46.2** |
| EMA + VICReg | 8 | 0.93 | 39.1 | 0.96 | **63.0** |
| EMA + VICReg | 32 | 0.99 | 63.5 | 1.01 | **68.3** |

* **Rank depends on the batch size under SIGReg and hardly under EMA + VICReg.** Per-step
  SIGReg goes from 5.7 to 20.1 when the batch grows 4x; the pooled variant from 32 to 46;
  EMA + VICReg from 63 to 68. This is the dependence the pilot's objective check should look
  at again at batch 128.
* **Not a quality ranking.** A higher rank is not a better representation: nothing here asks
  whether the dimensions carry the information a task needs (that is the benchmark's probes),
  and part of the spread can be sensor noise. The prediction error `pred` is measured against the
  online states for SIGReg and against the EMA states for the other, so it cannot be compared
  across objectives; under EMA it does not have to fall (0.64 to 0.69 at batch 8, 0.60 to 0.46 at
  batch 32) because the target moves.
* Caveats: 8 episodes of seated babbling and holding (low intrinsic dimension), a 2.4M model,
  1,000 steps, two seeds, untuned EMA / VICReg weights. A batch of 128 might lift per-step SIGReg
  as LeWM's does; that is what the GPU smoke test measures first.

## Probes of the encoded states (evaluation only)

`sensorium.model.probes` fits **ridge probes from the encoded state to privileged simulator
labels** on the H1 episodes. `train.probes = true` runs them on the final states; `scripts/
compare_objectives.py --probe` runs them on saved runs.

* **Labels** (per window of K observations): `contact_left`, `contact_right` (any contact of a
  `lh_*`/`lp_*` or `rh_*`/`rp_*` body), `contact_object` (an `obj_*` body touching a hand or pincer body; PR #15's 8-episode runs counted any object contact, which is true in every window of the 100-episode data because objects rest on the table, so the label changed and the PR #15 tables use the old one), `log_force` (mean
  `log1p` of the summed normal force), `tip` (the two hand tips' world position, 6-d). From
  `privileged/contacts`, `meta.body_names`, `privileged/tip_position`.
* **Protocol.** Leave-one-**episode**-out: no window of a test episode is in its training set. Ridge
  on features centred and divided by one global scale (not per column: rarely active touch
  channels would get huge weights), strength chosen by an inner episode-grouped 3-fold on the training
  episodes. Binary labels are scored by AUROC of the ridge output (0.5 = chance; threshold-free
  because the classes are imbalanced: 37 to 74% positive), continuous ones by pooled held-out
  R^2. `mean` averages `2 * (AUROC - 0.5)` and the R^2 values, so 0 is chance for every term.
  The encoder was trained on the same 8 episodes (self-supervised, no labels).
* **References.** `floor` = the same encoder at its random initialisation. `raw` = the non-visual
  model inputs of the same windows (touch after `log1p`, proprio groups, audio, last observation of
  each window), fitted with the same probe: what a perfect-memory encoder of those senses could offer.
* **Leakage guard.** Privileged arrays are read in `probes.py` and nowhere else in
  `sensorium.model`; the training path loads episodes with `Episode.load(path, privileged=False)`
  and builds batches from `model_inputs`. Tests enforce it: a spy on `Episode.load` while
  `SequenceDataset` samples (every call has `privileged=False` and returns no privileged array;
  batch keys are a subset of the senses, `action`, `present`, `region_mask`), and a source check
  that no other module in the package touches `.privileged` or `"privileged/..."`, and that every
  `Episode.load(` call in `data.py` passes `privileged=False`.

### Results (tiny 2.4M model, 8-episode H1 sample, 1,000 steps, 2 seeds; 1,192 windows)

| Encoder | Batch | Eval erank | contact L | contact R | contact obj | log force R^2 | tip R^2 | **mean** |
|---|---|---|---|---|---|---|---|---|
| SIGReg per step | 8 | 5.7 | 0.58 | 0.56 | 0.98 | 0.45 | 0.12 | **0.36** |
| SIGReg per step | 32 | 20.1 | 0.73 | 0.51 | 0.89 | 0.34 | -0.16 | **0.29** |
| SIGReg pooled | 8 | 32.4 | 0.65 | 0.61 | 0.86 | 0.17 | -0.12 | **0.26** |
| SIGReg pooled | 32 | 46.2 | 0.68 | 0.70 | 0.95 | 0.44 | -0.05 | **0.41** |
| EMA + VICReg | 8 | 62.8 | 0.62 | 0.64 | 0.62 | 0.00 | -0.12 | **0.13** |
| EMA + VICReg | 32 | 68.3 | 0.71 | 0.62 | 0.78 | 0.02 | -0.04 | **0.24** |
| **Random initialisation (floor)** | - | - | 0.72 | 0.77 | 0.98 | 0.32 | 0.13 | **0.48** |
| Raw non-visual inputs | - | - | 0.83 | 0.80 | 0.98 | 0.59 | 0.58 | **0.68** |

(The EMA batch-8 row has 3 seeds, one run from the postdoc's reproduction. Per-run `mean`:
SIGReg step 0.37/0.36 (b8), 0.33/0.24 (b32); pooled 0.27/0.24, 0.38/0.44; EMA 0.11/0.16/0.12,
0.08/0.40; floor 0.47/0.50.)

Probe `mean` against training steps (seed 0, batch 8):

| Objective | 0 (floor) | 100 | 300 | 1000 |
|---|---|---|---|---|
| SIGReg per step | 0.47 | 0.44 (erank 49) | 0.43 (14) | 0.37 (6) |
| SIGReg pooled | 0.47 | 0.51 (40) | 0.29 (38) | 0.27 (32) |
| EMA + VICReg | 0.47 | 0.51 (47) | 0.46 (51) | 0.11 (63) |

Without frames (touch, audio, proprio only): pooled SIGReg 0.30/0.31, EMA + VICReg 0.13/0.11
after 1,000 steps; random-init floor 0.60/0.59.

**Reading.**
* **The sample cannot separate the objectives, and no trained encoder beats the random one.**
  The random-initialisation floor (0.48) is above every trained configuration's mean
  (0.13 to 0.41; the best, pooled SIGReg at batch 32, is within seed noise of it). The probe
  protocol itself works: the raw inputs reach 0.68, with contact flags at 0.80 to 0.98.
* **Training lowered the probe score.** All three objectives start at the floor after 100 steps
  and fall afterwards (EMA + VICReg most: 0.51 to 0.11, while its effective rank rises from 47 to 63;
  per-step SIGReg loses rank and probe score together). Rank is not information: EMA + VICReg has
  the highest rank and the lowest probe score.
* **It is not the frames.** Without vision the floor is higher (0.60) and the trained encoders are
  no better (0.11 to 0.31), so the per-episode look does not explain the loss.
* **Seed noise is large** (EMA + VICReg at batch 32: 0.08 and 0.40), with 8 episodes and
  leave-one-episode-out folds in which only a few episodes have the contact being tested.
* **Likely reasons, not tested:** 8 episodes of seated babbling and holding is enough for a model
  to fit its own episodes' dynamics without keeping contact in a form that transfers to another
  episode; the tiny encoder compresses, whereas a random one keeps a random projection of everything.
  The test that would tell is a held-out-episode pretraining split (pretrain on some episodes,
  probe on others) on data large enough to have one: Track A's babbling data, not this sample.
  Until then a rank statistic or this probe table cannot pick SIGReg or EMA.

## The decisive held-out-episode test (120 episodes)

Question: does pretraining keep less contact information than a random encoder only because the
sample had 8 episodes? `scripts/decisive_test.py`; data from `configs/probe_test.toml`, which runs
**Track A's generator** (not a new source) at `main` commit `b44057d` (seated babbling, PR #14): 120
episodes of 600 steps (60 hand, 60 pincer; 519 MB; `data/probe_test`, untracked). The encoder is
pretrained on 100 (50 + 50) and probed on the 20 it never saw (10 + 10; leave-one-episode-out
inside that set, 1,500 windows), plus the same probes on 20 pretraining episodes ("seen").
Tiny model (2.4M), batch 16, **2,000 steps** (not the 3,000 steps at batch 32 of the brief: the
CPU was shared with another job, 9 to 14 sequences/s per run), 2 seeds, checkpoints at 100 / 300 /
1,000 / 2,000 steps (the learning rate follows one cosine schedule over the 2,000).

Held-out probe mean (`mean +- sd` over 2 seeds; seen episodes in brackets):

| Encoder | 100 | 300 | 1,000 | 2,000 |
|---|---|---|---|---|
| SIGReg pooled | 0.27 +- 0.03 (0.33) | 0.25 +- 0.00 (0.29) | 0.20 +- 0.01 (0.22) | **0.17 +- 0.02** (0.21) |
| EMA + VICReg | 0.26 +- 0.02 (0.32) | 0.29 +- 0.02 (0.33) | 0.30 +- 0.02 (0.31) | **0.29 +- 0.00** (0.29) |
| SIGReg pooled + sense dropout 0.5 (1,000-step schedule) | 0.29 +- 0.03 (0.33) | 0.32 +- 0.02 (0.34) | **0.35 +- 0.04** (0.37) | - |
| **Random init (floor)** | **0.31 +- 0.00** (0.38) | | | |
| Raw non-visual inputs | 0.22 (0.48) | | | |

* **No encoder trained as before beats the floor on held-out episodes.** EMA + VICReg reaches 0.29
  to 0.30 against 0.31 (one seed touches 0.32); pooled SIGReg falls to 0.17.
* **Pooled SIGReg keeps falling with training** (0.27 to 0.17), also on seen episodes, so it is not
  a generalisation gap: the encoder loses contact information. EMA + VICReg rises to 0.30 by 1,000
  steps and then stays.
* **Diagnosis (sense reliance = how far masking a sense moves the state, relative to the states'
  spread).** For pooled SIGReg the next-state error falls (0.56 to 0.13) while the state becomes
  proprio-dominated: reliance on `proprio_body` 0.41 to 0.79, on touch 0.07 to 0.04, vision 0.13
  to 0.04, audio 0.15 to 0.00. Its probe terms keep tip position (R^2 0.29, the floor's is 0.27) but
  lose contact (AUROC 0.64 / 0.56 / 0.54, force R^2 0.08). Masking touch or vision does nothing to its
  score; masking proprio *raises* it (0.17 to 0.25). So touch is down-weighted, not lost at the
  stem. EMA + VICReg also uses touch little (0.06 to 0.09) but keeps contact AUROC 0.66 to 0.67.
* **Test of that diagnosis (an intervention, not asked for in the brief).** Pooled SIGReg with
  sense dropout 0.5 (each sense dropped for a whole sequence) stops falling and rises: 0.29 -> 0.32
  -> 0.35 +- 0.04, with touch reliance back at 1.4 and `proprio_body` at 0.62. Per seed at 1,000
  steps: 0.31 (the floor) and **0.39** (above it: force R^2 0.33 vs 0.16, contact right 0.70 vs 0.66,
  object 0.73 vs 0.66). One seed clearly above, one at the floor; two seeds and a different
  schedule length, so it is a lead, not a result. The same dropout made EMA + VICReg worse (0.22,
  one seed).
* **Not only a small-sample artefact:** with 12x more episodes pooled SIGReg still loses contact
  information, EMA + VICReg stays below the floor, and the floor is above the raw inputs (0.22).

Caveats: tiny model only (no v0-size run); 2 seeds; batch 16 and 2,000 steps; the sense ablations
mask senses the model was not trained without (except with dropout); reliance is not comparable in
absolute terms between encoders; the label `contact_object` changed (see above).

## Input or teacher? A probe-level test for touch (B-teach-0)

`scripts/teacher_test.py`; same 120 episodes, split (pretrain 100, probe the 20 never seen), tiny
model, pooled SIGReg, batch 16 and 2,000 steps as the decisive test, so the schedules match. Arms
(`model.teacher`, `model.sense_dropout_senses`, `src/sensorium/model/teacher.py`):

* **a-input**: vision + touch + proprio (no audio), touch dropped in 50% of sequences (3 seeds);
* **b-teacher**: vision + proprio only. A touch-only world model (touch stem, projector, small
  predictor, SIGReg) trains alongside; an **EMA copy** of its encoder gives the targets; two heads
  on the main model predict them (from the state, the current touch latent; from the predictor's
  output, the next one). Touch is never an input of the state encoder and the targets are detached
  (6 seeds);
* **c-none**: vision + proprio only, no touch anywhere (6 seeds);
* **d-placebo**: as b, but the targets are the EMA latents of *another sequence's* touch (a roll by
  one in the batch): same statistics, no information about this sequence (6 seeds);
* **sd-all**: all six senses, dropout 0.5 on every sense (PR #18's lead, 5 seeds); **sd-touch**:
  all six senses, dropout on touch only (1 seed).

Every arm is probed on its **vision + proprio state** (touch, and audio, masked at probe time),
held-out episodes only, against the random-init floor of the same encoder (5 seeds). Tests
(`tests/test_model.py`): touch never enters the state encoder of the teacher arms (the states do not
change when the touch tensor is scrambled; encoder and teacher parameters are disjoint); the aux
loss trains the main model and the heads but gives no gradient to the touch encoder or its EMA
copy; the placebo targets are the real ones of the rolled batch; the EMA copy follows the encoder.

Held-out probe mean of the V+P state at 2,000 steps (mean +- sd over seeds; 0 = chance):

| Arm | Seeds | Probe mean | contact L | contact R | contact obj | force R^2 | tip R^2 |
|---|---|---|---|---|---|---|---|
| sd-all | 5 | 0.26 +- 0.04 | 0.69 | 0.61 | 0.60 | 0.21 | 0.26 |
| a-input | 3 | 0.21 +- 0.03 | 0.62 | 0.60 | 0.60 | 0.15 | 0.25 |
| **b-teacher** | 6 | **0.25 +- 0.03** | 0.70 | 0.60 | 0.62 | 0.12 | 0.31 |
| **c-none** | 6 | **0.22 +- 0.04** | 0.66 | 0.60 | 0.60 | 0.13 | 0.26 |
| **d-placebo** | 6 | **0.24 +- 0.04** | 0.69 | 0.60 | 0.59 | 0.13 | 0.30 |
| floor (V+P, random init) | 5 | 0.22 +- 0.02 | 0.70 | 0.57 | 0.58 | 0.14 | 0.26 |

Contrasts (difference of means +- standard error of the difference): **b - c = +0.03 +- 0.02**,
**b - d = +0.01 +- 0.02**, d - c = +0.02 +- 0.03, b - floor = +0.03 +- 0.02, c - floor = 0.00 +-
0.02, sd-all - floor = +0.04 +- 0.02. Per term, b - c is +0.04 +- 0.01 on contact left and +0.05 +- 0.01
on tip position, and d - c is +0.03 +- 0.01 and +0.04 +- 0.01: the same, so those gains belong to the
extra heads, not to touch.

* **Teacher use.** Against the true touch latents of held-out episodes, the heads of b predict
  R^2 = 0.43 +- 0.03 from the state (next touch from the predictor: 0.46 +- 0.03); the placebo's
  heads 0.11 +- 0.01. So the state does carry information that predicts touch latents, which
  never turned into more linearly decodable contact labels.
* **The dropout lead is not confirmed.** Five seeds of dropout on every sense, 2,000-step schedule:
  0.26 +- 0.04 (seeds 0.20 to 0.32) against 0.20 +- 0.03 for the random floor of the same encoder
  (+0.04 +- 0.02 vs the V+P floor), mean the same with touch present or masked at probe time (0.26
  vs 0.26), and below the random encoder with every sense present (0.30). The 0.39 of PR #18 was a
  good seed of a 1,000-step schedule. Dropout on touch only (sd-touch, a-input) is no better than
  no touch (0.21; 0.21 vs 0.22).
* **Reliance.** Without dropout the state is proprio-led in every arm (`proprio_body` 0.76 to
  0.79); a-input and sd-touch rely on touch (1.4) but their V+P-masked probe scores are no higher.

Caveats: tiny model; one probe family (contact flags, force, tip); with 6 seeds a difference below
about 0.04 on the probe mean is not resolved (SE 0.02); the aux target is a learned touch latent
that may mostly encode posture-dependent contact (the seated body touches the seat all the time),
which is predictable from vision and proprio but is not what the contact labels measure
(untested); the placebo can only be compared at equal architecture, and the arms' parameter counts
differ slightly (the teacher adds heads); `sd-touch` has one seed.

## Positive controls for the probe battery (B-ctrl-0)

**Training-set size for every number in this section:** 100 pretraining episodes of 600 steps =
60,000 env steps (about 50 minutes of experience at 20 Hz); an SSL run sees about 32,000
sequences (2,000 steps x 16); the supervised ceilings 16,000 (1,000 x 16); fine-tunes 4,800
(300 x 16) on 4 or 16 labelled episodes. Probes use the 20 other episodes.

Can the yardstick see a good representation at all? `scripts/controls_test.py`,
`src/sensorium/model/controls.py`; the 120 episodes and split of PR #18, the same probes
(leave-one-episode-out ridge on windows at stride 8, 20 held-out episodes), now with out-of-fold
predictions so every number has a 95% bootstrap CI over the held-out episodes.

| Control | n | Probe mean [95% CI over episodes] | contact L | contact R | contact obj | force R^2 | tip R^2 |
|---|---|---|---|---|---|---|---|
| floor, V+P (random init) | 5 | 0.22 [0.17, 0.27], seed sd 0.02 | 0.70 | 0.57 | 0.58 | 0.14 | 0.26 |
| floor, all senses | 5 | 0.30 [0.23, 0.35], seed sd 0.02 | 0.76 | 0.63 | 0.64 | 0.17 | 0.26 |
| SSL c-none (V+P, PR #20) | 6 | 0.22 [0.17, 0.26], seed sd 0.04 | 0.66 | 0.60 | 0.60 | 0.13 | 0.26 |
| SSL b-teacher (V+P) | 6 | 0.25 [0.20, 0.29], seed sd 0.03 | 0.70 | 0.60 | 0.62 | 0.12 | 0.31 |
| **supervised ceiling, V+P** | 3 | **0.64** [0.56, 0.71], seed sd 0.00 | 0.83 | 0.84 | 0.76 | 0.51 | 0.85 |
| **supervised ceiling, all senses** | 3 | **0.74** [0.67, 0.79], seed sd 0.01 | 0.90 | 0.88 | 0.85 | 0.59 | 0.84 |
| fine-tuned from scratch, 4 labelled episodes | 3 | 0.42 [0.33, 0.49], seed sd 0.03 | 0.74 | 0.72 | 0.66 | 0.31 | 0.54 |
| fine-tuned from SSL (c-none), 4 episodes | 3 | 0.41 [0.33, 0.47], seed sd 0.05 | 0.74 | 0.70 | 0.66 | 0.28 | 0.55 |
| fine-tuned from scratch, 16 labelled episodes | 3 | 0.51 [0.42, 0.58], seed sd 0.01 | 0.78 | 0.75 | 0.71 | 0.40 | 0.67 |
| fine-tuned from SSL (c-none), 16 episodes | 3 | 0.49 [0.41, 0.56], seed sd 0.01 | 0.76 | 0.73 | 0.72 | 0.42 | 0.64 |
| DINOv2 ViT-S/14 (frozen) | 1 | 0.15 [0.05, 0.22] | 0.68 | 0.49 | 0.46 | -0.05 | 0.52 |
| **DINOv2 + proprio** | 1 | **0.28** [0.21, 0.34] | 0.72 | 0.59 | 0.51 | 0.16 | 0.63 |
| proprio only | 1 | 0.12 [-0.28, 0.35] | 0.75 | 0.63 | 0.58 | 0.17 | -0.52 |
| raw V+P, PCA 128 (vision 16 px) | 1 | 0.14 [0.07, 0.20] | 0.71 | 0.51 | 0.45 | -0.03 | 0.40 |
| raw non-visual (touch, audio, proprio), PCA 128 | 1 | 0.26 [0.19, 0.31] | 0.79 | 0.67 | 0.58 | 0.14 | 0.09 |
| raw all senses, PCA 128 | 1 | 0.26 [0.16, 0.34] | 0.80 | 0.66 | 0.49 | 0.12 | 0.28 |

* **Read the CIs as episode-sampling only.** The bracketed intervals bootstrap the 20 held-out
  episodes with each encoder fixed; they do not include training-seed variance. A paired contrast
  such as b-teacher minus c-none = +0.031 [+0.010, +0.050] is therefore **not** evidence for the
  teacher: the minimum detectable difference with 6 seeds per arm is 0.083.
* **Supervised ceiling.** The tiny encoder trained directly on the probe labels on the 100
  pretraining episodes (`train_supervised`: the one place training reads labels, isolated from
  `pretrain.train`; BCE on the contact flags, MSE on standardised force and tip; 1,000 steps, batch
  16 sequences) scores 0.64 on the vision + proprio state and 0.74 with every sense present,
  held-out: **+0.42 +- 0.03 over the floor** (paired bootstrap SE; 95% CI [0.36, 0.49]). The battery
  has a dynamic range far above the 0.15 asked for, and the information is in vision + proprio.
* **DINOv2.** `facebookresearch/dinov2` ViT-S/14 distilled (22.06M parameters, measured), hub code
  pinned at commit `7764ea0`, weights `dinov2_vits14_pretrain.pth`, **Apache-2.0** (the model card;
  the README's non-commercial licence line belongs to its X-ray model); evaluation transform: resize
  256, centre-crop 224, ImageNet mean/std (the stored 256 px frames are cropped to 224 px), features =
  class token and mean patch token (768). Alone it is below the floor (0.15, paired difference
  -0.08 +- 0.05); **with proprio 0.28, +0.06 +- 0.035 over the V+P floor (95% CI [-0.00, +0.14])**,
  almost all from tip position (0.63 vs 0.26); its contact terms are at or below the floor.
* **Raw inputs done right.** PCA fitted on windows of 50 pretraining episodes (blocks scaled to
  unit RMS); PCA size changes little (32 vs 128). Raw V+P reaches 0.14 (vision at 16 px), raw
  non-visual 0.26, above the earlier un-reduced "raw 0.22" and equal to the floor with all senses:
  the earlier figure was partly overfitting. None comes near the supervised encoders.
* **SSL gives no few-shot benefit either.** Fine-tuning the c-none SSL encoder on 4 or 16 labelled
  episodes is **not better than fine-tuning from random weights** (SSL minus scratch: -0.014 +- 0.011
  at 4 episodes, -0.013 +- 0.010 at 16, paired SE); both clearly improve with more labelled episodes
  (scratch: 0.42, 0.51, 0.64 for 4, 16, 100 episodes).
* **Few-shot probes** (ridge on k labelled held-out episodes, fixed strength 1 x features, no search):
  at k = 2 the supervised V+P encoder scores 0.54 while every unsupervised feature is at or below
  0.11 (floor 0.03, SSL 0.02, DINO + proprio 0.05, proprio 0.11); at k = 19 DINO + proprio reaches
  0.36 against 0.26 for the floor.

### At 128 px (the tiny model rendered vision at 64 px)

`cpu_tiny` encodes frames at 64 px (patch 8), the resolution rejected for the data because a finger
is 1 to 2 px. Re-running at **vision 128 px, patch 16** (64 tokens either way), pooled SIGReg, batch 16,
2,000 steps, 3 seeds (training set as above):

| Encoder | 64 px: probe mean (n) | 128 px: probe mean (n) | 128 - 64 (paired SE) | contact L / R / obj, force R^2, tip R^2 at 64 px | the same at 128 px |
|---|---|---|---|---|---|
| floor, V+P | 0.22 (5) | 0.21 (5) | -0.016 (0.011) | 0.70 / 0.57 / 0.58, 0.14, 0.26 | 0.70 / 0.58 / 0.54, 0.14, 0.24 |
| SSL, V+P | 0.22 (6) | 0.23 (3) | +0.011 (0.008) | 0.66 / 0.60 / 0.60, 0.13, 0.26 | 0.66 / 0.64 / 0.58, 0.18, 0.23 |
| supervised ceiling, V+P | 0.64 (3) | 0.65 (3) | +0.004 (0.003) | 0.83 / 0.84 / 0.76, 0.51, 0.85 | 0.83 / 0.84 / 0.76, 0.53, 0.85 |
| floor, all senses | 0.30 (5) | 0.30 (5) | | 0.76 / 0.63 / 0.64, 0.17, 0.26 | 0.75 / 0.64 / 0.64, 0.18, 0.26 |
| SSL, all senses | 0.17 (2) | 0.22 (3) | +0.048 (0.007) | 0.64 / 0.56 / 0.54, 0.08, 0.29 | 0.67 / 0.58 / 0.57, 0.14, 0.29 |

* **Resolution is not what limits the probe.** The supervised encoder reads contact equally well at 64 px
  (0.64 vs 0.65), so 64 px already holds the information it needs; the floor does not move; SSL V+P
  moves by +0.01 +- 0.01 and stays at the floor (+0.024 +- 0.021 over the 128 px floor). SSL with all
  senses gains +0.05 (the 64 px runs are the decisive test's two seeds) but is still **below its 128 px
  floor (-0.087 +- 0.020)**.
* **CPU cost per training step** (one process, 16 sequences, forward + backward + AdamW, tiny model):
  V+P 0.06 s at 64 px, 0.09 s at 128 px (+50%); all senses 0.11 s and 0.13 s (+18%); 0.5 to 0.6 and
  0.9 to 1.0 GFLOPs per sequence; parameters +0.03M. Full runs are loader-bound at about 20 sequences/s,
  so wall time barely changes (c128 19.9, a128 17.0 sequences/s under contention; 64 px 14 to 23).
* **DINOv2** was already run at its native 224 px crop (above), the stored frames being 256 px.
* The loss curves (training terms, held-out loss, held-out probe against steps, floor lines, every
  seed) are in `.tmp/phd-b/ctrl0_128px_curves.png`: the held-out SSL loss falls and the probe score
  does not rise.

**Variance.** Two components. Episode sampling: the paired-bootstrap SE of a difference between two
encoders on the same 20 episodes is 0.01 to 0.02 (0.020 between two floor seeds). Training seeds: sd
of the probe mean 0.02 (floor) to 0.04 (SSL), 0.00 to 0.01 for supervised encoders. Minimum
detectable difference (80% power, 5% two-sided, 2.8 x SE of the difference):

| seeds per arm | 20 held-out episodes | 80 held-out episodes |
|---|---|---|
| 1 | 0.16 | 0.15 |
| 3 | 0.10 | 0.09 |
| 6 | 0.083 | 0.067 |
| 12 | 0.071 | 0.051 |
| 24 | 0.064 | 0.042 |

Seeds, not episodes, dominate: quadrupling the held-out episodes (80, which would leave 40 for
pretraining of the 120) barely helps; 24 seeds per arm halves it.

**Ridge-strength sensitivity.** The probe's inner search (3 episode-grouped folds) is a weak point for
high-dimensional features: DINO + proprio scores 0.18, 0.33, 0.36, 0.24, 0.10 at fixed strengths 0.01,
0.1, 1, 10, 100 x the number of features (search: 0.28; 5 or 10 inner folds: 0.31); proprio alone
0.10 to 0.32 (search 0.12, tip R^2 -0.52). Encoder states of 128 dimensions are stable (floor 0.22 to
0.26 for strengths up to 10). `probes.INNER_FOLDS` makes it a constant; PR #18 to #20 numbers use 3.

**Reproducibility.** A run is bit-identical for the same thread count (tested with and without loader
workers, `tests/test_model.py::test_the_same_thread_count_gives_the_same_run`); a different count
changes BLAS reduction order by about 1e-6 and training amplifies that to a visible gap within 50
steps (b-teacher, 100 steps: 3.958 vs 3.950 at 2 vs 8 threads; the loader and RNG streams are seeded and
unaffected). `train.threads` sets it; `run.json` records it. There is no unseeded RNG in the teacher,
EMA or placebo path.

## Consequence-rich data, objective variants and a converged fine-tune yardstick (B-conseq-0)

**Training-set size for every number in this section:** 100 pretraining episodes of 600 steps =
60,000 env steps (about 50 minutes at 20 Hz); an SSL run is 1,000 steps x 16 sequences = 16,000
sequences (one sequence = 16 model steps of 4 observations, 32 env steps in the tiny config of 8 x 4);
fine-tunes use 16 labelled episodes (9,600 env steps), 600 steps x 16. Probes use the 20 other
episodes, same battery and split as before. Tiny model, 64 px, pooled SIGReg, batch 16, 3 seeds per
arm. Code: `scripts/objective_test.py`, `scripts/finetune_test.py`, `src/sensorium/model/subsets.py`;
figures `.tmp/phd-b/conseq_subsets.png`, `conseq_objectives.png`, `conseq_finetune.png`.

**New model options (all optional, defaults unchanged).** `model.proprio_dropout` drops every
proprio group together for a whole sequence (vision must carry the state). `model.surprise` weights
each step's next-state loss by the model's own signal, normalised to mean 1 and clipped to
[0.2, 5]: `error` (its own detached prediction error) or `touch` (the step-to-step change of the
touch input). Neither uses privileged data. `data.subset_index` / `subset_key` restrict the
sequences drawn to one class of start positions of an offline index.

**The consequence-rich subset.** `probes.consequence_steps` marks a step when a hand or pincer
body touches an object and that object has moved more than 1 cm 10 steps (0.5 s) later. This reads
privileged contacts and object poses **offline, only to choose which windows are drawn**; the
sampler loads episodes with `privileged=False` and a test checks both. A window start is
*consequence-rich* if at least 25% of its 32 steps are consequence steps, *contact-only* (the
control) if at least 25% have hand-object contact but fewer than 25% have a consequence. Over the
100 pretraining episodes: 56,900 valid starts, 6,811 consequence-rich (12%, from 90 episodes),
4,982 contact-only (93 episodes). Hand-object contact is in 14% of steps, a consequence in 8%.

### Frozen probe: held-out probe mean of the encoder's state (mean +- sd over 3 seeds)

| arm | 250 steps | 500 | 750 | 1,000 |
|---|---|---|---|---|
| sub-rand (V+P, uniform sequences) | 0.22 +- 0.04 | 0.24 +- 0.06 | 0.25 +- 0.04 | 0.25 +- 0.03 |
| sub-conseq (V+P, consequence-rich only) | 0.24 +- 0.02 | 0.27 +- 0.02 | 0.26 +- 0.03 | 0.27 +- 0.04 |
| sub-contact (V+P, contact without consequence) | 0.23 +- 0.01 | 0.26 +- 0.03 | 0.26 +- 0.03 | 0.27 +- 0.03 |
| c-pdrop (V+P, proprio dropout 50%) | 0.18 +- 0.04 | 0.08 +- 0.04 | 0.05 +- 0.06 | 0.07 +- 0.04 |
| c-err (V+P, error-weighted loss) | 0.24 +- 0.02 | 0.27 +- 0.01 | 0.27 +- 0.01 | 0.27 +- 0.01 |
| a-base (all senses) | 0.29 +- 0.02 | 0.26 +- 0.01 | 0.28 +- 0.02 | 0.28 +- 0.01 |
| a-touch (all senses, touch-change-weighted loss) | 0.31 +- 0.01 | 0.34 +- 0.02 | 0.32 +- 0.02 | 0.32 +- 0.03 |
| floor V+P (random init), n 5 | 0.22 +- 0.02 | | | |
| floor all senses, n 5 | 0.30 +- 0.02 | | | |
| supervised ceiling V+P, n 3 (B-ctrl-0) | 0.64 | | | |

Contrasts at 1,000 steps (difference of seed means +- its standard error; MDD = 2.8 SE, the smallest
difference this design resolves). The CI of B-ctrl-0 covers episode sampling only; these include
training-seed variance.

| contrast | difference | SE | MDD |
|---|---|---|---|
| sub-conseq - sub-rand | +0.015 | 0.034 | 0.095 |
| sub-contact - sub-rand | +0.020 | 0.029 | 0.081 |
| sub-conseq - sub-contact | -0.005 | 0.031 | 0.087 |
| c-err - sub-rand | +0.015 | 0.023 | 0.064 |
| **c-pdrop - sub-rand** | **-0.178** | 0.037 | 0.105 |
| a-touch - a-base | +0.036 | 0.022 | 0.063 |
| sub-conseq - floor V+P | +0.045 | 0.027 | |
| a-base - floor all | -0.016 | 0.013 | |
| a-touch - floor all | +0.021 | 0.024 | |

Terms of the probe at 1,000 steps (contact L / R / object AUROC, force R^2, tip R^2): sub-rand
0.69 / 0.60 / 0.61 / 0.20 / 0.27; sub-conseq 0.70 / 0.62 / 0.63 / 0.20 / 0.26; floor V+P 0.70 /
0.57 / 0.58 / 0.14 / 0.26. No term moves by more than its seed spread.

### Fine-tune yardstick, run to convergence (16 labelled episodes, held-out probe of the fine-tuned state)

| init | recipe | 100 steps | 200 | 400 | 600 |
|---|---|---|---|---|---|
| random | std | 0.47 +- 0.00 | 0.53 +- 0.01 | 0.55 +- 0.02 | 0.56 +- 0.02 |
| sub-rand | std | 0.44 +- 0.01 | 0.50 +- 0.01 | 0.53 +- 0.01 | 0.54 +- 0.01 |
| sub-conseq | std | 0.45 +- 0.01 | 0.51 +- 0.01 | 0.53 +- 0.02 | 0.54 +- 0.02 |
| random | lpft | 0.17 +- 0.02 | 0.33 +- 0.01 | 0.40 +- 0.01 | 0.42 +- 0.01 |
| sub-rand | lpft | 0.25 +- 0.03 | 0.30 +- 0.01 | 0.37 +- 0.01 | 0.39 +- 0.01 |
| sub-conseq | lpft | 0.27 +- 0.03 | 0.32 +- 0.02 | 0.38 +- 0.02 | 0.40 +- 0.02 |

`std`: lr 3e-4 for everything. `lpft`: encoder frozen for 100 steps while the head fits, then the
encoder at 0.1 of the head's lr. Floor 0.22, ceiling 0.64. Fine-tune seed spread is 0.01 to 0.02, so
this yardstick resolves about 0.03 to 0.05 (MDD), far finer than the frozen probe. At 600 steps:
sub-rand - random -0.018 +- 0.012 (-0.028 +- 0.007 at 200 steps), sub-conseq - random -0.014 +- 0.018,
sub-conseq - sub-rand +0.004 +- 0.014.

### Reading

* **(a) Contact does not survive pretraining even in the consequence-rich subset.** The subset
  encoder is +0.015 +- 0.034 above the uniform one (MDD 0.095), the contact-only control is
  +0.020, and the three sit about +0.03 to +0.05 above the random-init floor (not resolved). The fine-tune
  yardstick says the same with ten times the resolution: neither SSL init beats training from
  scratch (-0.014 to -0.018, within 1.5 SE; at 200 steps the uniform init is slightly *worse*,
  -0.028 +- 0.007). So a better *sequence distribution* alone does not rescue this objective at this
  scale: the data hypothesis is not supported by the oracle subset. A caveat that cuts the other
  way: the subset holds 6,811 starts, a small fraction of a small run (16,000 draws, so each start is
  seen about 2.3 times), and the tiny model is small; a null here is not a null for v0 size.
* **The fine-tune yardstick was under-trained in B-ctrl-0 (300 steps: 0.51 from scratch, 0.49 from
  SSL; 600 steps: 0.56 and 0.54), but the conclusion did not change.** At 600 steps it is still
  rising slowly (+0.01 over the last 200 steps; ceiling 0.64), and both inits rise in parallel, so
  the init gap does not close with training. Linear-probe-then-fine-tune
  and a lower encoder lr are worse at every step and not converged at 600 (0.40 to 0.42).
* **(b) Proprio dropout 50% is clearly harmful**: -0.178 +- 0.037 against the uniform arm (above its
  MDD), the score falling from 0.18 at 250 steps to 0.05 to 0.07, and the state partly collapses (seed 0: effective
  rank 14 against 32 in the uniform arm, training prediction loss 0.11 against 0.21). My reading,
  not tested: with proprio missing half the time the prediction target gets easier to satisfy with a
  low-rank vision state, and the proprio-linked contact content is lost. I do not recommend it for this
  objective.
* **(b) Error-weighted loss** is neutral (+0.015 +- 0.023, above the floor by +0.045 +- 0.010 on
  V+P). **Touch-change-weighted loss with all senses** is +0.036 +- 0.022 above its unweighted twin,
  1.6 SE and below the 0.063 MDD, and +0.021 +- 0.024 above the all-sense floor: the first arm with all
  senses to sit at or above its floor at every step, but not resolved. With three seeds it is the only
  lead and it is worth the next seeds.
* The all-sense unweighted arm sits below its floor (-0.016 +- 0.013) as in B-ctrl-0; the held-out
  probe peaks early (a-base 0.29 at 250 steps, then 0.26 to 0.28), a weaker version of the
  peak-then-fall of the 2,000-step runs.

## Random play against scripted experiments (B-exp-0)

**Training-set size for every number in this section:** `rand` arm, 100 random-play episodes of 600
steps = 60,000 env steps; `exp` arms, 59 experiment episodes = about 35,000 env steps (hand press,
push, slide, topple and pincer push, topple; 10 or 9 per variant); every SSL run sees the same
16,000 sequences (1,000 steps x 16, tiny model, 64 px, pooled SIGReg, vision + proprio), 3 seeds.
Fine-tunes: 16 labelled episodes (9,600 env steps), 600 steps x 16. Code
`scripts/experiments_test.py`; figures `.tmp/phd-b/exp_frozen_randset.png`, `exp_frozen_expset.png`,
`exp_ft_randset.png`, `exp_ft_expset.png`.

**The batch as delivered** (`data/experiments_batch`, 79 episodes): hand press / push / slide /
topple 12 each, pincer push 11, topple 12, slide 6, press 2, **no knock**. Pincer slide and press
(weak, 6 and 2 episodes) are excluded; 2 episodes per remaining variant are held out (12 episodes,
8 of them hand). All arms share one channel space built from both kinds of pretraining episode (the
effort channels `eff_*`, 61 for hands, exist only in the experiments; they read 0 in random play).
`data.drop_prefix = ("eff_",)` leaves them out ("proprio without effort", arm `exp-noeff`);
`data.space_paths` builds the shared space.

**Windows with a consequence.** Same definition as above (at least 25% of the 32 steps are contact
that moves the object more than 1 cm within 0.5 s). Random play 12.0% of window starts; experiments
7.4% overall (hand topple 12.6%, hand push 9.3%, hand press 6.3%, hand slide 6.2%, pincer topple
5.6%, pincer push 3.8%). By this measure the scripted experiments are **not richer** than random
play; the criterion cannot tell a deliberate push from a bump, so this is a statement about the
criterion as much as about the data.

Two held-out probe sets, same battery: `rand` (the 20 held-out random-play episodes) and `exp` (12
held-out experiment episodes). The `exp` set has less range: floor 0.28, supervised ceiling 0.44
(0.67 on `rand`), and its object-contact term is at chance even for the supervised encoder (AUROC
0.40; positives are 12% of windows and zero in some episodes), so differences on `exp` are read
with less weight.

### Frozen probe (mean +- sd over 3 seeds; floor n 5)

| arm | 250 steps | 500 | 750 | 1,000 | | 250 | 500 | 750 | 1,000 |
|---|---|---|---|---|---|---|---|---|---|
| | **probe set `rand`** | | | | | **probe set `exp`** | | | |
| rand | 0.18 +- 0.03 | 0.20 +- 0.02 | 0.22 +- 0.02 | 0.23 +- 0.02 | | 0.23 +- 0.02 | 0.20 +- 0.03 | 0.21 +- 0.01 | 0.21 +- 0.02 |
| exp | 0.20 +- 0.00 | 0.20 +- 0.01 | 0.19 +- 0.01 | 0.19 +- 0.01 | | 0.20 +- 0.07 | 0.21 +- 0.03 | 0.19 +- 0.02 | 0.20 +- 0.02 |
| exp-conseq | 0.21 +- 0.01 | 0.19 +- 0.01 | 0.18 +- 0.02 | 0.18 +- 0.02 | | 0.20 +- 0.01 | 0.22 +- 0.03 | 0.21 +- 0.01 | 0.21 +- 0.01 |
| exp-noeff | 0.21 +- 0.01 | 0.19 +- 0.03 | 0.17 +- 0.04 | 0.17 +- 0.04 | | 0.14 +- 0.01 | 0.17 +- 0.01 | 0.12 +- 0.05 | 0.12 +- 0.05 |
| floor (random init) | 0.20 +- 0.06 | | | | | 0.28 +- 0.03 | | | |
| floor, no effort | 0.21 +- 0.02 | | | | | 0.25 +- 0.01 | | | |
| supervised ceiling | 0.67 +- 0.00 | | | | | 0.44 +- 0.01 | | | |

Contrasts at 1,000 steps (difference of seed means +- SE; MDD = 2.8 SE):

| contrast | probe set `rand` | probe set `exp` |
|---|---|---|
| exp - rand | -0.039 +- 0.015 (MDD 0.041) | -0.012 +- 0.018 (0.051) |
| exp-conseq - exp | -0.010 +- 0.018 (0.050) | +0.011 +- 0.017 (0.049) |
| exp-noeff - exp (effort as input) | -0.015 +- 0.031 (0.086) | **-0.076 +- 0.039** (0.110) |
| rand - floor | +0.031 +- 0.030 | -0.066 +- 0.019 |
| exp - floor | -0.007 +- 0.030 | -0.078 +- 0.021 |
| exp-noeff - floor no effort | -0.034 +- 0.031 | -0.130 +- 0.037 |

### Fine-tune yardstick (16 labelled episodes of the probed distribution; held-out probe of the fine-tuned state)

| init | `rand` 100 steps | 200 | 400 | 600 | | `exp` 100 steps | 200 | 400 | 600 |
|---|---|---|---|---|---|---|---|---|---|
| random | 0.47 +- 0.01 | 0.51 +- 0.01 | 0.55 +- 0.01 | 0.55 +- 0.01 | | 0.30 +- 0.03 | 0.34 +- 0.01 | 0.35 +- 0.01 | 0.37 +- 0.02 |
| rand SSL | 0.45 +- 0.01 | 0.52 +- 0.01 | 0.54 +- 0.01 | 0.55 +- 0.01 | | 0.28 +- 0.01 | 0.31 +- 0.01 | 0.32 +- 0.02 | 0.34 +- 0.02 |
| exp SSL | 0.43 +- 0.02 | 0.50 +- 0.01 | 0.54 +- 0.02 | 0.54 +- 0.01 | | 0.27 +- 0.02 | 0.30 +- 0.02 | 0.32 +- 0.02 | 0.34 +- 0.01 |
| exp-conseq SSL | 0.44 +- 0.01 | 0.51 +- 0.00 | 0.54 +- 0.01 | 0.55 +- 0.01 | | 0.29 +- 0.02 | 0.31 +- 0.01 | 0.33 +- 0.02 | 0.33 +- 0.02 |

At 600 steps on `exp`: rand SSL - random -0.030 +- 0.021, exp SSL - random -0.030 +- 0.017,
exp-conseq SSL - random -0.041 +- 0.020, exp SSL - rand SSL +0.001 +- 0.018; on `rand` all within
0.015 of random init (exp SSL - random -0.015 +- 0.011).

### Reading

* **Experiments did not beat random play here, and the random-play-set contrast is confounded.**
  The frozen probe of the `exp` encoder is -0.039 +- 0.015 against `rand` on the random-play set
  (about 2.6 SE, MDD 0.041) and indistinguishable on the experiment set (-0.012 +- 0.018). The first
  number mixes two things: 35,000 against 60,000 env steps of pretraining data, and an
  in-distribution advantage on a probe set made of random play. The size-matched control (`rand59`)
  was not run in B-exp-0 (it was run in B-s2b-0 below as `rand60`), so read it as "no gain", not as
  "experiments are worse". The consequence-rich windows of the experiments add nothing (-0.010,
  +0.011). On the experiment set every SSL arm sits 0.07 to 0.08 *below* its random-init floor
  (0.28), the same pattern as in B-conseq-0. **The batch had no knock episodes** (79 of the 120
  planned; pincer slide and press were excluded as weak).
* **The fine-tune yardstick agrees and is finer.** No SSL init, random-play or experiment, beats
  training from scratch; on the experiment set all three are 0.03 to 0.04 below it at 600 steps.
  Experiments do not make the encoder a better starting point for contact labels.
* **Effort as an input** helps on the experiment set: removing the `eff_*` channels lowers the
  probe by 0.076 +- 0.039 (2 SE, under the 0.110 MDD, so unresolved) and the no-effort encoders sit
  0.13 below their floor against 0.08 with effort. On random play there is no effort in the data and
  no effect (-0.015 +- 0.031). Effort carries contact force directly, so a gain is expected; it is a
  hypothesis for the 6-seed run and for a hidden-property label (mass, friction), not a result.
* **Caveats.** 59 experiment episodes (35,000 env steps) against 100 random-play (60,000): the
  experiments pool is smaller and has no knock programs and weak pincer variants; the `exp` probe set
  is 12 episodes, 8 of them hand; one batch, one seed set. This is the same model (tiny, 64 px) and
  objective that fails at its floor in B-conseq-0, so a null here says the data do not rescue it, not
  that the data are uninformative.

## Hand experiments against random play (B-s2b-0)

**Training-set size for every number in this section:** `s2b-*` arms pretrain on 60 hand-experiment
episodes of 600 steps = 36,000 env steps (push, topple, knock, press, slide, door, 10 per program;
Track A's `experiments_s2b`, hands only); `rand60-*` controls on a seeded random 60 of the 100
random-play pretraining episodes = 36,000 env steps (both effectors). Every SSL run sees the same
16,000 sequences (1,000 steps x 16, tiny model, 64 px, pooled SIGReg), 3 seeds. Fine-tunes: 16
labelled episodes (9,600 env steps), 600 steps x 16. Code `scripts/s2b_test.py`; figures
`.tmp/phd-b/s2b_frozen_expset.png`, `s2b_frozen_randset.png`, `s2b_frozen_expset_vp.png`.

Two held-out probe sets: `exp` (12 hand-experiment episodes, 2 per program, windows every 4 steps)
and `rand` (the 20 held-out random-play episodes, for comparability; every 8 steps). All arms share one
channel space built from both kinds of episode; `drop_prefix = ("eff_",)` removes the effort
channels (`-noeff` arms). `exp_vp` is an all-sense encoder probed with touch and audio masked.
Hidden-property probes (`probes.HIDDEN`, evaluation only, like the contact labels): R^2 of
leave-one-episode-out ridge probes of the log mass, friction and log softness (`solref_s`) of the
object a hand touches for most of the window, on windows that have one (275 windows in 12 episodes);
the ceiling is an encoder trained directly on those labels as well as the contact labels.

**What counts as a consequence.** By the 1 cm criterion hand topple has 25.8% consequence windows
against 15.6% for random-play hands, and the other hand programs 1.6% to 10.2%. Track A's
caused-change event spans (`probes.event_steps`, the `event` index key) saturate: at least 25% of the
steps of 63% of random-play windows (80% of random-play hand windows) are inside an event; for the
hand programs 27% (door), 36% (slide), 71% to 74% (press, knock, push), 80% (topple). An `event`
arm would therefore be close to uniform sampling, so none was trained.

### Frozen probe on the experiment set `exp` (mean +- sd over 3 seeds)

| arm | 250 steps | 500 | 750 | 1,000 |
|---|---|---|---|---|
| s2b-vp | 0.22 +- 0.01 | 0.15 +- 0.04 | 0.14 +- 0.05 | 0.13 +- 0.06 |
| rand60-vp | 0.28 +- 0.01 | 0.24 +- 0.07 | 0.25 +- 0.05 | 0.25 +- 0.05 |
| s2b-vp-noeff | 0.21 +- 0.01 | 0.17 +- 0.03 | 0.14 +- 0.04 | 0.12 +- 0.05 |
| s2b-all | 0.33 +- 0.00 | 0.31 +- 0.01 | 0.27 +- 0.02 | 0.25 +- 0.03 |
| rand60-all | 0.21 +- 0.03 | 0.23 +- 0.02 | 0.21 +- 0.05 | 0.22 +- 0.05 |
| **s2b-all-tw** | 0.35 +- 0.03 | 0.35 +- 0.02 | 0.33 +- 0.02 | **0.33 +- 0.02** |
| s2b-all-noeff | 0.34 +- 0.02 | 0.25 +- 0.06 | 0.21 +- 0.08 | 0.20 +- 0.08 |
| floor V+P / all / V+P no eff / all no eff (n 5 / 5 / 3 / 3) | 0.25 +- 0.03 / 0.28 +- 0.03 / 0.28 +- 0.04 / 0.34 +- 0.01 | | | |
| supervised ceiling V+P / all (n 3) | 0.45 +- 0.02 / 0.67 +- 0.01 | | | |

On `rand` at 1,000 steps: s2b-vp 0.09 +- 0.04, rand60-vp 0.24 +- 0.01, s2b-vp-noeff 0.08 +- 0.02,
s2b-all 0.22 +- 0.02, rand60-all 0.25 +- 0.01, s2b-all-tw 0.26 +- 0.02, s2b-all-noeff 0.25 +- 0.02;
floors 0.18 +- 0.04 (V+P) and 0.27 +- 0.02 (all). Masked `exp_vp` (all-sense arms): s2b-all 0.24 +-
0.03, rand60-all 0.21 +- 0.06, s2b-all-tw 0.25 +- 0.02, s2b-all-noeff 0.19 +- 0.07; floor 0.24 +- 0.03.

Contrasts at 1,000 steps (difference of seed means +- SE; MDD = 2.8 SE in brackets):

| contrast | `exp` | `rand` |
|---|---|---|
| s2b-vp - rand60-vp | -0.120 +- 0.054 (0.151) | **-0.147 +- 0.029 (0.081)** |
| s2b-all - rand60-all | +0.029 +- 0.043 (0.120) | -0.031 +- 0.018 (0.050) |
| **s2b-all-tw - s2b-all** (touch-change weighting) | **+0.079 +- 0.026 (0.072)** | +0.043 +- 0.023 (0.064) |
| s2b-vp-noeff - s2b-vp (effort as input, V+P) | -0.004 +- 0.053 (0.149) | -0.014 +- 0.031 (0.086) |
| s2b-all-noeff - s2b-all (effort as input, all) | -0.047 +- 0.061 (0.170) | +0.034 +- 0.022 (0.063) |
| s2b-vp - floor V+P | -0.126 +- 0.042 | -0.089 +- 0.033 |
| rand60-vp - floor V+P | -0.006 +- 0.040 | +0.058 +- 0.019 |
| s2b-all - floor all | -0.036 +- 0.026 | -0.059 +- 0.019 |
| s2b-all-tw - floor all | +0.043 +- 0.020 (0.057) | -0.017 +- 0.017 |

Terms at 1,000 steps on `exp` (contact L / R / object AUROC, force R^2, tip R^2): s2b-all 0.71 / 0.66 /
0.64 / 0.20 / 0.00; **s2b-all-tw 0.72 / 0.69 / 0.70 / 0.43 / -0.01**; rand60-all 0.72 / 0.60 / 0.63 / 0.16 /
0.03; s2b-vp 0.69 / 0.47 / 0.59 / 0.18 / -0.03; rand60-vp 0.75 / 0.65 / 0.63 / 0.19 / 0.01.

### Hidden-property probes (R^2 on contact windows; mean +- sd over seeds, 1,000 steps)

| | log mass | friction | log softness |
|---|---|---|---|
| s2b-vp / rand60-vp | -0.03 +- 0.08 / -0.08 +- 0.01 | -0.09 +- 0.01 / -0.07 +- 0.05 | -0.09 / -0.09 |
| s2b-all / s2b-all-tw / s2b-all-noeff | -0.06 +- 0.01 / -0.07 +- 0.02 / -0.06 +- 0.03 | -0.08 / -0.06 / -0.04 | -0.08 / -0.10 / -0.08 |
| floors (V+P, all) | -0.10 / -0.10 | +0.03 / -0.04 | -0.12 / -0.11 |
| supervised on these labels (V+P, all), n 3 | -0.08 +- 0.00 / -0.09 +- 0.01 | -0.07 +- 0.01 / -0.11 +- 0.04 | -0.10 +- 0.01 / -0.11 +- 0.01 |

About -0.09 is what a constant prediction scores under leave-one-episode-out (the held-out
episodes' mean differs from the training mean), so **every arm, the floors and the supervised
ceiling are at chance: the hidden-property probe has no range at this scale, and no contrast on
it (effort in or out) can be read.** Giving the ridge context (the state plus the mean of the
previous 4 or 8 windows and their difference, `scripts/hidden_context.py`) does not change that
(-0.04 to -0.11 for the SSL and the hidden-supervised encoders). A mass shows in how an object
*responds* to a push, which a 0.2 s state of a 16,000-sequence tiny model does not carry even when
trained on the label. A better-posed probe needs the response itself (object acceleration against
effort over a contact bout), or a longer encoder. **Parked** (supervisor/lead decision): if even a
supervised ceiling trained on the labels is at chance, the probe measures nothing at this scale, so
the response-based probe is not built now.

### Fine-tune yardstick (16 labelled s2b episodes, probed on `exp`; mean +- sd, 3 seeds)

| init | 100 steps | 200 | 400 | 600 |
|---|---|---|---|---|
| random | 0.40 +- 0.03 | 0.45 +- 0.01 | 0.46 +- 0.01 | 0.47 +- 0.01 |
| rand60-vp SSL | 0.42 +- 0.01 | 0.46 +- 0.01 | 0.47 +- 0.02 | 0.47 +- 0.02 |
| s2b-vp SSL | 0.38 +- 0.00 | 0.42 +- 0.01 | 0.44 +- 0.01 | 0.44 +- 0.02 |

At 600 steps s2b-vp - random = -0.030 +- 0.015, rand60-vp - random = -0.002 +- 0.016, s2b-vp -
rand60-vp = -0.027 +- 0.020 (-0.042 +- 0.013 at 200 steps).

### Reading

* **Hand experiments make the V+P encoder worse than random play of the same size**, not better:
  -0.147 +- 0.029 on the random-play set (above its 0.081 MDD) and -0.120 +- 0.054 on the experiment
  set, both curves falling from 250 steps (the peak-then-fall of B-ctrl-0: 0.22 to 0.13 on `exp`).
  The fine-tune yardstick agrees: the experiment-pretrained init is 0.03 below scratch and below the
  random-play init; the random-play init equals scratch.
* **With all senses the arms are indistinguishable** (s2b-all - rand60-all +0.029 +- 0.043 on `exp`,
  -0.031 +- 0.018 on `rand`), and both end at or below their floors.
* **Touch-change-weighted loss declines less than the unweighted arm; it does not "beat the floor".**
  On the experiment set `s2b-all` and `s2b-all-tw` both start above every floor seed at step 250
  (0.33 and 0.35, difference +0.020 +- 0.025), then `s2b-all` falls to 0.25 by step 1,000 and
  `s2b-all-tw` to 0.33: +0.079 +- 0.026 at 1,000 steps (a little above the 0.072 MDD), +0.040 +- 0.015
  at 500. On the random-play set the arms are +0.043 +- 0.023 apart at 1,000 steps and the weighted arm
  is at its floor (-0.017 +- 0.017). The same sign appeared on `probe_test` in B-conseq-0 (+0.036 +-
  0.022; +0.032 +- 0.020 at n = 6). The step-250 and step-1,000 numbers together say the weighting
  slows the loss of probe-relevant information, not that it adds any. The gain sits in force: log-force
  R^2 0.43 against 0.16 to 0.22 for every other arm (contact object 0.70 against 0.63 to 0.64). It
  does not show on the masked `exp_vp` probe (+0.008 +- 0.027), so the state uses touch more, rather
  than improving vision + proprio.
* **Every arm's probe score peaks at or before step 250 and falls with training** on the
  experiment set (V+P 0.22 to 0.13, all-sense 0.33 to 0.25, no-effort all-sense 0.34 to 0.20): the
  objective removes probe-relevant information as it trains, faster on experiment data. That, not
  the data, is the open question (see the objective sanity check below).
* **Effort as an input is not resolved**: -0.004 +- 0.053 (V+P), -0.047 +- 0.061 on `exp` and +0.034 +-
  0.022 on `rand` (all senses). The hidden-property probe that should have separated it is at chance
  (above).
* **Caveats.** 60 experiment episodes (36,000 env steps) and a 12-episode `exp` set; the supervised
  ceilings are trained on `exp`-distribution episodes only (so on `rand` they are not ceilings);
  one data release; the tiny model and an objective that already sits at its floor in B-conseq-0.

## Touch-change weighting and input + target at n = 6 (B-conseq-0 follow-up)

**Training-set size:** 100 random-play episodes = 60,000 env steps, 1,000 steps x 16 sequences (16,000),
tiny model, 64 px, pooled SIGReg, all six senses, 6 seeds per arm (`scripts/objective_test.py`,
`a-base` and `a-touch` seeds 0-5, `a-both` seeds 0-5). Probe set: the 20 held-out random-play
episodes; floor (all senses, 5 seeds) 0.30 +- 0.02. Figure `.tmp/phd-b/conseq_n6_touch.png`.

`a-base`: no change. `a-touch`: loss weighted by the step-to-step touch change (`model.surprise =
touch`). `a-both`: touch is an encoder input *and* a prediction target (`model.teacher = touch` with
touch in `senses`, the input + target cell of the H1 role grid; the aux loss of B-teach-0 on the
encoder's state, the teacher's own EMA target).

| arm | 250 steps | 500 | 750 | 1,000 |
|---|---|---|---|---|
| a-base | 0.28 +- 0.03 | 0.27 +- 0.04 | 0.28 +- 0.03 | 0.28 +- 0.03 |
| a-touch | 0.30 +- 0.02 | 0.32 +- 0.03 | 0.30 +- 0.03 | 0.31 +- 0.03 |
| a-both | 0.25 +- 0.02 | 0.25 +- 0.03 | 0.26 +- 0.02 | 0.27 +- 0.02 |

| contrast at 1,000 steps | difference +- SE | MDD |
|---|---|---|
| a-touch - a-base | +0.032 +- 0.020 | 0.056 |
| a-both - a-base | -0.007 +- 0.018 | 0.049 |
| a-both - a-touch | -0.039 +- 0.016 | 0.045 |
| a-base - floor | -0.019 +- 0.018 | |
| a-touch - floor | +0.012 +- 0.017 | |
| a-both - floor | -0.026 +- 0.014 | |

Terms at 1,000 steps (contact L / R / object AUROC, force R^2, tip R^2): a-base 0.68 / 0.62 / 0.63 /
0.21 / 0.32; a-touch 0.73 / 0.66 / 0.63 / 0.22 / 0.26; a-both 0.71 / 0.62 / 0.61 / 0.17 / 0.31.

* **Touch-change weighting: the lead shrinks at n = 6** (+0.032 +- 0.020, below the 0.056 MDD, against
  +0.036 at n = 3); the seed sd is 0.033 to 0.037, higher than the 0.02 to 0.03 assumed, so the MDD is
  0.056, not 0.045. The arm is at its floor, not above it (+0.012 +- 0.017). Two data sets side by
  side, step 250 next to step 1,000 (mean over seeds; the weighted arm "declines less", it does not
  rise above the floor):

  | data (probe set) | unweighted, 250 / 1,000 steps | touch-weighted, 250 / 1,000 steps | gap at 1,000 |
  |---|---|---|---|
  | random play, `probe_test` held-out (n 6; floor 0.30) | 0.28 / 0.28 | 0.30 / 0.31 | +0.032 +- 0.020 |
  | S2b hand experiments, `exp` (n 3; floor 0.28) | 0.33 / 0.25 | 0.35 / 0.33 | +0.079 +- 0.026 |
  | S2b hand experiments, random-play set (n 3; floor 0.27) | 0.26 / 0.22 | 0.29 / 0.26 | +0.043 +- 0.023 |

* **Input + target adds nothing here**: `a-both` is -0.007 +- 0.018 from `a-base` and -0.039 +- 0.016
  below `a-touch`. With touch also an input the teacher's latent is trivially predictable from the
  state, which is a reason, not a test.

## Objective sanity check: JEPA against two input-space references (B-obj-0)

Every JEPA arm of B-ctrl-0, B-conseq-0, B-exp-0 and B-s2b-0 sits at or below its random-init floor.
Is that our objective (pooled-SIGReg JEPA) or something common to any self-supervised objective at
this size? `src/sensorium/model/reference.py` adds two standard objectives on **the same encoder,
data, steps, probe battery and floors**; `scripts/objective_test.py` arms `ref-jepa`, `ref-mae`,
`ref-nextobs`. They are new code, not B-ctrl-0's controls (those were a supervised ceiling, frozen
DINOv2, PCA of raw inputs and fine-tunes; none is self-supervised reconstruction).

**Training-set size:** 100 random-play episodes = 60,000 env steps; every run 1,000 steps x 16
sequences (16,000), tiny model (dim 128, 64 px, vision + proprio), batch 16, 3 seeds, the
`probe_test` held-out 20 episodes as probe set. Step 0 is the untrained encoder of the same seed (the
floor, 5 further seeds in `floor V+P`). Probed at 0, 50, 100, 250, 500, 750 and 1,000 steps. Figure
`.tmp/phd-b/obj_sanity_curves.png`.

* `ref-jepa`: the pooled-SIGReg objective used so far.
* `ref-mae`: masked reconstruction. 75% of the frame's patches are grayed out and half of the proprio
  channels zeroed in the encoder's input; a two-layer decoder reads the encoder's state and
  reconstructs the masked patches (normalised pixels) and masked proprio values, loss on masked
  entries only. Deviations from MAE (He et al. 2022): pixel masking instead of dropped tokens, and a
  decoder that sees one state vector, because the probes read that vector.
* `ref-nextobs`: the action-conditioned predictor maps the state of step t to the next state and the
  decoder turns that into the whole next frame and next proprio (MSE in input space). No
  regulariser: the targets are inputs, so nothing can collapse.

| arm | 0 steps | 50 | 100 | 250 | 500 | 750 | 1,000 |
|---|---|---|---|---|---|---|---|
| ref-jepa | 0.23 +- 0.02 | 0.24 +- 0.03 | 0.23 +- 0.02 | 0.25 +- 0.00 | 0.26 +- 0.03 | 0.26 +- 0.03 | 0.26 +- 0.03 |
| ref-mae | 0.23 +- 0.02 | 0.14 +- 0.02 | 0.23 +- 0.03 | 0.20 +- 0.03 | 0.18 +- 0.01 | 0.19 +- 0.02 | 0.20 +- 0.02 |
| ref-nextobs | 0.23 +- 0.02 | 0.17 +- 0.03 | 0.14 +- 0.02 | 0.09 +- 0.02 | 0.13 +- 0.04 | 0.11 +- 0.05 | 0.09 +- 0.03 |
| floor V+P (random init, 5 seeds); supervised ceiling | 0.22 +- 0.02; 0.64 | | | | | | |

Change from the encoder's own step 0 (paired by seed, mean +- SE of 3): ref-jepa **+0.029 +- 0.007**,
ref-mae -0.033 +- 0.011, ref-nextobs -0.136 +- 0.013. Against each other at 1,000 steps: mae - jepa
-0.061 +- 0.025 (MDD 0.069), nextobs - jepa -0.165 +- 0.026 (0.074), nextobs - mae -0.104 +- 0.026 (0.072).
Terms at 1,000 steps (contact L / R / object AUROC, force R^2, tip R^2): jepa 0.67 / 0.62 / 0.61 /
0.20 / 0.28; mae 0.67 / 0.55 / 0.56 / 0.16 / 0.25; nextobs 0.65 / 0.48 / 0.50 / -0.11 / 0.32.

Geometry of the held-out states (`diagnostics.state_geometry`: effective rank = exp of the entropy of
the normalised singular values, 128 at most; mean over seeds), and per-dimension std:

| arm | rank 0 | 50 | 100 | 250 | 500 | 750 | 1,000 | std 0 / 1,000 |
|---|---|---|---|---|---|---|---|---|
| ref-jepa | 55.9 | 36.7 | 35.7 | 37.1 | 39.6 | 40.1 | 40.7 | 0.1 / 1.0 |
| ref-mae | 55.9 | 54.4 | 49.9 | 49.9 | 49.6 | 51.0 | 51.4 | 0.1 / 0.7 |
| ref-nextobs | 55.9 | 42.1 | 48.3 | 49.1 | 53.6 | 57.6 | 58.2 | 0.1 / 0.4 |

### Reading

* **Neither reference objective beats the floor; the JEPA objective is the best of the three.** MAE
  ends 0.03 below its own step 0 and next-observation prediction 0.14 below it (contact-right and
  contact-object AUROC fall to chance, force R^2 to -0.11), while JEPA ends 0.03 above it (4 SE
  within seeds, +0.035 +- 0.020 against the 5-seed floor). So the problem is not specific to the
  JEPA objective: at this size an encoder trained to reconstruct pixels carries *less* contact
  information than a random one, as Balestriero and LeCun (2024) argue reconstruction does
  (high-variance image structure is not the perception-relevant one).
* **It is not rank collapse.** JEPA's effective rank drops from 56 to 36 in the first 50 steps and
  recovers to 41 while its probe rises; next-observation prediction's rank rises to 58 while its
  probe falls to 0.09. Rank does not track the probe; what the objective *encodes* does.
* **So the setting is the open question**: scale (16,000 sequences of a 2.4M-parameter encoder; the
  reference objectives' held-out pixel loss falls only from 0.82 to 0.50 to 0.53, they are far from
  converged), the probe or the data. This check does not separate those three; it says the JEPA
  objective is not the culprit by itself, and that at this scale neither a latent nor an input-space
  objective yields a contact-aware encoder.
* **Closest published comparisons** (large scale, linear or attentive probes, so they say which
  objective wins at scale, not what happens at 16,000 sequences). I-JEPA (Assran et al. 2023,
  arXiv 2301.08243, Table 1; ImageNet-1K linear evaluation, top-1): MAE ViT-B/16 68.0, ViT-L/16 76.0,
  ViT-H/14 77.2 against I-JEPA ViT-B/16 72.9, ViT-L/16 77.5, ViT-H/14 79.3. V-JEPA (Bardes et al. 2024,
  arXiv 2404.08471, the target ablation, ViT-L/16, frozen attentive probe): pixel targets 68.6 on
  Kinetics-400, 66.0 on Something-Something-v2, 73.3 on ImageNet-1K against feature targets 73.7,
  66.2, 74.8, and about equal under fine-tuning (85.4 against 85.6 on K400). Our ordering (latent
  prediction above reconstruction) agrees with theirs; the difference is that here neither is above
  a random encoder.

### Geometry retrofitted onto the B-s2b-0 checkpoints (`scripts/collapse_retrofit.py`)

Effective rank of the held-out `exp` states (128 at most; mean over 3 seeds; step 0 = the random-init
floor of that arm's senses), next to the probe score of the same checkpoints (`exp`, section above):

| arm | rank 0 | 250 | 500 | 750 | 1,000 | probe 250 -> 1,000 |
|---|---|---|---|---|---|---|
| s2b-vp | 54.2 | 33.7 | 34.3 | 35.3 | 35.7 | 0.22 -> 0.13 |
| rand60-vp | 54.2 | 37.1 | 37.9 | 39.0 | 39.2 | 0.28 -> 0.25 |
| s2b-vp-noeff | 53.3 | 31.7 | 33.2 | 34.0 | 34.6 | 0.21 -> 0.12 |
| s2b-all | 54.9 | 37.2 | 40.2 | 41.0 | 41.3 | 0.33 -> 0.25 |
| rand60-all | 54.9 | 37.7 | 38.4 | 39.0 | 39.3 | 0.21 -> 0.22 |
| s2b-all-tw | 54.9 | 34.3 | 38.2 | 39.9 | 41.3 | 0.35 -> 0.33 |
| s2b-all-noeff | 52.4 | 35.4 | 38.5 | 38.7 | 39.1 | 0.34 -> 0.20 |

The rank drops by a third within the first 250 steps and then holds or rises slowly (per-dimension
std 0.1 -> 0.7 to 1.0), while the probe of the experiment arms keeps falling at constant rank
(`s2b-vp`: rank 34 -> 36, probe 0.22 -> 0.13). **Rank loss is not what costs the probe**; with the
reference objectives above (next-observation prediction at rank 58 and probe 0.09) it is what the
objective encodes, and the SIGReg setup is not the suspect by this measure.

## Input-only baselines and the steps curve (B-steps-0)

The sanity check leaves budget (1,000 x 16 steps) and the probe open. Two things here: what a linear
readout of the *inputs* already gives, and what 10 times more steps does to JEPA and to masked
reconstruction.

### Input-only baselines (`scripts/input_baselines.py`)

The battery's ridge probe on raw proprio, no encoder, same 20 held-out random-play episodes and
windows (stride 8) as every encoder. Floor (random-init V+P encoder) 0.22 +- 0.02; the best trained V+P
encoder so far 0.26 +- 0.03 (`ref-jepa`, 1,000 steps).

| raw input (no encoder) | channels | probe mean | contact L | contact R | contact obj | force R^2 | tip R^2 |
|---|---|---|---|---|---|---|---|
| proprio body, last obs | 42 | 0.05 | 0.74 | 0.57 | 0.53 | 0.16 | -0.56 |
| vestibular, last obs | 10 | 0.24 | 0.70 | 0.65 | 0.58 | 0.28 | 0.08 |
| proprio effector (hand / pincer joints), last obs | 128 | 0.00 | 0.62 | 0.48 | 0.43 | -0.02 | -0.01 |
| body + vestibular | 52 | 0.13 | 0.74 | 0.62 | 0.62 | 0.17 | -0.50 |
| all proprio, last obs | 180 | 0.12 | 0.75 | 0.63 | 0.58 | 0.17 | -0.52 |
| **all proprio, the K = 4 observations stacked** | 720 | **0.28** | 0.75 | 0.65 | 0.60 | 0.29 | 0.15 |

From B-ctrl-0 on the same episodes: raw vision thumbnail (16 px) + proprio, PCA 128: 0.14; raw
non-visual inputs (touch, audio, proprio), PCA 128: 0.26 (these include the contact signal itself).

**A linear readout of the window's own proprio (0.28) is at or above every trained V+P encoder (0.26
at best) and above the random-init floor (0.22).** Vestibular alone (10 channels) gives 0.24.
The tip-position term is what makes the last-observation rows poor (R^2 -0.5: a linear map from joint
angles to tip position extrapolates badly to unseen episodes); stacking the four observations
(velocities) fixes it. So the floor is largely what proprio gives linearly, and an encoder that
only matches the floor has learned nothing beyond a rotation of its proprio input.

### Random-init baselines for linear probes: a known effect

Random (untrained) convolutional features are a strong linear-probe reference in small-data vision, and
trained networks often only modestly improve on them:

* Jarrett et al. 2009, as quoted by Saxe et al. (ICML 2011): a one-layer convolution-pooling
  network with completely random filters reaches 53% on Caltech101 with a linear classifier, against
  54.2% after unsupervised pretraining and fine-tuning of the filters. Saxe et al. themselves: the best
  random-weight architecture on CIFAR-10 reaches 53.2 +- 0.3% against 59.5 +- 0.3% for the best trained
  one, and "many trained networks lose out ... to random-weight networks with different architectural
  parameters".
* Asano, Rupprecht and Vedaldi (ICLR 2020, Tables 1 and 3): linear classifiers on CIFAR-10 on the
  conv1 to conv4 features of a 4-block network, random 57.8 / 55.5 / 54.2 / 47.3 against fully
  supervised 66.5 / 70.1 / 72.4 / 75.9 and RotNet 64.4 / 65.6 / 65.6 / 59.1: a random first layer
  is 9 points below supervised, and random features lose with depth.

Both are image classification at a scale (tens of thousands of images, labels) unlike ours, and
neither is a controls-style multi-term probe, but they say the same thing: the random-init floor is
not a weak baseline, and at small scale training must beat architecture before it adds anything.

### Steps curve: protocol and decision rule (written before the runs finish)

Arms `long-jepa` (pooled SIGReg) and `long-mae` (masked reconstruction), V+P, the 100 random-play
episodes, 10,000 steps x 16 sequences (160,000 sequences, ten times the earlier runs; each sequence
covers 32 env steps, so every env step of the 60,000 is seen in about 85 sequences, 85 passes over the
data, against 8.5 at 1,000 steps), 3 seeds, probed at 0, 250, 1,000, 2,500, 5,000 and 10,000 steps with
the geometry of the held-out states and the held-out loss next to the training loss (a held-out
loss that turns up while the training loss falls is overfitting, not learning).

**Decision rule.** The change of the probe from the same seed's step 0 is computed per seed; its
MDD is 2.8 times the standard error of that change over the 3 seeds at the same step.

* If `long-jepa` at 10,000 steps beats its own step 0 by more than that MDD, the problem was budget,
  and the next question is how far it climbs (the slope between 2,500, 5,000 and 10,000).
* If not, steps are not the lever, and model size and data scale go to the supervisor as a
  compute decision.
* An input-only linear readout (0.28 above) is the second bar: an encoder that does not exceed it adds
  nothing to what proprio gives linearly.

### Steps curve: results (GCP, n2-highcpu-32, 3 seeds; `scripts/objective_test.py longreport`)

The runs were made on rented CPU (heavy training no longer runs on the Mac). Figure
`.tmp/phd-b/steps_curves.png`. Training-set size as above: 100 random-play episodes, 10,000 steps x 16
sequences (160,000 sequences, about 85 passes over the 60,000 env steps).

| arm | 0 steps | 250 | 1,000 | 2,500 | 5,000 | 10,000 |
|---|---|---|---|---|---|---|
| long-jepa | 0.23 +- 0.02 | 0.25 +- 0.00 | 0.24 +- 0.03 | 0.18 +- 0.01 | 0.15 +- 0.04 | 0.13 +- 0.03 |
| long-mae | 0.23 +- 0.02 | 0.20 +- 0.03 | 0.21 +- 0.03 | 0.22 +- 0.02 | 0.21 +- 0.01 | 0.22 +- 0.02 |
| floor V+P (5 seeds) | 0.22 +- 0.02 | | | | | |

Change from the same seed's step 0 (mean +- SE of 3; MDD = 2.8 SE): long-jepa +0.023 +- 0.010 at 250,
+0.007 +- 0.013 at 1,000, -0.049 +- 0.002 at 2,500, -0.084 +- 0.015 at 5,000 and **-0.104 +- 0.012 at
10,000 (MDD 0.032)**; long-mae -0.012 +- 0.016 at 10,000 (MDD 0.044).

| | 0 steps | 250 | 1,000 | 2,500 | 5,000 | 10,000 |
|---|---|---|---|---|---|---|
| long-jepa held-out / train prediction loss | 0.074 | 0.359 / 0.355 | 0.193 / 0.181 | 0.125 / 0.112 | 0.079 / 0.087 | 0.053 / 0.059 |
| long-mae held-out / train loss | 0.820 | 0.548 / 0.586 | 0.484 / 0.353 | 0.464 / 0.518 | 0.428 / 0.356 | 0.406 / 0.357 |
| long-jepa effective rank (of 128) | 55.9 | 37.0 | 40.6 | 40.9 | 41.3 | 41.2 |
| long-mae effective rank | 55.9 | 49.7 | 53.5 | 57.9 | 61.7 | 64.3 |

Terms of the probe, step 0 / 1,000 / 10,000 (contact L, R, object AUROC; force R^2; tip R^2): long-jepa
0.71, 0.59, 0.56, 0.14, 0.27 / 0.65, 0.62, 0.60, 0.18, 0.26 / 0.55, 0.55, 0.58, 0.09, 0.20; long-mae
0.71, 0.59, 0.56, 0.14, 0.27 / 0.69, 0.55, 0.60, 0.13, 0.24 / 0.67, 0.62, 0.61, 0.14, 0.14.

**Reading.** JEPA gets better at its own task (held-out prediction loss 0.36 to 0.05, tracking the
training loss: no overfitting, the held-out loss never turns up) while the contact information in its
state drains away (probe 0.25 at 250 steps to 0.13 at 10,000, left-hand contact AUROC 0.71 to 0.55,
force R^2 0.14 to 0.09), at constant effective rank (about 41): it is not collapse. Masked
reconstruction stays at about 0.22 with a still-falling loss and a rank that rises to 64. **By the rule
above, more steps are not the lever** (long-jepa at 10,000 steps is 0.104 *below* its own step 0, three
times the MDD, in the wrong direction), and the question becomes what the JEPA objective is allowed to
discard: nothing in it asks the state to keep touch-relevant information, which is the case for a
teacher (next section).

## B-teach-long: does a touch teacher stop the decay of contact information? (pre-registered)

**Written before any teacher run exists.** B-teach-0 found the touch teacher unresolved at 2,000
steps (target-only teacher minus placebo +0.01 +- 0.02, MDD 0.083, seeds 6). The steps curve above gives a much
sharper version of the same question: at 10,000 steps the plain JEPA (`long-jepa`, the c-none arm) has
lost 0.104 of probe score, so a teacher that held the state near its step-0 level would open a gap of
about 0.10, three times the MDD. `scripts/objective_test.py`, arms (same schedule, data, probe,
checkpoints as `long-jepa`, V+P, 3 seeds, 10,000 steps x 16, probed at 0 / 250 / 1,000 / 2,500 /
5,000 / 10,000):

* `long-teach`: `senses = V+P`, `teacher = touch`. Touch is never an input of the state encoder; an
  EMA touch encoder gives latent targets that two heads on the state and on the predictor output must
  predict (B-teach-0, after Touch Dreaming, arXiv 2604.13015).
* `long-placebo`: the same with the targets taken from another sequence of the batch (statistics of
  touch latents, no information about this sequence): the extra-heads control.
* `long-jepa` is the no-teacher arm and is **reused, not retrained**; same seeds, hence the same
  initial encoder as the teacher arms (a unit test checks it), so contrasts are paired by seed.

Teacher use is reported at every checkpoint as the held-out R^2 of the heads against the true EMA touch
latents of the same sequence (`touch_r2_now`, `touch_r2_next`; B-teach-0: 0.43 and 0.46 for the real
teacher, 0.11 for the placebo). If the real teacher's heads do not clearly beat the placebo's, a null
below says "the heads did not learn", not "a touch teacher is useless".

**Primary contrasts, at 10,000 steps** (also reported at 2,500 and 5,000): `long-teach - long-jepa` and
`long-teach - long-placebo`, mean over the 3 seeds of the same-seed differences of the probe mean
(V+P state, the 20 held-out episodes, as everywhere). A contrast is **resolved** if its absolute value
exceeds the threshold of 0.032 (2.8 x the 0.012 standard error of the long-jepa change from step 0 at
10,000 steps) *and* 2.8 times its own paired standard error. Outcomes, decided by
`objective_test.teacher_verdict` and stated here in advance:

| outcome | rule |
|---|---|
| **teacher prevents drift** | `teach - jepa` and `teach - placebo` both resolved and positive |
| **extra heads only** | `teach - jepa` and `placebo - jepa` resolved and positive, `teach - placebo` not resolved |
| **teacher harms** | `teach - jepa` resolved and negative |
| **null** | anything else (`teach - jepa` not resolved) |

Descriptive, not part of the rule: the change of each arm from its own step 0, held-out loss next to
training loss, effective rank. A positive outcome at 10,000 steps with the teacher heads at R^2 below
the placebo's would not be read as "prevents drift".

**Literature.** Lee et al. (ICRA 2019, "Making Sense of Vision and Touch", arXiv 1810.10191) used
contact prediction as a self-supervised auxiliary: a network fusing RGB, force-torque and proprio
*inputs* is trained to predict action-conditional optical flow, whether contact will occur in the
next control cycle (labels from simple heuristics on the force-torque signal) and the alignment of
vision and touch, and its representation feeds an RL policy for peg insertion (simulation and real
robot), improving sample efficiency. Known there: a contact-related auxiliary improves the
representation for a contact-rich task when touch is also an input at test time. New here: (i)
touch as a *latent* target of an EMA touch encoder inside a JEPA, never an input of the state we
probe, so the question is whether it changes what a vision + proprio state keeps; (ii) the
measurement is representation drift over 10,000 steps against a placebo-target control, not task
performance; (iii) the failure being tested (contact information lost while the self-supervised loss
falls) is the one the steps curve exposes, which Lee et al. do not look at.

### B-teach-long: results (GCP, 3 seeds; `scripts/objective_test.py longreport`)

**Pre-registered verdict at 10,000 steps: null.** `long-teach - long-jepa` is +0.082 +- 0.036 (n 3):
above the 0.032 threshold but not above 2.8 times its own paired standard error (0.101), so not
resolved; `long-teach - long-placebo` is +0.012 +- 0.007, under the threshold. Figure
`.tmp/phd-b/teach_long_curves.png`.

**Training-set size:** 100 random-play episodes (60,000 env steps), 10,000 steps x 16 sequences
(160,000 sequences, about 85 passes), V+P, tiny model, 3 seeds per arm, the 20 held-out random-play
episodes as probe set; floor (random-init V+P, 5 seeds) 0.22 +- 0.02. `long-jepa` is the runs of
B-steps-0 (same seeds, same initial encoder).

Probe mean (mean +- sd over seeds):

| arm | 0 steps | 250 | 1,000 | 2,500 | 5,000 | 10,000 |
|---|---|---|---|---|---|---|
| long-jepa | 0.23 +- 0.02 | 0.25 +- 0.00 | 0.24 +- 0.03 | 0.18 +- 0.01 | 0.15 +- 0.04 | 0.13 +- 0.03 |
| long-teach | 0.23 +- 0.02 | 0.26 +- 0.02 | 0.28 +- 0.02 | 0.23 +- 0.03 | 0.23 +- 0.01 | 0.21 +- 0.02 |
| long-placebo | 0.23 +- 0.02 | 0.26 +- 0.02 | 0.27 +- 0.02 | 0.22 +- 0.01 | 0.21 +- 0.01 | 0.20 +- 0.01 |

Change from the same seed's step 0 (mean +- SE of 3; MDD = 2.8 SE):

| arm | 250 | 1,000 | 2,500 | 5,000 | 10,000 |
|---|---|---|---|---|---|
| long-jepa | +0.023 +- 0.010 | +0.007 +- 0.013 | -0.049 +- 0.002 | -0.084 +- 0.015 | -0.104 +- 0.012 (MDD 0.032) |
| long-teach | +0.027 +- 0.001 | +0.050 +- 0.024 | +0.005 +- 0.035 | -0.004 +- 0.016 | -0.022 +- 0.024 (MDD 0.068) |
| long-placebo | +0.026 +- 0.009 | +0.039 +- 0.023 | -0.006 +- 0.020 | -0.021 +- 0.018 | -0.034 +- 0.018 (MDD 0.050) |

Pre-registered paired contrasts (same seed, same initial encoder; threshold 0.032):

| contrast | 2,500 | 5,000 | 10,000 |
|---|---|---|---|
| long-teach - long-jepa | +0.053 +- 0.033, not resolved | +0.080 +- 0.026, resolved | +0.082 +- 0.036, not resolved |
| long-teach - long-placebo | +0.011 +- 0.017, not resolved | +0.017 +- 0.002, not resolved | +0.012 +- 0.007, not resolved |
| long-placebo - long-jepa | +0.042 +- 0.018, not resolved | +0.063 +- 0.028, not resolved | +0.071 +- 0.029, not resolved |

Teacher use: held-out R^2 of the heads against the true EMA touch latents (now / next):

| arm | 0 steps | 250 | 1,000 | 2,500 | 5,000 | 10,000 |
|---|---|---|---|---|---|---|
| long-teach | -0.43 / -0.94 | 0.39 / 0.40 | 0.38 / 0.41 | 0.39 / 0.41 | 0.35 / 0.41 | 0.34 / 0.40 |
| long-placebo | -0.43 / -0.94 | 0.17 / 0.26 | 0.06 / 0.09 | 0.13 / 0.15 | 0.09 / 0.10 | 0.07 / 0.08 |

Held-out / training prediction loss and effective rank of the held-out states, 10,000 steps: jepa
0.053 / 0.059, rank 41.2; teach 0.061 / 0.064, rank 41.6; placebo 0.055 / 0.060, rank 41.1 (at
1,000 steps 0.193 / 0.181, 0.212 / 0.208, 0.208 / 0.206; no arm's held-out loss turns up).

**Reading (not pre-registered; descriptive).**

* Both auxiliary-head arms decay far less than plain JEPA: at 10,000 steps teach 0.21, placebo 0.20,
  jepa 0.13. The change from step 0 is -0.022 +- 0.024 (teach) and -0.034 +- 0.018 (placebo) against
  -0.104 +- 0.012. `placebo - jepa` is +0.071 +- 0.029, not resolved by the rule (2.8 own standard
  errors = 0.081).
* Real touch over the placebo is small and has the same sign at every late step (+0.011, +0.017,
  +0.012), all under 0.032.
* The heads do learn touch: R^2 0.34 / 0.40 at 10,000 steps against 0.07 / 0.08 for the placebo's
  heads on the real latents (placebo 0.17 / 0.26 at step 250 and falling). So, as in B-teach-0,
  touch-predictive information in the state does not become linearly decodable contact labels: a
  state that predicts touch latents with R^2 0.34 has the same contact probe as one whose heads
  predict another sequence's touch, within 0.012.
* Per probe term at 10,000 steps (contact L / R / object AUROC, force R^2, tip R^2): jepa 0.55 / 0.55 /
  0.58 / 0.09 / 0.20; teach 0.70 / 0.59 / 0.57 / 0.05 / 0.28; placebo 0.67 / 0.58 / 0.57 / 0.05 / 0.28.
  The teacher arms keep the left-contact and tip terms and lose force more than jepa does (R^2 0.14 at
  step 0).
* No arm beats the raw-proprio K = 4 linear readout (0.28, input-only baselines above) at 10,000
  steps; `long-teach` reaches 0.28 around 1,000 steps (equal to that baseline, not above it) and then
  decays, as does the placebo (0.27).

**Caveats.** n = 3 seeds. The rule's 2.8-times-own-standard-error term is strict at n = 3. At the
observed per-seed spread of the paired differences (0.062 for `teach - jepa`, 0.050 for `placebo -
jepa`), a +0.082 and a +0.071 difference would be resolved by that term at about 5 seeds (2.8 SE =
0.078) and at 4 to 5 seeds (0.070 at n = 4, 0.063 at n = 5) respectively; `teach - placebo` (+0.012,
spread 0.012) stays under the 0.032 threshold at any n. The 2.8 is the normal-approximation MDD
factor used throughout this document; with 3 seeds a t critical value would be larger. One data set
(random play), the tiny model, the 20-episode probe set.

## B-scale-0: does more pretraining data help the V+P JEPA? (pre-registered)

**Written before the data exist.** Every V+P encoder so far was pretrained on the same 100
random-play episodes. B-steps-0 showed the JEPA probe peaking near 1,000 steps (0.24 to 0.25) and
draining to 0.13 by 10,000, with no overfitting in the held-out loss. One reading is that with 100
episodes the run makes about 85 passes over the data and the objective learns to discard what it
does not need. A tenfold larger set at the same step budget tests that reading directly. Code:
`scripts/scale_test.py` (`split`, `train`, `probe`, `report`); paired statistics in
`sensorium.model.paired`.

**Data.**

* **Source:** `configs/scale_test.toml`. 1,110 new seated babbling episodes of 600 steps (master
  seed 2, so disjoint from `probe_test` and never mixed with it), in 30 blocks of 37:
  `hand_seated_00..14` and `pincer_seated_00..14`.
* **Split** (`scale_test.make_split`, drawn once with seed 0, written to `runs/scale/split.json`):
  1. Episodes are shuffled within each block.
  2. Held-out set: 100 episodes, 3 or 4 per block, 50 hand and 50 pincer.
  3. Pretraining order: the rest, in rounds that take one episode of every block in a random order
     alternating hand and pincer blocks. The subsets are the first 100, 300 and 1,000 episodes of
     that order, so they are **nested** (100 within 300 within 1,000) and stratified by block (3–4,
     10 and 33–34 per block, hand and pincer within one of half).
  4. The remaining 10 episodes are spare.

**Arms.**

* **Common recipe:** V+P, pooled SIGReg JEPA as `long-jepa`, tiny model at 64 px, 10,000 steps x 16
  sequences, checkpoints at 0, 250, 1,000, 2,500, 5,000 and 10,000, seeds 0 to 2.
* **Shared normalisation:** every arm fits its channel space and normalisation on the 100-episode
  subset (`data.space_paths`). So **the same seed gives the identical initial encoder at every data
  size and for the floor**. The smoke test confirms it: floor seed 0, `jepa-e100` step 0 and
  `mae-e1000` step 0 score the same.
* **Arms:** `jepa-e100`, `jepa-e300`, `jepa-e1000`, and `mae-e1000` (masked reconstruction on the
  1,000 episodes). With the frame cache below, all 12 runs fit on one machine at once.
* **The variable:** the number of **steps** is held equal across sizes. What changes is the number of
  passes: about 85 at 100 episodes, 28 at 300 and 8.5 at 1,000 (each sequence covers 32 env steps).
  A gain from 100 to 1,000 episodes therefore means "more distinct data at a fixed compute budget",
  and it cannot be separated from "fewer repeats of each episode". A null means more data at this
  budget and model size does not change what the state keeps.

**Bars**, all on the same 100 held-out episodes:

* the random-init V+P floor (5 seeds);
* the ridge probe on raw proprio with the K = 4 observations stacked;
* the supervised V+P ceiling on the 1,000 episodes (`sup-e1000`, 1,000 steps, 3 seeds).

**Evaluation.** The battery's probe: leave-one-episode-out ridge on windows at stride 8, here over
the 100 held-out episodes. Every encoder and bar is saved as out-of-fold predictions, with held-out
loss and effective rank per checkpoint.

**Primary contrast:** `jepa-e1000 - jepa-e100`, reported at two checkpoints:

* **The best step:** the checkpoint after step 0 where the average of the two arms' seed-mean probe
  is highest. It is one common step, so the choice favours neither arm.
* **10,000 steps.**

Pairing is by seed (same initial encoder) and by held-out episode. Each contrast reports:

* the seed SE (paired differences over 3 seeds);
* the episode-bootstrap SE (300 resamples of the 100 held-out episodes, `rng(0)`, the same resample
  for every encoder);
* the combined SE (the two in quadrature) and **MDD = 2.8 x combined SE**.

With 100 rather than 20 held-out episodes the episode SE should shrink by about sqrt 5 = 2.2.

A contrast is **resolved** when |difference| > MDD. Outcomes, decided by `scale_test.verdict` and
stated here in advance:

| outcome | rule |
|---|---|
| **data scale helps** | resolved and positive at both the best step and 10,000 |
| **helps only early** | resolved and positive at the best step, not resolved at 10,000 |
| **helps only late** | resolved and positive at 10,000 (more data slows the drift), not resolved at the best step |
| **more data harms** | resolved and negative at either checkpoint, positive at neither |
| **mixed** | resolved with opposite signs at the two checkpoints |
| **no effect** | resolved at neither (the MDDs state what could have been seen) |

**Reported with it, not part of the rule.**

* The secondary contrasts `e300 - e100`, `e1000 - e300` and `mae-e1000 - jepa-e1000`, at the same two
  checkpoints.
* The change of every arm from the same seed's step 0, at every checkpoint.
* Every arm, at the best step and at 10,000, minus the floor and minus the raw-proprio readout. These
  families are independent by seed, so their seed variances add, and the comparison is paired by
  episode.
* Held-out loss next to training loss, and effective rank.
* Whether a positive outcome also clears the floor, which decides how it reads:
  * *helps* and above the floor means data is a lever;
  * *helps* but still at or below the floor means the drift slows and the encoder still has not
    learned anything useful.
* Expectation, written down so it can be wrong: **helps only late** or **no effect**. Fewer passes
  should slow the drain from 0.25 to 0.13, but the early peak was never above the floor by more than
  its MDD.

**Literature** (data scaling of self-supervised encoders at small scale).

* Goyal et al. (ICCV 2019, "Scaling and benchmarking self-supervised visual representation
  learning", arXiv 1905.01235) scaled Jigsaw and Colorization pretraining from 1M to 100M images.
  * Transfer improves with data size, but Jigsaw "saturates (log-linearly)".
  * "Low capacity models like AlexNet do not show much improvement with more data", while
    ResNet-50 does.
  * Data helps only when the model has the capacity to use it.
* Cole et al. (CVPR 2022, "When does contrastive visual representation learning work?", arXiv
  2105.05837): on ImageNet-scale SimCLR, "the benefit of additional pretraining data beyond 500k
  images is modest".
* Our tiny model (2.4M parameters) and 60,000 to 600,000 env steps sit far below both. Capacity, not
  data, may bind: a null here would be a null for this model size, not for the thesis. That leaves
  model size as the next lever to put to the supervisor.
* I found no JEPA or world-model study that varies pretraining episodes at fixed steps at this scale.
  The B-steps-0 drift (better self-supervised loss, less contact information) has, as far as I
  found, no published counterpart to compare with.

**Loader cost and the frame cache.** I measured the training loop on the Mac: tiny V+P JEPA, batch
16, `OMP_NUM_THREADS=1`, 2 loader workers, steady state over 200 steps, a 1,000-entry pool (the
120 `probe_test` episodes repeated).

| loader | wall s / step | CPU s / step (all processes) | 10,000 steps, CPU |
|---|---|---|---|
| decode every episode (`cache_episodes` 4: almost every pick is a miss at 100 or 1,000 episodes) | 0.25 | **1.54** | 4.3 h |
| every episode already in memory (4-episode pool: a lower bound) | 0.089 | 0.12 | 20 min |
| **frame cache, frames at 128 px, memory-mapped** | **0.085** | **0.11** | **18 min** |

Decoding the stored 256 px mp4 (two episodes per step at `reuse` 8) is **93% of a step's CPU**, and
the per-worker LRU cache cannot hold a pool of 100 episodes, let alone 1,000. So the cost is the
same at every data size.

`sensorium.model.frame_cache` (opt-in: `[augment] frame_cache = <dir>`, default off) writes each
episode's `model_inputs` once as `.npy` files, which the loader memory-maps. The operating system's
page cache is then shared by every process, and a sequence reads only its frames.

* **What it stores:** only model inputs (no privileged state, no text; tested).
* **Full size:** with frames stored at full size, **batches are identical to decoding** (tested:
  every array, with random crops and touch noise on).
* **Downscaled:** frames can be stored downscaled. The random crop is still drawn in the 256 px
  grid with the same random calls, and taken from the downscale at the matching fractional box
  (`vision.train_view(source_side=...)`). So the random stream, the sequence starts and every
  non-vision array are identical, and vision differs only by resampling. On real episodes at 128 px
  the mean absolute difference is **0.69 grey levels out of 255** (0.92 at 96 px).
* **Cost and size:** building it costs about 1 CPU-s per episode. At 128 px it takes 32 MB per
  episode, 32 GB for the 1,000, which fits in the page cache of a 128 GB machine. At full size it
  would take 118 GB.
* **B-scale-0 uses the 128 px cache for every JEPA and MAE arm**, so contrasts between arms are
  unaffected. Absolute levels are not strictly comparable with B-steps-0 (decoded frames) to the
  last grey level. Held-out probes and held-out loss always decode at full resolution.
* **The supervised ceiling decodes**: its loader reads labels from the full episode.

**Compute plan** for an n2-standard-32 (32 vCPU, 128 GB). These are Mac core timings; a vCPU is a
hyperthread, so I allow 1.5 to 2 times longer.

| stage | processes x threads | VM wall |
|---|---|---|
| copy data, `split`, `cache` (1,000 episodes x about 1 CPU-s, 30 processes) | 30 x 1 | about 5 min + copy |
| 12 SSL runs at once (9 JEPA + 3 MAE), cached, 1 loader worker each | 12 x 2 threads (about 26 vCPU) | **about 30 to 50 min** |
| 3 supervised ceilings, decoding, 2 workers each | 3 x 1 thread | about 20 to 30 min |
| probes, 81 jobs at about 2 min each | 16 shards x 2 threads | about 15 min |
| **total** | | **about 1.5 to 2 h** |

The supervised ceilings run beside the probes. Memory: 32 GB of page cache plus 15 processes at
about 1.5 GB each is about 55 GB.

At about $1.55 per hour on demand for n2-standard-32 in us-central1 (third-party price listings, not
GCP's own page; Spot about $0.51 to $0.61), that is **about $2.5 to $3 on demand, MAE included**,
against about $8 to $9 and 6.5 to 7 h with decoding. Without the cache the runs would be
decode-bound at about 1.5 CPU-s per step: 12 runs x 4.3 CPU-hours, about 52 CPU-hours.

## GPU smoke test (prepared, not run)

`sensorium.model.gpu_smoke` measures, on one GPU, in three phases so a slow run says why:
(1) loader throughput for each `--workers` setting, cold (every pick decodes an episode) and
warm; (2) forward + backward + AdamW on one resident batch, fp32 and bf16, with seconds per
step, peak memory and GPU utilisation (`nvidia-smi` sampled each second by a thread); (3) the
real loop for `--minutes` at batch 128, bf16, logging loss, std, effective rank, sequences/s,
utilisation and memory. It writes `report.json` with the H100-hours per run the slowest
measured stage implies for 20 and 100 epochs of a 5M-step pool.

```sh
uvx modal run scripts/modal_gpu_smoke.py                   # Modal, H100 (SMOKE_GPU=A100-80GB for an A100)
scripts/gpu_smoke_ssh.sh root@<host> -p <port>             # any rented box with SSH (Vast.ai ...)
uv run python -m sensorium.model.gpu_smoke configs/model/cpu_tiny.toml --device cpu --minutes 0.2 \
    --batch-size 4 --workers 0 2 --paths 'data/sample_h1/humanoid/*.npz'     # CPU rehearsal
```

Only the CPU rehearsal has run. Untested until a GPU run: CUDA, bf16 on CUDA, `pin_memory`,
`nvidia-smi` sampling (its parser is unit-tested on a sample line), peak memory at batch 128
(not estimated; if it runs out, `--batch-size 64`), the Modal image and the Modal API calls (read
from the Modal docs, not executed), and the torch wheel's CUDA build on Modal's hosts (driver CUDA
13.0 per the Modal docs; torch 2.14.1 from PyPI bundles CUDA 13 libraries in the lock).

**Cost of one smoke run** (about 25 minutes: 3 for the loader and compute phases, 15 for
the timed run, the rest start-up and image build; 8 cores, 32 GiB):

| Where | Price used | Estimate |
|---|---|---|
| Modal, H100 (`H100!`, no H200 upgrade) | $0.001097/s, CPU $0.0000131/core-s, memory $0.00000222/GiB-s (modal.com/pricing, read 2026-10-14) | GPU $1.65 + CPU $0.16 + memory $0.11 = **about $1.9**; covered by the Starter plan's $30 monthly credit (a card on file is needed for GPUs) |
| Modal, A100 80GB | $0.000694/s | **about $1.3** |
| Vast.ai A100 80GB spot | $0.47 to $0.67/h (Vast.ai marketplace, 2026-09/10, not re-checked) | 25 min run + about 15 min set-up and sample upload = 40 min: **$0.31 to $0.45**; marketplace hosts vary in reliability and bandwidth |
| Vast.ai H100 spot | $1.33 to $2.00/h (Vast.ai marketplace) | **$0.89 to $1.33** |

An A100 run measures A100 throughput; the plan's "H100-eq hours" assume 1 H100-eq = 1.75
A100-h, which the smoke test can check if run on both. Modal's per-second billing
and free credit make it the cheaper way to learn *whether* the pipeline runs; Vast is cheaper for
anything repeated.

## Next

* **B-scale-0** (pre-registered above, data generating on GCP): JEPA on 100 / 300 / 1,000 new episodes
  at 10,000 steps, MAE on 1,000, with floor, raw-proprio and supervised bars on 100 held-out episodes.

* **B-teach-long** is done (pre-registered verdict: null; see its results above). Heavy training no
  longer runs on the Mac.
* **Dropped:** proprio dropout 50%. At 1,000 steps it is -0.178 +- 0.037 against the uniform arm
  (above its 0.105 MDD) and partly collapses the state (B-conseq-0); it is not part of any further run.
* **Held:** the scale curve (c-none on 25 / 50 / 100 episodes, width 256) and the masked-patch
  autoencoder. Reason: no encoder has yet beaten its random-init floor on any probe set (B-conseq-0,
  B-exp-0), so a curve over data size or a new objective family cannot be read until one does.
* Touch-change-weighted loss (`model.surprise = touch`) is the one lead of B-conseq-0 (+0.036 +- 0.022
  over its twin, below the 0.063 MDD): at n = 6 it is +0.032 +- 0.020 (MDD 0.056), and the input +
  target arm `a-both` is no better than the baseline (see the n = 6 section above).
* **Parked: touch-change weighting** (`model.surprise = touch`) and the input + target arm. Same sign
  four times (+0.036 +- 0.022 and +0.032 +- 0.020 on random play, +0.079 +- 0.026 and +0.043 +- 0.023
  on S2b), **never resolved**, the arm sits at its floor (+0.012 +- 0.017) with a flat curve, and every
  arm on random play is at or below the floor, so how the encoder learns cannot be read until it
  learns at all. Reopen only once an objective clearly beats its floor. Effective rank of the
  `a-*` checkpoints (mean over 6 seeds, steps 250 / 1,000): `a-base` 37.2 / 40.2, `a-touch` 35.4 / 40.0,
  `a-both` 36.2 / 36.7, flat while the probe is flat (`scripts/collapse_retrofit.py --objective`).
* The S2b hand-experiment study (`scripts/s2b_test.py`); the hidden-property probe is parked (no range at
  this scale, even for a supervised ceiling), as is the response-based version.
* Run the GPU smoke test above (supervisor's compute decision): throughput, utilisation, memory,
  and a first look at rank at B = 128 for both objectives (`model.objective`).
* The token-level variant (later brief); the DINOv2 stem option.
* Taxel coordinates as layout metadata from Track A, for a geometric touch stem shared by hands and
  pincers.
