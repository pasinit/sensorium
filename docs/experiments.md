# Experiments with consequences (Track A)

Direction of 2026-10-15: random poking teaches little, so the robot **acts on the world and
experiments**, and the data records what it caused. Arms, hands and the waist first; legs later.
This page is the contract; the design memo (repertoire, literature, effort) lives with Track A.

## What gets judged

`sensorium.caused_change` counts, per minute of episode and by outcome type, what the robot made
happen (`displaced`, `lifted`, `toppled`, `dropped`, `stacked`, `unstacked`, `opened`, `closed`,
`pushed_to_target`, `body_moved`), from privileged state at analysis time, and the **outcome-type
coverage** (types at or above one event per minute). Thresholds and their reasons are in the
module docstring, with a changelog. `scripts/caused_change_report.py <dir> [--vs <dir>] --plot out.png`
prints the table and draws events per minute against another data set (random play).

This replaces contact share as the headline of A1; contact share and safety (forces, falls, tips in
view, waiting, repertoire) stay as gated constraints (`make a1-gate`).

Random play (step 4, 10 episodes of 600 steps per effector): hands 23 displaced per minute and
coverage of 2 types, pincers 11 displaced, 2.2 lifted and coverage 3; everything compositional
(stack, open, push to target) is zero.

## Held-out tasks

`configs/splits/v3.toml` holds out T1 stack two blocks (pincers), T2 open the drawer (pincers),
T3 push a block onto a target and T4 reach the far-side object by turning the torso, with BiGym's
`StackBlocks` and `DrawerTopOpen`. Each source in `configs/sources.toml` lists its `programs` and
`props`; a split fails to load if a pretraining source runs a held-out program or uses a held-out
prop. The neighbouring skills stay in pretraining: push until it moves, lift and drop, topple,
slide, press, knock, the door, and turning without objects.

## Props and hidden properties (`h1_props.py`)

Recorder setting `props` (any of `tall_block`, `friction_patches`, `button`, `hinged_flap`,
`drawer`, `target_disc`; the drawer and the flaps do not go together: they share a corner).

| Prop | What it is | Logged as |
|---|---|---|
| `tall_block` | a 3 x 3 x 15 cm block and a 4 cm bottle, 15 cm tall (`obj_*` bodies, like the six table objects) | `object_pose`, `object_contact` |
| `friction_patches` | three strips across x with sliding friction 0.1 / 0.5 / 1.2; their colours are shuffled per episode | geoms `table_patch*` |
| `button` | two spring buttons (left and right), 8 mm travel, about 5 N at full press | `prop_state` (`prop_button_l/r`) |
| `hinged_flap` | two small hinged flaps that swing away when pushed from the robot's side | `prop_state` (`prop_flap_l/r`) |
| `drawer` | a drawer with a pinchable knob, 10 cm travel, opens towards the robot | `prop_state` (`prop_drawer`) |
| `target_disc` | a visual disc (no contact) at a per-episode position | `privileged/target` |

`privileged/prop_state` is the joint travel in [0, 1] of each prop joint (column names in
`meta.prop_names`). Objects are placed outside the zones the props occupy.

**Hidden properties** (`hidden_properties` setting): per object and episode, drawn from the seed
and stored in `privileged/hidden_properties` (columns in `meta.hidden_property_columns`), never
an input: mass 0.1-1.5 kg (log-uniform), sliding friction 0.2-1.0 (uniform), contact softness
(`solref` time constant) 0.01-0.06 s (log-uniform). An object's look does not depend on them. The
nominal values are restored at every reset.

## Effort channels (data card)

Setting `effort`: the actuator force of every body and effector joint is appended to `proprio` as
channels `eff_<actuator name>` (N m, N for the gripper), at the end of the list, so a loader can
include or drop them by prefix; hands add 61 channels, pincers 23. The real H1 reports motor
torque or current in its motor state, which is the reference. Noise model: additive Gaussian with
sigma = 1% of the actuator's force range, clipped to the range and quantised to 12 bits. The
simulated force is the position controller's output, so it is exact before the noise; it is not a
measured current.

## Experiment programs (`h1_experiments.py`)

Recorder policy `experiment` with a `program` setting repeats one program per episode (task
`exp_<program>`): `push` (12 cm from the side), `topple` (a push at a quarter, half or three
quarters of a tall object), `slide` (15 cm along a friction patch), `press` (a button, or an
object's top 1 cm deep, held), `knock` (a strike at one of three speeds) and `door` (push a flap
open). Both arms act independently; an arm that has no plan parks (counted as waiting). A
near-vertical pincer cannot reach past x of about 0.5 m, so the pincer programs choose targets
inside that range. `configs/experiments.toml` is the short check, `experiments_batch.toml` the
72,000-step batch for Track B.

## Kind of experience, coverage bar, reports

* The recorder's policy `experiment` is its own **kind of experience** (`KINDS` has `expert`,
  `noisy`, `babbling`, `experiment`): experiment against babbling is the contrast both tracks
  need, so pools stratify on it. Indexes built before 2026-10-15 have these episodes as
  `babbling`: re-index.
* Coverage counts the outcome types the repertoire can reach, which excludes `body_moved` (waist
  work is deferred): the bar is 6 of 9 types at one event per minute or more.
* `scripts/experiments_report.py <dir> <random-play dir> <prefix>`: a table per (effector, program),
  stacked events per minute against random play, histograms, a frame strip per variant.
* `scripts/experiment_outcomes.py <dir> [--plot out.png]`: attempts, intended outcomes per attempt,
  share of episodes with an outcome, share of steps on an object, on the table, and waiting.
* Experiment episodes log `privileged/phase` ([T, 2], per arm; `meta.phase_names`: travel,
  descent, contact, wait, retreat) so the split of arm time is measured, not inferred.
* `configs/experiments_batch.toml` (S2 part 1, stopped), `experiments_batch2.toml` (knock, hands'
  door), `experiments_s2b.toml` (denser programs, fresh master seed).

