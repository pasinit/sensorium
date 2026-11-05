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
* `lift_drop` runs with hands too since the hand's claw pinch (`docs/touch.md`), as the source
  `humanoid/hand/exp_lift_drop` (the hands grasp with the babbling's claw pinch: 2.3 lifts per
  episode in a 6-episode smoke run); stack and the drawer stay pincer-only.
* The stack task's success test (T1) needs the top block at rest on the other one (slower than
  2 cm/s, touching it, touched by no robot part) for its 1 s: a block held over the other does not
  count.


## Known gaps (2026-10-15)

* **Pincer slide, press and door wait 72-79% of the time** (S2b: 12 episodes each, `data/experiments_s2b`):
  a near-vertical pincer plans targets no further than x of about 0.5 m and cannot plan a low approach
  next to an object (the forearm meets the table). They are kept as a diagnostic, not as training data:
  pool specs should leave `humanoid/pincer/exp_slide`, `exp_press` and `exp_door` out until a pincer
  planner fixes them (pincer knock, topple, push and lift-drop plan well enough). **Pincer press has one
  episode with a 300 N effector-object peak**, above the 2F-85's 235 N grip and under the 470 N gate limit.
* **The drawer demonstrator (T2) succeeded by shoving (found 2026-11-05, fixed).**
  - In the demonstrator check (30 pincer episodes) all 30 "succeeded" by step 29. On its first
    travel the arm pressed the pincer into the cabinet at 37-118 N and shoved the drawer open; the
    knob was never pinched. The cabinets are `table_*` geoms, which an effector may touch.
  - The demonstrator now keeps its effector clear of the cabinets on its travel and approach
    (`BabblePolicy.avoid`: 8 mm, a posture every 0.03 rad along each path).
  - The task's success test now also requires the drawer to pass half its travel while both pads
    of a pincer hold its knob.
  - Held-out tasks record their sub-goals in `meta["milestones"]`: drawer pinched then opened,
    stack block picked then placed.
* **T2 redesigned: low cabinets in comfortable reach, and the table height (split v3.4).**
  - v3.3 (cabinets at the reach edge, |y| 0.28-0.33): demonstrator 7/30 (no plan in 20), replay of
    a successful demonstration 14/50. A blind pinch caught a knob within ~2 cm (7/7), sometimes
    within 2-5 cm (7/22), never beyond 5 cm (0/21).
  - Now each cabinet (8 x 10 x 3 cm, a 2.4 cm knob post) is drawn by its knob over x 0.42-0.62,
    |y| 0.10-0.30 on its side, yaw +-15 deg, its footprint 4 cm clear of the resting pincer and of
    the other cabinet; the table's height is drawn within +-3 cm (`table_jitter`, drawer scenes
    only), and the planners, the objects and the cabinets follow it.
* **T2's cabinets are posed per episode (split v3.3).**
  - At v3.2 the cabinets stood still: the replay floor (an unrelated demonstration played open
    loop) opened the drawer in 19 of 50 start states, 19 of the 21 whose demonstration was a
    success (Wilson 95% [71, 97]). No encoder could beat that floor by the pre-registered margin.
  - Each cabinet's front x (0.38-0.54), |y| (0.26-0.33) and yaw (+-15 deg) are drawn per
    episode, with its inner edge 19 cm or more from the midline (the resting pincers are at
    |y| ~0.12). The demonstrator pulls along the drawer's slide; objects keep off the footprints.
* **T3b carries T1's block (split v3.3; its v3.2 check was 3/30, Wilson 95% [3, 26]).**
  - At v3.2 the carried object was always the 7 cm wood box (the first object is the far one),
    which the disc's 17 cm rule put at the reach edge (|y| 0.25-0.31): no plan in 17 of 30
    episodes; 9 of the 13 grasps slipped in the pads (median mass ~0.7 kg; those lifted ~0.17 kg);
    2 boxes were dragged, still pinched, onto the disc and counted as at the target.
  - Now one 4 cm T1 block (prop `place_block`), alone, in T1's start region on the disc's side,
    10 cm or more from the disc. At the target and the success count only after the block rose
    in the pads; `released` is the success held for 1 s.
  - As a held-out test it is close to T1: the same object, grasp and start region, with a goal
    on the table instead of on a second block. It tests a new goal for a known grasp, not a new
    object or a new skill.
* **T3 is replaced in the pilot by T3b, place on target (pincers; split v3.2).**
  - The re-check after the closed-loop push change still fell under the agreed 30%: hands 6/30
    (reached 30/30, a median of 20 pushes), pincers 0/30 (the push mostly never started).
  - T3b's demonstrator (`plan_place_on_target`) pinches an object that began 17 cm or more from the
    disc, carries it with its bottom 6 cm over the table, sets it on the disc and lets go.
  - Success: within 3 cm of the disc's centre, still, touched by no robot part, for 1 s.
* **Push to the target (T3) failed (check of 2026-11-05: 1/4 hands, 0/11 pincers; changed).**
  - Its single straight push rarely ended within 3 cm.
  - The disc sat at the pincers' reach edge, so pushes often had no plan.
  - Now each push moves the object at most 6 cm, re-aimed from where it lies (the same object first,
    else the next one with a plan).
  - The disc and the far object are placed inside both effectors' reach (split v3.1,
    `configs/splits/v3.toml`).
  - The earlier note follows:
* **Push to the target (T3) is weak:** 3 of 12 demonstrator episodes succeed with hands and 1 of 12 with
  pincers; only the successful episodes should be offered as few-shot demonstrations.
* Some `displaced` events of the hand programs come from the hand sweeping objects during a low hover
  (6 cm over the table, below the tops of the 7 cm objects), not only from the intended contact; the
  push-to-target program hovers at 10 cm for this reason.
