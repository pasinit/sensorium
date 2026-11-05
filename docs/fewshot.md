# The few-shot harness (D0.6, minimum for the E1 pilot)

The E1 pilot scores pretrained sensory layers by how well a small head on their frozen states
learns the held-out tasks from k demonstrations ([experiment-01.md](experiment-01.md), §5-6). This
is the minimum version: frozen heads only. LoRA, the other vision baselines and the effector probe
come with the full E1.

## Parts

| part | code | what it does |
|---|---|---|
| encoders | `fewshot.layer_features`, `fewshot.dino_features_fn` | a pretrained layer's state for the window ending at each step (`model.planning.encode_at`: eval mode, centre crop, optional senses masked out), or the frozen DINOv2 ViT-S/14 baseline on the frame there (`model.controls.dino_features`) plus proprio |
| cache | `fewshot.build_cache`, `DemoCache` | features at every window end of the **successful** demonstrations (a weak demonstrator fails often: T3 succeeds 3/12 with hands), and the next `horizon` actions (the chunk) |
| head | `fewshot.BCHead`, `train_head` | two linear layers (hidden 256, ReLU) on standardised features -> the chunk; mean squared error, AdamW, 300 epochs; output clipped to [-1, 1]. One configuration (`BCConfig`) for every encoder and k |
| offline score | `fewshot.action_rmse` | RMS action error on the demonstrations the head did not see. Fast and needs no simulator: for iteration |
| closed loop | `bodies.h1_closed_loop` | **the pilot's primary score**: the head drives the arms in the simulator, the task's scripted program judges |
| command line | `scripts/fewshot.py` | `cache`, `sweep` (k x seeds, offline score), `closed-loop` (success rate with a Wilson interval) |

## Closed loop

The task's scripted program, in task mode (`h1_experiments.ExperimentPolicy`), is the judge.
- From the episode's seed it sets the scene up as for the demonstrations: the props, the layout and
  the arms' start.
- Its success test, held for 1 s (`SUCCESS_STEPS`), ends the episode as a success.
- It never moves the arms: a `LearnedPolicy` does.

Every `horizon` steps the policy encodes the last window of what the robot sensed and plays the
head's next `horizon` actions. `HumanoidRecorder.recent_episode` gives the model inputs only, no
privileged state.
- The recorder logs a step's senses before asking such a policy to act (`needs_observation`), so the
  head sees the current step.
- Other policies keep the old order, and their episodes are unchanged.

`HumanoidRecorder.record(..., wrap=...)` puts the learned policy in place of the program it made
from the same seeds.

**Seeds.**
- Evaluation episode i of a config block has seed `episode_seed(master_seed, "humanoid",
  "closed_loop", i, block)`.
- The demonstrations hash the policy `experiment`, so the evaluation starts from new layouts of the
  same task.

## Floors and sub-goals

**Null-policy floors.** `scripts/fewshot.py floor`, on the same closed-loop start states:
- `hold` keeps the start posture;
- `replay` plays an unrelated demonstration's actions open loop.

A task that either one solves cannot rank encoders. That is how the drawer's first demonstrator
was caught: it succeeded by a sweep. Both use `bodies.h1_closed_loop.ReplayPolicy`.

Both are open loop and judged on the physics state, so `--no-render` skips the cameras
(`HumanoidRecorder.render_eyes = False`: black frames, `meta["blind"]`, the same physics): two
256-pixel views per step in Mesa's software renderer on CPU machines. `floor` and `closed-loop` print
one line per finished episode, so a VM's stall rule can count episodes.

**Sub-goals.** Every held-out task records its sub-goals in order in `meta["milestones"]` and in
each closed-loop outcome, for split scoring:
- T1 `picked` -> `placed`;
- T2 `pinched` -> `opened`;
- T3 `reached` -> `at_target`;
- T3b `grasped` -> `lifted` -> `at_target` (only after the lift) -> `released` (the success,
  held 1 s).

T3 also records its pushes per episode in `meta["task_counts"]`: a multi-push demonstration is
harder to imitate from one example.

## Held-out tasks of the pilot (split v3.2)

| task | effector | demonstrator success | status |
|---|---|---|---|
| T1 stack two blocks | pincers | 15/29 = 52% [34, 69] (check of 2026-11-05) | demonstrations being generated |
| T2 open the drawer | pincers | fixed cabinets: replay 19/21 of successes; v3.3 (reach edge): demonstrator 7/30, replay 14/50; v3.4: low cabinets, knobs over 20 x 20 cm, table +-3 cm | check pending |
| T3 push to the target | hands, pincers | post-fix re-check 6/30 (hands) and 0/30 (pincers): under the agreed 30% | kept held out under its own name, not used in the pilot |
| T3b place on the target | pincers | v3.2 check 3/30 (the 7 cm wood box: no plan or slips); v3.3 carries T1's block, alone, 10 cm or more from the disc | check pending |

k ∈ {1, 5, 10, 50} needs at least 51 successful demonstrations per task and effector (one to
score). The closed-loop runs need their own start states: 50 per task gives Wilson intervals of
about ±12 points at 30%.
