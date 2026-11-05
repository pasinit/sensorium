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

## Held-out tasks of the pilot (split v3)

| task | effector | demonstrator success | status |
|---|---|---|---|
| T1 stack two blocks | pincers | not recorded (only a unit test shows a success) | to measure before the demos are generated |
| T2 open the drawer | pincers | not recorded (one unit-test episode succeeds within 400 steps) | to measure |
| T3 push to the target | hands, pincers | 3/12 and 1/12 (`docs/experiments.md`) | weak: many attempts per successful demo |

k ∈ {1, 5, 10, 50} needs at least 51 successful demonstrations per task and effector (one to
score), and the closed-loop runs need their own start states: 50 per task gives Wilson intervals of
about ±12 points at 30%.
