# Event-dense table data (Track A, from 2026-10-21)

The supervisor's direction: babbling has too little signal. The data should be mostly experiences in
which something happens (grasps, drops, falls, topples, collisions and impacts on the body, objects
seen moving), with a minority of rest (about 10-20%). This page defines how that is measured, the
baseline of the existing sets, and the generator setting built for it.

## The metric: `sensorium.event_rate`

Label-free: it uses what physically happens (privileged object poses and the per-step peak contact
forces), not what a program meant to do, and never reaches a model. Per control step it finds the
onsets of eight types: contact start and end (an effector on an object), grasp (an object touched
and 1 cm off its resting height for 2 steps), lift (touched and 5 cm up), drop (untouched, 3 cm up
and falling), fall (below the table's surface), topple (a non-spherical object's tilt crossing 45
degrees), and impact (the peak force of effector-object, limb-scene or self contacts rising past
20 N; effector-table contacts are left out, a hand pressing on the table re-crosses any threshold).
It also tracks one state, an object moving faster than 5 cm/s. A 1 s window is **eventful** if it
holds an onset or 3 steps of motion. **Rest** is the share of windows that are not. Thresholds and
reasons are in the module docstring.

## Baseline

Measured on 150 episodes of each set, 5 per block (copies of the cloud data):

| set | eventful | rest | onsets/min: contact start | grasp | lift | drop | fall | topple | impact |
|---|---|---|---|---|---|---|---|---|---|
| babble_v2 | 0.76 | 0.24 | 32.2 | 7.7 | 3.4 | 0.2 | 0.3 | 6.1 | 17.1 |
| variety | 0.67 | 0.33 | 33.4 | 7.9 | 2.9 | 0.7 | 0.3 | 5.8 | 17.1 |

Per block, variety ranges from 0.35 eventful (pincer door and slide) to 0.92 (hand babbling with the
props). Drops and falls are almost absent everywhere.

## The generator setting: a scene director (`bodies/h1_director.py`)

A recorder setting, `director`, intervenes in the scene, never in the robot. All interventions are
logged in `privileged/intervention`.

- **Stacks** (`stacks`, probability per episode): at reset, a box or cylinder is set on a wider one,
  for the arms to knock over.
- **Re-placing** (`respawn`): an object on the floor, or out of reach for 2 s, is dropped back over
  a free spot within reach.
- **Dropping when idle** (`idle_s`): after that long with nothing moving, the object dropped least
  recently is dropped from 15-30 cm. With probability `onto_hand` it lands on a hand (an impact on
  the body); otherwise it lands on a free spot on the table.

There is also a recorder setting `plan_tries`, which caps the babbling planner's goals per plan.

Ablation: 6 episodes per row on the Mac without rendering (same seeds per row). Every row has the
pretraining props (button, friction patches, hinged flap, tall block) except the first.

| effector / variant | eventful | rest | lifts/ep | topple/min | drop/min | fall/min | impact/min | contact | s/episode |
|---|---|---|---|---|---|---|---|---|---|
| hand: babble, no props | 0.82 | 0.18 | 0.5 | 10.0 | 0.0 | 0.0 | 11.7 | 0.45 | 75 |
| hand: + props | 0.83 | 0.17 | 0.8 | 7.3 | 1.7 | 0.3 | 13.3 | 0.41 | 84 |
| hand: + director, idle 1.5 s | 0.93 | 0.07 | 2.7 | 24.0 | 19.0 | 3.0 | 20.3 | 0.47 | 70 |
| hand: + director, idle 2.5 s | 0.88 | 0.12 | 1.2 | 25.0 | 16.3 | 3.3 | 21.3 | 0.43 | 71 |
| hand: + director, idle 3.5 s | 0.87 | 0.13 | 1.2 | 20.3 | 10.7 | 1.7 | 16.3 | 0.43 | 72 |
| pincer: babble, no props | 0.75 | 0.25 | 1.0 | 2.7 | 0.0 | 0.0 | 38.0 | 0.31 | 70 |
| pincer: + props | 0.87 | 0.13 | 1.2 | 7.7 | 1.7 | 0.7 | 49.3 | 0.41 | 92 |
| pincer: + director, idle 1.5 s | 0.93 | 0.07 | 1.3 | 22.7 | 15.3 | 1.3 | 54.0 | 0.42 | 69 |
| pincer: + director, idle 2.5 s | 0.92 | 0.08 | 2.2 | 19.7 | 16.0 | 1.3 | 48.3 | 0.46 | 82 |
| pincer: + director, idle 3.5 s | 0.89 | 0.11 | 1.7 | 17.0 | 11.7 | 1.3 | 46.0 | 0.44 | 84 |

The idle time sets the rest share. 2.5 s keeps it at 8-12%, close to the 10-20% asked for. A first
version re-placed objects as soon as they left the reach, before they could fall; it had no falls
and 20% rest without the idle drops. Capping `plan_tries` at 2 (with that first version and idle drops at 1.5 s) did
not change hands (0.94 eventful). For pincers it cut events (0.96 -> 0.83 eventful, impacts 42 ->
29/min) while running faster (87 -> 53 s per unrendered episode), so `configs/event_dense.toml`
leaves the planner as it is.

## Pilot: `configs/event_dense.toml`

32 rendered episodes on the Mac: the first 4 episodes of 4 hand and 4 pincer blocks. In the same
run, 3 episodes each of a hand and a pincer babble_v2 block, for speed. Tables and figures from
`scripts/event_report.py`.

| set | episodes | eventful | rest | contact start/min | grasp/min | lift/min | drop/min | fall/min | topple/min | impact/min | contact | lifts/ep | waiting | s/episode (Mac, same run) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| babble_v2 (sample of cloud data) | 150 | 0.76 | 0.24 | 32.2 | 7.7 | 3.4 | 0.2 | 0.3 | 6.1 | 17.1 | 0.35 | 1.1 | 0.10 | |
| variety (sample of cloud data) | 150 | 0.67 | 0.33 | 33.4 | 7.9 | 2.9 | 0.7 | 0.3 | 5.8 | 17.1 | 0.30 | 0.6 | 0.21 | |
| babble_v2 blocks, Mac | 6 | 0.80 | 0.20 | 35.0 | 9.7 | 3.7 | 0.7 | 0.0 | 8.0 | 14.7 | 0.38 | 1.5 | 0.12 | 119 |
| event_dense pilot | 32 | 0.88 | 0.12 | 42.2 | 16.7 | 11.9 | 15.2 | 0.8 | 19.5 | 30.4 | 0.40 | 1.0 | 0.08 | 108 |
| ... hands | 16 | 0.87 | 0.13 | 52.0 | 20.4 | 12.2 | 15.4 | 1.2 | 23.1 | 18.2 | 0.44 | 0.9 | 0.05 | 107 |
| ... pincers | 16 | 0.88 | 0.12 | 32.5 | 13.0 | 11.5 | 15.0 | 0.4 | 15.9 | 42.6 | 0.37 | 1.1 | 0.12 | 110 |

The director makes about 6 interventions per episode, one every 5 s. Recording costs no more than
babbling. Per episode, the largest effector-object force has median 67 N and max 256 N (babble_v2
on the Mac: 66 and 378). Limb contacts reach 846 N at most (babble_v2: 104), when an object is
dropped onto an arm: the "hurting" this data is meant to contain.

Figures: `runs/event/event_hist.png` (onsets per window and eventful share per episode, by
set) and `runs/event/event_strip.png` (head-camera frames around a drop onto a hand).

Caveats:
- `lift` (onset of touched and 5 cm up) also counts an object landing on a hand. Lifts the robot
  made itself are `lifts/ep` (`babble_stats.count_lifts`, held 0.5 s), which stays at about 1 per
  episode.
- Dropped objects appear in mid-air, inside the head camera's view at the low end of the drop
  heights (the strip shows one at +0.05 s). Dropping from higher would hide the appearance but hit
  harder.
- The director's events are not the robot's doing. `privileged/intervention` lets training tell
  them apart or weight them.
- The A1 force lines were set for babbling, not for this data: drops onto a body are meant to
  exceed them.

## Hand-sized objects (2026-10-24)

The six table objects are 3-10 cm across, small for a Shadow hand: its claw pinch lifts 28% of its
isolated grasps of them, and 45% [31, 60] of grasps of hand-sized props (`docs/touch.md`). The
`hand_sized` prop (`h1_props.HAND_SIZED_OBJECTS`) adds four free objects, and the hand blocks of
`configs/event_dense.toml` now have it (pincer blocks unchanged):

| object | shape | size | nominal mass |
|---|---|---|---|
| `obj_metal_can` | cylinder | 7 cm across, 16 cm tall | 0.35 kg |
| `obj_plastic_bottle` | cylinder | 6 cm across, 18 cm tall | 0.30 kg |
| `obj_wood_handbox` | box | 11 x 7 x 6 cm | 0.25 kg |
| `obj_rubber_ball` | sphere | 8 cm across | 0.15 kg |

- The nominal masses replace the material density, which would make a metal can 4.8 kg.
- With `hidden_properties` the mass, friction and softness are drawn per episode, as for every
  object (0.1-1.5 kg).
- A hand block now has 12 free objects; the placement relaxes its spacing on a crowded table, as
  with the other props.
- `tests/test_h1_props.py` checks that each object spawns, settles on the table within a short
  episode, and is reachable by the planner's claw pinch.
- The event rate and the grasp success with these objects are measured on the cloud, not on the
  Mac (`docs/learned-grasp.md`, pre-registration).
