# Reviewing the M1 input sample (car, drone, piano hands)

The goal is to check, by eye and ear, that the data is what the sensory
layer should learn from, *before* any GPU time is spent.

**Note (2026-10-11):** this sample predates the move to one humanoid body
([thesis.md](thesis.md#body-one-humanoid-whole-body-all-senses)). The piano
hands, their touch and the sound synthesis carry over to the H1; the car and
drone are now the alien-body ablation. The first humanoid sample arrives
with D0.2. Reviewing this one is still useful for the format, the audio and
the piano replay, but the open decisions below are no longer blocking.

```sh
make setup && make rp1m-toy && make sample && make browse
open data/sample/index.html
```

The sample (`configs/sample.toml`, master seed 0) has 16 episodes, about 5.5
minutes of experience (car 180 s, drone 68 s, piano 80 s). It takes ~1 min to generate on an M1 laptop and is 36 MB:

| Body | Episodes | Per episode | Policies |
|---|---|---|---|
| car | 6 | 300 steps × 0.1 s = 30 s | expert, noisy, explore (2 each) |
| drone | 6 | up to 600 steps × 0.025 s = 15 s | expert, noisy, explore (2 each); explore ends in a crash after ~4 s |
| piano | 4 | 400 steps × 0.05 s = 20 s | RP1M expert |

Each episode page shows the video as stored (64×64 in this first sample,
upscaled 4× without smoothing; 256×256 since 2026-10-12, see
[vision.md](vision.md)) with its sound, plots of proprioception, touch, actions and audio
features, with a cursor that follows the video, and the step descriptions
(click one to jump there).

## Checklist

1. **Vision.** Is 64×64 enough to see what matters? Road and traffic for the
   car, obstacles and horizon for the drone, hands and keys for the piano.
2. **Sound.** Does it follow the action? Engine pitch should rise with car
   speed, rotor tone should shift as the drone manoeuvres, and notes should
   start exactly when keys go down.
3. **Behaviour.** Do the policies look like expert, noisy and explore? Noisy
   should be visibly imperfect and explore clearly erratic. Neither should be
   so broken that nothing is learnable.
4. **Touch (piano).** Spikes should line up with key presses.
5. **Descriptions.** They should match what you see and hear at that moment.
6. **Timing.** Frame, sound and plots should line up when you scrub.

## Known issues and decisions for you

| # | Observation | Why | Proposal |
|---|---|---|---|
| 1 | **Piano actions look nearly bang-bang** (action heatmap is speckled) | RP1M experts are RL policies, which saturate their actions | Keep: it's the real expert data. Low-pass filtering would change the action→motion mapping. |
| 2 | **Piano touch is heavy-tailed**: median 0, 99th pct ~35, max ~890 | Kinematic replay can push a fingertip into a key, producing large contact forces | Keep raw values in the data; the training loader (next milestone) will apply `log1p`. |
| 3 | **Piano action→motion pairs come from RP1M's simulator, not ours** | RP1M actions don't reproduce its motion under our RoboPianist settings (key F1 0.06). RP1M's paper (App. B) lists its settings: gravity compensation on, stretch factor 1.25, trim silence, lookahead 10, 39-D action; it does not state the RoboPianist/MuJoCo version. We have not yet replayed with those settings. | First try the replay with RP1M's listed settings. If it still fails, regenerate expert data with our own policies for action-conditioned piano prediction. |
| 4 | **Drone sound barely changes** | Rotors stay near hover speed when cruising slowly | Acceptable. Explore (1–2 m/s) has more variation. |
| 5 | **The drone always flies the same arena** (PyBullet's sample obstacles) | gym-pybullet-drones ships one obstacle set | Fine for an ablation source. |
| 6 | **Sound is a deterministic function of state** (+ per-episode timbre) | No simulator renders audio | As planned (see E1 risks). H1, the audio ablation, tells us whether it matters. |
| 7 | **Piano camera is top-down, not first-person** | At 64 px the egocentric view hides the keys behind the forearms | On the H1 the view comes from the head cameras, so this becomes a question of head pose while seated at the piano (D0.5). |
| 8 | **The humanoid is not in this sample** | It did not exist when the sample was built | D0.2 built it: `make h1-assets && make sample-h1` gives 8 H1 episodes (hands and pincers; see [data-generation.md](data-generation.md#h1-body-bodieshumanoidpy)). |

## Bugs found while building (already fixed)

- MetaDrive returns **BGR**, not RGB: blue sky looked brown and yellow lines
  cyan. Frames are now flipped to RGB.
- MuJoCo's renderer fails to load in a process that already ran MetaDrive, so
  each body now generates in its own process.
