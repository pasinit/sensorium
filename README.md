# sensorium

**Can a *sensory layer* built from a plethora of different experiences turn task-specific learning into few-shot learning?**

A **sensory layer** encodes vision, sound, touch and proprioception, together
with the actions taken, into one latent state. It is pretrained without
rewards, labels or text, by predicting how the senses change when a body acts,
on one humanoid body (Unitree H1 with whole-body skin, egocentric eyes and
vestibular sense) across locomotion, reaching, home manipulation, piano and
kinds of experience (expert, noisy, task-free babbling), then adapted to new tasks
from a handful of examples (frozen head or LoRA). The bet: the *senses* and
the *variety of contact* it is built from matter more than the number of
skills it was shown. Language is only a readout of that latent state.

## Documents

| Doc | What |
|---|---|
| [docs/thesis.md](docs/thesis.md) | Goal, body choice, hypotheses H1–H4 with verified evidence and gaps, predictions, roadmap (D0, E1–E4) |
| [docs/dataset-proposal.md](docs/dataset-proposal.md) | **D0: Experientia**, a whole-body multisensory experience dataset on one humanoid, with controlled senses, experience kinds and sources, plus a held-out few-shot benchmark |
| [docs/experiment-01.md](docs/experiment-01.md) | **E1:** senses × kind of experience on one humanoid: design, conditions, the two end effectors, protocol, compute, milestones, risks |
| [docs/datasets.md](docs/datasets.md) | Simulator & dataset inventory, coverage matrix, what's missing |
| [docs/data-generation.md](docs/data-generation.md) | Unified episode format, synthetic audio, templated language, **manual** data tasks, quality gates |
| [docs/literature-review.md](docs/literature-review.md) | **What's already done**, a verdict per hypothesis, the body survey, and the superseded earlier framings |
| [docs/review-sample.md](docs/review-sample.md) | How to review the input sample, checklist, known issues and open decisions |
| [docs/touch.md](docs/touch.md) | How contacts become touch channels on the H1: dense skin, cube pads, onset / slip / vibration, load-time noise, with the literature behind each |
| [docs/vision.md](docs/vision.md) | What the cameras record (256 px, a room recoloured every episode), how frames are stored (H.264) and cropped at load time, with the literature and rendering costs |
| [docs/pools.md](docs/pools.md) | **D0.1:** experience-source registry, frozen split, episode index and equal-volume pools (senses, kind of experience, sources, effector ratio) |
| [docs/tracks.md](docs/tracks.md) | Team and research tracks: supervisor, postdoc, and two PhD tracks (A: Experientia dataset and benchmark; B: the sensory layer and E1), with their boundaries |
| [docs/roadmap.md](docs/roadmap.md) | **Status, next steps and the decision log** (why the plan changed three times on 2026-10-11) |
| [docs/references.md](docs/references.md) | Papers and links |

## Quickstart: generate and review the M1 input sample (car, drone, piano hands)

```sh
make setup        # uv env (Python 3.13), simulators pinned to exact commits
make rp1m-toy     # RP1M expert piano data, toy subset (351 MB, checksum-verified)
make sample       # 16 episodes: car, drone, piano (~1 min, CPU only)
make browse       # build data/sample/index.html, then open it in a browser
```

What to look for: [docs/review-sample.md](docs/review-sample.md).

## Quickstart: the H1 humanoid (D0.2)

```sh
make h1-assets    # H1 MJCF assets at pinned commits into data/external/ (~120 MB)
make sample-h1    # 8 episodes: Shadow hands and Robotiq pincers, standing and seated (~45 s)
open data/sample_h1/index.html
```

`make smoke` includes one humanoid episode (needs `make h1-assets`).

## Code

| Module | What |
|---|---|
| `src/sensorium/episode.py`, `codec.py` | The one episode format (save/load/validate, camera frames as H.264) and the model-input allow-list |
| `src/sensorium/bodies/` | `humanoid.py` (H1 with skin: `h1_scene.py` scene assembly, `h1_looks.py` per-episode colours and lights, `h1_sense.py` grid skin and contacts, `h1_skin.py` dense skin and contact dynamics, `h1_policies.py` behaviour), `car.py` (MetaDrive), `drone.py` (gym-pybullet-drones), `piano.py` (RoboPianist + RP1M re-rendering) |
| `src/sensorium/audio.py` | Synthetic sound per body + log-mel features |
| `src/sensorium/describe.py` | Templated step descriptions (language-probe labels only) |
| `src/sensorium/generate.py` | Config → episodes + `manifest.json` (seeds, versions, content hashes) |
| `src/sensorium/browse.py` | Episodes → static HTML review pages |
| `src/sensorium/touch_noise.py`, `vision.py` | Load-time touch perturbation (gain, dead channels, noise, binarisation) and image views (random / centre crop, resize, colour jitter) for the training loader |
| `src/sensorium/registry.py`, `index.py`, `pool.py` | Source registry; manifests → parquet index; equal-volume pretraining pools (`make index`, `make pool`) |
| `configs/` | `sample.toml` (review sample), `sample_h1.toml` (humanoid review sample), `smoke.toml` (1 tiny episode per body), `sources.toml` (source registry), `splits/v0.toml` (frozen split), `pools/` (pool specs) |

## Status

- **M0** (thesis, plan) done.
- **M1** (data pipeline + review sample) built and awaiting your data review.
- **Framing (2026-10-11):** one humanoid body (H1, whole-body skin), two end effectors (Shadow hands; Robotiq pincers for BiGym's human demos) with explicit controls; E1 tests which *senses* and which *kind of experience* (expert vs task-free babbling) make new skills few-shot, on **Experientia** (D0). Diversity is a secondary check; car and drone are an "alien body" ablation.
- **D0.1** (pool builder) and **D0.2** (the H1 in our episode format) built; the humanoid sample awaits your review: `make h1-assets && make sample-h1`.
- **Touch** (2026-10-12): dense skin on the hands and pincer pads, onset / slip / vibration, load-time noise ([docs/touch.md](docs/touch.md)).
- **Vision** (2026-10-12): 256 px frames for every body, stored as H.264; the H1 sees a room whose colours and lighting change every episode; crop and resize at load time ([docs/vision.md](docs/vision.md)).
- **Next:** D0.3 (babbling). See [docs/roadmap.md](docs/roadmap.md).

## Dev

Python 3.13, `uv`, and `make lint | format | typecheck | test | smoke`.
