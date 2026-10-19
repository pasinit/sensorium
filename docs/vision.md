# Vision: what the cameras record and how frames become model input

Updated 2026-10-12. Code: `episode.py` (`IMAGE_SIZE`), `codec.py` (video
storage), `bodies/h1_scene.py` (room, camera tilt), `bodies/h1_looks.py`
(per-episode appearance), `vision.py` (load-time crop and resize).

## Why this changed

The first humanoid sample stored 64 × 64 frames, the size the car, drone and
piano pipeline had used. On the H1 a Shadow-hand finger was 1–2 px wide and a
table object 2–4 px, so grasps were invisible. HumanoidBench's floor and sky
are the same light grey, so the standing scene looked like objects floating
in a void, and the top 40% of the seated view was empty. HumanoidBench itself
renders its eye cameras at 256 × 256 (`humanoid_bench/env.py`, both eyes).

## What the literature does (verified 2026-10-12)

| Family | Work | Stored / rendered | Model input | Augmentation |
|---|---|---|---|---|
| Pixel RL from scratch | DrQ-v2, arXiv:2107.09645 | 84 × 84, 3 frames stacked | 84 × 84 | pad 4 px, random 84 crop (±4 px shift), bilinear |
| Imitation from scratch | robomimic, arXiv:2108.03298 | 84 × 84 | random 76 × 76 crop | "pixel shift" crops; removing them hurts |
| Imitation from scratch | Diffusion Policy, arXiv:2303.04137 | 84, 96, 240 px sim; 320 × 240 real | 76, 84, 216 px; 288 × 216 | random crop in training, centre crop at test |
| Generalist policy | Octo, arXiv:2405.12213 | 256 (main), 128 (wrist) | 16 px patches: 256 + 64 tokens | random crop, resize, colour jitter; pixels to [-1, 1] |
| Video world model | V-JEPA 2 (and -AC), arXiv:2506.09985 | 256 (384 for ViT-g); robot clips 4 fps | 16 px patches, 2-frame tubelets | random resized crop, aspect 0.75–1.35 |
| World model on frozen features | DINO-WM, arXiv:2411.04983 | 224 | DINOv2 at 196 px: 196 tokens of 14 px | — |
| Frozen baseline | VC-1 (model card) | — | 224 × 224, 16 px patches | — |

DreamerV3 (arXiv:2301.04104) is often quoted at 64 × 64, but its paper gives
no image size. The shared conventions: **store larger than the model input
and take one random crop per sample (5–10% of the side), the centre crop at
test**; small CNNs read 84–96 px, transformers and pretrained encoders
224–256 px; normalisation lives in the model; colour jitter where data comes
from many scenes or real cameras.

## What is stored now

- **`rgb` is 256 × 256** for every body (`IMAGE_SIZE`). On the H1 the right
  eye stays in `privileged/rgb_right`, also 256 × 256 (v0 is monocular, see
  [data-generation.md](data-generation.md#h1-body-bodieshumanoidpy)).
- **H1 camera tilt 45°** (was 35°), 80° vertical field of view. The H1 has no
  neck, so one fixed tilt serves both postures: seated, the table fills the
  frame and the lap shows at the bottom; standing, the landmarks ahead stay
  in view.
- **A room.** A checkered floor and four walls 3 m from the robot (2.6 m
  high), one directional key light, a sky gradient. Textures are speckled
  near-white greys, so surfaces have grain to track and take their colour
  from the material. Walls collide and sound like the floor (label `wall`).
- **Scene statistics pinned.** MuJoCo scales the near clip and the shadow box
  by the model's extent, which the room would make 12.5 m: a 12.5 cm near clip
  and ragged shadows. `SCENE_EXTENT = 4` gives a 4 cm near clip, and
  `shadowclip 0.5` keeps shadows sharp within ±2 m of the robot (2048² map).

## Per-episode appearance (`h1_looks.py`)

Every episode redraws, from its own random stream (the fourth child of the
episode seed, so object layout, policy and sound are unchanged):

| What | Range |
|---|---|
| Floor tint | any hue, saturation 0–0.35, value 0.35–0.9 |
| Wall tint | any hue, saturation 0–0.25, value 0.55–1 |
| Table tint | any hue, saturation 0.1–0.6, value 0.3–0.8 |
| Seat tint | any hue, saturation 0–0.5, value 0.15–0.6 |
| Objects | own material colour, hue ±0.03, value ×0.8–1.15 |
| Key light | elevation 45–90°, any azimuth, strength 0.3–0.7, warm/cool ±8% |
| Headlight | ambient 0.15–0.35, diffuse 0.25–0.55 |

Objects keep their material's colour family (wood brown, plastic blue, metal
grey, rubber red) because what an object looks like should still predict how
it sounds and feels. The ranges are our choice, not from a paper. The draw is
stored in `meta["appearance"]`.

## Load-time view (`vision.py`)

`View(size=224, crop=0.9, brightness=0, contrast=0, saturation=0)`:

- `train_view(rgb, rng, view)`: one random square crop of 90% of the side
  (robomimic's and Diffusion Policy's ratio) for the whole clip, resized to
  `size` (bilinear, antialiased), then optional colour jitter, one draw per
  clip (torchvision `ColorJitter` semantics). Jitter is off by default: the
  H1 scene already varies colour and light per episode.
- `eval_view(rgb, view)`: the centre crop, resized, no jitter.

Output stays `uint8`; scaling and ImageNet statistics belong to each model.
Our encoder can read 224 (or 128 for a cheap pilot); VC-1 and DINOv2 baselines
read 224 from the same stored frames.

## Storage (`codec.py`)

256 px frames of a textured scene do not compress losslessly: ~50 KB per
frame with the `npz` zlib (lossless WebP ~44 KB), which would make the ~20M
steps ~2 TB per eye. Robot datasets store frames lossily: LeRobot as AV1
video, after checking that policies trained on it match those trained on raw
images; Open X-Embodiment as JPEG. Measured on one H1 episode (300 frames):

| Format | KB / frame | Mean abs. error (/255) | Decode 300 frames |
|---|---|---|---|
| PNG / lossless WebP | 61 / 44 | 0 | — |
| JPEG q95 / q90 | 20 / 14 | 1.6 / 2.0 | — |
| **H.264 CRF 18, `yuv444p`, keyframe every 20** | **2.5** | **1.6** | **0.08 s** |
| H.264 CRF 18, `yuv420p` | 2.2 | 2.3 | 0.06 s |
| AV1 CRF 30, `yuv420p`, keyframe every 2 (LeRobot) | 3.0 | 2.4 | 0.35 s |

`rgb` and `privileged/rgb_right` are stored as H.264 inside the episode file
(`rgb.mp4`), with full-resolution colour so small objects keep their colour.
Encoding is single-threaded with the bundled ffmpeg, so the same frames give
the same bytes. `Episode.load` decodes them; `content_hash` still hashes the
frames as rendered, before encoding. The H1 sample takes **7.4 KB per step**
with both eyes (17 MB for 2,400 steps; 15 MB at 64 px before), so ~20M steps
are ~150 GB.

## Costs

| | 64 px (before) | 256 px (now) |
|---|---|---|
| H1 seated recording, one process on the Mac (GPU rendering) | ~27 steps/s | ~21 steps/s |
| of which: two renders per step | ~16 ms | ~26 ms (the same at 64 px: the room and lights cost, not pixels; shadows ~4 ms) |
| Video encoding at save | — | ~8 ms per step (both eyes) |
| Software render, 1 thread (Linux container, llvmpipe), new scene, shadows on | ~215 ms / frame | ~260 ms / frame |
| Same, shadows off | — | ~60–80 ms / frame |

**Image size barely changes rendering cost; geometry does.** The H1 scene
draws ~400k mesh triangles (torso 47k, each pincer base 34k), twice per
frame with shadows. On a GPU (EGL) that is a few milliseconds; in software
it is ~260 ms per frame, ~2,900 core-hours for 20M steps. The plan is
to generate data on the GPU box that trains, so rendering runs on its
GPU via `MUJOCO_GL=egl`. On a CPU-only host, turn shadows off (~4× faster)
or decimate the visual meshes.

## Known issues

- On some episodes a low key light leaves faint horizontal shadow streaks on
  a wall (shadow acne at grazing angles).
- EGL rendering on the rented GPU box is not yet measured; check it on the
  first rental before generating D0 at scale.
- Stereo (feeding `rgb_right`) is a later ablation.
