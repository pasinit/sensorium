"""An opt-in on-disk cache of decoded model inputs, so the training loader does not decode video.

Decoding an episode's 600 stored 256 px frames (mp4) costs most of a training step on CPU
(``docs/model.md``, B-scale-0: 1.5 CPU-s per step against 0.12 with every episode in memory), and
the per-worker LRU cache of ``SequenceDataset`` cannot hold a pool of hundreds of episodes. This
module writes each episode's ``model_inputs`` once as plain ``.npy`` files, which the loader then
memory-maps: the operating system's page cache is shared by every process on the machine, and a
sequence reads only the frames it uses.

Frames can be stored downscaled (``size``): the loader then takes the same random crop, at the
same position of the original grid, from the downscale (``vision.train_view(source_side=...)``),
so the random stream is unchanged and only the resampling differs. ``size=None`` keeps the stored
side: batches are then identical to decoding (tested). Only ``model_inputs`` are written, so the
privileged state and text never enter the cache.

    uv run python -m sensorium.model.frame_cache <out_dir> <episode.npz>... [--size 128] [--jobs 8]
"""

import argparse
import json
import logging
import os
import shutil
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from sensorium.episode import Episode, model_inputs
from sensorium.model.data import Loaded, header_of
from sensorium.model.layout import INPUT_KEY, ChannelSpace, Header, check_senses
from sensorium.vision import resize

log = logging.getLogger("sensorium.model.frame_cache")
META = "meta.json"


def entry(root: Path, path: Path) -> Path:
    """The cache directory of one episode file."""
    return root / path.stem


def write_episode(path: Path, root: Path, size: int | None) -> Path:
    """Decode one episode and write its model inputs under ``root`` (atomically; skipped if
    present). ``size`` downscales the square frames (the whole frame, no crop)."""
    out = entry(root, path)
    if (out / META).exists():
        return out
    episode = Episode.load(path, privileged=False)
    inputs = model_inputs(episode)
    side = int(inputs["rgb"].shape[1]) if inputs["rgb"].size else 0
    if size is not None and side and size != side:
        inputs["rgb"] = resize(inputs["rgb"], size)
    tmp = root / f".{path.stem}.{os.getpid()}.tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    for key, value in inputs.items():
        np.save(tmp / f"{key}.npy", value)
    meta = {"header": asdict(header_of(episode)), "source_side": side, "source": str(path)}
    (tmp / META).write_text(json.dumps(meta))
    try:
        tmp.rename(out)
    except OSError:  # another process wrote it first
        shutil.rmtree(tmp)
    return out


def build_cache(paths: Sequence[Path], root: Path, size: int | None, jobs: int = 1) -> None:
    """Write every episode of ``paths`` into the cache at ``root`` (``jobs`` processes)."""
    root.mkdir(parents=True, exist_ok=True)
    if jobs <= 1:
        for path in paths:
            write_episode(Path(path), root, size)
        return
    with ProcessPoolExecutor(jobs) as pool:
        n = len(paths)
        for i, _ in enumerate(pool.map(write_episode, paths, [root] * n, [size] * n)):
            if (i + 1) % 50 == 0:
                log.info("cached %d / %d episodes", i + 1, len(paths))


def load_cached(path: Path, root: Path, space: ChannelSpace, senses: Sequence[str]) -> Loaded:
    """The ``Loaded`` episode of ``data.load_episode``, read from the cache (arrays memory-mapped).

    Raises:
        FileNotFoundError: if the episode is not in the cache (build it first).
    """
    out = entry(root, path)
    if not (out / META).exists():
        raise FileNotFoundError(f"{path.name} is not in the frame cache {root}")
    meta = json.loads((out / META).read_text())
    raw = meta["header"]
    header = Header(
        body=raw["body"],
        dt=raw["dt"],
        proprio_names=tuple(raw["proprio_names"]),
        touch_names=tuple(raw["touch_names"]),
        action_names=tuple(raw["action_names"]),
    )
    keys = sorted({INPUT_KEY[s] for s in check_senses(senses)} | {"action"})
    inputs = {k: np.load(out / f"{k}.npy", mmap_mode="r") for k in keys}
    source = int(meta["source_side"])
    stored = int(inputs["rgb"].shape[1]) if "rgb" in inputs and inputs["rgb"].size else source
    return Loaded(header, inputs, space.columns(header), source if stored != source else None)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--size", type=int, default=None, help="frame side (default: as stored)")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    build_cache(args.paths, args.out, args.size, args.jobs)
    log.info("cache %s: %d episodes", args.out, len(list(args.out.glob(f"*/{META}"))))


if __name__ == "__main__":
    main()
