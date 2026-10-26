"""Convert LeWorldModel's Push-T dataset (``sensorium.pusht``): episodes, LeWM's 64 px copy, split.

Usage (``<src>`` is ``pusht_expert_train.h5`` from ``quentinll/lewm-pusht``, after zstd):
    uv run python scripts/pusht_convert.py stats <src> <stats.json>
    uv run python scripts/pusht_convert.py episodes <src> <out_dir> --scale S [--shard I/N]
    uv run python scripts/pusht_convert.py lewm <src> <out_dir> <part.h5> [--shard I/N]
    uv run python scripts/pusht_convert.py merge <dst.h5> <part.h5> ...
    uv run python scripts/pusht_convert.py split <src> <split.json>
    uv run python scripts/pusht_convert.py report <src> <out_dir> <stats.json> <fig_dir>

- ``stats``: episode lengths and the ranges of action, proprio and state (no pixels read). It
  also gives ``action_scale``, the largest |action| (at least 1).
- ``episodes``: one ``out_dir/pusht/pusht-NNNNN.npz`` per source episode (``sensorium.pusht``).
  A shard is a contiguous range of episodes; files already there are skipped, so a stopped run
  resumes.
- ``lewm``: a shard of LeWM's 64 px HDF5. The pixels are the converted episodes' frames resized
  as our loader does (``pusht.lewm_frames``); action, proprio and state are the source's,
  unscaled. ``merge`` joins the shards in episode order.
- ``split``: LeWM's train/validation clips (``pusht.clip_split``) and an episode-level 90/10 split
  (same seed), as JSON.
- ``report``: a length histogram, a 224 vs 64 px frame strip of three episodes, and a markdown
  table of counts and ranges.
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sensorium import pusht  # noqa: E402
from sensorium.episode import Episode  # noqa: E402


def shard_range(n: int, shard: str) -> range:
    """Episodes of shard ``"I/N"`` out of ``n``: the I-th of N contiguous ranges."""
    i, k = (int(x) for x in shard.split("/"))
    if not 0 <= i < k:
        raise ValueError(f"shard {shard}: want 0 <= I < N")
    return range(i * n // k, (i + 1) * n // k)


def episode_path(out_dir: Path, index: int) -> Path:
    """Where episode ``index`` is written."""
    return out_dir / pusht.BODY / f"{pusht.BODY}-{index:05d}.npz"


def ranges(values: np.ndarray, names: list[str]) -> dict[str, dict[str, float]]:
    """Per column: min, 1st and 99th percentile, max."""
    out = {}
    for k, name in enumerate(names):
        column = values[:, k]
        p1, p99 = np.percentile(column, [1, 99])
        out[name] = {
            "min": float(column.min()),
            "p1": float(p1),
            "p99": float(p99),
            "max": float(column.max()),
        }
    return out


def stats(src: Path, out: Path) -> None:
    """Write the episode lengths and column ranges."""
    with h5py.File(src, "r") as f:
        lengths, _ = pusht.episode_index(f)
        action, proprio, state = (
            np.asarray(f[c][:], dtype=np.float64) for c in ("action", "proprio", "state")
        )
        shape = list(f["pixels"].shape)
    result = {
        "episodes": len(lengths),
        "steps": int(lengths.sum()),
        "pixels_shape": shape,
        "lengths": lengths.tolist(),
        "action": ranges(action.reshape(len(action), -1), pusht.ACTION_NAMES),
        "proprio": ranges(proprio, pusht.PROPRIO_NAMES),
        "state": ranges(state, pusht.STATE_NAMES),
        "action_scale": max(1.0, float(np.abs(action).max())),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result))
    print(f"{result['episodes']} episodes, {result['steps']} steps, scale {result['action_scale']}")


def episodes(src: Path, out_dir: Path, scale: float, shard: str, revision: str) -> None:
    """Write this shard's episodes."""
    with h5py.File(src, "r") as f:
        n = len(pusht.episode_index(f)[0])
        todo = shard_range(n, shard)
        for k, index in enumerate(todo):
            path = episode_path(out_dir, index)
            if path.exists():
                continue
            pusht.to_episode(index, pusht.read_episode(f, index), scale, revision).save(path)
            if k % 50 == 0:
                print(f"shard {shard}: {k + 1}/{len(todo)} (episode {index})", flush=True)
    print(f"shard {shard}: done, {len(todo)} episodes", flush=True)


def lewm(src: Path, out_dir: Path, part: Path, shard: str) -> None:
    """Write this shard of LeWM's 64 px HDF5."""

    def items(f: h5py.File, todo: range):  # type: ignore[no-untyped-def]
        for k, index in enumerate(todo):
            arrays = pusht.read_episode(f, index)
            ep = Episode.load(episode_path(out_dir, index), privileged=False)
            arrays["pixels"] = pusht.lewm_frames(ep)
            if k % 50 == 0:
                print(f"lewm {shard}: {k + 1}/{len(todo)}", flush=True)
            yield arrays

    with h5py.File(src, "r") as f:
        todo = shard_range(len(pusht.episode_index(f)[0]), shard)
        count = pusht.write_lewm_h5(part, items(f, todo))
    print(f"lewm {shard}: done, {count} episodes", flush=True)


def merge(dst: Path, parts: list[Path]) -> None:
    """Join shards of LeWM's HDF5, in the given order."""

    def items():  # type: ignore[no-untyped-def]
        for part in parts:
            with h5py.File(part, "r") as f:
                for index in range(len(pusht.episode_index(f)[0])):
                    yield pusht.read_episode(f, index)

    print(f"merged {pusht.write_lewm_h5(dst, items())} episodes into {dst}", flush=True)


def split(src: Path, out: Path) -> None:
    """Write LeWM's clip split and an episode-level split."""
    with h5py.File(src, "r") as f:
        lengths = pusht.episode_index(f)[0].tolist()
    clips = pusht.clip_split(lengths)
    order = np.random.default_rng(pusht.LEWM_SEED).permutation(len(lengths))
    n_train = round(pusht.LEWM_TRAIN * len(lengths))
    result = {
        "lewm_clips": {k: [list(c) for c in v] for k, v in clips.items()},
        "lewm_clip_settings": {
            "num_steps": pusht.LEWM_NUM_STEPS,
            "frameskip": pusht.LEWM_FRAMESKIP,
            "train": pusht.LEWM_TRAIN,
            "seed": pusht.LEWM_SEED,
        },
        "episodes": {
            "train": sorted(int(i) for i in order[:n_train]),
            "val": sorted(int(i) for i in order[n_train:]),
        },
    }
    out.write_text(json.dumps(result))
    print(
        f"clips train {len(clips['train'])} val {len(clips['val'])};",
        f"episodes {n_train} / {len(lengths) - n_train}",
    )


def report(src: Path, out_dir: Path, stats_path: Path, fig_dir: Path) -> None:
    """The data report: a length histogram, a 224 vs 64 px strip and a markdown table."""
    info = json.loads(stats_path.read_text())
    fig_dir.mkdir(parents=True, exist_ok=True)
    lengths = np.asarray(info["lengths"])
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(lengths, bins=40, color="#2a78d6")
    ax.set_xlabel("episode length (steps, 10 Hz)")
    ax.set_ylabel("episodes")
    ax.set_title(f"Push-T: {len(lengths)} episodes, {lengths.sum()} steps")
    fig.tight_layout()
    fig.savefig(fig_dir / "pusht_lengths.png", dpi=130)
    rows = []
    with h5py.File(src, "r") as f:
        n = len(pusht.episode_index(f)[0])
        for index in (0, n // 2, n - 1):
            source = pusht.read_episode(f, index)["pixels"]
            ep = Episode.load(episode_path(out_dir, index), privileged=False)
            small = pusht.lewm_frames(ep)
            picks = np.linspace(0, len(source) - 1, 8).astype(int)
            top = np.concatenate(list(source[picks]), axis=1)
            big = np.repeat(np.repeat(small[picks], 3, axis=1), 3, axis=2)  # 64 -> 192 px, nearest
            pad = np.full((len(picks), 192, pusht.SOURCE_SIZE - 192, 3), 255, np.uint8)
            bottom = np.concatenate(list(np.concatenate([big, pad], axis=2)), axis=1)
            rows += [top, bottom]
    plt.imsave(fig_dir / "pusht_strip_224_vs_64.png", np.concatenate(rows, axis=0))
    lines = [
        f"episodes {info['episodes']}, steps {info['steps']}, source pixels {info['pixels_shape']}",
        f"length: min {lengths.min()}, median {int(np.median(lengths))}, "
        f"mean {lengths.mean():.1f}, max {lengths.max()}",
        f"action scale (max |a|, at least 1): {info['action_scale']:.4g}",
        "",
        "| column | min | p1 | p99 | max |",
        "|---|---|---|---|---|",
    ]
    for group in ("action", "proprio", "state"):
        for name, r in info[group].items():
            cells = " | ".join(f"{r[k]:.4g}" for k in ("min", "p1", "p99", "max"))
            lines.append(f"| {group}/{name} | {cells} |")
    (fig_dir / "pusht_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    """Dispatch the subcommands."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("stats")
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    p = sub.add_parser("episodes")
    p.add_argument("src", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--scale", type=float, required=True)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--revision", default="")
    p = sub.add_parser("lewm")
    p.add_argument("src", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("part", type=Path)
    p.add_argument("--shard", default="0/1")
    p = sub.add_parser("merge")
    p.add_argument("dst", type=Path)
    p.add_argument("parts", type=Path, nargs="+")
    p = sub.add_parser("split")
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    p = sub.add_parser("report")
    p.add_argument("src", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("stats", type=Path)
    p.add_argument("fig_dir", type=Path)
    args = parser.parse_args()
    if args.command == "stats":
        stats(args.src, args.out)
    elif args.command == "episodes":
        episodes(args.src, args.out_dir, args.scale, args.shard, args.revision)
    elif args.command == "lewm":
        lewm(args.src, args.out_dir, args.part, args.shard)
    elif args.command == "merge":
        merge(args.dst, args.parts)
    elif args.command == "split":
        split(args.src, args.out)
    else:
        report(args.src, args.out_dir, args.stats, args.fig_dir)


if __name__ == "__main__":
    main()
