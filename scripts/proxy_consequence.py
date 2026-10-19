"""Does an input-only change score recover the oracle consequence windows? (offline, free)

The oracle selects training windows from privileged object motion (``probes.consequence_steps``:
a ``conseq`` window has at least 25% of its 32 steps in hand-object contact that moves the object
more than 1 cm within 0.5 s; an ``event`` window is the same share inside a Track A caused-change
event). A *curious* explorer cannot see object motion, only its own senses. This script scores every
32-step window by a signal computed from model inputs only and reports, per threshold (the top
``q`` of windows by score), precision and recall against the oracle, and the lift (precision over
the oracle's base rate):

* ``touch``: mean over the window of the summed absolute step-to-step change of the touch taxels;
* ``vision``: mean absolute difference of consecutive 64 px grayscale frames;
* ``both``: the two ranked scores summed.

    uv run python scripts/proxy_consequence.py --out .tmp/phd-b/proxy_consequence.md
"""

import argparse
import json
from pathlib import Path

import numpy as np

from sensorium.episode import Episode
from sensorium.model.probes import consequence_steps

DATA = Path("data/probe_test/humanoid")
SPLIT = Path("runs/decisive/split.json")
SPAN = 32
QUANTILES = (0.05, 0.10, 0.20, 0.40)


def window_mean(x: np.ndarray, span: int = SPAN) -> np.ndarray:
    """Mean of ``x`` over every window of ``span`` steps (``len - span + 1`` values)."""
    return np.convolve(x, np.ones(span), "valid") / span


def input_scores(path: Path) -> dict[str, np.ndarray]:
    """Window scores from model inputs only (the episode is loaded without privileged state)."""
    episode = Episode.load(path, privileged=False)
    touch = np.abs(np.diff(np.asarray(episode.touch, dtype=np.float32), axis=0)).sum(axis=1)
    frames = np.asarray(episode.rgb)[:, ::4, ::4].astype(np.float32).mean(axis=-1)
    vision = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    return {
        "touch": window_mean(np.concatenate([[0.0], touch])),
        "vision": window_mean(np.concatenate([[0.0], vision])),
    }


def oracle(path: Path) -> dict[str, np.ndarray]:
    """Oracle window flags (``conseq``, ``event``): at least 25% of the window's steps."""
    flags = consequence_steps(Episode.load(path))
    return {k: window_mean(flags[k].astype(float)) >= 0.25 for k in ("consequence", "event")}


def rank(x: np.ndarray) -> np.ndarray:
    """Ranks scaled to ``[0, 1]`` (the pooled windows of all episodes)."""
    return np.argsort(np.argsort(x)) / max(len(x) - 1, 1)


def evaluate(name: str, paths: list[Path]) -> list[str]:
    """Markdown rows for one data set."""
    scores: dict[str, list[np.ndarray]] = {"touch": [], "vision": []}
    truth: dict[str, list[np.ndarray]] = {"consequence": [], "event": []}
    for path in paths:
        for key, value in input_scores(path).items():
            scores[key].append(value)
        for key, value in oracle(path).items():
            truth[key].append(value)
    pooled = {k: np.concatenate(v) for k, v in scores.items()}
    pooled["both"] = rank(pooled["touch"]) + rank(pooled["vision"])
    rows = []
    for label, flags in truth.items():
        positive = np.concatenate(flags)
        base = positive.mean()
        for key, score in pooled.items():
            for q in QUANTILES:
                chosen = score >= np.quantile(score, 1 - q)
                hit = (chosen & positive).sum()
                precision = hit / max(chosen.sum(), 1)
                recall = hit / max(positive.sum(), 1)
                rows.append(
                    f"| {name} | {label} (base {base:.2f}) | {key} | top {int(q * 100)}% | "
                    f"{precision:.2f} | {recall:.2f} | {precision / max(base, 1e-9):.2f} |"
                )
    return rows


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=24, help="episodes per data set")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    rand = [DATA / n for n in json.loads(SPLIT.read_text())["pretrain"] if "hand" in n]
    exp = sorted(args.exp_dir.glob("*hand_*.npz"))
    sets = {
        "random play (hands)": [
            rand[i] for i in sorted(rng.choice(len(rand), args.episodes, replace=False))
        ],
        "experiments (hands)": [
            exp[i] for i in sorted(rng.choice(len(exp), args.episodes, replace=False))
        ],
    }
    lines = [
        "| data | oracle | score | selected | precision | recall | lift |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, paths in sets.items():
        lines += evaluate(name, paths)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))  # noqa: T201


if __name__ == "__main__":
    main()
