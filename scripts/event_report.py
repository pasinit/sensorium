"""Event-rate report of generated seated episodes: a table, a histogram and a frame strip.

Usage: uv run python scripts/event_report.py <out_dir> <name>=<glob> [<name>=<glob> ...]

Each ``<glob>`` matches episode files (``.../humanoid/*.npz``) of one set. Prints per set: episodes,
the share of 1 s windows with an event (``sensorium.event_rate``) and at rest, onsets per minute
by type, contact share, lifts per episode, the arms' waiting share, the director's interventions
per episode and the recording time per episode (``meta.wall_seconds``). Writes
``event_hist.png`` (per-window onset counts and per-episode eventful shares, one colour per set)
and ``event_strip.png`` (head-camera frames around the first drop onto a hand of the last set that
has one).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sensorium.babble_stats import a1_stats  # noqa: E402
from sensorium.bodies.h1_director import INTERVENTIONS  # noqa: E402
from sensorium.episode import Episode  # noqa: E402
from sensorium.event_rate import EVENT_TYPES, EventRate, event_rate  # noqa: E402

COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # categorical slots 1-4, fixed order


def measure(paths: list[str]) -> tuple[list[EventRate], list[dict[str, float]], str | None]:
    """Event rates and other statistics of the episodes, and a file with a drop onto a hand."""
    rates, stats, with_drop = [], [], None
    for path in paths:
        episode = Episode.load(path)
        rates.append(event_rate(episode))
        a1 = a1_stats(episode)
        codes = episode.privileged.get("intervention")
        drops = 0 if codes is None else int((codes > 0).sum())
        if codes is not None and with_drop is None and (codes == 3).any():
            with_drop = path
        seconds = episode.meta.get("wall_seconds")
        stats.append(
            {
                "contact": a1.object_contact_rate,
                "lifts": float(a1.lifts),
                "wait": a1.wait_share,
                "interventions": float(drops),
                "seconds": float(seconds) if isinstance(seconds, int | float) else float("nan"),
            }
        )
    return rates, stats, with_drop


def print_table(results: dict[str, tuple[list[EventRate], list[dict[str, float]]]]) -> None:
    """One markdown row per set."""
    print(
        "| set | episodes | eventful | rest | "
        + " | ".join(t.replace("_", " ") for t in EVENT_TYPES)
        + " | contact | lifts/ep | waiting | interventions/ep | s/episode |"
    )
    print("|---" * (len(EVENT_TYPES) + 10) + "|")
    for name, (rates, stats) in results.items():
        eventful = np.mean([r.eventful for r in rates])
        per_minute = [np.mean([r.per_minute[t] for r in rates]) for t in EVENT_TYPES]
        cells = [f"{np.nanmean([s[k] for s in stats]):.2f}" for k in ("contact", "lifts", "wait")]
        extra = [f"{np.nanmean([s[k] for s in stats]):.1f}" for k in ("interventions", "seconds")]
        print(
            f"| {name} | {len(rates)} | {eventful:.2f} | {1 - eventful:.2f} | "
            + " | ".join(f"{v:.1f}" for v in per_minute)
            + " | "
            + " | ".join(cells + extra)
            + " |"
        )


def plot_histogram(
    results: dict[str, tuple[list[EventRate], list[dict[str, float]]]], path: Path
) -> None:
    """Per-window onset counts and per-episode eventful shares."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.arange(0, 13) - 0.5
    for colour, (name, (rates, _)) in zip(COLOURS, results.items(), strict=False):
        counts = np.concatenate([r.windows for r in rates])
        left.hist(
            np.minimum(counts, 11),
            bins=bins,
            density=True,
            histtype="step",
            lw=2,
            color=colour,
            label=name,
        )
        right.hist(
            [r.eventful for r in rates],
            bins=np.linspace(0, 1, 21),
            histtype="step",
            lw=2,
            color=colour,
            label=name,
        )
    left.set_xlabel("onsets per 1 s window (11 = 11 or more)")
    left.set_ylabel("share of windows")
    right.set_xlabel("share of an episode's windows with an event")
    right.set_ylabel("episodes")
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
    left.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def plot_strip(path_npz: str, path: Path) -> None:
    """Head-camera frames from just before the first drop onto a hand, every 0.15 s."""
    episode = Episode.load(path_npz)
    codes = episode.privileged["intervention"]
    start = int(np.flatnonzero(codes == INTERVENTIONS.index("drop_hand"))[0])
    frames = [episode.rgb[min(start + k, episode.length - 1)] for k in range(-2, 22, 3)]
    fig, axes = plt.subplots(1, len(frames), figsize=(2 * len(frames), 2.3))
    for k, (ax, frame) in enumerate(zip(axes, frames, strict=True)):
        ax.imshow(frame)
        ax.set_title(f"{(-2 + 3 * k) * episode.dt:+.2f} s", fontsize=8)
        ax.axis("off")
    fig.suptitle("An object dropped onto a hand (head camera, t = 0 at the drop)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main() -> None:
    """Measure every set, print the table, write the figures."""
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    results, strip = {}, None
    for item in sys.argv[2:]:
        name, pattern = item.split("=", 1)
        folder, _, files = pattern.rpartition("/")
        rates, stats, with_drop = measure(sorted(str(p) for p in Path(folder).glob(files)))
        results[name] = (rates, stats)
        strip = with_drop or strip
    print_table(results)
    plot_histogram(results, out / "event_hist.png")
    if strip is not None:
        plot_strip(strip, out / "event_strip.png")
    print(f"wrote {out / 'event_hist.png'}" + (f" and {out / 'event_strip.png'}" if strip else ""))


if __name__ == "__main__":
    main()
