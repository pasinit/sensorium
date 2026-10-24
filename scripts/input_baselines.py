"""Input-only baselines in the same table as the encoders (B-steps-0, step 1).

The ridge probe of the battery (leave-one-episode-out, the 20 held-out random-play episodes of
``runs/decisive/split.json``, windows every 8 steps) on **raw proprio** channels, without any
encoder, so that "what proprio gives linearly" can be read against the random-init floor
(V+P 0.22) and the trained encoders:

* each proprio group alone (body, vestibular, effector), the last observation of the window;
* body + vestibular, and all proprio (last observation);
* all proprio over the whole window (the K = 4 observations stacked: velocities included).

The PCA-of-raw-inputs baselines of B-ctrl-0 (raw vision thumbnail + proprio, raw non-visual inputs)
are on the same 20 episodes and the same windows; their numbers are quoted from ``docs/model.md``.

    uv run python scripts/input_baselines.py --out .tmp/phd-b/input_baselines.md
"""

import argparse
import json
from pathlib import Path

import numpy as np

from sensorium.episode import Episode
from sensorium.model.controls import build_probe_data
from sensorium.model.data import build, load_episode
from sensorium.model.pretrain import build_space, load_config
from sensorium.model.probes import predict_probes

DATA = Path("data/probe_test/humanoid")
SPLIT = Path("runs/decisive/split.json")
STRIDE = 8
GROUPS = ("proprio_body", "vestibular", "proprio_effector")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(Path("configs/model/cpu_tiny_a0.toml"))
    split = json.loads(SPLIT.read_text())
    space = build_space([DATA / n for n in split["pretrain"]], cfg.data.stats_episodes)
    k = cfg.model.substeps

    def features(episode: Episode, ends: np.ndarray) -> dict[str, np.ndarray]:
        arrays = build(load_episode(episode, space, GROUPS), space, GROUPS, ends, k)
        last = {g: arrays[g][:, -1] for g in GROUPS}
        return {
            "proprio_body": last["proprio_body"],
            "vestibular": last["vestibular"],
            "proprio_effector": last["proprio_effector"],
            "body + vestibular": np.concatenate([last["proprio_body"], last["vestibular"]], 1),
            "all proprio (last obs)": np.concatenate([last[g] for g in GROUPS], 1),
            "all proprio (K = 4 stacked)": np.concatenate(
                [arrays[g].reshape(len(ends), -1) for g in GROUPS], 1
            ),
        }

    data = build_probe_data([DATA / n for n in split["heldout"]], k, STRIDE, features)
    lines = [
        "| raw input (no encoder) | channels | probe mean | contact L | contact R | contact obj |"
        " force R2 | tip R2 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, x in data.features.items():
        s = predict_probes(data, name).scores()
        lines.append(
            f"| {name} | {x.shape[1]} | {s['mean']:.2f} | {s['contact_left_auroc']:.2f} | "
            f"{s['contact_right_auroc']:.2f} | {s['contact_object_auroc']:.2f} | "
            f"{s['log_force_r2']:.2f} | {s['tip_r2']:.2f} |"
        )
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))  # noqa: T201


if __name__ == "__main__":
    main()
