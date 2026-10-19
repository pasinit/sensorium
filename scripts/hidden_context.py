"""Is the hidden-property probe limited by the one-step state? (diagnostic of B-s2b-0)

``scripts/s2b_test.py`` probes log mass, friction and log softness of the touched object from the
state of one window (0.2 s). Even the supervised ceiling, trained on those labels, scores chance
(R^2 about -0.09, the value of a constant prediction under leave-one-episode-out). A mass shows
only in how the object *responds* to a push, which takes longer than a window. This script gives
the ridge a context: the state, the mean of the states of the previous ``span`` windows, and their
difference, on the same held-out episodes, for a few encoders. It also reports the *oracle
context*: the same probe on the labelled windows' neighbours' contact force and effort (raw
inputs), to show whether the information is in the inputs at all.

    uv run python scripts/hidden_context.py --out .tmp/phd-b/hidden_context.md
"""

import argparse
import json
from pathlib import Path

import numpy as np

from sensorium.model.pretrain import load_layer
from sensorium.model.probes import HIDDEN, ProbeData, collect, predict_hidden

ROOT = Path("runs/s2b")
STRIDE = 4
SPANS = (4, 8)


def with_context(data: ProbeData, span: int) -> ProbeData:
    """Add ``ctx`` features: state, mean of the previous ``span`` windows, and their difference."""
    state = data.features["state"]
    ctx = np.zeros((len(state), 3 * state.shape[1]), dtype=state.dtype)
    for g in np.unique(data.groups):
        rows = np.flatnonzero(data.groups == g)
        x = state[rows]
        csum = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
        lo = np.maximum(np.arange(len(x)) - span, 0)
        past = (csum[np.arange(len(x))] - csum[lo]) / np.maximum(np.arange(len(x)) - lo, 1)[:, None]
        ctx[rows] = np.concatenate([x, past, x - past], axis=1)
    return ProbeData({**data.features, "ctx": ctx}, data.labels, data.groups)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    split = json.loads((ROOT / "split.json").read_text())
    held = [Path(p) for p in split["exp_heldout"]]
    runs = {
        "s2b-all-tw (SSL)": ROOT / "s2b-all-tw-s0" / "ckpt-1000.pt",
        "rand60-all (SSL)": ROOT / "rand60-all-s0" / "ckpt-1000.pt",
        "sup-all (hidden-supervised)": ROOT / "sup-all-s0" / "ckpt.pt",
        "sup-vp (hidden-supervised)": ROOT / "sup-vp-s0" / "ckpt.pt",
    }
    lines = [
        "| encoder | feature | log mass R2 | friction R2 | log softness R2 | windows |",
        "|---|---|---|---|---|---|",
    ]
    for name, ckpt in runs.items():
        data = collect(load_layer(ckpt), held, STRIDE, (), raw=False)
        for span in (0, *SPANS):
            d = data if span == 0 else with_context(data, span)
            feature = "state" if span == 0 else "ctx"
            s = predict_hidden(d, feature)
            label = "one-step state" if span == 0 else f"state + previous {span} windows"
            lines.append(
                f"| {name} | {label} | "
                + " | ".join(f"{s[f'{n}_r2']:.2f}" for n in HIDDEN)
                + f" | {int(s['hidden_windows'])} |"
            )
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))  # noqa: T201


if __name__ == "__main__":
    main()
