"""Effective rank and std of the held-out states of existing checkpoints (B-s2b-0 retrofit).

The s2b and rand60 runs of ``scripts/s2b_test.py`` logged the probe score of every checkpoint but
not the geometry of their states. This encodes the held-out experiment windows (``exp``, stride 8)
with every saved checkpoint and with the random-init floors (step 0), and writes the effective rank
(``exp`` of the entropy of the normalised singular values of the centred states) and the mean
per-dimension std to ``runs/s2b/geometry/<run>-n<step>.json``. If the rank falls while the probe
score falls, the objective is removing information rather than the data lacking it.

    uv run python scripts/collapse_retrofit.py
    uv run python scripts/collapse_retrofit.py --report
"""

import argparse
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from sensorium.episode import Episode
from sensorium.model.data import LoaderConfig
from sensorium.model.diagnostics import state_geometry
from sensorium.model.pretrain import load_layer
from sensorium.model.sensory_layer import SensoryLayer, SensoryModel

sys.path.insert(0, str(Path(__file__).parent))
import objective_test as objective  # noqa: E402
import s2b_test as study  # noqa: E402

OUT = study.ROOT / "geometry"
STRIDE = 8


def geometry(layer: SensoryLayer) -> dict[str, float]:
    """Geometry of the layer's states on the held-out experiment windows (natural inputs)."""
    states = []
    for path in study.paths("exp_heldout"):
        episode = Episode.load(path, privileged=False)
        states.append(layer.encode(episode, stride=STRIDE)[0])
    return state_geometry(np.concatenate(states))


def retrofit() -> None:
    """Compute the geometry of every checkpoint and floor that has no file yet."""
    OUT.mkdir(parents=True, exist_ok=True)
    base = study.tiny()
    for tag, (senses, drop, seeds) in study.FLOORS.items():
        space = study.shared_space(drop)
        for seed in range(seeds):
            target = OUT / f"{tag}-s{seed}-n0.json"
            if target.exists():
                continue
            torch.manual_seed(seed)
            model = replace(base.model, senses=senses or base.model.senses)
            loader = LoaderConfig(
                steps=model.steps, substeps=model.substeps, view_size=model.vision_size
            )
            layer = SensoryLayer(SensoryModel(model, space), space, loader)
            target.write_text(json.dumps(geometry(layer)))
    for run in sorted(study.ROOT.glob("*-s?")):
        if run.name.startswith(("ft-", "sup-")):
            continue
        for step in study.CKPTS:
            target = OUT / f"{run.name}-n{step}.json"
            ckpt = run / f"ckpt-{step}.pt"
            if ckpt.exists() and not target.exists():
                target.write_text(json.dumps(geometry(load_layer(ckpt))))
                print("geometry", target.name, flush=True)  # noqa: T201


def retrofit_objective() -> None:
    """Geometry of the all-sense ``a-*`` checkpoints of ``objective_test.py`` (random-play set)."""
    out = objective.ROOT / "geometry"
    out.mkdir(parents=True, exist_ok=True)
    for run in sorted(objective.ROOT.glob("a-*-s?")):
        for step in objective.CKPTS:
            target, ckpt = out / f"{run.name}-n{step}.json", run / f"ckpt-{step}.pt"
            if not ckpt.exists() or target.exists():
                continue
            layer = load_layer(ckpt)
            states = [
                layer.encode(Episode.load(p, privileged=False), stride=STRIDE)[0]
                for p in objective.files("heldout")
            ]
            target.write_text(json.dumps(state_geometry(np.concatenate(states))))
            print("geometry", target.name, flush=True)  # noqa: T201


def report() -> None:
    """Effective rank and std against steps (mean over seeds), step 0 = the floor."""
    files = {p.stem: json.loads(p.read_text()) for p in OUT.glob("*.json")}
    steps = (0, *study.CKPTS)
    for term in ("erank", "std"):
        print(
            f"\n| arm ({term}, mean over seeds) | " + " | ".join(f"{s} steps" for s in steps) + " |"
        )  # noqa: T201
        print("|---|" + "---|" * len(steps))  # noqa: T201
        for arm in study.ARMS:
            floor = (
                "floor-" + ("vp" if "-vp" in arm else "all") + ("-noeff" if "noeff" in arm else "")
            )
            cells = []
            for step in steps:
                name = floor if step == 0 else arm
                vals = [
                    v[term]
                    for k, v in files.items()
                    if (m := re.fullmatch(r"(.+)-s(\d+)-n(\d+)", k))
                    and m[1] == name
                    and int(m[3]) == step
                ]
                cells.append(f"{np.mean(vals):.1f}" if vals else "-")
            print(f"| {arm} | " + " | ".join(cells) + " |")  # noqa: T201


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--objective", action="store_true", help="the a-* arms of objective_test")
    args = parser.parse_args()
    if args.objective:
        retrofit_objective()
    elif args.report:
        report()
    else:
        retrofit()


if __name__ == "__main__":
    main()
