"""Measure a model configuration: parameters, tokens, FLOPs, CPU speed and loader throughput.

Usage:
    make measure
    uv run python -m sensorium.model.measure <config.toml> --paths '<episode glob>'

Prints a markdown table. FLOPs are counted by ``torch.utils.flop_counter`` on one
training sequence (forward + backward, 2 FLOPs per multiply-add). Speeds are for the
machine it runs on; the GPU numbers in ``docs/model.md`` are estimates from the FLOPs.
"""

import argparse
import logging
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

from sensorium.model.data import SequenceDataset
from sensorium.model.pretrain import build_space, expand, load_config, resolve_data
from sensorium.model.sensory_layer import SensoryModel, count_parameters

log = logging.getLogger("sensorium.model.measure")


def training_flops(model: SensoryModel, batch: dict[str, torch.Tensor]) -> float:
    """FLOPs of one forward + backward pass on ``batch`` (divide by its batch size)."""
    with FlopCounterMode(display=False) as counter:
        model(batch)["loss"].backward()
    return float(counter.get_total_flops())


def loader_throughput(dataset: SequenceDataset, n: int, seed: int = 0) -> tuple[float, float]:
    """Sequences per second of the loader in this process: ``(stream, warm)``.

    ``stream`` = the training iterator from an empty cache (episodes are decoded as they
    are picked, ``cfg.reuse`` sequences per pick); ``warm`` = random sequences with every
    episode already cached (cutting and augmenting only).
    """
    dataset._cache.clear()  # noqa: SLF001
    stream = iter(dataset)
    start = time.perf_counter()
    for _ in range(n):
        next(stream)
    cold = n / (time.perf_counter() - start)
    rng = np.random.default_rng(seed)
    for p in dataset.paths:
        dataset.load(p)
    start = time.perf_counter()
    for _ in range(n):
        dataset.sample(rng)
    return cold, n / (time.perf_counter() - start)


def measure(config: Path, paths: Sequence[str], batch_size: int, steps: int) -> dict[str, float]:
    """Run the measurements for one config on some episode files."""
    cfg = load_config(config)
    files, senses = (
        resolve_data(cfg)
        if not paths
        else (
            [f for p in paths for f in expand(p)],
            cfg.model.senses,
        )
    )
    space = build_space(files, 16)
    model = SensoryModel(cfg.model, space)
    counts = count_parameters(model)
    dataset = SequenceDataset(files, space, senses, cfg.loader(), seed=0)
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    flops = training_flops(model, batch)
    cold, warm = loader_throughput(dataset, 16)
    loader = iter(DataLoader(dataset, batch_size=batch_size))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    next_batch = next(loader)
    start = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad()
        model(next_batch)["loss"].backward()
        opt.step()
    step_s = (time.perf_counter() - start) / steps
    out = {
        "parameters_M": counts["total"] / 1e6,
        "tokens_per_step": 1 + sum(model.n_tokens.values()),
        "train_GFLOPs_per_sequence": flops / 1e9,
        "cpu_train_seq_per_s": batch_size / step_s,
        "loader_stream_seq_per_s": cold,
        "loader_warm_seq_per_s": warm,
    }
    for name, count in counts.items():
        log.info("params %-18s %8.2fM", name, count / 1e6)
    return out


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--paths", nargs="*", default=[], help="episode globs (default: the config's)"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for key, value in measure(args.config, args.paths, args.batch_size, args.steps).items():
        log.info("%-28s %.3f", key, value)


if __name__ == "__main__":
    main()
