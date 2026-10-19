"""GPU smoke test of the sensory layer: loader scaling, compute throughput, a timed run.

Prepared on CPU, **not yet run on a GPU**. Run it on one rented GPU box (see
``scripts/modal_gpu_smoke.py`` and ``docs/model.md``):

    uv run python -m sensorium.model.gpu_smoke configs/model/v0.toml \\
        --paths 'data/sample_h1/humanoid/*.npz' --minutes 15 --workers 4 8

Three phases, so a slow run says *why*:

1. **Loader** (before CUDA is touched): batches of ``--batch-size`` from a ``DataLoader`` with
   each ``--workers`` setting, cold (one decoded episode kept: every pick decodes) and, when the
   data fits, warm. This is the decode-bound number a real pool will see.
2. **Compute**: forward + backward + AdamW on one real batch kept on the GPU, fp32 and bf16:
   seconds per step, sequences/s, peak memory, GPU utilisation. Data loading excluded.
3. **Train**: the real loop (``pretrain.train``) for ``--minutes``, bf16, ``--workers[-1]``
   loader workers; logs loss, std, effective rank, sequences/s, mean GPU utilisation and
   memory at each logged step.

Writes ``<out>/report.json`` and prints a markdown summary, including the H100-hours per
run the measured throughput implies for the planned pool (5M env steps = 78k sequences per
epoch). GPU utilisation comes from ``nvidia-smi`` sampled by a background thread.
"""

import argparse
import json
import logging
import os
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sensorium.model.data import SequenceDataset
from sensorium.model.pretrain import (
    DataConfig,
    RunConfig,
    TrainConfig,
    build_space,
    load_config,
    pick_device,
    resolve_data,
    train,
)
from sensorium.model.sensory_layer import SensoryModel

log = logging.getLogger("sensorium.model.gpu_smoke")

SEQUENCES_PER_EPOCH = 5_000_000 // 64
"""Non-overlapping 64-env-step sequences in a pool of 5M env steps (``docs/model.md``)."""
SMI_QUERY = "utilization.gpu,memory.used,power.draw"


def parse_smi(line: str) -> dict[str, float]:
    """Parse one ``nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw`` CSV line.

    Raises:
        ValueError: if the line does not have the three numeric fields.
    """
    util, mem, power = (float(x) for x in line.split(","))
    return {"gpu_util_pct": util, "gpu_mem_used_mib": mem, "gpu_power_w": power}


def read_smi() -> dict[str, float]:
    """One reading of the first GPU, or ``{}`` when ``nvidia-smi`` is missing or fails."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={SMI_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.splitlines()
        return parse_smi(out[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}


class Sampler:
    """Averages a GPU reading taken every ``period`` seconds in a background thread.

    Args:
        read: Function returning one reading (default: ``read_smi``).
        period: Seconds between readings.
    """

    def __init__(
        self, read: Callable[[], dict[str, float]] = read_smi, period: float = 1.0
    ) -> None:
        self.read, self.period = read, period
        self._readings: list[dict[str, float]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.period):
            reading = self.read()
            if reading:
                with self._lock:
                    self._readings.append(reading)

    def __enter__(self) -> "Sampler":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2 * self.period)

    def take(self) -> dict[str, float]:
        """Mean (utilisation, power) and max (memory) of the readings since the last call."""
        with self._lock:
            readings, self._readings = self._readings, []
        if not readings:
            return {}

        def mean(key: str) -> float:
            return sum(r[key] for r in readings) / len(readings)

        return {
            "gpu_util_pct": mean("gpu_util_pct"),
            "gpu_power_w": mean("gpu_power_w"),
            "gpu_mem_used_mib": max(r["gpu_mem_used_mib"] for r in readings),
        }


def loader_phase(
    cfg: RunConfig,
    paths: Sequence[Path],
    senses: Sequence[str],
    workers: Sequence[int],
    seconds: float,
) -> dict[str, dict[str, float]]:
    """Sequences/s of a ``DataLoader`` for each worker count, cold and warm-cache.

    Cold keeps one decoded episode per worker, so (almost) every pick decodes from disk, as a
    real pool much larger than memory would; warm keeps the configured cache.
    """
    space = build_space(paths, cfg.data.stats_episodes)
    out: dict[str, dict[str, float]] = {}
    for n in workers:
        row = {}
        for label, cache in (("cold", 1), ("warm", cfg.augment.cache_episodes)):
            loader_cfg = replace(cfg.loader(), cache_episodes=cache)
            dataset = SequenceDataset(paths, space, senses, loader_cfg, seed=cfg.seed)
            loader = iter(
                DataLoader(
                    dataset,
                    batch_size=cfg.train.batch_size,
                    num_workers=n,
                    prefetch_factor=4 if n else None,
                    persistent_workers=False,
                )
            )
            next(loader)  # start the workers
            start, seqs = time.perf_counter(), 0
            while time.perf_counter() - start < seconds:
                seqs += next(loader)["action"].size(0)
            row[f"{label}_seq_per_s"] = seqs / (time.perf_counter() - start)
            del loader
        out[str(n)] = row
        log.info("loader, %d workers: %s", n, row)
    return out


def compute_phase(
    cfg: RunConfig, paths: Sequence[Path], senses: Sequence[str], device: torch.device, steps: int
) -> dict[str, dict[str, float]]:
    """Seconds per optimiser step on one resident real batch, per precision."""
    space = build_space(paths, cfg.data.stats_episodes)
    dataset = SequenceDataset(paths, space, senses, cfg.loader(), seed=cfg.seed)
    batch = {
        k: v.to(device)
        for k, v in next(iter(DataLoader(dataset, batch_size=cfg.train.batch_size))).items()
    }
    out: dict[str, dict[str, float]] = {}
    for precision in ("fp32", "bf16"):
        model = SensoryModel(cfg.model, space).to(device)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        amp = torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with Sampler() as sampler:
            for i in range(steps + 3):
                if i == 3:
                    _sync(device)
                    start = time.perf_counter()
                    sampler.take()
                with amp:
                    loss = model(batch)["loss"]
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                model.update_target(0.99)
            _sync(device)
            step_s = (time.perf_counter() - start) / steps
            util = sampler.take()
        out[precision] = {
            "step_s": step_s,
            "seq_per_s": cfg.train.batch_size / step_s,
            "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30
            if device.type == "cuda"
            else 0.0,
            **util,
        }
        log.info("compute %s: %s", precision, out[precision])
    return out


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def plan_hours(
    seq_per_s: float, epochs: int, sequences_per_epoch: int = SEQUENCES_PER_EPOCH
) -> float:
    """GPU-hours for ``epochs`` passes over a 5M-step pool at a measured throughput."""
    return epochs * sequences_per_epoch / seq_per_s / 3600.0


def hardware() -> dict[str, object]:
    """What the run ran on."""
    gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu.name if gpu else None,
        "gpu_mem_gib": round(gpu.total_memory / 2**30, 1) if gpu else None,
        "cpu_count": os.cpu_count(),
    }


def smoke(
    cfg: RunConfig,
    minutes: float,
    workers: Sequence[int],
    loader_seconds: float,
    compute_steps: int,
) -> dict[str, object]:
    """Run the three phases and return the report."""
    paths, senses = resolve_data(cfg)
    report: dict[str, object] = {"hardware": hardware(), "config": asdict(cfg.train)}
    report["loader"] = loader_phase(cfg, paths, senses, workers, loader_seconds)
    device = pick_device(cfg.train.device)
    report["compute"] = compute_phase(cfg, paths, senses, device, compute_steps)
    if minutes > 0:
        with Sampler() as sampler:
            tc = replace(cfg.train, max_seconds=60.0 * minutes, num_workers=workers[-1])
            final = train(replace(cfg, train=tc), on_log=lambda _m: sampler.take())
        report["train"] = final
        if device.type == "cuda":
            report["train_peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
    report["plan_hours_per_run"] = _plan(report)
    return report


def _plan(report: dict[str, object]) -> dict[str, float]:
    """H100-hours implied for 20 and 100 epochs: the slower of train, compute and loader."""
    candidates = []
    train_m = report.get("train")
    if isinstance(train_m, dict):
        candidates.append(float(train_m["seq_per_s"]))
    compute = report["compute"]
    loader = report["loader"]
    assert isinstance(compute, dict) and isinstance(loader, dict)
    candidates.append(float(compute["bf16"]["seq_per_s"]))
    candidates.append(max(float(v["cold_seq_per_s"]) for v in loader.values()))
    rate = min(candidates)
    return {
        "seq_per_s": rate,
        "epochs_20": plan_hours(rate, 20),
        "epochs_100": plan_hours(rate, 100),
    }


def run(argv: Sequence[str] | None = None) -> dict[str, object]:
    """Parse the command line, run the smoke test, write ``report.json``, return the report."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--paths", nargs="*", default=[], help="episode globs (default: the config's)"
    )
    parser.add_argument("--minutes", type=float, default=15.0, help="length of the train phase")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, nargs="*", default=[4, 8])
    parser.add_argument("--loader-seconds", type=float, default=20.0)
    parser.add_argument("--compute-steps", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=Path("runs/gpu-smoke"))
    args = parser.parse_args(argv)
    base = load_config(args.config)
    data = DataConfig(tuple(args.paths), "", base.data.stats_episodes) if args.paths else base.data
    tc = TrainConfig(
        **{
            **asdict(base.train),
            "batch_size": args.batch_size,
            "device": args.device,
            "precision": "bf16",
            "out_dir": str(args.out),
            "log_every": 25,
            "steps": 100_000,
            "num_workers": args.workers[-1],
        }
    )
    cfg = RunConfig(f"{base.name}-smoke", base.seed, data, base.augment, base.model, tc)
    report = smoke(cfg, args.minutes, args.workers, args.loader_seconds, args.compute_steps)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    log.info("report: %s", args.out / "report.json")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point (logs the report)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    log.info("%s", json.dumps(run(argv), indent=1, default=str))


if __name__ == "__main__":
    main()
