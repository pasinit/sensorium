"""Pretraining loop and CLI of the sensory layer.

Usage:
    sensorium-pretrain configs/model/cpu_tiny.toml

The config is a TOML file (``configs/model/``); see ``docs/model.md``. A run writes
``<out_dir>/<name>/`` with ``config.toml``, ``log.jsonl`` (one line per logged step) and
``ckpt.pt``. Everything is seeded: the same config and seed give the same run when
``num_workers = 0``.
"""

import argparse
import json
import logging
import math
import time
import tomllib
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sensorium.episode import Episode
from sensorium.model.data import LoaderConfig, SequenceDataset, compute_stats, spacing
from sensorium.model.events import load_start_weights
from sensorium.model.layout import INPUT_KEY, ChannelSpace, check_senses, read_header
from sensorium.model.probes import run_probes
from sensorium.model.sensory_layer import (
    ModelConfig,
    SensoryLayer,
    SensoryModel,
    collapse_metrics,
    count_parameters,
)
from sensorium.model.subsets import IndexedSequences, load_index

log = logging.getLogger("sensorium.model.pretrain")


@dataclass(frozen=True)
class DataConfig:
    """Which episodes to train on.

    Attributes:
        paths: Episode files or glob patterns (relative to the working directory).
        pool: Optional ``pool.json`` from ``sensorium-pool``; replaces ``paths`` and
            restricts the senses to the pool's.
        stats_episodes: Episodes used to fit the proprio and audio normalisation.
        subset_index: Optional index of ``subsets.build_index``; with ``subset_key`` it restricts
            the sequences drawn to one class of start positions (``conseq``, ``contact``, ``all``).
        subset_key: The class to draw from (needs ``subset_index``).
        space_paths: Optional episode files (or patterns) to build the channel space and its
            normalisation from, instead of the training files: lets runs on different data share
            one space (so a model trained on one kind of episode can be probed on another).
        drop_prefix: Proprio channel-name prefixes left out of the channel space (``("eff_",)``
            removes the actuator-effort channels: "proprio without effort").
        event_index: Optional start-weight file of ``sensorium.model.events``: sequences are
            drawn toward event windows (label-free signals) instead of uniformly.
    """

    paths: tuple[str, ...] = ()
    pool: str = ""
    stats_episodes: int = 64
    subset_index: str = ""
    subset_key: str = ""
    space_paths: tuple[str, ...] = ()
    drop_prefix: tuple[str, ...] = ()
    event_index: str = ""


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation settings (defaults follow LeWM's recipe where it gives one).

    Attributes:
        steps: Optimiser steps.
        batch_size: Sequences per step.
        lr, weight_decay: AdamW settings (LeWM: 5e-5 and 1e-3, bf16 on a GPU).
        warmup: Linear warm-up steps; cosine decay to 10% afterwards.
        grad_clip: Gradient-norm clip.
        log_every, save_every: Step intervals for the progress line and the checkpoint.
        save_steps: Steps at which to also keep a numbered copy ``ckpt-<step>.pt`` (to probe
            the encoder along training).
        num_workers: Loader processes (0 = in the main process, deterministic).
        out_dir: Where runs go (git-ignored).
        device: ``auto`` (CUDA if available), ``cpu`` or ``cuda``.
        precision: ``fp32`` or ``bf16`` (autocast of the forward pass).
        max_seconds: Stop after this wall-clock time (0 = run all ``steps``); the learning-rate
            schedule still follows ``steps``.
        eval_windows: Windows of the data encoded at the end to measure the states' std and
            effective rank in eval mode (0 = skip).
        threads: ``torch.set_num_threads`` (0 = leave the default). A run is bit-identical only for
            the same thread count: BLAS reductions change order with it and training amplifies the
            1e-6 differences within tens of steps (``run.json`` records the count used).
        probes: Also fit the privileged-label probes of ``probes.py`` on the final states
            (evaluation only; H1 episodes with privileged contacts).
    """

    steps: int = 300
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-3
    warmup: int = 20
    grad_clip: float = 1.0
    log_every: int = 10
    save_every: int = 0
    save_steps: tuple[int, ...] = ()
    num_workers: int = 0
    out_dir: str = "runs"
    device: str = "auto"
    precision: str = "fp32"
    max_seconds: float = 0.0
    eval_windows: int = 256
    probes: bool = False
    threads: int = 0


@dataclass(frozen=True)
class AugmentConfig:
    """Load-time augmentation switches (``[augment]``)."""

    crop: float = 0.9
    touch_noise: bool = True
    cache_episodes: int = 16
    frame_cache: str = ""  # a sensorium.model.frame_cache directory; "" decodes every episode
    reuse: int = 8  # sequences cut from an episode per pick (a batch of 16 spans 2 episodes)


@dataclass(frozen=True)
class RunConfig:
    """A whole run: name, seed and the four sub-configs."""

    name: str = "run"
    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def loader(self) -> LoaderConfig:
        """The loader configuration implied by the model's sequence geometry."""
        return LoaderConfig(
            steps=self.model.steps,
            substeps=self.model.substeps,
            view_size=self.model.vision_size,
            crop=self.augment.crop,
            touch_noise=self.augment.touch_noise,
            cache_episodes=self.augment.cache_episodes,
            frame_cache=self.augment.frame_cache,
            reuse=self.augment.reuse,
        )


def _table(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    """The TOML table ``key`` of a config (empty if absent)."""
    table = raw.get(key, {})
    if not isinstance(table, dict):
        raise ValueError(f"[{key}] must be a table")
    return table


def _section[T](cls: type[T], table: Mapping[str, object], where: str) -> T:
    """Build a dataclass from a TOML table, rejecting unknown keys (lists become tuples)."""
    names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(table) - names)
    if unknown:
        raise ValueError(f"[{where}] has unknown keys {unknown}; known: {sorted(names)}")
    return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in table.items()})


def parse_config(raw: Mapping[str, object]) -> RunConfig:
    """Build a ``RunConfig`` from parsed TOML.

    Raises:
        ValueError: on an unknown section or key, or an invalid value.
    """
    unknown = sorted(set(raw) - {"name", "seed", "data", "augment", "model", "train"})
    if unknown:
        raise ValueError(f"unknown config sections or keys {unknown}")
    return RunConfig(
        name=str(raw.get("name", "run")),
        seed=int(str(raw.get("seed", 0))),
        data=_section(DataConfig, _table(raw, "data"), "data"),
        augment=_section(AugmentConfig, _table(raw, "augment"), "augment"),
        model=_section(ModelConfig, _table(raw, "model"), "model"),
        train=_section(TrainConfig, _table(raw, "train"), "train"),
    )


def load_config(path: Path) -> RunConfig:
    """Read a run config from a TOML file."""
    return parse_config(tomllib.loads(path.read_text()))


def expand(pattern: str) -> list[Path]:
    """Files matching a path or glob pattern (sorted)."""
    p = Path(pattern).expanduser()
    base, rel = (Path(p.anchor), p.relative_to(p.anchor)) if p.is_absolute() else (Path(), p)
    return sorted(base.glob(str(rel)))


def senses_for_pool(senses: Sequence[str], pool_keys: Sequence[str]) -> tuple[str, ...]:
    """Configured senses whose input array the pool keeps (``pool.senses`` are input keys)."""
    return tuple(s for s in check_senses(senses) if INPUT_KEY[s] in pool_keys)


def resolve_data(cfg: RunConfig) -> tuple[list[Path], tuple[str, ...]]:
    """Episode files and effective senses of a run.

    Raises:
        FileNotFoundError: if no episode file matches.
    """
    senses = check_senses(cfg.model.senses)
    if cfg.data.pool:
        pool = json.loads(Path(cfg.data.pool).read_text())
        paths = [Path(e["path"]) for e in pool["episodes"]]
        kept = senses_for_pool(senses, pool["spec"]["senses"])
        if kept != senses:
            log.warning("pool senses restrict the model senses %s to %s", senses, kept)
        senses = kept
    else:
        paths = [f for pattern in cfg.data.paths for f in expand(pattern)]
    if not paths:
        raise FileNotFoundError(f"no episode files match {cfg.data.paths or cfg.data.pool!r}")
    return paths, senses


def build_space(
    paths: Sequence[Path], stats_episodes: int, drop_prefix: Sequence[str] = ()
) -> ChannelSpace:
    """The union channel space of some episode files, with normalisation fitted.

    ``drop_prefix`` leaves proprio channels with these name prefixes out of the space.
    """
    space = ChannelSpace.from_headers((read_header(p) for p in paths), drop_prefix)
    compute_stats(paths, space, stats_episodes)
    return space


def lr_at(step: int, tc: TrainConfig) -> float:
    """Learning rate at an optimiser step: linear warm-up, then cosine to 10%."""
    if step < tc.warmup:
        return tc.lr * (step + 1) / tc.warmup
    progress = (step - tc.warmup) / max(1, tc.steps - tc.warmup)
    return tc.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def save_checkpoint(
    path: Path, model: SensoryModel, space: ChannelSpace, cfg: RunConfig, step: int
) -> None:
    """Write model weights with everything needed to rebuild the layer."""
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": asdict(cfg.model),
            "loader_config": asdict(cfg.loader()),
            "space": space.to_json(),
            "step": step,
        },
        path,
    )


def load_layer(path: Path) -> SensoryLayer:
    """Rebuild a trained layer from a checkpoint written by ``save_checkpoint``."""
    raw = torch.load(path, weights_only=True)
    mc = dict(raw["model_config"])
    mc["senses"] = tuple(mc["senses"])
    space = ChannelSpace.from_json(raw["space"])
    model = SensoryModel(ModelConfig(**mc), space)
    model.load_state_dict(raw["model"])
    return SensoryLayer(model, space, LoaderConfig(**raw["loader_config"]))


def pick_device(name: str) -> torch.device:
    """``auto`` -> CUDA when available, else CPU; otherwise the named device.

    Raises:
        ValueError: on an unknown name or ``cuda`` without a GPU.
    """
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name not in ("cpu", "cuda") or (name == "cuda" and not torch.cuda.is_available()):
        raise ValueError(f"device {name!r} is not available (want auto, cpu or cuda)")
    return torch.device(name)


def _batches(
    dataset: SequenceDataset, tc: TrainConfig, seed: int, device: torch.device
) -> Iterator[dict[str, torch.Tensor]]:
    """An endless iterator of collated batches (pinned when the device is a GPU)."""
    loader = DataLoader(
        dataset,
        batch_size=tc.batch_size,
        num_workers=tc.num_workers,
        generator=torch.Generator().manual_seed(seed),
        persistent_workers=tc.num_workers > 0,
        prefetch_factor=4 if tc.num_workers > 0 else None,
        pin_memory=device.type == "cuda",
    )
    return iter(loader)


def ema_at(step: int, mc: ModelConfig, total: int) -> float:
    """EMA momentum at an optimiser step: linear from ``ema_momentum`` to ``ema_momentum_end``."""
    frac = min(1.0, (step - 1) / max(1, total - 1))
    return mc.ema_momentum + frac * (mc.ema_momentum_end - mc.ema_momentum)


def state_statistics(layer: SensoryLayer, paths: Sequence[Path], windows: int) -> dict[str, float]:
    """Std and effective rank of encoded states (eval mode, centre crop), a fixed sample.

    About ``windows`` windows spread evenly over the episodes, so numbers are comparable
    across runs whatever their batch size (the in-batch values depend on it).
    """
    per = max(1, windows // max(1, len(paths)))
    states = []
    for path in paths:
        episode = Episode.load(path, privileged=False)
        first = (layer.model.cfg.substeps - 1) * spacing(episode.dt)
        stride = max(1, (episode.length - first) // per)
        states.append(layer.encode(episode, stride=stride)[0][:per])
    stacked = torch.as_tensor(np.concatenate(states))[None]
    return {f"eval_{k}": v for k, v in collapse_metrics(stacked).items()}


def _record(
    step: int,
    result: dict[str, torch.Tensor],
    tc: TrainConfig,
    elapsed: float,
    on_log: Callable[[dict[str, float]], dict[str, float]] | None,
) -> dict[str, float]:
    """The metrics of a logged step (plus whatever ``on_log`` adds)."""
    metrics = {
        "step": step,
        "loss": float(result["loss"].detach()),
        "pred": float(result["pred"].detach()),
        "reg": float(result["reg"].detach()),
        **collapse_metrics(result["states"]),
        "seq_per_s": step * tc.batch_size / elapsed,
        "elapsed_s": elapsed,
    }
    metrics |= {k: float(result[k]) for k in ("aux", "teacher_own") if k in result}
    if on_log is not None:
        metrics |= on_log(metrics)
    return metrics


def make_dataset(
    cfg: RunConfig, paths: Sequence[Path], space: ChannelSpace, senses: Sequence[str]
) -> SequenceDataset:
    """The training sequences of a run: uniform, restricted to a class of an index, or weighted
    toward events (``data.event_index``)."""
    if cfg.data.subset_index:
        return IndexedSequences(
            paths,
            space,
            senses,
            cfg.loader(),
            load_index(Path(cfg.data.subset_index)),
            cfg.data.subset_key,
            seed=cfg.seed,
        )
    weights = load_start_weights(Path(cfg.data.event_index)) if cfg.data.event_index else None
    return SequenceDataset(paths, space, senses, cfg.loader(), seed=cfg.seed, start_weights=weights)


def save_step(
    step: int, out: Path, model: SensoryModel, space: ChannelSpace, cfg: RunConfig
) -> None:
    """Write ``ckpt-<step>.pt`` if ``step`` is one of ``train.save_steps`` (0: before training)."""
    if step in cfg.train.save_steps:
        save_checkpoint(out / f"ckpt-{step}.pt", model, space, cfg, step)


def train(
    cfg: RunConfig, on_log: Callable[[dict[str, float]], dict[str, float]] | None = None
) -> dict[str, float]:
    """Run the pretraining loop; returns the metrics of the last logged step.

    Writes ``config.toml``, ``log.jsonl`` and ``ckpt.pt`` under ``<out_dir>/<name>/``.

    Args:
        cfg: The run.
        on_log: Optional hook called with each logged step's metrics; the dict it returns
            is added to them (the GPU smoke test adds utilisation and memory).
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    tc = cfg.train
    if tc.threads:
        torch.set_num_threads(tc.threads)
    device = pick_device(tc.device)
    paths, senses = resolve_data(cfg)
    space_files = [f for pattern in cfg.data.space_paths for f in expand(pattern)] or paths
    space = build_space(space_files, cfg.data.stats_episodes, cfg.data.drop_prefix)
    model_cfg = ModelConfig(**{**asdict(cfg.model), "senses": senses})
    cfg = RunConfig(cfg.name, cfg.seed, cfg.data, cfg.augment, model_cfg, tc)
    model = SensoryModel(model_cfg, space).to(device)
    counts = count_parameters(model)
    log.info(
        "%d episodes, senses %s, %.1fM parameters, %s %s, objective %s",
        len(paths),
        senses,
        counts["total"] / 1e6,
        device,
        tc.precision,
        model_cfg.objective,
    )
    dataset = make_dataset(cfg, paths, space, model.data_senses)
    batches = _batches(dataset, tc, cfg.seed, device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=tc.lr, weight_decay=tc.weight_decay
    )
    out = Path(tc.out_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.toml").write_text(_dump_config(cfg))
    (out / "run.json").write_text(
        json.dumps(
            {
                "threads": torch.get_num_threads(),
                "torch": torch.__version__,
                "device": str(device),
                "num_workers": tc.num_workers,
                "seed": cfg.seed,
            }
        )
    )
    metrics: dict[str, float] = {}
    save_step(0, out, model, space, cfg)  # the untrained encoder (the floor of this seed)
    model.train()
    start = time.perf_counter()
    amp = torch.autocast(device.type, dtype=torch.bfloat16, enabled=tc.precision == "bf16")
    with (out / "log.jsonl").open("w") as journal:
        for step in range(1, tc.steps + 1):
            for group in opt.param_groups:
                group["lr"] = lr_at(step - 1, tc)
            batch = {k: v.to(device, non_blocking=True) for k, v in next(batches).items()}
            with amp:
                result = model(batch)
            opt.zero_grad(set_to_none=True)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            opt.step()
            model.update_target(ema_at(step, model_cfg, tc.steps))
            elapsed = time.perf_counter() - start
            last = step == tc.steps or 0 < tc.max_seconds <= elapsed
            if step % tc.log_every == 0 or last:
                metrics = _record(step, result, tc, elapsed, on_log)
                journal.write(json.dumps(metrics) + "\n")
                journal.flush()
                log.info(_line(metrics, tc.steps))
            if tc.save_every and step % tc.save_every == 0:
                save_checkpoint(out / "ckpt.pt", model, space, cfg, step)
            save_step(step, out, model, space, cfg)
            if last:
                break
        save_checkpoint(out / "ckpt.pt", model, space, cfg, int(metrics["step"]))
        if tc.eval_windows:
            layer = SensoryLayer(model, space, cfg.loader())
            metrics |= state_statistics(layer, paths, tc.eval_windows)
            journal.write(json.dumps(metrics) + "\n")
            log.info("eval states: std %.3f erank %.1f", metrics["eval_std"], metrics["eval_erank"])
        if tc.probes:
            scores = run_probes(SensoryLayer(model, space, cfg.loader()), paths)["state"]
            metrics |= {f"probe_{k}": v for k, v in scores.items()}
            journal.write(json.dumps(metrics) + "\n")
            log.info("probes: mean %.3f", metrics["probe_mean"])
    log.info("saved %s", out / "ckpt.pt")
    return metrics


def _line(m: dict[str, float], total: int) -> str:
    """The progress line of a logged step."""
    extra = "".join(f" {k} {v:.3g}" for k, v in m.items() if k.startswith("gpu_"))
    return (
        f"step {m['step']:.0f}/{total} loss {m['loss']:.4f} pred {m['pred']:.4f} "
        f"reg {m['reg']:.3f} std {m['std']:.3f} erank {m['erank']:.1f} | "
        f"{m['seq_per_s']:.2f} seq/s{extra}"
    )


def _dump_config(cfg: RunConfig) -> str:
    """The effective config as TOML (written next to the run)."""
    lines = [f'name = "{cfg.name}"', f"seed = {cfg.seed}"]
    for section in ("data", "augment", "model", "train"):
        lines.append(f"\n[{section}]")
        for key, value in asdict(getattr(cfg, section)).items():
            lines.append(
                f"{key} = {json.dumps(list(value) if isinstance(value, tuple) else value)}"
            )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point: ``sensorium-pretrain <config.toml>``."""
    parser = argparse.ArgumentParser(description="Pretrain the sensory layer.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--name", help="override the run name")
    parser.add_argument("--steps", type=int, help="override train.steps")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    if args.name:
        cfg = RunConfig(args.name, cfg.seed, cfg.data, cfg.augment, cfg.model, cfg.train)
    if args.steps:
        tc = TrainConfig(**{**asdict(cfg.train), "steps": args.steps})
        cfg = RunConfig(cfg.name, cfg.seed, cfg.data, cfg.augment, cfg.model, tc)
    train(cfg)


if __name__ == "__main__":
    main()
