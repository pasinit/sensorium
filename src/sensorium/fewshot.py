"""The few-shot benchmark harness, minimum for the E1 pilot (D0.6; docs/experiment-01.md).

A frozen encoder turns held-out task demonstrations into features; a small behaviour-cloning head
learns the next actions from ``k`` of them; it is scored offline by its action error on the other
demonstrations, and in closed loop by the task's success test in the simulator
(``bodies.h1_closed_loop``).

- **Features** (``FeatureFn``): the state of a pretrained sensory layer for the window ending at
  each step (``layer_features``: ``model.planning.encode_at``, eval mode, centre crop), or the
  frozen DINOv2 baseline on the frame at each step (``dino_features_fn``), optionally with the
  proprio vector.
- **Cache** (``DemoCache``, ``build_cache``): features at every window end of the demonstrations
  that succeeded, and the ``horizon`` actions that follow each (the action chunk the head predicts).
- **Head** (``BCHead``, ``train_head``): a two-layer MLP on standardised features, trained with mean
  squared error on the chunk. Its output is clipped to the action range [-1, 1].
- **Offline score** (``action_rmse``): RMS action error on episodes the head did not see.
"""

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sensorium.episode import Episode
from sensorium.model.controls import dino_features
from sensorium.model.data import spacing
from sensorium.model.planning import encode_at
from sensorium.model.sensory_layer import SensoryLayer

FeatureFn = Callable[[Episode, np.ndarray], np.ndarray]
"""``features(episode, ends) -> [len(ends), D]``: features of the windows ending at ``ends``."""


def layer_features(layer: SensoryLayer, drop: Sequence[str] = ()) -> FeatureFn:
    """A frozen sensory layer's states ``s`` (``drop``: senses masked out of its fusion)."""

    def features(episode: Episode, ends: np.ndarray) -> np.ndarray:
        return np.asarray(encode_at(layer, episode, ends, drop)[0], dtype=np.float32)

    return features


def dino_features_fn(model: nn.Module, proprio: bool = True) -> FeatureFn:
    """The frozen DINOv2 baseline: class and mean patch token of the frame at each window end
    (``model.controls.dino_features``), and the proprio vector there if ``proprio``."""

    def features(episode: Episode, ends: np.ndarray) -> np.ndarray:
        out = dino_features(model, np.asarray(episode.rgb)[ends])
        if proprio:
            out = np.concatenate([out, np.asarray(episode.proprio)[ends]], axis=1)
        return np.asarray(out, dtype=np.float32)

    return features


def first_step(dt: float, substeps: int) -> int:
    """The first step a window of ``substeps`` observations can end at, for control period ``dt``
    (``SensoryLayer.encode``)."""
    return (substeps - 1) * spacing(dt)


def first_end(episode: Episode, substeps: int) -> int:
    """``first_step`` of an episode."""
    return first_step(episode.dt, substeps)


@dataclass
class DemoCache:
    """Features and action chunks of demonstrations, one row per window end.

    Attributes:
        features: ``[n, D]`` float32.
        targets: ``[n, horizon, A]`` float32, the actions from the window's end on.
        episode: ``[n]`` index of the row's demonstration in ``sources``.
        ends: ``[n]`` the window's last step.
        sources: The demonstration files, in order.
        action_names: The action columns.
    """

    features: np.ndarray
    targets: np.ndarray
    episode: np.ndarray
    ends: np.ndarray
    sources: list[str]
    action_names: list[str]

    @property
    def horizon(self) -> int:
        """Actions per chunk."""
        return int(self.targets.shape[1])

    def save(self, path: Path) -> None:
        """Write the cache as ``.npz``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=self.features,
            targets=self.targets,
            episode=self.episode,
            ends=self.ends,
            sources=np.array(self.sources),
            action_names=np.array(self.action_names),
        )

    @classmethod
    def load(cls, path: Path) -> "DemoCache":
        """Read a cache written by ``save``."""
        with np.load(path) as f:
            return cls(
                f["features"],
                f["targets"],
                f["episode"],
                f["ends"],
                [str(x) for x in f["sources"]],
                [str(x) for x in f["action_names"]],
            )


def chunks(action: np.ndarray, ends: np.ndarray, horizon: int) -> np.ndarray:
    """``[n, horizon, A]``: the actions from each end on, the last one repeated past the end."""
    idx = np.minimum(np.asarray(ends)[:, None] + np.arange(horizon)[None], len(action) - 1)
    return np.asarray(action, dtype=np.float32)[idx]


def build_cache(
    paths: Sequence[Path],
    features: FeatureFn,
    horizon: int,
    substeps: int,
    stride: int = 1,
    successful_only: bool = True,
) -> DemoCache:
    """Features and chunks at every ``stride``-th window end of the demonstrations in ``paths``.

    With ``successful_only``, episodes whose ``meta["success"]`` is not true are left out (the
    demonstrators of a weak task fail often; only successes are demonstrations).

    Raises:
        ValueError: if no episode is left, or the episodes' action columns differ.
    """
    rows: dict[str, list[np.ndarray]] = {"features": [], "targets": [], "episode": [], "ends": []}
    sources: list[str] = []
    names: list[str] | None = None
    for path in paths:
        ep = Episode.load(path)
        if successful_only and not ep.meta.get("success"):
            continue
        if names is None:
            names = list(ep.action_names)
        elif list(ep.action_names) != names:
            raise ValueError(f"{path}: action columns differ from the first episode's")
        ends = np.arange(first_end(ep, substeps), ep.length, stride)
        if len(ends) == 0:
            continue
        rows["features"].append(features(ep, ends))
        rows["targets"].append(chunks(ep.action, ends, horizon))
        rows["episode"].append(np.full(len(ends), len(sources)))
        rows["ends"].append(ends)
        sources.append(str(path))
    if not sources or names is None:
        raise ValueError("no demonstration left to cache")
    return DemoCache(
        np.concatenate(rows["features"]).astype(np.float32),
        np.concatenate(rows["targets"]).astype(np.float32),
        np.concatenate(rows["episode"]),
        np.concatenate(rows["ends"]),
        sources,
        names,
    )


def choose_demos(n_episodes: int, k: int, seed: int) -> list[int]:
    """``k`` demonstration indices of ``n_episodes``, drawn without replacement by ``seed``.

    Raises:
        ValueError: if ``k`` exceeds the demonstrations available.
    """
    if not 0 < k <= n_episodes:
        raise ValueError(f"k = {k} with {n_episodes} demonstrations")
    return sorted(int(i) for i in np.random.default_rng(seed).choice(n_episodes, k, replace=False))


@dataclass
class BCConfig:
    """Training of the behaviour-cloning head (fixed for every encoder and k)."""

    hidden: int = 256
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch: int = 256


class BCHead(nn.Module):
    """Two-layer MLP: standardised features -> an action chunk ``[horizon, A]`` in [-1, 1]."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, dim: int, horizon: int, actions: int, hidden: int) -> None:
        super().__init__()
        self.horizon, self.actions = horizon, actions
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, horizon * actions)
        )
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[n, D]`` raw features -> ``[n, horizon, A]`` (unclipped)."""
        out: torch.Tensor = self.net((x - self.mean) / self.std)
        return out.view(-1, self.horizon, self.actions)

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> np.ndarray:
        """``[n, D]`` -> ``[n, horizon, A]`` actions, clipped to [-1, 1]."""
        self.eval()
        out = self(torch.as_tensor(np.asarray(features, dtype=np.float32)))
        return np.asarray(out.clamp(-1.0, 1.0).numpy())


def train_head(cache: DemoCache, demos: Sequence[int], cfg: BCConfig, seed: int) -> BCHead:
    """A head trained on the rows of the demonstrations ``demos`` (indices into the cache).

    Raises:
        ValueError: if ``demos`` select no rows.
    """
    rows = np.isin(cache.episode, list(demos))
    if not rows.any():
        raise ValueError("the chosen demonstrations have no rows")
    torch.manual_seed(seed)
    x = torch.as_tensor(cache.features[rows])
    y = torch.as_tensor(cache.targets[rows])
    head = BCHead(x.shape[1], cache.horizon, y.shape[2], cfg.hidden)
    head.mean.copy_(x.mean(0))
    head.std.copy_(x.std(0, unbiased=False).clamp_min(1e-6))
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(seed)
    head.train()
    for _ in range(cfg.epochs):
        order = torch.randperm(len(x), generator=gen)
        for i in range(0, len(x), cfg.batch):
            b = order[i : i + cfg.batch]
            loss = nn.functional.mse_loss(head(x[b]), y[b])
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
    head.eval()
    return head


def action_rmse(head: BCHead, cache: DemoCache, episodes: Sequence[int]) -> float:
    """RMS error of the head's chunks on the rows of ``episodes`` (indices into the cache)."""
    rows = np.isin(cache.episode, list(episodes))
    if not rows.any():
        return float("nan")
    pred = head.predict(cache.features[rows])
    return float(np.sqrt(np.mean((pred - cache.targets[rows]) ** 2)))


def save_head(head: BCHead, cfg: BCConfig, path: Path, meta: dict[str, object]) -> None:
    """Write a head with its config and provenance (``meta``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": head.state_dict(),
            "dim": int(head.mean.numel()),
            "horizon": head.horizon,
            "actions": head.actions,
            "config": asdict(cfg),
            "meta": meta,
        },
        path,
    )


def load_head(path: Path) -> BCHead:
    """Rebuild a head written by ``save_head``."""
    raw = torch.load(path, weights_only=True)
    head = BCHead(raw["dim"], raw["horizon"], raw["actions"], raw["config"]["hidden"])
    head.load_state_dict(raw["state"])
    return head.eval()
