"""A touch teacher: touch that is only ever a prediction *target* of the vision + proprio state.

In the target-only arm of the "input or teacher" test the state encoder never sees touch. A
``TouchTeacher`` encodes touch on its own (touch stem, projector, and its own small action-
conditioned predictor, trained with next-latent MSE plus SIGReg: a touch-only world model). An
EMA copy of its encoder gives the targets, as the tactile latents of Touch Dreaming (HTD, Niu et
al. 2026, arXiv:2604.13015: EMA target encoder; latent targets beat raw touch). Two heads on the
main model predict them: from the state ``s_t`` the latent of the *current* touch, and from the
predictor's output the latent of the *next* touch. Targets are detached: no gradient reaches the
teacher through the aux loss, and none of the main model's inputs is touch.

``placebo``: the targets come from the touch of *another sequence of the batch* (a roll by one), so
they have the statistics of touch latents and no information about this sequence.
"""

import copy
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from sensorium.model.layout import ChannelSpace
from sensorium.model.modules import Encoder as TransformerEncoder
from sensorium.model.modules import SIGReg, TouchStem, VectorStem


def mlp(dim: int) -> nn.Sequential:
    """Linear - BatchNorm - GELU - Linear (the projector of the state encoder)."""
    return nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Linear(dim, dim))


class TouchEncoder(nn.Module):
    """Touch ``[B, T, K, U]`` + region mask ``[B, R]`` -> latents ``[B, T, dim]``."""

    def __init__(
        self,
        space: ChannelSpace,
        dim: int,
        touch_dim: int,
        depth: int,
        heads: int,
        tokens: int,
        substeps: int,
    ) -> None:
        super().__init__()
        self.stem = TouchStem(
            torch.as_tensor(space.region_table()),
            space.region_meta(),
            space.size("touch"),
            substeps,
            touch_dim,
            depth,
            heads,
            tokens,
            dim,
        )
        self.projector = mlp(dim)

    def forward(self, touch: Tensor, region_mask: Tensor) -> Tensor:
        """Encode every step of a batch of sequences."""
        b, t = touch.shape[:2]
        mask = region_mask.repeat_interleave(t, 0)
        pooled = self.stem(touch.flatten(0, 1), mask).mean(1)
        return self.projector(pooled).view(b, t, -1)


class TouchTeacher(nn.Module):
    """The touch-only world model, its EMA target, and the two aux heads of the main model.

    Args:
        space: Channel space (touch channels and regions).
        dim: Latent and head width (the main model's ``dim``).
        touch_dim, depth, heads, tokens, substeps: The touch stem (as in the main config).
        action_channels: Union action channels (for the teacher's own action stem).
        steps: Sequence length (position table of the teacher's predictor).
        momentum: EMA momentum of the target encoder.
        placebo: Use the touch of another sequence of the batch as the aux target.
    """

    def __init__(
        self,
        space: ChannelSpace,
        dim: int,
        touch_dim: int,
        depth: int,
        heads: int,
        tokens: int,
        substeps: int,
        action_channels: int,
        steps: int,
        momentum: float,
        placebo: bool,
    ) -> None:
        super().__init__()
        self.placebo, self.momentum = placebo, momentum
        self.encoder = TouchEncoder(space, dim, touch_dim, depth, heads, tokens, substeps)
        self.target = copy.deepcopy(self.encoder)
        self.target.requires_grad_(False)
        self.action = VectorStem(substeps, action_channels, dim, dim)
        self.pos = nn.Parameter(torch.randn(1, steps, dim) * 0.02)
        self.predictor = TransformerEncoder(dim, 1, heads, cond=True)
        self.pred_proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.head_now = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.head_next = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    @torch.no_grad()
    def update_target(self) -> None:
        """Move the EMA target encoder towards the online touch encoder."""
        for tp, op in zip(self.target.parameters(), self.encoder.parameters(), strict=True):
            tp.mul_(self.momentum).add_(op.detach(), alpha=1.0 - self.momentum)
        for tb, ob in zip(self.target.buffers(), self.encoder.buffers(), strict=True):
            tb.copy_(ob)

    @torch.no_grad()
    def targets(self, touch: Tensor, region_mask: Tensor) -> Tensor:
        """Aux targets ``[B, T, dim]``: EMA latents of this batch's touch (placebo: another's)."""
        if self.placebo:
            touch, region_mask = touch.roll(1, 0), region_mask.roll(1, 0)
        return self.target(touch, region_mask)

    def own_loss(
        self, batch: Mapping[str, Tensor], sigreg: SIGReg, weight: float, pool_time: bool
    ) -> Tensor:
        """The teacher's own objective: next touch latent MSE + SIGReg (touch-only world model)."""
        z = self.encoder(batch["touch"], batch["region_mask"])
        cond = self.action(batch["action"].flatten(0, 1)).view_as(z)
        t = z.size(1) - 1
        h = self.predictor(z[:, :t] + self.pos[:, :t], cond=cond[:, :t], causal=True)
        mse = (self.pred_proj(h) - z[:, 1:]).pow(2).mean()
        return mse + weight * sigreg(z, pool_time=pool_time)

    def aux_loss(
        self, batch: Mapping[str, Tensor], states: Tensor, trunk: Tensor, has_touch: Tensor
    ) -> Tensor:
        """Mean of the two head errors against the (detached) targets, over sequences with touch.

        Args:
            batch: The batch (``touch`` and ``region_mask`` are read here, as targets only).
            states: Main-model states ``[B, T, dim]``.
            trunk: Main predictor output ``[B, T-1, dim]`` (before its projection).
            has_touch: ``[B]`` float, 1 where a sequence has touch.
        """
        target = self.targets(batch["touch"], batch["region_mask"])
        weight = has_touch * (has_touch.roll(1, 0) if self.placebo else has_touch)
        now = (self.head_now(states) - target).pow(2).mean((1, 2))
        nxt = (self.head_next(trunk) - target[:, 1:]).pow(2).mean((1, 2))
        return ((now + nxt) / 2 * weight).sum() / weight.sum().clamp_min(1.0)
