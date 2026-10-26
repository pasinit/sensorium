"""The sensory layer v0: per-sense stems -> fusion encoder -> one state per step -> predictor.

Per model step (0.2 s) every configured sense becomes a few tokens (vision 8, touch 6,
audio 1, each proprio group 1); a bidirectional **fusion encoder** reads them with a
summary token and its output, projected, is the **state** ``s_t`` (``encode`` returns it).
A **causal predictor**, modulated by the action chunk with AdaLN-zero, predicts
``s_{t+1}`` from ``s_{<=t}``. The loss is the next-state MSE plus SIGReg on the states, as
in LeWorldModel: no EMA target, no stop-gradient, one tunable weight.

A sense that is not configured has no stem and no token; a sense an episode lacks (a car
has no skin) is masked. Training can also drop whole senses at random (sense dropout), so a
layer trained with all senses can be run without some.

Design sources and what is unverified: ``docs/model.md``.
"""

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from torch import Tensor, nn

from sensorium.episode import Episode
from sensorium.model import reference
from sensorium.model.data import LoaderConfig, build, load_episode, spacing
from sensorium.model.layout import ChannelSpace, check_senses
from sensorium.model.modules import (
    Encoder,
    SIGReg,
    TouchStem,
    VectorStem,
    VisionStem,
)
from sensorium.model.reference import PROPRIO_SENSES
from sensorium.model.teacher import TouchTeacher
from sensorium.vision import eval_view

REFERENCE = ("mae", "nextobs")
OBJECTIVES = ("sigreg", "ema_vicreg", "mae", "nextobs")
TEACHERS = ("none", "touch", "touch_placebo")
DEFAULT_RECON = 0.5
"""Reconstruction weight of the default ``sigreg`` model (B-fix-0 / B-fix-1: one masked pass,
the decoder on the pre-projector state). ``recon_weight = 0`` gives the plain JEPA ("A0")."""


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters of the sensory layer (defaults: the ~40M v0 of ``docs/model.md``).

    Attributes:
        senses: Configured senses, any subset of ``layout.SENSES``.
        dim: Width of the fusion encoder, predictor and state.
        fusion_depth, predictor_depth, heads: Depths and attention heads of those two.
        vision_size, vision_patch: Input side and patch side (px).
        vision_dim, vision_depth, vision_heads, vision_tokens: The vision ViT and its pooling.
        touch_dim, touch_depth, touch_heads, touch_tokens: The region transformer and pooling.
        steps: Model steps ``T`` per sequence (the predictor's context length).
        substeps: Observations ``K`` stacked per model step.
        sense_dropout: Probability of dropping each droppable sense for a whole sequence.
        sense_dropout_senses: The senses that can be dropped (empty: all configured senses).
        proprio_dropout: Probability of dropping *all* proprio groups together for a whole
            sequence (vision has to carry the state then).
        surprise: ``none``, ``error`` (weight each step's next-state loss by the model's own
            detached prediction error) or ``touch`` (by the change of the touch input from step to
            step; needs the touch sense); weights are normalised to mean 1 and clipped to
            ``[0.2, 5]``. No privileged data.
        teacher: ``none``, ``touch`` (touch is a prediction target of the state, see
            ``teacher.py``) or ``touch_placebo`` (the targets come from another sequence). Touch
            may or may not be in ``senses``: without it touch is *only* a target (the target-only
            arm), with it touch is both an input and a target (the default model of H1, "a sense
            is an input and a teacher").
        teacher_weight, teacher_momentum: Weight of the aux loss and EMA momentum of the target.
        sigreg_weight, sigreg_proj: SIGReg weight and number of random projections.
        sigreg_pool_time: Compare the B * T states at once instead of the B states of each step
            (for small batches; see ``modules.SIGReg``).
        objective: ``sigreg``, ``ema_vicreg`` (see ``SensoryModel``) or the input-space
            references ``mae`` and ``nextobs`` (``reference.py``; vision and proprio senses only).
        mae_ratio, mae_proprio_ratio: Share of frame patches / proprio channels hidden by ``mae``.
        ema_momentum, ema_momentum_end: EMA momentum of the target encoder, linearly
            annealed over the run (``ema_vicreg`` only).
        recon_weight: ``sigreg`` only: weight of an MAE-style reconstruction term. ``None`` (the
            default) means ``DEFAULT_RECON`` for ``sigreg`` with vision and 0 otherwise (``recon``);
            0 is the plain JEPA of B-variety-0 ("A0"). A
            decoder reads the *pre-projector* state of a masked copy of the batch and reconstructs
            the masked patches and proprio channels (``mae_ratio``, ``mae_proprio_ratio``; the
            ``mae`` objective's targets), so the encoder must keep what next-state prediction can
            drop; the projector and the next-state loss are unchanged.
        recon_source: ``z`` (the default: the decoder reads the pre-projector summary) or ``s``
            (B-rep-0: it reads the projector output, the state itself).
        recon_proprio: ``hidden`` (the default: the proprio reconstruction is scored on the
            channels the mask hid) or ``all`` (B-rep-0 arm 1b: on every proprio channel, hidden
            and visible, against the clean input).
        skip_tokens: B-skip-0: ``z = summary + W [fused sense tokens] + b``, the fusion's sense
            tokens (absent senses zeroed) concatenated and mapped linearly onto the summary.
            ``W`` and ``b`` start at zero (no random draw, so every other parameter initialises
            as without the option and step 0 is the same model).
        skip_projector: B-skip-0: ``s = projector(z) + z`` (a residual around the projector).
        summary_tokens: Summary tokens ``K`` of the fusion (default 1). With ``K > 1`` every
            token is projected by the shared projector and the state is their concatenation
            (``K * dim``); the predictor maps it to ``dim`` and back, SIGReg is applied per token
            and averaged, and the reconstruction decoder reads the sum of the tokens (a linear map
            on the concatenation with weights tied across tokens). ``sigreg`` without a teacher
            only.
        recon_input: ``masked`` (the default, B-fix-1: one pass; the batch is masked once and
            the encoder runs once on it; the next-state loss and SIGReg use its projected
            states, the decoder reads its pre-projector states, scored on the hidden entries) or
            ``separate`` (B-fix-0: two encoder passes, the next-state objective on the clean
            batch and the reconstruction on a masked copy). Training objective only: ``embed``
            and ``encode`` always see clean inputs.
        target_grad: ``sigreg`` only: gradients flow into the next-state targets (LeWM). False
            stops them (the encoder is then shaped by the prediction only through its inputs).
        vicreg_var_weight, vicreg_cov_weight: Weights of VICReg's variance and covariance
            terms against an invariance (prediction) weight of 1 (VICReg's own 25 : 25 : 1
            is 1 : 1 : 0.04).
    """

    senses: tuple[str, ...] = (
        "vision",
        "touch",
        "audio",
        "proprio_body",
        "vestibular",
        "proprio_effector",
    )
    dim: int = 512
    fusion_depth: int = 4
    predictor_depth: int = 4
    heads: int = 8
    vision_size: int = 128
    vision_patch: int = 16
    vision_dim: int = 192
    vision_depth: int = 12
    vision_heads: int = 3
    vision_tokens: int = 8
    touch_dim: int = 256
    touch_depth: int = 2
    touch_heads: int = 4
    touch_tokens: int = 6
    steps: int = 16
    substeps: int = 4
    sense_dropout: float = 0.0
    sense_dropout_senses: tuple[str, ...] = ()
    proprio_dropout: float = 0.0
    mae_ratio: float = 0.75
    mae_proprio_ratio: float = 0.5
    surprise: str = "none"
    teacher: str = "none"
    teacher_weight: float = 1.0
    teacher_momentum: float = 0.99
    sigreg_weight: float = 0.1
    sigreg_proj: int = 1024
    sigreg_pool_time: bool = False
    objective: str = "sigreg"
    ema_momentum: float = 0.99
    ema_momentum_end: float = 0.999
    vicreg_var_weight: float = 1.0
    vicreg_cov_weight: float = 0.04
    target_grad: bool = True
    recon_weight: float | None = None
    recon_input: str = "masked"
    recon_source: str = "z"
    recon_proprio: str = "hidden"
    summary_tokens: int = 1
    skip_tokens: bool = False
    skip_projector: bool = False

    def __post_init__(self) -> None:
        check_senses(self.senses)
        if not 0.0 <= self.sense_dropout < 1.0:
            raise ValueError(f"sense_dropout must be in [0, 1), got {self.sense_dropout}")
        check_senses(self.sense_dropout_senses)
        if self.surprise not in ("none", "error", "touch"):
            raise ValueError(f"surprise must be none, error or touch, got {self.surprise!r}")
        if self.surprise == "touch" and "touch" not in self.senses:
            raise ValueError("surprise = touch needs the touch sense as an input")
        if not 0.0 <= self.proprio_dropout < 1.0:
            raise ValueError(f"proprio_dropout must be in [0, 1), got {self.proprio_dropout}")
        if self.teacher not in TEACHERS:
            raise ValueError(f"teacher must be one of {TEACHERS}, got {self.teacher!r}")
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {self.objective!r}")
        if self.objective in REFERENCE and set(self.senses) - {"vision", *PROPRIO_SENSES}:
            raise ValueError(
                f"objective {self.objective!r} supports vision and proprio senses only"
            )
        self._check_recon()

    def _check_recon(self) -> None:
        """Validate the reconstruction options (``recon_weight``, ``recon_input``)."""
        if self.recon_weight is not None and self.recon_weight < 0:
            raise ValueError(f"recon_weight must be >= 0, got {self.recon_weight}")
        if self.recon_weight and (self.objective != "sigreg" or "vision" not in self.senses):
            raise ValueError("recon_weight > 0 needs the sigreg objective and the vision sense")
        if self.recon_input not in ("separate", "masked"):
            raise ValueError(f"recon_input must be separate or masked, got {self.recon_input!r}")
        if self.recon_source not in ("z", "s"):
            raise ValueError(f"recon_source must be z or s, got {self.recon_source!r}")
        if self.recon_proprio not in ("hidden", "all"):
            raise ValueError(f"recon_proprio must be hidden or all, got {self.recon_proprio!r}")
        if self.summary_tokens < 1:
            raise ValueError(f"summary_tokens must be >= 1, got {self.summary_tokens}")
        if self.summary_tokens > 1 and (self.objective != "sigreg" or self.teacher != "none"):
            raise ValueError("summary_tokens > 1 needs the sigreg objective without a teacher")

    @property
    def state_dim(self) -> int:
        """Width of the state ``s``: ``summary_tokens * dim``."""
        return self.summary_tokens * self.dim

    @property
    def recon(self) -> float:
        """The reconstruction weight in use: ``recon_weight`` if set, else the default
        (``DEFAULT_RECON`` for the ``sigreg`` objective with vision, 0 otherwise)."""
        if self.recon_weight is not None:
            return self.recon_weight
        return DEFAULT_RECON if self.objective == "sigreg" and "vision" in self.senses else 0.0

    def resolved(self) -> "ModelConfig":
        """This config with ``recon_weight`` written out (what checkpoints store, so a later
        change of the default cannot change how a saved model is rebuilt)."""
        return replace(self, recon_weight=self.recon)


class StateEncoder(nn.Module):
    """Stems and fusion encoder: a batch of windows -> states ``[B, T, dim]``.

    The part of the model that the EMA objective copies as its target and that ``encode``
    runs. A sense that is not configured has no stem and no token.

    Args:
        cfg: Hyperparameters.
        space: The channel space of the data (sizes of the touch and proprio stems).
    """

    def __init__(self, cfg: ModelConfig, space: ChannelSpace) -> None:
        super().__init__()
        self.cfg = cfg
        self.senses = check_senses(cfg.senses)
        if "touch" in self.senses and not space.regions:
            raise ValueError("the touch sense is configured but no episode in the data has touch")
        d = cfg.dim
        self.n_tokens = {s: self._tokens(s) for s in self.senses}
        self.stems = nn.ModuleDict({s: self._stem(s, space) for s in self.senses})
        k = cfg.summary_tokens
        self.summary = nn.Parameter(torch.randn(1, k, d) * 0.02)
        self.slot = nn.Parameter(torch.randn(1, k + sum(self.n_tokens.values()), d) * 0.02)
        self.fusion = Encoder(d, cfg.fusion_depth, cfg.heads)
        self.projector = nn.Sequential(
            nn.Linear(d, d), nn.BatchNorm1d(d), nn.GELU(), nn.Linear(d, d)
        )
        if cfg.skip_tokens:  # zeros, no random draw: the other parameters are unchanged
            n = sum(self.n_tokens.values())
            self.skip_weight = nn.Parameter(torch.zeros(k * d, n * d))
            self.skip_bias = nn.Parameter(torch.zeros(k * d))

    def _tokens(self, sense: str) -> int:
        c = self.cfg
        return {"vision": c.vision_tokens, "touch": c.touch_tokens}.get(sense, 1)

    def _stem(self, sense: str, space: ChannelSpace) -> nn.Module:
        c = self.cfg
        if sense == "vision":
            return VisionStem(
                c.vision_size,
                c.vision_patch,
                c.vision_dim,
                c.vision_depth,
                c.vision_heads,
                c.vision_tokens,
                c.dim,
            )
        if sense == "touch":
            table = torch.as_tensor(space.region_table())
            return TouchStem(
                table,
                space.region_meta(),
                space.size("touch"),
                c.substeps,
                c.touch_dim,
                c.touch_depth,
                c.touch_heads,
                c.touch_tokens,
                c.dim,
            )
        return VectorStem(
            c.substeps, space.size("audio" if sense == "audio" else sense), c.dim, c.dim
        )

    def sense_tokens(self, batch: Mapping[str, Tensor]) -> list[Tensor]:
        """Tokens ``[N, n, dim]`` of every configured sense, ``N = B * T``."""
        out = []
        for sense in self.senses:
            x = batch[sense].flatten(0, 1)
            stem = self.stems[sense]
            if sense == "touch":
                region = batch["region_mask"].repeat_interleave(batch["action"].size(1), 0)
                out.append(stem(x, region))
            else:
                y = stem(x)
                out.append(y if y.dim() == 3 else y[:, None])
        return out

    def keep_mask(self, present: Tensor, drop: bool) -> Tensor:
        """``[B, S]`` senses to use: present in the episode and, if ``drop``, not dropped."""
        keep = present.clone()
        p = self.cfg.sense_dropout
        if drop and p > 0.0:
            only = self.cfg.sense_dropout_senses
            droppable = torch.tensor(
                [not only or s in only for s in self.senses], device=keep.device
            )
            keep &= (torch.rand(keep.shape, device=keep.device) >= p) | ~droppable
            empty = ~keep.any(-1)
            pick = torch.randint(keep.size(1), (keep.size(0),), device=keep.device)
            keep[empty, pick[empty]] = True
        q = self.cfg.proprio_dropout
        if drop and q > 0.0:
            is_proprio = torch.tensor(
                [s.startswith(("proprio", "vestib")) for s in self.senses], device=keep.device
            )
            gone = torch.rand(keep.size(0), 1, device=keep.device) < q
            keep &= ~(gone & is_proprio[None])
        return keep

    def forward(self, batch: Mapping[str, Tensor], drop: bool = False) -> Tensor:
        """States ``[B, T, dim]`` of a batch (see ``data.build`` for the batch layout).

        Args:
            batch: Tensors per sense, ``action``, ``present`` and ``region_mask``.
            drop: Apply sense dropout (training only).
        """
        return self.project(self.summary_token(batch, drop))

    def summary_token(self, batch: Mapping[str, Tensor], drop: bool = False) -> Tensor:
        """The fusion's summary tokens, concatenated ``[B, T, K * dim]``: the state before the
        projector."""
        b, t = batch["action"].shape[:2]
        k = self.cfg.summary_tokens
        keep = self.keep_mask(batch["present"], drop).repeat_interleave(t, 0)  # [N, S]
        tokens = [self.summary.expand(b * t, -1, -1), *self.sense_tokens(batch)]
        valid = [torch.ones(b * t, k, dtype=torch.bool, device=keep.device)]
        valid += [keep[:, i : i + 1].expand(-1, n) for i, n in enumerate(self.n_tokens.values())]
        x = torch.cat(tokens, 1) + self.slot
        valid_all = torch.cat(valid, 1)
        fused = self.fusion(x, mask=valid_all[:, None, None, :])
        z = fused[:, :k].reshape(b * t, -1)
        if self.cfg.skip_tokens:
            sense = (fused[:, k:] * valid_all[:, k:, None]).flatten(1)  # absent senses zeroed
            z = z + sense @ self.skip_weight.T + self.skip_bias
        return z.reshape(b, t, -1)

    def project(self, summary: Tensor) -> Tensor:
        """Projector output ``[B, T, K * dim]`` of summary tokens ``[B, T, K * dim]`` (the
        state): the shared projector applied to every token (plus the summary itself with
        ``skip_projector``)."""
        b, t = summary.shape[:2]
        s = self.projector(summary.reshape(-1, self.cfg.dim)).view(b, t, -1)
        return s + summary if self.cfg.skip_projector else s


def surprise_weights(kind: str, err: Tensor, batch: Mapping[str, Tensor]) -> Tensor:
    """Per-(sequence, step) loss weights from the model's own signals (mean 1, clipped to [0.2, 5]).

    ``error``: the detached next-state prediction error. ``touch``: the summed absolute change of
    the touch input between a step and the next. ``none``: ones. Only model inputs and the model's
    own errors are used.
    """
    if kind == "none":
        return torch.ones_like(err)
    if kind == "error":
        signal = err.detach()
    else:
        touch = batch["touch"].mean(2)  # [B, T, U], the window of K sub-steps averaged
        signal = (touch[:, 1:] - touch[:, :-1]).abs().sum(-1).detach()
    weights = (signal / signal.mean().clamp_min(1e-8)).clamp(0.2, 5.0)
    return weights / weights.mean()


def vicreg_terms(states: Tensor) -> tuple[Tensor, Tensor]:
    """VICReg's variance and covariance terms of ``[B, T, D]`` states (Bardes et al. 2022).

    The variance term pushes every dimension's std over the ``B * T`` samples to at least
    1 (hinge); the covariance term pushes the off-diagonal covariances to 0.
    """
    x = states.reshape(-1, states.size(-1)).float()
    x = x - x.mean(0, keepdim=True)
    std = torch.sqrt(x.var(0) + 1e-4)
    var = torch.relu(1.0 - std).mean()
    cov = (x.T @ x) / max(x.size(0) - 1, 1)
    off = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return var, off / x.size(1)


class SensoryModel(nn.Module):
    """The trainable part: state encoder, action encoder, predictor and the objective.

    Two objectives (``ModelConfig.objective``): ``sigreg`` (LeWorldModel: next-state MSE
    plus SIGReg, gradients through both sides) and ``ema_vicreg`` (the predictor regresses
    the states of an EMA copy of the encoder, with VICReg's variance and covariance terms on
    the online states against collapse; ``update_target`` after every optimiser step).

    Args:
        cfg: Hyperparameters.
        space: The channel space of the data.
    """

    def __init__(self, cfg: ModelConfig, space: ChannelSpace) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.encoder = StateEncoder(cfg, space)
        self.senses = self.encoder.senses
        self.n_tokens = self.encoder.n_tokens
        self.action = VectorStem(cfg.substeps, space.size("action"), d, d)
        self.pos = nn.Parameter(torch.randn(1, cfg.steps, d) * 0.02)
        self.predictor = Encoder(d, cfg.predictor_depth, cfg.heads, cond=True)
        self.pred_proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.sigreg = SIGReg(num_proj=cfg.sigreg_proj)
        self.data_senses = check_senses(
            sorted(set(cfg.senses) | ({"touch"} if cfg.teacher != "none" else set()))
        )
        self.teacher: TouchTeacher | None = None
        if cfg.teacher != "none":
            if not space.regions:
                raise ValueError("a touch teacher needs touch channels in the data")
            self.teacher = TouchTeacher(
                space,
                d,
                cfg.touch_dim,
                cfg.touch_depth,
                cfg.touch_heads,
                cfg.touch_tokens,
                cfg.substeps,
                space.size("action"),
                cfg.steps,
                cfg.teacher_momentum,
                cfg.teacher == "touch_placebo",
            )
        self.decoder: reference.InputDecoder | None = None
        if cfg.objective in REFERENCE:
            side = cfg.vision_size // cfg.vision_patch
            proprio = sum(space.size(s) for s in self.senses if s in PROPRIO_SENSES)
            self.decoder = reference.InputDecoder(
                d, 4 * d, side * side, cfg.vision_patch**2 * 3, proprio
            )
        self.target: StateEncoder | None = None
        if cfg.objective == "ema_vicreg":
            self.target = copy.deepcopy(self.encoder)
            self.target.requires_grad_(False)
        self.recon: reference.InputDecoder | None = None
        if cfg.recon > 0:  # created last: the other modules' initialisation is unchanged
            side = cfg.vision_size // cfg.vision_patch
            proprio = sum(space.size(s) for s in self.senses if s in PROPRIO_SENSES)
            self.recon = reference.InputDecoder(
                d, 4 * d, side * side, cfg.vision_patch**2 * 3, proprio
            )
        self.state_in: nn.Module = nn.Identity()  # K > 1: the state (K * dim) to dim and back
        self.state_out: nn.Module = nn.Identity()
        if cfg.summary_tokens > 1:  # created after everything else, as ``recon``
            self.state_in = nn.Linear(cfg.state_dim, d)
            self.state_out = nn.Linear(d, cfg.state_dim)

    def _view(self, batch: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """The batch as the state encoder sees it: ``present`` restricted to its own senses."""
        if batch["present"].shape[1] == len(self.senses):  # already only the encoder's senses
            return batch
        cols = [self.data_senses.index(s) for s in self.senses]
        return {**batch, "present": batch["present"][:, cols]}

    def embed(self, batch: Mapping[str, Tensor]) -> Tensor:
        """States ``[B, T, dim]`` of the online encoder (sense dropout only while training)."""
        return self.encoder(self._view(batch), drop=self.training)

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        """Move the EMA targets (EMA objective, touch teacher) towards their online encoders."""
        if self.teacher is not None:
            self.teacher.update_target()
        if self.target is None:
            return
        for tp, op in zip(self.target.parameters(), self.encoder.parameters(), strict=True):
            tp.mul_(momentum).add_(op.detach(), alpha=1.0 - momentum)
        for tb, ob in zip(self.target.buffers(), self.encoder.buffers(), strict=True):
            tb.copy_(ob)

    def trunk(self, states: Tensor, actions: Tensor) -> Tensor:
        """The predictor's output ``[B, T, dim]`` (causal, action-conditioned), pre-projection."""
        t = states.size(1)
        return self.predictor(self.state_in(states) + self.pos[:, :t], cond=actions, causal=True)

    def predict(self, states: Tensor, actions: Tensor) -> Tensor:
        """Next states ``[B, T, state_dim]`` from states and action conditioning (causal)."""
        return self.state_out(self.pred_proj(self.trunk(states, actions)))

    def regulariser(self, states: Tensor) -> Tensor:
        """SIGReg of the states: on the whole state, or with several summary tokens the mean of
        its value on each token's ``dim`` columns."""
        c = self.cfg
        if c.summary_tokens == 1:
            return self.sigreg(states, pool_time=c.sigreg_pool_time)
        parts = states.reshape(*states.shape[:-1], c.summary_tokens, c.dim).unbind(-2)
        return torch.stack([self.sigreg(p, pool_time=c.sigreg_pool_time) for p in parts]).mean()

    def decoder_input(self, states: Tensor) -> Tensor:
        """What the reconstruction decoder reads of ``[..., K * dim]`` states: the sum of the
        tokens (a linear map on the concatenation, tied across tokens; the identity for K = 1)."""
        k = self.cfg.summary_tokens
        return states if k == 1 else states.reshape(*states.shape[:-1], k, self.cfg.dim).sum(-2)

    def _reference_loss(
        self, batch: Mapping[str, Tensor], decoder: "reference.InputDecoder"
    ) -> dict[str, Tensor]:
        """Losses of the input-space reference objectives (``mae``, ``nextobs``)."""
        c = self.cfg
        pixels = reference.patchify(batch["vision"], c.vision_patch)
        proprio = reference.proprio_last(batch, self.senses)
        if c.objective == "mae":
            states, pix, prop_loss = self._masked_recon(batch, decoder, pre=False)
        else:
            states = self.embed(batch)
            cond = self.action(batch["action"].flatten(0, 1)).view_as(states)
            nxt = self.pred_proj(self.trunk(states[:, :-1], cond[:, :-1]))
            vis, prop = decoder(nxt)
            pix = (vis - pixels[:, 1:]).pow(2).mean()
            prop_loss = (prop - proprio[:, 1:]).pow(2).mean() if prop.size(-1) else pix * 0.0
        loss = pix + prop_loss
        return {
            "loss": loss,
            "pred": loss.detach(),
            "reg": torch.zeros((), device=states.device),
            "pix": pix.detach(),
            "prop": prop_loss.detach(),
            "states": states,
        }

    def _masked_recon(
        self, batch: Mapping[str, Tensor], decoder: "reference.InputDecoder", pre: bool
    ) -> tuple[Tensor, Tensor, Tensor]:
        """MAE-style masked reconstruction (``reference.mask_batch``): the states of the masked
        batch (projector output, or the summary token if ``pre``) and the pixel and proprio
        losses on the masked entries only."""
        c = self.cfg
        masked, hidden, channels = reference.mask_batch(
            batch, self.senses, c.vision_patch, c.mae_ratio, c.mae_proprio_ratio
        )
        view = self._view(masked)
        states = self.encoder.summary_token(view, drop=self.training) if pre else self.embed(masked)
        pix, prop_loss = self._recon_loss(
            batch, decoder, self.decoder_input(states), hidden, channels
        )
        return states, pix, prop_loss

    def _recon_loss(
        self,
        batch: Mapping[str, Tensor],
        decoder: "reference.InputDecoder",
        states: Tensor,
        hidden: Tensor,
        channels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Pixel and proprio losses of the decoder on ``states``, scored on the entries the
        masks hid (``reference.mask_batch``), against the clean batch."""
        c = self.cfg
        vis, prop = decoder(states)
        pixels = reference.patchify(batch["vision"], c.vision_patch)
        proprio = reference.proprio_last(batch, self.senses)
        err = (vis - pixels).pow(2).mean(-1)
        pix = (err * hidden).sum() / hidden.sum().clamp_min(1)
        prop_err = (prop - proprio).pow(2)
        if c.recon_proprio == "all":  # every channel, hidden or not
            channels = torch.ones_like(channels)
        prop_loss = (prop_err * channels).sum() / channels.sum().clamp_min(1)
        return pix, prop_loss

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Losses of one batch.

        Returns:
            ``loss``; ``pred`` (next-state MSE against the online states for ``sigreg``,
            against the EMA states for ``ema_vicreg``: not comparable across objectives);
            ``reg`` (the weighted regulariser, plus ``reg_var`` and ``reg_cov`` for
            ``ema_vicreg``); ``states`` (online).
        """
        if self.decoder is not None:
            return self._reference_loss(batch, self.decoder)
        c = self.cfg
        single: tuple[Tensor, Tensor, Tensor] | None = None
        if self.recon is not None and c.recon_input == "masked":
            masked, hidden, channels = reference.mask_batch(
                batch, self.senses, c.vision_patch, c.mae_ratio, c.mae_proprio_ratio
            )
            summary = self.encoder.summary_token(self._view(masked), drop=self.training)
            states = self.encoder.project(summary)
            read = states if c.recon_source == "s" else summary
            single = (self.decoder_input(read), hidden, channels)
        else:
            states = self.embed(batch)
        cond = self.action(batch["action"].flatten(0, 1)).view(*states.shape[:2], -1)
        trunk = self.trunk(states[:, :-1], cond[:, :-1])
        pred = self.state_out(self.pred_proj(trunk))
        extra: dict[str, Tensor] = {}
        if self.teacher is not None:
            has = batch["present"][:, self.data_senses.index("touch")].float()
            aux = self.teacher.aux_loss(batch, states, trunk, has)
            own = self.teacher.own_loss(batch, self.sigreg, c.sigreg_weight, c.sigreg_pool_time)
            extra = {"aux": aux.detach(), "teacher_own": own.detach()}
            extra_loss = c.teacher_weight * aux + own
        else:
            extra_loss = torch.zeros((), device=states.device)
        if self.target is None:
            nxt = states[:, 1:] if c.target_grad else states[:, 1:].detach()
            err = (pred - nxt).pow(2).mean(-1)  # [B, T-1]
            mse = err.mean()
            weighted = (surprise_weights(c.surprise, err, batch) * err).mean()
            reg = self.regulariser(states)
            if self.recon is not None:
                if single is not None:
                    pix, prop_loss = self._recon_loss(batch, self.recon, *single)
                else:
                    pre = c.recon_source == "z"
                    _, pix, prop_loss = self._masked_recon(batch, self.recon, pre=pre)
                recon = pix + prop_loss
                extra = {**extra, "recon": recon.detach()}
                extra_loss = extra_loss + c.recon * recon
            return {
                "loss": weighted + c.sigreg_weight * reg + extra_loss,
                "pred": mse,
                "reg": c.sigreg_weight * reg,
                "states": states,
                **extra,
            }
        with torch.no_grad():
            target = self.target(self._view(batch), drop=False)
        mse = (pred - target[:, 1:]).pow(2).mean()
        var, cov = vicreg_terms(states)
        reg = c.vicreg_var_weight * var + c.vicreg_cov_weight * cov
        return {
            "loss": mse + reg + extra_loss,
            "pred": mse,
            "reg": reg,
            "reg_var": var.detach(),
            "reg_cov": cov.detach(),
            "states": states,
            **extra,
        }


def collapse_metrics(states: Tensor) -> dict[str, float]:
    """Std of the states per dimension (mean over dimensions) and their effective rank.

    The effective rank is ``exp`` of the entropy of the normalised singular values of the
    centred ``[B * T, D]`` states: ``D`` for isotropic embeddings, 1 for a collapsed one.
    """
    x = states.detach().flatten(0, 1).float()
    x = x - x.mean(0, keepdim=True)
    sv = torch.linalg.svdvals(x)
    p = sv / sv.sum().clamp_min(1e-12)
    erank = torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum())
    return {"std": float(x.std(0).mean()), "erank": float(erank)}


@dataclass
class SensoryLayer:
    """A trained model together with what is needed to feed it episodes.

    Attributes:
        model: The trained ``SensoryModel`` (put in eval mode by ``encode``).
        space: The channel space it was built for.
        loader: How windows are cut (``substeps``) and cropped (``view_size``).
    """

    model: SensoryModel
    space: ChannelSpace
    loader: LoaderConfig = field(default_factory=LoaderConfig)

    @torch.no_grad()
    def encode(
        self, episode: Episode, stride: int = 1, batch: int = 32, drop: Sequence[str] = ()
    ) -> tuple[np.ndarray, np.ndarray]:
        """States of an episode, one per window of K observations.

        Inputs go through ``model_inputs`` (no privileged state, no text), the centre
        crop and no touch noise. The state of a window depends only on that window, so
        any ``stride`` is allowed; ``stride = K * spacing`` gives the training rate.

        Args:
            episode: Any episode of a body the layer's channel space contains.
            stride: Env steps between consecutive windows.
            batch: Windows encoded at once.
            drop: Senses to mask in the fusion (their tokens are ignored), to see what the
                states owe to each sense; the layer is not retrained for it.

        Returns:
            ``(states [n, dim], ends [n])``: the state and the index of each window's last step.

        Raises:
            ValueError: if the episode is shorter than one window.
        """
        k = self.model.cfg.substeps
        loaded = load_episode(episode, self.space, self.model.senses)
        first = (k - 1) * spacing(loaded.header.dt)
        if loaded.length <= first:
            raise ValueError(f"episode of {loaded.length} steps is shorter than one window")
        ends = np.arange(first, loaded.length, stride)
        was_training = self.model.training
        self.model.eval()
        device = next(self.model.parameters()).device
        states = []
        for i in range(0, len(ends), batch):
            e = ends[i : i + batch]
            rgb = None
            if "vision" in self.model.senses:
                rgb = eval_view(loaded.inputs["rgb"][e], self.loader.view)
            arrays = build(loaded, self.space, self.model.senses, e, k, rgb=rgb)
            arrays = dict(arrays)
            arrays["present"] = arrays["present"] & ~np.isin(self.model.senses, list(drop))
            batch_t = {k: v.to(device) for k, v in _as_batch(arrays, len(e)).items()}
            states.append(self.model.embed(batch_t).squeeze(1).cpu().numpy())
        self.model.train(was_training)
        return np.concatenate(states), ends


def _as_batch(arrays: Mapping[str, np.ndarray], n: int) -> dict[str, Tensor]:
    """Windows built by ``data.build`` -> a batch of ``n`` one-step sequences."""
    out = {
        k: torch.as_tensor(v)[:, None]
        for k, v in arrays.items()
        if k not in ("present", "region_mask")
    }
    out["present"] = torch.as_tensor(arrays["present"])[None].expand(n, -1)
    out["region_mask"] = torch.as_tensor(arrays["region_mask"])[None].expand(n, -1)
    return out


def count_parameters(model: SensoryModel) -> dict[str, int]:
    """Trainable parameters per part (each sense's stem on its own) and in total.

    The EMA copy of the encoder (``ema_vicreg``) is not trainable and is not counted.
    """
    train = {id(p) for p in model.parameters() if p.requires_grad}

    def count(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if id(p) in train)

    enc = model.encoder
    counts = {
        "fusion": count(enc.fusion),
        "projector": count(enc.projector),
        "action": count(model.action),
        "predictor": count(model.predictor),
        "pred_proj": count(model.pred_proj),
    }
    if model.teacher is not None:
        counts["teacher"] = count(model.teacher)
    for sense, stem in enc.stems.items():
        counts[f"stem.{sense}"] = count(stem)
    counts["slot/summary/pos"] = enc.slot.numel() + enc.summary.numel() + model.pos.numel()
    counts["total"] = sum(p.numel() for p in model.parameters() if id(p) in train)
    return counts
