"""Building blocks of the sensory layer: transformer blocks, stems and the SIGReg loss.

``SIGReg``, ``modulate`` and the AdaLN-zero block follow LeWorldModel (Maes, Le Lidec,
Scieur, LeCun and Balestriero 2026, arXiv:2603.19312; github.com/lucas-maes/le-wm,
MIT). The learned-query pooling of the stems follows HPT (Wang, Chen, Zhao and He 2024,
arXiv:2409.20537; github.com/liruiw/HPT, MIT). The code is rewritten, not copied,
except the SIGReg statistic.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """AdaLN modulation."""
    return x * (1 + scale) + shift


class SIGReg(nn.Module):
    """Sketched isotropic Gaussian regulariser (LeWM, MIT; the Epps-Pulley statistic).

    Projects embeddings on random unit directions and compares the empirical
    characteristic function of each 1-D projection with a standard normal's. Low for
    isotropic Gaussian embeddings, high when they collapse.

    Args:
        knots: Integration nodes on ``[0, 3]``.
        num_proj: Random directions per call.
    """

    t: Tensor
    phi: Tensor
    weights: Tensor

    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(
        self, emb: Tensor, pool_time: bool = False, generator: torch.Generator | None = None
    ) -> Tensor:
        """Mean statistic over time steps and projections.

        Args:
            emb: ``[B, T, D]`` embeddings. By default the statistic compares the B samples
                of each time step with a Gaussian (LeWM, batch 128); with ``pool_time`` it
                compares all B * T samples at once, which needs far fewer sequences per
                batch to keep the embedding from collapsing in rank.
            pool_time: Pool the time steps into one sample set.
            generator: Optional generator for the random directions (determinism).
        """
        if pool_time:
            emb = emb.reshape(-1, 1, emb.size(-1))
        proj = emb.transpose(0, 1).float()  # [T, B, D]
        a = torch.randn(proj.size(-1), self.num_proj, device=proj.device, generator=generator)
        a = a / a.norm(p=2, dim=0)
        x_t = (proj @ a).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        return ((err @ self.weights) * proj.size(-2)).mean()


class Block(nn.Module):
    """Pre-norm transformer block; with ``cond=True`` an AdaLN-zero block (LeWM).

    Args:
        dim: Width.
        heads: Attention heads.
        cond: Whether the block is modulated by a conditioning vector per token.
        mlp_ratio: Hidden width of the MLP over ``dim``.
    """

    def __init__(self, dim: int, heads: int, cond: bool = False, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim)
        )
        affine = not cond
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=affine, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=affine, eps=1e-6)
        self.modulation: nn.Sequential | None = None
        if cond:
            last = nn.Linear(dim, 6 * dim)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            self.modulation = nn.Sequential(nn.SiLU(), last)

    def attend(self, x: Tensor, mask: Tensor | None, causal: bool) -> Tensor:
        """Multi-head self-attention; ``mask`` is ``[N, 1, 1, S]`` bool (True = may attend)."""
        n, s, d = x.shape
        q, k, v = self.qkv(x).view(n, s, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=causal and mask is None
        )
        return self.proj(out.transpose(1, 2).reshape(n, s, d))

    def forward(
        self,
        x: Tensor,
        cond: Tensor | None = None,
        mask: Tensor | None = None,
        causal: bool = False,
    ) -> Tensor:
        """Apply the block to ``[N, S, D]`` tokens."""
        if self.modulation is None or cond is None:
            x = x + self.attend(self.norm1(x), mask, causal)
            return x + self.mlp(self.norm2(x))
        sh1, sc1, g1, sh2, sc2, g2 = self.modulation(cond).chunk(6, dim=-1)
        x = x + g1 * self.attend(modulate(self.norm1(x), sh1, sc1), mask, causal)
        return x + g2 * self.mlp(modulate(self.norm2(x), sh2, sc2))


class Encoder(nn.Module):
    """A stack of blocks followed by a LayerNorm."""

    def __init__(self, dim: int, depth: int, heads: int, cond: bool = False) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(Block(dim, heads, cond) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: Tensor,
        cond: Tensor | None = None,
        mask: Tensor | None = None,
        causal: bool = False,
    ) -> Tensor:
        """Apply all blocks to ``[N, S, D]`` tokens."""
        for block in self.blocks:
            x = block(x, cond, mask, causal)
        return self.norm(x)


class Pool(nn.Module):
    """Learned-query cross-attention pooling to a fixed number of tokens (HPT-style)."""

    def __init__(self, dim: int, tokens: int, heads: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, tokens, dim) * dim**-0.5)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """``[N, S, D]`` -> ``[N, tokens, D]``; ``key_padding_mask`` is True where ignored."""
        out, _ = self.attn(
            self.query.expand(x.size(0), -1, -1), x, x, key_padding_mask=key_padding_mask
        )
        return self.norm(out)


def sincos(n: int, dim: int) -> Tensor:
    """``[n, dim]`` 1-D sine-cosine position table."""
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    freq = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    table = torch.zeros(n, dim)
    table[:, 0::2], table[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
    return table


class VisionStem(nn.Module):
    """A small ViT (default ViT-Tiny: 192 wide, 12 layers) pooled to a few tokens.

    Args:
        size: Input side (px); frames are ``uint8`` ``[N, size, size, 3]``.
        patch: Patch side (px).
        dim, depth, heads: ViT width, depth and heads.
        tokens: Number of pooled output tokens.
        out_dim: Width of the output tokens (a linear projection after the pooling).
    """

    def __init__(
        self, size: int, patch: int, dim: int, depth: int, heads: int, tokens: int, out_dim: int
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        side = size // patch
        self.register_buffer("pos", sincos(side * side, dim)[None], persistent=False)
        self.encoder = Encoder(dim, depth, heads)
        self.pool = Pool(dim, tokens, heads)
        self.out = nn.Linear(dim, out_dim)

    def forward(self, frames: Tensor) -> Tensor:
        """``[N, H, W, 3]`` uint8 -> ``[N, tokens, out_dim]``."""
        x = (frames.permute(0, 3, 1, 2).float() / 255.0 - 0.5) / 0.5
        x = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos
        return self.out(self.pool(self.encoder(x)))


class UnionLinear(nn.Module):
    """A linear map from the K stacked sub-steps of a union channel group to ``dim``."""

    def __init__(self, substeps: int, channels: int, dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(substeps * channels, dim)

    def forward(self, x: Tensor) -> Tensor:
        """``[N, K, C]`` -> ``[N, dim]``."""
        return self.net(x.flatten(1))


class VectorStem(nn.Module):
    """MLP over a stacked channel group: one token (or one conditioning vector)."""

    def __init__(self, substeps: int, channels: int, dim: int, hidden: int) -> None:
        super().__init__()
        self.inp = UnionLinear(substeps, channels, hidden)
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, x: Tensor) -> Tensor:
        """``[N, K, C]`` -> ``[N, dim]``."""
        return self.out(self.inp(x))


class TouchStem(nn.Module):
    """Skin regions -> region tokens -> a few pooled tokens.

    Each region has the taxels and the 3 dynamics channels of one skin patch. A region
    token is a linear map of its K stacked sub-steps, with weights per *channel* (so the
    body skin is the same weights in hand and pincer episodes) plus an identity embedding
    from the region's name parts (block, side, segment). A small transformer mixes the
    region tokens (absent regions masked) and a learned-query pool reduces them.

    Args:
        table: ``[R, P]`` union touch column of each region's channels, ``-1`` padding.
        meta: Per region ``(block, side, part)`` (``ChannelSpace.region_meta``).
        n_channels: Union touch channels.
        substeps: K.
        dim, depth, heads: Width, depth and heads of the region transformer.
        tokens: Pooled output tokens.
        out_dim: Width of the output tokens (a linear projection after the pooling).
    """

    table: Tensor
    block: Tensor
    side: Tensor
    part: Tensor

    def __init__(
        self,
        table: Tensor,
        meta: list[tuple[int, int, str]],
        n_channels: int,
        substeps: int,
        dim: int,
        depth: int,
        heads: int,
        tokens: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        n_regions, width = table.shape
        self.substeps = substeps
        self.register_buffer("table", table, persistent=False)
        self.weight = nn.Parameter(
            torch.randn(substeps, n_channels, dim) * (width * substeps) ** -0.5
        )
        self.bias = nn.Parameter(torch.zeros(n_regions, dim))
        parts = sorted({m[2] for m in meta})
        self.register_buffer("block", torch.tensor([m[0] for m in meta]), persistent=False)
        self.register_buffer("side", torch.tensor([m[1] for m in meta]), persistent=False)
        self.register_buffer(
            "part", torch.tensor([parts.index(m[2]) for m in meta]), persistent=False
        )
        self.block_emb, self.side_emb = nn.Embedding(3, dim), nn.Embedding(3, dim)
        self.part_emb = nn.Embedding(len(parts), dim)
        self.encoder = Encoder(dim, depth, heads)
        self.pool = Pool(dim, tokens, heads)
        self.out = nn.Linear(dim, out_dim)

    def forward(self, touch: Tensor, region_mask: Tensor) -> Tensor:
        """``[N, K, U]`` touch and ``[N, R]`` bool region mask -> ``[N, tokens, out_dim]``."""
        valid = self.table >= 0
        safe = self.table.clamp(min=0)
        gathered = touch[:, :, safe] * valid  # [N, K, R, P]
        weight = self.weight[:, safe] * valid[None, :, :, None]  # [K, R, P, D]
        tokens = torch.einsum("nkrp,krpd->nrd", gathered, weight) + self.bias
        tokens = tokens + (
            self.block_emb(self.block) + self.side_emb(self.side) + self.part_emb(self.part)
        )
        # A sequence with no skin at all (a car) would mask every key: let it see its
        # tokens instead; the model masks the touch tokens of such sequences anyway.
        keep = region_mask | ~region_mask.any(-1, keepdim=True)
        pooled = self.pool(
            self.encoder(tokens, mask=keep[:, None, None, :]), key_padding_mask=~keep
        )
        return self.out(pooled)
