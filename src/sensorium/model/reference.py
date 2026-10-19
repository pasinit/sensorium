"""Reference objectives in input space, to tell the JEPA objective from the setting.

Three studies put every pooled-SIGReg JEPA arm at or below its random-init floor. Two standard
objectives that predict *inputs* instead of latents, on the same encoder, data, steps and probe
battery, say whether the objective or the setting (scale, probe) is the problem:

* ``mae``: masked reconstruction (He et al. 2022). 75% of the frame's patches are grayed out and
  half of the proprio channels zeroed *in the encoder's input*; a decoder reads the encoder's state
  and reconstructs the masked patches (normalised pixels) and the masked proprio values, loss on
  the masked entries only. Unlike MAE the masking is on pixels, not on dropped tokens, and the
  decoder sees only the one state vector (which is what the probes read);
* ``nextobs``: next-observation prediction in input space. The action-conditioned predictor maps
  the state of step ``t`` to the next state and the decoder turns *that* into the whole next frame
  (all patches) and the next proprio, MSE; the targets are inputs, so nothing can collapse and no
  regulariser is used.

Both support the vision and proprio senses only (touch and audio have no decoder here).
"""

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

PROPRIO_SENSES = ("proprio_body", "vestibular", "proprio_effector")
GRAY = 128


class InputDecoder(nn.Module):
    """States ``[..., dim]`` -> normalised frame patches ``[..., P, patch*patch*3]`` and proprio.

    Args:
        dim: State width.
        hidden: Width of the decoder's hidden layers.
        patches: Number of patches of a frame.
        patch_dim: Values per patch (``patch * patch * 3``).
        proprio_dim: Number of proprio channels reconstructed (0: none).
    """

    def __init__(self, dim: int, hidden: int, patches: int, patch_dim: int, proprio_dim: int):
        super().__init__()
        self.patches, self.patch_dim = patches, patch_dim
        self.vision = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, patches * patch_dim)
        )
        self.proprio = (
            nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, proprio_dim))
            if proprio_dim
            else None
        )

    def forward(self, states: Tensor) -> tuple[Tensor, Tensor]:
        """``(patches [..., P, pd], proprio [..., C])`` (proprio is empty without channels)."""
        patches = self.vision(states).unflatten(-1, (self.patches, self.patch_dim))
        proprio = self.proprio(states) if self.proprio is not None else states[..., :0]
        return patches, proprio


def patchify(frames: Tensor, patch: int) -> Tensor:
    """``[B, T, H, W, 3]`` uint8 -> ``[B, T, P, patch*patch*3]`` pixels scaled to ``[-1, 1]``."""
    b, t, h, w, _ = frames.shape
    x = (frames.float() / 255.0 - 0.5) / 0.5
    x = x.view(b, t, h // patch, patch, w // patch, patch, 3).permute(0, 1, 2, 4, 3, 5, 6)
    return x.reshape(b, t, (h // patch) * (w // patch), patch * patch * 3)


def proprio_last(batch: Mapping[str, Tensor], senses: Sequence[str]) -> Tensor:
    """The last observation of every proprio sense, concatenated: ``[B, T, C]``."""
    parts = [batch[s][:, :, -1, :].float() for s in senses if s in PROPRIO_SENSES]
    return torch.cat(parts, dim=-1) if parts else batch["action"][..., :0, 0].float()


def mask_batch(
    batch: Mapping[str, Tensor], senses: Sequence[str], patch: int, ratio: float, prop_ratio: float
) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    """Mask the encoder's inputs MAE-style.

    Returns:
        ``(masked batch, patch_mask [B, T, P] bool, channel_mask [B, T, C] bool)``; the masks are
        True where the input was hidden (and the decoder is scored).
    """
    out = dict(batch)
    frames = batch["vision"]
    b, t, h, w, _ = frames.shape
    side = h // patch
    hidden = torch.rand(b, t, side, side, device=frames.device) < ratio
    pixels = hidden.repeat_interleave(patch, 2).repeat_interleave(patch, 3)
    out["vision"] = torch.where(pixels[..., None], torch.full_like(frames, GRAY), frames)
    masks = []
    for sense in senses:
        if sense not in PROPRIO_SENSES:
            continue
        x = batch[sense]
        gone = torch.rand(x.size(0), x.size(1), 1, x.size(3), device=x.device) < prop_ratio
        out[sense] = x.masked_fill(gone, 0.0)
        masks.append(gone[:, :, 0, :])
    channels = torch.cat(masks, dim=-1) if masks else hidden.new_zeros(b, t, 0)
    return out, hidden.flatten(2), channels
