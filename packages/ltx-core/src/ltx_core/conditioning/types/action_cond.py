"""Raw-action conditioning for action-conditioned video generation (GVT / TIA2V).

The rendered context video (the per-frame Franka render) rides on the existing
``VideoConditionByReferenceLatent`` path — it is image-shaped and already encoded
through the VAE. This module adds the complementary path for the **raw numeric
action vector** (e.g. 7 Franka joint commands + 1 gripper, per timestep), which
has no pixel representation: a small MLP projects each per-frame action vector
into the transformer's hidden dim, and the resulting tokens are concatenated to
the latent sequence MosaicMem-style. Self-attention then mixes the action tokens
with the video tokens; the model learns to use them as a precise pose hint.

Temporal RoPE positions are aligned with the latent frame grid (one action token
per latent frame). Spatial RoPE positions are pinned to a fixed negative sentinel
so the action tokens are positionally distinguishable from any real video patch.
"""

from __future__ import annotations

import torch
from torch import nn

from ltx_core.conditioning.item import ConditioningItem
from ltx_core.conditioning.mask_utils import update_attention_mask
from ltx_core.tools import VideoLatentTools
from ltx_core.types import LatentState

ACTION_SPATIAL_SENTINEL: float = -1.0


class ActionMLPProjector(nn.Module):
    """Project per-frame action vectors into transformer hidden tokens.

    ``forward((B, F, action_dim)) -> (B, F, hidden_dim)`` where ``F`` is the latent
    frame count. The action stream must be pre-resampled to that latent rate by the
    dataloader (one action sample per latent frame).

    Two-layer SiLU MLP. Intended to be trained from scratch as part of a LoRA fine-tune;
    its weights live alongside (not inside) the base LTX-2 checkpoint.
    """

    def __init__(self, action_dim: int, hidden_dim: int, mid_dim: int | None = None) -> None:
        super().__init__()
        if action_dim <= 0 or hidden_dim <= 0:
            raise ValueError(f"action_dim and hidden_dim must be positive, got {action_dim}, {hidden_dim}")
        mid = mid_dim if mid_dim is not None else max(4 * action_dim, hidden_dim // 4)
        self.net = nn.Sequential(
            nn.Linear(action_dim, mid),
            nn.SiLU(),
            nn.Linear(mid, hidden_dim),
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        if action.dim() != 3:
            raise ValueError(f"action must be (B, F, action_dim), got shape {tuple(action.shape)}")
        return self.net(action)


class VideoConditionByActionTokens(ConditioningItem):
    """Append projected action tokens to the video latent sequence (MosaicMem-style concat).

    Args:
        tokens: Projected action tokens of shape ``(B, F_latent, D_hidden)``, where
            ``F_latent`` matches the target video's latent frame count and ``D_hidden``
            matches the transformer's input dim. Produced by :class:`ActionMLPProjector`.
        strength: ``1.0`` (default) keeps the action tokens clean / non-denoised, the
            standard for control-style conditioning. ``0.0`` would treat them as another
            noisy stream to denoise (unusual; provided for symmetry with the reference-
            video path).
    """

    def __init__(self, tokens: torch.Tensor, strength: float = 1.0) -> None:
        if tokens.dim() != 3:
            raise ValueError(f"tokens must be (B, F, D), got shape {tuple(tokens.shape)}")
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"strength must be in [0, 1], got {strength}")
        self.tokens = tokens
        self.strength = strength

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: VideoLatentTools,
    ) -> LatentState:
        b, n_action, _ = self.tokens.shape
        device = latent_state.latent.device
        dtype = latent_state.latent.dtype

        # Temporal positions in seconds, mirroring the convention used by
        # `get_pixel_coords` + the pipeline's fps division (see reference_video_cond.py).
        # One action token per latent frame, so latent_coord = [f, f+1).
        scale_t = float(latent_tools.scale_factors.time)
        fps = float(latent_tools.fps)
        frame_idx = torch.arange(n_action, dtype=torch.float32, device=device)
        pixel_start = frame_idx * scale_t
        pixel_end = (frame_idx + 1.0) * scale_t
        if latent_tools.causal_fix:
            pixel_start = (pixel_start + 1.0 - scale_t).clamp(min=0)
            pixel_end = (pixel_end + 1.0 - scale_t).clamp(min=0)
        time_bounds = torch.stack([pixel_start, pixel_end], dim=-1) / fps  # (T, 2)

        # Spatial axes pinned to a fixed negative sentinel — clearly out-of-band vs
        # any real video patch (whose pixel coords are non-negative).
        spatial_bounds = torch.full(
            (n_action, 2),
            fill_value=ACTION_SPATIAL_SENTINEL,
            device=device,
            dtype=torch.float32,
        )

        # Stack into (3, T, 2) — axes (time, height, width) — then broadcast to (B, 3, T, 2)
        # to match get_pixel_coords' output layout.
        positions = torch.stack([time_bounds, spatial_bounds, spatial_bounds], dim=0)
        positions = positions.unsqueeze(0).expand(b, -1, -1, -1).contiguous()

        tokens = self.tokens.to(device=device, dtype=dtype)

        # Clean conditioning by default (strength=1.0 -> denoise weight 0).
        denoise_mask = torch.full(
            size=(b, n_action, 1),
            fill_value=1.0 - self.strength,
            device=latent_state.denoise_mask.device,
            dtype=latent_state.denoise_mask.dtype,
        )

        new_attention_mask = update_attention_mask(
            latent_state=latent_state,
            attention_mask=None,
            num_noisy_tokens=latent_tools.target_shape.token_count(),
            num_new_tokens=n_action,
            batch_size=b,
            device=device,
            dtype=dtype,
        )

        return LatentState(
            latent=torch.cat([latent_state.latent, tokens], dim=1),
            denoise_mask=torch.cat([latent_state.denoise_mask, denoise_mask], dim=1),
            positions=torch.cat(
                [latent_state.positions, positions.to(latent_state.positions)],
                dim=2,
            ),
            clean_latent=torch.cat([latent_state.clean_latent, tokens], dim=1),
            attention_mask=new_attention_mask,
        )
