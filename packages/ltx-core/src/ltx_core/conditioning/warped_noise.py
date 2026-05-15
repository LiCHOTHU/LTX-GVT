"""Warped-noise initialisation for action-conditioned video diffusion (Go-with-the-Flow style).

Reference: Burgert et al., "Go-with-the-Flow: Motion-Controllable Video Diffusion Models
Using Real-Time Warped Noise", CVPR 2025 oral. https://arxiv.org/abs/2501.08331

The core idea: replace the standard i.i.d. Gaussian noise seed with a noise tensor whose
*temporal* statistics are warped along an optical flow field, while preserving *spatial*
Gaussianity. The video diffusion model trained (or fine-tuned) on this noise then generates
motion that follows the flow — for GVT this lets us bake the FK-derived robot motion into
the initial noise for free, on top of whatever conditioning (action tokens, IC-LoRA prior
video, PRoPE) we're already using.

This module is a self-contained PyTorch reimplementation of the essential mechanism. It
avoids the upstream ``rp`` / ``CommonSource`` dependency by replacing those helpers with
``torch.nn.functional`` equivalents. It captures the *backward-gather + regaussianize*
path, which is the simplest variant that preserves unit variance in expectation; the
upstream code additionally implements a forward-scatter (shrink) path for higher fidelity
on fast-flow regions.

Public API:

    apply_warped_noise(noise, flow)  -> torch.Tensor
        The main entry point. Takes a video noise latent ``(B, C, F, H, W)`` and a per-
        frame-transition flow ``(B, 2, F-1, H, W)`` and returns warped noise of the same
        shape. The first frame is left as-is; frame ``t`` for ``t > 0`` is the flow-warp
        of frame ``t-1``, regaussianised to unit variance.

    regaussianize(x)  -> torch.Tensor
        Per-frame zero-mean unit-variance normalisation, used internally and exposed for
        custom pipelines.

All operations are differentiable so the function can sit inside a training-time graph if
desired, though the typical use is at sample time (no gradient through the noise warp).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _build_sample_grid(flow_dxdy: torch.Tensor) -> torch.Tensor:
    """Construct an absolute sample grid for ``F.grid_sample`` from a relative flow.

    Args:
        flow_dxdy: ``(B, 2, H, W)`` flow in pixel units; flow[:, 0] is ``dx``, flow[:, 1]
            is ``dy``. Sign convention: a pixel that should *move to* ``(x + dx, y + dy)``
            in the output gets sampled *from* ``(x - dx, y - dy)`` in the input — i.e.
            this is the backward-gather grid for warping the *previous* frame *forward*
            along the flow.

    Returns:
        Sample grid of shape ``(B, H, W, 2)`` with values in ``[-1, 1]``, ready for
        ``F.grid_sample(..., align_corners=False)``.
    """
    b, _, h, w = flow_dxdy.shape
    device = flow_dxdy.device
    dtype = flow_dxdy.dtype

    # Base grid in pixel coordinates (y-first to match torch's NCHW convention).
    ys = torch.arange(h, device=device, dtype=dtype)
    xs = torch.arange(w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W) each
    grid_x = grid_x.unsqueeze(0).expand(b, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(b, -1, -1)

    # Backward gather: sample from (x - dx, y - dy) to warp pixel at (x, y) FORWARD.
    src_x = grid_x - flow_dxdy[:, 0]
    src_y = grid_y - flow_dxdy[:, 1]

    # Normalise to [-1, 1] for grid_sample (align_corners=False convention).
    norm_x = 2.0 * (src_x + 0.5) / w - 1.0
    norm_y = 2.0 * (src_y + 0.5) / h - 1.0

    return torch.stack([norm_x, norm_y], dim=-1)  # (B, H, W, 2)


def regaussianize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-frame zero-mean unit-variance normalisation across spatial axes.

    Args:
        x: Tensor of shape ``(B, C, H, W)`` or ``(B, C, F, H, W)``.
        eps: Numerical floor on the std.

    Returns:
        Tensor of the same shape; each ``(C, H, W)`` slab (per batch, per frame) is
        normalised independently to mean 0 / std 1.
    """
    if x.ndim == 4:
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True, unbiased=False)
    elif x.ndim == 5:
        mean = x.mean(dim=(3, 4), keepdim=True)
        std = x.std(dim=(3, 4), keepdim=True, unbiased=False)
    else:
        raise ValueError(f"x must be 4-D or 5-D, got shape {tuple(x.shape)}")
    return (x - mean) / std.clamp(min=eps)


def apply_warped_noise(noise: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp an i.i.d. Gaussian noise latent along a flow field, preserving spatial Gaussianity.

    Args:
        noise: Initial noise of shape ``(B, C, F, H, W)``. Assumed to be sampled i.i.d. from
            a unit-variance Gaussian, but the function only requires zero-mean / unit-std
            *per frame* (which is what ``torch.randn`` gives you in expectation).
        flow: Per-frame-transition flow ``(B, 2, F-1, H, W)``. Index ``t`` contains the
            (dx, dy) displacement, in pixel units of the noise's spatial grid, of pixels
            going from frame ``t`` to frame ``t+1``. If you have RGB flow at a higher
            resolution, downsample it to the latent grid before passing in.

    Returns:
        Warped noise of shape ``(B, C, F, H, W)``. Frame 0 is the input noise's frame 0
        unchanged; frame ``t > 0`` is ``regaussianize(grid_sample(frame_{t-1}, flow_{t-1}))``.

    Notes:
        Sign convention for the flow: positive ``dx`` means the corresponding pixel moves
        rightward by ``dx`` pixels between consecutive frames; positive ``dy`` means it
        moves downward. This matches the FK-derived flow we compute from the action stream
        (project_points difference between consecutive frames in image coordinates).
    """
    if noise.ndim != 5:
        raise ValueError(f"noise must be (B, C, F, H, W), got shape {tuple(noise.shape)}")
    b, c, f_frames, h, w = noise.shape
    if flow.shape != (b, 2, f_frames - 1, h, w):
        raise ValueError(
            f"flow must have shape (B, 2, F-1, H, W) = ({b}, 2, {f_frames - 1}, {h}, {w}); "
            f"got {tuple(flow.shape)}"
        )

    # Build the output frame-by-frame as a list to keep the autograd graph clean
    # (in-place assignment into a pre-allocated buffer breaks gradient flow).
    frames: list[torch.Tensor] = [noise[:, :, 0]]
    for t in range(1, f_frames):
        prev = frames[-1]                              # (B, C, H, W)
        flow_t = flow[:, :, t - 1]                     # (B, 2, H, W)
        sample_grid = _build_sample_grid(flow_t)       # (B, H, W, 2)
        warped = F.grid_sample(
            prev, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )                                              # (B, C, H, W)
        frames.append(regaussianize(warped))
    return torch.stack(frames, dim=2)
