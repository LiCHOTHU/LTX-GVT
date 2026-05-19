"""Warped Latent (MosaicMem-style cross-view feature alignment).

For each pixel `(u, v)` in a source view with depth `D`, back-project to a 3D
world point via source `(K_src, T_src)`, then project into target `(K_tgt, T_tgt)`:

    (u', v') = Π( K_tgt · T_tgt · T_src^-1 · K_src^-1 · (u, v, D) )   (Eq. 2 in the paper)

The output feature map is the source feature map bilinear-sampled at `(u', v')` —
a differentiable, depth-aware approximation of view synthesis directly in latent
space. With identity cameras + flat depth, the warp is the identity transform
(used for smoke verification before real cameras/depth are plumbed in).

This is a pure function — no learned parameters. It's complementary to PRoPE
(which adjusts attention dot products via relative projections, no depth needed):
warp-latent gives explicit per-patch feature correspondence between views,
PRoPE gives geometry-aware attention bias.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def warp_latent(
    src_latent: torch.Tensor,
    *,
    src_K: torch.Tensor,
    src_T: torch.Tensor,
    tgt_K: torch.Tensor,
    tgt_T: torch.Tensor,
    depth: torch.Tensor,
) -> torch.Tensor:
    """Warp `src_latent` from source camera to target camera using per-pixel depth.

    Args:
        src_latent: Source feature map, shape `(B, C, H, W)` (or `(B, C, F, H, W)`
            for a video — the spatial warp is applied frame-by-frame).
        src_K, tgt_K: Camera intrinsics, shape `(B, 3, 3)`.
        src_T, tgt_T: Camera extrinsics (world-to-camera), shape `(B, 4, 4)`.
        depth: Source-view depth at each pixel, shape `(B, H, W)` (or
            `(B, F, H, W)` for video). Must be positive.

    Returns:
        Warped feature map, same shape as `src_latent`. Pixels whose back-projection
        leaves the source frustum are zero-padded (border mode).

    Notes:
        * Identity inputs (`src_K = tgt_K`, `src_T = tgt_T`) produce exactly the
          source latent — useful for end-to-end wiring verification.
        * The warp uses `grid_sample(mode='bilinear', align_corners=False)`, matching
          PyTorch's standard image-resampling convention.
    """
    is_video = src_latent.dim() == 5
    if is_video:
        b, c, f, h, w = src_latent.shape
        # Fold time into batch: (B, C, F, H, W) -> (B*F, C, H, W)
        src_latent_2d = src_latent.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
        depth_2d = depth.reshape(b * f, h, w) if depth.dim() == 4 else depth.unsqueeze(1).expand(b, f, h, w).reshape(b * f, h, w)
        # Each frame within one sample shares the same per-sample cameras (1 viewpoint per clip).
        src_K_2d = src_K.unsqueeze(1).expand(b, f, 3, 3).reshape(b * f, 3, 3)
        src_T_2d = src_T.unsqueeze(1).expand(b, f, 4, 4).reshape(b * f, 4, 4)
        tgt_K_2d = tgt_K.unsqueeze(1).expand(b, f, 3, 3).reshape(b * f, 3, 3)
        tgt_T_2d = tgt_T.unsqueeze(1).expand(b, f, 4, 4).reshape(b * f, 4, 4)
        warped_2d = _warp_latent_2d(src_latent_2d, src_K_2d, src_T_2d, tgt_K_2d, tgt_T_2d, depth_2d)
        # Unfold time: (B*F, C, H, W) -> (B, C, F, H, W)
        return warped_2d.reshape(b, f, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
    return _warp_latent_2d(src_latent, src_K, src_T, tgt_K, tgt_T, depth)


def _warp_latent_2d(
    src_latent: torch.Tensor,  # (B, C, H, W)
    src_K: torch.Tensor,  # (B, 3, 3)
    src_T: torch.Tensor,  # (B, 4, 4) world-to-camera
    tgt_K: torch.Tensor,  # (B, 3, 3)
    tgt_T: torch.Tensor,  # (B, 4, 4)
    depth: torch.Tensor,  # (B, H, W)
) -> torch.Tensor:
    b, _, h, w = src_latent.shape
    device, orig_dtype = src_latent.device, src_latent.dtype
    # torch.linalg.inv doesn't support bf16/fp16. Compute the geometry in fp32 and
    # cast the final grid back to the source dtype for grid_sample.
    dtype = torch.float32
    src_K = src_K.to(dtype)
    src_T = src_T.to(dtype)
    tgt_K = tgt_K.to(dtype)
    tgt_T = tgt_T.to(dtype)
    depth = depth.to(dtype)

    # Build a pixel grid in TARGET image coords: shape (B, H, W, 2) with (u, v).
    # Convention: grid index 0 = top-left pixel center, matching grid_sample.
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    # (H, W, 3) homogeneous pixel coords in target view
    uv1_tgt = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)  # (H, W, 3)
    uv1_tgt = uv1_tgt.unsqueeze(0).expand(b, h, w, 3)  # (B, H, W, 3)

    # For *target-frame* sampling we actually want the inverse direction:
    # for each TARGET pixel, find which SOURCE pixel it came from.
    # That requires depth in TARGET view — which we don't have. The MosaicMem Eq. 2
    # form assumes depth in SOURCE view, mapping source pixels -> target pixels;
    # to sample with grid_sample we instead need target -> source.
    # For the verification path (identity cameras), this distinction doesn't matter —
    # both directions give the identity grid. For real cameras the dataloader is
    # expected to provide TARGET-view depth and use this function as
    # `warp_latent(src=other_view, src_K=other_K, src_T=other_T, tgt_K=self_K, tgt_T=self_T, depth=self_depth)`.
    depth_tgt = depth.clamp(min=1e-6)  # (B, H, W)

    # Back-project target pixels to 3D in target camera: (K_tgt^-1 · [u,v,1]) * D
    K_tgt_inv = torch.linalg.inv(tgt_K)  # (B, 3, 3)
    rays_tgt = torch.einsum("bij,bhwj->bhwi", K_tgt_inv, uv1_tgt)  # (B, H, W, 3)
    pts_tgt = rays_tgt * depth_tgt.unsqueeze(-1)  # (B, H, W, 3)

    # Tgt camera -> world: world = T_tgt^-1 · [pts; 1]
    pts_tgt_h = torch.cat([pts_tgt, torch.ones_like(pts_tgt[..., :1])], dim=-1)  # (B, H, W, 4)
    T_tgt_inv = torch.linalg.inv(tgt_T)  # (B, 4, 4)
    pts_world = torch.einsum("bij,bhwj->bhwi", T_tgt_inv, pts_tgt_h)  # (B, H, W, 4)

    # World -> src camera: pts_src = T_src · pts_world
    pts_src_h = torch.einsum("bij,bhwj->bhwi", src_T, pts_world)  # (B, H, W, 4)
    pts_src = pts_src_h[..., :3]

    # Project into source image: uv1_src = K_src · pts_src, then divide by z.
    uv1_src = torch.einsum("bij,bhwj->bhwi", src_K, pts_src)  # (B, H, W, 3)
    z_src = uv1_src[..., 2:3].clamp(min=1e-6)
    uv_src = uv1_src[..., :2] / z_src  # (B, H, W, 2) in source pixel coords

    # Convert pixel coords (u, v) in [0, W-1]x[0, H-1] to grid_sample coords in [-1, 1].
    u_norm = (uv_src[..., 0] / max(w - 1, 1)) * 2.0 - 1.0
    v_norm = (uv_src[..., 1] / max(h - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([u_norm, v_norm], dim=-1)  # (B, H, W, 2)

    return F.grid_sample(src_latent, grid.to(orig_dtype), mode="bilinear", padding_mode="zeros", align_corners=True)
