"""Analytical motion flow from a Franka joint trajectory.

For each frame transition the function projects the 9 Franka link keypoints into a
given camera, takes the per-keypoint pixel displacement, and propagates it to nearby
pixels (nearest-keypoint within a radius). This gives a sparse 2-D flow field that
agrees exactly with what optical flow would recover *on the arm*; background pixels
are zero by construction. The result is suitable as a motion prior for warped-noise
video diffusion (Go-with-the-Flow style) without running an external flow estimator.

Conventions:
  * Joint stream shape ``(T, 7)`` of float radians.
  * Intrinsics ``K`` shape ``(3, 3)``, expressed in the same pixel grid as the
    visualisation resolution.
  * Extrinsics ``cam2base`` shape ``(4, 4)`` (camera-frame -> base-frame).
    To project base-frame points into the camera we invert this internally.
  * For the wrist camera, ``cam2base`` is time-varying; pass a per-frame
    ``cam2base`` of shape ``(T, 4, 4)`` instead of a single matrix.

Outputs ``(T-1, 2, H, W)`` where channel 0 is ``dx`` (positive = rightward in the
image plane) and channel 1 is ``dy`` (positive = downward). Flow is in pixel units
of the image grid; downsample to the latent grid by mean-pooling along (H, W) and
sub-sampling along T as the caller sees fit.
"""

from __future__ import annotations

import numpy as np

from ltx_action_cond.kinematics import fk
from ltx_action_cond.projection import project_points


def _project_keypoints_all_frames(
    joints: np.ndarray,
    K: np.ndarray,
    cam2base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run FK on every frame and project keypoints into the camera.

    Returns ``(pix, in_front)`` where ``pix`` is ``(T, 9, 2)`` pixel coords and
    ``in_front`` is ``(T, 9)`` bool — True when the keypoint has positive depth.
    Accepts a static ``cam2base`` ``(4, 4)`` or a per-frame ``(T, 4, 4)``.
    """
    T = joints.shape[0]
    if cam2base.ndim == 2:
        cam2base_per_frame = np.broadcast_to(cam2base, (T, 4, 4))
    elif cam2base.ndim == 3 and cam2base.shape == (T, 4, 4):
        cam2base_per_frame = cam2base
    else:
        raise ValueError(f"cam2base must be (4,4) or (T,4,4), got shape {cam2base.shape}")

    pix = np.zeros((T, 9, 2), dtype=np.float64)
    in_front = np.zeros((T, 9), dtype=bool)
    for t in range(T):
        kpts = fk(joints[t])                                          # (9, 3) base-frame
        base2cam_t = np.linalg.inv(cam2base_per_frame[t])             # (4, 4)
        pix_t, z_t = project_points(kpts, base2cam_t, K)              # (9, 2), (9,)
        pix[t] = pix_t
        in_front[t] = z_t > 0
    return pix, in_front


def scene_plane_flow_for_view(
    K: np.ndarray,
    cam2base: np.ndarray,
    image_size: tuple[int, int],
    plane_z: float = 0.0,
) -> np.ndarray:
    """Per-pixel optical flow from camera egomotion, assuming the scene lies on a plane.

    Each pixel is back-projected through the camera at frame ``t`` onto the plane
    ``z = plane_z`` (base-frame), then forward-projected through the camera at frame
    ``t+1``. The difference is the egomotion-induced flow that scene point would have.
    For a fixed camera (static ``cam2base``) this is identically zero; for the wrist
    camera, which rides on the hand, it captures the dominant motion in the view
    (table sliding past the camera).

    Args:
        K: ``(3, 3)`` intrinsic matrix in pixels.
        cam2base: ``(T, 4, 4)`` per-frame camera-to-base extrinsic (the wrist case).
            A static ``(4, 4)`` is also accepted but will trivially produce zero flow.
        image_size: ``(W, H)`` pixel size.
        plane_z: Plane height in base coords (Franka tables are typically at z=0).

    Returns:
        Flow ``(T-1, 2, H, W)`` float32. Pixels whose back-projected ray is parallel
        to (or below) the plane get zero flow.
    """
    W, H = image_size
    if cam2base.ndim == 2:
        cam2base = np.broadcast_to(cam2base, (1, 4, 4))
    T = cam2base.shape[0]
    if T < 2:
        return np.zeros((0, 2, H, W), dtype=np.float32)

    # Per-pixel rays in camera frame at every pixel coordinate.
    K_inv = np.linalg.inv(K).astype(np.float64)
    ys = np.arange(H, dtype=np.float64) + 0.5  # pixel centres
    xs = np.arange(W, dtype=np.float64) + 0.5
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    ones = np.ones_like(grid_x)
    homog = np.stack([grid_x, grid_y, ones], axis=-1)                  # (H, W, 3)
    rays_cam = homog @ K_inv.T                                          # (H, W, 3)

    flow = np.zeros((T - 1, 2, H, W), dtype=np.float32)
    for t in range(T - 1):
        R_t = cam2base[t, :3, :3]; o_t = cam2base[t, :3, 3]
        # Ray in base frame at frame t.
        rays_base = rays_cam @ R_t.T                                    # (H, W, 3)
        denom = rays_base[..., 2]
        # Distance along ray to plane z=plane_z (skip pixels with dir.z ~ 0 or behind).
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (plane_z - o_t[2]) / denom
        valid = np.isfinite(s) & (s > 0)
        s = np.where(valid, s, 0.0)
        # 3-D point on the plane in base frame.
        P_base = o_t[None, None, :] + rays_base * s[..., None]          # (H, W, 3)

        # Project that point through the camera at frame t+1.
        base2cam_tp1 = np.linalg.inv(cam2base[t + 1])
        Rn = base2cam_tp1[:3, :3]; tn = base2cam_tp1[:3, 3]
        P_cam_tp1 = P_base @ Rn.T + tn[None, None, :]                   # (H, W, 3)
        z_tp1 = P_cam_tp1[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u_tp1 = K[0, 0] * P_cam_tp1[..., 0] / z_tp1 + K[0, 2]
            v_tp1 = K[1, 1] * P_cam_tp1[..., 1] / z_tp1 + K[1, 2]
        valid_proj = valid & (z_tp1 > 0)
        dx = np.where(valid_proj, u_tp1 - grid_x, 0.0).astype(np.float32)
        dy = np.where(valid_proj, v_tp1 - grid_y, 0.0).astype(np.float32)
        flow[t, 0] = dx
        flow[t, 1] = dy
    return flow


def flow_from_joints_for_view(
    joints: np.ndarray,
    K: np.ndarray,
    cam2base: np.ndarray,
    image_size: tuple[int, int],
    radius_pixels: float = 18.0,
    max_speed_pixels: float = 60.0,
    plane_z: float | None = None,
) -> np.ndarray:
    """Analytical per-pixel optical flow for a single camera view.

    Args:
        joints: ``(T, 7)`` Franka joint trajectory.
        K: ``(3, 3)`` intrinsic matrix at the target ``image_size`` resolution.
        cam2base: ``(4, 4)`` static extrinsic OR ``(T, 4, 4)`` per-frame extrinsic
            (the latter for the wrist camera, which rides on the hand).
        image_size: ``(W, H)`` target image size in pixels.
        radius_pixels: How far a pixel can be from a Franka keypoint and still
            inherit its velocity. A pixel beyond this radius from every keypoint
            uses the background convention (see ``plane_z``).
        max_speed_pixels: Clip per-keypoint frame-to-frame velocities at this
            magnitude to suppress numerical blowups near the camera plane.
        plane_z: If set, the background (off-arm) flow is computed via
            :func:`scene_plane_flow_for_view` at this plane height instead of being
            zeroed. For fixed exterior cameras this is a no-op (static cam2base
            yields zero plane flow); for the wrist camera (per-frame cam2base) this
            captures the dominant egomotion of the scene under the camera. A common
            value for Franka setups is ``0.0`` (table at base-frame z=0).

    Returns:
        Flow ``(T-1, 2, H, W)`` float32 in pixel units. Channel 0 = dx, 1 = dy.
    """
    if joints.ndim != 2 or joints.shape[1] != 7:
        raise ValueError(f"joints must be (T, 7), got {joints.shape}")
    W, H = image_size
    if H <= 0 or W <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")

    # 1. Project keypoints into every frame.
    pix_T9, in_front_T9 = _project_keypoints_all_frames(joints, K, cam2base)
    T = pix_T9.shape[0]
    if T < 2:
        return np.zeros((0, 2, H, W), dtype=np.float32)

    # 2. Per-frame-transition keypoint velocities. Mask out keypoints that are
    # behind the camera or off-screen in either endpoint, and clip outliers.
    velocities = pix_T9[1:] - pix_T9[:-1]                              # (T-1, 9, 2)
    in_front_pair = in_front_T9[:-1] & in_front_T9[1:]                 # (T-1, 9)
    speed = np.linalg.norm(velocities, axis=-1)                        # (T-1, 9)
    valid = in_front_pair & (speed < max_speed_pixels)                 # (T-1, 9)

    # 3. Background flow: zero for fixed exteriors, scene-plane egomotion when plane_z
    # is supplied (this is what makes the wrist view's flow correct).
    if plane_z is not None:
        flow = scene_plane_flow_for_view(K, cam2base, (W, H), plane_z=plane_z)
    else:
        flow = np.zeros((T - 1, 2, H, W), dtype=np.float32)

    # 4. For each pixel, find the nearest keypoint at the *current* frame and assign
    # its velocity if within radius. Vectorised across pixels per frame.
    ys = np.arange(H, dtype=np.float32)
    xs = np.arange(W, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")                # (H, W) each
    grid_xy = np.stack([grid_x, grid_y], axis=-1)                       # (H, W, 2)

    radius_sq = radius_pixels ** 2

    for t in range(T - 1):
        kp_xy = pix_T9[t]                                              # (9, 2)
        vel = velocities[t]                                            # (9, 2)
        mask = valid[t]                                                # (9,)
        if not mask.any():
            continue
        kp_xy = kp_xy[mask]                                            # (K_valid, 2)
        vel = vel[mask]                                                # (K_valid, 2)

        # Pairwise squared distances pixel -> keypoint. Use broadcasting.
        # (H, W, 1, 2) - (K_valid, 2) -> (H, W, K_valid, 2) -> sum along last
        diff = grid_xy[:, :, None, :] - kp_xy[None, None, :, :]        # (H, W, K, 2)
        dist_sq = (diff ** 2).sum(axis=-1)                              # (H, W, K)
        nearest_idx = dist_sq.argmin(axis=-1)                           # (H, W)
        nearest_d2 = np.take_along_axis(dist_sq, nearest_idx[..., None], axis=-1)[..., 0]
        in_range = nearest_d2 <= radius_sq                              # (H, W)

        # Gather per-pixel velocity from the nearest valid keypoint. Where the pixel
        # is within radius, OVERRIDE the background plane flow with the arm flow;
        # outside that radius, keep whatever the background carries (zero for fixed
        # cams, egomotion for the wrist).
        flow_xy_pixels = vel[nearest_idx]                               # (H, W, 2)
        flow[t, 0] = np.where(in_range, flow_xy_pixels[..., 0], flow[t, 0])
        flow[t, 1] = np.where(in_range, flow_xy_pixels[..., 1], flow[t, 1])
    return flow


def flow_to_rgb(
    flow: np.ndarray,
    max_magnitude: float | None = None,
    value: float = 1.0,
) -> np.ndarray:
    """Visualise a 2-D flow field as an RGB image with hue = direction, saturation = speed.

    Args:
        flow: ``(2, H, W)`` or ``(T, 2, H, W)``.
        max_magnitude: Normalisation constant for saturation. If ``None``, the per-input
            max is used (per-frame for a sequence).
        value: HSV "value" channel — 1.0 for bright on black.

    Returns:
        ``(H, W, 3)`` uint8 if input is 3-D, ``(T, H, W, 3)`` uint8 if input is 4-D.
    """
    if flow.ndim == 3:
        squeeze = True
        flow = flow[None]
    elif flow.ndim == 4:
        squeeze = False
    else:
        raise ValueError(f"flow must be (2, H, W) or (T, 2, H, W), got {flow.shape}")

    T, _, H, W = flow.shape
    fx = flow[:, 0]                                                    # (T, H, W)
    fy = flow[:, 1]
    mag = np.sqrt(fx * fx + fy * fy)                                    # (T, H, W)
    angle = np.arctan2(fy, fx)                                         # in [-pi, pi]
    hue = (angle / (2 * np.pi)) + 0.5                                  # in [0, 1]
    if max_magnitude is None:
        per_frame_max = mag.reshape(T, -1).max(axis=1, keepdims=True)
        per_frame_max = np.maximum(per_frame_max, 1e-6).reshape(T, 1, 1)
        sat = np.clip(mag / per_frame_max, 0.0, 1.0)
    else:
        sat = np.clip(mag / max(max_magnitude, 1e-6), 0.0, 1.0)
    val = np.full_like(hue, value)
    hsv = np.stack([hue, sat, val], axis=-1)                            # (T, H, W, 3)

    # HSV -> RGB (vectorised).
    rgb = _hsv_to_rgb(hsv)
    rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    return rgb[0] if squeeze else rgb


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Vectorised HSV->RGB. ``hsv`` shape ``(..., 3)`` with values in ``[0, 1]``."""
    h = hsv[..., 0] * 6.0
    s = hsv[..., 1]
    v = hsv[..., 2]
    i = np.floor(h).astype(np.int32) % 6
    f = h - np.floor(h)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    r = np.where(i == 0, v, np.where(i == 1, q, np.where(i == 2, p, np.where(i == 3, p, np.where(i == 4, t, v)))))
    g = np.where(i == 0, t, np.where(i == 1, v, np.where(i == 2, v, np.where(i == 3, q, np.where(i == 4, p, p)))))
    b = np.where(i == 0, p, np.where(i == 1, p, np.where(i == 2, t, np.where(i == 3, v, np.where(i == 4, v, q)))))
    return np.stack([r, g, b], axis=-1)
