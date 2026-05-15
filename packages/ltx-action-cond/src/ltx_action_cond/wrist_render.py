"""Wrist-camera synthetic view generation.

Pipeline:

  1. **Hand-eye mount** -- a fixed `T_cam_to_hand` constant relating the wrist
     camera to the Franka hand frame. DROID does NOT ship this; we either
     hand-tune a nominal (`NOMINAL_T_CAM_TO_HAND` below) or solve hand-eye
     calibration from data (see `solve_hand_eye_at_frame`). Given the mount,
     wrist pose per frame is:
         T_wrist_to_base(t) = T_hand_to_base(q(t)) @ T_cam_to_hand

  2. **Scene reconstruction at t=0** -- two strategies:
     - `reconstruct_scene_from_stereo`: SIFT match ext1 <-> ext2, triangulate
       to a sparse 3D point cloud. Robust to non-planar scenes but feature
       matching can be sparse at the 320x180 RLDS resolution.
     - `reconstruct_scene_from_plane`: assume the workspace is a horizontal
       plane at z=z_table in base; back-project every ext1 pixel onto that
       plane. Dense, fast, but objects above/below the plane are placed at
       the wrong height.

  3. **Wrist render** -- for each frame t:
       project the t=0 scene cloud into the wrist camera via
       `T_base_to_wrist(t)` and `K_wrist`, splat colours with z-buffer,
       then composite the Franka mesh (rendered at the same wrist pose) on top.

Output: a synthetic wrist view per frame, action-consistent. Has holes
(unobserved regions at t=0) and is stale for dynamic content.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# ---------- nominal wrist mount ----------

def _nominal_t_cam_to_hand() -> np.ndarray:
    """A reasonable nominal `T_cam_to_hand` based on the DROID assembly description.

    The ZED Mini is bolted to the side of the Franka hand, optical axis
    roughly parallel to the gripper-pointing direction (+Z of the panda_hand
    frame, toward the fingers and workspace).

    Conventions (panda_hand frame):
      +Z = toward fingertips (forward in gripping direction)
      +X = perpendicular to gripper jaws (in the open/close plane)
      +Y = along the gripper jaw motion
    Conventions (OpenCV camera frame):
      +Z = optical axis (forward)
      +X = image right
      +Y = image down

    Mount: camera origin is offset ~5 cm forward (+Z hand) and ~3 cm to one
    side (+Y hand). Optical axis aligned with hand +Z. Image-right (cam +X)
    aligned with hand +Y. Image-down (cam +Y) aligned with hand -X.

    This is a starting point; refine with hand-eye calibration if needed.
    """
    R_cam_to_hand = np.array([
        [0.0,  0.0, 1.0],   # cam X (image right)  = hand +Y? No: this places hand+Y where cam+X is
        [0.0, -1.0, 0.0],   # cam Y (image down)   = -hand Y? Actually we map column-wise.
        [1.0,  0.0, 0.0],   # cam Z (optical axis) = hand +X
    ])
    # Re-derive cleanly: R has columns = camera axes in hand frame
    # We want:
    #   cam Z-axis -> hand +Z direction  => col 2 = (0, 0, 1)
    #   cam X-axis -> hand +Y direction  => col 0 = (0, 1, 0)
    #   cam Y-axis -> hand -X direction  => col 1 = (-1, 0, 0)
    R = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    # Camera placed UP the arm from hand origin (-Z in hand = toward the wrist
    # body, away from the fingertips) and offset to the side. From here the
    # camera looks "through" the gripper toward the workspace.
    T[:3, 3] = np.array([0.0, -0.06, -0.08])
    return T


NOMINAL_T_CAM_TO_HAND = _nominal_t_cam_to_hand()


# ---------- stereo scene reconstruction ----------

@dataclass
class SceneCloud:
    points_base: np.ndarray  # (N, 3) base-frame 3D points
    colors: np.ndarray       # (N, 3) uint8 RGB
    ext1_pixels: np.ndarray  # (N, 2) the pixel in ext1 that contributed each point
    ext2_pixels: np.ndarray  # (N, 2) the pixel in ext2
    inlier_match_count: int


def _sift_match(img1: np.ndarray, img2: np.ndarray,
                ratio: float = 0.75,
                nfeatures: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """SIFT detect + ratio test + Lowe filter. Returns (pts1, pts2) as float32 (M,2)."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.02)
    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(des1, des2, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2


def reconstruct_scene_from_plane(
    ext1_img: np.ndarray,
    K1: np.ndarray,
    cam2base_1: np.ndarray,
    *,
    plane_z: float = 0.0,
    sample_stride: int = 1,
    max_distance_from_base: float = 1.5,
    max_distance_from_cam: float = 3.0,
) -> "SceneCloud":
    """Approximate scene as a horizontal plane at z=plane_z in base frame.

    For each ext1 pixel (sub-sampled by `sample_stride`), back-project the ray
    into base coords and intersect with the plane. Returns the (3D base point,
    color) per pixel that hits the plane in front of the camera within
    reasonable distance.

    This is dense and fast (no feature matching) but misplaces objects above
    or below the plane. Good enough for textured-tabletop scenes; the wrist
    camera is usually pointing roughly down, so plane-warped texture is a
    decent first approximation of what it should see.
    """
    H, W = ext1_img.shape[:2]
    # Sample pixel grid
    us, vs = np.meshgrid(np.arange(0, W, sample_stride), np.arange(0, H, sample_stride))
    us = us.ravel().astype(np.float64)
    vs = vs.ravel().astype(np.float64)

    # Back-project to camera-frame rays
    Kinv = np.linalg.inv(K1)
    pts_h = np.stack([us, vs, np.ones_like(us)], axis=1)  # (M, 3)
    rays_cam = pts_h @ Kinv.T  # (M, 3), z=1
    # Rotate rays into base frame
    R = cam2base_1[:3, :3]
    t = cam2base_1[:3, 3]
    rays_base = rays_cam @ R.T  # (M, 3) -- direction vectors

    # Intersect each ray with plane z = plane_z
    # Camera origin in base = t. Point on ray: t + s * rays_base. Solve t.z + s * d.z = plane_z
    dz = rays_base[:, 2]
    # Avoid div-by-zero
    valid = np.abs(dz) > 1e-6
    s = np.where(valid, (plane_z - t[2]) / np.where(valid, dz, 1.0), -1.0)
    P_base = t[None, :] + s[:, None] * rays_base  # (M, 3)

    # Filter: in front of camera (s > 0) and reasonable distance
    cam_dist = np.linalg.norm(P_base - t[None, :], axis=1)
    base_dist = np.linalg.norm(P_base[:, :2], axis=1)
    keep = valid & (s > 0.01) & (cam_dist < max_distance_from_cam) & (base_dist < max_distance_from_base)
    if not keep.any():
        return SceneCloud(
            points_base=np.zeros((0, 3), np.float64),
            colors=np.zeros((0, 3), np.uint8),
            ext1_pixels=np.zeros((0, 2), np.float32),
            ext2_pixels=np.zeros((0, 2), np.float32),
            inlier_match_count=0,
        )
    P_base = P_base[keep]
    us = us[keep]; vs = vs[keep]
    # Sample colors directly (us, vs are integer aligned since stride is int)
    ius = us.astype(int); ivs = vs.astype(int)
    colors = ext1_img[ivs, ius]

    return SceneCloud(
        points_base=P_base,
        colors=colors,
        ext1_pixels=np.stack([us, vs], axis=1).astype(np.float32),
        ext2_pixels=np.zeros((P_base.shape[0], 2), np.float32),
        inlier_match_count=int(P_base.shape[0]),
    )


def reconstruct_scene(
    ext1_img: np.ndarray,
    K1: np.ndarray,
    cam2base_1: np.ndarray,
    ext2_img: np.ndarray,
    K2: np.ndarray,
    cam2base_2: np.ndarray,
    *,
    max_reproj_err_px: float = 2.0,
    z_clip_min: float = 0.05,
    z_clip_max: float = 3.0,
) -> SceneCloud:
    """Triangulate a sparse 3D point cloud in base coords from two calibrated views.

    Filters using:
      - RANSAC fundamental-matrix consistency
      - reprojection error <= `max_reproj_err_px` in both views
      - 3D point depth in [z_clip_min, z_clip_max] when expressed in each cam frame
    """
    pts1, pts2 = _sift_match(ext1_img, ext2_img)
    if pts1.shape[0] < 16:
        return SceneCloud(
            points_base=np.zeros((0, 3), np.float64),
            colors=np.zeros((0, 3), np.uint8),
            ext1_pixels=np.zeros((0, 2), np.float32),
            ext2_pixels=np.zeros((0, 2), np.float32),
            inlier_match_count=0,
        )
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.5, 0.999)
    if F is None or mask is None:
        return SceneCloud(
            points_base=np.zeros((0, 3), np.float64),
            colors=np.zeros((0, 3), np.uint8),
            ext1_pixels=np.zeros((0, 2), np.float32),
            ext2_pixels=np.zeros((0, 2), np.float32),
            inlier_match_count=0,
        )
    mask = mask.ravel().astype(bool)
    pts1 = pts1[mask]
    pts2 = pts2[mask]

    # Projection matrices in base coordinates: P_i = K_i @ [R_i | t_i] = K_i @ inv(cam2base_i)
    base2cam_1 = np.linalg.inv(cam2base_1)
    base2cam_2 = np.linalg.inv(cam2base_2)
    P1 = K1 @ base2cam_1[:3, :]
    P2 = K2 @ base2cam_2[:3, :]
    pts1_h = pts1.T  # (2, M)
    pts2_h = pts2.T
    X4 = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)  # (4, M)
    X = (X4[:3] / X4[3]).T  # (M, 3) in base coords

    # Reprojection error check
    def reproj_err(P, X3, gt):
        Xh = np.concatenate([X3, np.ones((X3.shape[0], 1))], axis=1)  # (M, 4)
        uvw = Xh @ P.T  # (M, 3)
        uv = uvw[:, :2] / uvw[:, 2:3]
        return np.linalg.norm(uv - gt, axis=1)
    e1 = reproj_err(P1, X, pts1)
    e2 = reproj_err(P2, X, pts2)

    # Depth check in each camera
    X_cam1 = (base2cam_1 @ np.concatenate([X, np.ones((X.shape[0], 1))], axis=1).T).T[:, :3]
    X_cam2 = (base2cam_2 @ np.concatenate([X, np.ones((X.shape[0], 1))], axis=1).T).T[:, :3]
    z1 = X_cam1[:, 2]
    z2 = X_cam2[:, 2]
    ok = (e1 < max_reproj_err_px) & (e2 < max_reproj_err_px)
    ok &= (z1 > z_clip_min) & (z1 < z_clip_max)
    ok &= (z2 > z_clip_min) & (z2 < z_clip_max)
    X = X[ok]
    pts1 = pts1[ok]
    pts2 = pts2[ok]

    # Color: bilinear sample from ext1 at the keypoint location
    h, w = ext1_img.shape[:2]
    u = np.clip(pts1[:, 0], 0, w - 1)
    v = np.clip(pts1[:, 1], 0, h - 1)
    u0 = np.floor(u).astype(int); v0 = np.floor(v).astype(int)
    u1_ = np.clip(u0 + 1, 0, w - 1); v1_ = np.clip(v0 + 1, 0, h - 1)
    du = (u - u0)[:, None]; dv = (v - v0)[:, None]
    c00 = ext1_img[v0, u0].astype(np.float32)
    c01 = ext1_img[v0, u1_].astype(np.float32)
    c10 = ext1_img[v1_, u0].astype(np.float32)
    c11 = ext1_img[v1_, u1_].astype(np.float32)
    colors = (c00 * (1 - du) * (1 - dv) + c01 * du * (1 - dv)
              + c10 * (1 - du) * dv + c11 * du * dv).clip(0, 255).astype(np.uint8)

    return SceneCloud(
        points_base=X.astype(np.float64),
        colors=colors,
        ext1_pixels=pts1.astype(np.float32),
        ext2_pixels=pts2.astype(np.float32),
        inlier_match_count=int(ok.sum()),
    )


# ---------- hand-eye calibration ----------

@dataclass
class HandEyeResult:
    T_cam_to_hand: np.ndarray  # 4x4
    n_inliers: int
    mean_reproj_err_px: float
    used_frame_idx: int


def solve_hand_eye_at_frame(
    wrist_img: np.ndarray,
    ext1_img: np.ndarray,
    K1: np.ndarray,
    cam2base_1: np.ndarray,
    scene: SceneCloud,
    K_wrist: np.ndarray,
    T_hand_to_base: np.ndarray,
    *,
    used_frame_idx: int = 0,
    ratio: float = 0.78,
    pnp_err_px: float = 4.0,
) -> HandEyeResult | None:
    """Match wrist features to ext1 features whose 3D positions we already know
    from `scene`, then solve PnP for T_base_to_wrist, then recover T_cam_to_hand.

    Requires the scene to have been triangulated from (ext1, ext2). We use ext1
    here as the matching reference since `scene.ext1_pixels` ties scene points
    to ext1 pixel locations.
    """
    if scene.points_base.shape[0] < 16:
        return None

    # SIFT match wrist <-> ext1
    pts_w, pts_e = _sift_match(wrist_img, ext1_img, ratio=ratio)
    if pts_w.shape[0] < 8:
        return None

    # For each (pts_w, pts_e), find the nearest scene.ext1_pixels point.
    # Match only if within a tight radius (pixel-accurate). KD-tree-ish search.
    e_scene = scene.ext1_pixels  # (N, 2)
    if len(e_scene) == 0:
        return None
    # Use scipy/np broadcasted distance for simplicity (N typically <2000)
    d2 = ((pts_e[:, None, :] - e_scene[None, :, :]) ** 2).sum(-1)  # (M, N)
    nn_idx = d2.argmin(axis=1)
    nn_dist = np.sqrt(d2[np.arange(len(pts_e)), nn_idx])
    keep = nn_dist < 2.0  # px tolerance for matching wrist's-ext1-feature to scene's-ext1-feature
    if keep.sum() < 8:
        return None

    pts_w = pts_w[keep]
    pts_e = pts_e[keep]
    nn_idx = nn_idx[keep]
    obj_pts = scene.points_base[nn_idx].astype(np.float64)  # (M, 3)
    img_pts = pts_w.astype(np.float64).reshape(-1, 1, 2)
    K = K_wrist.astype(np.float64)

    # solvePnP with RANSAC for outlier rejection
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts.reshape(-1, 1, 3), img_pts, K, distCoeffs=None,
        reprojectionError=pnp_err_px, iterationsCount=200,
        confidence=0.999, flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < 6:
        return None
    inliers = inliers.ravel()

    R, _ = cv2.Rodrigues(rvec)
    T_base_to_wrist = np.eye(4)
    T_base_to_wrist[:3, :3] = R
    T_base_to_wrist[:3, 3] = tvec.ravel()
    T_wrist_to_base = np.linalg.inv(T_base_to_wrist)
    T_cam_to_hand = np.linalg.inv(T_hand_to_base) @ T_wrist_to_base

    # Compute final reprojection error on inliers
    Xh = np.concatenate([obj_pts[inliers], np.ones((len(inliers), 1))], axis=1)
    uvw = (K @ (T_base_to_wrist @ Xh.T)[:3]).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    err = float(np.mean(np.linalg.norm(uv - pts_w[inliers], axis=1)))

    return HandEyeResult(
        T_cam_to_hand=T_cam_to_hand,
        n_inliers=int(len(inliers)),
        mean_reproj_err_px=err,
        used_frame_idx=used_frame_idx,
    )


# ---------- wrist view splat renderer ----------

def render_wrist_scene_splat(
    scene: SceneCloud,
    T_wrist_to_base: np.ndarray,
    K_wrist: np.ndarray,
    out_size: tuple[int, int],
    *,
    point_radius_px: int = 2,
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> tuple[np.ndarray, np.ndarray]:
    """Project a coloured point cloud through the wrist camera. Returns (rgb HxWx3, depth HxW).

    Points behind the camera (z<=0 in camera frame) or outside the image are
    skipped. Each surviving point is drawn as a small filled disk; z-buffer
    breaks ties so closer points win.
    """
    W, H = out_size
    base_to_wrist = np.linalg.inv(T_wrist_to_base)
    pts = scene.points_base
    if pts.shape[0] == 0:
        return (np.full((H, W, 3), bg_color, np.uint8),
                np.zeros((H, W), np.float32))
    Ph = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)  # (N, 4)
    Pc = (base_to_wrist @ Ph.T).T[:, :3]  # (N, 3) in wrist cam frame
    z = Pc[:, 2]
    front = z > 1e-3
    if not front.any():
        return (np.full((H, W, 3), bg_color, np.uint8),
                np.zeros((H, W), np.float32))
    Pc = Pc[front]; cols = scene.colors[front]
    uvw = (K_wrist @ Pc.T).T
    u = uvw[:, 0] / uvw[:, 2]
    v = uvw[:, 1] / uvw[:, 2]
    z = Pc[:, 2]
    iu = np.round(u).astype(int)
    iv = np.round(v).astype(int)
    ok = (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)
    iu = iu[ok]; iv = iv[ok]; z = z[ok]; cols = cols[ok]
    # Order by depth (far first so close overwrites)
    order = np.argsort(-z)
    iu, iv, z, cols = iu[order], iv[order], z[order], cols[order]

    rgb = np.full((H, W, 3), bg_color, dtype=np.uint8)
    depth = np.zeros((H, W), dtype=np.float32)
    r = point_radius_px
    for du in range(-r, r + 1):
        for dv in range(-r, r + 1):
            if du * du + dv * dv > r * r:
                continue
            nu = iu + du; nv = iv + dv
            m = (nu >= 0) & (nu < W) & (nv >= 0) & (nv < H)
            rgb[nv[m], nu[m]] = cols[m]
            depth[nv[m], nu[m]] = z[m]
    return rgb, depth


def render_wrist_scene_splat_vec(
    scene: SceneCloud,
    T_wrist_to_base: np.ndarray,
    K_wrist: np.ndarray,
    out_size: tuple[int, int],
    *,
    point_radius_px: int = 2,
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised-disk variant of `render_wrist_scene_splat`.

    Same projection, same z-buffer semantics. The 29-iteration (du, dv) disk
    loop is fused into one broadcast over a precomputed (D, 2) offset table,
    so all D disk pixels per point are written by a single `np.add.at`-style
    scatter. Pixel output is identical to the loop variant up to back-to-front
    paint order.
    """
    W, H = out_size
    base_to_wrist = np.linalg.inv(T_wrist_to_base)
    pts = scene.points_base
    if pts.shape[0] == 0:
        return (np.full((H, W, 3), bg_color, np.uint8),
                np.zeros((H, W), np.float32))

    Ph = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    Pc = (base_to_wrist @ Ph.T).T[:, :3]
    front = Pc[:, 2] > 1e-3
    if not front.any():
        return (np.full((H, W, 3), bg_color, np.uint8),
                np.zeros((H, W), np.float32))
    Pc = Pc[front]; cols = scene.colors[front]
    uvw = (K_wrist @ Pc.T).T
    u = uvw[:, 0] / uvw[:, 2]
    v = uvw[:, 1] / uvw[:, 2]
    z = Pc[:, 2]
    iu = np.round(u).astype(np.int32)
    iv = np.round(v).astype(np.int32)
    in_bounds = (iu >= -point_radius_px) & (iu < W + point_radius_px) \
                & (iv >= -point_radius_px) & (iv < H + point_radius_px)
    iu = iu[in_bounds]; iv = iv[in_bounds]
    cols = cols[in_bounds]; z = z[in_bounds]

    # Sort N points back-to-front FIRST (cheap: 57K elements, not 1.7M)
    # so that the (N, D) explosion preserves back-to-front row order and
    # `arr[flat] = vals` does last-write-wins on the closer points.
    order = np.argsort(-z)
    iu = iu[order]; iv = iv[order]
    cols = cols[order]; z = z[order]

    # Precompute disk offsets (D, 2) -- D=29 for r=3
    r = point_radius_px
    offs = np.array(
        [(du, dv) for du in range(-r, r + 1) for dv in range(-r, r + 1)
         if du * du + dv * dv <= r * r],
        dtype=np.int32,
    )
    D = offs.shape[0]

    # Broadcast points x offsets -> (N, D)
    NU = iu[:, None] + offs[None, :, 0]
    NV = iv[:, None] + offs[None, :, 1]
    m = (NU >= 0) & (NU < W) & (NV >= 0) & (NV < H)
    flat = (NV * W + NU)[m]  # row-major: all 29 disk px of pt 0, then pt 1, etc.
    cols_b = np.broadcast_to(cols[:, None, :], (cols.shape[0], D, 3))[m]
    z_b = np.broadcast_to(z[:, None], (z.shape[0], D))[m]

    rgb = np.full((H * W, 3), bg_color, dtype=np.uint8)
    depth = np.zeros((H * W,), dtype=np.float32)
    rgb[flat] = cols_b      # NumPy 1D last-write-wins: closer pts overwrite
    depth[flat] = z_b
    return rgb.reshape(H, W, 3), depth.reshape(H, W)


_GPU_OFFSET_CACHE: dict = {}


def render_wrist_scene_splat_gpu(
    scene: SceneCloud,
    T_wrist_to_base: np.ndarray,
    K_wrist: np.ndarray,
    out_size: tuple[int, int],
    *,
    point_radius_px: int = 2,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    device: str = "cuda",
    _scene_cache: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """torch/CUDA splat. Vectorised disk offsets, scene resident on GPU.

    For best throughput across many frames, pass a `_scene_cache` dict (it is
    populated on first call and reused; only depends on `scene` identity).
    Otherwise the scene tensors are re-uploaded each call.
    """
    import torch

    W, H = out_size
    pts_n = scene.points_base.shape[0]
    if pts_n == 0:
        return (np.full((H, W, 3), bg_color, np.uint8),
                np.zeros((H, W), np.float32))

    # Cache scene tensors on GPU (re-upload only if scene identity changes).
    # Points held in float64 to match the baseline projection rounding.
    if _scene_cache is None or _scene_cache.get("id") != id(scene):
        pts_t = torch.from_numpy(scene.points_base).to(device=device, dtype=torch.float64)
        cols_t = torch.from_numpy(scene.colors).to(device=device, dtype=torch.uint8)
        Ph_t = torch.cat([pts_t, torch.ones(pts_n, 1, device=device, dtype=torch.float64)], dim=1)
        if _scene_cache is not None:
            _scene_cache.clear()
            _scene_cache.update({"id": id(scene), "Ph": Ph_t, "cols": cols_t})
    else:
        Ph_t = _scene_cache["Ph"]
        cols_t = _scene_cache["cols"]

    key = (point_radius_px, device)
    if key not in _GPU_OFFSET_CACHE:
        r = point_radius_px
        offs_np = np.array(
            [(du, dv) for du in range(-r, r + 1) for dv in range(-r, r + 1)
             if du * du + dv * dv <= r * r],
            dtype=np.int64,
        )
        _GPU_OFFSET_CACHE[key] = torch.from_numpy(offs_np).to(device)
    offs_t = _GPU_OFFSET_CACHE[key]
    D = offs_t.shape[0]

    T_b2w = torch.from_numpy(np.linalg.inv(T_wrist_to_base)).to(
        device=device, dtype=torch.float64,
    )
    K_t = torch.from_numpy(K_wrist).to(device=device, dtype=torch.float64)

    Pc = (T_b2w @ Ph_t.T).T[:, :3]
    z = Pc[:, 2]
    front = z > 1e-3
    Pc = Pc[front]
    cols_f = cols_t[front]
    z = z[front]
    if Pc.shape[0] == 0:
        return (np.full((H, W, 3), bg_color, np.uint8),
                np.zeros((H, W), np.float32))

    uvw = (K_t @ Pc.T).T
    u = uvw[:, 0] / uvw[:, 2]
    v = uvw[:, 1] / uvw[:, 2]
    iu = torch.round(u).to(torch.int64)
    iv = torch.round(v).to(torch.int64)

    # Sort N points back-to-front FIRST (so the post-explosion last-write-wins
    # at duplicate flat indices resolves to the closer point).
    order = torch.argsort(-z, stable=True)
    iu = iu[order]; iv = iv[order]; cols_f = cols_f[order]; z = z[order]

    NU = iu[:, None] + offs_t[None, :, 0]
    NV = iv[:, None] + offs_t[None, :, 1]
    mask = (NU >= 0) & (NU < W) & (NV >= 0) & (NV < H)
    flat = (NV * W + NU)[mask]
    cols_b = cols_f[:, None, :].expand(-1, D, -1)[mask]
    z_b = z[:, None].expand(-1, D)[mask]

    bg = torch.tensor(bg_color, device=device, dtype=torch.uint8)
    rgb = bg.expand(H * W, 3).clone()
    depth = torch.zeros(H * W, device=device, dtype=torch.float32)
    rgb[flat] = cols_b           # back-to-front order via row-major flatten
    depth[flat] = z_b.to(torch.float32)
    return (rgb.reshape(H, W, 3).cpu().numpy(),
            depth.reshape(H, W).cpu().numpy())


def render_wrist_via_plane_homography(
    ext1_img: np.ndarray,
    K_ext1: np.ndarray,
    cam2base_ext1: np.ndarray,
    T_wrist_to_base: np.ndarray,
    K_wrist: np.ndarray,
    out_size: tuple[int, int],
    *,
    plane_z: float = 0.0,
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Dense backwards-warp from ext1 to wrist via the table plane.

    For each wrist pixel, ray-cast through the plane z=plane_z in base coords,
    project back into ext1, sample its color bilinearly. Every wrist pixel
    gets a color (no splat gaps even when the wrist is close to the table) and
    the cost is the same as forward splatting -- one matrix multiply per pixel.

    Returns a dense HxWx3 RGB image (uint8). Pixels whose rays don't hit the
    plane in front of either camera get `bg_color`.
    """
    W, H = out_size
    He, We = ext1_img.shape[:2]

    # Build the grid of wrist pixels
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    us = us.ravel().astype(np.float64)
    vs = vs.ravel().astype(np.float64)

    # Back-project each wrist pixel to a ray in the wrist camera frame
    K_wrist_inv = np.linalg.inv(K_wrist)
    pts_h = np.stack([us, vs, np.ones_like(us)], axis=1)  # (N, 3)
    rays_cam = pts_h @ K_wrist_inv.T  # (N, 3) -- direction in wrist cam frame

    # Rotate to base frame and translate origin
    R_w2b = T_wrist_to_base[:3, :3]
    t_w2b = T_wrist_to_base[:3, 3]
    rays_base = rays_cam @ R_w2b.T  # (N, 3)

    # Intersect each ray with the plane z = plane_z
    dz = rays_base[:, 2]
    eps = 1e-6
    safe_dz = np.where(np.abs(dz) > eps, dz, eps)
    s_param = (plane_z - t_w2b[2]) / safe_dz
    in_front_wrist = (np.abs(dz) > eps) & (s_param > 0)
    P_base = t_w2b[None] + s_param[:, None] * rays_base  # (N, 3)

    # Project P_base into ext1
    base2cam_ext1 = np.linalg.inv(cam2base_ext1)
    P_ext1 = P_base @ base2cam_ext1[:3, :3].T + base2cam_ext1[:3, 3]  # (N, 3)
    in_front_ext1 = P_ext1[:, 2] > 1e-3
    px = P_ext1 @ K_ext1.T  # (N, 3)
    safe_z = np.where(in_front_ext1, px[:, 2], 1.0)
    u_e = px[:, 0] / safe_z
    v_e = px[:, 1] / safe_z

    in_bounds = (u_e >= 0) & (u_e < We - 1) & (v_e >= 0) & (v_e < He - 1)
    mask = in_front_wrist & in_front_ext1 & in_bounds

    # Bilinear sample ext1 at (u_e, v_e)
    u_c = np.clip(u_e, 0, We - 1)
    v_c = np.clip(v_e, 0, He - 1)
    u0 = np.floor(u_c).astype(int); v0 = np.floor(v_c).astype(int)
    u1 = np.clip(u0 + 1, 0, We - 1); v1 = np.clip(v0 + 1, 0, He - 1)
    du = (u_c - u0)[:, None]; dv = (v_c - v0)[:, None]
    c00 = ext1_img[v0, u0].astype(np.float32)
    c01 = ext1_img[v0, u1].astype(np.float32)
    c10 = ext1_img[v1, u0].astype(np.float32)
    c11 = ext1_img[v1, u1].astype(np.float32)
    colors = (c00 * (1 - du) * (1 - dv) + c01 * du * (1 - dv)
              + c10 * (1 - du) * dv + c11 * du * dv).clip(0, 255).astype(np.uint8)
    # Apply mask
    bg = np.array(bg_color, dtype=np.uint8)
    out = np.where(mask[:, None], colors, bg[None, :])
    return out.reshape(H, W, 3)


def composite_wrist_with_mesh(
    scene_rgb: np.ndarray,
    mesh_rgb: np.ndarray,
) -> np.ndarray:
    """Place the rendered Franka mesh on top of the scene splat wherever the mesh covers."""
    cov = (mesh_rgb.sum(-1) > 0)
    out = scene_rgb.copy()
    out[cov] = mesh_rgb[cov]
    return out
