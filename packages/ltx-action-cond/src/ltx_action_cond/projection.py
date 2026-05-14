"""Project base-frame 3D points into pixel coordinates of a calibrated camera."""

from __future__ import annotations

import numpy as np


def project_points(points_base: np.ndarray, base2cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project N points expressed in robot-base frame into image pixels.

    Args:
        points_base: (N, 3) base-frame points.
        base2cam:    (4, 4) transform mapping base-frame -> camera-frame. Note that
                     `cam2base_extrinsics.json` gives you `T_base_cam`, so pass
                     `np.linalg.inv(T_base_cam)` here.
        K:           (3, 3) pinhole intrinsics at the TARGET image resolution.

    Returns:
        pix:    (N, 2) pixel coords (may fall outside the image, no clipping).
        depth:  (N,)   z-in-cam (negative or zero means behind the camera).
    """
    if points_base.ndim != 2 or points_base.shape[1] != 3:
        raise ValueError(f"points_base must be (N,3), got {points_base.shape}")
    if base2cam.shape != (4, 4):
        raise ValueError(f"base2cam must be (4,4), got {base2cam.shape}")
    if K.shape != (3, 3):
        raise ValueError(f"K must be (3,3), got {K.shape}")

    n = points_base.shape[0]
    p_h = np.hstack([points_base, np.ones((n, 1), dtype=points_base.dtype)]).T  # (4, N)
    p_cam = (base2cam @ p_h)[:3]  # (3, N)
    z = p_cam[2]
    safe_z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    pix = (K @ p_cam)[:2] / safe_z
    return pix.T, z
