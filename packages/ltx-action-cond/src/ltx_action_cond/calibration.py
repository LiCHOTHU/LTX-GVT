"""Parse the April 2025 DROID calibration JSONs.

Conventions verified against the official `CalibrationExample.ipynb` shipped in
the calibration release:

  * Extrinsic is a 6-vector `[tx, ty, tz, rx, ry, rz]` where (rx, ry, rz) are
    Euler angles in xyz convention. The resulting 4x4 matrix maps CAMERA-frame
    coordinates to BASE-frame coordinates (`T_base_cam`). To project base-frame
    points into the camera, invert it.
  * Intrinsics is `[fx, cx, fy, cy]` (note the order: NOT [fx, fy, cx, cy]).
    The intrinsic resolution is native ZED 2 (1280x720); DROID RLDS frames are
    320x180, so the K matrix must be rescaled before projection.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def extrinsic_6dof_to_matrix(vec6: list[float] | np.ndarray) -> np.ndarray:
    """[tx,ty,tz, rx,ry,rz_euler_xyz] -> 4x4 matrix mapping cam-frame -> base-frame."""
    v = np.asarray(vec6, dtype=np.float64)
    if v.shape != (6,):
        raise ValueError(f"extrinsic 6-DoF must be shape (6,), got {v.shape}")
    m = np.eye(4)
    m[:3, :3] = R.from_euler("xyz", v[3:6]).as_matrix()
    m[:3, 3] = v[:3]
    return m


def intrinsics_to_K(camera_matrix: list[float]) -> np.ndarray:
    """DROID JSON `cameraMatrix` field [fx, cx, fy, cy] -> 3x3 K (native ZED 2 resolution)."""
    if len(camera_matrix) != 4:
        raise ValueError(f"expected [fx, cx, fy, cy], got {camera_matrix!r}")
    fx, cx, fy, cy = camera_matrix
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def rescale_K(K: np.ndarray, src_size: tuple[int, int], dst_size: tuple[int, int]) -> np.ndarray:
    """Rescale a 3x3 K matrix from src image size to dst image size.

    Sizes are (W, H). Used to go from native 1280x720 intrinsics down to the 320x180
    that DROID's RLDS frames are stored at.
    """
    sx = dst_size[0] / src_size[0]
    sy = dst_size[1] / src_size[1]
    out = K.copy().astype(np.float64)
    out[0, 0] *= sx  # fx
    out[0, 2] *= sx  # cx
    out[1, 1] *= sy  # fy
    out[1, 2] *= sy  # cy
    return out
