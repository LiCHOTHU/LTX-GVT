"""Draw the projected Franka skeleton onto a PIL image.

Stage 1 prior: just dots and lines. Stage 2 (per-link color mask from URDF meshes)
will live next to this in a `mesh_rendering` submodule.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

# Edges between keypoint indices: 0=base, 1..7=joint origins, 8=flange.
SKELETON_EDGES: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
]

# Default colors. (kept module-level so callers can tweak.)
JOINT_COLOR = (255, 80, 80)    # red
EDGE_COLOR = (80, 200, 255)    # cyan
BASE_COLOR = (255, 255, 80)    # yellow
FLANGE_COLOR = (80, 255, 120)  # green


def draw_skeleton(
    img: Image.Image,
    pix: np.ndarray,
    depth: np.ndarray,
    *,
    point_radius: int = 3,
    line_width: int = 2,
) -> None:
    """In-place draw 9 keypoints + 8 edges onto `img`.

    Args:
        img:    target PIL RGB image (modified in place).
        pix:    (9, 2) pixel coords from `project_points`.
        depth:  (9,) z-in-cam from `project_points`; keypoints with depth<=0
                are skipped (behind the camera).
        point_radius / line_width: cosmetic.
    """
    if pix.shape != (9, 2) or depth.shape != (9,):
        raise ValueError(f"expected 9 keypoints, got pix {pix.shape} depth {depth.shape}")

    d = ImageDraw.Draw(img)
    w, h = img.size

    def clip(p: np.ndarray) -> tuple[float, float]:
        # Pillow tolerates out-of-canvas coords but very large values can cause overflow.
        return float(np.clip(p[0], -w, 2 * w)), float(np.clip(p[1], -h, 2 * h))

    for i, j in SKELETON_EDGES:
        if depth[i] <= 0 or depth[j] <= 0:
            continue
        d.line([clip(pix[i]), clip(pix[j])], fill=EDGE_COLOR, width=line_width)

    for i, p in enumerate(pix):
        if depth[i] <= 0:
            continue
        x, y = clip(p)
        color = BASE_COLOR if i == 0 else FLANGE_COLOR if i == 8 else JOINT_COLOR
        d.ellipse([x - point_radius, y - point_radius, x + point_radius, y + point_radius], fill=color)
