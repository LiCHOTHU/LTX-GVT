"""Apply scale correction to VGGT-recovered wrist pose for IPRL and re-validate.

Scale factor = true_focal / vggt_predicted_focal at VGGT's internal resolution.
We scale only the TRANSLATION of T_wrist_to_ext1 (rotation is scale-invariant).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "packages/ltx-action-cond/src")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from ltx_action_cond.kinematics import fk_urdf
from ltx_action_cond.wrist_render import reconstruct_scene_from_plane, render_wrist_scene_splat

EP_SHORT = "IPRL+w026bb9b+2023-04-20-23h-40m-21s"
WRIST_ROOT = Path("experiments/droid_action_cond/outputs/wrist_demo")
CALIB_DIR = Path("experiments/droid_action_cond/outputs/wrist_calib_vggt")


def label(img: np.ndarray, text: str, font_size: int = 16) -> np.ndarray:
    h = font_size + 8
    bar = Image.new("RGB", (img.shape[1], h), color=(0, 0, 0))
    d = ImageDraw.Draw(bar)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        f = ImageFont.load_default()
    d.text((6, 4), text, fill=(255, 255, 255), font=f)
    return np.concatenate([np.array(bar), img], axis=0)


def render_with(T_cam_to_hand, d, t=0):
    scene = reconstruct_scene_from_plane(d["ext1_frames"][0], d["K1"], d["cam2base_1"], plane_z=0.0)
    q = d["cmd_joint_position"][t]
    Twb = fk_urdf(q)["hand"] @ T_cam_to_hand
    H, W = d["wrist_frames"][t].shape[:2]
    rgb, _ = render_wrist_scene_splat(scene, Twb, d["K_wrist"], (W, H), point_radius_px=3)
    return rgb


def main():
    ep_dir = WRIST_ROOT / EP_SHORT
    d = np.load(ep_dir / "data.npz")
    calib = json.loads((CALIB_DIR / f"{EP_SHORT}.json").read_text())

    # Original VGGT-derived T_cam_to_hand
    T_cam_to_hand_orig = np.array(calib["T_cam_to_hand"])
    print(f"Original T_cam_to_hand translation: {T_cam_to_hand_orig[:3, 3]}")

    # Apply scale to T_wrist_to_ext1 translation, recompose
    T_wrist_to_ext1 = np.array(calib["T_wrist_to_ext1"]).copy()
    cam2base_1 = d["cam2base_1"]
    T_hand_to_base = fk_urdf(d["cmd_joint_position"][0])["hand"]
    # Principled scale: set |t_wrist_to_ext1| equal to the known FK-derived
    # |t_hand_to_ext1| (wrist is mounted close to hand, so distances ≈ equal).
    T_hand_to_ext1 = np.linalg.inv(cam2base_1) @ T_hand_to_base
    true_dist = float(np.linalg.norm(T_hand_to_ext1[:3, 3]))
    vggt_dist = float(np.linalg.norm(T_wrist_to_ext1[:3, 3]))
    principled_scale = true_dist / vggt_dist
    print(f"Principled scale (|hand_to_ext1| / |wrist_to_ext1_vggt|): {principled_scale:.3f}")
    SCALES = [principled_scale, 1.5, 2.0]

    rendered_variants = []
    rendered_variants.append(("VGGT raw (scale=1.0)", render_with(T_cam_to_hand_orig, d, t=0), T_cam_to_hand_orig[:3, 3]))
    for s in SCALES:
        T_wrist_to_ext1_scaled = T_wrist_to_ext1.copy()
        T_wrist_to_ext1_scaled[:3, 3] *= s
        T_wrist_to_base_scaled = cam2base_1 @ T_wrist_to_ext1_scaled
        T_cam_to_hand_scaled = np.linalg.inv(T_hand_to_base) @ T_wrist_to_base_scaled
        rgb = render_with(T_cam_to_hand_scaled, d, t=0)
        rendered_variants.append((f"scale={s}", rgb, T_cam_to_hand_scaled[:3, 3]))

    # Layout: top row = ext1[0], wrist GT[0]; bottom rows = scale variants
    gt = d["wrist_frames"][0]
    ext1 = d["ext1_frames"][0]
    H, W = gt.shape[:2]; s = 3
    top_cells = [
        label(np.array(Image.fromarray(ext1).resize((W*s, H*s), Image.LANCZOS)), "ext1 @ t=0"),
        label(np.array(Image.fromarray(gt).resize((W*s, H*s), Image.LANCZOS)), "wrist GT @ t=0"),
    ]
    sep = np.zeros((top_cells[0].shape[0], 6, 3), dtype=np.uint8)
    top = np.concatenate([top_cells[0], sep, top_cells[1]], axis=1)

    var_cells = []
    for name, rgb, t_cam in rendered_variants:
        cell = label(
            np.array(Image.fromarray(rgb).resize((W*s, H*s), Image.LANCZOS)),
            f"{name}  T_cam_to_hand_t=[{t_cam[0]:.2f}, {t_cam[1]:.2f}, {t_cam[2]:.2f}]",
        )
        var_cells.append(cell)
    # All variants in one row
    row = var_cells[0]
    for c in var_cells[1:]:
        row = np.concatenate([row, sep, c], axis=1)
    # pad top OR row to same width
    if row.shape[1] > top.shape[1]:
        top = np.concatenate([top, np.zeros((top.shape[0], row.shape[1] - top.shape[1], 3), dtype=np.uint8)], axis=1)
    elif row.shape[1] < top.shape[1]:
        row = np.concatenate([row, np.zeros((row.shape[0], top.shape[1] - row.shape[1], 3), dtype=np.uint8)], axis=1)
    gap = np.full((10, top.shape[1], 3), 80, dtype=np.uint8)
    out = np.concatenate([top, gap, row], axis=0)
    out_path = CALIB_DIR / f"{EP_SHORT}_scale_sweep.png"
    Image.fromarray(out).save(out_path)
    print(f"wrote {out_path}  shape={out.shape}")


if __name__ == "__main__":
    main()
