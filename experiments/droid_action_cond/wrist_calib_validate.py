"""Validation for VGGT-recovered T_cam_to_hand:

For each episode, render wrist[t=0] using the plane-warp scene from ext1[0]
and the VGGT-derived wrist pose at t=0. Compare visually to the GT wrist[0].

If VGGT got the wrist pose right, the rendered view should overlap the GT
view geometrically -- same scene content in the same image regions.

Per episode, saves outputs/wrist_calib_vggt/<ep>/_validate.png:
  ext1[0]  |  wrist GT[0]  |  wrist rendered[0]  |  side-by-side overlay
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


def render_with_pose(data: dict, T_cam_to_hand: np.ndarray, t: int, point_radius: int = 3) -> np.ndarray:
    """Render wrist[t] from ext1[0] plane-warp using the given T_cam_to_hand."""
    scene = reconstruct_scene_from_plane(data["ext1_frames"][0], data["K1"], data["cam2base_1"], plane_z=0.0)
    q = data["cmd_joint_position"][t]
    T_hand_to_base = fk_urdf(q)["hand"]
    T_wrist_to_base = T_hand_to_base @ T_cam_to_hand
    H, W = data["wrist_frames"][t].shape[:2]
    rgb, _ = render_wrist_scene_splat(scene, T_wrist_to_base, data["K_wrist"], (W, H),
                                       point_radius_px=point_radius)
    return rgb


def main():
    for ep_dir in sorted(WRIST_ROOT.iterdir()):
        if not (ep_dir / "data.npz").exists(): continue
        calib_path = CALIB_DIR / f"{ep_dir.name}.json"
        if not calib_path.exists():
            print(f"  no VGGT calib for {ep_dir.name}, skipping")
            continue
        calib = json.loads(calib_path.read_text())
        T_cam_to_hand = np.array(calib["T_cam_to_hand"])
        d = np.load(ep_dir / "data.npz")
        ext1_t0 = d["ext1_frames"][0]
        gt_t0 = d["wrist_frames"][0]
        rendered_t0 = render_with_pose({k: d[k] for k in d.files}, T_cam_to_hand, t=0, point_radius=3)
        # Overlay GT + rendered (50/50 blend) so they can be eyeballed
        ov = (0.5 * gt_t0.astype(np.float32) + 0.5 * rendered_t0.astype(np.float32)).clip(0,255).astype(np.uint8)
        # 3x upscale + label
        s = 3
        cells = [
            label(np.array(Image.fromarray(ext1_t0).resize((ext1_t0.shape[1]*s, ext1_t0.shape[0]*s), Image.LANCZOS)),
                  f"ext1 t=0"),
            label(np.array(Image.fromarray(gt_t0).resize((gt_t0.shape[1]*s, gt_t0.shape[0]*s), Image.LANCZOS)),
                  f"wrist GT t=0"),
            label(np.array(Image.fromarray(rendered_t0).resize((rendered_t0.shape[1]*s, rendered_t0.shape[0]*s), Image.LANCZOS)),
                  f"wrist rendered t=0 (VGGT pose)"),
            label(np.array(Image.fromarray(ov).resize((ov.shape[1]*s, ov.shape[0]*s), Image.LANCZOS)),
                  f"50/50 overlay GT vs render"),
        ]
        sep = np.zeros((cells[0].shape[0], 6, 3), dtype=np.uint8)
        out = cells[0]
        for c in cells[1:]:
            out = np.concatenate([out, sep, c], axis=1)
        # Title
        title = label(np.zeros((1, out.shape[1], 3), dtype=np.uint8),
                      f"{ep_dir.name}  |  T_cam_to_hand_t = {[round(x,3) for x in calib['translation_cam_to_hand_m']]}",
                      font_size=18)
        out = np.concatenate([title, out], axis=0)
        Image.fromarray(out).save(CALIB_DIR / f"{ep_dir.name}_validate.png")
        print(f"  wrote validation for {ep_dir.name}")


if __name__ == "__main__":
    main()
