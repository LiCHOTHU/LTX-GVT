"""Render wrist videos using GT wrist[0] as the scene source (not ext1[0]).

This guarantees the rendered wrist[0] matches GT wrist[0] exactly. For t>0,
we warp wrist[0] through the relative wrist motion (FK + T_cam_to_hand).

The model receives (action, observation=wrist[0]) — so wrist[0] is given.

Math for any t:
    T_wrist[0]_to_base = T_hand[0] @ T_cam_to_hand
    T_wrist[t]_to_base = T_hand[t] @ T_cam_to_hand
    For each wrist[t] pixel:
        back-project through K_wrist, ray in wrist[t] frame
        transform to base frame using T_wrist[t]_to_base
        intersect with plane z=plane_z
        project to wrist[0] frame using inv(T_wrist[0]_to_base)
        sample wrist[0] image at the projected pixel
"""

from __future__ import annotations

import json, os, sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "packages/ltx-action-cond/src")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from ltx_action_cond.kinematics import fk_urdf
from ltx_action_cond.wrist_render import render_wrist_via_plane_homography

WRIST_ROOT = Path("experiments/droid_action_cond/outputs/wrist_demo")
CALIB_DIR = Path("experiments/droid_action_cond/outputs/wrist_calib_vggt")


def label(img, text, font_size=18):
    h = font_size + 10
    bar = Image.new("RGB", (img.shape[1], h), color=(0, 0, 0))
    d = ImageDraw.Draw(bar)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        f = ImageFont.load_default()
    d.text((8, 5), text, fill=(255, 255, 255), font=f)
    return np.concatenate([np.array(bar), img], axis=0)


def derive_T_cam_to_hand(d, calib) -> np.ndarray:
    """Use the v1 calibration with the principled scale heuristic, falling
    back to ext2-anchored variant if a `*_ext2anchor.json` exists for this episode."""
    T_wrist_to_ext = np.array(calib.get("T_wrist_to_ext2", calib.get("T_wrist_to_ext1")))
    use_ext2 = "T_wrist_to_ext2" in calib
    cam2base_anchor = d["cam2base_2" if use_ext2 else "cam2base_1"]
    T_hand_to_base_t0 = fk_urdf(d["cmd_joint_position"][0])["hand"]
    T_hand_to_anchor = np.linalg.inv(cam2base_anchor) @ T_hand_to_base_t0
    true_dist = float(np.linalg.norm(T_hand_to_anchor[:3, 3]))
    vggt_dist = float(np.linalg.norm(T_wrist_to_ext[:3, 3]))
    scale = true_dist / vggt_dist if vggt_dist > 1e-6 else 1.0
    T_w_scaled = T_wrist_to_ext.copy()
    T_w_scaled[:3, 3] *= scale
    T_wrist_to_base_t0 = cam2base_anchor @ T_w_scaled
    return np.linalg.inv(T_hand_to_base_t0) @ T_wrist_to_base_t0


def render_episode(ep_short: str):
    d = np.load(WRIST_ROOT / ep_short / "data.npz")
    # Prefer ext2-anchored calib if present (better for TRI), else default
    cal_path = CALIB_DIR / f"{ep_short}_ext2anchor.json"
    if cal_path.exists():
        calib = json.loads(cal_path.read_text())
        T_cam_to_hand = np.array(calib["T_cam_to_hand"])
        src_tag = "ext2-anchored"
    else:
        calib = json.loads((CALIB_DIR / f"{ep_short}.json").read_text())
        T_cam_to_hand = derive_T_cam_to_hand(d, calib)
        src_tag = "ext1-anchored"

    joints = d["cmd_joint_position"]
    wrist_gt = d["wrist_frames"]
    K_wrist = d["K_wrist"]
    H, W = wrist_gt[0].shape[:2]
    s = 3
    T_frames = joints.shape[0]

    # Pose at t=0
    T_hand_0 = fk_urdf(joints[0])["hand"]
    T_wrist0_to_base = T_hand_0 @ T_cam_to_hand
    wrist0_img = wrist_gt[0]

    frames = []
    for t in range(T_frames):
        T_hand_t = fk_urdf(joints[t])["hand"]
        T_wrist_t_to_base = T_hand_t @ T_cam_to_hand
        # Use the existing plane-homography fn with WRIST[0] as the source view
        # (treating wrist[0] as if it were "ext1[0]" -- same intrinsics K_wrist).
        rgb = render_wrist_via_plane_homography(
            wrist0_img, K_wrist, T_wrist0_to_base,
            T_wrist_t_to_base, K_wrist, (W, H), plane_z=0.0,
        )
        gt_up = np.array(Image.fromarray(wrist_gt[t]).resize((W * s, H * s), Image.LANCZOS))
        rd_up = np.array(Image.fromarray(rgb).resize((W * s, H * s), Image.LANCZOS))
        gt_l = label(gt_up, f"GT wrist  |  {ep_short[:32]}  t={t}")
        rd_l = label(rd_up, f"rendered (anchor=GT wrist[0], {src_tag})  t={t}")
        sep = np.zeros((gt_l.shape[0], 8, 3), dtype=np.uint8)
        frames.append(np.concatenate([gt_l, sep, rd_l], axis=1))

    out_path = CALIB_DIR / f"{ep_short}_wrist0anchor_video.mp4"
    imageio.mimsave(out_path, frames, fps=15, macro_block_size=1)
    print(f"  wrote {out_path.name}  ({len(frames)} frames, {out_path.stat().st_size/1024:.0f} KB)")
    return out_path


def main():
    for ep_dir in sorted(WRIST_ROOT.iterdir()):
        if not (ep_dir / "data.npz").exists(): continue
        print(f"=== {ep_dir.name} ===")
        render_episode(ep_dir.name)


if __name__ == "__main__":
    main()
