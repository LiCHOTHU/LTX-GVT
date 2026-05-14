"""Render full wrist videos for all 5 episodes using VGGT pose + principled scale.

For each episode:
  - Apply scale to VGGT-recovered T_wrist_to_ext1 so |t_wrist_to_ext1| =
    |t_hand_to_ext1| at t=0 (from FK).
  - Recompose T_cam_to_hand.
  - Render every frame's wrist view from the t=0 ext1 plane-warp scene.
  - Save as a side-by-side GT-vs-rendered MP4.

Outputs:
  outputs/wrist_calib_vggt/<ep>_wrist_video.mp4
  outputs/wrist_calib_vggt/_all_wrist_videos_grid.mp4   (5 episodes stacked, single video)
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


def render_episode(ep_short: str) -> tuple[Path, list[np.ndarray]]:
    """Returns (output mp4 path, per-frame stacked images)."""
    d = np.load(WRIST_ROOT / ep_short / "data.npz")
    calib = json.loads((CALIB_DIR / f"{ep_short}.json").read_text())
    T_wrist_to_ext1 = np.array(calib["T_wrist_to_ext1"])
    cam2base_1 = d["cam2base_1"]

    T_hand_to_base_t0 = fk_urdf(d["cmd_joint_position"][0])["hand"]
    T_hand_to_ext1_t0 = np.linalg.inv(cam2base_1) @ T_hand_to_base_t0
    true_dist = float(np.linalg.norm(T_hand_to_ext1_t0[:3, 3]))
    vggt_dist = float(np.linalg.norm(T_wrist_to_ext1[:3, 3]))
    scale = true_dist / vggt_dist if vggt_dist > 1e-6 else 1.0

    T_wrist_to_ext1_scaled = T_wrist_to_ext1.copy()
    T_wrist_to_ext1_scaled[:3, 3] *= scale
    T_wrist_to_base_t0 = cam2base_1 @ T_wrist_to_ext1_scaled
    T_cam_to_hand = np.linalg.inv(T_hand_to_base_t0) @ T_wrist_to_base_t0
    print(f"  scale={scale:.3f}  T_cam_to_hand_t={T_cam_to_hand[:3, 3]}")

    ext1_t0 = d["ext1_frames"][0]
    K_ext1 = d["K1"]
    joints = d["cmd_joint_position"]
    wrist_gt = d["wrist_frames"]
    K_wrist = d["K_wrist"]
    T_frames = joints.shape[0]
    H, W = wrist_gt[0].shape[:2]
    s = 3
    frames_out = []
    for t in range(T_frames):
        T_hand_to_base = fk_urdf(joints[t])["hand"]
        T_wrist_to_base = T_hand_to_base @ T_cam_to_hand
        rgb = render_wrist_via_plane_homography(
            ext1_t0, K_ext1, cam2base_1, T_wrist_to_base, K_wrist, (W, H), plane_z=0.0,
        )
        gt_up = np.array(Image.fromarray(wrist_gt[t]).resize((W * s, H * s), Image.LANCZOS))
        rd_up = np.array(Image.fromarray(rgb).resize((W * s, H * s), Image.LANCZOS))
        gt_l = label(gt_up, f"wrist GT  |  {ep_short[:30]}  t={t}")
        rd_l = label(rd_up, f"wrist rendered (VGGT+scale={scale:.2f})  t={t}")
        sep = np.zeros((gt_l.shape[0], 8, 3), dtype=np.uint8)
        frames_out.append(np.concatenate([gt_l, sep, rd_l], axis=1))

    out = CALIB_DIR / f"{ep_short}_wrist_video.mp4"
    imageio.mimsave(out, frames_out, fps=15, macro_block_size=1)
    print(f"  wrote {out.name}  ({len(frames_out)} frames, {out.stat().st_size / 1024:.0f} KB)")
    return out, frames_out


def main():
    episode_frames = []
    target_W = None
    for ep_dir in sorted(WRIST_ROOT.iterdir()):
        if not (ep_dir / "data.npz").exists(): continue
        calib_p = CALIB_DIR / f"{ep_dir.name}.json"
        if not calib_p.exists():
            print(f"  skip {ep_dir.name}: no VGGT calib")
            continue
        print(f"=== {ep_dir.name} ===")
        _, frames = render_episode(ep_dir.name)
        episode_frames.append((ep_dir.name, frames))
        if target_W is None: target_W = frames[0].shape[1]


if __name__ == "__main__":
    main()
