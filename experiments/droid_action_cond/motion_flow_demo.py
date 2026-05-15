"""Motion-flow visualisation demo across 5 DROID-100 episodes and all 3 views.

For each episode in ``outputs/wrist_demo/<ep>/`` (already populated by `wrist_demo.py`),
produces ``outputs/motion_flow_demo/<ep>/<view>.mp4`` with four columns per row:

  [ GT | rendered Franka | rendered-on-GT | motion flow (HSV) ]

and a stacked headline mp4 showing all three views together.

The motion flow comes from ``ltx_action_cond.motion_flow.flow_from_joints_for_view``:
analytical per-pixel optical flow built by FK-projecting the 9 Franka link keypoints
into each camera and propagating their pixel-velocity to nearby pixels. For the wrist
camera the extrinsic is time-varying (``T_hand_to_base(t) @ NOMINAL_T_CAM_TO_HAND``),
so flow is computed with the same per-frame pose used by the wrist renderer.

Run with ``conda env ltx`` activated (the only deps are torch/numpy/imageio/pyrender):

    python motion_flow_demo.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Use EGL backend for pyrender (no X server needed).
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from ltx_action_cond.kinematics import fk_urdf  # noqa: E402
from ltx_action_cond.mesh_rendering import FrankaMeshRenderer  # noqa: E402
from ltx_action_cond.motion_flow import flow_from_joints_for_view, flow_to_rgb  # noqa: E402
from ltx_action_cond.wrist_render import (  # noqa: E402
    NOMINAL_T_CAM_TO_HAND,
    reconstruct_scene_from_plane,
    render_wrist_scene_splat,
)

WRIST_CALIB_DIR = Path(__file__).parent / "outputs" / "wrist_calib_vggt"
SOURCE_DIR = Path(__file__).parent / "outputs" / "wrist_demo"
OUT_ROOT = Path(__file__).parent / "outputs" / "motion_flow_demo"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

FPS = 15
UPSCALE = 3


def short_id(ep_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_+-]", "_", ep_id)[:40]


def label_strip(text: str, w: int, h: int = 28, font_size: int = 18) -> np.ndarray:
    img = Image.new("RGB", (w, h), color=(0, 0, 0))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    d.text((8, 4), text, fill=(240, 240, 240), font=font)
    return np.array(img)


def upscale(frames: list[np.ndarray], factor: int) -> list[np.ndarray]:
    return [
        np.array(Image.fromarray(f).resize((f.shape[1] * factor, f.shape[0] * factor), Image.LANCZOS))
        for f in frames
    ]


def blend(gt: np.ndarray, render: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    cov = (render.sum(-1) > 0).astype(np.float32)[..., None]
    return (
        gt.astype(np.float32) * (1 - alpha * cov) + render.astype(np.float32) * (alpha * cov)
    ).clip(0, 255).astype(np.uint8)


def panels_for_view(
    gt: np.ndarray,
    rendered: np.ndarray,
    flow_rgb_frames: np.ndarray,
    label_tail: str,
) -> list[np.ndarray]:
    """Build a per-frame 4-column strip [GT | rendered | overlay | flow]."""
    T = gt.shape[0]
    flow_padded = np.concatenate([flow_rgb_frames, flow_rgb_frames[-1:]], axis=0) if flow_rgb_frames.shape[0] == T - 1 else flow_rgb_frames

    gtU = upscale(list(gt), UPSCALE)
    rU = upscale([rendered[i] for i in range(T)], UPSCALE)
    ovU = upscale([blend(gt[i], rendered[i]) for i in range(T)], UPSCALE)
    flU = upscale([flow_padded[i] for i in range(T)], UPSCALE)

    label_h = 28
    labels = [label_strip(f"{tag} | {label_tail}", w=gtU[0].shape[1], h=label_h) for tag in ("GT", "rendered", "overlay", "flow")]
    sep = np.zeros((gtU[0].shape[0] + label_h, 6, 3), dtype=np.uint8)

    strips = []
    for i in range(T):
        row = np.concatenate(
            [
                np.concatenate([labels[0], gtU[i]], axis=0),
                sep,
                np.concatenate([labels[1], rU[i]], axis=0),
                sep,
                np.concatenate([labels[2], ovU[i]], axis=0),
                sep,
                np.concatenate([labels[3], flU[i]], axis=0),
            ],
            axis=1,
        )
        strips.append(row)
    return strips


def per_frame_wrist_cam2base(joints: np.ndarray, T_cam_to_hand: np.ndarray) -> np.ndarray:
    """Build (T, 4, 4) cam2base for the wrist: T_hand_to_base(joints[t]) @ T_cam_to_hand."""
    T = joints.shape[0]
    out = np.zeros((T, 4, 4), dtype=np.float64)
    for t in range(T):
        T_hand_to_base = fk_urdf(joints[t])["hand"]
        out[t] = T_hand_to_base @ T_cam_to_hand
    return out


def load_wrist_T_cam_to_hand(ep_id: str) -> np.ndarray:
    """Per-episode wrist mount transform from VGGT calibration; fall back to nominal."""
    p = WRIST_CALIB_DIR / f"{ep_id}.json"
    if p.exists():
        d = json.loads(p.read_text())
        if "T_cam_to_hand" in d:
            return np.asarray(d["T_cam_to_hand"], dtype=np.float64)
    return NOMINAL_T_CAM_TO_HAND.copy()


def process_episode(ep_dir: Path, renderer: FrankaMeshRenderer) -> None:
    info_path = ep_dir / "info.json"
    data_path = ep_dir / "data.npz"
    if not (info_path.exists() and data_path.exists()):
        return
    info = json.loads(info_path.read_text())
    ep_id = info["episode_id"]
    inst = info["institution"]

    d = np.load(data_path)
    joints = d["cmd_joint_position"]
    grippers = d["gripper_for_render"].reshape(-1)
    K1 = d["K1"]; K2 = d["K2"]; K_wrist = d["K_wrist"]
    cam2base_1 = d["cam2base_1"]; cam2base_2 = d["cam2base_2"]
    ext1f = d["ext1_frames"]; ext2f = d["ext2_frames"]; wristf = d["wrist_frames"]
    T = ext1f.shape[0]
    H, W = ext1f.shape[1], ext1f.shape[2]

    print(f"=== {ep_id}  T={T} ===")
    T_cam_to_hand = load_wrist_T_cam_to_hand(ep_id)
    cam2base_wrist_per_frame = per_frame_wrist_cam2base(joints, T_cam_to_hand)

    # --- 1. Motion flow per view (analytical, FK-projected). ---
    # Fixed-exterior cameras: arm flow only (plane flow is zero for a static camera).
    flow_ext1 = flow_from_joints_for_view(joints, K1, cam2base_1, (W, H))
    flow_ext2 = flow_from_joints_for_view(joints, K2, cam2base_2, (W, H))
    # Wrist: scene-plane egomotion dominates; arm overrides where keypoints project on-screen.
    flow_wrist = flow_from_joints_for_view(
        joints, K_wrist, cam2base_wrist_per_frame, (W, H), plane_z=0.0
    )

    # Per-view 99th-percentile normaliser. Using the raw max would let a handful of
    # grazing-ray outliers (typical for the wrist camera looking near the table plane)
    # wash everything else out. p99 keeps typical motion visibly saturated; the rare
    # pixels above p99 just clip.
    def per_view_p99(flow: np.ndarray) -> float:
        if flow.size == 0:
            return 1e-3
        mag = np.linalg.norm(flow, axis=1)
        return float(max(np.percentile(mag, 99.0), 1e-3))

    max_ext1 = per_view_p99(flow_ext1)
    max_ext2 = per_view_p99(flow_ext2)
    max_wrist = per_view_p99(flow_wrist)
    flow_rgb_ext1 = flow_to_rgb(flow_ext1, max_magnitude=max_ext1)
    flow_rgb_ext2 = flow_to_rgb(flow_ext2, max_magnitude=max_ext2)
    flow_rgb_wrist = flow_to_rgb(flow_wrist, max_magnitude=max_wrist)

    # --- 2. Per-view renders. ---
    # ext1 / ext2 : photoreal Franka mesh (Stage 2) at the fixed camera pose.
    # wrist       : "wrist0 anchor" approach — scene reconstructed from GT wrist[0]
    #               through the VGGT-calibrated wrist pose at t=0, then re-rendered
    #               through the wrist pose at every frame t. Mirrors
    #               `wrist_video_from_wrist0.py`. This is the version that matches
    #               GT wrist[0] exactly at t=0 and stays close downstream.
    scene = reconstruct_scene_from_plane(
        wristf[0], K_wrist, cam2base_wrist_per_frame[0], plane_z=0.0, sample_stride=1
    )

    ext1_renders, ext2_renders, wrist_renders = [], [], []
    for t in range(T):
        ext1_renders.append(renderer.render(joints[t], K1, cam2base_1, gripper_position=float(grippers[t]))[0])
        ext2_renders.append(renderer.render(joints[t], K2, cam2base_2, gripper_position=float(grippers[t]))[0])
        scene_rgb, _ = render_wrist_scene_splat(
            scene, cam2base_wrist_per_frame[t], K_wrist, (W, H), point_radius_px=3
        )
        wrist_renders.append(scene_rgb)
    ext1_renders = np.stack(ext1_renders, axis=0)
    ext2_renders = np.stack(ext2_renders, axis=0)
    wrist_renders = np.stack(wrist_renders, axis=0)

    # --- 3. Per-view 4-column strip mp4s. ---
    out_dir = OUT_ROOT / short_id(ep_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    ext1_strip = panels_for_view(ext1f, ext1_renders, flow_rgb_ext1, f"ext1 | {inst}")
    ext2_strip = panels_for_view(ext2f, ext2_renders, flow_rgb_ext2, f"ext2 | {inst}")
    wrist_strip = panels_for_view(wristf, wrist_renders, flow_rgb_wrist, f"wrist | {inst}")

    imageio.mimsave(out_dir / "ext1.mp4", ext1_strip, fps=FPS, macro_block_size=1)
    imageio.mimsave(out_dir / "ext2.mp4", ext2_strip, fps=FPS, macro_block_size=1)
    imageio.mimsave(out_dir / "wrist.mp4", wrist_strip, fps=FPS, macro_block_size=1)

    # --- 4. Stacked headline mp4 (3 views, 4 panels each). ---
    max_w = max(ext1_strip[0].shape[1], ext2_strip[0].shape[1], wrist_strip[0].shape[1])

    def pad_to(rows: list[np.ndarray], target_w: int) -> list[np.ndarray]:
        out = []
        for r in rows:
            pad = target_w - r.shape[1]
            if pad > 0:
                r = np.concatenate([r, np.zeros((r.shape[0], pad, 3), dtype=np.uint8)], axis=1)
            out.append(r)
        return out

    ext1_pad = pad_to(ext1_strip, max_w)
    ext2_pad = pad_to(ext2_strip, max_w)
    wrist_pad = pad_to(wrist_strip, max_w)
    title = label_strip(
        f"{ep_id} | {inst} | T={T} | flow norm (px): ext1={max_ext1:.1f}, ext2={max_ext2:.1f}, wrist={max_wrist:.1f}",
        w=max_w, h=36, font_size=20,
    )
    gap = np.full((6, max_w, 3), 80, dtype=np.uint8)
    stacked = [
        np.concatenate([title, ext1_pad[i], gap, ext2_pad[i], gap, wrist_pad[i]], axis=0)
        for i in range(T)
    ]
    imageio.mimsave(out_dir / "headline_all.mp4", stacked, fps=FPS, macro_block_size=1)
    print(f"  wrote {out_dir.relative_to(Path.cwd()) if out_dir.is_relative_to(Path.cwd()) else out_dir} / headline_all.mp4  ({T} frames)")


def main() -> None:
    ep_dirs = sorted(p for p in SOURCE_DIR.iterdir() if p.is_dir() and (p / "data.npz").exists())
    if not ep_dirs:
        raise SystemExit(f"No episode data found under {SOURCE_DIR}. Run wrist_demo.py first.")
    print(f"Processing {len(ep_dirs)} episodes from {SOURCE_DIR}")
    H = W = 0
    for ep_dir in ep_dirs:
        d = np.load(ep_dir / "data.npz")
        H, W = d["ext1_frames"].shape[1], d["ext1_frames"].shape[2]
        break
    renderer = FrankaMeshRenderer(size=(W, H), mode="photo")
    try:
        for ep_dir in ep_dirs:
            process_episode(ep_dir, renderer)
    finally:
        renderer.close()
    print(f"\nAll outputs under {OUT_ROOT}")


if __name__ == "__main__":
    main()
