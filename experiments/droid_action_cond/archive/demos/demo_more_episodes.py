"""Render the photoreal + overlay videos for a few additional IoU-calibrated episodes
so we can eyeball whether the prior generalizes across institutions and scenes.

Two-pass to avoid an EGL/CUDA segfault when TF and pyrender share a process:
  Pass 1 (--extract): TF loads DROID-100, writes per-episode .npz with arrays + calib.
  Pass 2 (--render):  reads the .npz files, runs pyrender + writes videos. No TF.

Default (no flag): runs both passes in subprocesses sequentially.

Outputs:
  outputs/extra_episodes/<short_id>/
    info.json                        IoU + n_steps + camera + length
    gt.mp4                           ground truth video at 15Hz
    photo.mp4                        rendered Franka alone
    overlay.mp4                      rendered Franka blended on GT
    headline_3up.mp4                 [GT | photoreal | overlay] side-by-side
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CALIB_DIR = Path(os.environ.get("DROID_CALIB_DIR", "/storage/project/r-agarg35-0/lwang831/droid/calibration"))
DROID100 = Path(os.environ.get("DROID_RLDS_DIR", "/storage/project/r-agarg35-0/lwang831/droid/droid_100/1.0.0"))
ROOT = Path(__file__).parent / "outputs" / "extra_episodes"
ROOT.mkdir(parents=True, exist_ok=True)

# Skip the one we've been using as the primary demo.
PRIMARY_EPISODE_ID = "WEIRD+f1c42455+2024-01-09-03h-11m-42s"
# How many additional episodes to produce
N_EXTRA = 3


def short_id(ep_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_+-]", "_", ep_id)[:40]


def label_strip(text: str, w: int, h: int = 22) -> np.ndarray:
    img = Image.new("RGB", (w, h), color=(0, 0, 0))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    d.text((6, 2), text, fill=(240, 240, 240), font=font)
    return np.array(img)


def attach_label(frames: list[np.ndarray], text: str) -> list[np.ndarray]:
    lbl = label_strip(text, w=frames[0].shape[1])
    return [np.concatenate([lbl, f], axis=0) for f in frames]


def blend(gt: np.ndarray, render: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    coverage = (render.sum(-1) > 0).astype(np.float32)[..., None]
    out = gt.astype(np.float32) * (1 - alpha * coverage) + render.astype(np.float32) * (alpha * coverage)
    return out.clip(0, 255).astype(np.uint8)


def extract_one(builder, calib, cand) -> Path | None:
    """Pass 1: pull arrays + calibration out of TFDS, save to .npz. Skip bad intrinsics."""
    from ltx_action_cond.calibration import extrinsic_6dof_to_matrix, intrinsics_to_K, rescale_K
    from ltx_action_cond.droid import extract_episode_arrays, serial_to_rls_image_key

    intr = calib.intrinsics[cand.episode_id][cand.serial]
    native_w, native_h = intr["width"], intr["height"]
    if native_w <= 0 or native_h <= 0:
        print(f"  skipped: bad native intrinsic size ({native_w}x{native_h})")
        return None
    cam2base_mat = extrinsic_6dof_to_matrix(calib.cam2base[cand.episode_id][cand.serial])
    K_native = intrinsics_to_K(intr["cameraMatrix"])
    img_key = serial_to_rls_image_key(calib.camera_serials, cand.episode_id, cand.serial)
    role = {v: k for k, v in calib.camera_serials[cand.episode_id].items()}[cand.serial]
    out_dir = ROOT / short_id(cand.episode_id)
    out_dir.mkdir(exist_ok=True)

    ds = builder.as_dataset(split="train")
    chosen = None
    for i, ep in enumerate(ds):
        if i == cand.ep_index:
            chosen = ep
            break
    assert chosen is not None
    arrays = extract_episode_arrays(chosen, img_key=img_key)
    gt_frames = arrays["frames"]
    T, H, W = gt_frames.shape[:3]
    K_rls = rescale_K(K_native, src_size=(native_w, native_h), dst_size=(W, H))

    npz = out_dir / "arrays.npz"
    np.savez(
        npz,
        cmd_joint_position=arrays["cmd_joint_position"],
        cmd_gripper_position=arrays["cmd_gripper_position"],
        gt_frames=gt_frames,
        cam2base=cam2base_mat,
        K_rls=K_rls,
    )
    info = {
        "episode_id": cand.episode_id,
        "relpath": cand.relpath,
        "camera_serial": cand.serial,
        "camera_role": role,
        "img_key": img_key,
        "quality_metric_IoU": cand.quality,
        "n_steps": int(T),
        "duration_sec_at_15Hz": round(T / 15.0, 2),
    }
    with (out_dir / "info.json").open("w") as f:
        json.dump(info, f, indent=2)
    print(f"  extracted -> {out_dir}/  T={T} {W}x{H}")
    return out_dir


def render_one(ep_dir: Path) -> None:
    """Pass 2: read .npz, render videos with pyrender. No TF here."""
    from ltx_action_cond.mesh_rendering import FrankaMeshRenderer

    with (ep_dir / "info.json").open() as f:
        info = json.load(f)
    data = np.load(ep_dir / "arrays.npz")
    cmd_joints = data["cmd_joint_position"]
    cmd_grippers = data["cmd_gripper_position"]
    gt_frames = data["gt_frames"]
    cam2base_mat = data["cam2base"]
    K_rls = data["K_rls"]
    T, H, W = gt_frames.shape[:3]
    print(f"=== {info['episode_id']}  T={T} {W}x{H} ===")

    renderer = FrankaMeshRenderer(size=(W, H), mode="photo")
    photo_frames, overlay_frames = [], []
    for t in range(T):
        color, _ = renderer.render(cmd_joints[t], K_rls, cam2base_mat,
                                   gripper_position=float(cmd_grippers[t]))
        photo_frames.append(color)
        overlay_frames.append(blend(gt_frames[t], color))
    renderer.close()

    label = f"{info['episode_id'][:24]} | IoU {info['quality_metric_IoU']:.2f} | {info['camera_role']}"
    gt_lbl = attach_label(list(gt_frames), f"GT | {label}")
    photo_lbl = attach_label(photo_frames, f"photoreal | {label}")
    overlay_lbl = attach_label(overlay_frames, f"overlay | {label}")

    imageio.mimsave(ep_dir / "gt.mp4", gt_lbl, fps=15, macro_block_size=1)
    imageio.mimsave(ep_dir / "photo.mp4", photo_lbl, fps=15, macro_block_size=1)
    imageio.mimsave(ep_dir / "overlay.mp4", overlay_lbl, fps=15, macro_block_size=1)

    sep = np.zeros((gt_lbl[0].shape[0], 4, 3), dtype=np.uint8)
    three = [
        np.concatenate([gt_lbl[t], sep, photo_lbl[t], sep, overlay_lbl[t]], axis=1)
        for t in range(T)
    ]
    imageio.mimsave(ep_dir / "headline_3up.mp4", three, fps=15, macro_block_size=1)
    print(f"  wrote 4 videos -> {ep_dir}/")


def extract_pass() -> list[Path]:
    import tensorflow_datasets as tfds
    from ltx_action_cond.droid import CalibrationIndex, find_iou_calibrated_episodes

    calib = CalibrationIndex.load(CALIB_DIR)
    builder = tfds.builder_from_directory(str(DROID100))
    ds = builder.as_dataset(split="train")
    candidates = find_iou_calibrated_episodes(ds, calib, min_quality=0.85)
    candidates = [c for c in candidates if c.episode_id != PRIMARY_EPISODE_ID]
    candidates.sort(key=lambda c: (-c.quality, -c.n_steps))
    print(f"Top non-primary IoU candidates: {len(candidates)}")
    for c in candidates[:6]:
        print(f"  IoU={c.quality:.3f}  n={c.n_steps:4d}  ep={c.episode_id[:50]}")

    extracted: list[Path] = []
    for cand in candidates:
        if len(extracted) >= N_EXTRA:
            break
        d = extract_one(builder, calib, cand)
        if d is not None:
            extracted.append(d)
    return extracted


def render_pass() -> None:
    for ep_dir in sorted(ROOT.iterdir()):
        if not (ep_dir / "arrays.npz").exists():
            continue
        render_one(ep_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["extract", "render", "both"], default="both")
    args = parser.parse_args()
    if args.mode in ("extract", "both"):
        if args.mode == "both":
            # Split: TF in one process, pyrender in another (they segfault together on EGL).
            subprocess.run([sys.executable, __file__, "--mode", "extract"], check=True)
            subprocess.run([sys.executable, __file__, "--mode", "render"], check=True)
            return
        extract_pass()
    if args.mode == "render":
        render_pass()


if __name__ == "__main__":
    main()
