"""Render `headline_big`-style videos (GT | rendered | rendered-on-GT, 4x upscaled)
for a curated set of diverse DROID-100 episodes spanning multiple institutions
and camera serials.

Two-pass to avoid the TF+pyrender EGL/CUDA segfault we hit elsewhere:
  Pass 1 (--mode extract): TF loads DROID-100, writes per-episode arrays.npz + info.json.
  Pass 2 (--mode render):  reads .npz, runs pyrender, writes headline_big.mp4 (no TF).

Default (no flag): runs both passes in subprocesses sequentially.

Outputs:
  outputs/headline_diverse/<short_id>/
    info.json
    headline_big.mp4   GT | rendered Franka | rendered on GT, 4x upscale, labeled, fps=15
"""

from __future__ import annotations

import os

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CALIB_DIR = Path(os.environ.get("DROID_CALIB_DIR", "/storage/project/r-agarg35-0/lwang831/droid/calibration"))
DROID100 = Path(os.environ.get("DROID_RLDS_DIR", "/storage/project/r-agarg35-0/lwang831/droid/droid_100/1.0.0"))
ROOT = Path(__file__).parent / "outputs" / "headline_diverse"
ROOT.mkdir(parents=True, exist_ok=True)

# Curated for institution/camera diversity. Each must be IoU-calibrated. The script
# walks this list in order, skipping any episode with bad native intrinsics, and
# stops once we have TARGET_COUNT good renders.
TARGET_EPISODES: list[str] = [
    "AUTOLab+84bd5053+2023-07-14-16h-55m-45s",
    "AUTOLab+0d4edc83+2023-10-27-20h-16m-47s",
    "CLVR+236539bc+2023-05-09-02h-08m-20s",
    "ILIAD+sbd7d2c6+2023-04-21-18h-25m-07s",
    "RAIL+80edfcb1+2023-06-29-13h-20m-45s",
    "IPRL+5085c3ce+2023-10-14-21h-06m-37s",
    "TRI+52ca9b6a+2023-11-01-16h-08m-47s",
    "RPL+32cba90c+2023-12-04-18h-30m-04s",
]
TARGET_COUNT = 5  # stop after this many successful renders
RPL_FRAME_CAP = 450  # long RPL trajectory; cap at ~30s for demo size


def short_id(ep_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_+-]", "_", ep_id)[:40]


def label_strip(text: str, w: int, h: int = 32, font_size: int = 22) -> np.ndarray:
    img = Image.new("RGB", (w, h), color=(0, 0, 0))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    d.text((8, 4), text, fill=(240, 240, 240), font=font)
    return np.array(img)


def attach_label(frames: list[np.ndarray], text: str) -> list[np.ndarray]:
    lbl = label_strip(text, w=frames[0].shape[1])
    return [np.concatenate([lbl, f], axis=0) for f in frames]


def blend(gt: np.ndarray, render: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    coverage = (render.sum(-1) > 0).astype(np.float32)[..., None]
    out = gt.astype(np.float32) * (1 - alpha * coverage) + render.astype(np.float32) * (alpha * coverage)
    return out.clip(0, 255).astype(np.uint8)


def upscale(frames: list[np.ndarray], factor: int) -> list[np.ndarray]:
    return [
        np.array(Image.fromarray(f).resize(
            (f.shape[1] * factor, f.shape[0] * factor), Image.LANCZOS,
        ))
        for f in frames
    ]


def extract_pass() -> None:
    """TF pass: walk TARGET_EPISODES and write arrays.npz + info.json."""
    import tensorflow_datasets as tfds
    from ltx_action_cond.calibration import extrinsic_6dof_to_matrix, intrinsics_to_K, rescale_K
    from ltx_action_cond.droid import (
        CalibrationIndex,
        extract_episode_arrays,
        find_iou_calibrated_episodes,
        serial_to_rls_image_key,
    )

    calib = CalibrationIndex.load(CALIB_DIR)
    builder = tfds.builder_from_directory(str(DROID100))
    ds = builder.as_dataset(split="train")
    by_id = {c.episode_id: c for c in find_iou_calibrated_episodes(ds, calib, min_quality=0.70)}

    # Build a map: ep_index -> candidate, only for the ones we want.
    wanted = []
    for ep_id in TARGET_EPISODES:
        if ep_id not in by_id:
            print(f"  not IoU-calibrated, skipping: {ep_id}")
            continue
        wanted.append(by_id[ep_id])
    wanted_idx = {c.ep_index: c for c in wanted}
    print(f"Want {len(wanted)} episodes; iterating dataset...")

    success = 0
    cached_eps = {}
    for i, ep in enumerate(ds):
        if i in wanted_idx:
            cached_eps[i] = ep
        if len(cached_eps) >= len(wanted_idx):
            break

    for cand in wanted:
        ep = cached_eps.get(cand.ep_index)
        if ep is None:
            print(f"  episode index {cand.ep_index} missing for {cand.episode_id}")
            continue

        intr = calib.intrinsics[cand.episode_id][cand.serial]
        native_w, native_h = intr["width"], intr["height"]
        if native_w <= 0 or native_h <= 0:
            print(f"  skipped (bad intrinsics {native_w}x{native_h}): {cand.episode_id}")
            continue

        cam2base_mat = extrinsic_6dof_to_matrix(calib.cam2base[cand.episode_id][cand.serial])
        K_native = intrinsics_to_K(intr["cameraMatrix"])
        img_key = serial_to_rls_image_key(calib.camera_serials, cand.episode_id, cand.serial)
        role = {v: k for k, v in calib.camera_serials[cand.episode_id].items()}[cand.serial]
        arrays = extract_episode_arrays(ep, img_key=img_key)
        gt_frames = arrays["frames"]

        # Cap RPL (1139-frame) for demo size.
        if cand.episode_id.startswith("RPL+") and gt_frames.shape[0] > RPL_FRAME_CAP:
            gt_frames = gt_frames[:RPL_FRAME_CAP]
            arrays["cmd_joint_position"] = arrays["cmd_joint_position"][:RPL_FRAME_CAP]
            arrays["cmd_gripper_position"] = arrays["cmd_gripper_position"][:RPL_FRAME_CAP]

        T, H, W = gt_frames.shape[:3]
        K_rls = rescale_K(K_native, src_size=(native_w, native_h), dst_size=(W, H))
        out_dir = ROOT / short_id(cand.episode_id)
        out_dir.mkdir(exist_ok=True)
        np.savez(
            out_dir / "arrays.npz",
            cmd_joint_position=arrays["cmd_joint_position"],
            cmd_gripper_position=arrays["cmd_gripper_position"],
            gt_frames=gt_frames,
            cam2base=cam2base_mat,
            K_rls=K_rls,
        )
        info = {
            "episode_id": cand.episode_id,
            "institution": cand.episode_id.split("+")[0],
            "camera_serial": cand.serial,
            "camera_role": role,
            "img_key": img_key,
            "quality_metric_IoU": cand.quality,
            "n_steps": int(T),
            "duration_sec_at_15Hz": round(T / 15.0, 2),
            "native_camera_size": [int(native_w), int(native_h)],
        }
        with (out_dir / "info.json").open("w") as f:
            json.dump(info, f, indent=2)
        success += 1
        print(f"  extracted ({success}/{TARGET_COUNT}+): inst={info['institution']:8s}  "
              f"cam={cand.serial[-6:]}  IoU={cand.quality:.2f}  T={T}  -> {out_dir.name}/")


def render_pass() -> None:
    """Render pass: read .npz, produce headline_big.mp4 for each episode."""
    from ltx_action_cond.mesh_rendering import FrankaMeshRenderer

    dirs = sorted([d for d in ROOT.iterdir() if (d / "arrays.npz").exists()])
    print(f"Rendering {len(dirs)} episodes...")
    rendered = 0
    for ep_dir in dirs:
        with (ep_dir / "info.json").open() as f:
            info = json.load(f)
        data = np.load(ep_dir / "arrays.npz")
        cmd_joints = data["cmd_joint_position"]
        cmd_grippers = data["cmd_gripper_position"]
        gt_frames = data["gt_frames"]
        cam2base_mat = data["cam2base"]
        K_rls = data["K_rls"]
        T, H, W = gt_frames.shape[:3]

        print(f"=== {info['episode_id']}  inst={info['institution']}  IoU={info['quality_metric_IoU']:.2f}  T={T} ===")
        r = FrankaMeshRenderer(size=(W, H), mode="photo")
        photo_frames, overlay_frames = [], []
        for t in range(T):
            color, _ = r.render(cmd_joints[t], K_rls, cam2base_mat,
                                gripper_position=float(cmd_grippers[t]))
            photo_frames.append(color)
            overlay_frames.append(blend(gt_frames[t], color))
        r.close()

        # 4x upscale + label + 3-up.
        scale = 4
        gt_up = upscale(list(gt_frames), scale)
        photo_up = upscale(photo_frames, scale)
        ovl_up = upscale(overlay_frames, scale)
        label_tail = f"{info['institution']} | IoU {info['quality_metric_IoU']:.2f} | {info['camera_role']}"
        gt_lbl = attach_label(gt_up, f"GT | {label_tail}")
        ph_lbl = attach_label(photo_up, f"rendered | {label_tail}")
        ov_lbl = attach_label(ovl_up, f"rendered on GT | {label_tail}")

        sep_h = gt_lbl[0].shape[0]
        sep = np.zeros((sep_h, 8, 3), dtype=np.uint8)
        three = [
            np.concatenate([gt_lbl[t], sep, ph_lbl[t], sep, ov_lbl[t]], axis=1)
            for t in range(T)
        ]
        out = ep_dir / "headline_big.mp4"
        imageio.mimsave(out, three, fps=15, macro_block_size=1)
        kb = out.stat().st_size / 1024
        print(f"  wrote {out.relative_to(ROOT.parent)}  ({kb:.0f} KB, {T} frames)")
        rendered += 1
        if rendered >= TARGET_COUNT and len(dirs) > TARGET_COUNT:
            # Render a couple extras since we have them, but log when target hit.
            pass
    print(f"\nDone. {rendered} headline_big videos under {ROOT}/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["extract", "render", "both"], default="both")
    args = parser.parse_args()
    if args.mode == "both":
        subprocess.run([sys.executable, __file__, "--mode", "extract"], check=True)
        subprocess.run([sys.executable, __file__, "--mode", "render"], check=True)
        return
    if args.mode == "extract":
        extract_pass()
    elif args.mode == "render":
        render_pass()


if __name__ == "__main__":
    main()
