"""Pick the highest-IoU calibrated DROID-100 episode and dump everything needed
to build a per-frame projection prior on a single fixed exterior view.

Writes (under outputs/):
  episode.npz   joints, action, gt_frames, K_rls (320x180), cam2base 4x4
  meta.json     episode_id, camera_serial/role, n_steps, IoU
  01_gt/*.png   the GT video frames at native RLDS resolution
"""

from __future__ import annotations

import os

import json
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

from ltx_action_cond.calibration import extrinsic_6dof_to_matrix, intrinsics_to_K, rescale_K
from ltx_action_cond.droid import (
    CalibrationIndex,
    extract_episode_arrays,
    find_iou_calibrated_episodes,
    serial_to_rls_image_key,
)

CALIB_DIR = Path(os.environ.get("DROID_CALIB_DIR", "/storage/project/r-agarg35-0/lwang831/droid/calibration"))
DROID100 = Path(os.environ.get("DROID_RLDS_DIR", "/storage/project/r-agarg35-0/lwang831/droid/droid_100/1.0.0"))
OUT = Path(__file__).parent / "outputs"
GT_DIR = OUT / "01_gt"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    GT_DIR.mkdir(exist_ok=True)

    calib = CalibrationIndex.load(CALIB_DIR)
    builder = tfds.builder_from_directory(str(DROID100))
    ds = builder.as_dataset(split="train")

    candidates = find_iou_calibrated_episodes(ds, calib, min_quality=0.0)
    candidates.sort(key=lambda c: (-c.quality, -c.n_steps))
    print(f"Calibrated IoU candidates: {len(candidates)}")
    for c in candidates[:5]:
        print(f"  IoU={c.quality:.3f}  n={c.n_steps:4d}  ep={c.episode_id[:60]}")

    if not candidates:
        raise SystemExit("no candidates")

    top = [c for c in candidates if c.quality >= 0.85]
    top.sort(key=lambda c: -c.n_steps)
    best = top[0] if top else candidates[0]
    print(f"\n=== CHOSEN: idx={best.ep_index} IoU={best.quality:.3f} n={best.n_steps} ===")

    # Calibration for chosen episode
    cam2base_mat = extrinsic_6dof_to_matrix(calib.cam2base[best.episode_id][best.serial])
    intr = calib.intrinsics[best.episode_id][best.serial]
    K_native = intrinsics_to_K(intr["cameraMatrix"])
    native_w, native_h = intr["width"], intr["height"]
    img_key = serial_to_rls_image_key(calib.camera_serials, best.episode_id, best.serial)
    role = {v: k for k, v in calib.camera_serials[best.episode_id].items()}[best.serial]

    # Pull the chosen episode (re-iterate since we exhausted ds)
    ds2 = builder.as_dataset(split="train")
    chosen_ep = None
    for i, ep in enumerate(ds2):
        if i == best.ep_index:
            chosen_ep = ep
            break
    assert chosen_ep is not None

    arrays = extract_episode_arrays(chosen_ep, img_key=img_key)
    joints = arrays["joint_position"]
    cmd_joints = arrays["cmd_joint_position"]
    actions = arrays["action"]
    grippers = arrays["gripper_position"]
    cmd_grippers = arrays["cmd_gripper_position"]
    frames = arrays["frames"]
    img_h, img_w = frames.shape[1:3]

    K_rls = rescale_K(K_native, src_size=(native_w, native_h), dst_size=(img_w, img_h))
    print(f"\nshapes: joint={joints.shape}  cmd={cmd_joints.shape}  action={actions.shape}  frames={frames.shape}")
    print(f"native intrinsics {native_w}x{native_h} -> RLDS {img_w}x{img_h}")
    print(f"|cmd - obs|_max joint diff: {np.abs(cmd_joints - joints).max():.4f} rad")

    for t, frame in enumerate(frames):
        Image.fromarray(frame).save(GT_DIR / f"gt_{t:04d}.png")

    np.savez(
        OUT / "episode.npz",
        joint_position=joints,
        cmd_joint_position=cmd_joints,
        action=actions,
        gripper_position=grippers,
        cmd_gripper_position=cmd_grippers,
        gt_frames=frames,
        cam2base=cam2base_mat,
        K_native=K_native,
        K_rls=K_rls,
        native_size=np.array([native_w, native_h]),
        rls_size=np.array([img_w, img_h]),
    )
    meta = {
        "episode_id": best.episode_id,
        "relpath": best.relpath,
        "camera_serial": best.serial,
        "camera_role": role,
        "img_key": img_key,
        "quality_metric_IoU": best.quality,
        "n_steps": int(joints.shape[0]),
        "duration_sec_at_15Hz": round(joints.shape[0] / 15.0, 2),
        "K_rls": K_rls.tolist(),
        "cam2base_4x4": cam2base_mat.tolist(),
    }
    with (OUT / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nWrote {OUT}/episode.npz, meta.json, 01_gt/ ({len(frames)} frames)")


if __name__ == "__main__":
    main()
