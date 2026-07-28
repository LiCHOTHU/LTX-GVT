"""Render headline_big-style videos for BOTH exterior cameras of each episode.

The April 2025 DROID calibration release only ships an IoU-calibrated extrinsic
for ONE of the two exterior cameras per episode. To get the OTHER camera's pose
in the target episode's base frame, we:

  1. Use the target episode's own (calibrated) camera-to-base for the OWN camera.
  2. Estimate the rig geometry T_{other_to_own} between the two cameras from
     sibling episodes (same lab, same camera pair):
         A has `own` calibrated  -> T_own_to_base_A
         B has `other` calibrated -> T_other_to_base_B
         T_other_to_own = inv(T_own_to_base_A) @ T_other_to_base_B
     Aggregated over many pairs by median (translation) + best-quality
     (rotation).
  3. Compose for the target:
         T_other_to_base_target = T_own_to_base_target @ T_other_to_own

This anchors the OTHER camera to the TARGET episode's actual base pose
(via the OWN camera), rather than blindly transplanting the sibling's
camera-to-base (which is wrong whenever the robot base moved between
sibling and target). The remaining assumption is rig stability across
sibling pairs, which is checked via the spread of T_{other_to_own}
estimates.

Two-pass to avoid the TF+pyrender EGL/CUDA segfault:
  Pass 1 (--mode extract): TF loads DROID-100, writes per-episode arrays_<role>.npz
                           for BOTH cameras (the calibrated one + the transplanted one).
  Pass 2 (--mode render):  pyrender renders both cameras, stacks vertically.

Default (no flag): runs both passes in subprocesses sequentially.

Outputs:
  outputs/both_cams/<short_id>/
    info.json
    cam_ext1.mp4              GT | render | overlay  for ext1 (4x upscale)
    cam_ext2.mp4              GT | render | overlay  for ext2 (4x upscale)
    headline_both_cams.mp4    two rows stacked: ext1 over ext2
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
ROOT = Path(__file__).parent / "outputs" / "both_cams"
ROOT.mkdir(parents=True, exist_ok=True)

# Selected for institution diversity. Each has a strict-pair sibling for the
# other camera. Listed best-to-worst by combined IoU; first 5 will be used.
TARGET_EPISODES: list[str] = [
    "WEIRD+f1c42455+2024-01-09-03h-11m-42s",     # 323 frames -- the primary demo episode
    "IPRL+w026bb9b+2023-04-20-23h-40m-21s",      # 89  frames, Stanford mobile cart
    "TRI+52ca9b6a+2023-11-07-15h-04m-09s",       # 284 frames, TRI kitchen
    "AUTOLab+0d4edc83+2023-10-27-20h-16m-47s",   # 87  frames, Berkeley AUTOLab
    "CLVR+236539bc+2023-05-09-02h-08m-20s",      # 167 frames, CLVR
    "ILIAD+sbd7d2c6+2023-04-21-18h-25m-07s",     # 129 frames, ILIAD (spare in case one fails)
]
TARGET_COUNT = 5


def short_id(ep_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_+-]", "_", ep_id)[:40]


def label_strip(text: str, w: int, h: int = 36, font_size: int = 24) -> np.ndarray:
    img = Image.new("RGB", (w, h), color=(0, 0, 0))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    d.text((10, 4), text, fill=(240, 240, 240), font=font)
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


# --- Pass 1: extract ---

def extract_pass() -> None:
    import tensorflow_datasets as tfds
    from ltx_action_cond.calibration import extrinsic_6dof_to_matrix, intrinsics_to_K, rescale_K
    from ltx_action_cond.droid import (
        CalibrationIndex,
        ROLE_TO_IMG_KEY,
        extract_episode_arrays,
        find_iou_calibrated_episodes,
        resolve_both_cams_cam2base,
        resolve_gripper_signal,
    )

    calib = CalibrationIndex.load(CALIB_DIR)
    builder = tfds.builder_from_directory(str(DROID100))
    ds = builder.as_dataset(split="train")

    iou_eps = {c.episode_id: c for c in find_iou_calibrated_episodes(ds, calib, min_quality=0.70)}
    wanted = [iou_eps[eid] for eid in TARGET_EPISODES if eid in iou_eps]
    print(f"Want {len(wanted)} episodes; iterating dataset to cache ep dicts...")

    wanted_idx = {c.ep_index for c in wanted}
    cached: dict[int, object] = {}
    for i, ep in enumerate(ds):
        if i in wanted_idx:
            cached[i] = ep
        if len(cached) >= len(wanted_idx):
            break

    success = 0
    for cand in wanted:
        if success >= TARGET_COUNT:
            break
        ep = cached.get(cand.ep_index)
        if ep is None:
            continue

        ep_id = cand.episode_id
        sers = calib.camera_serials[ep_id]
        ext1, ext2 = sers.get("ext1_cam_serial"), sers.get("ext2_cam_serial")
        own_serial = cand.serial
        other_serial = ext2 if own_serial == ext1 else ext1
        own_role = "ext1" if own_serial == ext1 else "ext2"
        other_role = "ext2" if own_role == "ext1" else "ext1"

        intr_own = calib.intrinsics[ep_id][own_serial]
        intr_oth = calib.intrinsics[ep_id][other_serial]
        if intr_own.get("width", 0) <= 0 or intr_oth.get("width", 0) <= 0:
            print(f"  skipping {ep_id}: bad intrinsics")
            continue

        # Resolve cam2base for BOTH cameras using the best available source:
        # superset (both cams directly) -> target's own cam2cam rig + own
        # cam2base -> sibling cam2cam rig + own cam2base.
        resolved = resolve_both_cams_cam2base(calib, ep_id)
        if resolved is None:
            print(f"  skipping {ep_id}: no resolved cam2base for both cameras")
            continue
        cam2base_own = resolved[f"{own_role}_cam2base"]
        cam2base_oth = resolved[f"{other_role}_cam2base"]
        own_src = resolved[f"{own_role}_source"]
        oth_src = resolved[f"{other_role}_source"]
        rig_source_ep = resolved.get("rig_source_ep") or ep_id
        rig_window = resolved.get("rig_time_window_days")

        K_own_native = intrinsics_to_K(intr_own["cameraMatrix"])
        K_oth_native = intrinsics_to_K(intr_oth["cameraMatrix"])

        own_img_key = ROLE_TO_IMG_KEY[f"{own_role}_cam_serial"]
        oth_img_key = ROLE_TO_IMG_KEY[f"{other_role}_cam_serial"]
        arr_own = extract_episode_arrays(ep, img_key=own_img_key)
        arr_oth = extract_episode_arrays(ep, img_key=oth_img_key)

        T, H_own, W_own = arr_own["frames"].shape[:3]
        K_own_rls = rescale_K(K_own_native,
                              src_size=(intr_own["width"], intr_own["height"]),
                              dst_size=(W_own, H_own))
        _, H_oth, W_oth = arr_oth["frames"].shape[:3]
        K_oth_rls = rescale_K(K_oth_native,
                              src_size=(intr_oth["width"], intr_oth["height"]),
                              dst_size=(W_oth, H_oth))

        out_dir = ROOT / short_id(ep_id)
        out_dir.mkdir(exist_ok=True)
        gripper_for_render, gripper_src = resolve_gripper_signal(
            arr_own["cmd_gripper_position"], arr_own["gripper_position"],
        )
        np.savez(
            out_dir / "shared.npz",
            cmd_joint_position=arr_own["cmd_joint_position"],
            cmd_gripper_position=arr_own["cmd_gripper_position"],
            obs_gripper_position=arr_own["gripper_position"],
            gripper_for_render=gripper_for_render,
        )
        print(f"     gripper source: {gripper_src}")
        for role, arr, cam2base, K, calibrated_here, source_ep, iou, source_kind in [
            (own_role,   arr_own, cam2base_own, K_own_rls, True,  ep_id,         cand.quality, own_src),
            (other_role, arr_oth, cam2base_oth, K_oth_rls, False, rig_source_ep, cand.quality, oth_src),
        ]:
            np.savez(
                out_dir / f"cam_{role}.npz",
                gt_frames=arr["frames"],
                cam2base=cam2base,
                K_rls=K,
            )
            (out_dir / f"cam_{role}.json").write_text(json.dumps({
                "role": role,
                "calibrated_in_this_episode": calibrated_here,
                "extrinsic_source": source_kind,
                "extrinsic_source_episode": source_ep,
                "extrinsic_IoU": iou,
            }, indent=2))

        info = {
            "episode_id": ep_id,
            "institution": ep_id.split("+")[0],
            "n_steps": int(T),
            "duration_sec_at_15Hz": round(T / 15.0, 2),
            "own_role": own_role,
            "other_role": other_role,
            "ext1_serial": ext1,
            "ext2_serial": ext2,
            "cam2base_resolution": {
                f"{own_role}_source": own_src,
                f"{other_role}_source": oth_src,
                "rig_source_ep": rig_source_ep,
                "rig_time_window_days": rig_window,
            },
        }
        (out_dir / "info.json").write_text(json.dumps(info, indent=2))
        success += 1
        print(f"  extracted ({success}/{TARGET_COUNT}): {ep_id}  "
              f"own={own_role}({own_src})  other={other_role}({oth_src})  "
              f"rig_src={rig_source_ep[:40]}  window={rig_window}d")


# --- Pass 2: render ---

def render_one_camera(joints, grippers, gt_frames, cam2base, K_rls, role,
                      calibrated, source_kind, src_ep, iou):
    """Return list of 3-up frames (GT | rendered | overlay), 4x upscale, labeled."""
    from ltx_action_cond.mesh_rendering import FrankaMeshRenderer
    T, H, W = gt_frames.shape[:3]
    r = FrankaMeshRenderer(size=(W, H), mode="photo")
    photo, ovl = [], []
    for t in range(T):
        color, _ = r.render(joints[t], K_rls, cam2base, gripper_position=float(grippers[t]))
        photo.append(color)
        ovl.append(blend(gt_frames[t], color))
    r.close()
    gt_up = upscale(list(gt_frames), 4)
    ph_up = upscale(photo, 4)
    ov_up = upscale(ovl, 4)
    if source_kind == "superset":
        src = "superset (direct)"
    elif source_kind == "self_calibrated":
        src = "calibrated"
    elif source_kind == "rig_from_target_cam2cam":
        src = "cam2cam (direct rig, this ep)"
    elif source_kind == "rig_from_sibling_cam2cam":
        src = f"cam2cam rig via {src_ep[:20]}"
    else:
        src = f"unknown<-{src_ep[:20]}"
    base = f"{role} | {src}"
    gt_lbl = attach_label(gt_up, f"GT | {base}")
    ph_lbl = attach_label(ph_up, f"rendered | {base}")
    ov_lbl = attach_label(ov_up, f"rendered on GT | {base}")
    sep_h = gt_lbl[0].shape[0]
    sep = np.zeros((sep_h, 8, 3), dtype=np.uint8)
    three = [
        np.concatenate([gt_lbl[t], sep, ph_lbl[t], sep, ov_lbl[t]], axis=1)
        for t in range(T)
    ]
    return three


def render_pass() -> None:
    for ep_dir in sorted(ROOT.iterdir()):
        if not (ep_dir / "info.json").exists():
            continue
        info = json.loads((ep_dir / "info.json").read_text())
        ep_id = info["episode_id"]
        shared = np.load(ep_dir / "shared.npz")
        joints = shared["cmd_joint_position"]
        # Use the rescaled gripper signal if it was saved (prefers obs over cmd
        # when obs has usable range), else fall back to raw cmd.
        grippers = shared["gripper_for_render"] if "gripper_for_render" in shared else shared["cmd_gripper_position"]
        print(f"=== {ep_id}  inst={info['institution']}  T={info['n_steps']} ===")

        per_cam_videos: dict[str, list[np.ndarray]] = {}
        for role in ("ext1", "ext2"):
            cam_npz = np.load(ep_dir / f"cam_{role}.npz")
            cam_meta = json.loads((ep_dir / f"cam_{role}.json").read_text())
            three = render_one_camera(
                joints, grippers,
                cam_npz["gt_frames"], cam_npz["cam2base"], cam_npz["K_rls"],
                role,
                cam_meta["calibrated_in_this_episode"],
                cam_meta.get("extrinsic_source", "transplant"),
                cam_meta["extrinsic_source_episode"],
                cam_meta["extrinsic_IoU"],
            )
            imageio.mimsave(ep_dir / f"cam_{role}.mp4", three, fps=15, macro_block_size=1)
            per_cam_videos[role] = three
            print(f"  wrote cam_{role}.mp4  ({len(three)} frames)")

        # Stack ext1 over ext2 (with a thick separator + a centered label between them).
        ext1_frames, ext2_frames = per_cam_videos["ext1"], per_cam_videos["ext2"]
        n = min(len(ext1_frames), len(ext2_frames))
        W = ext1_frames[0].shape[1]
        gap = np.full((10, W, 3), 32, dtype=np.uint8)
        title_top = label_strip(f"{ep_id}  |  inst={info['institution']}  |  T={info['n_steps']} frames",
                                w=W, h=42, font_size=26)
        stacked = [
            np.concatenate([title_top, ext1_frames[t], gap, ext2_frames[t]], axis=0)
            for t in range(n)
        ]
        out = ep_dir / "headline_both_cams.mp4"
        imageio.mimsave(out, stacked, fps=15, macro_block_size=1)
        kb = out.stat().st_size / 1024
        print(f"  wrote {out.name}  ({kb:.0f} KB)")


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
    else:
        render_pass()


if __name__ == "__main__":
    main()
