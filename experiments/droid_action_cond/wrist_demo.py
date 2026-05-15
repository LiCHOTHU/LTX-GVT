"""Wrist-camera synthetic-view demo for 5 randomly sampled DROID-100 episodes.

For each episode, produces:
  outputs/wrist_demo/<ep>/
    info.json
    ext1.mp4              [GT | rendered Franka | overlay], 3x upscale
    ext2.mp4              [GT | rendered Franka | overlay], 3x upscale
    wrist.mp4             [GT | rendered synthetic], 3x upscale
    headline_all.mp4      ext1 over ext2 over wrist, single stacked video

Wrist render pipeline:
  - reconstruct a base-frame coloured point cloud at t=0 by back-projecting
    ext1 onto a horizontal plane (z = table_z in base coords)
  - per frame, project the cloud through the wrist camera using
    T_wrist_to_base(t) = T_hand_to_base(q(t)) @ T_cam_to_hand
    where T_cam_to_hand is the NOMINAL mount transform.
  - composite the Franka mesh (rendered at the same wrist pose) on top.

Two-pass to avoid TF + pyrender EGL conflict.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CALIB_DIR = Path("/home/licho/data/droid/calibration")
DROID100 = Path("/home/licho/data/droid/droid_100/1.0.0")
ROOT = Path(__file__).parent / "outputs" / "context"
ROOT.mkdir(parents=True, exist_ok=True)

N_EPISODES = 5
RANDOM_SEED = 42
PLANE_Z_DEFAULT = 0.0  # table height in base frame (Franka standard: base mount at z=0)


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


def attach_label(frames: list[np.ndarray], text: str) -> list[np.ndarray]:
    lbl = label_strip(text, w=frames[0].shape[1])
    return [np.concatenate([lbl, f], axis=0) for f in frames]


def upscale(frames: list[np.ndarray], factor: int) -> list[np.ndarray]:
    return [
        np.array(Image.fromarray(f).resize((f.shape[1] * factor, f.shape[0] * factor), Image.LANCZOS))
        for f in frames
    ]


def blend(gt: np.ndarray, render: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    cov = (render.sum(-1) > 0).astype(np.float32)[..., None]
    return (gt.astype(np.float32) * (1 - alpha * cov) + render.astype(np.float32) * (alpha * cov)).clip(0, 255).astype(np.uint8)


def pick_episodes(builder, calib) -> list:
    from ltx_action_cond.droid import find_iou_calibrated_episodes
    ds = builder.as_dataset(split="train")
    cands = find_iou_calibrated_episodes(ds, calib, min_quality=0.70)
    # Filter: must have both ext cam intrinsics + wrist intrinsics + valid widths
    good = []
    for c in cands:
        sers = calib.camera_serials[c.episode_id]
        ext1 = sers.get("ext1_cam_serial")
        ext2 = sers.get("ext2_cam_serial")
        wrist = sers.get("wrist_cam_serial")
        intr = calib.intrinsics[c.episode_id]
        if not (ext1 and ext2 and wrist): continue
        if not all(s in intr and intr[s].get("width", 0) > 0 for s in (ext1, ext2, wrist)):
            continue
        good.append(c)
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(good)
    return good[:N_EPISODES]


def extract_pass() -> None:
    import tensorflow_datasets as tfds
    from ltx_action_cond.calibration import extrinsic_6dof_to_matrix, intrinsics_to_K, rescale_K
    from ltx_action_cond.droid import (
        CalibrationIndex, ROLE_TO_IMG_KEY, extract_episode_arrays,
        resolve_both_cams_cam2base, resolve_gripper_signal,
    )

    calib = CalibrationIndex.load(CALIB_DIR)
    builder = tfds.builder_from_directory(str(DROID100))
    picks = pick_episodes(builder, calib)
    print(f"Picked {len(picks)} episodes (seed={RANDOM_SEED}):")
    for c in picks:
        print(f"  IoU={c.quality:.3f}  T={c.n_steps:4d}  {c.episode_id}")

    wanted_idx = {c.ep_index: c for c in picks}
    ds = builder.as_dataset(split="train")
    cached: dict[int, object] = {}
    for i, ep in enumerate(ds):
        if i in wanted_idx:
            cached[i] = ep
        if len(cached) >= len(wanted_idx): break

    for cand in picks:
        ep = cached[cand.ep_index]
        ep_id = cand.episode_id
        sers = calib.camera_serials[ep_id]
        ext1_s, ext2_s, wrist_s = sers["ext1_cam_serial"], sers["ext2_cam_serial"], sers["wrist_cam_serial"]
        intr = calib.intrinsics[ep_id]

        res = resolve_both_cams_cam2base(calib, ep_id)
        if res is None:
            print(f"  skip {ep_id}: no cam2base resolution"); continue
        cam2base_1 = res["ext1_cam2base"]; cam2base_2 = res["ext2_cam2base"]

        arr_ext1 = extract_episode_arrays(ep, img_key=ROLE_TO_IMG_KEY["ext1_cam_serial"])
        arr_ext2 = extract_episode_arrays(ep, img_key=ROLE_TO_IMG_KEY["ext2_cam_serial"])
        # Wrist stream
        wrist_frames = []
        for step in ep["steps"]:
            wrist_frames.append(step["observation"]["wrist_image_left"].numpy())
        wrist_frames = np.stack(wrist_frames, axis=0)

        T, H, W = arr_ext1["frames"].shape[:3]
        def K_for(serial, sz):
            i = intr[serial]
            return rescale_K(intrinsics_to_K(i["cameraMatrix"]), (i["width"], i["height"]), sz)
        K1 = K_for(ext1_s, (W, H))
        K2 = K_for(ext2_s, (arr_ext2["frames"].shape[2], arr_ext2["frames"].shape[1]))
        K_wrist = K_for(wrist_s, (wrist_frames.shape[2], wrist_frames.shape[1]))

        gripper_for_render, gripper_src = resolve_gripper_signal(
            arr_ext1["cmd_gripper_position"], arr_ext1["gripper_position"])

        out_dir = ROOT / short_id(ep_id); out_dir.mkdir(exist_ok=True)
        np.savez(
            out_dir / "data.npz",
            cmd_joint_position=arr_ext1["cmd_joint_position"],
            gripper_for_render=gripper_for_render,
            ext1_frames=arr_ext1["frames"],
            ext2_frames=arr_ext2["frames"],
            wrist_frames=wrist_frames,
            K1=K1, K2=K2, K_wrist=K_wrist,
            cam2base_1=cam2base_1, cam2base_2=cam2base_2,
        )
        info = {
            "episode_id": ep_id, "institution": ep_id.split("+")[0],
            "n_steps": int(T), "ext1_source": res["ext1_source"],
            "ext2_source": res["ext2_source"], "gripper_source": gripper_src,
            "iou": float(cand.quality),
        }
        (out_dir / "info.json").write_text(json.dumps(info, indent=2))
        print(f"  extracted {ep_id}  T={T}")


WRIST_CALIB_DIR = Path(__file__).parent / "outputs" / "wrist_calib_vggt"


def _derive_T_cam_to_hand(d: dict, calib: dict) -> np.ndarray:
    """v1-style VGGT calibration with FK-anchored scale fix (mirrors wrist_video_from_wrist0.py)."""
    from ltx_action_cond.kinematics import fk_urdf
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


def _load_T_cam_to_hand(ep_short: str, d: dict) -> tuple[np.ndarray, str]:
    """Load the VGGT-calibrated wrist mount transform for this episode.

    Prefers `<ep>_ext2anchor.json` if present (better for TRI / some labs); otherwise
    falls back to `<ep>.json` and applies the FK-anchored scale fix in
    :func:`_derive_T_cam_to_hand`. **Never** falls back to the nominal mount.
    """
    ext2_path = WRIST_CALIB_DIR / f"{ep_short}_ext2anchor.json"
    default_path = WRIST_CALIB_DIR / f"{ep_short}.json"
    if ext2_path.exists():
        calib = json.loads(ext2_path.read_text())
        return np.array(calib["T_cam_to_hand"]), "ext2-anchored"
    if not default_path.exists():
        raise FileNotFoundError(
            f"VGGT wrist calibration missing for episode {ep_short}: "
            f"neither {ext2_path.name} nor {default_path.name} exists. "
            f"Run calibrate_wrist_vggt(_v2).py first."
        )
    calib = json.loads(default_path.read_text())
    return _derive_T_cam_to_hand(d, calib), "ext1-anchored"


def render_pass() -> None:
    import os; os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from ltx_action_cond.mesh_rendering import FrankaMeshRenderer
    from ltx_action_cond.kinematics import fk_urdf
    from ltx_action_cond.wrist_render import render_wrist_via_plane_homography

    for ep_dir in sorted(ROOT.iterdir()):
        if not (ep_dir / "data.npz").exists(): continue
        info = json.loads((ep_dir / "info.json").read_text())
        d = np.load(ep_dir / "data.npz")
        joints = d["cmd_joint_position"]
        # Gripper signal for the FIXED-camera Franka mesh: the resolved (obs-preferred,
        # rescaled-to-[0,1]) signal from `resolve_gripper_signal`. This is what tracks
        # the *actual* finger width in GT video — the rendered fingers open/close in
        # sync with what the GT shows them doing.
        grippers = d["gripper_for_render"].reshape(-1)
        ext1f = d["ext1_frames"]; ext2f = d["ext2_frames"]; wristf = d["wrist_frames"]
        K1 = d["K1"]; K2 = d["K2"]; K_wrist = d["K_wrist"]
        cam2base_1 = d["cam2base_1"]; cam2base_2 = d["cam2base_2"]
        T = ext1f.shape[0]
        H, W = ext1f.shape[1], ext1f.shape[2]
        Hw, Ww = wristf.shape[1], wristf.shape[2]
        assert (W, H) == (Ww, Hw), f"viewport mismatch ext={W}x{H} wrist={Ww}x{Hw}"
        print(f"=== {info['episode_id']}  T={T} ===")

        # === Wrist recipe (standard): anchor on GT wrist[0] + per-episode VGGT mount. ===
        # NEVER use NOMINAL_T_CAM_TO_HAND here; load `T_cam_to_hand` from
        # outputs/wrist_calib_vggt/<ep>.json (with FK-anchored scale fix).
        ep_short = ep_dir.name
        T_cam_to_hand, calib_src = _load_T_cam_to_hand(ep_short, d)
        T_hand_to_base_0 = fk_urdf(joints[0])["hand"]
        T_wrist0_to_base = T_hand_to_base_0 @ T_cam_to_hand
        wrist0_img = wristf[0]
        print(f"  wrist calib: {calib_src}")

        # Fixed cameras: photoreal Franka mesh + matching gripper.
        r = FrankaMeshRenderer(size=(W, H), mode="photo")
        ext1_renders, ext2_renders, wrist_renders = [], [], []
        for t in range(T):
            c_ext1, _ = r.render(joints[t], K1, cam2base_1, gripper_position=float(grippers[t]))
            ext1_renders.append(c_ext1)
            c_ext2, _ = r.render(joints[t], K2, cam2base_2, gripper_position=float(grippers[t]))
            ext2_renders.append(c_ext2)

            # Wrist: plane-homography from GT wrist[0] through the VGGT-calibrated
            # wrist pose at frame t. By construction t=0 matches GT wrist[0] exactly.
            # No Franka mesh in the wrist view — the camera is mounted on the gripper.
            T_wrist_t_to_base = fk_urdf(joints[t])["hand"] @ T_cam_to_hand
            wrist_renders.append(
                render_wrist_via_plane_homography(
                    wrist0_img, K_wrist, T_wrist0_to_base,
                    T_wrist_t_to_base, K_wrist, (W, H), plane_z=0.0,
                )
            )
        r.close()

        # Save per-camera headline videos at 3x upscale
        scale = 3
        def headline3(gt, render, label_tail):
            gtU = upscale(list(gt), scale); rU = upscale(render, scale)
            ovU = upscale([blend(gt[i], render[i]) for i in range(len(render))], scale)
            gtL = attach_label(gtU, f"GT | {label_tail}")
            rL = attach_label(rU, f"rendered | {label_tail}")
            ovL = attach_label(ovU, f"overlay | {label_tail}")
            sep = np.zeros((gtL[0].shape[0], 6, 3), dtype=np.uint8)
            return [np.concatenate([gtL[i], sep, rL[i], sep, ovL[i]], axis=1) for i in range(len(gtL))]

        def headline_wrist(gt, render, label_tail):
            gtU = upscale(list(gt), scale); rU = upscale(render, scale)
            gtL = attach_label(gtU, f"GT wrist | {label_tail}")
            rL = attach_label(rU, f"rendered wrist | {label_tail}")
            sep = np.zeros((gtL[0].shape[0], 6, 3), dtype=np.uint8)
            return [np.concatenate([gtL[i], sep, rL[i]], axis=1) for i in range(len(gtL))]

        ext1_video = headline3(ext1f, ext1_renders, f"ext1 | {info['institution']}")
        ext2_video = headline3(ext2f, ext2_renders, f"ext2 | {info['institution']}")
        wrist_video = headline_wrist(wristf, wrist_renders, f"wrist | {info['institution']}")

        imageio.mimsave(ep_dir / "ext1.mp4", ext1_video, fps=15, macro_block_size=1)
        imageio.mimsave(ep_dir / "ext2.mp4", ext2_video, fps=15, macro_block_size=1)
        imageio.mimsave(ep_dir / "wrist.mp4", wrist_video, fps=15, macro_block_size=1)

        # Stack all three with title strip; pad wrist row width to match ext rows
        max_w = max(ext1_video[0].shape[1], ext2_video[0].shape[1], wrist_video[0].shape[1])
        def pad_to(rows, target_w):
            out = []
            for r in rows:
                pad = target_w - r.shape[1]
                if pad > 0:
                    pad_arr = np.zeros((r.shape[0], pad, 3), dtype=np.uint8)
                    r = np.concatenate([r, pad_arr], axis=1)
                out.append(r)
            return out
        ext1_pad = pad_to(ext1_video, max_w)
        ext2_pad = pad_to(ext2_video, max_w)
        wrist_pad = pad_to(wrist_video, max_w)
        title = label_strip(f"{info['episode_id']} | {info['institution']} | IoU {info['iou']:.2f} | T={T}",
                             w=max_w, h=36, font_size=22)
        gap = np.full((6, max_w, 3), 80, dtype=np.uint8)
        stacked = [np.concatenate([title, ext1_pad[i], gap, ext2_pad[i], gap, wrist_pad[i]], axis=0)
                    for i in range(T)]
        imageio.mimsave(ep_dir / "headline_all.mp4", stacked, fps=15, macro_block_size=1)
        print(f"  wrote 4 videos ({len(ext1_video)} frames each)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["extract", "render", "both"], default="both")
    args = parser.parse_args()
    if args.mode == "both":
        subprocess.run([sys.executable, __file__, "--mode", "extract"], check=True)
        subprocess.run([sys.executable, __file__, "--mode", "render"], check=True)
        return
    (extract_pass if args.mode == "extract" else render_pass)()


if __name__ == "__main__":
    main()
