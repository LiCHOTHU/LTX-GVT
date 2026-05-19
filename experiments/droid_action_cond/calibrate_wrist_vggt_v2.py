"""Improved wrist hand-eye calibration using FK-anchored scale.

The old approach assumed |t_wrist_to_ext1| ≈ |t_hand_to_ext1| (wrist mount is
near hand → distances roughly equal). This breaks when the mount offset isn't
small compared to the hand-to-ext1 distance, giving non-physical T_cam_to_hand
values (e.g., TRI: 19 cm "behind" the hand origin).

Better: feed VGGT three images simultaneously -- ext1[0], wrist[0],
wrist[t_far] -- and use the FK-known hand motion as a direct scale anchor.

  motion_FK = |hand_pos(t_far) - hand_pos(0)|   (known exactly from FK)
  motion_VGGT = |t_wrist[t_far] - t_wrist[0]|   (in VGGT's world frame)
  scale = motion_FK / motion_VGGT

This is mount-independent: any wrist that moves rigidly with the hand has
exactly the hand's motion magnitude.

After calibration:
  T_wrist_to_ext1_scaled = T_wrist_to_ext1 with translation * scale
  T_wrist_to_base(0) = cam2base_ext1 @ T_wrist_to_ext1_scaled
  T_cam_to_hand = inv(T_hand_to_base(0)) @ T_wrist_to_base(0)

We also use t_far to verify: T_cam_to_hand computed from wrist[t_far] should
match the one from wrist[0] (it's a constant mount). The agreement is a
self-consistency check.
"""

from __future__ import annotations

import json, os, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "packages/ltx-action-cond/src")

from ltx_action_cond.kinematics import fk_urdf

_HERE = Path(__file__).parent
WRIST_ROOT = Path(os.environ.get("WRIST_DATA_ROOT", _HERE / "outputs" / "context"))
OUT_DIR = Path(os.environ.get("WRIST_CALIB_OUT", _HERE / "outputs" / "wrist_calib_vggt"))
OUT_DIR.mkdir(exist_ok=True)


def load_vggt():
    from vggt.models.vggt import VGGT
    device = os.environ.get("VGGT_DEVICE", "cpu")
    print(f"Loading VGGT on {device}...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    return model, device


def run_vggt_triple(model, device, images_rgb: list[np.ndarray]):
    """Run VGGT on N images, return list of (cam-to-world 4x4) matrices and intrinsics."""
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    tmpdir = OUT_DIR / "_tmp"; tmpdir.mkdir(exist_ok=True)
    paths = []
    for i, rgb in enumerate(images_rgb):
        p = tmpdir / f"img{i}.png"
        Image.fromarray(rgb).save(p)
        paths.append(str(p))
    images = load_and_preprocess_images(paths).to(device)
    for p in paths:
        Path(p).unlink()

    with torch.no_grad():
        tokens, ps_idx = model.aggregator(images[None])
        pose_enc = model.camera_head(tokens)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    E = extrinsic[0].detach().cpu().numpy()  # (N, 3, 4) world-to-cam
    K = intrinsic[0].detach().cpu().numpy()  # (N, 3, 3)

    def w2c_to_c2w(rt):
        T = np.eye(4); T[:3, :4] = rt
        return np.linalg.inv(T)
    Ts_c2w = [w2c_to_c2w(E[i]) for i in range(E.shape[0])]
    return Ts_c2w, K


def find_far_frame(joints: np.ndarray, min_motion_m: float = 0.10) -> int:
    """Pick a frame where the hand has moved by at least `min_motion_m` from t=0.
    Returns the smallest such t, or the frame with the largest motion if none reach the threshold."""
    T_hand_0 = fk_urdf(joints[0])["hand"]
    p0 = T_hand_0[:3, 3]
    dists = []
    for t in range(1, joints.shape[0]):
        T_hand_t = fk_urdf(joints[t])["hand"]
        dists.append(np.linalg.norm(T_hand_t[:3, 3] - p0))
    dists = np.array(dists)
    above = np.where(dists >= min_motion_m)[0]
    if len(above) > 0:
        return int(above[0]) + 1
    return int(np.argmax(dists)) + 1


def calibrate_episode(model, device, ep_dir: Path) -> dict | None:
    d = np.load(ep_dir / "data.npz")
    joints = d["cmd_joint_position"]
    if joints.shape[0] < 5:
        return None
    cam2base_ext1 = d["cam2base_1"]

    # Pick t_far
    t_far = find_far_frame(joints, min_motion_m=0.10)
    T_hand_0 = fk_urdf(joints[0])["hand"]
    T_hand_far = fk_urdf(joints[t_far])["hand"]
    motion_fk = float(np.linalg.norm(T_hand_far[:3, 3] - T_hand_0[:3, 3]))

    # Run VGGT on [ext1[0], wrist[0], wrist[t_far]]
    imgs = [d["ext1_frames"][0], d["wrist_frames"][0], d["wrist_frames"][t_far]]
    Ts_c2w, Ks_vggt = run_vggt_triple(model, device, imgs)
    T_ext1_c2w = Ts_c2w[0]
    T_wrist0_c2w = Ts_c2w[1]
    T_wrist_far_c2w = Ts_c2w[2]
    # Express wrist[0], wrist[t_far] in ext1[0] frame
    T_wrist0_to_ext1 = np.linalg.inv(T_ext1_c2w) @ T_wrist0_c2w
    T_wrist_far_to_ext1 = np.linalg.inv(T_ext1_c2w) @ T_wrist_far_c2w
    motion_vggt = float(np.linalg.norm(T_wrist_far_to_ext1[:3, 3] - T_wrist0_to_ext1[:3, 3]))

    if motion_vggt < 1e-3:
        print(f"  motion_vggt ~ 0, skipping")
        return None
    scale = motion_fk / motion_vggt

    # Apply scale to wrist[0] translation in ext1 frame, then derive T_cam_to_hand
    T_wrist0_to_ext1_scaled = T_wrist0_to_ext1.copy()
    T_wrist0_to_ext1_scaled[:3, 3] *= scale
    T_wrist0_to_base = cam2base_ext1 @ T_wrist0_to_ext1_scaled
    T_cam_to_hand = np.linalg.inv(T_hand_0) @ T_wrist0_to_base

    # Self-consistency: compute T_cam_to_hand from t_far too; should agree
    T_wrist_far_to_ext1_scaled = T_wrist_far_to_ext1.copy()
    T_wrist_far_to_ext1_scaled[:3, 3] *= scale
    T_wrist_far_to_base = cam2base_ext1 @ T_wrist_far_to_ext1_scaled
    T_cam_to_hand_far = np.linalg.inv(T_hand_far) @ T_wrist_far_to_base
    consistency_mm = float(1000 * np.linalg.norm(T_cam_to_hand[:3, 3] - T_cam_to_hand_far[:3, 3]))

    return {
        "T_cam_to_hand": T_cam_to_hand.tolist(),
        "T_cam_to_hand_from_t_far": T_cam_to_hand_far.tolist(),
        "self_consistency_translation_mm": consistency_mm,
        "scale": scale,
        "motion_fk_m": motion_fk,
        "motion_vggt_m": motion_vggt,
        "t_far": int(t_far),
        "translation_m": T_cam_to_hand[:3, 3].tolist(),
    }


def main():
    model, device = load_vggt()
    results = {}
    for ep_dir in sorted(WRIST_ROOT.iterdir()):
        if not (ep_dir / "data.npz").exists(): continue
        print(f"\n=== {ep_dir.name} ===")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            r = calibrate_episode(model, device, ep_dir)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")
            continue
        if r is None:
            print(f"  skipped")
            continue
        print(f"  t_far={r['t_far']}, FK motion={r['motion_fk_m']:.3f} m, "
              f"VGGT motion={r['motion_vggt_m']:.3f} m, scale={r['scale']:.3f}")
        print(f"  T_cam_to_hand translation (cm): {np.round(np.array(r['translation_m'])*100, 1).tolist()}")
        print(f"  self-consistency (t_far vs t=0): {r['self_consistency_translation_mm']:.1f} mm")
        out = OUT_DIR / f"{ep_dir.name}_v2.json"
        out.write_text(json.dumps(r, indent=2))
        results[ep_dir.name] = r

    if results:
        Ts = np.array([np.array(v["translation_m"]) for v in results.values()])
        print(f"\nAcross {len(results)} episodes (cm):")
        for name, v in results.items():
            t = np.array(v["translation_m"]) * 100
            mag = float(np.linalg.norm(t)) / 100  # back to m for printing
            print(f"  {name[:45]:45s}  t=[{t[0]:6.1f}, {t[1]:6.1f}, {t[2]:6.1f}]  |t|={mag*100:.1f} cm")


if __name__ == "__main__":
    main()
