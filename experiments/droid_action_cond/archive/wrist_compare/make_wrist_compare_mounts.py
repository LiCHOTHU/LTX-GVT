#!/usr/bin/env python
"""Race wrist-pose sources: GT | P1 ours (consensus+measured FK) | P2b DROID official.

P2b = per-frame wrist extrinsics shipped in raw DROID trajectory.h5
(observation/camera_extrinsics/<wrist>_left = pose(cartesian_position) @ gripper_offset,
scipy euler "xyz", verified exact). RLDS frames = raw[:-1] (verified).

Same protocol as make_wrist_compare_multi: raw wrist0 plane-warp (simple mode),
two 33-frame chunks (start + mid), photometric error + coverage per arm.

Outputs -> outputs/wrist_compare_multi/<ep>__mounts.mp4 (+ ALL__mounts.mp4)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, "packages/ltx-action-cond/src")
from ltx_action_cond.kinematics import fk_urdf
from ltx_action_cond.wrist_render import render_wrist_via_plane_homography

_HERE = Path(__file__).parent
DATA_ROOT = Path(os.environ.get(
    "WRIST_DATA_ROOT", "/storage/scratch1/8/lwang831/gvt_dataset_full/scratch/context"))
TMP = Path("/storage/project/r-agarg35-0/lwang831/tmp")
OBS = _HERE / "outputs" / "wrist_calib_vggt_multi" / "_obs"
CONS = np.load(_HERE / "outputs" / "wrist_calib_vggt_multi_obs" / "_consensus_T_cam_to_hand.npy")
META = json.load(open(TMP / "droid_test5_metadata_serials.json"))
OUT = _HERE / "outputs" / "wrist_compare_multi"

CHUNK = 33
UPSCALE = 2
FPS = 10
TITLES = ("GT wrist", "OURS: consensus + measured FK", "DROID official extrinsics")


def pose(v: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = v[:3]
    T[:3, :3] = R.from_euler("xyz", v[3:]).as_matrix()
    return T


def label_strip(text: str, w: int, h: int = 16) -> np.ndarray:
    img = Image.new("RGB", (w, h), (25, 25, 25))
    ImageDraw.Draw(img).text((4, 2), text, fill=(240, 240, 240))
    return np.asarray(img)


def panel_header(w_panel: int, w_sep: int) -> np.ndarray:
    sep = np.full((18, w_sep, 3), 60, np.uint8)
    parts = []
    for i, t in enumerate(TITLES):
        if i:
            parts.append(sep)
        parts.append(label_strip(t, w_panel, h=18))
    return np.concatenate(parts, axis=1)


def warp_chunk_T(d, T_all: np.ndarray, t0: int) -> np.ndarray:
    """Plane-warp wrist0 given per-frame cam-in-base poses T_all (T,4,4)."""
    wristf = d["wrist_frames"]
    K_w = d["K_wrist"]
    H, W = wristf.shape[1:3]
    out = []
    for t in range(t0, t0 + CHUNK):
        out.append(render_wrist_via_plane_homography(
            wristf[t0], K_w, T_all[t0], T_all[t], K_w, (W, H), plane_z=0.0))
    return np.stack(out, 0)


def photometric(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """Mean |err| on valid (nonblack) pred pixels + coverage fraction."""
    m = pred.sum(-1) > 0
    if m.sum() < 100:
        return float("nan"), 0.0
    err = np.abs(gt.astype(np.float32) - pred.astype(np.float32)).mean(-1)
    return float(err[m].mean()), float(m.mean())


def build_episode(ep: str):
    d = np.load(DATA_ROOT / ep / "data.npz")
    obs = np.load(OBS / f"{ep}.npz")
    q, cart = obs["joint_position"], obs["cartesian_position"]
    ws = META[ep]["wrist_cam_serial"]
    with h5py.File(TMP / f"droid_raw_traj/{ep}.h5") as f:
        off = pose(f[f"observation/camera_extrinsics/{ws}_left_gripper_offset"][0])

    T = d["wrist_frames"].shape[0]
    assert len(q) == T, (len(q), T)
    T_ours = np.stack([fk_urdf(q[t])["hand"] @ CONS for t in range(T)])
    T_droid = np.stack([pose(cart[t]) @ off for t in range(T)])

    starts = [0]
    mid = (T - CHUNK) // 2
    if mid > CHUNK // 2:
        starts.append(mid)

    H, W = d["wrist_frames"].shape[1:3]
    sep = np.full((H, 4, 3), 60, np.uint8)
    header = panel_header(W, 4)
    rows, stats = [], []
    for t0 in starts:
        gt = d["wrist_frames"][t0:t0 + CHUNK]
        ours = warp_chunk_T(d, T_ours, t0)
        droid = warp_chunk_T(d, T_droid, t0)
        e_o, c_o = photometric(gt, ours)
        e_d, c_d = photometric(gt, droid)
        stats.append((ep, t0, e_o, c_o, e_d, c_d))
        for i in range(CHUNK):
            row = np.concatenate([gt[i], sep, ours[i], sep, droid[i]], axis=1)
            strip = label_strip(f"{ep[:44]}   chunk@{t0}  t={t0+i:3d}", row.shape[1])
            rows.append(np.concatenate([strip, header, row], axis=0))
    vid = np.repeat(np.repeat(np.stack(rows, 0), UPSCALE, 1), UPSCALE, 2)
    outp = OUT / f"{ep}__mounts.mp4"
    imageio.mimsave(outp, vid, fps=FPS, macro_block_size=1)
    print(f"wrote {outp}")
    return vid, stats


def main():
    all_stats, vids = [], []
    for ep in sorted(META):
        vid, stats = build_episode(ep)
        vids.append(vid)
        all_stats += stats
    if len(vids) > 1:
        minT = min(v.shape[0] for v in vids)
        minW = min(v.shape[2] for v in vids)
        hsep = np.full((1, 6, minW, 3), 60, np.uint8)
        combined = np.concatenate(
            [np.concatenate([v[:minT, :, :minW], np.repeat(hsep, minT, 0)], axis=1)
             for v in vids], axis=1)
        imageio.mimsave(OUT / "ALL__mounts.mp4", combined, fps=FPS, macro_block_size=1)
        print("wrote ALL__mounts.mp4")
    print(f"\n{'episode':46s} {'chunk':>5s} {'OURS err':>9s} {'cov':>5s} {'DROID err':>10s} {'cov':>5s}")
    for ep, t0, e_o, c_o, e_d, c_d in all_stats:
        print(f"{ep[:46]:46s} {t0:5d} {e_o:9.2f} {c_o:5.2f} {e_d:10.2f} {c_d:5.2f}")


if __name__ == "__main__":
    main()
