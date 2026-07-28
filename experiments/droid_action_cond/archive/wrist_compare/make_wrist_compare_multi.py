#!/usr/bin/env python
"""Compare wrist-context warps: GT | v1 (single-pair VGGT) | multi (trajectory hand-eye).

For each episode, renders the RAW wrist0 plane-warp (the production "zero-op on
frame 0" context, simple mode — no multiview fill, so calibration quality is the
ONLY thing that differs between the two right panels) through:

  v1    : T_cam_to_hand from the single (ext1_t0, wrist_t0) VGGT pair with the
          FK-anchored scale fix at load time (what the current pipeline uses).
  multi : T_cam_to_hand from calibrate_wrist_vggt_multi.py (trajectory solve).

Two 33-frame chunks per episode (start + mid-episode, each re-anchored on its
own wrist frame 0, exactly like chunk_context.render_chunk does), concatenated
in time. One mp4 per episode + a vertically stacked ALL.mp4.

Outputs -> experiments/droid_action_cond/outputs/wrist_compare_multi/
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw

sys.path.insert(0, "packages/ltx-action-cond/src")

from ltx_action_cond.kinematics import fk_urdf
from ltx_action_cond.wrist_render import render_wrist_via_plane_homography

_HERE = Path(__file__).parent
DATA_ROOT = Path(os.environ.get(
    "WRIST_DATA_ROOT", "/storage/scratch1/8/lwang831/gvt_dataset_full/scratch/context"))
V1_CALIB = Path(os.environ.get(
    "V1_CALIB_DIR", "/storage/scratch1/8/lwang831/gvt_dataset_full/scratch/wrist_calib_vggt"))
MULTI_CALIB = Path(os.environ.get(
    "MULTI_CALIB_DIR", _HERE / "outputs" / "wrist_calib_vggt_multi"))
OUT = _HERE / "outputs" / "wrist_compare_multi"
OUT.mkdir(parents=True, exist_ok=True)

CHUNK = 33
UPSCALE = 2
FPS = 10
# When set (dir of <ep>.npz with measured `joint_position`), the AFTER panel
# renders with measured joints; BEFORE stays on commanded joints (production).
JOINTS_OBS_DIR = os.environ.get("JOINTS_OBS_DIR", "")
OUT_SUFFIX = os.environ.get("OUT_SUFFIX", "__before_after")

# Reuse the v1 load-time derivation from chunk_context.py (same code path as
# the production build) without importing its module-level CALIB_DIR defaults.
_spec = importlib.util.spec_from_file_location("cc", _HERE / "chunk_context.py")
_cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cc)
derive_v1 = _cc._derive_T_cam_to_hand_v1


def label_strip(text: str, w: int, h: int = 16) -> np.ndarray:
    img = Image.new("RGB", (w, h), (25, 25, 25))
    ImageDraw.Draw(img).text((4, 2), text, fill=(240, 240, 240))
    return np.asarray(img)


PANEL_TITLES = (
    ("GT wrist", "BEFORE: v1 + cmd joints", "AFTER: multi + measured joints")
    if JOINTS_OBS_DIR else
    ("GT wrist", "BEFORE: v1 single-pair", "AFTER: multi hand-eye"))


def panel_header(w_panel: int, w_sep: int) -> np.ndarray:
    """One header row with a title strip over each of the three panels."""
    sep = np.full((18, w_sep, 3), 60, np.uint8)
    parts = []
    for i, t in enumerate(PANEL_TITLES):
        if i:
            parts.append(sep)
        parts.append(label_strip(t, w_panel, h=18))
    return np.concatenate(parts, axis=1)


def warp_chunk(d, X: np.ndarray, t0: int, q: np.ndarray | None = None) -> np.ndarray:
    """Raw wrist0 plane-warp for frames [t0, t0+CHUNK), anchored at t0."""
    wristf = d["wrist_frames"]
    if q is None:
        q = d["cmd_joint_position"]
    K_w = d["K_wrist"]
    H, W = wristf.shape[1:3]
    wrist0 = wristf[t0]
    T_w0 = fk_urdf(q[t0])["hand"] @ X
    out = []
    for t in range(t0, t0 + CHUNK):
        T_wt = fk_urdf(q[t])["hand"] @ X
        out.append(render_wrist_via_plane_homography(
            wrist0, K_w, T_w0, T_wt, K_w, (W, H), plane_z=0.0))
    return np.stack(out, 0)


def upscale(arr: np.ndarray, k: int) -> np.ndarray:
    return np.repeat(np.repeat(arr, k, axis=1), k, axis=2)


def build_episode(ep: str) -> np.ndarray | None:
    npz = DATA_ROOT / ep / "data.npz"
    v1p = V1_CALIB / f"{ep}.json"
    mp = MULTI_CALIB / f"{ep}.json"
    if not (npz.exists() and v1p.exists() and mp.exists()):
        print(f"skip {ep}: missing inputs")
        return None
    d = np.load(npz)
    T_total = d["wrist_frames"].shape[0]
    X_v1 = derive_v1(d, json.loads(v1p.read_text()))
    X_multi = np.array(json.loads(mp.read_text())["T_cam_to_hand"])
    q_after = None  # default: commanded joints in both arms
    if JOINTS_OBS_DIR:
        obs_npz = Path(JOINTS_OBS_DIR) / f"{ep}.npz"
        if obs_npz.exists():
            q_after = np.load(obs_npz)["joint_position"]

    starts = [0]
    mid = (T_total - CHUNK) // 2
    if mid > CHUNK // 2:
        starts.append(mid)

    H, W = d["wrist_frames"].shape[1:3]
    sep = np.full((H, 4, 3), 60, np.uint8)
    header = panel_header(W, 4)
    rows = []
    for t0 in starts:
        gt = d["wrist_frames"][t0:t0 + CHUNK]
        v1 = warp_chunk(d, X_v1, t0)
        mu = warp_chunk(d, X_multi, t0, q=q_after)
        for i in range(CHUNK):
            row = np.concatenate([gt[i], sep, v1[i], sep, mu[i]], axis=1)
            strip = label_strip(f"{ep[:44]}   chunk@{t0}  t={t0+i:3d}", row.shape[1])
            rows.append(np.concatenate([strip, header, row], axis=0))
    vid = upscale(np.stack(rows, 0), UPSCALE)
    outp = OUT / f"{ep}{OUT_SUFFIX}.mp4"
    imageio.mimsave(outp, vid, fps=FPS, macro_block_size=1)
    print(f"wrote {outp}  ({vid.shape})")
    return vid


def main():
    eps = [s.strip() for s in os.environ.get("COMPARE_EPS", "").split(",") if s.strip()]
    if not eps:
        eps = sorted(p.stem for p in MULTI_CALIB.glob("*.json") if not p.stem.startswith("_"))
    vids = [v for v in (build_episode(ep) for ep in eps) if v is not None]
    if len(vids) > 1:
        minT = min(v.shape[0] for v in vids)
        minW = min(v.shape[2] for v in vids)
        hsep = np.full((1, 6, minW, 3), 60, np.uint8)
        combined = np.concatenate(
            [np.concatenate([v[:minT, :, :minW], np.repeat(hsep, minT, 0)], axis=1)
             for v in vids], axis=1)
        allp = OUT / f"ALL{OUT_SUFFIX}.mp4"
        imageio.mimsave(allp, combined, fps=FPS, macro_block_size=1)
        print(f"wrote {allp}  ({combined.shape})")


if __name__ == "__main__":
    main()
