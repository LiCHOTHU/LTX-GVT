#!/usr/bin/env python
"""Prototype: 3D (VGGT-depth) ext-camera fill for the wrist context vs the planar fill.

The production wrist context warps the raw wrist0 plate, then fills the regions
that plate never saw with the two FIXED cameras' plates — but through the same
flat z=0 plane homography, so all off-plane structure misprojects in the fill.

This prototype replaces ONLY the fill with true 3D reprojection:

  1. One VGGT pass per episode on [ext1_t0, ext2_t0, wrist x3] (the wrist frames
     give the model multi-view context; only the ext depth maps are consumed).
  2. Metric scale from the KNOWN ext1<->ext2 baseline:
         s = ||t_gt(e1,e2)|| / ||t_vggt(e1,e2)||
     (independent of the hand-eye solve; only VGGT *depth* is trusted, poses for
     unprojection are the calibrated DROID extrinsics).
  3. Unproject ext1/ext2 pixels at VGGT's internal resolution with the rescaled
     CALIBRATED intrinsics + GT cam2base -> one merged metric point cloud.
  4. Per target frame t: z-buffer splat the cloud into the wrist view at
     T_wrist_t = fk(q_t) @ T_cam_to_hand(multi), small-hole closing, then
     composite the (identical in both arms) raw wrist0 plane-warp plate on top.

  BEFORE arm = same wrist plate over plane-homography fill (production logic,
  raw t0 ext frames). AFTER arm = same wrist plate over the depth-splat fill.
  Both arms use the NEW multi hand-eye calibration, so the fill is the ONLY
  difference.

Outputs (same folder as the calibration A/B):
  outputs/wrist_compare_multi/<ep>__fill_before_after.mp4   GT | BEFORE | AFTER
  outputs/wrist_compare_multi/ALL_fill_before_after.mp4
  per-episode photometric table on the fill region (printed).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_HF_FALLBACK = "/storage/project/r-agarg35-0/lwang831/hf_cache"
_hf = os.environ.get("HF_HOME", "")
if (not _hf) or _hf == "/" or _hf.startswith("/huggingface"):
    os.environ["HF_HOME"] = _HF_FALLBACK
for _k in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
    _v = os.environ.get(_k, "")
    if _v and (_v == "/" or _v.startswith("/huggingface")):
        os.environ.pop(_k, None)

import cv2
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image, ImageDraw

sys.path.insert(0, "packages/ltx-action-cond/src")

from ltx_action_cond.kinematics import fk_urdf
from ltx_action_cond.wrist_render import render_wrist_via_plane_homography

_HERE = Path(__file__).parent
DATA_ROOT = Path(os.environ.get(
    "WRIST_DATA_ROOT", "/storage/scratch1/8/lwang831/gvt_dataset_full/scratch/context"))
MULTI_CALIB = Path(os.environ.get(
    "MULTI_CALIB_DIR", _HERE / "outputs" / "wrist_calib_vggt_multi"))
OUT = _HERE / "outputs" / "wrist_compare_multi"
OUT.mkdir(parents=True, exist_ok=True)

CHUNK = 33
UPSCALE = 2
FPS = 10
CONF_MIN = float(os.environ.get("DEPTH_CONF_MIN", "1.5"))
PANEL_TITLES = ("GT wrist", "BEFORE: plane fill", "AFTER: depth fill")


# ------------------------- VGGT depth -------------------------

def load_vggt():
    from vggt.models.vggt import VGGT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading VGGT on {device}...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    return model, device


def vggt_ext_depth(model, device, d) -> tuple[np.ndarray, ...]:
    """Run VGGT on [ext1_0, ext2_0, wrist x3]. Returns, for ext1 and ext2:
    rgb (h,w,3 uint8), metric depth (h,w), conf (h,w) — at VGGT internal res —
    plus the metric scale used."""
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    T_total = d["wrist_frames"].shape[0]
    sel = [0, T_total // 2, T_total - 1]
    frames = [d["ext1_frames"][0], d["ext2_frames"][0]] + [d["wrist_frames"][t] for t in sel]
    tmpdir = OUT / "_tmp"
    tmpdir.mkdir(exist_ok=True)
    paths = []
    for i, f in enumerate(frames):
        p = tmpdir / f"d{i:03d}.png"
        Image.fromarray(f).save(p)
        paths.append(str(p))
    images = load_and_preprocess_images(paths).to(device)
    for p in paths:
        Path(p).unlink()

    use_fp16 = device == "cuda" and torch.cuda.get_device_capability()[0] < 8
    amp_dtype = torch.float16 if use_fp16 else torch.bfloat16
    with torch.no_grad():
        if device == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                tokens, ps_idx = model.aggregator(images[None])
        else:
            tokens, ps_idx = model.aggregator(images[None])
        pose_enc = model.camera_head(tokens)[-1].float()
        extrinsic, _ = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        depth, depth_conf = model.depth_head(tokens, images=images[None], patch_start_idx=ps_idx)

    E = extrinsic[0].detach().cpu().numpy()        # (N,3,4) w2c
    D = depth[0, ..., 0].float().cpu().numpy()     # (N,h,w) vggt units
    C = depth_conf[0].float().cpu().numpy()        # (N,h,w)
    rgb = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

    # metric scale from the known ext1<->ext2 baseline
    def c2w(rt):
        T = np.eye(4); T[:3, :4] = rt
        return np.linalg.inv(T)
    T_e2_rel = np.linalg.inv(c2w(E[0])) @ c2w(E[1])
    base_vggt = float(np.linalg.norm(T_e2_rel[:3, 3]))
    base_gt = float(np.linalg.norm(
        (np.linalg.inv(d["cam2base_1"]) @ d["cam2base_2"])[:3, 3]))
    s = base_gt / base_vggt if base_vggt > 1e-6 else 1.0
    return rgb[0], s * D[0], C[0], rgb[1], s * D[1], C[1], s


def unproject(rgb, depth, conf, K_orig, cam2base, orig_wh) -> tuple[np.ndarray, np.ndarray]:
    """Pixels at internal res -> (P,3) base-frame points + (P,3) uint8 colors."""
    h, w = depth.shape
    W0, H0 = orig_wh
    K = np.diag([w / W0, h / H0, 1.0]) @ K_orig       # intrinsics at internal res
    v, u = np.mgrid[0:h, 0:w]
    keep = conf >= CONF_MIN
    rays = np.stack([u[keep], v[keep], np.ones(keep.sum())], -1) @ np.linalg.inv(K).T
    pts_cam = rays * depth[keep][:, None]
    pts = pts_cam @ cam2base[:3, :3].T + cam2base[:3, 3]
    return pts, rgb[keep]


def splat(pts, cols, T_cam_to_base, K, out_wh) -> np.ndarray:
    """Z-buffer point splat into a (H,W,3) uint8 image (far->near overwrite)."""
    W, H = out_wh
    w2c = np.linalg.inv(T_cam_to_base)
    pc = pts @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    ok = z > 0.05
    pc, z, cols_ = pc[ok], z[ok], cols[ok]
    uv = pc @ K.T
    u = np.round(uv[:, 0] / z).astype(int)
    v = np.round(uv[:, 1] / z).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z, cols_ = u[ok], v[ok], z[ok], cols_[ok]
    order = np.argsort(-z)                      # far first, near wins
    img = np.zeros((H, W, 3), np.uint8)
    img[v[order], u[order]] = cols_[order]
    # close 1-2 px splatting gaps (max-dilate only into empty pixels)
    for _ in range(2):
        mask = img.sum(-1) > 0
        dil = cv2.dilate(img, np.ones((3, 3), np.uint8))
        img = np.where(mask[..., None], img, dil)
    return img


# ------------------------- compositing -------------------------

def over(fg: np.ndarray, bg: np.ndarray) -> np.ndarray:
    m = (fg.sum(-1, keepdims=True) > 0)
    return np.where(m, fg, bg)


def label_strip(text, w, h=16):
    img = Image.new("RGB", (w, h), (25, 25, 25))
    ImageDraw.Draw(img).text((4, 2), text, fill=(240, 240, 240))
    return np.asarray(img)


def panel_header(w_panel, w_sep):
    sep = np.full((18, w_sep, 3), 60, np.uint8)
    parts = []
    for i, t in enumerate(PANEL_TITLES):
        if i:
            parts.append(sep)
        parts.append(label_strip(t, w_panel, h=18))
    return np.concatenate(parts, axis=1)


def build_episode(model, device, ep: str):
    npz = DATA_ROOT / ep / "data.npz"
    mp = MULTI_CALIB / f"{ep}.json"
    if not (npz.exists() and mp.exists()):
        print(f"skip {ep}: missing inputs")
        return None
    d = np.load(npz)
    X = np.array(json.loads(mp.read_text())["T_cam_to_hand"])
    q = d["cmd_joint_position"]
    wf = d["wrist_frames"]
    Kw, K1, K2 = d["K_wrist"], d["K1"], d["K2"]
    c2b1, c2b2 = d["cam2base_1"], d["cam2base_2"]
    T_total, H, W = wf.shape[:3]

    rgb1, dep1, conf1, rgb2, dep2, conf2, s = vggt_ext_depth(model, device, d)
    pts1, col1 = unproject(rgb1, dep1, conf1, K1, c2b1, (W, H))
    pts2, col2 = unproject(rgb2, dep2, conf2, K2, c2b2, (W, H))
    pts = np.concatenate([pts1, pts2], 0)
    cols = np.concatenate([col1, col2], 0)
    kept = len(pts) / (2 * dep1.size)
    print(f"  scale={s:.4f}  cloud={len(pts):,} pts ({100*kept:.0f}% kept @conf>={CONF_MIN})")

    starts = [0]
    mid = (T_total - CHUNK) // 2
    if mid > CHUNK // 2:
        starts.append(mid)

    sep = np.full((H, 4, 3), 60, np.uint8)
    header = panel_header(W, 4)
    rows = []
    err_b, err_a = [], []
    for t0 in starts:
        wrist0 = wf[t0]
        T_w0 = fk_urdf(q[t0])["hand"] @ X
        for t in range(t0, t0 + CHUNK):
            T_wt = fk_urdf(q[t])["hand"] @ X
            plate = render_wrist_via_plane_homography(
                wrist0, Kw, T_w0, T_wt, Kw, (W, H), plane_z=0.0)
            # BEFORE: planar fill from raw t0 ext frames (production logic)
            e1 = render_wrist_via_plane_homography(
                d["ext1_frames"][0], K1, c2b1, T_wt, Kw, (W, H), plane_z=0.0)
            e2 = render_wrist_via_plane_homography(
                d["ext2_frames"][0], K2, c2b2, T_wt, Kw, (W, H), plane_z=0.0)
            before = over(plate, over(e1, e2))
            # AFTER: depth-splat fill
            after = over(plate, splat(pts, cols, T_wt, Kw, (W, H)))
            # photometric error on the FILL region only (plate-invalid pixels)
            fill_m = (plate.sum(-1) == 0)
            gt = wf[t]
            for img, acc in ((before, err_b), (after, err_a)):
                m = fill_m & (img.sum(-1) > 0)
                if m.sum() > 100:
                    acc.append(np.abs(img[m].astype(float) - gt[m].astype(float)).mean())
            row = np.concatenate([gt, sep, before, sep, after], axis=1)
            strip = label_strip(f"{ep[:44]}   chunk@{t0}  t={t:3d}", row.shape[1])
            rows.append(np.concatenate([strip, header, row], axis=0))

    vid = np.repeat(np.repeat(np.stack(rows, 0), UPSCALE, 1), UPSCALE, 2)
    outp = OUT / f"{ep}__fill_before_after.mp4"
    imageio.mimsave(outp, vid, fps=FPS, macro_block_size=1)
    eb, ea = np.mean(err_b), np.mean(err_a)
    print(f"  fill-region err: BEFORE(plane)={eb:.2f}  AFTER(depth)={ea:.2f}"
          f"  ({100*(ea-eb)/eb:+.0f}%)")
    print(f"  wrote {outp}")
    return vid, eb, ea


def main():
    eps = [s.strip() for s in os.environ.get("COMPARE_EPS", "").split(",") if s.strip()]
    if not eps:
        eps = sorted(p.stem for p in MULTI_CALIB.glob("*.json") if not p.stem.startswith("_"))
    model, device = load_vggt()
    vids, table = [], []
    for ep in eps:
        print(f"\n=== {ep} ===")
        try:
            r = build_episode(model, device, ep)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")
            if device == "cuda":
                torch.cuda.empty_cache()
            continue
        if r is not None:
            vids.append(r[0])
            table.append((ep, r[1], r[2]))
        if device == "cuda":
            torch.cuda.empty_cache()

    if len(vids) > 1:
        minT = min(v.shape[0] for v in vids)
        minW = min(v.shape[2] for v in vids)
        combined = np.concatenate(
            [np.concatenate([v[:minT, :, :minW],
                             np.full((minT, 6, minW, 3), 60, np.uint8)], axis=1)
             for v in vids], axis=1)
        allp = OUT / "ALL_fill_before_after.mp4"
        imageio.mimsave(allp, combined, fps=FPS, macro_block_size=1)
        print(f"\nwrote {allp}")

    print(f"\n{'episode':44s} {'plane':>7s} {'depth':>7s}")
    for ep, eb, ea in table:
        print(f"{ep:44s} {eb:7.2f} {ea:7.2f}")


if __name__ == "__main__":
    main()
