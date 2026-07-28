#!/usr/bin/env python
"""Tier-2 prototype: monocular-depth wrist reprojection vs flat-plane warp.

Pipeline (per episode, chunk_00):
  1. Monocular metric depth on the RAW wrist t0 frame (Depth-Anything-V2-Metric).
  2. Back-project every pixel to a base-frame point cloud via K_wrist + the FK
     wrist pose at t0.
  3. Scale-anchor to the table: monocular metric scale is domain-mismatched, so
     we fit the dominant plane and rescale the cloud (radially from the camera
     origin, which FK fixes) so that plane sits at z=0. This ties absolute scale
     to FK geometry without touching the camera pose -> no GT leak.
  4. For each frame t, splat the cloud into the wrist camera at the FK pose
     (existing render_wrist_scene_splat_gpu, true z-buffered reprojection).

Output per episode: GT | flat z=0 warp | depth-splat   (so parallax handling is
directly comparable to Tier-1's flat warp).

Usage: python tier2_proto.py [ep_substr ...]   (default: all 5)
"""
import os
import sys
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "..", "..", "packages", "ltx-action-cond", "src"))
from ltx_action_cond.kinematics import fk_urdf  # noqa: E402
from ltx_action_cond.wrist_render import (  # noqa: E402
    SceneCloud, render_wrist_scene_splat_gpu, render_wrist_via_plane_homography,
)

ROOT = os.path.join(os.path.dirname(__file__), "outputs", "wrist_compare2")
OUT = os.path.join(ROOT, "tier2_proto")
os.makedirs(OUT, exist_ok=True)
CHUNK = "chunk_00"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

_PROC = _MODEL = None


def get_depth(rgb):
    """Monocular depth (meters, model scale) resized to rgb's HxW."""
    global _PROC, _MODEL
    if _MODEL is None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        _PROC = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
        _MODEL = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).to("cuda").eval()
    H, W = rgb.shape[:2]
    inp = _PROC(images=Image.fromarray(rgb), return_tensors="pt").to("cuda")
    with torch.no_grad():
        d = _MODEL(**inp).predicted_depth  # (1,h,w)
    d = torch.nn.functional.interpolate(d[:, None], size=(H, W),
                                        mode="bilinear", align_corners=False)
    return d[0, 0].cpu().numpy().astype(np.float64)


def lift_cloud(rgb, depth, K, cam2base, stride=1):
    """Back-project pixels to a base-frame coloured cloud (camera-z = depth)."""
    H, W = depth.shape
    us, vs = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))
    us = us.ravel(); vs = vs.ravel()
    z = depth[vs, us]
    rays = np.stack([us, vs, np.ones_like(us)], 1).astype(np.float64) @ np.linalg.inv(K).T
    Pc = rays * z[:, None]            # camera frame, z forward
    Pb = Pc @ cam2base[:3, :3].T + cam2base[:3, 3]
    cols = rgb[vs, us]
    return Pb, cols


def anchor_to_table(Pb, cam_origin, thresh=0.03):
    """Rescale cloud radially from cam_origin so the dominant plane -> z=0."""
    # RANSAC-lite dominant plane via repeated SVD on z-low band, then deterministic refit
    z = Pb[:, 2]
    # seed: take the lowest 60% in z as table candidates (robot/objects sit above)
    lo = z < np.percentile(z, 60)
    P = Pb[lo]
    cen = P.mean(0)
    _, _, Vt = np.linalg.svd(P - cen)
    n = Vt[-1]; n = n * np.sign(n[2] + 1e-9)
    d = n @ cen
    inl = np.abs(Pb @ n - d) < thresh
    z_plane = Pb[inl, 2].mean() if inl.any() else cen[2]
    oz = cam_origin[2]
    denom = oz - z_plane
    s = oz / denom if abs(denom) > 1e-6 else 1.0
    s = float(np.clip(s, 0.2, 5.0))     # guard against pathological scale
    Ps = cam_origin[None] + s * (Pb - cam_origin[None])
    return Ps, s, float(z_plane)


def up(img, k=3):
    return np.repeat(np.repeat(img, k, 0), k, 1)


def lab(img, t):
    o = img.copy(); o[:12] = (o[:12] * 0.35).astype(np.uint8); return o


def vsep(H, w=4):
    return np.zeros((H, w, 3), np.uint8) + 60


def main():
    want = sys.argv[1:]
    eps = sorted(d for d in os.listdir(os.path.join(ROOT, "multiview"))
                 if os.path.isdir(os.path.join(ROOT, "multiview", d)))
    if want:
        eps = [e for e in eps if any(w in e for w in want)]
    nz = np.array([0.0, 0.0, 1.0])
    for ep in eps:
        d = np.load(os.path.join(ROOT, "multiview", ep, CHUNK, "data.npz"))
        wrist0 = d["wrist_frames"][0]
        Kw = d["K_wrist"]; j = d["cmd_joint_position"]; Tc2h = d["T_cam_to_hand"]
        H, W = wrist0.shape[:2]; T = j.shape[0]
        Tw0 = fk_urdf(j[0])["hand"] @ Tc2h

        # --- build clouds: wrist (raw, gripper rides along) + 2 exterior (clean,
        #     robot inpainted out) -> fuse for hole-fill + surround awareness ---
        K1, K2 = d["K1"], d["K2"]
        c2b1, c2b2 = d["cam2base_1"], d["cam2base_2"]
        cln1, cln2 = d["ext1_clean_bg"], d["ext2_clean_bg"]

        def build(src, K, c2b):
            dep = get_depth(src)
            P, c = lift_cloud(src, dep, K, c2b)
            Ps, s, zpl = anchor_to_table(P, c2b[:3, 3])
            return Ps, c, s, zpl

        Pw, cw, sw, zw = build(wrist0, Kw, Tw0)        # wrist-only cloud
        P1, c1, s1, _ = build(cln1, K1, c2b1)
        P2, c2, s2, _ = build(cln2, K2, c2b2)
        # wrist-only scene
        scene_w = SceneCloud(points_base=Pw, colors=cw,
                             ext1_pixels=np.zeros((len(Pw), 2), np.float32),
                             ext2_pixels=np.zeros((len(Pw), 2), np.float32),
                             inlier_match_count=len(Pw))
        # fused scene (wrist + ext1 + ext2)
        Pf = np.concatenate([Pw, P1, P2], 0)
        cf = np.concatenate([cw, c1, c2], 0)
        scene_f = SceneCloud(points_base=Pf, colors=cf,
                             ext1_pixels=np.zeros((len(Pf), 2), np.float32),
                             ext2_pixels=np.zeros((len(Pf), 2), np.float32),
                             inlier_match_count=len(Pf))
        print(f"{ep[:30]:30s} scale w/e1/e2={sw:.2f}/{s1:.2f}/{s2:.2f} "
              f"pts w={len(Pw)} fused={len(Pf)}")

        frames = []
        cw_cache, cf_cache = {}, {}
        for t in range(T):
            Twt = fk_urdf(j[t])["hand"] @ Tc2h
            gt = d["wrist_frames"][t]
            flat = render_wrist_via_plane_homography(
                wrist0, Kw, Tw0, Twt, Kw, (W, H), plane_z=0.0)
            sp_w, _ = render_wrist_scene_splat_gpu(
                scene_w, Twt, Kw, (W, H), point_radius_px=2, _scene_cache=cw_cache)
            sp_f, _ = render_wrist_scene_splat_gpu(
                scene_f, Twt, Kw, (W, H), point_radius_px=2, _scene_cache=cf_cache)
            row = np.concatenate([lab(gt, "GT"), vsep(H), lab(flat, "flat z=0"),
                                  vsep(H), lab(sp_w, "depth wrist"),
                                  vsep(H), lab(sp_f, "depth fused")], 1)
            frames.append(up(row))
        outp = os.path.join(OUT, f"{ep}.mp4")
        imageio.mimsave(outp, frames, fps=10)
        print(f"{'':32s} -> {os.path.basename(outp)}")
    print(f"\nout: {OUT}\nlegend: GT | flat-plane warp | monocular-depth splat")


if __name__ == "__main__":
    main()
