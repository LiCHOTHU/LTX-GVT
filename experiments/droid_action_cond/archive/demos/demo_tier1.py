#!/usr/bin/env python
"""Tier-1 demo: evaluate plane-height sensitivity + seam feathering.

Since stereo plane-FIT is unrecoverable at 320x180 (see explore_plane_fit.py:
0 triangulated points on every episode), we instead probe Tier-1's two ideas
directly so the user can judge them:

  DEMO A  plane sweep   : GT | z=-0.10 | z=0.00 | z=+0.10  (raw wrist0 warp)
          -> if quality barely changes across heights, the error is PARALLAX
             (off-plane content), which no plane choice can fix. If one height
             clearly wins, a plane-fit is worth pursuing by some other means.

  DEMO B  seam feather  : GT | hard-priority multiview | feathered multiview
          -> tests whether blending the wrist patch into the fixed-camera fill
             removes the visible seam at the wrist-coverage boundary.

Writes per-episode mp4s under outputs/wrist_compare2/tier1_demo/.
"""
import os
import sys
import numpy as np
import cv2
import imageio.v2 as imageio

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "..", "..", "packages", "ltx-action-cond", "src"))
from ltx_action_cond.kinematics import fk_urdf  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "outputs", "wrist_compare2")
OUT = os.path.join(ROOT, "tier1_demo")
os.makedirs(OUT, exist_ok=True)
CHUNK = "chunk_00"


def warp_via_plane(src, K_src, c2b_src, Tw2b, K_w, out_size, n, d, bg=(0, 0, 0)):
    """Backwards-warp src into the wrist cam through plane n.X = d."""
    W, H = out_size
    He, We = src.shape[:2]
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    us = us.ravel().astype(np.float64); vs = vs.ravel().astype(np.float64)
    rays = np.stack([us, vs, np.ones_like(us)], 1) @ np.linalg.inv(K_w).T
    Rw, tw = Tw2b[:3, :3], Tw2b[:3, 3]
    rb = rays @ Rw.T
    denom = rb @ n
    safe = np.where(np.abs(denom) > 1e-9, denom, 1e-9)
    s = (d - tw @ n) / safe
    in_front = (np.abs(denom) > 1e-9) & (s > 0)
    P = tw[None] + s[:, None] * rb
    b2c = np.linalg.inv(c2b_src)
    Pc = P @ b2c[:3, :3].T + b2c[:3, 3]
    fsrc = Pc[:, 2] > 1e-3
    px = Pc @ K_src.T
    z = np.where(fsrc, px[:, 2], 1.0)
    ue, ve = px[:, 0] / z, px[:, 1] / z
    inb = (ue >= 0) & (ue < We - 1) & (ve >= 0) & (ve < He - 1)
    m = in_front & fsrc & inb
    uc = np.clip(ue, 0, We - 1); vc = np.clip(ve, 0, He - 1)
    u0 = np.floor(uc).astype(int); v0 = np.floor(vc).astype(int)
    u1 = np.clip(u0 + 1, 0, We - 1); v1 = np.clip(v0 + 1, 0, He - 1)
    du = (uc - u0)[:, None]; dv = (vc - v0)[:, None]
    c00 = src[v0, u0].astype(np.float32); c01 = src[v0, u1].astype(np.float32)
    c10 = src[v1, u0].astype(np.float32); c11 = src[v1, u1].astype(np.float32)
    col = (c00*(1-du)*(1-dv)+c01*du*(1-dv)+c10*(1-du)*dv+c11*du*dv)
    col = col.clip(0, 255).astype(np.uint8)
    out = np.where(m[:, None], col, np.array(bg, np.uint8)[None]).reshape(H, W, 3)
    return out


def hard_fill(wrist, e1, e2):
    """Priority wrist > e1 > e2 (hard mask)."""
    fill = np.where((e1.sum(-1) > 0)[..., None], e1, e2)
    return np.where((wrist.sum(-1) > 0)[..., None], wrist, fill)


def feathered_fill(wrist, e1, e2, feather_px=12):
    """Blend the wrist patch into the fixed-camera fill over a feather band."""
    fill = np.where((e1.sum(-1) > 0)[..., None], e1, e2).astype(np.float32)
    cov = (wrist.sum(-1) > 0).astype(np.uint8)
    # distance INTO the wrist-covered region from its boundary
    dist = cv2.distanceTransform(cov, cv2.DIST_L2, 5)
    alpha = np.clip(dist / max(feather_px, 1), 0, 1)[..., None]  # 1 deep inside, 0 at edge
    blended = alpha * wrist.astype(np.float32) + (1 - alpha) * fill
    # outside wrist coverage entirely -> pure fill
    out = np.where(cov[..., None] > 0, blended, fill)
    return out.clip(0, 255).astype(np.uint8)


def up(img, k=3):
    return np.repeat(np.repeat(img, k, 0), k, 1)


def lab(img, txt):
    o = img.copy(); o[:12] = (o[:12] * 0.35).astype(np.uint8); return o


def vsep(H, w=4):
    return np.zeros((H, w, 3), np.uint8) + 60


def main():
    eps = sorted(d for d in os.listdir(os.path.join(ROOT, "multiview"))
                 if os.path.isdir(os.path.join(ROOT, "multiview", d)))
    nz = np.array([0.0, 0.0, 1.0])
    for ep in eps:
        d = np.load(os.path.join(ROOT, "multiview", ep, CHUNK, "data.npz"))
        wrist0 = d["wrist_frames"][0]
        Kw, K1, K2 = d["K_wrist"], d["K1"], d["K2"]
        c2b1, c2b2 = d["cam2base_1"], d["cam2base_2"]
        cln1, cln2 = d["ext1_clean_bg"], d["ext2_clean_bg"]
        joints, Tc2h = d["cmd_joint_position"], d["T_cam_to_hand"]
        H, W = wrist0.shape[:2]; T = joints.shape[0]
        Tw0 = fk_urdf(joints[0])["hand"] @ Tc2h

        framesA, framesB = [], []
        for t in range(T):
            Twt = fk_urdf(joints[t])["hand"] @ Tc2h
            gt = d["wrist_frames"][t]
            # DEMO A: plane height sweep
            zlo = warp_via_plane(wrist0, Kw, Tw0, Twt, Kw, (W, H), nz, -0.10)
            z0 = warp_via_plane(wrist0, Kw, Tw0, Twt, Kw, (W, H), nz, 0.00)
            zhi = warp_via_plane(wrist0, Kw, Tw0, Twt, Kw, (W, H), nz, 0.10)
            rowA = np.concatenate([lab(gt, "GT"), vsep(H), lab(zlo, "z=-.10"),
                                   vsep(H), lab(z0, "z=0"), vsep(H),
                                   lab(zhi, "z=+.10")], 1)
            framesA.append(up(rowA))
            # DEMO B: seam feather (multiview fill)
            wv = warp_via_plane(wrist0, Kw, Tw0, Twt, Kw, (W, H), nz, 0.0)
            e1 = warp_via_plane(cln1, K1, c2b1, Twt, Kw, (W, H), nz, 0.0)
            e2 = warp_via_plane(cln2, K2, c2b2, Twt, Kw, (W, H), nz, 0.0)
            hard = hard_fill(wv, e1, e2)
            feat = feathered_fill(wv, e1, e2)
            rowB = np.concatenate([lab(gt, "GT"), vsep(H), lab(hard, "hard seam"),
                                   vsep(H), lab(feat, "feathered")], 1)
            framesB.append(up(rowB))

        pa = os.path.join(OUT, f"{ep}__A_planesweep.mp4")
        pb = os.path.join(OUT, f"{ep}__B_feather.mp4")
        imageio.mimsave(pa, framesA, fps=10)
        imageio.mimsave(pb, framesB, fps=10)
        print(f"{ep[:34]:34s}  wrote A(planesweep) + B(feather)")
    print(f"\nout: {OUT}")
    print("A = GT | z=-.10 | z=0 | z=+.10   (does plane height matter? or is it parallax?)")
    print("B = GT | hard-priority fill | feathered fill   (does blending kill the seam?)")


if __name__ == "__main__":
    main()
