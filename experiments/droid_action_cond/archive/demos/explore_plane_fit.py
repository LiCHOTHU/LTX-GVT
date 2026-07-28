#!/usr/bin/env python
"""Tier-1 exploration: does fitting the table plane beat the hardcoded z=0 plane?

For each episode's chunk_00 we:
  1. SIFT-triangulate a sparse 3D cloud from the two fixed exterior cams at t=0
     (the only metric 3D we can get without depth).
  2. RANSAC-fit a plane to that cloud -> general plane (normal n, offset d).
  3. Report how far the fitted plane is from z=0 (height at workspace center +
     tilt angle). This tells us whether a plane-fit is worth wiring in.
  4. Render the wrist context two ways for visual comparison:
       (a) current hardcoded horizontal plane z=0
       (b) the RANSAC-fitted (possibly tilted) plane
     stacked GT | z=0 | fitted, per episode.

No library changes — the general-plane warp is written inline here so we can
evaluate before committing to a pipeline edit.
"""
import os
import sys
import numpy as np
import imageio.v2 as imageio

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "..", "..", "packages", "ltx-action-cond", "src"))
from ltx_action_cond.wrist_render import reconstruct_scene  # noqa: E402
from ltx_action_cond.kinematics import fk_urdf  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "outputs", "wrist_compare2")
OUT = os.path.join(ROOT, "plane_fit_explore")
os.makedirs(OUT, exist_ok=True)
CHUNK = "chunk_00"


def fit_plane_ransac(pts, n_iter=2000, thresh=0.01, seed_stride=7):
    """Fit n.X = d to a point cloud. Returns (n unit, d, inlier_mask).

    Horizontal-biased: we orient the normal so n.z > 0. thresh in metres.
    Deterministic sampling (no RNG) for reproducibility across resume.
    """
    N = pts.shape[0]
    if N < 10:
        return np.array([0.0, 0.0, 1.0]), 0.0, np.zeros(N, bool)
    best_inl = None
    best_cnt = -1
    idx = np.arange(N)
    for it in range(n_iter):
        # deterministic triplet
        a = (it * seed_stride) % N
        b = (it * seed_stride + 1 + it % 13) % N
        c = (it * seed_stride + 2 + (it * 3) % 29) % N
        if len({a, b, c}) < 3:
            continue
        p0, p1, p2 = pts[a], pts[b], pts[c]
        nrm = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm = nrm / ln
        if nrm[2] < 0:
            nrm = -nrm
        d = nrm @ p0
        dist = np.abs(pts @ nrm - d)
        inl = dist < thresh
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_inl = inl
    # refit on inliers (least squares plane)
    P = pts[best_inl]
    centroid = P.mean(0)
    U, S, Vt = np.linalg.svd(P - centroid)
    nrm = Vt[-1]
    if nrm[2] < 0:
        nrm = -nrm
    d = nrm @ centroid
    return nrm, float(d), best_inl


def warp_via_plane(src_img, K_src, cam2base_src, T_wrist_to_base, K_wrist,
                   out_size, n, d, bg=(0, 0, 0)):
    """Backwards-warp src_img into the wrist camera through a GENERAL plane n.X=d."""
    W, H = out_size
    He, We = src_img.shape[:2]
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    us = us.ravel().astype(np.float64); vs = vs.ravel().astype(np.float64)
    Kinv = np.linalg.inv(K_wrist)
    rays_cam = np.stack([us, vs, np.ones_like(us)], 1) @ Kinv.T
    Rw = T_wrist_to_base[:3, :3]; tw = T_wrist_to_base[:3, 3]
    rays_base = rays_cam @ Rw.T
    # intersect ray o + s*r with plane n.X = d  ->  s = (d - n.o)/(n.r)
    denom = rays_base @ n
    eps = 1e-9
    safe = np.where(np.abs(denom) > eps, denom, eps)
    s = (d - tw @ n) / safe
    in_front = (np.abs(denom) > eps) & (s > 0)
    P = tw[None] + s[:, None] * rays_base
    b2c = np.linalg.inv(cam2base_src)
    Pc = P @ b2c[:3, :3].T + b2c[:3, 3]
    front_src = Pc[:, 2] > 1e-3
    px = Pc @ K_src.T
    z = np.where(front_src, px[:, 2], 1.0)
    ue = px[:, 0] / z; ve = px[:, 1] / z
    inb = (ue >= 0) & (ue < We - 1) & (ve >= 0) & (ve < He - 1)
    mask = in_front & front_src & inb
    uc = np.clip(ue, 0, We - 1); vc = np.clip(ve, 0, He - 1)
    u0 = np.floor(uc).astype(int); v0 = np.floor(vc).astype(int)
    u1 = np.clip(u0 + 1, 0, We - 1); v1 = np.clip(v0 + 1, 0, He - 1)
    du = (uc - u0)[:, None]; dv = (vc - v0)[:, None]
    c00 = src_img[v0, u0].astype(np.float32); c01 = src_img[v0, u1].astype(np.float32)
    c10 = src_img[v1, u0].astype(np.float32); c11 = src_img[v1, u1].astype(np.float32)
    col = (c00*(1-du)*(1-dv)+c01*du*(1-dv)+c10*(1-du)*dv+c11*du*dv).clip(0, 255).astype(np.uint8)
    out = np.where(mask[:, None], col, np.array(bg, np.uint8)[None])
    return out.reshape(H, W, 3)


def upscale(img, k=3):
    return np.repeat(np.repeat(img, k, 0), k, 1)


def main():
    eps = sorted(d for d in os.listdir(os.path.join(ROOT, "multiview"))
                 if os.path.isdir(os.path.join(ROOT, "multiview", d)))
    print(f"{len(eps)} episodes\n")
    print(f"{'episode':40s} {'pts':>5} {'inl':>5} {'h@ctr(m)':>9} {'tilt(deg)':>9} {'rms(mm)':>8}")
    print("-" * 90)

    for ep in eps:
        d = np.load(os.path.join(ROOT, "multiview", ep, CHUNK, "data.npz"))
        ext1, ext2 = d["ext1_frames"][0], d["ext2_frames"][0]
        K1, K2, Kw = d["K1"], d["K2"], d["K_wrist"]
        c2b1, c2b2 = d["cam2base_1"], d["cam2base_2"]
        joints = d["cmd_joint_position"]; Tc2h = d["T_cam_to_hand"]

        scene = reconstruct_scene(ext1, K1, c2b1, ext2, K2, c2b2)
        npts = scene.points_base.shape[0]
        if npts < 10:
            print(f"{ep:40s} {npts:5d}   too few points -> skip")
            continue
        n, dd, inl = fit_plane_ransac(scene.points_base)
        # height of plane directly under workspace centre (x=y=0): z = (dd - n.x*0 - n.y*0)/n.z
        h_ctr = dd / n[2] if abs(n[2]) > 1e-6 else float("nan")
        tilt = np.degrees(np.arccos(np.clip(abs(n[2]), 0, 1)))
        resid = np.abs(scene.points_base[inl] @ n - dd)
        rms = float(np.sqrt((resid**2).mean()) * 1000) if inl.any() else float("nan")
        print(f"{ep:40s} {npts:5d} {int(inl.sum()):5d} {h_ctr:9.3f} {tilt:9.2f} {rms:8.1f}")

        # visual: render wrist context with z=0 vs fitted plane (use raw wrist0 source)
        wrist0 = d["wrist_frames"][0]
        T_hand0 = fk_urdf(joints[0])["hand"]; Tw0 = T_hand0 @ Tc2h
        H, W = wrist0.shape[:2]
        frames = []
        T = joints.shape[0]
        n_horiz = np.array([0.0, 0.0, 1.0]); d_horiz = 0.0
        for t in range(T):
            Twt = fk_urdf(joints[t])["hand"] @ Tc2h
            a = warp_via_plane(wrist0, Kw, Tw0, Twt, Kw, (W, H), n_horiz, d_horiz)
            b = warp_via_plane(wrist0, Kw, Tw0, Twt, Kw, (W, H), n, dd)
            gt = d["wrist_frames"][t]
            sep = np.zeros((H, 4, 3), np.uint8) + 60
            row = np.concatenate([gt, sep, a, sep, b], 1)
            frames.append(upscale(row))
        outp = os.path.join(OUT, f"{ep}.mp4")
        imageio.mimsave(outp, frames, fps=10)
        print(f"{'':40s} -> wrote {os.path.basename(outp)}  (GT | z=0 | fitted)")

    print("\nlegend: each mp4 is  GT wrist | z=0 plane warp | RANSAC-fitted plane warp")


if __name__ == "__main__":
    main()
