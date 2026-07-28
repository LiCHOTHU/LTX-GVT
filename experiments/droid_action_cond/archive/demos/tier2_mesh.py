#!/usr/bin/env python
"""Tier-2 (mesh): monocular-depth reprojection rendered as a RASTERIZED SURFACE.

Point-splatting (tier2_proto.py) gave speckle/holes. The fix is to render the
depth as a connected surface: triangulate each camera's depth grid into a
textured mesh (vertex colors = pixels), cull triangles that span depth
discontinuities (so we don't stretch skin across object edges), and rasterize
with pyrender (FLAT shading, proper z-buffer). Multiple cameras' meshes share
one scene so occlusion is resolved correctly.

Output per episode: GT | flat z=0 warp | mesh wrist-only | mesh fused (3 cams).

Usage: python tier2_mesh.py [ep_substr ...]
"""
import os
import sys
import numpy as np
import trimesh
import imageio.v2 as imageio

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import pyrender  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "..", "..", "packages", "ltx-action-cond", "src"))
from ltx_action_cond.kinematics import fk_urdf  # noqa: E402
from ltx_action_cond.wrist_render import render_wrist_via_plane_homography  # noqa: E402
from tier2_proto import get_depth, lift_cloud, anchor_to_table, up, lab, vsep  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "outputs", "wrist_compare2")
OUT = os.path.join(ROOT, "tier2_mesh")
os.makedirs(OUT, exist_ok=True)
CHUNK = "chunk_00"
_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def grid_faces(H, W):
    idx = np.arange(H * W).reshape(H, W)
    v00 = idx[:-1, :-1].ravel(); v01 = idx[:-1, 1:].ravel()
    v10 = idx[1:, :-1].ravel(); v11 = idx[1:, 1:].ravel()
    f1 = np.stack([v00, v01, v10], 1)
    f2 = np.stack([v01, v11, v10], 1)
    return np.concatenate([f1, f2], 0)


def build_mesh(rgb, K, cam2base, disc=0.06):
    """Lift depth -> base-frame textured mesh, culling discontinuity triangles."""
    depth = get_depth(rgb)
    P, cols = lift_cloud(rgb, depth, K, cam2base)
    P, s, zpl = anchor_to_table(P, cam2base[:3, 3])
    H, W = rgb.shape[:2]
    faces = grid_faces(H, W)
    # cull triangles whose longest 3D edge exceeds `disc` (object boundaries)
    e0 = np.linalg.norm(P[faces[:, 0]] - P[faces[:, 1]], axis=1)
    e1 = np.linalg.norm(P[faces[:, 1]] - P[faces[:, 2]], axis=1)
    e2 = np.linalg.norm(P[faces[:, 2]] - P[faces[:, 0]], axis=1)
    keep = np.maximum.reduce([e0, e1, e2]) < disc
    faces = faces[keep]
    tm = trimesh.Trimesh(vertices=P, faces=faces, vertex_colors=cols, process=False)
    return pyrender.Mesh.from_trimesh(tm, smooth=False), s, zpl


def render_meshes(renderer, meshes, K, T_cam_to_base, W, H):
    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[0, 0, 0])
    for m in meshes:
        scene.add(m)
    cam = pyrender.IntrinsicsCamera(K[0, 0], K[1, 1], K[0, 2], K[1, 2],
                                    znear=0.01, zfar=20.0)
    scene.add(cam, pose=T_cam_to_base @ _CV_TO_GL)
    # FLAT = unlit (vertex colors straight through); SKIP_CULL_FACES = the depth
    # grid is single-winding and may face away from the moved camera -> render both sides.
    flags = pyrender.RenderFlags.FLAT | pyrender.RenderFlags.SKIP_CULL_FACES
    color, _ = renderer.render(scene, flags=flags)
    return color


def main():
    want = sys.argv[1:]
    eps = sorted(d for d in os.listdir(os.path.join(ROOT, "multiview"))
                 if os.path.isdir(os.path.join(ROOT, "multiview", d)))
    if want:
        eps = [e for e in eps if any(w in e for w in want)]

    sample = np.load(os.path.join(ROOT, "multiview", eps[0], CHUNK, "data.npz"))
    H, W = sample["wrist_frames"].shape[1:3]
    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)

    for ep in eps:
        d = np.load(os.path.join(ROOT, "multiview", ep, CHUNK, "data.npz"))
        wrist0 = d["wrist_frames"][0]
        Kw, K1, K2 = d["K_wrist"], d["K1"], d["K2"]
        c2b1, c2b2 = d["cam2base_1"], d["cam2base_2"]
        cln1, cln2 = d["ext1_clean_bg"], d["ext2_clean_bg"]
        j, Tc2h = d["cmd_joint_position"], d["T_cam_to_hand"]
        T = j.shape[0]
        Tw0 = fk_urdf(j[0])["hand"] @ Tc2h

        mw, sw, _ = build_mesh(wrist0, Kw, Tw0)
        m1, s1, _ = build_mesh(cln1, K1, c2b1)
        m2, s2, _ = build_mesh(cln2, K2, c2b2)
        print(f"{ep[:30]:30s} scale w/e1/e2={sw:.2f}/{s1:.2f}/{s2:.2f}")

        frames = []
        for t in range(T):
            Twt = fk_urdf(j[t])["hand"] @ Tc2h
            gt = d["wrist_frames"][t]
            flat = render_wrist_via_plane_homography(
                wrist0, Kw, Tw0, Twt, Kw, (W, H), plane_z=0.0)
            mesh_w = render_meshes(renderer, [mw], Kw, Twt, W, H)
            mesh_f = render_meshes(renderer, [mw, m1, m2], Kw, Twt, W, H)
            row = np.concatenate([lab(gt, "GT"), vsep(H), lab(flat, "flat z=0"),
                                  vsep(H), lab(mesh_w, "mesh wrist"),
                                  vsep(H), lab(mesh_f, "mesh fused")], 1)
            frames.append(up(row))
        outp = os.path.join(OUT, f"{ep}.mp4")
        imageio.mimsave(outp, frames, fps=10)
        print(f"{'':30s} -> {os.path.basename(outp)}")
    renderer.delete()
    print(f"\nout: {OUT}\nlegend: GT | flat warp | mesh wrist-only | mesh fused")


if __name__ == "__main__":
    main()
