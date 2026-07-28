#!/usr/bin/env python
"""Build LIBERO per-chunk context data (GT + geometric-prior contexts) for one demo.

Computes the context **exactly the way the DROID pipeline does** (experiments/
droid_action_cond/chunk_context.py). LIBERO is a simulator, so the camera poses,
intrinsics, joint angles and gripper width are all exact — but the prior itself is
still built the DROID way, never by reading the simulator's ground-truth render:

  FIXED cameras (agentview, frontview) — DROID's "moving robot over a clean plate":
    1. once per chunk, render the FK-posed Franka *mesh* (pyrender, mode="photo")
       at the chunk-t0 joints and use it as a stencil to erase the robot from the
       GT t0 frame (dilate + TELEA inpaint) -> clean background plate;
    2. per frame, render the bare mesh at that frame's joints and paste its
       non-black pixels over the clean plate (`_composite_over_bg`).
    The robot is a rendered mesh posed by fk_urdf(q); the only GT pixels used are
    the t0 background (a conditioning input). This is byte-for-byte DROID's
    `render_chunk` logic, fed LIBERO's exact K / cam2base / joints / gripper.

  WRIST camera (robot0_eye_in_hand) — DROID's plane-homography kernel
    (`render_wrist_via_plane_homography`) warping the chunk-t0 wrist frame through
    the wrist-pose trajectory, plus a depth-reprojection variant for comparison.

pyrender (EGL) and mujoco (EGL) cannot share one process, so this runs in two
passes like chunk_context.py: `--mode capture` (robosuite -> intermediate npz) then
`--mode render` (pyrender + numpy -> outputs). Default `--mode both` subprocesses each.

Outputs under experiments/droid_action_cond/outputs/libero_context_chunks/<id>/:
  chunk_NN/data.npz   GT frames + per-camera context arrays + cameras + actions
  chunk_NN/info.json  metadata
  chunk_NN/headline.mp4  per view a [GT | prior1 | prior2] row, all views stacked
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_TIME = os.environ.get("GVT_TIME", "") not in ("", "0")
# Debug artifacts (headline.mp4, frontview output, wrist_context_single, clean_plate)
# are NOT consumed downstream (encode reads only agentview+wrist frames/context;
# build_libero_cameras reads only agentview+wrist K/cam2world). Default OFF for speed;
# set GVT_DEBUG=1 to restore them. frontview is still captured + used as a fuse source.
_DEBUG = os.environ.get("GVT_DEBUG", "") not in ("", "0")

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import logging as _logging

_ORIG_FH = _logging.FileHandler


class _RedirectFileHandler(_ORIG_FH):
    def __init__(self, filename, *a, **k):
        if str(filename) == "/tmp/robosuite.log":
            filename = os.path.join(os.environ.get("TMPDIR", os.path.expanduser("~")), "robosuite.log")
        super().__init__(filename, *a, **k)


_logging.FileHandler = _RedirectFileHandler

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # replay_libero
sys.path.insert(0, str(_HERE.parents[1] / "packages" / "ltx-action-cond" / "src"))
sys.path.insert(0, str(_HERE.parents[1] / "experiments" / "droid_action_cond"))  # chunk_context helpers

WRIST_CAM = "robot0_eye_in_hand"
DEFAULT_CAMS = ["agentview", "frontview", WRIST_CAM]
ROBOT_BASE_BODY = "robot0_link0"  # fk_urdf treats link0 as the identity base frame
ROBOT_INSTANCE_KEYS = ("Panda", "Gripper", "Mount")  # instance-name substrings that are "the robot"
GRIPPER_MAX_WIDTH = 0.08  # franka_hand fully-open finger separation (m); maps qpos -> gripper_position
SPLAT_RADIUS = 2  # disk radius (px) when splatting the fused cloud into the wrist view
CHUNK_LEN = int(os.environ.get("CHUNK_LEN", 33))
OUT_ROOT = _HERE.parents[1] / "experiments" / "droid_action_cond" / "outputs" / "libero_context_chunks"


# ----------------------------------------------------------------------------- chunking (DROID's exact convention)
from chunk_context import chunk_starts  # noqa: E402


# ----------------------------------------------------------------------------- geometry (wrist depth prior)
def depth_reproject(src_rgb, src_depth, K, C_src, C_tgt, out_hw):
    """Forward-warp the source frame into the target camera via exact depth + poses."""
    H, W = src_rgb.shape[:2]
    Ho, Wo = out_hw
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    d = src_depth.ravel().astype(np.float64)
    pts = np.stack([us.ravel(), vs.ravel(), np.ones(H * W)], 1).astype(np.float64)
    cam = (pts @ np.linalg.inv(K).T) * d[:, None]
    world = cam @ C_src[:3, :3].T + C_src[:3, 3]
    Cti = np.linalg.inv(C_tgt)
    pc = world @ Cti[:3, :3].T + Cti[:3, 3]
    z = pc[:, 2]
    px = pc @ K.T
    safe = np.where(z > 1e-4, z, 1.0)
    ut = np.round(px[:, 0] / safe).astype(int)
    vt = np.round(px[:, 1] / safe).astype(int)
    valid = (d > 1e-4) & (z > 1e-4) & (ut >= 0) & (ut < Wo) & (vt >= 0) & (vt < Ho)
    order = np.argsort(-z)
    sel = order[valid[order]]
    out = np.zeros((Ho * Wo, 3), np.uint8)
    out[vt[sel] * Wo + ut[sel]] = src_rgb.reshape(-1, 3)[sel]
    return out.reshape(Ho, Wo, 3)


def estimate_plane_z(depth, K, cam2world):
    """Dominant-surface world-z: median z of unprojected valid points (table plane prior)."""
    H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    pts = np.stack([us.ravel(), vs.ravel(), np.ones(H * W)], 1).astype(np.float64)
    cam = (pts @ np.linalg.inv(K).T) * depth.ravel()[:, None]
    world = cam @ cam2world[:3, :3].T + cam2world[:3, 3]
    zs = world[:, 2][depth.ravel() > 1e-4]
    return float(np.median(zs)) if zs.size else 0.0


def _unproject(rgb, depth, K, cam2world, robot_mask=None):
    """Lift one camera's RGB-D to a coloured world point cloud (scene only if mask given)."""
    H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    u, v, dd = us.ravel(), vs.ravel(), depth.ravel()
    cam = (np.stack([u, v, np.ones_like(u)], 1).astype(np.float64) @ np.linalg.inv(K).T) * dd[:, None]
    world = cam @ cam2world[:3, :3].T + cam2world[:3, 3]
    valid = dd > 1e-4
    if robot_mask is not None:
        valid &= robot_mask.ravel() == 0  # drop robot pixels -> scene-only cloud
    return world[valid], rgb.reshape(-1, 3)[valid].astype(np.uint8)


def fuse_scene_cloud(cap, cams, s):
    """Fuse every camera's chunk-t0 RGB-D into one robot-masked world point cloud."""
    from ltx_action_cond.wrist_render import SceneCloud
    pts, cols = [], []
    for c in cams:
        p, cc = _unproject(cap[f"{c}_frames"][s], cap[f"{c}_depth"][s], cap[f"{c}_K"],
                           cap[f"{c}_cam2world"][s], cap[f"{c}_robot_mask"][s])
        pts.append(p); cols.append(cc)
    P = np.concatenate(pts, 0); C = np.concatenate(cols, 0)
    z = np.zeros((len(P), 2), np.float32)
    return SceneCloud(points_base=P, colors=C, ext1_pixels=z, ext2_pixels=z, inlier_match_count=len(P))


# ----------------------------------------------------------------------------- pass 1: capture (robosuite only)
def _body_pose(sim, name):
    try:
        p = sim.data.get_body_xpos(name).copy()
        R = sim.data.get_body_xmat(name).reshape(3, 3).copy()
    except Exception:
        bid = [sim.model.body_id2name(i) for i in range(sim.model.nbody)].index(name)
        p = sim.data.body_xpos[bid].copy()
        R = sim.data.body_xmat[bid].reshape(3, 3).copy()
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = p
    return M


def _object_boxes(env, sim):
    """Resolve the BDDL goal objects -> (names, body_names, local_center, local_half).

    Each object's STATIC body-local axis-aligned box is the union AABB of its geoms
    (geom_pos +/- geom_size; geom orientation ignored -> a conservative over-cover,
    which is fine for a 2D pixel mask). The per-frame WORLD pose is read separately via
    _body_pose, so the 8 world corners at frame t = obj_pose[t] @ (center +/- half).
    Body resolution mirrors gen_libero_play.object_positions / bowl_radius."""
    names, bodies, centers, halves = [], [], [], []
    for n in getattr(env.env, "obj_of_interest", None) or []:
        bid = bname = None
        for cand in (f"{n}_main", n, f"{n}_1_main", f"{n}_g0"):
            try:
                bid, bname = sim.model.body_name2id(cand), cand
                break
            except Exception:
                continue
        if bid is None:
            continue
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for gi in range(sim.model.ngeom):
            if sim.model.geom_bodyid[gi] == bid:
                gp = np.asarray(sim.model.geom_pos[gi], np.float64)
                gs = np.asarray(sim.model.geom_size[gi], np.float64)
                lo, hi = np.minimum(lo, gp - gs), np.maximum(hi, gp + gs)
        if not np.isfinite(lo).all():  # body with no sized geoms -> small default box
            lo, hi = np.full(3, -0.02), np.full(3, 0.02)
        names.append(n)
        bodies.append(bname)
        centers.append((lo + hi) / 2.0)
        halves.append((hi - lo) / 2.0)
    return (names, bodies,
            np.asarray(centers, np.float64).reshape(-1, 3),
            np.asarray(halves, np.float64).reshape(-1, 3))


def capture(hdf5, demo_key, cams, res, out_npz):
    """Replay recorded states; save GT frames + camera geometry + joints/gripper/base.
    No pyrender here — this is the mujoco/robosuite half (separate process)."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
        get_real_depth_map,
    )
    from replay_libero import localize_model_xml

    wrist = WRIST_CAM if WRIST_CAM in cams else cams[-1]

    import h5py
    with h5py.File(hdf5, "r") as h:
        data = h["data"]
        lang = json.loads(data.attrs["problem_info"])["language_instruction"]
        recorded = data.attrs["bddl_file_name"]
        bddl = os.path.join(get_libero_path("bddl_files"), Path(recorded).parent.name, Path(recorded).name)
        grp = data[demo_key]
        states = grp["states"][:]
        actions = grp["actions"][:]
        model_xml = localize_model_xml(grp.attrs["model_file"])
        T = states.shape[0]

        _t = time.perf_counter() if _TIME else 0.0
        env = OffScreenRenderEnv(
            bddl_file_name=bddl, camera_names=cams,
            camera_heights=res, camera_widths=res, camera_depths=True,
            camera_segmentations="instance",
        )
        try:
            env.reset()
            # LIBERO version drift: a few objects were renamed with a "new_" prefix
            # (e.g. salad_dressing -> new_salad_dressing) AFTER the libero_90 demos
            # were recorded, so the recorded model_file still uses the OLD instance
            # names. The BDDL-built model references the CURRENT names, so the geom
            # id-mapping in reset_from_xml_string fails ("No geom new_salad_dressing_1_g0
            # ... available salad_dressing_1_g0"). Rename any legacy "<X>_<n>" instance
            # prefix in the recorded XML to its current "new_<X>_<n>" so they line up.
            for inst in env.env.model.instances_to_ids.keys():
                if inst.startswith("new_") and inst not in model_xml:
                    legacy = inst[len("new_"):]
                    if re.search(r"\b" + re.escape(legacy), model_xml):
                        model_xml = re.sub(r"\b" + re.escape(legacy), inst, model_xml)
            env.reset_from_xml_string(model_xml)
            sim = env.env.sim
            # Robot instance seg ids: seg value == index in instances_to_ids + 1.
            inst_names = list(env.env.model.instances_to_ids.keys())
            robot_ids = [i + 1 for i, n in enumerate(inst_names) if any(k in n for k in ROBOT_INSTANCE_KEYS)]
            assert robot_ids, f"no robot instance found among {inst_names}"
            # Object-centric AL signal: the BDDL goal objects' static local boxes +
            # per-frame world poses, so a downstream pass can project them into each
            # camera and mask the curiosity to object regions (not wrist look-around).
            obj_names, obj_bodies, obj_lc, obj_lh = _object_boxes(env, sim)
            obj_pose = []
            Ks = {c: get_camera_intrinsic_matrix(sim, c, res, res) for c in cams}
            frames = {c: [] for c in cams}
            depths = {c: [] for c in cams}
            masks = {c: [] for c in cams}
            cam2world = {c: [] for c in cams}
            joints, gripper, base2world = [], [], []
            if _TIME:
                _t_setup = time.perf_counter() - _t; _t = time.perf_counter()
            for t in range(T):
                obs = env.regenerate_obs_from_state(states[t])
                for c in cams:
                    # robosuite renders bottom-up; flip to image-natural top-down (matches the OpenCV extrinsic).
                    frames[c].append(obs[f"{c}_image"][::-1].copy())
                    depths[c].append(get_real_depth_map(sim, obs[f"{c}_depth"])[::-1, :, 0].copy())
                    seg = obs[f"{c}_segmentation_instance"][::-1, :, 0]
                    masks[c].append(np.isin(seg, robot_ids).astype(np.uint8))
                    cam2world[c].append(get_camera_extrinsic_matrix(sim, c).copy())
                joints.append(np.asarray(obs["robot0_joint_pos"][:7], np.float64))
                gq = np.abs(np.asarray(obs["robot0_gripper_qpos"], np.float64)).sum()
                gripper.append(float(np.clip(1.0 - gq / GRIPPER_MAX_WIDTH, 0.0, 1.0)))  # 0=open, 1=closed
                base2world.append(_body_pose(sim, ROBOT_BASE_BODY))
                obj_pose.append(np.stack([_body_pose(sim, b) for b in obj_bodies], 0)
                                if obj_bodies else np.zeros((0, 4, 4)))
            if _TIME:
                _t_loop = time.perf_counter() - _t
        finally:
            env.close()

    save = {
        "lang": np.array(lang), "bddl": np.array(bddl), "T": np.int64(T),
        "cams": np.array(cams), "wrist": np.array(wrist), "res": np.int64(res),
        "actions": actions, "joints": np.stack(joints, 0), "gripper": np.array(gripper),
        "base2world": np.stack(base2world, 0),
        "robot_instances": np.array([inst_names[i - 1] for i in robot_ids]),
        # object boxes: static local box + per-frame world pose (T, n_obj, 4, 4)
        "obj_names": np.array(obj_names), "obj_local_center": obj_lc, "obj_local_half": obj_lh,
        "obj_pose": np.stack(obj_pose, 0) if obj_pose else np.zeros((T, 0, 4, 4)),
    }
    for c in cams:
        save[f"{c}_frames"] = np.stack(frames[c], 0)
        save[f"{c}_depth"] = np.stack(depths[c], 0).astype(np.float32)
        save[f"{c}_robot_mask"] = np.stack(masks[c], 0)
        save[f"{c}_cam2world"] = np.stack(cam2world[c], 0)
        save[f"{c}_K"] = Ks[c]
    _t = time.perf_counter() if _TIME else 0.0
    # Intermediate is scratch (TMPDIR, deleted after render). Uncompressed save is
    # mmap-fast to read and skips zlib on both write and read; compression buys
    # nothing here. The render pass materializes it once (see np.load below).
    np.savez(out_npz, **save)
    if _TIME:
        _t_save = time.perf_counter() - _t
        print(f"[TIME-capture] T={T} cams={len(cams)} | env_setup={_t_setup:6.2f}s  "
              f"frame_loop={_t_loop:6.2f}s ({_t_loop/max(T,1)*1000:5.1f}ms/frame, "
              f"{_t_loop/max(T*len(cams),1)*1000:5.1f}ms/frame-cam)  npz_save={_t_save:6.2f}s")
    print(f"[capture] T={T} task='{lang}' robot={[inst_names[i-1] for i in robot_ids]} -> {out_npz}")


# ----------------------------------------------------------------------------- viz
def _label(img, text):
    import cv2
    # np.array(copy=True) so cv2's in-place draws never touch the caller's array.
    # `gt[t]` passed here is a view into the shared source frames; ascontiguousarray
    # alone does NOT copy an already-contiguous view, so drawing would corrupt it.
    img = np.array(img, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (img.shape[1], 16), (0, 0, 0), -1)
    cv2.putText(img, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def build_headline(triples_by_cam, cams, n, scale=2):
    import cv2
    out = []
    for t in range(n):
        rows = []
        for c in cams:
            gt, p1, p2, (l1, l2) = triples_by_cam[c]
            rows.append(np.concatenate([_label(gt[t], f"GT {c}"), _label(p1[t], l1), _label(p2[t], l2)], axis=1))
        f = np.concatenate(rows, axis=0)
        out.append(cv2.resize(f, (f.shape[1] * scale, f.shape[0] * scale), interpolation=cv2.INTER_NEAREST))
    return out


# ----------------------------------------------------------------------------- pass 2: render (pyrender + numpy)
def render(in_npz, out_dir, ep_id, fps):
    """Build the DROID-style context from the captured arrays. No mujoco here."""
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import imageio.v2 as imageio
    from ltx_action_cond.mesh_rendering import FrankaMeshRenderer
    from ltx_action_cond.wrist_render import render_wrist_scene_splat_vec
    from chunk_context import _clean_background_via_stencil, _composite_over_bg

    # Materialize every array ONCE. np.load returns a lazy NpzFile that does NOT
    # cache: each d[key] re-inflates the full array from zlib. fuse_scene_cloud and
    # the per-frame loops index d[...][s] many times per chunk, so without this the
    # whole intermediate gets decompressed ~100x (the old fuse bottleneck).
    d = dict(np.load(in_npz, allow_pickle=False))
    cams = [str(c) for c in d["cams"]]
    wrist = str(d["wrist"])
    res = int(d["res"])
    T = int(d["T"])
    fixed = [c for c in cams if c != wrist]
    out_hw = (res, res)

    starts = chunk_starts(T, CHUNK_LEN)
    print(f"[render] {ep_id}  T={T}  {len(starts)} x {CHUNK_LEN} chunks: {starts}")

    # One mesh renderer for all cams (LIBERO uses the Franka hand gripper).
    _t = time.perf_counter() if _TIME else 0.0
    renderer = FrankaMeshRenderer(size=(res, res), mode="photo", gripper="franka_hand")
    _acc = {"renderer_init": (time.perf_counter() - _t) if _TIME else 0.0,
            "fuse": 0.0, "wrist_splat": 0.0, "wrist_mesh": 0.0, "depth_reproject": 0.0,
            "fixed_mesh": 0.0, "save": 0.0}
    # Fixed-cam extrinsics are constant; precompute cam2base = inv(base2world) @ cam2world.
    base2world = d["base2world"][0]
    w2base = np.linalg.inv(base2world)
    cam2base = {c: w2base @ d[f"{c}_cam2world"][0] for c in fixed}
    joints = d["joints"]
    gripper = d["gripper"]

    try:
        for ci, s in enumerate(starts):
            n = CHUNK_LEN
            save = {"actions": d["actions"][s:s + n], "start": np.int64(s), "chunk_len": np.int64(n),
                    "joints": joints[s:s + n], "gripper": gripper[s:s + n]}
            # object boxes for the object-centric AL signal: slice poses to this chunk
            # ([s:s+n], identical to cam2world), carry the static local box through.
            if "obj_pose" in d and d["obj_pose"].size:
                save["obj_pose"] = d["obj_pose"][s:s + n]           # (n, n_obj, 4, 4)
                save["obj_names"] = d["obj_names"]
                save["obj_local_center"] = d["obj_local_center"]
                save["obj_local_half"] = d["obj_local_half"]
            triples = {}

            for c in cams:
                tag = c.replace(WRIST_CAM, "wrist")
                # frontview is consumed only as a fuse source (read from `d` inside
                # fuse_scene_cloud), never output downstream. Skip its render+save in
                # production; keep it under GVT_DEBUG for the headline comparison.
                is_used = (c == wrist) or (tag == "agentview")
                if not _DEBUG and not is_used:
                    continue
                K = d[f"{c}_K"]
                gt = d[f"{c}_frames"][s:s + n]
                save[f"{tag}_frames"] = gt
                save[f"{tag}_K"] = K
                save[f"{tag}_cam2world"] = d[f"{c}_cam2world"][s:s + n]

                if c == wrist:
                    # Fuse all cameras' chunk-t0 RGB-D (robot-masked) into one world cloud,
                    # splat it into the wrist camera per frame, then composite the FK gripper
                    # mesh on top (the moving part the static cloud can't supply). Multi-view
                    # fusion fills the disocclusion holes a single-view warp leaves.
                    _t = time.perf_counter() if _TIME else 0.0
                    scene = fuse_scene_cloud(d, cams, s)
                    if _TIME: _acc["fuse"] += time.perf_counter() - _t
                    src_rgb, src_depth, C_src = d[c + "_frames"][s], d[f"{wrist}_depth"][s], d[c + "_cam2world"][s]
                    fused, single = [], []
                    for t in range(n):
                        C_tgt = d[c + "_cam2world"][s + t]
                        _t = time.perf_counter() if _TIME else 0.0
                        splat, _ = render_wrist_scene_splat_vec(scene, C_tgt, K, out_hw, point_radius_px=SPLAT_RADIUS)
                        if _TIME: _acc["wrist_splat"] += time.perf_counter() - _t
                        c2b = w2base @ C_tgt
                        _t = time.perf_counter() if _TIME else 0.0
                        grip, _ = renderer.render(joints[s + t], K, c2b, gripper_position=float(gripper[s + t]), gripper_only=True)
                        if _TIME: _acc["wrist_mesh"] += time.perf_counter() - _t
                        fused.append(_composite_over_bg(grip, splat))
                        if _DEBUG:  # single-view warp: comparison-only, unused by encode
                            _t = time.perf_counter() if _TIME else 0.0
                            single.append(depth_reproject(src_rgb, src_depth, K, C_src, C_tgt, out_hw))
                            if _TIME: _acc["depth_reproject"] += time.perf_counter() - _t
                    fused = np.stack(fused, 0)
                    save[f"{tag}_context_fused"] = fused
                    if _DEBUG:
                        single = np.stack(single, 0)
                        save[f"{tag}_context_single"] = single
                        triples[c] = (gt, fused, single, ("3-cam fused", "wrist-only"))
                else:
                    # DROID fixed-cam: FK-mesh stencil -> clean plate -> composite per-frame mesh.
                    c2b = cam2base[c]
                    _t = time.perf_counter() if _TIME else 0.0
                    stencil, _ = renderer.render(joints[s], K, c2b, gripper_position=float(gripper[s]))
                    clean_bg = _clean_background_via_stencil(gt[0], stencil)
                    robot = []
                    for t in range(n):
                        bare, _ = renderer.render(joints[s + t], K, c2b, gripper_position=float(gripper[s + t]))
                        robot.append(_composite_over_bg(bare, clean_bg))
                    if _TIME: _acc["fixed_mesh"] += time.perf_counter() - _t
                    robot = np.stack(robot, 0)
                    save[f"{tag}_context_robot"] = robot
                    if _DEBUG:
                        save[f"{tag}_clean_plate"] = clean_bg
                        triples[c] = (gt, robot, np.repeat(clean_bg[None], n, 0), ("mesh render", "clean plate"))

            cdir = out_dir / f"chunk_{ci:02d}"
            cdir.mkdir(parents=True, exist_ok=True)
            _t = time.perf_counter() if _TIME else 0.0
            np.savez_compressed(cdir / "data.npz", **save)
            with open(cdir / "info.json", "w") as f:
                json.dump({
                    "ep_id": ep_id, "task": str(d["lang"]), "bddl": str(d["bddl"]),
                    "chunk_index": ci, "chunk_count": len(starts), "start": int(s),
                    "chunk_len": int(n), "T": int(T), "cams": cams, "res": res,
                    "method": "fixed: FK Franka mesh over inpainted plate (DROID); "
                              "wrist: 3-cam t0 RGB-D fused cloud splat + FK gripper mesh",
                    "contexts": {"fixed": ["mesh-render+clean-plate"],
                                 "wrist": ["fused (3-cam cloud + gripper mesh)", "single (wrist-only, comparison)"]},
                }, f, indent=2)
            if _DEBUG:  # headline.mp4 is a visual-debug strip, not consumed downstream
                imageio.mimsave(cdir / "headline.mp4", build_headline(triples, cams, n), fps=fps, macro_block_size=1)
            if _TIME: _acc["save"] += time.perf_counter() - _t
            print(f"  [chunk {ci:02d}] start={s} -> {cdir}")
    finally:
        renderer.close()
    if _TIME:
        nch = len(starts)
        print(f"[TIME-render] {nch} chunks x {CHUNK_LEN}fr, {len(cams)} cams ({len(fixed)} fixed + wrist) | "
              f"renderer_init={_acc['renderer_init']:6.2f}s")
        for k in ("fuse", "wrist_splat", "wrist_mesh", "depth_reproject", "fixed_mesh", "save"):
            print(f"               {k:16s}={_acc[k]:7.2f}s")
        print(f"               render_compute_total="
              f"{sum(_acc[k] for k in ('fuse','wrist_splat','wrist_mesh','depth_reproject','fixed_mesh')):7.2f}s "
              f"(+save {_acc['save']:.1f}s, +init {_acc['renderer_init']:.1f}s)")
    print(f"[done] {len(starts)} chunks -> {out_dir}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--demo", default="0")
    ap.add_argument("--cams", default=",".join(DEFAULT_CAMS))
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    ap.add_argument("--mode", choices=["capture", "render", "both"], default="both")
    ap.add_argument("--inter", default="", help="intermediate npz path (capture writes / render reads)")
    ap.add_argument("--ep_id", default="", help="explicit output id (default: <stem>_<demo_key>); "
                    "pass this to avoid the long-task-name collision when names exceed the old 80-char cap")
    args = ap.parse_args()

    cams = args.cams.split(",")
    demo_key = f"demo_{args.demo}"
    # NOTE: do NOT truncate — LIBERO task stems can be ~90 chars, so a [:80] cap
    # dropped the `_demo_N` suffix and collided all demos of a task into one dir.
    ep_id = args.ep_id or f"{Path(args.hdf5).stem}_{demo_key}"
    out_dir = Path(args.out_root) / ep_id
    tmp = os.environ.get("TMPDIR", "/tmp")
    inter = args.inter or os.path.join(tmp, f"libero_capture_{ep_id}.npz")

    if args.mode == "both":
        # pyrender (EGL) and mujoco (EGL) can't share a process -> subprocess each pass.
        base = [sys.executable, "-u", __file__, "--hdf5", args.hdf5, "--demo", args.demo,
                "--cams", args.cams, "--res", str(args.res), "--fps", str(args.fps),
                "--out_root", args.out_root, "--inter", inter, "--ep_id", ep_id]
        subprocess.run(base + ["--mode", "capture"], check=True)
        subprocess.run(base + ["--mode", "render"], check=True)
        return
    if args.mode == "capture":
        capture(args.hdf5, demo_key, cams, args.res, inter)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    render(inter, out_dir, ep_id, args.fps)


if __name__ == "__main__":
    main()
