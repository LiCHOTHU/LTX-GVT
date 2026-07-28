"""Per-chunk DROID context renderer (33-frame windows, GT-anchored).

Splits each episode into non-overlapping 33-frame chunks, with the last
chunk overlap-shifted to cover the tail so no frames are padded or dropped.
Each chunk's wrist context is anchored to that chunk's own ground-truth t=0
wrist frame (and chunk-t=0 FK pose), instead of the episode's global t=0.
This stops plane-homography drift from accumulating across long episodes.

Pipeline reuses what's already extracted by `wrist_demo.py`:
  inputs:  outputs/context/<ep>/data.npz           (GT frames, intrinsics, cam2base, joints, gripper)
           outputs/wrist_calib_vggt/<ep>.json      (v1 VGGT calibration with FK-anchored scale fix at load time)
  outputs: outputs/context_chunks/<ep>/chunk_<NN>/
              data.npz        (chunk-local slices + rendered ext1/ext2/wrist context as arrays)
              info.json
              ext1.mp4 ext2.mp4 wrist.mp4 headline_all.mp4

Pick the 3 episodes via env var CHUNK_EPS (comma-separated short names) or
take the default 3 (diverse-length subset of the seed-42 verified set).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess

import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages" / "ltx-action-cond" / "src"))

_HERE = Path(__file__).parent
SRC_ROOT = Path(os.environ.get("CHUNK_SRC_ROOT", _HERE / "outputs" / "context"))
CALIB_DIR = Path(os.environ.get("CHUNK_CALIB_DIR", _HERE / "outputs" / "wrist_calib_vggt"))
OUT_ROOT = Path(os.environ.get("CHUNK_OUT_ROOT", _HERE / "outputs" / "context_chunks"))
DROID_CALIB_DIR = Path(os.environ.get("DROID_CALIB_DIR",
                                       "/storage/project/r-agarg35-0/lwang831/droid/calibration"))
DROID_RLDS_DIR = Path(os.environ.get("DROID_RLDS_DIR",
                                      "/storage/project/r-agarg35-0/lwang831/droid/droid_100/1.0.0"))
# Raw-DROID trajectory.h5 mirror (download_droid_traj.py). Holds the OFFICIAL
# per-frame wrist camera extrinsics that the RLDS export dropped — preferred
# over any VGGT-estimated mount (raced on 5 eps 2026-06-06: ties/wins 8/10
# chunks, survives close-ups; see make_wrist_compare_mounts.py).
RAW_TRAJ_DIR = Path(os.environ.get("RAW_TRAJ_DIR",
                                    "/storage/project/r-agarg35-0/lwang831/droid/raw_traj"))

CHUNK_LEN = int(os.environ.get("CHUNK_LEN", 33))


def _short_id(ep_id: str) -> str:
    """Mirror wrist_demo.short_id so source dir names match."""
    return re.sub(r"[^A-Za-z0-9_+-]", "_", ep_id)[:40]

DEFAULT_EPS = [
    "IPRL+w026bb9b+2023-04-20-23h-40m-21s",       # T=89  → 3 chunks (overlap-tail)
    "CLVR+13759f6e+2023-06-05-23h-50m-15s",       # T=186 → 6 chunks
    "AUTOLab+cf2a60c6+2023-11-27-09h-44m-12s",    # T=661 → 21 chunks (long; biggest anchoring win)
    "AUTOLab+84bd5053+2023-07-14-16h-55m-45s",    # newly added; calibration pre-staged
    "TRI+52ca9b6a+2024-01-02-13h-33m-44s",        # newly added; calibration pre-staged
]


def chunk_starts(T: int, chunk: int) -> list[int]:
    """Non-overlapping windows of length `chunk`, with the final window
    overlap-shifted to T-chunk so no frames are padded or dropped.

    T=89, chunk=33  -> [0, 33, 56]    (last starts at T-chunk=56, overlaps prev by 10)
    T=99, chunk=33  -> [0, 33, 66]    (exact, no overlap)
    T=92, chunk=33  -> [0, 33, 59]    (last at 92-33=59)
    """
    if T < chunk:
        return []
    starts = list(range(0, T - chunk + 1, chunk))
    if starts[-1] + chunk < T:
        starts.append(T - chunk)
    return starts


def _derive_T_cam_to_hand_v1(d: dict, calib_v1: dict) -> np.ndarray:
    """Same FK-anchored scale fix as wrist_demo._derive_T_cam_to_hand. Lives here
    so chunk_context.py doesn't import private helpers from wrist_demo."""
    from ltx_action_cond.kinematics import fk_urdf
    T_wrist_to_ext = np.array(
        calib_v1.get("T_wrist_to_ext2", calib_v1.get("T_wrist_to_ext1"))
    )
    use_ext2 = "T_wrist_to_ext2" in calib_v1
    cam2base_anchor = d["cam2base_2" if use_ext2 else "cam2base_1"]
    T_hand_to_base_t0 = fk_urdf(d["cmd_joint_position"][0])["hand"]
    T_hand_to_anchor = np.linalg.inv(cam2base_anchor) @ T_hand_to_base_t0
    true_dist = float(np.linalg.norm(T_hand_to_anchor[:3, 3]))
    vggt_dist = float(np.linalg.norm(T_wrist_to_ext[:3, 3]))
    scale = true_dist / vggt_dist if vggt_dist > 1e-6 else 1.0
    T_w = T_wrist_to_ext.copy()
    T_w[:3, 3] *= scale
    T_wrist_to_base_t0 = cam2base_anchor @ T_w
    return np.linalg.inv(T_hand_to_base_t0) @ T_wrist_to_base_t0


def load_T_cam_to_hand(ep_short: str, d: dict) -> np.ndarray:
    """Load the per-episode VGGT wrist mount transform. Same precedence as wrist_demo."""
    ext2_path = CALIB_DIR / f"{ep_short}_ext2anchor.json"
    default_path = CALIB_DIR / f"{ep_short}.json"
    if ext2_path.exists():
        return np.array(json.loads(ext2_path.read_text())["T_cam_to_hand"])
    if not default_path.exists():
        raise FileNotFoundError(
            f"VGGT calibration missing for {ep_short}: expected {default_path}"
        )
    return _derive_T_cam_to_hand_v1(d, json.loads(default_path.read_text()))


def _pose_xyz(v: np.ndarray) -> np.ndarray:
    """DROID 6-vec (xyz position + scipy extrinsic-"xyz" euler) -> 4x4. Verified
    exact against raw trajectory.h5 composition (extrinsics = pose(cartesian_position)
    @ pose(gripper_offset), error 0.0000 deg/mm)."""
    from scipy.spatial.transform import Rotation as R
    T = np.eye(4)
    T[:3, 3] = v[:3]
    T[:3, :3] = R.from_euler("xyz", v[3:]).as_matrix()
    return T


def load_official_wrist_poses(ep_short: str, d: dict) -> np.ndarray | None:
    """Per-frame OFFICIAL wrist-camera-in-base poses (T,4,4) from the raw-DROID
    trajectory.h5 mirror. Returns None when unavailable so callers can fall back
    to the VGGT mount.

    Alignment: the raw export usually has 1 extra trailing frame vs RLDS (a few
    eps have 0 or 2). The start offset is resolved per episode by matching the
    commanded joint trajectory, which exists bit-near-identically in both exports.

    Mirrored mounts (~2.7% of the fleet; gripper_offset yaw > 0, cleanly bimodal
    vs the normal ~-90 deg): the camera is physically rotated 180 deg about its
    optical axis but the recorded video is auto-rotated upright, so the official
    extrinsics must be composed with Rz(180 deg) to describe the *image* frame.
    Without it the warp's apparent motion is exactly inverted (cos=-1.0 vs GT).
    """
    import h5py

    traj_path = RAW_TRAJ_DIR / ep_short / "trajectory.h5"
    meta_path = RAW_TRAJ_DIR / ep_short / "metadata.json"
    if not (traj_path.exists() and meta_path.exists()):
        return None
    wrist_serial = json.loads(meta_path.read_text())["wrist_cam_serial"]
    T = d["wrist_frames"].shape[0]
    with h5py.File(traj_path, "r") as f:
        key = f"observation/camera_extrinsics/{wrist_serial}_left"
        if key not in f:
            print(f"  WARN {ep_short}: {key} missing in trajectory.h5; falling back to VGGT")
            return None
        ext = f[key][:]
        off_key = f"{key}_gripper_offset"
        yaw_deg = float(np.degrees(f[off_key][0][5])) if off_key in f else -90.0
        raw_cmd = f["action/joint_position"][:]
    # Known mount conventions are cleanly bimodal at yaw ~ -90 (normal) and
    # ~ +90 (mirrored); anything else is an UNKNOWN convention — refuse rather
    # than warp with a possibly-wrong pose (fallback arm takes over).
    if not 70.0 < abs(yaw_deg) < 110.0:
        print(f"  WARN {ep_short}: unknown wrist mount convention "
              f"(gripper_offset yaw={yaw_deg:.1f} deg); falling back to VGGT")
        return None
    mirrored = yaw_deg > 0
    n_extra = len(raw_cmd) - T
    if n_extra < 0:
        print(f"  WARN {ep_short}: raw traj shorter than RLDS ({len(raw_cmd)} < {T}); falling back to VGGT")
        return None
    npz_cmd = d["cmd_joint_position"]
    errs = [float(np.abs(raw_cmd[s:s + T] - npz_cmd).max()) for s in range(n_extra + 1)]
    best_s = int(np.argmin(errs))
    if errs[best_s] > 1e-3:
        print(f"  WARN {ep_short}: official-traj cmd-joint alignment err={errs[best_s]:.2e}; falling back to VGGT")
        return None
    poses = np.stack([_pose_xyz(v) for v in ext[best_s:best_s + T]])
    if mirrored:
        print(f"  mirrored wrist mount: composing official poses with Rz(180)")
        rz180 = np.diag([-1.0, -1.0, 1.0, 1.0])
        poses = poses @ rz180
    return poses


# -------- augment pass: pull cmd_gripper_position out of RLDS and add it to source data.npz --------

def augment_pass(ep_list: list[str]) -> None:
    """Read each source data.npz; if it doesn't already have `cmd_gripper_position`,
    load the matching RLDS episode and grab the commanded gripper trajectory from
    `step.action_dict.gripper_position`. Save it back in-place.

    This is the *action* the policy issued (0=open, 1=closed) -- the right signal
    to drive the rendered fingers in the action-prior video. The previously-saved
    `gripper_for_render` was obs-preferred and so tracks the *physical* width,
    which can lag, plateau, or rescale relative to the command.

    Must run in a TF subprocess because pyrender/EGL doesn't coexist with TF
    in the same process.
    """
    import tensorflow_datasets as tfds  # noqa: E402
    from ltx_action_cond.droid import (  # noqa: E402
        CalibrationIndex,
        ROLE_TO_IMG_KEY,
        extract_episode_arrays,
        find_iou_calibrated_episodes,
    )

    needed = []
    for ep_short in ep_list:
        npz_path = SRC_ROOT / ep_short / "data.npz"
        if not npz_path.exists():
            print(f"  SKIP augment {ep_short}: no source npz at {npz_path}")
            continue
        existing = np.load(npz_path)
        if "cmd_gripper_position" in existing.files:
            print(f"  ALREADY augmented: {ep_short}")
            continue
        needed.append(ep_short)
    if not needed:
        print("  All sources already have cmd_gripper_position; skipping TF load.")
        return

    print(f"  Loading TFDS to augment {len(needed)} source(s): {needed}")
    calib = CalibrationIndex.load(DROID_CALIB_DIR)
    builder = tfds.builder_from_directory(str(DROID_RLDS_DIR))
    ds = builder.as_dataset(split="train")
    # Same bound as wrist_demo.pick_episodes: on the full 95K-episode DROID-1.0.1
    # an unbounded find_iou_calibrated_episodes scan reads >1 TB. Cap via env to
    # match whatever subset the universe was enumerated against (use the same
    # N as enumerate_universe.py's --max_episodes / WRIST_DEMO_MAX_SCAN).
    max_scan = int(os.environ.get("CHUNK_AUGMENT_MAX_SCAN",
                                  os.environ.get("WRIST_DEMO_MAX_SCAN", "0")))
    if max_scan > 0:
        ds = ds.take(max_scan)
        print(f"  [augment] CHUNK_AUGMENT_MAX_SCAN={max_scan}: limiting RLDS scan.")
    # count_steps=False: skip OOM-inducing step iteration (we don't use n_steps
    # in augment_pass; src/<ep>/data.npz already has cmd_joint_position).
    cands = find_iou_calibrated_episodes(ds, calib, min_quality=0.70, count_steps=False)
    by_short = {_short_id(c.episode_id): c for c in cands}

    picks = []
    for ep_short in needed:
        if ep_short not in by_short:
            print(f"  WARN: {ep_short} not in IoU-calibrated candidate set; skipping augment")
            continue
        picks.append(by_short[ep_short])
    if not picks:
        return

    # Stream-process (same OOM fix as wrist_demo.extract_pass): with 10K+ eps
    # caching all TFDS dicts blows 48 GB.
    wanted_idx = {c.ep_index: c for c in picks}
    remaining = set(wanted_idx)
    for i, ep in enumerate(ds):
        if not remaining:
            break
        if i not in wanted_idx:
            continue
        cand = wanted_idx[i]
        remaining.discard(i)
        arr = extract_episode_arrays(ep, img_key=ROLE_TO_IMG_KEY["ext1_cam_serial"])
        cmd_gripper = arr["cmd_gripper_position"]
        obs_gripper = arr["gripper_position"]
        ep_short = _short_id(cand.episode_id)
        src = SRC_ROOT / ep_short / "data.npz"
        d = dict(np.load(src))
        d["cmd_gripper_position"] = cmd_gripper
        d["obs_gripper_position"] = obs_gripper  # keep for diagnostics
        np.savez(src, **d)
        print(f"  augmented {ep_short}  (T={len(cmd_gripper)}, "
              f"cmd range [{cmd_gripper.min():.2f}, {cmd_gripper.max():.2f}], "
              f"obs range [{obs_gripper.min():.2f}, {obs_gripper.max():.2f}])")


# -------- video helpers (lifted from wrist_demo.py) --------

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
    return (gt.astype(np.float32) * (1 - alpha * cov)
            + render.astype(np.float32) * (alpha * cov)).clip(0, 255).astype(np.uint8)


# -------- core chunked renderer --------

def _clean_background_via_stencil(
    gt_frame_0: np.ndarray,
    robot_render_t0: np.ndarray,
    *,
    dilate_px: int = 7,
    inpaint_radius: int = 5,
) -> np.ndarray:
    """Erase the t=0 robot from `gt_frame_0` using `robot_render_t0` as a stencil.

    The robot mesh and the real Franka in the GT frame don't agree exactly
    (small FK/mesh mismatch, finger geometry, cabling), so we dilate the stencil
    mask by `dilate_px` to over-cover. The erased region is then filled via TELEA
    inpainting from neighbour pixels — typically just table/wall texture in DROID.

    Returns an HxWx3 uint8 RGB image with no robot present.
    """
    mask = (robot_render_t0.sum(-1) > 0).astype(np.uint8) * 255
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
    bgr = cv2.cvtColor(gt_frame_0, cv2.COLOR_RGB2BGR)
    clean_bgr = cv2.inpaint(bgr, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)


def _composite_over_bg(render: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Paint `render`'s non-black pixels over `bg`. Used to drop the rendered Franka
    onto the cleaned background plate, producing a single-robot frame."""
    mask = (render.sum(-1) > 0)[..., None]
    return np.where(mask, render, bg)


def _wrist_motion_direction_cos(gt: np.ndarray, ctx: np.ndarray) -> list[float]:
    """Pose-convention tripwire: per-frame cosines between GT and context
    apparent motion in the wrist view. Validates the OUTPUT, so it catches any
    pose convention error (unhandled mirrored mount, bad lab calib, frame
    misalignment) regardless of cause.

    Signature matters, not just the median: a true pose inversion is
    CONSISTENT (cos ~ -1 at every scoreable frame, magnitudes matching GT);
    off-plane clutter parallax gives MIXED signs with inflated magnitudes
    (verified on the AUTOLab+7cb094d5 mirrored mount vs WEIRD clutter chunk).

    Tracks LK features in the top 60% of the frame (away from the gripper)
    from frame 0 to every 4th frame; frames with < 2 px of GT motion carry no
    direction signal and are skipped. Empty list = nothing scoreable (static
    chunk / featureless context) — absence of evidence, not a pass.
    """
    def med_disp(a, b):
        ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        m = np.zeros_like(ga)
        m[: int(0.6 * ga.shape[0]), :] = 255
        pts = cv2.goodFeaturesToTrack(ga, 200, 0.01, 5, mask=m)
        if pts is None:
            return None
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(ga, gb, pts, None)
        good = st.reshape(-1) == 1
        if good.sum() < 10:
            return None
        return np.median((nxt - pts).reshape(-1, 2)[good], axis=0)

    cosines = []
    for t in range(4, gt.shape[0], 4):
        g = med_disp(gt[0], gt[t])
        c = med_disp(ctx[0], ctx[t])
        if g is None or c is None or np.linalg.norm(g) < 2.0 or np.linalg.norm(c) < 1.0:
            continue
        cosines.append(float(np.dot(g, c) / (np.linalg.norm(g) * np.linalg.norm(c))))
    return cosines


def render_chunk(
    *,
    out_dir: Path,
    info_base: dict,
    joints: np.ndarray,
    cmd_gripper: np.ndarray,
    obs_gripper: np.ndarray | None,
    ext1f: np.ndarray, ext2f: np.ndarray, wristf: np.ndarray,
    K1: np.ndarray, K2: np.ndarray, K_wrist: np.ndarray,
    cam2base_1: np.ndarray, cam2base_2: np.ndarray,
    T_cam_to_hand: np.ndarray | None,
    renderer,  # FrankaMeshRenderer; type omitted to keep top-level free of pyrender import
    T_wrist_cam_all: np.ndarray | None = None,  # (T,4,4) official per-frame wrist cam poses
) -> None:
    """Render all 33 frames of one chunk. All array inputs are already chunk-sliced.

    Per-frame ext1/ext2 context: rendered Franka composited over a once-per-chunk
    "clean background plate" built by stencil-erasing the t=0 robot from GT frame 0
    and TELEA-inpainting the hole. This produces a single-robot training sample
    (no t=0 ghost) so the model only sees the action-conditioned moving arm.

    Per-frame wrist context: plane-homography from chunk-local GT wrist frame 0
    through the VGGT-calibrated wrist pose at frame t.

    The rendered fingers follow `cmd_gripper` (the commanded action), NOT the
    observed gripper width.
    """
    from ltx_action_cond.kinematics import fk_urdf
    from ltx_action_cond.wrist_render import render_wrist_via_plane_homography

    T = ext1f.shape[0]
    H, W = ext1f.shape[1:3]

    # Chunk-local wrist anchor: GT wrist frame and wrist cam pose at chunk-t=0.
    # Preferred source: OFFICIAL per-frame extrinsics from raw trajectory.h5
    # (measured gripper pose ∘ lab hand-eye calibration). Fallback: FK(cmd
    # joints) ∘ VGGT-estimated mount (the legacy pipeline).
    wrist0_img = wristf[0]
    if T_wrist_cam_all is not None:
        wrist_pose_at = lambda t: T_wrist_cam_all[t]  # noqa: E731
    else:
        wrist_pose_at = lambda t: fk_urdf(joints[t])["hand"] @ T_cam_to_hand  # noqa: E731
    T_wrist0_to_base = wrist_pose_at(0)

    # Build the cleaned ext1/ext2 backgrounds ONCE per chunk using the t=0 stencil.
    g0 = float(cmd_gripper[0])
    stencil_ext1, _ = renderer.render(joints[0], K1, cam2base_1, gripper_position=g0)
    stencil_ext2, _ = renderer.render(joints[0], K2, cam2base_2, gripper_position=g0)
    clean_bg_ext1 = _clean_background_via_stencil(ext1f[0], stencil_ext1)
    clean_bg_ext2 = _clean_background_via_stencil(ext2f[0], stencil_ext2)

    # Wrist context has three modes (env WRIST_CONTEXT_MODE):
    #   "simple"        — single plane-homography on the raw wrist0_img (legacy).
    #                     Cheap but the gripper, which moves WITH the camera in
    #                     reality, gets warped like a world-frame object.
    #   "gripper_aware" — inpaint the t=0 gripper out of wrist0_img, warp only
    #                     the cleaned scene plate, then composite the gripper
    #                     back on top (paste real pixels when fingers haven't
    #                     moved, otherwise render the mesh).
    #   "multiview"     — gripper_aware, but the scene plate is fused from ALL
    #                     THREE calibrated views: warp the wrist-t0 plate AND the
    #                     two fixed-cam clean plates (ext1, ext2) into the frame-t
    #                     wrist view and stack them (priority wrist > ext1 > ext2).
    #                     The fixed cameras see the broader workspace, so they fill
    #                     the regions the wrist-t0 view never captured (no black
    #                     holes / stretched edges) instead of leaving gaps. Still a
    #                     planar warp, so off-plane geometry is approximate.
    #   "multiview_gated" — multiview, but the exterior fill is CONSENSUS-GATED:
    #                     a fill pixel is kept only where the ext1-warp and the
    #                     ext2-warp AGREE photometrically. Two independent views
    #                     warped through the same z=0 homography only coincide
    #                     where the content truly lies on the plane, so off-plane
    #                     ext content (objects with height, walls, inpaint smears)
    #                     is rejected instead of being painted in the wrong place.
    #                     Pixels seen by only one ext camera can't be verified and
    #                     stay black. Gate threshold: WRIST_FILL_TAU (mean |ext1-ext2|
    #                     per pixel, default 30).
    wrist_mode = os.environ.get("WRIST_CONTEXT_MODE", "multiview").lower()
    assert wrist_mode in ("simple", "gripper_aware", "multiview", "multiview_gated"), \
        f"WRIST_CONTEXT_MODE must be 'simple', 'gripper_aware', 'multiview' or 'multiview_gated', got {wrist_mode!r}"
    fill_tau = float(os.environ.get("WRIST_FILL_TAU", "30"))

    # The wrist context applies ZERO operation to the first wrist frame: we warp
    # the RAW wrist0 image directly as the camera moves (plane homography). No
    # gripper detection, no stencil, no inpaint/blur, no mesh composite. The
    # wrist camera is rigidly attached to the gripper, so the gripper in the raw
    # frame is already where it belongs — we just move the camera. The only
    # difference between the modes is gap-fill coverage:
    #   simple / gripper_aware : raw wrist0 warp only (uncovered regions stay black)
    #   multiview              : raw wrist0 warp, gaps filled by the two exterior plates

    ext1_renders, ext2_renders, wrist_renders = [], [], []
    for t in range(T):
        g = float(cmd_gripper[t])
        # Bare-mesh renders (Franka on black) — keep these so the data.npz preserves
        # the mask if a downstream consumer wants it; the composited "*_context"
        # arrays below are the actual training signal.
        c1_bare, _ = renderer.render(joints[t], K1, cam2base_1, gripper_position=g)
        c2_bare, _ = renderer.render(joints[t], K2, cam2base_2, gripper_position=g)
        ext1_renders.append(_composite_over_bg(c1_bare, clean_bg_ext1))
        ext2_renders.append(_composite_over_bg(c2_bare, clean_bg_ext2))

        T_wrist_t_to_base = wrist_pose_at(t)
        # Warp the RAW wrist0 frame (gripper and all) as the camera moves.
        bg_t = render_wrist_via_plane_homography(
            wrist0_img, K_wrist, T_wrist0_to_base,
            T_wrist_t_to_base, K_wrist, (W, H), plane_z=0.0,
        )
        if wrist_mode in ("multiview", "multiview_gated"):
            # Fill what the wrist-t0 view missed with the two fixed cameras'
            # clean plates (same table plane, different vantage). Priority
            # wrist > ext1 > ext2: keep raw wrist pixels where it saw the scene,
            # else ext1, else ext2. (Exterior fill only touches regions the
            # wrist plate never covered — the wrist frame itself is untouched.)
            e1_t = render_wrist_via_plane_homography(
                clean_bg_ext1, K1, cam2base_1,
                T_wrist_t_to_base, K_wrist, (W, H), plane_z=0.0,
            )
            e2_t = render_wrist_via_plane_homography(
                clean_bg_ext2, K2, cam2base_2,
                T_wrist_t_to_base, K_wrist, (W, H), plane_z=0.0,
            )
            if wrist_mode == "multiview_gated":
                # Keep fill only where BOTH ext warps see the pixel AND agree —
                # plane-induced-homography consistency: true only for content
                # actually on the z=0 plane. Off-plane content projects to
                # different wrist-view locations from the two vantages.
                v1m = e1_t.sum(-1) > 0
                v2m = e2_t.sum(-1) > 0
                diff = np.abs(e1_t.astype(np.float32) - e2_t.astype(np.float32)).mean(-1)
                agree = v1m & v2m & (diff < fill_tau)
                fill = np.where(agree[..., None], e1_t, 0).astype(e1_t.dtype)
                bg_t = _composite_over_bg(bg_t, fill)
            else:
                bg_t = _composite_over_bg(bg_t, _composite_over_bg(e1_t, e2_t))
        wrist_t = bg_t
        wrist_renders.append(wrist_t)

    ext1_renders = np.stack(ext1_renders, 0)
    ext2_renders = np.stack(ext2_renders, 0)
    wrist_renders = np.stack(wrist_renders, 0)

    # Validate the wrist warp's motion direction against GT (caught the
    # mirrored-mount inversion). A consistent cos ~ -1 across frames means the
    # context moves opposite to reality — worse than no context. Mixed signs
    # with a low median are off-plane clutter parallax (pose fine): recorded
    # for audit, no inversion alarm.
    cosines = _wrist_motion_direction_cos(wristf, wrist_renders)
    motion_cos = float(np.median(cosines)) if cosines else None
    info_base["wrist_motion_cos"] = motion_cos
    info_base["wrist_motion_cosines"] = [round(c, 3) for c in cosines]
    if cosines and motion_cos <= -0.8 and sum(c < 0 for c in cosines) >= 0.8 * len(cosines):
        print(f"  WARN {out_dir}: wrist context motion INVERTED vs GT "
              f"(median cos={motion_cos:.2f}, {sum(c < 0 for c in cosines)}/{len(cosines)} "
              f"frames negative) — check wrist pose convention")

    # Save chunk-local data (GT slices + composited context arrays + clean bg plates).
    save_kwargs = dict(
        cmd_joint_position=joints,
        cmd_gripper_position=cmd_gripper,
        ext1_frames=ext1f, ext2_frames=ext2f, wrist_frames=wristf,
        ext1_context=ext1_renders, ext2_context=ext2_renders, wrist_context=wrist_renders,
        ext1_clean_bg=clean_bg_ext1, ext2_clean_bg=clean_bg_ext2,
        K1=K1, K2=K2, K_wrist=K_wrist,
        cam2base_1=cam2base_1, cam2base_2=cam2base_2,
    )
    if T_cam_to_hand is not None:
        save_kwargs["T_cam_to_hand"] = T_cam_to_hand
    # Per-frame wrist camera poses in the base frame — ALWAYS saved, whichever
    # source produced them (official raw-traj extrinsics or FK∘VGGT fallback).
    # The PRoPE cameras stream (build_dataset_resumable.build_chunk_cameras)
    # needs the full wrist trajectory.
    save_kwargs["wrist_cam_poses"] = (
        T_wrist_cam_all if T_wrist_cam_all is not None
        else np.stack([wrist_pose_at(t) for t in range(T)])
    )
    if obs_gripper is not None:
        save_kwargs["obs_gripper_position"] = obs_gripper
    np.savez_compressed(out_dir / "data.npz", **save_kwargs)

    # Headline videos for visual verification.
    scale = 3
    def headline3(gt, render, label_tail):
        gtU = upscale(list(gt), scale); rU = upscale(list(render), scale)
        ovU = upscale([blend(gt[i], render[i]) for i in range(len(render))], scale)
        gtL = attach_label(gtU, f"GT | {label_tail}")
        rL = attach_label(rU, f"context | {label_tail}")
        ovL = attach_label(ovU, f"overlay | {label_tail}")
        sep = np.zeros((gtL[0].shape[0], 6, 3), dtype=np.uint8)
        return [np.concatenate([gtL[i], sep, rL[i], sep, ovL[i]], axis=1) for i in range(len(gtL))]

    def headline_wrist(gt, render, label_tail):
        gtU = upscale(list(gt), scale); rU = upscale(list(render), scale)
        gtL = attach_label(gtU, f"GT wrist | {label_tail}")
        rL = attach_label(rU, f"context wrist | {label_tail}")
        sep = np.zeros((gtL[0].shape[0], 6, 3), dtype=np.uint8)
        return [np.concatenate([gtL[i], sep, rL[i]], axis=1) for i in range(len(gtL))]

    inst = info_base["institution"]
    chunk_tag = f"{inst} ch{info_base['chunk_index']:02d}/{info_base['chunk_count']:02d} [{info_base['start']}:{info_base['start']+T}]"
    ext1_video = headline3(ext1f, ext1_renders, f"ext1 | {chunk_tag}")
    ext2_video = headline3(ext2f, ext2_renders, f"ext2 | {chunk_tag}")
    wrist_video = headline_wrist(wristf, wrist_renders, f"wrist | {chunk_tag}")

    imageio.mimsave(out_dir / "ext1.mp4", ext1_video, fps=15, macro_block_size=1)
    imageio.mimsave(out_dir / "ext2.mp4", ext2_video, fps=15, macro_block_size=1)
    imageio.mimsave(out_dir / "wrist.mp4", wrist_video, fps=15, macro_block_size=1)

    max_w = max(ext1_video[0].shape[1], ext2_video[0].shape[1], wrist_video[0].shape[1])
    def pad_to(rows, target_w):
        out = []
        for r in rows:
            pad = target_w - r.shape[1]
            if pad > 0:
                r = np.concatenate([r, np.zeros((r.shape[0], pad, 3), dtype=np.uint8)], axis=1)
            out.append(r)
        return out
    ext1_pad = pad_to(ext1_video, max_w)
    ext2_pad = pad_to(ext2_video, max_w)
    wrist_pad = pad_to(wrist_video, max_w)
    title = label_strip(
        f"{info_base['episode_id']} | {chunk_tag}",
        w=max_w, h=36, font_size=22,
    )
    gap = np.full((6, max_w, 3), 80, dtype=np.uint8)
    stacked = [np.concatenate([title, ext1_pad[i], gap, ext2_pad[i], gap, wrist_pad[i]], axis=0)
               for i in range(T)]
    imageio.mimsave(out_dir / "headline_all.mp4", stacked, fps=15, macro_block_size=1)

    (out_dir / "info.json").write_text(json.dumps(info_base, indent=2))


def render_pass(ep_list: list[str]) -> None:
    """Read each source episode, slice into 33-frame chunks (overlap-tail to cover
    the full length without padding), render the action-prior context per chunk
    using the CMD gripper signal, save chunk dirs."""
    # EGL/pyrender imports happen here (not at module top) so the augment pass
    # can run with TF in a different subprocess without an EGL/TF conflict.
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from ltx_action_cond.mesh_rendering import FrankaMeshRenderer

    print(f"Chunking {len(ep_list)} episodes into {CHUNK_LEN}-frame windows")
    print(f"  SRC_ROOT  = {SRC_ROOT}")
    print(f"  CALIB_DIR = {CALIB_DIR}")
    print(f"  OUT_ROOT  = {OUT_ROOT}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for ep_short in ep_list:
        src_dir = SRC_ROOT / ep_short
        npz_path = src_dir / "data.npz"
        info_path = src_dir / "info.json"
        if not npz_path.exists():
            print(f"\n  SKIP {ep_short}: no source data at {npz_path}")
            continue
        ep_info = json.loads(info_path.read_text())
        d = dict(np.load(npz_path))
        if "cmd_gripper_position" not in d:
            print(f"\n  SKIP {ep_short}: missing cmd_gripper_position; run --mode augment first")
            continue
        T = d["ext1_frames"].shape[0]
        starts = chunk_starts(T, CHUNK_LEN)
        if not starts:
            print(f"\n  SKIP {ep_short}: T={T} < chunk={CHUNK_LEN}")
            continue
        cmd_g = d["cmd_gripper_position"].reshape(-1)
        obs_g = d["obs_gripper_position"].reshape(-1) if "obs_gripper_position" in d else None
        print(f"\n=== {ep_short}  T={T}  -> {len(starts)} chunks at {starts} ===")
        print(f"  cmd_gripper range: [{cmd_g.min():.3f}, {cmd_g.max():.3f}]")

        # Wrist pose source: official raw-trajectory extrinsics first, VGGT fallback.
        T_wrist_all = load_official_wrist_poses(ep_short, d)
        T_cam_to_hand = None if T_wrist_all is not None else load_T_cam_to_hand(ep_short, d)
        wrist_pose_source = "official_raw_traj" if T_wrist_all is not None else "vggt_calib"
        print(f"  wrist pose source: {wrist_pose_source}")
        H, W = d["ext1_frames"].shape[1:3]
        renderer = FrankaMeshRenderer(
            size=(W, H), mode="photo",
            gripper=os.environ.get("CHUNK_GRIPPER", "robotiq_2f85"),
            robotiq_mount_yaw=float(os.environ.get("ROBOTIQ_MOUNT_YAW", 1.5 * np.pi)),
            robotiq_mount_z_offset=float(os.environ.get("ROBOTIQ_MOUNT_Z", 0.0)),
        )
        try:
            ep_out_root = OUT_ROOT / ep_short
            ep_out_root.mkdir(parents=True, exist_ok=True)
            only_idx_env = os.environ.get("CHUNK_INDEX", "").strip()
            only_idx = int(only_idx_env) if only_idx_env else None
            for ci, start in enumerate(starts):
                if only_idx is not None and ci != only_idx:
                    continue
                end = start + CHUNK_LEN
                chunk_dir = ep_out_root / f"chunk_{ci:02d}"
                chunk_dir.mkdir(exist_ok=True)
                info_base = {
                    "episode_id": ep_info["episode_id"],
                    "institution": ep_info["institution"],
                    "iou": ep_info.get("iou"),
                    "n_steps_episode": T,
                    "chunk_index": ci,
                    "chunk_count": len(starts),
                    "chunk_len": CHUNK_LEN,
                    "start": start,
                    "end": end,
                    "overlap_with_prev": max(0, starts[ci - 1] + CHUNK_LEN - start) if ci > 0 else 0,
                    "gripper_signal": "cmd_gripper_position (action)",
                    "wrist_pose_source": wrist_pose_source,
                }
                render_chunk(
                    out_dir=chunk_dir,
                    info_base=info_base,
                    joints=d["cmd_joint_position"][start:end],
                    cmd_gripper=cmd_g[start:end],
                    obs_gripper=obs_g[start:end] if obs_g is not None else None,
                    ext1f=d["ext1_frames"][start:end],
                    ext2f=d["ext2_frames"][start:end],
                    wristf=d["wrist_frames"][start:end],
                    K1=d["K1"], K2=d["K2"], K_wrist=d["K_wrist"],
                    cam2base_1=d["cam2base_1"], cam2base_2=d["cam2base_2"],
                    T_cam_to_hand=T_cam_to_hand,
                    renderer=renderer,
                    T_wrist_cam_all=T_wrist_all[start:end] if T_wrist_all is not None else None,
                )
                print(f"  chunk_{ci:02d}  [{start:4d}:{end:4d}]  "
                      f"overlap_prev={info_base['overlap_with_prev']:2d}  "
                      f"cmd_g[{cmd_g[start]:.2f}->{cmd_g[end - 1]:.2f}]")
        finally:
            renderer.close()

    print("\nDone.")


def _resolve_ep_list() -> list[str]:
    # Prefer CHUNK_EPS_FILE (newline-list) — bypasses MAX_ARG_STRLEN cap on
    # environment strings when the build orchestrator passes 10K+ short IDs.
    eps_file = os.environ.get("CHUNK_EPS_FILE", "").strip()
    if eps_file and Path(eps_file).exists():
        return [ln.strip() for ln in Path(eps_file).read_text().splitlines() if ln.strip()]
    wanted = os.environ.get("CHUNK_EPS", "").strip()
    if wanted:
        return [e.strip() for e in wanted.split(",") if e.strip()]
    return DEFAULT_EPS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["augment", "render", "both"], default="both")
    args = parser.parse_args()
    ep_list = _resolve_ep_list()

    if args.mode == "both":
        # Augment (TF) and render (EGL) cannot share one process; subprocess each.
        subprocess.run([sys.executable, __file__, "--mode", "augment"], check=True)
        subprocess.run([sys.executable, __file__, "--mode", "render"], check=True)
        return
    if args.mode == "augment":
        augment_pass(ep_list)
        return
    render_pass(ep_list)


if __name__ == "__main__":
    main()
