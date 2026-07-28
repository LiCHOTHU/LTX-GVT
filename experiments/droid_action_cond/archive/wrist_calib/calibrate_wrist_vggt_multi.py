"""Multi-frame VGGT wrist calibration (v2) — trajectory hand-eye solve.

The v1 calibration (calibrate_wrist_vggt.py) runs VGGT on ONE wide-baseline
(ext1_t0, wrist_t0) pair and anchors scale with a single-frame distance ratio.
That makes the whole episode's wrist context hostage to one hard pairwise pose
and one scale heuristic.

v2 instead runs VGGT ONCE on [ext1_t0, ext2_t0, wrist_t0, wrist_t1, ... wrist_tM]
(M wrist frames spanning the episode) and solves for the constant mount
transform X = T_cam_to_hand jointly against the FK hand trajectory:

  For each selected frame i (episode frame t_i):
      T_wrist_i_to_ext1   from VGGT (relative pose, up-to-scale translation)
      T_hand_i_to_base    from FK  (metric, essentially ground truth)
  Constraint:  cam2base_1 @ Scale(s) @ T_wrist_i_to_ext1 = T_hand_i_to_base @ X

  Rotation:    R_x_i = R_hand_i^T @ R_ext1 @ R_wrist_i  -> chordal mean (SVD),
               with outlier rejection on geodesic distance.
  Translation: R_ext1 @ t_wrist_i * s - R_hand_i @ p_x = p_hand_i - t_ext1
               -> one 3-row linear block per frame, unknowns (s, p_x) in R^4,
               least squares over all inlier frames, residual-based rejection.

Free cross-check: VGGT also poses ext2_t0, and we KNOW the metric ext1<->ext2
relative pose from the DROID calibration — its rotation error and implied scale
are reported per episode as a model-accuracy diagnostic (never used in the solve).

Output: outputs/wrist_calib_vggt_multi/<ep>.json with a DIRECT (metric)
T_cam_to_hand — no load-time scale fix needed — plus diagnostics.

Env:
  WRIST_DATA_ROOT     episode dirs with data.npz   (default: full-build scratch context)
  WRIST_CALIB_MULTI_OUT  output dir                (default: outputs/wrist_calib_vggt_multi)
  VGGT_DEVICE         cuda|cpu        (default cuda if available)
  VGGT_MULTI_NFRAMES  wrist frames fed to VGGT     (default 12)
  VGGT_MULTI_EPS      comma-separated episode dir names to process (default: all)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Same HF-cache fix as v1: a bogus HF_HOME (e.g. "/huggingface" from an
# unexpanded XDG var on compute nodes) must be replaced before any hub import.
_HF_FALLBACK = "/storage/project/r-agarg35-0/lwang831/hf_cache"
_hf = os.environ.get("HF_HOME", "")
if (not _hf) or _hf == "/" or _hf.startswith("/huggingface"):
    os.environ["HF_HOME"] = _HF_FALLBACK
for _k in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
    _v = os.environ.get(_k, "")
    if _v and (_v == "/" or _v.startswith("/huggingface")):
        os.environ.pop(_k, None)

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "packages/ltx-action-cond/src")

from ltx_action_cond.kinematics import fk_urdf

_HERE = Path(__file__).parent
WRIST_ROOT = Path(os.environ.get(
    "WRIST_DATA_ROOT",
    "/storage/scratch1/8/lwang831/gvt_dataset_full/scratch/context"))
OUT_DIR = Path(os.environ.get(
    "WRIST_CALIB_MULTI_OUT", _HERE / "outputs" / "wrist_calib_vggt_multi"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_WRIST_FRAMES = int(os.environ.get("VGGT_MULTI_NFRAMES", "12"))
# Optional dir of <ep>.npz files with MEASURED `joint_position` (extracted from
# RLDS observation.joint_position). When set and present, FK runs on measured
# joints instead of the commanded ones stored in data.npz — removing the
# command-tracking lag from the hand-eye solve.
JOINTS_OBS_DIR = os.environ.get("JOINTS_OBS_DIR", "")
ROT_OUTLIER_DEG = 5.0          # drop frames whose per-frame R_x deviates more than this
TRANS_OUTLIER_FACTOR = 2.5     # drop frames with residual > factor * median residual


def load_vggt():
    from vggt.models.vggt import VGGT
    device = os.environ.get(
        "VGGT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading VGGT on {device}...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    return model, device


def run_vggt_multi(model, device, frames: list[np.ndarray]):
    """Run VGGT on a list of HxWx3 uint8 frames. Returns (N,4,4) cam-to-world
    matrices in VGGT's internal (up-to-scale) world frame, anchored at image 0."""
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    tmpdir = OUT_DIR / "_tmp"
    tmpdir.mkdir(exist_ok=True)
    paths = []
    for i, f in enumerate(frames):
        p = tmpdir / f"f{i:03d}.png"
        Image.fromarray(f).save(p)
        paths.append(str(p))
    images = load_and_preprocess_images(paths).to(device)
    for p in paths:
        Path(p).unlink()

    is_cuda = device == "cuda"
    # sm80+ -> bf16 (as upstream); V100 (sm70) -> fp16 autocast; cpu -> fp32.
    if is_cuda and torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.bfloat16
    elif is_cuda:
        dtype = torch.float16
    else:
        dtype = None
    with torch.no_grad():
        if dtype is not None:
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                tokens, _ = model.aggregator(images[None])
        else:
            tokens, _ = model.aggregator(images[None])
        # Camera head in fp32 — pose decode is cheap and fp16-sensitive.
        pose_enc = model.camera_head(tokens)[-1].float()
        extrinsic, _ = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    E = extrinsic[0].detach().cpu().numpy()  # (N,3,4) world-to-cam [R|t], OpenCV
    c2w = []
    for rt in E:
        T = np.eye(4)
        T[:3, :4] = rt
        c2w.append(np.linalg.inv(T))
    return np.stack(c2w, 0)


def _rot_geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _chordal_mean(Rs: np.ndarray) -> np.ndarray:
    """SVD-projected mean of (N,3,3) rotations."""
    M = Rs.sum(axis=0)
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def pick_wrist_indices(T: int, n: int) -> list[int]:
    idx = np.unique(np.linspace(0, T - 1, n).round().astype(int))
    return idx.tolist()


def calibrate_episode(model, device, ep_dir: Path) -> dict | None:
    d = np.load(ep_dir / "data.npz")
    wristf = d["wrist_frames"]
    q = d["cmd_joint_position"]
    if JOINTS_OBS_DIR:
        obs_npz = Path(JOINTS_OBS_DIR) / f"{ep_dir.name}.npz"
        if obs_npz.exists():
            q_obs = np.load(obs_npz)["joint_position"]
            assert q_obs.shape == q.shape, f"obs joints shape {q_obs.shape} != cmd {q.shape}"
            q = q_obs
            print("  using MEASURED joints for FK")
    cam2base_1 = d["cam2base_1"]
    cam2base_2 = d["cam2base_2"]
    T_total = wristf.shape[0]

    sel = pick_wrist_indices(T_total, N_WRIST_FRAMES)
    frames = [d["ext1_frames"][0], d["ext2_frames"][0]] + [wristf[t] for t in sel]
    c2w = run_vggt_multi(model, device, frames)  # (2 + M, 4, 4)

    # Relative to ext1 (image 0). VGGT's world is image-0-anchored, but compose
    # generally rather than assuming c2w[0] == I.
    T_to_ext1 = np.einsum("ij,njk->nik", np.linalg.inv(c2w[0]), c2w)

    # ---- diagnostic: VGGT ext2 pose vs DROID-calibrated GT (never used in solve)
    T_ext2_gt = np.linalg.inv(cam2base_1) @ cam2base_2
    T_ext2_vggt = T_to_ext1[1]
    ext2_rot_err = _rot_geodesic_deg(T_ext2_vggt[:3, :3], T_ext2_gt[:3, :3])
    n_v = float(np.linalg.norm(T_ext2_vggt[:3, 3]))
    ext2_scale = float(np.linalg.norm(T_ext2_gt[:3, 3])) / n_v if n_v > 1e-6 else float("nan")

    # ---- per-frame quantities
    R1, t1 = cam2base_1[:3, :3], cam2base_1[:3, 3]
    wrist_rel = T_to_ext1[2:]                                   # (M,4,4)
    T_hand = np.stack([fk_urdf(q[t])["hand"] for t in sel], 0)  # (M,4,4) metric

    # ---- rotation: per-frame estimate of R_x, robust chordal mean
    Rx_i = np.einsum("nij,nik->njk",
                     T_hand[:, :3, :3],                          # transposed via einsum
                     np.einsum("ij,njk->nik", R1, wrist_rel[:, :3, :3]))
    inl = np.arange(len(sel))
    for _ in range(2):  # mean -> reject -> re-mean
        Rx = _chordal_mean(Rx_i[inl])
        devs = np.array([_rot_geodesic_deg(Rx_i[j], Rx) for j in range(len(sel))])
        new_inl = np.where(devs <= max(ROT_OUTLIER_DEG, 2.0 * np.median(devs[inl])))[0]
        if len(new_inl) < 3:
            break
        inl = new_inl
    rot_dev_inl = float(np.median([_rot_geodesic_deg(Rx_i[j], Rx) for j in inl]))

    # ---- translation + scale: linear LSQ over inlier frames
    def solve_s_px(idxs):
        A = np.zeros((3 * len(idxs), 4))
        b = np.zeros(3 * len(idxs))
        for r, j in enumerate(idxs):
            A[3 * r:3 * r + 3, 0] = R1 @ wrist_rel[j, :3, 3]
            A[3 * r:3 * r + 3, 1:4] = -T_hand[j, :3, :3]
            b[3 * r:3 * r + 3] = T_hand[j, :3, 3] - t1
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        res = (A @ x - b).reshape(-1, 3)
        return x[0], x[1:4], np.linalg.norm(res, axis=1)

    s, p_x, res = solve_s_px(inl)
    med = np.median(res)
    keep = inl[res <= max(TRANS_OUTLIER_FACTOR * med, 0.005)]   # floor 5 mm
    if len(keep) >= 3 and len(keep) < len(inl):
        s, p_x, res = solve_s_px(keep)
        inl = keep
    res_mm = 1000.0 * res

    T_cam_to_hand = np.eye(4)
    T_cam_to_hand[:3, :3] = Rx
    T_cam_to_hand[:3, 3] = p_x

    out = {
        "version": "multi_v2",
        "T_cam_to_hand": T_cam_to_hand.tolist(),
        "translation_cam_to_hand_m": p_x.tolist(),
        "vggt_world_scale": float(s),
        "frames_used": [int(sel[j]) for j in inl],
        "frames_selected": [int(t) for t in sel],
        "n_inliers": int(len(inl)),
        "rot_dev_median_deg": rot_dev_inl,
        "trans_residual_mm": {"median": float(np.median(res_mm)),
                              "max": float(res_mm.max())},
        "ext2_check": {"rot_err_deg": ext2_rot_err,
                       "implied_scale": ext2_scale,
                       "solved_scale": float(s)},
    }
    return out


def main():
    eps_filter = {s.strip() for s in os.environ.get("VGGT_MULTI_EPS", "").split(",") if s.strip()}
    model, device = load_vggt()
    results = {}
    for ep_dir in sorted(WRIST_ROOT.iterdir()):
        if not (ep_dir / "data.npz").exists():
            continue
        if eps_filter and ep_dir.name not in eps_filter:
            continue
        print(f"\n=== {ep_dir.name} ===")
        try:
            r = calibrate_episode(model, device, ep_dir)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")
            if device == "cuda":
                torch.cuda.empty_cache()
            continue
        print(f"  T_cam_to_hand t = {np.array(r['translation_cam_to_hand_m']).round(4)}"
              f"  (|t|={np.linalg.norm(r['translation_cam_to_hand_m']):.3f} m)")
        print(f"  scale={r['vggt_world_scale']:.4f}  inliers={r['n_inliers']}/{len(r['frames_selected'])}"
              f"  rot_dev={r['rot_dev_median_deg']:.2f}°"
              f"  trans_res={r['trans_residual_mm']['median']:.1f} mm (max {r['trans_residual_mm']['max']:.1f})")
        print(f"  ext2 check: rot_err={r['ext2_check']['rot_err_deg']:.2f}°"
              f"  implied_scale={r['ext2_check']['implied_scale']:.4f} vs solved {r['vggt_world_scale']:.4f}")
        (OUT_DIR / f"{ep_dir.name}.json").write_text(json.dumps(r, indent=2))
        results[ep_dir.name] = r
        if device == "cuda":
            torch.cuda.empty_cache()

    if results:
        Ts = np.array([np.array(v["T_cam_to_hand"]) for v in results.values()])
        t_med = np.median(Ts[:, :3, 3], axis=0)
        t_spread = Ts[:, :3, 3].max(0) - Ts[:, :3, 3].min(0)
        print(f"\nAcross {len(results)} episodes:")
        print(f"  median T_cam_to_hand translation: {t_med.round(4)}")
        print(f"  per-axis spread (max-min): {t_spread.round(4)}  (mm: {1000*np.abs(t_spread).max():.1f})")
        agg = {
            "translation_median_m": t_med.tolist(),
            "translation_spread_m": t_spread.tolist(),
            "per_episode": {k: {"t": v["translation_cam_to_hand_m"],
                                "ext2_rot_err_deg": v["ext2_check"]["rot_err_deg"]}
                            for k, v in results.items()},
        }
        (OUT_DIR / "_aggregate.json").write_text(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
