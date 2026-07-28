"""Use VGGT to estimate the wrist camera pose from (ext1, wrist) image pairs,
then recover the constant `T_cam_to_hand` mount transform.

Algorithm:
  1. For each of our 5 episodes, run VGGT on the two images
     [ext1_t0 (PIL), wrist_t0 (PIL)] -> per-image (extrinsic 4x4, intrinsic 3x3).
     VGGT outputs camera-to-world matrices with image 0 (=ext1) at world origin.
  2. VGGT operates at its own internal resolution and outputs intrinsics
     RESCALED to that resolution. We don't need the intrinsics (we already
     have the calibrated DROID intrinsics) -- only the RELATIVE pose:
         T_wrist_to_ext1 = extrinsic[1]
  3. Compose with known ext1 cam2base to get wrist pose in base:
         T_wrist_to_base(t=0) = T_ext1_to_base @ T_wrist_to_ext1
  4. With FK at t=0:
         T_cam_to_hand = inv(T_hand_to_base(0)) @ T_wrist_to_base(0)
  5. Save to a per-episode JSON and a sanity image.

Validation step (the user's proposed test):
  - Reconstruct the scene at t=0 from ext1 (plane warp).
  - Render through the wrist camera using the VGGT-recovered T_cam_to_hand.
  - Compare visually with GT wrist[0]. Should match closely if VGGT is right.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Fix HF cache BEFORE any huggingface_hub import. The user's .bashrc sets
# HF_HOME=${XDG_CACHE_HOME}/huggingface, which on slurm compute nodes (no
# XDG_CACHE_HOME) expands to "/huggingface" — unwritable. Anything <= a single
# leading slash or pointing under /huggingface is treated as bogus and replaced
# with the project HF cache where VGGT-1B is pre-downloaded.
_HF_FALLBACK = "/storage/project/r-agarg35-0/lwang831/hf_cache"
_hf = os.environ.get("HF_HOME", "")
if (not _hf) or _hf == "/" or _hf.startswith("/huggingface"):
    os.environ["HF_HOME"] = _HF_FALLBACK
# HF_HUB_CACHE / HUGGINGFACE_HUB_CACHE override HF_HOME silently if set badly.
for _k in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
    _v = os.environ.get(_k, "")
    if _v and (_v == "/" or _v.startswith("/huggingface")):
        os.environ.pop(_k, None)

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "packages/ltx-action-cond/src")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from ltx_action_cond.calibration import intrinsics_to_K, rescale_K
from ltx_action_cond.kinematics import fk_urdf

_HERE = Path(__file__).parent
WRIST_ROOT = Path(os.environ.get("WRIST_DATA_ROOT", _HERE / "outputs" / "context"))
OUT_DIR = Path(os.environ.get("WRIST_CALIB_OUT", _HERE / "outputs" / "wrist_calib_vggt"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
# Optional filter: short-id list. If set, only these directories under
# WRIST_ROOT are processed (used by the resumable orchestrator). Prefer the
# file variant — a single env string > 128 KB hits Linux MAX_ARG_STRLEN.
def _load_eps_filter() -> set[str]:
    f = os.environ.get("VGGT_EPS_FILTER_FILE", "").strip()
    if f and Path(f).exists():
        return {ln.strip() for ln in Path(f).read_text().splitlines() if ln.strip()}
    return {s.strip() for s in os.environ.get("VGGT_EPS_FILTER", "").split(",") if s.strip()}

_EPS_FILTER = _load_eps_filter()


def load_vggt():
    from vggt.models.vggt import VGGT
    # Force CPU — the GPU is being held by the wan finetune.
    device = os.environ.get("VGGT_DEVICE", "cpu")
    print(f"Loading VGGT on {device}...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    return model, device


def _free_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run_vggt_pair(model, device, ext1_rgb: np.ndarray, wrist_rgb: np.ndarray):
    """Run VGGT on (ext1, wrist). Returns:
        T_ext1_to_world: 4x4 (= identity by construction)
        T_wrist_to_world: 4x4 (= the relative pose, ext1 is world origin)
        K_ext1_internal, K_wrist_internal: 3x3 intrinsics at VGGT's internal resolution
        internal_size: (H, W) the resolution VGGT processed at
    """
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    # VGGT's loader expects file paths -- save temporary PNGs.
    tmpdir = OUT_DIR / "_tmp"; tmpdir.mkdir(exist_ok=True)
    p1 = tmpdir / "ext1.png"; p2 = tmpdir / "wrist.png"
    Image.fromarray(ext1_rgb).save(p1)
    Image.fromarray(wrist_rgb).save(p2)
    images = load_and_preprocess_images([str(p1), str(p2)]).to(device)
    p1.unlink(); p2.unlink()

    is_cuda = (device == "cuda")
    dtype = torch.bfloat16 if (is_cuda and torch.cuda.get_device_capability()[0] >= 8) else torch.float32
    with torch.no_grad():
        if is_cuda:
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                tokens, ps_idx = model.aggregator(images[None])
                pose_enc = model.camera_head(tokens)[-1]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        else:
            tokens, ps_idx = model.aggregator(images[None])
            pose_enc = model.camera_head(tokens)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    # extrinsic shape: (1, N, 3, 4)  -- N=2 here. World-to-cam OR cam-to-world?
    # The VGGT docs in their README say "OpenCV convention (camera from world)" with
    # the column on the right being translation. The pose_encoding_to_extri_intri
    # output is consistent with cv2 convention: a 3x4 [R|t] world-to-camera matrix.
    E = extrinsic[0].detach().cpu().numpy()  # (N, 3, 4)
    K = intrinsic[0].detach().cpu().numpy()  # (N, 3, 3)

    # Convert each 3x4 [R|t] world-to-cam to 4x4 cam-to-world
    def w2c_to_c2w(rt: np.ndarray) -> np.ndarray:
        T = np.eye(4); T[:3, :4] = rt
        return np.linalg.inv(T)

    T_ext1_c2w = w2c_to_c2w(E[0])
    T_wrist_c2w = w2c_to_c2w(E[1])
    H, W = images.shape[-2:]
    return T_ext1_c2w, T_wrist_c2w, K[0], K[1], (H, W)


def calibrate_episode(model, device, ep_dir: Path, calib_info: dict) -> dict | None:
    """Compute T_cam_to_hand for one episode using VGGT pose."""
    data = np.load(ep_dir / "data.npz")
    ext1_img = data["ext1_frames"][0]
    wrist_img = data["wrist_frames"][0]
    cam2base_1 = data["cam2base_1"]  # 4x4
    q0 = data["cmd_joint_position"][0]
    info = json.loads((ep_dir / "info.json").read_text())

    # Run VGGT
    T_ext1_c2w, T_wrist_c2w, K_ext1_vggt, K_wrist_vggt, vggt_hw = run_vggt_pair(
        model, device, ext1_img, wrist_img,
    )
    # VGGT places image 0 (ext1) at world origin -- so T_wrist_c2w is the
    # pose of wrist in ext1's frame
    T_wrist_to_ext1 = np.linalg.inv(T_ext1_c2w) @ T_wrist_c2w
    # Compose with known ext1 cam2base
    T_wrist_to_base = cam2base_1 @ T_wrist_to_ext1
    # Compose with FK to recover T_cam_to_hand
    T_hand_to_base = fk_urdf(q0)["hand"]
    T_cam_to_hand = np.linalg.inv(T_hand_to_base) @ T_wrist_to_base

    return {
        "T_cam_to_hand": T_cam_to_hand.tolist(),
        "T_wrist_to_ext1": T_wrist_to_ext1.tolist(),
        "T_wrist_to_base": T_wrist_to_base.tolist(),
        "T_hand_to_base_t0": T_hand_to_base.tolist(),
        "vggt_resolution_hw": list(vggt_hw),
        "K_ext1_vggt": K_ext1_vggt.tolist(),
        "K_wrist_vggt": K_wrist_vggt.tolist(),
        "translation_cam_to_hand_m": T_cam_to_hand[:3, 3].tolist(),
    }


def main():
    model, device = load_vggt()
    results = {}
    for ep_dir in sorted(WRIST_ROOT.iterdir()):
        if not (ep_dir / "data.npz").exists():
            continue
        if _EPS_FILTER and ep_dir.name not in _EPS_FILTER:
            continue
        print(f"\n=== {ep_dir.name} ===")
        info = json.loads((ep_dir / "info.json").read_text())
        _free_gpu()
        try:
            r = calibrate_episode(model, device, ep_dir, info)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")
            _free_gpu()
            continue
        t = np.array(r["translation_cam_to_hand_m"])
        print(f"  T_cam_to_hand translation: {t}")
        # Save per-episode JSON
        out_path = OUT_DIR / f"{ep_dir.name}.json"
        out_path.write_text(json.dumps(r, indent=2))
        results[ep_dir.name] = r

    # Aggregate: T_cam_to_hand should be approximately the same across episodes
    # (it's a fixed physical mount). Median translation gives the consensus.
    if results:
        Ts = np.array([np.array(v["T_cam_to_hand"]) for v in results.values()])
        t_med = np.median(Ts[:, :3, 3], axis=0)
        t_spread = Ts[:, :3, 3].max(0) - Ts[:, :3, 3].min(0)
        print(f"\nAcross {len(results)} episodes:")
        print(f"  median T_cam_to_hand translation: {t_med}")
        print(f"  per-axis spread (max-min): {t_spread}  (mm: {1000*np.abs(t_spread).max():.1f})")
        agg = {
            "translation_median_m": t_med.tolist(),
            "translation_spread_m": t_spread.tolist(),
            "per_episode_translations_m": {
                k: v["translation_cam_to_hand_m"] for k, v in results.items()
            },
        }
        (OUT_DIR / "_aggregate.json").write_text(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
