"""Build the per-frame projection prior for the chosen DROID episode.

Reads outputs/episode.npz, runs Franka FK on the commanded joint trajectory,
projects link keypoints into the calibrated camera, and writes BOTH stages of
prior:

  outputs/02_context/ctx_NNNN.png        Stage 1: skeleton on black (THE MODEL INPUT)
  outputs/03_overlay/ovr_NNNN.png        Stage 1: skeleton on GT     (sanity only)
  outputs/04_context_mesh/cmsh_NNNN.png  Stage 2: per-link color mask on black
  outputs/05_overlay_mesh/omsh_NNNN.png  Stage 2: mesh blended on GT  (sanity only)
  outputs/prior_report.json              geometry sanity stats

Stage 2 needs the Franka FER visual meshes at:
  /home/licho/data/robot_assets/franka_description/meshes/robots/fer/visual/*.dae
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ltx_action_cond import draw_skeleton, fk, project_points
from ltx_action_cond.mesh_rendering import FrankaMeshRenderer

OUT = Path(__file__).parent / "outputs"
CTX_DIR = OUT / "02_context"
OVR_DIR = OUT / "03_overlay"
CMSH_DIR = OUT / "04_context_mesh"      # mask mode: per-link flat colors
OMSH_DIR = OUT / "05_overlay_mesh"      # mask overlaid on GT
CPHO_DIR = OUT / "06_context_photo"     # photo mode: textured Franka render
OPHO_DIR = OUT / "07_overlay_photo"     # photo overlaid on GT


def blend(gt: np.ndarray, mesh_rgb: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    """Alpha-blend the rendered mesh on top of GT only where the mesh covers."""
    coverage = (mesh_rgb.sum(-1) > 0).astype(np.float32)[..., None]
    out = gt.astype(np.float32) * (1 - alpha * coverage) + mesh_rgb.astype(np.float32) * (alpha * coverage)
    return out.clip(0, 255).astype(np.uint8)


def main() -> None:
    for d in (CTX_DIR, OVR_DIR, CMSH_DIR, OMSH_DIR, CPHO_DIR, OPHO_DIR):
        d.mkdir(exist_ok=True)

    data = np.load(OUT / "episode.npz")
    joints = data["cmd_joint_position"]
    grippers = data["cmd_gripper_position"]  # drives finger opening (0=open, 1=closed)
    actions = data["action"]
    gt_frames = data["gt_frames"]
    cam2base = data["cam2base"]
    K = data["K_rls"]

    T, H, W = gt_frames.shape[:3]
    base2cam = np.linalg.inv(cam2base)
    print(f"T={T}  size={W}x{H}")

    # --- Sanity at t=0 ---
    kpts0 = fk(joints[0])
    pix_kp0, _ = project_points(kpts0, base2cam, K)
    pix_cart, _ = project_points(actions[:, :3], base2cam, K)
    err_t0 = float(np.linalg.norm(pix_kp0[8] - pix_cart[0]))
    print(f"flange (FK) t=0 = ({pix_kp0[8, 0]:.2f}, {pix_kp0[8, 1]:.2f})")
    print(f"cart   EE  t=0 = ({pix_cart[0, 0]:.2f}, {pix_cart[0, 1]:.2f})")
    print(f"pixel error: {err_t0:.3f}px")

    # --- Stage 1: skeleton ---
    print(f"\nStage 1: rendering skeleton ({T} frames)...")
    pix_per_frame, z_per_frame = [], []
    t0 = time.perf_counter()
    for t in range(T):
        kpts = fk(joints[t])
        pix, z = project_points(kpts, base2cam, K)
        pix_per_frame.append(pix)
        z_per_frame.append(z)
        ctx = Image.new("RGB", (W, H), color=(0, 0, 0))
        draw_skeleton(ctx, pix, z)
        ctx.save(CTX_DIR / f"ctx_{t:04d}.png")
        ovr = Image.fromarray(gt_frames[t])
        draw_skeleton(ovr, pix, z)
        ovr.save(OVR_DIR / f"ovr_{t:04d}.png")
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    # --- Stage 2a: per-link mask ---
    print(f"\nStage 2a: rendering per-link mask ({T} frames)...")
    renderer = FrankaMeshRenderer(size=(W, H), mode="mask")
    t0 = time.perf_counter()
    mesh_coverage = []
    for t in range(T):
        color, _ = renderer.render(joints[t], K, cam2base, gripper_position=float(grippers[t]))
        Image.fromarray(color).save(CMSH_DIR / f"cmsh_{t:04d}.png")
        Image.fromarray(blend(gt_frames[t], color)).save(OMSH_DIR / f"omsh_{t:04d}.png")
        mesh_coverage.append(float((color.sum(-1) > 0).mean()))
    renderer.close()
    elapsed_mask = time.perf_counter() - t0
    print(f"  done in {elapsed_mask:.1f}s  ({1000 * elapsed_mask / T:.1f} ms/frame)")

    # --- Stage 2b: photoreal Franka render (textured) ---
    print(f"\nStage 2b: rendering photoreal Franka ({T} frames)...")
    renderer = FrankaMeshRenderer(size=(W, H), mode="photo")
    t0 = time.perf_counter()
    for t in range(T):
        color, _ = renderer.render(joints[t], K, cam2base, gripper_position=float(grippers[t]))
        Image.fromarray(color).save(CPHO_DIR / f"cpho_{t:04d}.png")
        Image.fromarray(blend(gt_frames[t], color, alpha=0.85)).save(OPHO_DIR / f"opho_{t:04d}.png")
    renderer.close()
    elapsed_photo = time.perf_counter() - t0
    print(f"  done in {elapsed_photo:.1f}s  ({1000 * elapsed_photo / T:.1f} ms/frame)")

    # --- Stats ---
    pix_arr = np.stack(pix_per_frame, axis=0)
    depth_arr = np.stack(z_per_frame, axis=0)
    inside = (
        (pix_arr[..., 0] >= 0) & (pix_arr[..., 0] < W)
        & (pix_arr[..., 1] >= 0) & (pix_arr[..., 1] < H)
        & (depth_arr > 0)
    )
    report = {
        "n_frames": int(T),
        "img_size": [int(W), int(H)],
        "flange_t0_pix": [float(pix_kp0[8, 0]), float(pix_kp0[8, 1])],
        "cart_target_t0_pix": [float(pix_cart[0, 0]), float(pix_cart[0, 1])],
        "flange_vs_cart_pix_err_t0": err_t0,
        "frac_keypoints_inside_image": float(inside.mean()),
        "mean_flange_depth_m": float(np.mean([z[8] for z in z_per_frame])),
        "mesh_coverage_mean_frac": float(np.mean(mesh_coverage)),
        "mask_render_ms_per_frame": float(1000 * elapsed_mask / T),
        "photo_render_ms_per_frame": float(1000 * elapsed_photo / T),
    }
    with (OUT / "prior_report.json").open("w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
