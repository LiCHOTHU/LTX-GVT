"""Visual verification that the GPU-vectorised splat matches the CPU baseline.

For 5 episodes under outputs/context/<ep>/data.npz, renders:

  ext1.mp4   GT  |  Franka mesh rendered          (fixed cam, mesh — unaffected by this work)
  ext2.mp4   GT  |  Franka mesh rendered          (fixed cam, mesh — unaffected by this work)
  wrist.mp4  GT  |  GPU splat (NEW)  |  CPU splat (OLD baseline)  |  homography (canonical)

The GPU vs CPU columns are the actual verification: same scene, same poses, same
algorithm, only the splat backend differs. If they disagree visibly the
optimisation has a bug. The homography column shows what `wrist_demo.py` ships
to outputs/context — the canonical wrist context render is the homography path,
not the splat. Useful here only as a reference.

Outputs land under experiments/droid_action_cond/outputs/verify_splat_gpu/<ep>/.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "packages/ltx-action-cond/src")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from ltx_action_cond.kinematics import fk_urdf  # noqa: E402
from ltx_action_cond.mesh_rendering import FrankaMeshRenderer  # noqa: E402
from ltx_action_cond.wrist_render import (  # noqa: E402
    reconstruct_scene_from_plane,
    render_wrist_scene_splat,        # CPU baseline
    render_wrist_scene_splat_gpu,    # new GPU fast path
    render_wrist_via_plane_homography,
)

SRC = Path("experiments/droid_action_cond/outputs/context")
WRIST_CALIB_DIR = Path("experiments/droid_action_cond/outputs/wrist_calib_vggt")
OUT_ROOT = Path("experiments/droid_action_cond/outputs/verify_splat_gpu")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def _derive_T_cam_to_hand(d, calib) -> np.ndarray:
    """Same FK-anchored scale fix as wrist_demo._derive_T_cam_to_hand."""
    T_wrist_to_ext = np.array(calib.get("T_wrist_to_ext2", calib.get("T_wrist_to_ext1")))
    use_ext2 = "T_wrist_to_ext2" in calib
    cam2base_anchor = d["cam2base_2" if use_ext2 else "cam2base_1"]
    T_hand_to_base_t0 = fk_urdf(d["cmd_joint_position"][0])["hand"]
    T_hand_to_anchor = np.linalg.inv(cam2base_anchor) @ T_hand_to_base_t0
    true_dist = float(np.linalg.norm(T_hand_to_anchor[:3, 3]))
    vggt_dist = float(np.linalg.norm(T_wrist_to_ext[:3, 3]))
    scale = true_dist / vggt_dist if vggt_dist > 1e-6 else 1.0
    T_w_scaled = T_wrist_to_ext.copy()
    T_w_scaled[:3, 3] *= scale
    T_wrist_to_base_t0 = cam2base_anchor @ T_w_scaled
    return np.linalg.inv(T_hand_to_base_t0) @ T_wrist_to_base_t0


def _load_T_cam_to_hand(ep_short: str, d) -> np.ndarray:
    p2 = WRIST_CALIB_DIR / f"{ep_short}_ext2anchor.json"
    if p2.exists():
        return np.array(json.loads(p2.read_text())["T_cam_to_hand"])
    p1 = WRIST_CALIB_DIR / f"{ep_short}.json"
    if not p1.exists():
        raise FileNotFoundError(f"no VGGT calib for {ep_short}")
    return _derive_T_cam_to_hand(d, json.loads(p1.read_text()))


def label(img: np.ndarray, text: str, font_size: int = 16) -> np.ndarray:
    h = font_size + 10
    bar = Image.new("RGB", (img.shape[1], h), color=(0, 0, 0))
    d = ImageDraw.Draw(bar)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        f = ImageFont.load_default()
    d.text((6, 4), text, fill=(255, 255, 255), font=f)
    return np.concatenate([np.array(bar), img], axis=0)


def upscale(img: np.ndarray, s: int = 3) -> np.ndarray:
    return np.array(Image.fromarray(img).resize((img.shape[1] * s, img.shape[0] * s), Image.LANCZOS))


def hstack(*imgs: np.ndarray, sep_px: int = 6) -> np.ndarray:
    sep = np.zeros((imgs[0].shape[0], sep_px, 3), dtype=np.uint8)
    out = imgs[0]
    for im in imgs[1:]:
        out = np.concatenate([out, sep, im], axis=1)
    return out


def process_episode(ep_dir: Path, mesh_renderer: FrankaMeshRenderer) -> dict:
    ep_short = ep_dir.name
    d = np.load(ep_dir / "data.npz")
    joints = d["cmd_joint_position"]
    grippers = d["gripper_for_render"].reshape(-1)
    ext1f = d["ext1_frames"]; ext2f = d["ext2_frames"]; wristf = d["wrist_frames"]
    K1 = d["K1"]; K2 = d["K2"]; K_wrist = d["K_wrist"]
    cam2base_1 = d["cam2base_1"]; cam2base_2 = d["cam2base_2"]
    T = ext1f.shape[0]
    H, W = ext1f.shape[1], ext1f.shape[2]

    T_cam_to_hand = _load_T_cam_to_hand(ep_short, d)
    T_hand_0 = fk_urdf(joints[0])["hand"]
    T_wrist0_to_base = T_hand_0 @ T_cam_to_hand
    wrist0_img = wristf[0]

    # Scenes for the two splat backends. Use wrist[0] as the source (same
    # anchor the canonical homography uses) so the comparison is apples-to-apples.
    scene_gpu = reconstruct_scene_from_plane(wrist0_img, K_wrist, T_wrist0_to_base,
                                              plane_z=0.0, sample_stride=2)  # NEW fast
    scene_cpu = reconstruct_scene_from_plane(wrist0_img, K_wrist, T_wrist0_to_base,
                                              plane_z=0.0, sample_stride=1)  # OLD baseline
    cache_gpu: dict = {}

    ext1_frames, ext2_frames, wrist_frames = [], [], []
    t_gpu = t_cpu = t_homog = t_mesh = 0.0
    s_up = 3

    for t in range(T):
        T_wrist_t_to_base = fk_urdf(joints[t])["hand"] @ T_cam_to_hand

        ts = time.perf_counter()
        c_ext1, _ = mesh_renderer.render(joints[t], K1, cam2base_1, gripper_position=float(grippers[t]))
        c_ext2, _ = mesh_renderer.render(joints[t], K2, cam2base_2, gripper_position=float(grippers[t]))
        t_mesh += time.perf_counter() - ts

        ts = time.perf_counter()
        w_gpu, _ = render_wrist_scene_splat_gpu(
            scene_gpu, T_wrist_t_to_base, K_wrist, (W, H),
            point_radius_px=3, _scene_cache=cache_gpu,
        )
        t_gpu += time.perf_counter() - ts

        ts = time.perf_counter()
        w_cpu, _ = render_wrist_scene_splat(
            scene_cpu, T_wrist_t_to_base, K_wrist, (W, H), point_radius_px=3,
        )
        t_cpu += time.perf_counter() - ts

        ts = time.perf_counter()
        w_hom = render_wrist_via_plane_homography(
            wrist0_img, K_wrist, T_wrist0_to_base,
            T_wrist_t_to_base, K_wrist, (W, H), plane_z=0.0,
        )
        t_homog += time.perf_counter() - ts

        # Build per-camera rows
        ext1_frames.append(hstack(
            label(upscale(ext1f[t], s_up), f"GT ext1 t={t}"),
            label(upscale(c_ext1, s_up),    f"mesh ext1 t={t}"),
        ))
        ext2_frames.append(hstack(
            label(upscale(ext2f[t], s_up), f"GT ext2 t={t}"),
            label(upscale(c_ext2, s_up),    f"mesh ext2 t={t}"),
        ))
        wrist_frames.append(hstack(
            label(upscale(wristf[t], s_up), f"GT wrist t={t}"),
            label(upscale(w_gpu, s_up),     f"GPU splat (NEW) t={t}"),
            label(upscale(w_cpu, s_up),     f"CPU splat (OLD) t={t}"),
            label(upscale(w_hom, s_up),     f"homography (canonical)"),
        ))

    out_dir = OUT_ROOT / ep_short
    out_dir.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_dir / "ext1.mp4",  ext1_frames,  fps=15, macro_block_size=1)
    imageio.mimsave(out_dir / "ext2.mp4",  ext2_frames,  fps=15, macro_block_size=1)
    imageio.mimsave(out_dir / "wrist.mp4", wrist_frames, fps=15, macro_block_size=1)

    return {
        "ep": ep_short, "T": T,
        "mesh_ms":  1000 * t_mesh  / (2 * T),   # per single render
        "gpu_ms":   1000 * t_gpu   / T,
        "cpu_ms":   1000 * t_cpu   / T,
        "homog_ms": 1000 * t_homog / T,
    }


def main() -> None:
    eps = sorted(p for p in SRC.iterdir() if p.is_dir() and (p / "data.npz").exists())
    if not eps:
        raise SystemExit(f"no episodes under {SRC}")
    print(f"verifying GPU splat on {len(eps)} episodes from {SRC}")
    H = W = 0
    d0 = np.load(eps[0] / "data.npz")
    H, W = d0["ext1_frames"].shape[1], d0["ext1_frames"].shape[2]
    mesh = FrankaMeshRenderer(size=(W, H), mode="photo")
    stats = []
    try:
        for ep in eps:
            print(f"  {ep.name} ...")
            stats.append(process_episode(ep, mesh))
    finally:
        mesh.close()
    print("\n=== per-render avg (ms/frame) ===")
    print(f"  {'episode':45s}  {'T':>4s}   mesh    GPU    CPU   homog")
    for s in stats:
        print(f"  {s['ep']:45s}  {s['T']:>4d}  {s['mesh_ms']:5.2f}  {s['gpu_ms']:5.2f}  {s['cpu_ms']:5.2f}  {s['homog_ms']:5.2f}")
    print(f"\noutputs under {OUT_ROOT}")


if __name__ == "__main__":
    main()
