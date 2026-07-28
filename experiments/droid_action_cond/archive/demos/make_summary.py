"""Assemble human-readable outputs/ for the chosen DROID episode.

Layout produced:
  outputs/
    README.txt
    meta.json
    prior_report.json
    episode.npz
    01_gt/              GT frames
    02_context/         skeleton on black           (Stage 1 model input)
    03_overlay/         skeleton on GT              (Stage 1 sanity)
    04_context_mesh/    per-link color mesh mask    (Stage 2 model input)
    05_overlay_mesh/    mesh blended on GT          (Stage 2 sanity)
    summary_timeline.png   one PNG: 5 timestamps x 5 columns (gt | skel | overlay | mesh | overlay_mesh)
    videos/                various MP4s
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "outputs"


def label_strip(text: str, w: int, h: int = 18) -> np.ndarray:
    img = Image.new("RGB", (w, h), color=(0, 0, 0))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    d.text((4, 2), text, fill=(240, 240, 240), font=font)
    return np.array(img)


def attach_label(stack: list[np.ndarray], text: str) -> list[np.ndarray]:
    lbl = label_strip(text, w=stack[0].shape[1])
    return [np.concatenate([lbl, f], axis=0) for f in stack]


def load_seq(dir_path: Path, prefix: str) -> list[np.ndarray]:
    files = sorted(dir_path.glob(f"{prefix}_*.png"))
    return [np.array(Image.open(p)) for p in files]


def main() -> None:
    gt = load_seq(OUT / "01_gt", "gt")
    skel = load_seq(OUT / "02_context", "ctx")
    ovr = load_seq(OUT / "03_overlay", "ovr")
    mesh = load_seq(OUT / "04_context_mesh", "cmsh")
    omsh = load_seq(OUT / "05_overlay_mesh", "omsh")
    photo = load_seq(OUT / "06_context_photo", "cpho")
    opho = load_seq(OUT / "07_overlay_photo", "opho")
    n = len(gt)
    H, W = gt[0].shape[:2]
    print(f"n_frames={n} size={W}x{H}")

    videos_dir = OUT / "videos"
    videos_dir.mkdir(exist_ok=True)

    cols = [
        (gt,    "GT"),
        (skel,  "skeleton"),
        (mesh,  "per-link mask"),
        (photo, "photoreal Franka"),
        (opho,  "photoreal on GT"),
    ]

    labeled = [(attach_label(frames, name), name) for frames, name in cols]

    # 1. individual labeled videos
    for frames, name in labeled:
        slug = name.split(" ")[0].lower().rstrip(":")
        imageio.mimsave(videos_dir / f"{slug}.mp4", frames, fps=15, macro_block_size=1)

    # 2. headline comparison: GT | photoreal | photoreal-on-GT
    sep_v = np.zeros((H + 18, 2, 3), dtype=np.uint8)
    triplet = [
        np.concatenate([labeled[0][0][t], sep_v, labeled[3][0][t], sep_v, labeled[4][0][t]], axis=1)
        for t in range(n)
    ]
    imageio.mimsave(videos_dir / "gt_vs_photo_vs_overlay.mp4", triplet, fps=15, macro_block_size=1)

    # 3. all-5 side-by-side
    quint = [
        np.concatenate(
            [labeled[0][0][t], sep_v, labeled[1][0][t], sep_v, labeled[2][0][t],
             sep_v, labeled[3][0][t], sep_v, labeled[4][0][t]],
            axis=1,
        )
        for t in range(n)
    ]
    imageio.mimsave(videos_dir / "all_five.mp4", quint, fps=15, macro_block_size=1)

    # 4. timeline summary: 5 timestamps x 5 columns
    timestamps = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
    col_w = W
    row_h = H + 18
    pad = 4
    title_h = 22
    sheet_w = title_h + len(cols) * col_w + (len(cols) - 1) * pad
    sheet_h = title_h + len(timestamps) * row_h + (len(timestamps) - 1) * pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(0, 0, 0))

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    d = ImageDraw.Draw(sheet)
    for ci, (_, name) in enumerate(cols):
        x = title_h + ci * (col_w + pad) + 4
        d.text((x, 4), name, fill=(240, 240, 240), font=font)

    for ri, t in enumerate(timestamps):
        y = title_h + ri * (row_h + pad)
        d.text((4, y + row_h // 2 - 6), f"t={t:03d}", fill=(240, 240, 240), font=font)
        for ci, (frames, _) in enumerate(cols):
            lbl = label_strip(f"t={t:03d}", w=col_w)
            tile = np.concatenate([lbl, frames[t]], axis=0)
            sheet.paste(Image.fromarray(tile), (title_h + ci * (col_w + pad), y))
    sheet.save(OUT / "summary_timeline.png")

    # 5. README
    with (OUT / "meta.json").open() as f:
        meta = json.load(f)
    with (OUT / "prior_report.json").open() as f:
        rpt = json.load(f)
    (OUT / "README.txt").write_text(f"""Action-conditioned video model: projection priors for one DROID episode (single fixed exterior view).

Episode:    {meta['episode_id']}
Camera:     {meta['camera_role']} (serial {meta['camera_serial']}) -> RLDS key {meta['img_key']}
IoU:        {meta['quality_metric_IoU']:.3f}  (April 2025 cam2base extrinsic)
Length:     {meta['n_steps']} frames at 15Hz ({meta['duration_sec_at_15Hz']}s)
Image:      320 x 180 (every output here is at this resolution)

Geometry sanity:
  flange (FK on commanded joint) t=0 = ({rpt['flange_t0_pix'][0]:.1f}, {rpt['flange_t0_pix'][1]:.1f})
  cartesian-EE target             t=0 = ({rpt['cart_target_t0_pix'][0]:.1f}, {rpt['cart_target_t0_pix'][1]:.1f})
  pixel error                          = {rpt['flange_vs_cart_pix_err_t0']:.2f} px (sub-pixel)
  fraction of FK keypoints in frame    = {rpt['frac_keypoints_inside_image']*100:.0f}%
  mean mesh coverage                   = {rpt['mesh_coverage_mean_frac']*100:.1f}% of pixels
  mask render speed                    = {rpt['mask_render_ms_per_frame']:.1f} ms/frame
  photoreal render speed               = {rpt['photo_render_ms_per_frame']:.1f} ms/frame
  (pyrender + EGL on RTX 5090)

Layout:
  README.txt              this file
  meta.json               episode + calibration metadata
  prior_report.json       sanity stats
  episode.npz             joints (obs+cmd), action, gripper, gt_frames, K_rls, cam2base
  summary_timeline.png    5 timestamps x 5 columns mosaic
  01_gt/                  ground truth video frames                       ({meta['n_steps']} PNGs)
  02_context/             skeleton on black           (Stage 1 input)     ({meta['n_steps']} PNGs)
  03_overlay/             skeleton on GT              (Stage 1 sanity)    ({meta['n_steps']} PNGs)
  04_context_mesh/        per-link color mask         (Stage 2a input)    ({meta['n_steps']} PNGs)
  05_overlay_mesh/        mask on GT                  (Stage 2a sanity)   ({meta['n_steps']} PNGs)
  06_context_photo/       textured Franka on black    (Stage 2b input)    ({meta['n_steps']} PNGs)
  07_overlay_photo/       textured Franka on GT       (Stage 2b sanity)   ({meta['n_steps']} PNGs)
  videos/
    gt.mp4              skeleton.mp4              per-link.mp4
    photoreal.mp4       photoreal::.mp4
    gt_vs_photo_vs_overlay.mp4   headline 3-up
    all_five.mp4

Modes:
  Stage 1 (skeleton):   sparse, ~5 keypoints visible per frame. Quick sanity.
  Stage 2a (per-link mask): articulation-aware segmentation (one color per link).
                            High SNR for the model to read pose identity.
  Stage 2b (photoreal): real Franka textures + directional lighting.
                        Looks like the actual robot floating in a black void.

Mesh assets: /home/licho/data/robot_assets/franka_description/meshes/robots/fer/visual/
Kinematics:  URDF FK from kinematics.yaml; matches DH FK to machine precision.
Fingers:     2x finger.dae instantiated at hand-local (0, +-q, 0.0584); q from gripper_position.
""")
    print(f"wrote {OUT}/summary_timeline.png and {videos_dir}/ ({len(list(videos_dir.iterdir()))} videos)")


if __name__ == "__main__":
    main()
