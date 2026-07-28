"""Produce a larger variety of demo videos from the already-rendered outputs.

All written under outputs/demo_videos/. Reads the existing 01_gt/, 02_context/,
04_context_mesh/, 06_context_photo/, 07_overlay_photo/ image sequences.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "outputs"
DEMO = OUT / "demo_videos"
DEMO.mkdir(exist_ok=True)


def load_seq(d: Path, prefix: str) -> list[np.ndarray]:
    return [np.array(Image.open(p)) for p in sorted(d.glob(f"{prefix}_*.png"))]


def label_strip(text: str, w: int, h: int = 24, font_size: int = 16) -> np.ndarray:
    img = Image.new("RGB", (w, h), color=(0, 0, 0))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    d.text((6, 2), text, fill=(240, 240, 240), font=font)
    return np.array(img)


def attach_label(frames: list[np.ndarray], text: str, font_size: int = 16) -> list[np.ndarray]:
    h = max(24, font_size + 6)
    lbl = label_strip(text, w=frames[0].shape[1], h=h, font_size=font_size)
    return [np.concatenate([lbl, f], axis=0) for f in frames]


def stack_horizontal(seqs: list[list[np.ndarray]], sep_w: int = 4) -> list[np.ndarray]:
    n = len(seqs[0])
    H = seqs[0][0].shape[0]
    sep = np.zeros((H, sep_w, 3), dtype=np.uint8)
    out = []
    for t in range(n):
        parts = [seqs[0][t]]
        for s in seqs[1:]:
            parts.append(sep)
            parts.append(s[t])
        out.append(np.concatenate(parts, axis=1))
    return out


def upscale(frames: list[np.ndarray], factor: int) -> list[np.ndarray]:
    return [
        np.array(Image.fromarray(f).resize(
            (f.shape[1] * factor, f.shape[0] * factor), Image.LANCZOS,
        ))
        for f in frames
    ]


def save(path: Path, frames: list[np.ndarray], fps: int = 15, loop_gif: int = 0) -> None:
    if path.suffix == ".gif":
        imageio.mimsave(path, frames, duration=1.0 / fps, loop=loop_gif)
    else:
        imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
    print(f"  wrote {path.name}  ({path.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    gt = load_seq(OUT / "01_gt", "gt")
    skel = load_seq(OUT / "02_context", "ctx")
    mask = load_seq(OUT / "04_context_mesh", "cmsh")
    photo = load_seq(OUT / "06_context_photo", "cpho")
    opho = load_seq(OUT / "07_overlay_photo", "opho")
    n = len(gt)
    print(f"Loaded {n} frames at {gt[0].shape[1]}x{gt[0].shape[0]}")

    # --- 1. Focused 2-ups at native resolution ---
    save(DEMO / "01_headline_gt_vs_photo.mp4",
         stack_horizontal([attach_label(gt, "GT"), attach_label(photo, "rendered Franka")]))
    save(DEMO / "02_proof_gt_vs_overlay.mp4",
         stack_horizontal([attach_label(gt, "GT"), attach_label(opho, "rendered on GT")]))
    save(DEMO / "03_priors_mask_vs_photo.mp4",
         stack_horizontal([attach_label(mask, "per-link mask"), attach_label(photo, "photoreal")]))

    # --- 2. 4x upscaled (1280x720) ---
    gt_up = upscale(gt, 4)
    photo_up = upscale(photo, 4)
    opho_up = upscale(opho, 4)
    save(DEMO / "04_photo_big.mp4", attach_label(photo_up, "photoreal Franka (4x)", font_size=28))
    save(DEMO / "05_overlay_big.mp4", attach_label(opho_up, "rendered on GT (4x)", font_size=28))
    save(DEMO / "06_headline_big.mp4",
         stack_horizontal([attach_label(gt_up, "GT", font_size=28),
                           attach_label(photo_up, "rendered", font_size=28),
                           attach_label(opho_up, "rendered on GT", font_size=28)]))

    # --- 3. 4-up everything (native) ---
    save(DEMO / "07_everything_4up.mp4",
         stack_horizontal([attach_label(gt, "GT"),
                           attach_label(skel, "skeleton"),
                           attach_label(mask, "mask"),
                           attach_label(photo, "photoreal")]))

    # --- 4. Speed variations ---
    save(DEMO / "08_slow_overlay.mp4", attach_label(opho_up, "slow-mo (5 fps)", font_size=28), fps=5)
    save(DEMO / "09_double_speed_overlay.mp4", attach_label(opho, "2x speed (30 fps)"), fps=30)

    # --- 5. Loops ---
    save(DEMO / "10_boomerang_overlay.mp4",
         attach_label(opho + list(reversed(opho)), "boomerang loop"))
    save(DEMO / "11_loop3x_overlay.mp4", attach_label(opho * 3, "3x loop"))

    # --- 6. GIF for browser-friendly viewing ---
    save(DEMO / "12_headline.gif",
         stack_horizontal([attach_label(gt, "GT"),
                           attach_label(photo, "rendered"),
                           attach_label(opho, "overlay")]))

    print(f"\nWrote {len(list(DEMO.glob('*')))} files under {DEMO}/")


if __name__ == "__main__":
    main()
