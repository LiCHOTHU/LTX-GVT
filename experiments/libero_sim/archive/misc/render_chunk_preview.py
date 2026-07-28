#!/usr/bin/env python3
"""Render AL context chunks (data.npz) into side-by-side mp4 panels for eyeballing.

Each chunk's data.npz holds the GT sim frames and the rendered context priors. We
stack them per-frame into one panel so you can see, for each candidate clip, what the
model is being conditioned on (context_*) vs. the ground-truth target (*_frames):

    [ agentview GT | agentview context | wrist GT | wrist context ]

These are TRAINING INPUTS, not model generations -- they show the data the AL loop is
feeding, not the world model's output quality.
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def label(img: np.ndarray, text: str) -> np.ndarray:
    # cheap top-left text via a darkened strip; avoids font deps
    out = img.copy()
    out[:16, :, :] = (out[:16, :, :] * 0.35).astype(np.uint8)
    return out


def make_panel(npz_path: Path) -> np.ndarray | None:
    z = np.load(npz_path)
    keys = ["agentview_frames", "agentview_context_robot", "wrist_frames", "wrist_context_fused"]
    have = [k for k in keys if k in z.files]
    if "agentview_frames" not in have:
        return None
    tiles = [z[k] for k in have]  # each (T,H,W,3)
    T = min(t.shape[0] for t in tiles)
    frames = []
    for f in range(T):
        row = np.concatenate([label(t[f], k) for t, k in zip(tiles, have)], axis=1)
        frames.append(row)
    return np.stack(frames)  # (T, H, W*ntiles, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-root", required=True, help="round dir's context/ folder")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=10, help="how many chunks")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--stride", type=int, default=1,
                    help="take every Nth chunk for episode spread (default 1)")
    args = ap.parse_args()

    root = Path(args.context_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    chunks = sorted(root.glob("*/chunk_*/data.npz"))
    if args.stride > 1:
        chunks = chunks[:: args.stride]
    chunks = chunks[: args.n]

    manifest = []
    for i, c in enumerate(chunks):
        panel = make_panel(c)
        if panel is None:
            print(f"[skip] {c} (no agentview frames)")
            continue
        ep = c.parent.parent.name
        ck = c.parent.name
        name = f"{i:02d}__{ep[:70]}__{ck}.mp4"
        dst = out / name
        imageio.mimwrite(dst, panel, fps=args.fps, codec="libx264",
                         output_params=["-pix_fmt", "yuv420p"])
        info = {}
        ij = c.parent / "info.json"
        if ij.exists():
            info = json.loads(ij.read_text())
        manifest.append({"file": name, "ep": ep, "chunk": ck,
                         "task": info.get("task"), "frames": int(panel.shape[0]),
                         "panel_wh": [int(panel.shape[2]), int(panel.shape[1])]})
        print(f"[ok] {dst}  ({panel.shape[0]} frames, {panel.shape[2]}x{panel.shape[1]})")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[done] {len(manifest)} panels -> {out}")
    print(f"  layout per frame: [ agentview GT | agentview context | wrist GT | wrist context ]")


if __name__ == "__main__":
    main()
