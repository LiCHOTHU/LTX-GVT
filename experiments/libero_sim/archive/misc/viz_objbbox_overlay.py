#!/usr/bin/env python3
# ruff: noqa: T201
"""Smoke viz: draw the projected object 3D-bbox onto the GT frames of a context chunk.

Confirms the object-centric AL signal's geometry end-to-end WITHOUT the 22B model:
reuses build_libero_objbbox.build_one (object capture -> 8 world corners) and
score_chunk_objbbox_pixel._project_boxes (the exact projection the scorer uses), then
overlays the 2D box hull on agentview + wrist GT frames. If the boxes sit on the
tomato_sauce / basket in both views, the capture + projection are correct.

    python viz_objbbox_overlay.py --chunk <ctx>/<ep>/chunk_00/data.npz --out overlay.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from build_libero_objbbox import build_one
from score_chunk_objbbox_pixel import _project_boxes


def _draw(img: np.ndarray, boxes, color=(255, 0, 0), thick=2) -> np.ndarray:
    out = img.copy()
    H, W = out.shape[:2]
    for (u0, v0, u1, v1) in boxes:
        for t in range(thick):
            for y in (v0 + t, v1 - 1 - t):
                if 0 <= y < H:
                    out[y, max(0, u0):min(W, u1)] = color
            for x in (u0 + t, u1 - 1 - t):
                if 0 <= x < W:
                    out[max(0, v0):min(H, v1), x] = color
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=Path, required=True, help="path to a chunk's data.npz")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", default="0,mid,last")
    args = ap.parse_args()

    d = np.load(args.chunk, allow_pickle=False)
    corners = build_one(args.chunk)["corners"].numpy()  # (n_obj, F, 8, 3)
    names = [str(x) for x in d["obj_names"]] if "obj_names" in d.files else []
    F = d["agentview_frames"].shape[0]
    print(f"chunk F={F}  objects={names}  corners={corners.shape}")
    if corners.size == 0:
        raise SystemExit("no object corners in this chunk — object capture failed")

    sel = []
    for tok in args.frames.split(","):
        sel.append(0 if tok == "0" else (F - 1 if tok == "last" else F // 2))
    sel = sorted(set(sel))

    views = [("agentview", d["agentview_K"], d["agentview_cam2world"], d["agentview_frames"]),
             ("wrist", d["wrist_K"], d["wrist_cam2world"], d["wrist_frames"])]
    rows = []
    for f in sel:
        tiles = []
        for vname, K, c2w, frames in views:
            vm = np.linalg.inv(c2w[f].astype(np.float64))  # world->cam (== cameras stream)
            boxes = _project_boxes(corners[:, f], vm, K.astype(np.float64))
            tiles.append(_draw(frames[f], boxes))
            print(f"  frame {f:2d} {vname:9s}: {len(boxes)} box(es) visible -> {boxes}")
        rows.append(np.concatenate(tiles, axis=1))  # [agentview | wrist]
    grid = np.concatenate(rows, axis=0)             # rows = frames
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(args.out, grid)
    print(f"\nWrote {args.out}  ({grid.shape[1]}x{grid.shape[0]}; cols=[agentview|wrist], rows={sel})")


if __name__ == "__main__":
    main()
