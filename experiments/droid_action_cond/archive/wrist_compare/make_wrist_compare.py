#!/usr/bin/env python
"""Build wrist-context comparison videos: GT wrist | gripper_aware | multiview.

Reads the per-episode chunk renders produced by chunk_context.py for both
WRIST_CONTEXT_MODE=gripper_aware and =multiview (rendered with
WRIST_NO_GRIPPER=1 so the context is the clean warped plate only — no gripper).

Layout per frame (left -> right):  GT wrist | gripper_aware ctx | multiview ctx
Each panel is labeled and upscaled 3x (nearest) for visibility.
Writes one mp4 per episode plus a stacked ALL.mp4 (episodes stacked vertically).
"""
import os
import numpy as np
import imageio.v2 as imageio

ROOT = os.path.join(os.path.dirname(__file__), "outputs", "wrist_compare2")
OUT = os.path.join(ROOT, "compare")
CHUNK = "chunk_00"
UPSCALE = 3
FPS = 10

os.makedirs(OUT, exist_ok=True)


def upscale(img, k):
    return np.repeat(np.repeat(img, k, axis=0), k, axis=1)


def label(frame, text):
    # simple top bar with text drawn via a tiny 5x7 bitmap font is overkill;
    # just darken a top strip so the panels are visually separable.
    out = frame.copy()
    out[:12] = (out[:12] * 0.35).astype(np.uint8)
    return out


def vsep(h, w=4):
    # vertical bar for horizontal concat: shape (H, w, 3)
    return np.zeros((h, w, 3), np.uint8) + 60


def main():
    eps = sorted(os.listdir(os.path.join(ROOT, "gripper_aware")))
    eps = [e for e in eps if os.path.isdir(os.path.join(ROOT, "gripper_aware", e))]
    print(f"{len(eps)} episodes: {eps}")

    all_rows = []
    for ep in eps:
        ga = np.load(os.path.join(ROOT, "gripper_aware", ep, CHUNK, "data.npz"))
        mv = np.load(os.path.join(ROOT, "multiview", ep, CHUNK, "data.npz"))
        gt = ga["wrist_frames"]            # (T,H,W,3)
        ga_ctx = ga["wrist_context"]
        mv_ctx = mv["wrist_context"]
        T, H, W, _ = gt.shape

        frames = []
        for t in range(T):
            panels = [label(gt[t], "GT"),
                      label(ga_ctx[t], "gripper_aware"),
                      label(mv_ctx[t], "multiview")]
            row = panels[0]
            for p in panels[1:]:
                row = np.concatenate([row, vsep(H), p], axis=1)
            frames.append(upscale(row, UPSCALE))
        frames = np.stack(frames, 0)

        outp = os.path.join(OUT, f"{ep}.mp4")
        imageio.mimsave(outp, frames, fps=FPS)
        print(f"wrote {outp}  ({frames.shape})")
        all_rows.append(frames)

    # stack all episodes vertically into one ALL5 video (pad to min T)
    minT = min(r.shape[0] for r in all_rows)
    W = min(r.shape[2] for r in all_rows)
    combined = []
    for t in range(minT):
        col = all_rows[0][t][:, :W]
        for r in all_rows[1:]:
            col = np.concatenate([col, np.zeros((4, W, 3), np.uint8) + 60,
                                  r[t][:, :W]], axis=0)
        combined.append(col)
    combined = np.stack(combined, 0)
    allp = os.path.join(OUT, "ALL5_wrist_compare.mp4")
    imageio.mimsave(allp, combined, fps=FPS)
    print(f"wrote {allp}  ({combined.shape})")


if __name__ == "__main__":
    main()
