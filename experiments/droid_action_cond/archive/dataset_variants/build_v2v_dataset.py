"""Assemble one or more chunks into the trainer's v2v IC-LoRA dataset layout.

Inputs:
  outputs/context_chunks_wrist_new/<ep>/chunk_NN/data.npz
    ext1_frames, ext2_frames, wrist_frames       — GT video per view (33, 180, 320, 3) uint8
    ext1_context, ext2_context, wrist_context    — rendered context per view (same shape)
    cmd_joint_position, cmd_gripper_position     — (33, 7), (33,)

Outputs (under --out_root):
  videos/<chunk_id>_target.mp4        — 1024x576x33 tiled GT video
  videos/<chunk_id>_reference.mp4     — 1024x576x33 tiled rendered context
  dataset.json                        — [{media_path, reference_path, caption}, ...]
  actions/videos/<chunk_id>.pt        — {"latents": (33, 8) float32}

After this script, run:
  python packages/ltx-trainer/scripts/process_dataset.py <out_root>/dataset.json \
      --resolution-buckets "1024x576x33" \
      --model-path <LTX-2.3 .safetensors> \
      --text-encoder-path <gemma dir> \
      --reference-column reference_path \
      --output-dir <out_root>/precomputed

That populates latents/, reference_latents/, conditions/ in the trainer-expected layout.
The conditions/ fix-up move (process_dataset stages text embeddings beside source MP4s) is
handled by --fixup-after at the end of this script when invoked with --do-precompute.

Pick chunks with --chunks "ep1/chunk_NN,ep2/chunk_MM" or use --all to take everything under
the chunk-source root.
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
CHUNK_ROOT = Path(os.environ.get("CHUNK_SRC_ROOT", _HERE / "outputs" / "context_chunks_wrist_new"))

TILE_H, TILE_W = 288, 512      # per-tile pixel size (after upscale)
FULL_H, FULL_W = 576, 1024     # 2x2 tiled
NUM_FRAMES = 33
FPS = 15

DEFAULT_CAPTION = (
    "A Franka robot arm with a Robotiq parallel-jaw gripper manipulates objects on a "
    "tabletop. Two exterior camera views and a wrist camera view are shown in a 2x2 grid."
)


def _upscale_lanczos(frames: np.ndarray, h_out: int, w_out: int) -> np.ndarray:
    """Per-frame Lanczos upscale. frames: (T, H, W, 3) uint8 -> (T, h_out, w_out, 3) uint8."""
    out = np.empty((frames.shape[0], h_out, w_out, 3), dtype=np.uint8)
    for t in range(frames.shape[0]):
        out[t] = cv2.resize(frames[t], (w_out, h_out), interpolation=cv2.INTER_LANCZOS4)
    return out


def _tile_2x2(top_left: np.ndarray, top_right: np.ndarray, bottom_right: np.ndarray) -> np.ndarray:
    """[ext1 | ext2 / blank | wrist] 2x2 tile. Each input (T, TILE_H, TILE_W, 3)."""
    t = top_left.shape[0]
    out = np.zeros((t, FULL_H, FULL_W, 3), dtype=np.uint8)
    out[:, :TILE_H, :TILE_W] = top_left
    out[:, :TILE_H, TILE_W:] = top_right
    # out[:, TILE_H:, :TILE_W] = 0  # blank quadrant (already zero)
    out[:, TILE_H:, TILE_W:] = bottom_right
    return out


def chunk_id_for(ep: str, chunk_dir: str) -> str:
    """ep = 'CLVR+13759f6e+...', chunk_dir = 'chunk_03' -> 'CLVR_13759f6e_..._c03'."""
    ep_short = ep.replace("+", "_")
    chunk_short = chunk_dir.replace("chunk_", "c")
    return f"{ep_short}_{chunk_short}"


def build_one_chunk(chunk_path: Path, out_root: Path, caption: str) -> dict:
    data = dict(np.load(chunk_path / "data.npz"))
    ep = chunk_path.parent.name
    cid = chunk_id_for(ep, chunk_path.name)

    # 1. Tile target + reference videos.
    ext1_gt = _upscale_lanczos(data["ext1_frames"], TILE_H, TILE_W)
    ext2_gt = _upscale_lanczos(data["ext2_frames"], TILE_H, TILE_W)
    wrist_gt = _upscale_lanczos(data["wrist_frames"], TILE_H, TILE_W)
    target_tiled = _tile_2x2(ext1_gt, ext2_gt, wrist_gt)

    ext1_ctx = _upscale_lanczos(data["ext1_context"], TILE_H, TILE_W)
    ext2_ctx = _upscale_lanczos(data["ext2_context"], TILE_H, TILE_W)
    wrist_ctx = _upscale_lanczos(data["wrist_context"], TILE_H, TILE_W)
    reference_tiled = _tile_2x2(ext1_ctx, ext2_ctx, wrist_ctx)

    videos_dir = out_root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    target_path = videos_dir / f"{cid}_target.mp4"
    reference_path = videos_dir / f"{cid}_reference.mp4"
    iio.imwrite(target_path, target_tiled, fps=FPS, macro_block_size=1)
    iio.imwrite(reference_path, reference_tiled, fps=FPS, macro_block_size=1)

    # 2. Dump actions: 7 cmd_joints + 1 cmd_gripper -> (33, 8) float32.
    joints = data["cmd_joint_position"].astype(np.float32)  # (33, 7)
    grip = data["cmd_gripper_position"].reshape(-1, 1).astype(np.float32)  # (33, 1)
    actions = np.concatenate([joints, grip], axis=-1)  # (33, 8)
    # Filename must match the latent/reference/conditions stem so PrecomputedDataset's
    # path-set matching picks it up. process_dataset.py derives that stem from the source
    # MP4 (here: <cid>_target.mp4 -> <cid>_target.pt).
    actions_dir = out_root / "precomputed" / "actions" / "videos"
    actions_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"latents": torch.from_numpy(actions)}, actions_dir / f"{target_path.stem}.pt")

    return {
        "chunk_id": cid,
        "media_path": str(target_path),
        "reference_path": str(reference_path),
        "caption": caption,
        "action_shape": list(actions.shape),
        "joint_range": [float(joints.min()), float(joints.max())],
        "gripper_range": [float(grip.min()), float(grip.max())],
    }


def fixup_conditions(out_root: Path) -> int:
    """process_dataset.py writes the text-embedding .pt next to the source MP4 instead of
    under precomputed/conditions/videos/. Move them after preprocessing so the trainer's
    PrecomputedDataset can find them.

    Returns the number of files moved.
    """
    src_dir = out_root / "videos"
    dst_dir = out_root / "precomputed" / "conditions" / "videos"
    dst_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for p in src_dir.glob("*_target.pt"):
        # Strip the "_target" suffix so the condition file matches the latent file name.
        new_name = p.name.replace("_target.pt", "_target.pt")
        shutil.move(str(p), str(dst_dir / new_name))
        moved += 1
    # Also handle the case where process_dataset.py names them <stem>.pt without _target.
    for p in src_dir.glob("*.pt"):
        shutil.move(str(p), str(dst_dir / p.name))
        moved += 1
    return moved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunks",
        type=str,
        default="",
        help='Comma-separated list of "<ep_name>/chunk_NN". Empty + --all picks everything.',
    )
    parser.add_argument("--all", action="store_true", help="Build all chunks under CHUNK_SRC_ROOT.")
    parser.add_argument(
        "--episode",
        type=str,
        default="",
        help='Shortcut: build every chunk under one episode dir, e.g. "CLVR+13759f6e+...".',
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=_HERE / "outputs" / "v2v_dataset_single",
        help="Output dataset directory.",
    )
    parser.add_argument("--caption", type=str, default=DEFAULT_CAPTION)
    parser.add_argument(
        "--do_precompute",
        action="store_true",
        help="After tiling, invoke process_dataset.py to encode latents/conditions/refs.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/gemma-3-12b-it-qat-q4_0-unquantized",
    )
    args = parser.parse_args()

    args.out_root = args.out_root.resolve()  # absolute; process_dataset.py expects unambiguous paths
    args.out_root.mkdir(parents=True, exist_ok=True)

    # Resolve chunk list.
    if args.episode:
        ep_dir = CHUNK_ROOT / args.episode
        chunk_list = sorted(
            f"{args.episode}/{d.name}"
            for d in ep_dir.iterdir()
            if d.is_dir() and (d / "data.npz").exists()
        )
    elif args.chunks:
        chunk_list = [s.strip() for s in args.chunks.split(",") if s.strip()]
    elif args.all:
        chunk_list = []
        for ep_dir in sorted(CHUNK_ROOT.iterdir()):
            if not ep_dir.is_dir():
                continue
            for chunk_dir in sorted(ep_dir.iterdir()):
                if chunk_dir.is_dir() and (chunk_dir / "data.npz").exists():
                    chunk_list.append(f"{ep_dir.name}/{chunk_dir.name}")
    else:
        print("Provide --chunks 'ep/chunk_NN,...', --episode <ep>, or --all", file=sys.stderr)
        return 1

    print(f"Building {len(chunk_list)} chunk(s) -> {args.out_root}")
    t0 = time.time()

    entries = []
    for c in chunk_list:
        chunk_path = CHUNK_ROOT / c
        if not (chunk_path / "data.npz").exists():
            print(f"  SKIP {c} (no data.npz)")
            continue
        entry = build_one_chunk(chunk_path, args.out_root, args.caption)
        print(f"  {c} -> {entry['chunk_id']}  "
              f"target={Path(entry['media_path']).name}  "
              f"action_shape={entry['action_shape']}  "
              f"gripper_range={entry['gripper_range']}")
        entries.append({
            "media_path": entry["media_path"],
            "reference_path": entry["reference_path"],
            "caption": entry["caption"],
        })

    dataset_json = args.out_root / "dataset.json"
    dataset_json.write_text(json.dumps(entries, indent=2))
    t_tile_done = time.time()
    print(f"\nWrote {dataset_json}  ({len(entries)} entries)")
    print(f"[TIMING] tile+action+json: {t_tile_done - t0:.1f} s for {len(entries)} chunk(s) "
          f"({(t_tile_done - t0)/max(len(entries),1):.1f} s/chunk avg)")

    if args.do_precompute:
        print("\nInvoking process_dataset.py ...")
        cmd = [
            sys.executable,
            "-u",
            str(_HERE.parent.parent / "packages" / "ltx-trainer" / "scripts" / "process_dataset.py"),
            str(dataset_json),
            "--resolution-buckets",
            f"{FULL_W}x{FULL_H}x{NUM_FRAMES}",
            "--model-path",
            args.model_path,
            "--text-encoder-path",
            args.text_encoder_path,
            "--reference-column",
            "reference_path",
            "--output-dir",
            str(args.out_root / "precomputed"),
            "--vae-tiling",
        ]
        print(" ".join(cmd))
        t_pre_start = time.time()
        rc = subprocess.run(cmd).returncode
        t_pre_done = time.time()
        if rc != 0:
            print(f"process_dataset.py exited with rc={rc}", file=sys.stderr)
            return rc
        moved = fixup_conditions(args.out_root)
        print(f"\nMoved {moved} text-embedding .pt files into precomputed/conditions/videos/")
        print(f"[TIMING] process_dataset.py: {t_pre_done - t_pre_start:.1f} s for {len(entries)} chunk(s) "
              f"({(t_pre_done - t_pre_start)/max(len(entries),1):.1f} s/chunk avg, includes one-time model load)")
        print(f"[TIMING] total wall-clock: {time.time() - t0:.1f} s")

    print("\nDONE. Final dataset layout:")
    for p in sorted(args.out_root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(args.out_root)
            print(f"  {rel}  ({p.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
