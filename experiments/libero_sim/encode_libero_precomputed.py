#!/usr/bin/env python
"""Encode LIBERO context chunks into the trainer's `precomputed/` schema.

This is the LIBERO analogue of DROID's Rung 4 (build_dataset_resumable.py:459-658).
The LIBERO build (build_libero_context.py) writes raw per-chunk npz with 3 camera
views; this script turns each chunk into the exact 4-file precomputed layout the
v2v IC-LoRA + action-cond trainer consumes:

    <out_root>/precomputed/
      latents/videos/<cid>_target.pt            VAE latents of the GT 2x2-tiled video
      reference_latents/videos/<cid>_target.pt  VAE latents of the context-prior tiled video
      conditions/videos/<cid>_target.pt         Gemma text embeddings of the task caption
      actions/videos/<cid>_target.pt            {"latents": (CHUNK_LEN, 8)} joints+gripper

Tiling matches DROID EXACTLY (1024x576, 2x2, three 512x288 tiles + one blank) so
the same model/config trains on both datasets and they can be mixed later:

      +-----------------+-----------------+
      | agentview (GT)  | frontview (GT)  |     reference uses the *_context priors
      +-----------------+-----------------+     instead of the GT *_frames
      |     (blank)     |  wrist (GT)     |
      +-----------------+-----------------+

VAE encode + caption embed are delegated to the trainer's process_dataset.py
(same call DROID makes), so the latent space + text embeddings match the data
the model was trained on. Action tensors are built directly (no model).

Usage (needs a big GPU for process_dataset.py's VAE+Gemma):
    python encode_libero_precomputed.py \
        --src_root /storage/scratch1/8/lwang831/libero90_context \
        --out_root /storage/scratch1/8/lwang831/libero90_precomputed_smoke \
        --max_chunks 4
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Reuse DROID's proven tiling/encode/atomic helpers — single source of truth for
# the precomputed geometry so DROID and LIBERO land byte-compatible layouts.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_DROID = _REPO / "experiments" / "droid_action_cond"
sys.path.insert(0, str(_DROID))
from build_dataset_resumable import (  # noqa: E402
    CHUNK_LEN,
    _output_looks_valid,
    atomic_replace,
    clean_partial_outputs,
    mark_done,
    precomputed_paths,
    write_mp4_atomic,
    write_torch_atomic,
)

# 2-view tile geometry: agentview (left) | wrist (right), native 256x256 each,
# horizontally concatenated -> 256x512 (H x W). Replaces the inherited DROID
# 3-view 2x2 (1024x576) layout; deliberately NO upscale (raw native resolution).
NEW_H, NEW_W = 256, 512


def _hconcat2(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Horizontally concat two (F,H,W,C) uint8 view stacks -> (F,H,2W,C). No upscale."""
    if left.shape != right.shape:
        raise ValueError(f"view shape mismatch: {left.shape} vs {right.shape}")
    return np.concatenate([left, right], axis=2)

DEFAULT_MODEL_PATH = "/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
DEFAULT_TEXT_ENCODER_PATH = "/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/gemma-3-12b-it-qat-q4_0-unquantized"
PROCESS_DATASET = _REPO / "packages" / "ltx-trainer" / "scripts" / "process_dataset.py"


def caption_for(task: str) -> str:
    task = (task or "").strip().rstrip(".")
    action = f" performing the task: {task}" if task else ""
    return (
        f"A Franka robot arm with a parallel-jaw gripper{action}. The agentview and "
        "wrist camera views are shown side by side."
    )


def cid_for(chunk_npz: Path) -> str:
    # <out_root>/<ep_id>/chunk_NN/data.npz  ->  <ep_id>_chunk_NN
    return f"{chunk_npz.parent.parent.name}_{chunk_npz.parent.name}"


def enumerate_chunks(src_root: Path) -> list[Path]:
    return [Path(p) for p in sorted(glob.glob(str(src_root / "*" / "chunk_*" / "data.npz")))]


def build_chunk_videos(npz_path: Path, cid: str, staging: Path) -> tuple[Path, Path, np.ndarray, str]:
    """Tile one LIBERO chunk into target.mp4 + reference.mp4; return (tgt, ref, actions, caption)."""
    staging.mkdir(parents=True, exist_ok=True)
    d = dict(np.load(npz_path, allow_pickle=True))

    # 2-view layout: agentview (left) | wrist (right) at native 256x256 -> 256x512.
    target = _hconcat2(d["agentview_frames"], d["wrist_frames"])
    reference = _hconcat2(d["agentview_context_robot"], d["wrist_context_fused"])

    tgt = staging / f"{cid}_target.mp4"
    ref = staging / f"{cid}_reference.mp4"
    write_mp4_atomic(target, tgt)
    write_mp4_atomic(reference, ref)

    # Action vector: 7 joint angles + 1 gripper = 8D (matches DROID action_dim).
    actions = np.concatenate(
        [d["joints"].astype(np.float32),
         d["gripper"].reshape(-1, 1).astype(np.float32)],
        axis=-1,
    )

    # Caption from the episode's info.json task string (falls back to generic).
    task = ""
    info = npz_path.parent.parent / "chunk_00" / "info.json"
    if not info.exists():
        info = npz_path.parent / "info.json"
    if info.exists():
        try:
            task = json.loads(info.read_text()).get("task", "")
        except Exception:
            pass
    return tgt, ref, actions, caption_for(task)


def process_batch(batch: list[tuple[str, Path]], out_root: Path,
                  model_path: str, text_encoder_path: str,
                  load_text_encoder_in_8bit: bool) -> tuple[int, int]:
    """Encode one batch of (cid, npz_path) into precomputed/. Mirrors DROID's process_batch."""
    if not batch:
        return 0, 0
    batch_id = f"libbatch_{os.getpid()}_{len(batch)}"
    staging = out_root / "staging" / batch_id
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    entries = []
    actions_map: dict[str, np.ndarray] = {}
    for cid, npz_path in batch:
        try:
            tgt, ref, actions, cap = build_chunk_videos(npz_path, cid, staging / "videos")
            entries.append({"media_path": str(tgt), "reference_path": str(ref), "caption": cap})
            actions_map[cid] = actions
        except Exception as exc:
            print(f"  [WARN] {cid} pre-stage failed: {exc!r}", file=sys.stderr)

    if not entries:
        shutil.rmtree(staging, ignore_errors=True)
        return 0, len(batch)

    dataset_json = staging / "dataset.json"
    dataset_json.write_text(json.dumps(entries, indent=2))

    pre_out = staging / "precomputed"
    cmd = [
        sys.executable, "-u", str(PROCESS_DATASET), str(dataset_json),
        "--resolution-buckets", f"{NEW_W}x{NEW_H}x{CHUNK_LEN}",
        "--model-path", model_path,
        "--text-encoder-path", text_encoder_path,
        "--reference-column", "reference_path",
        "--output-dir", str(pre_out),
        "--vae-tiling",
    ]
    if load_text_encoder_in_8bit:
        cmd.append("--load-text-encoder-in-8bit")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"  [ERROR] process_dataset.py rc={rc}; batch aborted", file=sys.stderr)
        shutil.rmtree(staging, ignore_errors=True)
        return 0, len(batch)

    src_conditions = staging / "videos"  # conditions land beside the source MP4s
    n_committed = n_failed = 0
    for cid in actions_map:
        try:
            staged = {
                "latents": pre_out / "latents" / "videos" / f"{cid}_target.pt",
                "reference_latents": pre_out / "reference_latents" / "videos" / f"{cid}_target.pt",
                "conditions": src_conditions / f"{cid}_target.pt",
            }
            for k, p in staged.items():
                if not _output_looks_valid(p):
                    raise RuntimeError(f"staged {k} missing/corrupt: {p}")
            clean_partial_outputs(out_root, cid)
            final = precomputed_paths(out_root, cid)
            for k in ("latents", "reference_latents", "conditions"):
                atomic_replace(staged[k], final[k])
            write_torch_atomic({"latents": torch.from_numpy(actions_map[cid])}, final["actions"])
            # Validate ONLY what this script commits. `cameras` is produced by the
            # separate build_libero_cameras.py pass (same as the GT set), so it is
            # absent here and must not gate the commit.
            for k in ("latents", "reference_latents", "conditions", "actions"):
                if not _output_looks_valid(final[k]):
                    raise RuntimeError(f"post-commit {k} unreadable: {final[k]}")
            mark_done(out_root, cid)
            n_committed += 1
        except Exception as exc:
            print(f"  [WARN] {cid} commit failed: {exc!r}", file=sys.stderr)
            clean_partial_outputs(out_root, cid)
            n_failed += 1

    shutil.rmtree(staging, ignore_errors=True)
    return n_committed, n_failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", type=Path, default=Path("/storage/scratch1/8/lwang831/libero90_context"))
    ap.add_argument("--out_root", type=Path, default=Path("/storage/scratch1/8/lwang831/libero90_precomputed_smoke"))
    ap.add_argument("--max_chunks", type=int, default=0, help="0 = all enumerated chunks")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--shard", type=str, default="0/1",
                    help='"i/N" worker shard. Disjoint stable-hash partition '
                         "(sha1(cid) %% N == i); auto-inferred from SLURM_ARRAY_TASK_ID/COUNT.")
    ap.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--text_encoder_path", default=DEFAULT_TEXT_ENCODER_PATH)
    ap.add_argument("--load_text_encoder_in_8bit", action="store_true")
    ap.add_argument("--stage_only", action="store_true",
                    help="Build target/reference mp4 + actions + dataset.json only (NO model); for CPU smoke-test.")
    args = ap.parse_args()

    args.out_root = args.out_root.resolve()
    (args.out_root / "_done").mkdir(parents=True, exist_ok=True)
    (args.out_root / "staging").mkdir(parents=True, exist_ok=True)

    # Shard identity: explicit --shard "i/N", else inferred from the Slurm array.
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    if args.shard == "0/1" and os.environ.get("SLURM_ARRAY_TASK_COUNT"):
        shard_i = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        shard_n = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
    if not (0 <= shard_i < shard_n):
        print(f"[error] bad shard {shard_i}/{shard_n}", file=sys.stderr)
        return 1

    chunks = enumerate_chunks(args.src_root)
    if not chunks:
        print(f"[error] no chunks under {args.src_root}", file=sys.stderr)
        return 1
    if args.max_chunks > 0:
        chunks = chunks[: args.max_chunks]
    work = [(cid_for(c), c) for c in chunks]
    # Disjoint stable-hash shard (matches DROID's shard_filter_universe): each cid
    # lands on exactly one shard, so an N-way array never double-encodes a chunk.
    if shard_n > 1:
        work = [
            (cid, c) for cid, c in work
            if int.from_bytes(hashlib.sha1(cid.encode()).digest()[:8], "big") % shard_n == shard_i
        ]
    n_shard = len(work)
    work = [(cid, c) for cid, c in work if not (args.out_root / "_done" / cid).exists()]
    print(f"[encode] shard {shard_i}/{shard_n} | {len(chunks)} total chunks | "
          f"{n_shard} in shard | {len(work)} pending -> {args.out_root}", flush=True)

    if args.stage_only:
        stage = args.out_root / "staging" / "stage_only"
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        entries = []
        for cid, npz_path in work:
            tgt, ref, actions, cap = build_chunk_videos(npz_path, cid, stage / "videos")
            entries.append({"media_path": str(tgt), "reference_path": str(ref),
                            "caption": cap, "cid": cid, "action_shape": list(actions.shape)})
            print(f"  staged {cid}: target={tgt.name} ref={ref.name} actions={actions.shape} "
                  f"a_range=[{actions.min():.3f},{actions.max():.3f}]")
        (stage / "dataset.json").write_text(json.dumps(entries, indent=2))
        print(f"[stage_only] wrote {len(entries)} entries -> {stage/'dataset.json'}")
        return 0

    committed = failed = 0
    t0 = time.time()
    for i in range(0, len(work), args.batch_size):
        nb_c, nb_f = process_batch(
            work[i : i + args.batch_size], args.out_root,
            args.model_path, args.text_encoder_path, args.load_text_encoder_in_8bit)
        committed += nb_c
        failed += nb_f
        print(f"  batch {i//args.batch_size+1}: +{nb_c} (-{nb_f}) | total {committed}/{len(work)} "
              f"| {time.time()-t0:.0f}s", flush=True)
    print(f"\nDONE. committed={committed} failed={failed} elapsed={time.time()-t0:.0f}s")
    # Success == nothing failed. An empty work list (everything already encoded in a
    # prior run -> 0 pending, committed=0) is a legitimate no-op, NOT a failure; only
    # a real per-chunk failure should propagate a non-zero exit.
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
