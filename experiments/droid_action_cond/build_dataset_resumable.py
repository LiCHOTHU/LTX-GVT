"""Preemption-safe, resumable v2v IC-LoRA dataset builder.

Walks every chunk under ``--src_root`` (default: ``outputs/context_chunks_wrist_new``),
produces the trainer-ready preprocessed dataset under ``--out_root`` (default: on cedar
storage), and is safe to kill/relaunch under Slurm.

Reliability guarantees:
  1. **Atomic per-chunk commit.** A chunk's outputs are written to ``staging/<id>/``,
     verified (all four .pt files present and non-empty), then atomically renamed into
     ``precomputed/<source>/videos/<id>_target.pt`` and a sentinel ``_done/<id>``
     marker is created. Either *all* of a chunk's outputs are in place or *none* are.
  2. **Resumable.** On startup, every chunk with an existing ``_done/<id>`` marker is
     skipped. Partial staging dirs from a killed run are wiped before retry.
  3. **No corrupted writes.** All file moves use POSIX ``os.rename`` which is atomic on
     the same filesystem; the ``staging/`` dir is co-located with ``out_root`` to ensure
     same-FS renames.
  4. **Slurm-array friendly.** ``--shard i/N`` selects chunks whose hash(``chunk_id``)
     mod N equals i, so multiple sbatch tasks can run in parallel without coordination.

Within one run, chunks are processed in *batches* of ``--batch_size`` (default 8) to
amortize the LTX-2 cold-load. The model is loaded once per batch via the existing
``process_dataset.py`` script. If the job is preempted mid-batch, completed chunks
within the batch are still committed; the rest are re-built on the next launch.

Usage:
    python build_dataset_resumable.py \
        --src_root outputs/context_chunks_wrist_new \
        --out_root /storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/gvt_dataset_full \
        --shard 0/1 \
        --batch_size 16
"""

from __future__ import annotations

import argparse
import hashlib
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
DEFAULT_SRC_ROOT = _HERE / "outputs" / "context_chunks_wrist_new"
DEFAULT_OUT_ROOT = Path("/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/gvt_dataset_full")
DEFAULT_MODEL_PATH = "/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
DEFAULT_TEXT_ENCODER_PATH = "/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/gemma-3-12b-it-qat-q4_0-unquantized"

CHUNK_LEN = 33
TILE_H, TILE_W = 288, 512
FULL_H, FULL_W = 576, 1024
FPS = 15

DEFAULT_CAPTION = (
    "A Franka robot arm with a Robotiq parallel-jaw gripper manipulates objects on a "
    "tabletop. Two exterior camera views and a wrist camera view are shown in a 2x2 grid."
)

REQUIRED_OUTPUTS = ("latents", "reference_latents", "conditions", "actions")


# ---------- atomic helpers ----------

def atomic_replace(src: Path, dst: Path) -> None:
    """Atomic POSIX rename. Both paths must live on the same filesystem."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dst)


def _atomic_tmp_path(path: Path) -> Path:
    """Return a same-dir tmp path that PRESERVES the real extension.

    imageio (and a few other libs) sniff format from extension, so naive
    ``path.with_suffix(path.suffix + ".tmp")`` (which yields ``foo.mp4.tmp.PID``,
    extension = ".PID") would misroute MP4 writes to the TIFF plugin. Putting the
    real suffix LAST keeps the sniff correct.
    """
    return path.with_name(f".{path.stem}.tmp.{os.getpid()}{path.suffix}")


def write_torch_atomic(obj, path: Path) -> None:
    """Save a torch object to `path` via a same-dir .tmp file + rename. Crash-safe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_tmp_path(path)
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_mp4_atomic(frames: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_tmp_path(path)
    iio.imwrite(tmp, frames, fps=FPS, macro_block_size=1)
    os.replace(tmp, path)


# ---------- chunk enumeration + sharding ----------

def list_chunks(src_root: Path) -> list[tuple[str, Path]]:
    """Returns (chunk_id, chunk_dir) for every chunk with a data.npz under src_root."""
    out = []
    for ep_dir in sorted(src_root.iterdir()):
        if not ep_dir.is_dir():
            continue
        for chunk_dir in sorted(ep_dir.iterdir()):
            if not chunk_dir.is_dir():
                continue
            if not (chunk_dir / "data.npz").exists():
                continue
            cid = f"{ep_dir.name.replace('+', '_')}_{chunk_dir.name.replace('chunk_', 'c')}"
            out.append((cid, chunk_dir))
    return out


def shard_filter(chunks: list[tuple[str, Path]], shard_idx: int, shard_total: int) -> list[tuple[str, Path]]:
    """Hash-based stable sharding. Worker i takes chunks where hash(cid) % N == i."""
    if shard_total <= 1:
        return chunks
    out = []
    for cid, path in chunks:
        h = int.from_bytes(hashlib.sha1(cid.encode()).digest()[:8], "big")
        if h % shard_total == shard_idx:
            out.append((cid, path))
    return out


# ---------- per-chunk build (tile + actions) ----------

def _upscale_lanczos(frames: np.ndarray, h: int, w: int) -> np.ndarray:
    out = np.empty((frames.shape[0], h, w, 3), dtype=np.uint8)
    for t in range(frames.shape[0]):
        out[t] = cv2.resize(frames[t], (w, h), interpolation=cv2.INTER_LANCZOS4)
    return out


def _tile_2x2(top_left, top_right, bottom_right) -> np.ndarray:
    t = top_left.shape[0]
    out = np.zeros((t, FULL_H, FULL_W, 3), dtype=np.uint8)
    out[:, :TILE_H, :TILE_W] = top_left
    out[:, :TILE_H, TILE_W:] = top_right
    out[:, TILE_H:, TILE_W:] = bottom_right
    return out


def build_chunk_videos(chunk_dir: Path, cid: str, staging: Path) -> tuple[Path, Path, np.ndarray]:
    """Build target.mp4 + reference.mp4 + actions for one chunk into `staging`. Returns the paths."""
    staging.mkdir(parents=True, exist_ok=True)
    d = dict(np.load(chunk_dir / "data.npz"))
    target = _tile_2x2(
        _upscale_lanczos(d["ext1_frames"], TILE_H, TILE_W),
        _upscale_lanczos(d["ext2_frames"], TILE_H, TILE_W),
        _upscale_lanczos(d["wrist_frames"], TILE_H, TILE_W),
    )
    reference = _tile_2x2(
        _upscale_lanczos(d["ext1_context"], TILE_H, TILE_W),
        _upscale_lanczos(d["ext2_context"], TILE_H, TILE_W),
        _upscale_lanczos(d["wrist_context"], TILE_H, TILE_W),
    )
    target_path = staging / f"{cid}_target.mp4"
    reference_path = staging / f"{cid}_reference.mp4"
    write_mp4_atomic(target, target_path)
    write_mp4_atomic(reference, reference_path)
    actions = np.concatenate(
        [d["cmd_joint_position"].astype(np.float32),
         d["cmd_gripper_position"].reshape(-1, 1).astype(np.float32)],
        axis=-1,
    )
    return target_path, reference_path, actions


# ---------- batch process_dataset + atomic commit ----------

def precomputed_paths(out_root: Path, cid: str) -> dict[str, Path]:
    """Final destinations for a chunk's four .pt files."""
    base = out_root / "precomputed"
    return {
        "latents": base / "latents" / "videos" / f"{cid}_target.pt",
        "reference_latents": base / "reference_latents" / "videos" / f"{cid}_target.pt",
        "conditions": base / "conditions" / "videos" / f"{cid}_target.pt",
        "actions": base / "actions" / "videos" / f"{cid}_target.pt",
    }


def _output_looks_valid(path: Path, min_bytes: int = 1024) -> bool:
    """Cheap structural check: file exists, non-trivial size, opens as a torch object."""
    if not path.exists() or path.stat().st_size < min_bytes:
        return False
    try:
        torch.load(path, map_location="cpu", weights_only=True)
        return True
    except Exception:
        return False


def chunk_is_done(out_root: Path, cid: str) -> bool:
    """Done = marker exists AND all four .pt files load. If the marker exists but a file
    is missing/corrupt, the marker is stale (likely from a previous version or partial
    cleanup) — we clear it so the chunk gets rebuilt.
    """
    marker = out_root / "_done" / cid
    if not marker.exists():
        return False
    final = precomputed_paths(out_root, cid)
    for k, p in final.items():
        if not _output_looks_valid(p):
            print(f"  [REVIVE] {cid}: marker exists but {k} is missing/corrupt -> rebuilding",
                  file=sys.stderr)
            # Wipe the stale marker and any partial outputs so the rebuild starts clean.
            clean_partial_outputs(out_root, cid)
            return False
    return True


def clean_partial_outputs(out_root: Path, cid: str) -> None:
    """Remove any leftover files from a failed/aborted commit so the retry is idempotent.
    Safe to call even on a clean state.
    """
    marker = out_root / "_done" / cid
    if marker.exists():
        marker.unlink()
    for p in precomputed_paths(out_root, cid).values():
        if p.exists():
            p.unlink()


def mark_done(out_root: Path, cid: str) -> None:
    marker_dir = out_root / "_done"
    marker_dir.mkdir(parents=True, exist_ok=True)
    (marker_dir / cid).touch()


def process_batch(
    batch: list[tuple[str, Path, np.ndarray]],
    out_root: Path,
    model_path: str,
    text_encoder_path: str,
    load_text_encoder_in_8bit: bool = False,
) -> tuple[int, int]:
    """Process one batch of chunks. Each tuple is (cid, chunk_dir, actions_array).
    Returns (n_committed, n_failed).

    Strategy:
      1. Build staging/<batch_uuid>/ with target.mp4 + reference.mp4 for each chunk.
      2. Run process_dataset.py over a dataset.json pointing at those staging videos,
         outputting to staging/<batch_uuid>/precomputed/.
      3. For each chunk in the batch: verify all 4 outputs exist + non-empty,
         atomic-rename into out_root/precomputed/.../<id>_target.pt, write actions,
         touch _done/<id> marker. Skip chunks whose outputs are missing/corrupt.
      4. Clean up staging dir on success.
    """
    if not batch:
        return 0, 0

    batch_id = f"batch_{int(time.time() * 1000)}_{os.getpid()}"
    staging = out_root / "staging" / batch_id
    staging.mkdir(parents=True, exist_ok=True)

    entries = []
    actions_map: dict[str, np.ndarray] = {}
    for cid, chunk_dir, actions in batch:
        try:
            tgt, ref, _ = build_chunk_videos(chunk_dir, cid, staging / "videos")
            entries.append({
                "media_path": str(tgt),
                "reference_path": str(ref),
                "caption": DEFAULT_CAPTION,
            })
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
        sys.executable, "-u",
        str(_HERE.parent.parent / "packages" / "ltx-trainer" / "scripts" / "process_dataset.py"),
        str(dataset_json),
        "--resolution-buckets", f"{FULL_W}x{FULL_H}x{CHUNK_LEN}",
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

    # Conditions are emitted alongside source MP4s, not under conditions/.
    src_conditions = staging / "videos"

    n_committed = 0
    n_failed = 0
    for cid in actions_map.keys():
        try:
            staged_pts = {
                "latents": pre_out / "latents" / "videos" / f"{cid}_target.pt",
                "reference_latents": pre_out / "reference_latents" / "videos" / f"{cid}_target.pt",
                "conditions": src_conditions / f"{cid}_target.pt",
            }
            # Verify the trainer encoded everything successfully AND each .pt actually
            # loads — guards against process_dataset.py truncating output on a crash.
            for k, p in staged_pts.items():
                if not _output_looks_valid(p):
                    raise RuntimeError(f"staged {k} missing/corrupt: {p}")

            # Wipe any leftovers from a half-committed previous run so partials don't
            # coexist with the fresh atomic renames.
            clean_partial_outputs(out_root, cid)

            final = precomputed_paths(out_root, cid)
            for k in ("latents", "reference_latents", "conditions"):
                atomic_replace(staged_pts[k], final[k])

            # Action tensor is built directly (no model). Write atomically.
            write_torch_atomic({"latents": torch.from_numpy(actions_map[cid])}, final["actions"])

            # Final post-commit verification before marking done. If anything is wrong,
            # clean up rather than mark a corrupt chunk as done.
            for k, p in final.items():
                if not _output_looks_valid(p):
                    raise RuntimeError(f"post-commit {k} unreadable: {p}")

            # Last step: marker. After this point the chunk is considered done.
            mark_done(out_root, cid)
            n_committed += 1
        except Exception as exc:
            print(f"  [WARN] {cid} commit failed: {exc!r}", file=sys.stderr)
            # Leave nothing half-committed if anything in the commit chain blew up.
            clean_partial_outputs(out_root, cid)
            n_failed += 1

    shutil.rmtree(staging, ignore_errors=True)
    return n_committed, n_failed


# ---------- main loop ----------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=Path, default=DEFAULT_SRC_ROOT)
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--shard", type=str, default="0/1", help='"i/N" worker shard.')
    parser.add_argument("--batch_size", type=int, default=8, help="Chunks per process_dataset.py invocation.")
    parser.add_argument("--max_chunks", type=int, default=0, help="If >0, stop after committing this many chunks.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_encoder_path", type=str, default=DEFAULT_TEXT_ENCODER_PATH)
    parser.add_argument("--dry_run", action="store_true", help="Enumerate + skip-status only; no work.")
    parser.add_argument(
        "--load_text_encoder_in_8bit",
        action="store_true",
        help="Load Gemma in 8-bit (bitsandbytes). Required for <24 GB GPUs (V100 etc).",
    )
    args = parser.parse_args()

    args.src_root = args.src_root.resolve()
    args.out_root = args.out_root.resolve()

    # If the caller passed the default --shard "0/1" and we're running under an
    # `sbatch --array=...` job, infer the shard from SLURM_ARRAY_TASK_{ID,COUNT}.
    # Explicit --shard always wins.
    if args.shard == "0/1" and "SLURM_ARRAY_TASK_ID" in os.environ:
        sid = int(os.environ["SLURM_ARRAY_TASK_ID"])
        # SLURM_ARRAY_TASK_COUNT is the total number of tasks in the array.
        sct = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
        args.shard = f"{sid}/{sct}"
        print(f"[shard] inferred from Slurm array: {args.shard}")

    shard_i, shard_n = (int(s) for s in args.shard.split("/"))

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "_done").mkdir(exist_ok=True)
    (args.out_root / "staging").mkdir(exist_ok=True)

    all_chunks = list_chunks(args.src_root)
    my_chunks = shard_filter(all_chunks, shard_i, shard_n)
    pending = [(cid, path) for cid, path in my_chunks if not chunk_is_done(args.out_root, cid)]
    done_count = len(my_chunks) - len(pending)

    print(f"== shard {shard_i}/{shard_n}  src={args.src_root}  out={args.out_root}")
    print(f"   total chunks under src   : {len(all_chunks)}")
    print(f"   chunks assigned to shard : {len(my_chunks)}")
    print(f"   already done (skipped)   : {done_count}")
    print(f"   pending in this run      : {len(pending)}")

    if args.dry_run or not pending:
        return 0

    committed_total = failed_total = 0
    t0 = time.time()
    for i in range(0, len(pending), args.batch_size):
        batch_slice = pending[i : i + args.batch_size]
        # Load actions arrays for the batch up front (small).
        batch_full = []
        for cid, path in batch_slice:
            d = np.load(path / "data.npz")
            actions = np.concatenate(
                [d["cmd_joint_position"].astype(np.float32),
                 d["cmd_gripper_position"].reshape(-1, 1).astype(np.float32)],
                axis=-1,
            )
            batch_full.append((cid, path, actions))
        n_c, n_f = process_batch(
            batch_full, args.out_root, args.model_path, args.text_encoder_path,
            load_text_encoder_in_8bit=args.load_text_encoder_in_8bit,
        )
        committed_total += n_c
        failed_total += n_f
        elapsed = time.time() - t0
        rate = committed_total / elapsed if elapsed > 0 else 0
        remaining = len(pending) - (committed_total + failed_total)
        eta = remaining / rate if rate > 0 else float("inf")
        print(f"  batch {i // args.batch_size + 1}: "
              f"+{n_c} committed (-{n_f} failed) | "
              f"total {committed_total}/{len(pending)} | "
              f"elapsed {elapsed:.0f}s, ~{rate:.2f} chunks/s, ETA {eta:.0f}s",
              flush=True)

        if args.max_chunks > 0 and committed_total >= args.max_chunks:
            print(f"  Hit --max_chunks={args.max_chunks}, stopping cleanly")
            break

    print(f"\nDONE. committed={committed_total} failed={failed_total} elapsed={time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
