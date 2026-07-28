"""Preemption-safe, resumable v2v IC-LoRA dataset builder for the full DROID-100
chunk universe.

Reads ``out_root/_universe.json`` (produced by enumerate_universe.py) for the
authoritative chunk list, then drives a four-rung ladder per chunk:

  Rung 1 — extract episode (TF subprocess: wrist_demo.py --mode extract).
           Produces outputs/context/<short>/data.npz + info.json (atomic per ep).
  Rung 2 — VGGT wrist calibration (PyTorch subprocess: calibrate_wrist_vggt.py).
           Produces outputs/wrist_calib_vggt/<short>.json (atomic per ep).
  Rung 3 — chunk context render (TF subprocess: chunk_context.py --mode augment,
           then EGL subprocess: chunk_context.py --mode render). Produces
           outputs/context_chunks_wrist_new/<short>/chunk_NN/ (atomic per chunk).
  Rung 4 — tile + process_dataset.py + atomic commit on cedar. Same as before.

Each rung writes to a same-FS staging dir, verifies, then atomic-renames into
its final location. Killing the orchestrator at any point leaves at most one
half-built tmp dir, which is wiped on the next launch.

Reliability guarantees:
  1. **Atomic per-chunk commit.** All four .pt files of a chunk are written into
     ``staging/<batch_uuid>/`` then atomic-renamed into
     ``precomputed/<source>/videos/<id>_target.pt`` with a sentinel ``_done/<id>``.
  2. **Atomic upstream artifacts.** Each rung-1/2/3 output is a same-dir tmp →
     rename, so a half-finished upstream artifact never leaks into the cache.
  3. **Resumable.** A relaunch reads ``_universe.json``, skips chunks whose
     ``_done/<id>`` already exists, and only rebuilds the rungs (1-4) that are
     missing for each pending chunk.
  4. **Hash-sharded parallelism.** ``--shard i/N`` lets multiple sbatch tasks
     cooperate without coordination — but the watchdog by default runs ONE
     sbatch at a time, so the common case is N=1.

Usage:
    python build_dataset_resumable.py \
        --out_root /storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/gvt_dataset_full \
        --shard 0/1 \
        --batch_size 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
# Built data is WRITTEN here. cedar0 filled to 100%, so the build target lives on
# the per-user scratch1 quota (15 TB, auto-purges files unused for ~60 days — fine
# for an active build; move the finished dataset somewhere durable afterwards).
# The DROID RLDS source + model checkpoints below stay on cedar (read-only inputs).
DEFAULT_OUT_ROOT = Path("/storage/scratch1/8/lwang831/gvt_dataset_full")
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


# ---------- universe + sharding ----------

# Per-episode scratch (Phase A npz, Phase B calib JSON, Phase C mp4s). Lives on
# scratch1 alongside the build target (cedar0 is full, $HOME is a 20 GB cap). These
# are regenerable intermediates — resume keys off _done/<cid> markers, not scratch —
# so a fresh empty dir on scratch1 is fine. Env vars override for local demos.
_DEFAULT_SCRATCH = Path(
    "/storage/scratch1/8/lwang831/gvt_dataset_full/scratch"
)
CONTEXT_ROOT = Path(os.environ.get("GVT_CONTEXT_ROOT", _DEFAULT_SCRATCH / "context"))
WRIST_CALIB_ROOT = Path(os.environ.get("GVT_WRIST_CALIB_ROOT", _DEFAULT_SCRATCH / "wrist_calib_vggt"))
CHUNKS_ROOT = Path(os.environ.get("GVT_CHUNKS_ROOT", _DEFAULT_SCRATCH / "context_chunks_wrist_new"))


def load_universe(out_root: Path) -> list[dict]:
    """Read _universe.json (produced by enumerate_universe.py). Raises if missing."""
    path = out_root / "_universe.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Universe manifest missing: {path}. Run enumerate_universe.py first."
        )
    return json.loads(path.read_text())


def shard_filter_universe(universe: list[dict], shard_idx: int, shard_total: int) -> list[dict]:
    if shard_total <= 1:
        return universe
    out = []
    for u in universe:
        h = int.from_bytes(hashlib.sha1(u["cid"].encode()).digest()[:8], "big")
        if h % shard_total == shard_idx:
            out.append(u)
    return out


def chunk_dir_for(ep_short: str, chunk_idx: int) -> Path:
    """Final location of a rendered chunk's data.npz (after atomic move)."""
    return CHUNKS_ROOT / ep_short / f"chunk_{chunk_idx:02d}"


# ---------- atomic helpers for upstream artifacts ----------

def _atomic_replace_dir(src: Path, dst: Path) -> None:
    """Atomic dir-level rename. Wipes dst if it exists first.
    Both paths must live on the same filesystem.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        # Move existing aside; if rename succeeds, drop the old. Doing this with
        # a rename instead of rmtree means we never delete a good dir before
        # we have the new one in place.
        backup = dst.with_name(f".{dst.name}.bak.{os.getpid()}")
        os.replace(dst, backup)
        try:
            os.replace(src, dst)
        except Exception:
            os.replace(backup, dst)
            raise
        shutil.rmtree(backup, ignore_errors=True)
    else:
        os.replace(src, dst)


def _data_npz_has_keys(path: Path, required: tuple[str, ...]) -> bool:
    try:
        with np.load(path) as d:
            return all(k in d.files for k in required)
    except Exception:
        return False


def _episode_extracted(ep_short: str) -> bool:
    npz = CONTEXT_ROOT / ep_short / "data.npz"
    # Require cmd_gripper_position (from augment pass); without it, chunk_context
    # render skips the episode.
    return _data_npz_has_keys(
        npz,
        ("cmd_joint_position", "ext1_frames", "ext2_frames", "wrist_frames",
         "K1", "K2", "K_wrist", "cam2base_1", "cam2base_2", "cmd_gripper_position"),
    )


def _vggt_done(ep_short: str) -> bool:
    p = WRIST_CALIB_ROOT / f"{ep_short}.json"
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text())
        return "T_cam_to_hand" in d or "T_wrist_to_ext1" in d or "T_wrist_to_ext2" in d
    except Exception:
        return False


def _chunk_rendered(ep_short: str, chunk_idx: int) -> bool:
    npz = chunk_dir_for(ep_short, chunk_idx) / "data.npz"
    return _data_npz_has_keys(
        npz,
        ("ext1_frames", "ext2_frames", "wrist_frames",
         "ext1_context", "ext2_context", "wrist_context",
         "cmd_joint_position", "cmd_gripper_position"),
    )


# ---------- rung 1: extract (TF subprocess) ----------

def _run_phase_a_extract(needed_eps: list[str], *, py: str) -> None:
    """Run wrist_demo.py --mode extract for the given DROID episode IDs, atomically
    landing each ep dir at outputs/context/<short>/.

    All staging happens in a per-pid tmp tree adjacent to CONTEXT_ROOT so
    `os.replace` is same-FS. The subprocess sees WRIST_EXTRACT_OUT pointing at
    that staging root.
    """
    if not needed_eps:
        return
    stage_root = CONTEXT_ROOT.parent / f".stage_extract.{os.getpid()}"
    if stage_root.exists():
        shutil.rmtree(stage_root, ignore_errors=True)
    stage_root.mkdir(parents=True)
    # Pass IDs via file: Linux caps a single env-var string at MAX_ARG_STRLEN
    # (128 KB on this kernel). 10K+ DROID episode IDs blow past that.
    eps_file = stage_root / "eps.txt"
    eps_file.write_text("\n".join(needed_eps) + "\n")
    env = {
        **os.environ,
        "WRIST_DEMO_EPS_FILE": str(eps_file),
        "WRIST_EXTRACT_OUT": str(stage_root),
        # No N cap — process all IDs we asked for.
        "WRIST_DEMO_N": str(len(needed_eps)),
    }
    env.pop("WRIST_DEMO_EPS", None)
    cmd = [py, "-u", str(_HERE / "wrist_demo.py"), "--mode", "extract"]
    print(f"[Phase A] extract {len(needed_eps)} eps via wrist_demo.py …", flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise RuntimeError(f"wrist_demo --mode extract failed rc={rc}")
    moved = 0
    for ep_id in needed_eps:
        ep_short = _short_id_for(ep_id)
        src = stage_root / ep_short
        if not (src / "data.npz").exists():
            print(f"  [WARN] extract produced no data.npz for {ep_short}", file=sys.stderr)
            continue
        dst = CONTEXT_ROOT / ep_short
        _atomic_replace_dir(src, dst)
        moved += 1
    shutil.rmtree(stage_root, ignore_errors=True)
    print(f"[Phase A] atomically committed {moved}/{len(needed_eps)} extracts", flush=True)


def _run_phase_a_augment(needed_eps: list[str], *, py: str) -> None:
    """Run chunk_context.py --mode augment to add cmd_gripper_position to each
    extracted data.npz IN-PLACE. (chunk_context's augment_pass uses np.savez
    overwrite — not atomic, but the parent data.npz is also recoverable from the
    RLDS, so if a write is interrupted, the next launch's _episode_extracted()
    check fails the file and re-augments.)
    """
    if not needed_eps:
        return
    # CHUNK_EPS expects short IDs (per chunk_context._resolve_ep_list/_short_id).
    short_ids = [_short_id_for(e) for e in needed_eps]
    # File-based: avoid MAX_ARG_STRLEN cap (see _run_phase_a_extract note).
    tmp_eps = Path(tempfile.gettempdir()) / f"chunk_eps.augment.{os.getpid()}.txt"
    tmp_eps.write_text("\n".join(short_ids) + "\n")
    env = {**os.environ, "CHUNK_EPS_FILE": str(tmp_eps)}
    env.pop("CHUNK_EPS", None)
    cmd = [py, "-u", str(_HERE / "chunk_context.py"), "--mode", "augment"]
    print(f"[Phase A] augment cmd_gripper for {len(short_ids)} eps …", flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    tmp_eps.unlink(missing_ok=True)
    if rc != 0:
        raise RuntimeError(f"chunk_context --mode augment failed rc={rc}")


# ---------- rung 2: VGGT (PyTorch subprocess) ----------

def _run_phase_b_vggt(needed_eps_short: list[str], *, py: str) -> None:
    """Run calibrate_wrist_vggt.py for the given short-id episodes. The script
    iterates outputs/context/, but we restrict via VGGT_EPS_FILTER, then redirect
    output to a per-pid staging dir for atomic per-ep commit.
    """
    if not needed_eps_short:
        return
    stage_dir = WRIST_CALIB_ROOT.parent / f".stage_vggt.{os.getpid()}"
    if stage_dir.exists():
        shutil.rmtree(stage_dir, ignore_errors=True)
    stage_dir.mkdir(parents=True)
    # Guard against the .bashrc-leaked HF_HOME=/huggingface (XDG_CACHE_HOME
    # unset on compute nodes). Force a writable cache path so HF downloads
    # don't try to mkdir at filesystem root.
    _hf = os.environ.get("HF_HOME", "")
    if (not _hf) or _hf == "/" or _hf.startswith("/huggingface"):
        _hf = "/storage/project/r-agarg35-0/lwang831/hf_cache"
    eps_file = stage_dir / "eps.txt"
    eps_file.write_text("\n".join(needed_eps_short) + "\n")
    env = {
        **os.environ,
        # File-based: avoid MAX_ARG_STRLEN (see _run_phase_a_extract note).
        "VGGT_EPS_FILTER_FILE": str(eps_file),
        "WRIST_CALIB_OUT": str(stage_dir),
        # Default VGGT_DEVICE=cpu unless caller overrides — keeps GPU free for
        # the EGL+PyTorch trainer load in phases C/D.
        "VGGT_DEVICE": os.environ.get("VGGT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
        "HF_HOME": _hf,
        "HF_HUB_CACHE": f"{_hf}/hub",
    }
    # Strip any inherited broken pointers that override HF_HOME silently.
    for _bad_key in ("HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "VGGT_EPS_FILTER"):
        env.pop(_bad_key, None)
    cmd = [py, "-u", str(_HERE / "calibrate_wrist_vggt.py")]
    print(f"[Phase B] VGGT calibrate {len(needed_eps_short)} eps "
          f"on {env['VGGT_DEVICE']} …", flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise RuntimeError(f"calibrate_wrist_vggt failed rc={rc}")
    WRIST_CALIB_ROOT.mkdir(parents=True, exist_ok=True)
    moved = 0
    for ep_short in needed_eps_short:
        src = stage_dir / f"{ep_short}.json"
        if not src.exists():
            print(f"  [WARN] VGGT produced no json for {ep_short}", file=sys.stderr)
            continue
        dst = WRIST_CALIB_ROOT / f"{ep_short}.json"
        atomic_replace(src, dst)
        moved += 1
    shutil.rmtree(stage_dir, ignore_errors=True)
    print(f"[Phase B] atomically committed {moved}/{len(needed_eps_short)} calibs", flush=True)


# ---------- rung 3: chunk render (EGL subprocess) ----------

def _run_phase_c_render(needed_chunks: list[tuple[str, int]], *, py: str) -> None:
    """Run chunk_context.py --mode render for episodes that have at least one
    un-rendered chunk. Per-chunk granularity is too fine (pyrender warm-up),
    so we render all chunks of each ep into a staging dir then atomic-rename
    just the ones we needed.

    needed_chunks: list of (ep_short, chunk_idx) pairs.
    """
    if not needed_chunks:
        return
    eps_to_render = sorted({ep for ep, _ in needed_chunks})
    stage_root = CHUNKS_ROOT.parent / f".stage_render.{os.getpid()}"
    if stage_root.exists():
        shutil.rmtree(stage_root, ignore_errors=True)
    stage_root.mkdir(parents=True)
    eps_file = stage_root / "eps.txt"
    eps_file.write_text("\n".join(eps_to_render) + "\n")
    env = {
        **os.environ,
        # File-based: avoid MAX_ARG_STRLEN (see _run_phase_a_extract note).
        "CHUNK_EPS_FILE": str(eps_file),
        "CHUNK_OUT_ROOT": str(stage_root),
        "PYOPENGL_PLATFORM": "egl",
    }
    env.pop("CHUNK_EPS", None)
    cmd = [py, "-u", str(_HERE / "chunk_context.py"), "--mode", "render"]
    print(f"[Phase C] render {len(needed_chunks)} chunks "
          f"across {len(eps_to_render)} eps …", flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise RuntimeError(f"chunk_context --mode render failed rc={rc}")
    moved = 0
    for ep_short, ci in needed_chunks:
        src = stage_root / ep_short / f"chunk_{ci:02d}"
        if not (src / "data.npz").exists():
            print(f"  [WARN] no rendered chunk for {ep_short} chunk_{ci:02d}",
                  file=sys.stderr)
            continue
        dst = chunk_dir_for(ep_short, ci)
        _atomic_replace_dir(src, dst)
        moved += 1
    shutil.rmtree(stage_root, ignore_errors=True)
    print(f"[Phase C] atomically committed {moved}/{len(needed_chunks)} chunk renders",
          flush=True)


def _short_id_for(ep_id: str) -> str:
    """Mirror wrist_demo.short_id."""
    import re
    return re.sub(r"[^A-Za-z0-9_+-]", "_", ep_id)[:40]


def ensure_upstream_for(pending: list[dict], *, py: str) -> list[dict]:
    """Run rungs 1-3 to ensure every pending chunk has a rendered context dir.
    Returns the subset of `pending` whose chunk context now exists (may be
    smaller than `pending` if some episodes failed mid-rung).
    """
    # Rung 1 — extract: only episodes whose context/<ep>/data.npz is missing
    # the wrist-extract output altogether (no `cmd_joint_position` etc).
    eps_need_extract = sorted({
        u["ep_id"] for u in pending
        if not (CONTEXT_ROOT / u["ep_short"] / "data.npz").exists()
    })
    _run_phase_a_extract(eps_need_extract, py=py)

    # Rung 1b — augment: any episode whose data.npz lacks cmd_gripper_position.
    eps_need_augment = sorted({
        u["ep_id"] for u in pending
        if not _episode_extracted(u["ep_short"])
    })
    _run_phase_a_augment(eps_need_augment, py=py)

    # Rung 2 — VGGT: only episodes whose calib JSON is missing.
    eps_need_vggt = sorted({u["ep_short"] for u in pending if not _vggt_done(u["ep_short"])})
    _run_phase_b_vggt(eps_need_vggt, py=py)

    # Rung 3 — render: chunks whose final dir is missing data.npz.
    chunks_need_render = [
        (u["ep_short"], u["chunk_idx"]) for u in pending
        if not _chunk_rendered(u["ep_short"], u["chunk_idx"])
    ]
    _run_phase_c_render(chunks_need_render, py=py)

    # Filter to chunks whose context now exists. Anything still missing → failed
    # upstream; skip for this run, the next launch will retry that rung.
    ready = [u for u in pending if _chunk_rendered(u["ep_short"], u["chunk_idx"])]
    dropped = len(pending) - len(ready)
    if dropped:
        print(f"[upstream] {dropped} chunks still un-rendered after rungs 1-3 — "
              f"will retry on next launch.", flush=True)
    return ready


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


def build_chunk_cameras(d: dict) -> dict[str, torch.Tensor]:
    """PRoPE ``cameras`` stream for the 2x2 DROID panel — same schema as the
    LIBERO stream (build_libero_cameras.py): {"viewmats": (V,F,4,4) world->camera
    float32, "Ks": (V,3,3) float32}.

    V=3 views in TILE order [ext1 (top-left), ext2 (top-right), wrist
    (bottom-right)]; the bottom-left tile is black and carries no camera.
    World = robot base frame (shared by all three cam2base/extrinsics).
    ext1/ext2 are fixed mounts (pose tiled across F); the wrist trajectory
    comes from ``wrist_cam_poses`` (official raw-traj extrinsics incl. the
    mirrored-mount Rz(180) fix, or FK∘VGGT fallback — whatever render_chunk
    used). Ks are rescaled to the TILE resolution the panel upscales to.

    NOTE: the trainer's _build_per_token_cameras maps view-from-width-halves
    (LIBERO 2-view L|R layout). DROID PRoPE finetuning additionally needs that
    mapping extended to this 2x2 quadrant layout — the data saved here is
    layout-complete for it.
    """
    F = int(d["ext1_frames"].shape[0])

    def K_at_tile(K: np.ndarray, frames: np.ndarray) -> np.ndarray:
        h, w = frames.shape[1:3]
        s = np.diag([TILE_W / w, TILE_H / h, 1.0])
        return s @ np.asarray(K, dtype=np.float64)

    Ks = np.stack([
        K_at_tile(d["K1"], d["ext1_frames"]),
        K_at_tile(d["K2"], d["ext2_frames"]),
        K_at_tile(d["K_wrist"], d["wrist_frames"]),
    ])
    wrist_poses = np.asarray(d["wrist_cam_poses"], dtype=np.float64)  # (F,4,4) cam->base
    if wrist_poses.shape != (F, 4, 4):
        raise ValueError(f"wrist_cam_poses shape {wrist_poses.shape}, expected ({F},4,4)")
    viewmats = np.stack([
        np.broadcast_to(np.linalg.inv(np.asarray(d["cam2base_1"], dtype=np.float64)), (F, 4, 4)),
        np.broadcast_to(np.linalg.inv(np.asarray(d["cam2base_2"], dtype=np.float64)), (F, 4, 4)),
        np.linalg.inv(wrist_poses),
    ])
    return {
        "viewmats": torch.from_numpy(np.ascontiguousarray(viewmats)).to(torch.float32),
        "Ks": torch.from_numpy(np.ascontiguousarray(Ks)).to(torch.float32),
    }


# ---------- batch process_dataset + atomic commit ----------

def precomputed_paths(out_root: Path, cid: str) -> dict[str, Path]:
    """Final destinations for a chunk's four .pt files."""
    base = out_root / "precomputed"
    return {
        "latents": base / "latents" / "videos" / f"{cid}_target.pt",
        "reference_latents": base / "reference_latents" / "videos" / f"{cid}_target.pt",
        "conditions": base / "conditions" / "videos" / f"{cid}_target.pt",
        "actions": base / "actions" / "videos" / f"{cid}_target.pt",
        # PRoPE camera stream (added 2026-06-07). Listing it here makes
        # chunk_is_done() auto-REVIVE every chunk built before the official-pose
        # + cameras upgrade — exactly the invalidation we want: the whole corpus
        # rebuilds with the new context method.
        "cameras": base / "cameras" / "videos" / f"{cid}_target.pt",
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
    cameras_map: dict[str, dict[str, torch.Tensor]] = {}
    for cid, chunk_dir, actions in batch:
        try:
            tgt, ref, _ = build_chunk_videos(chunk_dir, cid, staging / "videos")
            cameras_map[cid] = build_chunk_cameras(np.load(chunk_dir / "data.npz"))
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

            # PRoPE cameras (viewmats/Ks), aligned 1:1 with the latents stream.
            write_torch_atomic(cameras_map[cid], final["cameras"])

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
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--shard", type=str, default="0/1", help='"i/N" worker shard.')
    parser.add_argument("--batch_size", type=int, default=8, help="Chunks per process_dataset.py invocation.")
    parser.add_argument("--max_chunks", type=int, default=0, help="If >0, stop after committing this many chunks.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_encoder_path", type=str, default=DEFAULT_TEXT_ENCODER_PATH)
    parser.add_argument("--dry_run", action="store_true", help="Enumerate + skip-status only; no work.")
    parser.add_argument(
        "--skip_upstream", action="store_true",
        help="Don't run rungs 1-3; assume context chunks are already rendered. "
             "Useful for local testing when CHUNKS_ROOT is pre-populated.",
    )
    parser.add_argument(
        "--load_text_encoder_in_8bit",
        action="store_true",
        help="Load Gemma in 8-bit (bitsandbytes). Required for <24 GB GPUs (V100 etc).",
    )
    parser.add_argument(
        "--only_cids", type=str, default="",
        help="Comma-separated cid list. If set, only these chunks are processed "
             "(intersected with the manifest). Smoke-test / single-chunk use.",
    )
    args = parser.parse_args()

    args.out_root = args.out_root.resolve()

    # If the caller passed the default --shard "0/1" and we're running under an
    # `sbatch --array=...` job, infer the shard from SLURM_ARRAY_TASK_{ID,COUNT}.
    if args.shard == "0/1" and "SLURM_ARRAY_TASK_ID" in os.environ:
        sid = int(os.environ["SLURM_ARRAY_TASK_ID"])
        sct = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
        args.shard = f"{sid}/{sct}"
        print(f"[shard] inferred from Slurm array: {args.shard}")

    shard_i, shard_n = (int(s) for s in args.shard.split("/"))

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "_done").mkdir(exist_ok=True)
    (args.out_root / "staging").mkdir(exist_ok=True)
    CONTEXT_ROOT.mkdir(parents=True, exist_ok=True)
    WRIST_CALIB_ROOT.mkdir(parents=True, exist_ok=True)
    CHUNKS_ROOT.mkdir(parents=True, exist_ok=True)

    universe = load_universe(args.out_root)
    if args.only_cids:
        wanted = {s.strip() for s in args.only_cids.split(",") if s.strip()}
        universe = [u for u in universe if u["cid"] in wanted]
        print(f"[only_cids] restricting universe to {len(universe)} chunks")
    my_universe = shard_filter_universe(universe, shard_i, shard_n)
    pending = [u for u in my_universe if not chunk_is_done(args.out_root, u["cid"])]
    done_count = len(my_universe) - len(pending)

    print(f"== shard {shard_i}/{shard_n}  out={args.out_root}")
    print(f"   universe total chunks    : {len(universe)}")
    print(f"   chunks assigned to shard : {len(my_universe)}")
    print(f"   already done (skipped)   : {done_count}")
    print(f"   pending in this run      : {len(pending)}")

    if args.dry_run or not pending:
        return 0

    # Rungs 1-3: ensure each pending chunk has its rendered context dir.
    # We may end up with fewer chunks than `pending` if an upstream rung fails
    # for some episode; the missing ones get retried on the next launch.
    if args.skip_upstream:
        ready = [u for u in pending if _chunk_rendered(u["ep_short"], u["chunk_idx"])]
        print(f"[upstream] --skip_upstream: {len(ready)}/{len(pending)} ready")
    else:
        ready = ensure_upstream_for(pending, py=sys.executable)

    if not ready:
        print("Nothing ready after upstream; exiting.")
        return 0

    committed_total = failed_total = 0
    t0 = time.time()
    for i in range(0, len(ready), args.batch_size):
        batch_slice = ready[i : i + args.batch_size]
        # Build per-chunk tuples (cid, chunk_dir_path, actions_array).
        batch_full = []
        for u in batch_slice:
            cdir = chunk_dir_for(u["ep_short"], u["chunk_idx"])
            d = np.load(cdir / "data.npz")
            actions = np.concatenate(
                [d["cmd_joint_position"].astype(np.float32),
                 d["cmd_gripper_position"].reshape(-1, 1).astype(np.float32)],
                axis=-1,
            )
            batch_full.append((u["cid"], cdir, actions))
        n_c, n_f = process_batch(
            batch_full, args.out_root, args.model_path, args.text_encoder_path,
            load_text_encoder_in_8bit=args.load_text_encoder_in_8bit,
        )
        committed_total += n_c
        failed_total += n_f
        elapsed = time.time() - t0
        rate = committed_total / elapsed if elapsed > 0 else 0
        remaining = len(ready) - (committed_total + failed_total)
        eta = remaining / rate if rate > 0 else float("inf")
        print(f"  batch {i // args.batch_size + 1}: "
              f"+{n_c} committed (-{n_f} failed) | "
              f"total {committed_total}/{len(ready)} | "
              f"elapsed {elapsed:.0f}s, ~{rate:.2f} chunks/s, ETA {eta:.0f}s",
              flush=True)

        if args.max_chunks > 0 and committed_total >= args.max_chunks:
            print(f"  Hit --max_chunks={args.max_chunks}, stopping cleanly")
            break

    print(f"\nDONE. committed={committed_total} failed={failed_total} elapsed={time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
