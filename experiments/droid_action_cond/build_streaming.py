"""Per-episode streaming dataset builder.

Replaces the batched 4-phase pipeline (Phase A extract all eps → A.5 augment →
B VGGT all → C render all → D process all) with a single-process inline loop:
for each episode found in TFDS, run extract → VGGT → render → process → commit
_done markers, then move to the next episode.

Why: the batched pipeline spawned huge subprocesses that OOM'd at full-DROID
scale (10K+ eps), kept 0 visible progress for hours, and lost partial work on
preemption. This loop streams TFDS once, loads heavy models once, and commits
chunk-by-chunk so progress is visible immediately and a SIGTERM only loses one
in-flight episode.

Resumable: any episode whose chunks are all already in _done/ is skipped via the
ep_index filter. Per-chunk _done markers from prior runs (including the existing
droid_100 corpus) are honored verbatim — cid is stable across DROID releases.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

# Force HF cache BEFORE any HF import (.bashrc leaks HF_HOME=/huggingface on
# compute nodes where XDG_CACHE_HOME is unset). Same guard the phase scripts use.
_HF_FALLBACK = "/storage/project/r-agarg35-0/lwang831/hf_cache"
_hf = os.environ.get("HF_HOME", "")
if (not _hf) or _hf == "/" or _hf.startswith("/huggingface"):
    os.environ["HF_HOME"] = _HF_FALLBACK
for _k in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
    _v = os.environ.get(_k, "")
    if _v and (_v == "/" or _v.startswith("/huggingface")):
        os.environ.pop(_k, None)
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
# Hide GPU from TF before any tensorflow import. TFDS pulls in tensorflow which
# by default reserves ~all GPU memory, leaving none for VGGT / Gemma / LTX-VAE.
# We use TF only as the TFDS data loader (CPU is fine).
os.environ.setdefault("CUDA_VISIBLE_DEVICES_TF", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import torch

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent.parent / "packages" / "ltx-action-cond" / "src"))

from ltx_action_cond.droid import (
    CalibrationIndex, ROLE_TO_IMG_KEY, extract_episode_arrays,
    resolve_both_cams_cam2base, resolve_gripper_signal,
)
from ltx_action_cond.calibration import intrinsics_to_K, rescale_K

# Reuse the existing orchestrator's helpers (atomic IO, paths, Phase D).
sys.path.insert(0, str(_HERE))
from build_dataset_resumable import (  # noqa: E402
    CONTEXT_ROOT, WRIST_CALIB_ROOT, CHUNKS_ROOT,
    CHUNK_LEN, DEFAULT_OUT_ROOT, DEFAULT_MODEL_PATH, DEFAULT_TEXT_ENCODER_PATH,
    atomic_replace, _atomic_replace_dir, _atomic_tmp_path,
    load_universe, shard_filter_universe,
    chunk_dir_for, chunk_is_done, process_batch,
)
# chunk_context and calibrate_wrist_vggt each carry their OWN home-relative path
# defaults (CHUNK_{SRC,CALIB,OUT}_* and WRIST_{DATA_ROOT,CALIB_OUT}). build_streaming
# writes Phase A/B to CONTEXT_ROOT / WRIST_CALIB_ROOT (scratch1), so we pin those
# modules to the SAME roots via env vars set BEFORE importing them. Otherwise
# load_T_cam_to_hand reads calib from the stale home dir (every chunk SKIP-Cs) and
# run_vggt_pair writes its _tmp PNGs into $HOME. Setting before import also redirects
# calibrate_wrist_vggt's import-time OUT_DIR.mkdir() onto scratch1.
# build_dataset_resumable's roots are the single source of truth.
os.environ["CHUNK_SRC_ROOT"] = str(CONTEXT_ROOT)
os.environ["CHUNK_CALIB_DIR"] = str(WRIST_CALIB_ROOT)
os.environ["CHUNK_OUT_ROOT"] = str(CHUNKS_ROOT)
os.environ["WRIST_DATA_ROOT"] = str(CONTEXT_ROOT)
os.environ["WRIST_CALIB_OUT"] = str(WRIST_CALIB_ROOT)
# PRODUCTION wrist context decision (2026-06-07): raw wrist0 plane-warp with
# official poses, NO ext-camera fill (user rejected the fusion as geometrically
# wrong). chunk_context's own default is still "multiview" — pin it here.
os.environ.setdefault("WRIST_CONTEXT_MODE", "simple")

from chunk_context import (  # noqa: E402
    chunk_starts, load_T_cam_to_hand, load_official_wrist_poses, render_chunk,
)

DROID_RLDS_DIR = Path(os.environ.get(
    "DROID_RLDS_DIR",
    "/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/droid_full/1.0.1",
))
DROID_CALIB_DIR = Path(os.environ.get(
    "DROID_CALIB_DIR",
    "/storage/project/r-agarg35-0/lwang831/droid/calibration",
))


# ---------- per-episode Phase A (extract) ----------

def _short_id(ep_id: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_+-]", "_", ep_id)[:40]


def extract_episode_context(ep, calib: CalibrationIndex, ep_id: str) -> dict | None:
    """Pull all 3-camera arrays + joints + grippers from one TFDS episode.
    Returns the dict ready to be saved as outputs/context/<short>/data.npz, or
    None if the episode is structurally unusable (missing serials, no cam2base).
    """
    sers = calib.camera_serials.get(ep_id)
    if not sers:
        return None
    ext1_s = sers.get("ext1_cam_serial")
    ext2_s = sers.get("ext2_cam_serial")
    wrist_s = sers.get("wrist_cam_serial")
    if not (ext1_s and ext2_s and wrist_s):
        return None
    intr = calib.intrinsics.get(ep_id, {})
    if not all(s in intr and intr[s].get("width", 0) > 0 for s in (ext1_s, ext2_s, wrist_s)):
        return None

    res = resolve_both_cams_cam2base(calib, ep_id)
    if res is None:
        return None
    cam2base_1 = res["ext1_cam2base"]
    cam2base_2 = res["ext2_cam2base"]

    arr_ext1 = extract_episode_arrays(ep, img_key=ROLE_TO_IMG_KEY["ext1_cam_serial"])
    arr_ext2 = extract_episode_arrays(ep, img_key=ROLE_TO_IMG_KEY["ext2_cam_serial"])
    wrist_frames = []
    for step in ep["steps"]:
        wrist_frames.append(step["observation"]["wrist_image_left"].numpy())
    wrist_frames = np.stack(wrist_frames, axis=0)

    T, H, W = arr_ext1["frames"].shape[:3]
    def K_for(serial, sz):
        i = intr[serial]
        return rescale_K(intrinsics_to_K(i["cameraMatrix"]), (i["width"], i["height"]), sz)
    K1 = K_for(ext1_s, (W, H))
    K2 = K_for(ext2_s, (arr_ext2["frames"].shape[2], arr_ext2["frames"].shape[1]))
    K_wrist = K_for(wrist_s, (wrist_frames.shape[2], wrist_frames.shape[1]))

    cmd_gripper, gripper_src = resolve_gripper_signal(
        arr_ext1["cmd_gripper_position"], arr_ext1["gripper_position"]
    )

    return {
        "arrays": {
            "cmd_joint_position": arr_ext1["cmd_joint_position"],
            "gripper_for_render": cmd_gripper,
            "ext1_frames": arr_ext1["frames"],
            "ext2_frames": arr_ext2["frames"],
            "wrist_frames": wrist_frames,
            "K1": K1, "K2": K2, "K_wrist": K_wrist,
            "cam2base_1": cam2base_1, "cam2base_2": cam2base_2,
            # Augment-pass fields (used by chunk_context.render_chunk via SRC_ROOT load):
            "cmd_gripper_position": arr_ext1["cmd_gripper_position"],
            "obs_gripper_position": arr_ext1["gripper_position"],
        },
        "info": {
            "episode_id": ep_id,
            "institution": ep_id.split("+")[0],
            "n_steps": int(T),
            "ext1_source": res["ext1_source"],
            "ext2_source": res["ext2_source"],
            "gripper_source": gripper_src,
        },
    }


def write_context_atomic(ep_short: str, payload: dict) -> Path:
    """Atomically write context/<short>/{data.npz, info.json}."""
    final = CONTEXT_ROOT / ep_short
    stage = final.with_name(f".{ep_short}.stage.{os.getpid()}")
    if stage.exists():
        import shutil
        shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    np.savez(stage / "data.npz", **payload["arrays"])
    (stage / "info.json").write_text(json.dumps(payload["info"], indent=2))
    _atomic_replace_dir(stage, final)
    return final


def write_calib_atomic(ep_short: str, calib_result: dict) -> Path:
    WRIST_CALIB_ROOT.mkdir(parents=True, exist_ok=True)
    dst = WRIST_CALIB_ROOT / f"{ep_short}.json"
    tmp = _atomic_tmp_path(dst)
    tmp.write_text(json.dumps(calib_result, indent=2))
    os.replace(tmp, dst)
    return dst


def write_chunk_dir_atomic(ep_short: str, chunk_idx: int, stage_dir: Path) -> Path:
    """Move staged chunk dir into its final CHUNKS_ROOT location atomically."""
    dst = chunk_dir_for(ep_short, chunk_idx)
    _atomic_replace_dir(stage_dir, dst)
    return dst


# ---------- main streaming loop ----------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text_encoder_path", default=DEFAULT_TEXT_ENCODER_PATH)
    parser.add_argument("--load_text_encoder_in_8bit", action="store_true", default=True)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--max_eps", type=int, default=0, help="0 = process until walltime")
    parser.add_argument("--max_chunks", type=int, default=0)
    args = parser.parse_args()

    if args.shard == "0/1" and "SLURM_ARRAY_TASK_ID" in os.environ:
        sid = int(os.environ["SLURM_ARRAY_TASK_ID"])
        sct = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
        args.shard = f"{sid}/{sct}"
    shard_i, shard_n = (int(s) for s in args.shard.split("/"))

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "_done").mkdir(exist_ok=True)
    (args.out_root / "staging").mkdir(exist_ok=True)
    CONTEXT_ROOT.mkdir(parents=True, exist_ok=True)
    WRIST_CALIB_ROOT.mkdir(parents=True, exist_ok=True)
    CHUNKS_ROOT.mkdir(parents=True, exist_ok=True)

    universe = load_universe(args.out_root)
    my_universe = shard_filter_universe(universe, shard_i, shard_n)
    # Group pending chunks by ep_index. Skip eps whose chunks are all _done.
    by_ep_idx: dict[int, dict] = {}
    for u in my_universe:
        idx = int(u["ep_index"])
        if chunk_is_done(args.out_root, u["cid"]):
            continue
        if idx not in by_ep_idx:
            by_ep_idx[idx] = {
                "ep_id": u["ep_id"],
                "ep_short": u["ep_short"],
                "chunks": [],
            }
        by_ep_idx[idx]["chunks"].append({
            "cid": u["cid"], "chunk_idx": int(u["chunk_idx"]),
            "start": int(u["start"]), "end": int(u["end"]),
        })

    print(f"== streaming build  shard={shard_i}/{shard_n}  out={args.out_root}", flush=True)
    print(f"   universe total chunks    : {len(universe)}", flush=True)
    print(f"   pending eps (un-done)    : {len(by_ep_idx)}", flush=True)
    if not by_ep_idx:
        print("Nothing to do.")
        return 0

    # ----- Load TFDS + heavy models once -----
    print("[init] loading calibration index …", flush=True)
    calib = CalibrationIndex.load(DROID_CALIB_DIR)
    print("[init] opening TFDS (hiding GPU from TF first) …", flush=True)
    import tensorflow as tf
    # CRITICAL: TF must not see the GPU or it'll claim all 46 GB on the L40S,
    # leaving none for VGGT / LTX-VAE / Gemma. TFDS is CPU-only for our use.
    tf.config.set_visible_devices([], "GPU")
    import tensorflow_datasets as tfds
    builder = tfds.builder_from_directory(str(DROID_RLDS_DIR))
    ds = builder.as_dataset(split="train")

    # VGGT is only the FALLBACK wrist-pose source (episodes without official
    # raw-trajectory extrinsics — should be none for the current universe).
    # Lazy-load so the common path never pays the model load or the GPU memory.
    _vggt_state: dict = {}

    def get_vggt():
        if "model" not in _vggt_state:
            print("[lazy] loading VGGT (fallback pose path) …", flush=True)
            from calibrate_wrist_vggt import load_vggt
            _vggt_state["model"], _vggt_state["device"] = load_vggt()
        return _vggt_state["model"], _vggt_state["device"]

    print("[init] loading pyrender (FrankaMeshRenderer) …", flush=True)
    from ltx_action_cond.mesh_rendering import FrankaMeshRenderer
    # Renderer is per-resolution; lazily build per ep (TILE size varies).
    renderer_cache: dict[tuple[int, int], FrankaMeshRenderer] = {}

    def get_renderer(W: int, H: int) -> FrankaMeshRenderer:
        key = (W, H)
        if key not in renderer_cache:
            renderer_cache[key] = FrankaMeshRenderer(
                size=(W, H), mode="photo",
                gripper=os.environ.get("CHUNK_GRIPPER", "robotiq_2f85"),
                robotiq_mount_yaw=float(os.environ.get("ROBOTIQ_MOUNT_YAW", 1.5 * np.pi)),
                robotiq_mount_z_offset=float(os.environ.get("ROBOTIQ_MOUNT_Z", 0.0)),
            )
        return renderer_cache[key]

    print(f"[init] done. streaming TFDS for {len(by_ep_idx)} target episodes.", flush=True)
    t_start = time.time()
    eps_completed = 0
    chunks_committed_total = 0
    chunks_failed_total = 0

    for i, ep in enumerate(ds):
        if not by_ep_idx:
            break
        if i not in by_ep_idx:
            continue
        info = by_ep_idx.pop(i)
        ep_id = info["ep_id"]
        ep_short = info["ep_short"]
        wanted_chunks = info["chunks"]
        t_ep = time.time()
        print(f"\n=== ep_index={i}  {ep_id}  ({len(wanted_chunks)} chunks pending) ===",
              flush=True)

        # ----- Phase A: extract context arrays + save -----
        try:
            payload = extract_episode_context(ep, calib, ep_id)
        except Exception as exc:
            print(f"  [SKIP] extract failed: {type(exc).__name__}: {exc!s:.200}", flush=True)
            continue
        if payload is None:
            print(f"  [SKIP] not wrist-capable", flush=True)
            continue
        ctx_dir = write_context_atomic(ep_short, payload)
        T = payload["info"]["n_steps"]
        print(f"  [A] context saved (T={T})", flush=True)

        # ----- Phase B: wrist pose source -----
        # Preferred: OFFICIAL per-frame extrinsics from raw trajectory.h5
        # (downloaded for the full universe; includes the mirrored-mount
        # Rz(180) correction + unknown-convention refusal inside the loader).
        # Fallback only: VGGT consensus calibration (legacy pipeline).
        d = payload["arrays"]
        T_wrist_all = load_official_wrist_poses(ep_short, d)
        if T_wrist_all is not None:
            T_cam_to_hand = None
            wrist_pose_source = "official_raw_traj"
            print(f"  [B] official wrist poses (raw trajectory.h5) — VGGT skipped", flush=True)
        else:
            wrist_pose_source = "vggt_calib"
            print(f"  [B] WARN no official wrist poses for {ep_short}; VGGT fallback", flush=True)
            calib_json_path = WRIST_CALIB_ROOT / f"{ep_short}.json"
            if calib_json_path.exists():
                print(f"  [B] calib already present, skipping VGGT", flush=True)
            else:
                try:
                    from calibrate_wrist_vggt import calibrate_episode
                    vggt_model, vggt_device = get_vggt()
                    vggt_info = json.loads((ctx_dir / "info.json").read_text())
                    calib_result = calibrate_episode(vggt_model, vggt_device, ctx_dir, vggt_info)
                    if calib_result is None:
                        raise RuntimeError("calibrate_episode returned None")
                    # Mirror the per-ep aggregate field calibrate_wrist_vggt.main() adds.
                    calib_result["translation_cam_to_hand_m"] = list(
                        np.array(calib_result["T_cam_to_hand"])[:3, 3]
                    )
                    write_calib_atomic(ep_short, calib_result)
                    print(f"  [B] VGGT done, t_cam_to_hand={calib_result['translation_cam_to_hand_m']}",
                          flush=True)
                except Exception as exc:
                    print(f"  [SKIP-B] VGGT failed: {type(exc).__name__}: {exc!s:.200}", flush=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

        # ----- Phase C: render each pending chunk -----
        if T_wrist_all is None:
            try:
                T_cam_to_hand = load_T_cam_to_hand(ep_short, d)
            except Exception as exc:
                print(f"  [SKIP-C] load_T_cam_to_hand failed: {exc!s:.200}", flush=True)
                continue
        H, W = d["ext1_frames"].shape[1:3]
        renderer = get_renderer(W, H)
        all_starts = chunk_starts(T, CHUNK_LEN)
        rendered_chunks: list[tuple[str, Path, int, int]] = []
        for ch in wanted_chunks:
            ci = ch["chunk_idx"]
            start = ch["start"]
            end = ch["end"]
            if ci >= len(all_starts) or all_starts[ci] != start:
                print(f"  [WARN] chunk {ci} start mismatch (universe={start} "
                      f"vs computed={all_starts[ci] if ci < len(all_starts) else 'OOR'})",
                      flush=True)
                continue
            chunk_stage = (CHUNKS_ROOT / ep_short).with_name(
                f".{ep_short}.chunk_{ci:02d}.stage.{os.getpid()}"
            )
            if chunk_stage.exists():
                import shutil
                shutil.rmtree(chunk_stage, ignore_errors=True)
            chunk_stage.mkdir(parents=True)
            info_base = {
                "episode_id": ep_id,
                "institution": payload["info"]["institution"],
                "n_steps_episode": T,
                "chunk_index": ci,
                "chunk_count": len(all_starts),
                "chunk_len": CHUNK_LEN,
                "start": start,
                "end": end,
                "overlap_with_prev": (
                    max(0, all_starts[ci - 1] + CHUNK_LEN - start) if ci > 0 else 0
                ),
                "gripper_signal": "cmd_gripper_position (action)",
                "wrist_pose_source": wrist_pose_source,
            }
            try:
                render_chunk(
                    out_dir=chunk_stage,
                    info_base=info_base,
                    joints=d["cmd_joint_position"][start:end],
                    cmd_gripper=d["cmd_gripper_position"][start:end],
                    obs_gripper=d["obs_gripper_position"][start:end],
                    ext1f=d["ext1_frames"][start:end],
                    ext2f=d["ext2_frames"][start:end],
                    wristf=d["wrist_frames"][start:end],
                    K1=d["K1"], K2=d["K2"], K_wrist=d["K_wrist"],
                    cam2base_1=d["cam2base_1"], cam2base_2=d["cam2base_2"],
                    T_cam_to_hand=T_cam_to_hand,
                    renderer=renderer,
                    T_wrist_cam_all=(
                        T_wrist_all[start:end] if T_wrist_all is not None else None
                    ),
                )
                final = write_chunk_dir_atomic(ep_short, ci, chunk_stage)
                rendered_chunks.append((ch["cid"], final, start, end))
                print(f"  [C] chunk_{ci:02d} rendered", flush=True)
            except Exception as exc:
                print(f"  [SKIP-C chunk {ci}] {type(exc).__name__}: {exc!s:.200}", flush=True)
                import shutil
                shutil.rmtree(chunk_stage, ignore_errors=True)

        if not rendered_chunks:
            print(f"  [D] no chunks rendered → nothing to commit", flush=True)
            continue

        # ----- Phase D: build panels + VAE/text encode + commit _done -----
        d_batch = []
        for cid, cdir, start, end in rendered_chunks:
            actions = np.concatenate(
                [d["cmd_joint_position"][start:end].astype(np.float32),
                 d["cmd_gripper_position"][start:end].reshape(-1, 1).astype(np.float32)],
                axis=-1,
            )
            d_batch.append((cid, cdir, actions))
        n_c, n_f = process_batch(
            d_batch, args.out_root, args.model_path, args.text_encoder_path,
            load_text_encoder_in_8bit=args.load_text_encoder_in_8bit,
        )
        chunks_committed_total += n_c
        chunks_failed_total += n_f
        eps_completed += 1
        elapsed = time.time() - t_start
        print(f"  [D] committed {n_c}/{len(d_batch)} chunks  "
              f"(ep_total={chunks_committed_total} fail={chunks_failed_total}  "
              f"ep_dt={time.time() - t_ep:.0f}s  total_dt={elapsed:.0f}s)", flush=True)

        # Free per-ep memory: arrays can be 100s of MB.
        del payload, d
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if args.max_eps > 0 and eps_completed >= args.max_eps:
            print(f"  hit --max_eps={args.max_eps}, stopping cleanly", flush=True)
            break
        if args.max_chunks > 0 and chunks_committed_total >= args.max_chunks:
            print(f"  hit --max_chunks={args.max_chunks}, stopping cleanly", flush=True)
            break

    print(f"\nDONE. eps={eps_completed} chunks_committed={chunks_committed_total} "
          f"chunks_failed={chunks_failed_total} elapsed={time.time() - t_start:.0f}s",
          flush=True)
    if by_ep_idx:
        print(f"  ({len(by_ep_idx)} target eps remained un-reached — next launch picks up)",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
