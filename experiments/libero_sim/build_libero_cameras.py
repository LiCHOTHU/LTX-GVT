#!/usr/bin/env python
"""Build the PRoPE ``cameras/`` stream for the LIBERO-90 precomputed dataset.

The 2-view encode (encode_libero_precomputed.py) writes 4 streams per chunk
(latents / reference_latents / conditions / actions) but NOT cameras. PRoPE needs
per-token camera geometry, and LIBERO is sim so the geometry is exact ground truth
already sitting in every raw context npz:

    <view>_K          (3,3)      OpenCV pinhole intrinsics @ native 256x256
    <view>_cam2world  (F,4,4)    per-frame camera->world (OpenCV; +z forward, y down)

This script is a pure CPU file-shuffle (no model, no GPU): for each ALREADY-ENCODED
chunk it reads only those small arrays from the raw npz (np.load is lazy — it does
NOT read the 15 MB of frames), converts to PRoPE's world->camera convention, and
writes one tiny tensor file aligned 1:1 with the latents.

Output (mirrors the other streams' layout exactly):
    <out_root>/precomputed/cameras/videos/<cid>_target.pt
      {"viewmats": (V, F, 4, 4),   # world->camera = inverse(cam2world), float32
       "Ks":       (V, 3, 3)}      # intrinsics (time-invariant), float32
    with V=2 views in canonical order [agentview (left tile), wrist (right tile)],
    matching encode's _hconcat2(agentview, wrist).

ALIGNMENT (why this is correct, not guessed):
  * convention: viewmats = inverse(cam2world). Proven by build_libero_context.py's
    depth_reproject/_unproject, which warp pixels with exactly K + cam2world (+ its
    inverse for the target cam) — the same math that produced the validated context
    priors.
  * view order: encode concatenates agentview|wrist left|right -> view 0 / view 1.
  * temporal: ALL F poses are stored here (full fidelity). The per-token frame
    resampling to the latent-frame count happens in the trainer's _build_per_token_cameras
    using the SAME linspace decimation the action tokens use, so cameras stay aligned
    with actions and video latents.

KEYED OFF THE LATENTS STREAM: we enumerate existing latents/videos/*_target.pt and
emit one cameras file per latents file, so the cameras count always matches latents
(PrecomputedDataset requires equal per-source counts). Re-run after the encode
finishes to pick up any stragglers. Idempotent + resumable (skips existing).
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch

VIEWS = ("agentview", "wrist")  # canonical order: left tile, right tile (matches encode _hconcat2)


def cid_from_latents(p: Path) -> str:
    """latents/videos/<cid>_target.pt -> <cid>."""
    name = p.name
    if not name.endswith("_target.pt"):
        raise ValueError(f"unexpected latents filename: {name}")
    return name[: -len("_target.pt")]


def npz_for_cid(cid: str, src_root: Path) -> Path:
    """<cid> == <ep_id>_chunk_NN  ->  <src_root>/<ep_id>/chunk_NN/data.npz.

    Mirrors encode_libero_precomputed.cid_for: cid = parent.parent.name + '_' + parent.name,
    where parent.name == 'chunk_NN'. rsplit on the LAST '_chunk_' recovers (ep_id, NN).
    """
    ep_id, chunk_nn = cid.rsplit("_chunk_", 1)
    return src_root / ep_id / f"chunk_{chunk_nn}" / "data.npz"


def build_one(npz_path: Path) -> dict[str, torch.Tensor]:
    """Read cameras from one raw npz -> {'viewmats': (V,F,4,4), 'Ks': (V,3,3)} float32."""
    d = np.load(npz_path, allow_pickle=True)  # lazy: only the keys we touch are read
    viewmats, Ks = [], []
    for v in VIEWS:
        c2w = np.asarray(d[f"{v}_cam2world"], dtype=np.float64)  # (F,4,4) camera->world
        K = np.asarray(d[f"{v}_K"], dtype=np.float64)            # (3,3)
        if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
            raise ValueError(f"{npz_path}: {v}_cam2world has shape {c2w.shape}, expected (F,4,4)")
        if K.shape != (3, 3):
            raise ValueError(f"{npz_path}: {v}_K has shape {K.shape}, expected (3,3)")
        w2c = np.linalg.inv(c2w)  # world->camera per frame == PRoPE viewmats
        viewmats.append(w2c)
        Ks.append(K)
    vm = torch.from_numpy(np.stack(viewmats, axis=0)).to(torch.float32)  # (V,F,4,4)
    ks = torch.from_numpy(np.stack(Ks, axis=0)).to(torch.float32)        # (V,3,3)
    return {"viewmats": vm, "Ks": ks}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", type=Path, default=Path("/storage/scratch1/8/lwang831/libero90_context"),
                    help="raw context npz root (holds <view>_K / <view>_cam2world)")
    ap.add_argument("--out_root", type=Path, default=Path("/storage/scratch1/8/lwang831/libero90_precomputed"),
                    help="precomputed dataset root (the dir that CONTAINS precomputed/)")
    ap.add_argument("--max_chunks", type=int, default=0, help="0 = all encoded chunks")
    ap.add_argument("--shard", type=str, default="0/1", help='"i/N" round-robin shard for parallel runs')
    args = ap.parse_args()

    pre = args.out_root / "precomputed"
    latents_dir = pre / "latents" / "videos"
    cam_dir = pre / "cameras" / "videos"
    cam_dir.mkdir(parents=True, exist_ok=True)

    latents = sorted(glob.glob(str(latents_dir / "*_target.pt")))
    if not latents:
        print(f"[error] no latents under {latents_dir} — run the encode first", file=sys.stderr)
        return 1
    if args.max_chunks > 0:
        latents = latents[: args.max_chunks]

    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    work = [Path(p) for k, p in enumerate(latents) if k % shard_n == shard_i]

    print(f"[cameras] {len(latents)} encoded chunks | shard {shard_i}/{shard_n} -> {len(work)} to consider")
    done = skipped = failed = 0
    for i, lp in enumerate(work):
        cid = cid_from_latents(lp)
        out = cam_dir / f"{cid}_target.pt"
        if out.exists():
            skipped += 1
            continue
        npz = npz_for_cid(cid, args.src_root)
        try:
            if not npz.exists():
                raise FileNotFoundError(f"raw npz missing: {npz}")
            payload = build_one(npz)
            tmp = out.with_suffix(".pt.tmp")
            torch.save(payload, tmp)
            tmp.replace(out)  # atomic
            done += 1
        except Exception as exc:
            print(f"  [WARN] {cid}: {exc!r}", file=sys.stderr)
            failed += 1
        if (i + 1) % 1000 == 0:
            print(f"  ...{i + 1}/{len(work)}  (wrote {done}, skipped {skipped}, failed {failed})", flush=True)

    print(f"[cameras] done. wrote={done} skipped={skipped} failed={failed} -> {cam_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
