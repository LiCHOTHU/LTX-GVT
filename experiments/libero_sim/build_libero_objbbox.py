#!/usr/bin/env python
"""Build the object-bounding-box stream for the object-centric active-learning signal.

Companion to build_libero_cameras.py. The context build (build_libero_context.py)
now records, per chunk, the BDDL goal objects' STATIC body-local box (center/half)
and their per-frame WORLD pose. This pass turns those into the 8 world-space corners
of each object's oriented 3D box, per frame, so the scorer can project them into the
agentview + wrist cameras (via the cameras/ stream's viewmats/Ks) and mask the
curiosity to object regions instead of the whole frame.

Output (mirrors cameras/ layout, keyed off the latents stream so counts match):
    <out_root>/precomputed/objbbox/videos/<cid>_target.pt
      {"corners": (n_obj, F, 8, 3) float32 world-space corners,
       "names":   list[str] of length n_obj}

A chunk with no objects (or built before this field existed) yields an empty
corners tensor; the scorer treats that as "no object mask" (ranks last).
Idempotent + resumable (skips existing) + shardable, exactly like build_libero_cameras.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch

# 8 unit-cube corner sign combinations, multiplied by the half-extents.
SIGNS = np.array([[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)
                  for sz in (-1.0, 1.0)], dtype=np.float64)  # (8, 3)


def cid_from_latents(p: Path) -> str:
    name = p.name
    if not name.endswith("_target.pt"):
        raise ValueError(f"unexpected latents filename: {name}")
    return name[: -len("_target.pt")]


def npz_for_cid(cid: str, src_root: Path) -> Path:
    """<cid> == <ep_id>_chunk_NN -> <src_root>/<ep_id>/chunk_NN/data.npz (mirrors cameras)."""
    ep_id, chunk_nn = cid.rsplit("_chunk_", 1)
    return src_root / ep_id / f"chunk_{chunk_nn}" / "data.npz"


def build_one(npz_path: Path) -> dict:
    """Read object box from one chunk npz -> {'corners':(n_obj,F,8,3), 'names':[...]}."""
    d = np.load(npz_path, allow_pickle=False)  # lazy: only the obj_* keys are read
    if "obj_pose" not in d.files or np.asarray(d["obj_pose"]).size == 0:
        return {"corners": torch.zeros(0, 0, 8, 3, dtype=torch.float32), "names": []}
    pose = np.asarray(d["obj_pose"], dtype=np.float64)          # (F, n_obj, 4, 4)
    center = np.asarray(d["obj_local_center"], dtype=np.float64)  # (n_obj, 3)
    half = np.asarray(d["obj_local_half"], dtype=np.float64)      # (n_obj, 3)
    names = [str(x) for x in d["obj_names"]]
    n_obj = pose.shape[1]
    if n_obj == 0:
        return {"corners": torch.zeros(0, 0, 8, 3, dtype=torch.float32), "names": []}

    local = center[:, None, :] + SIGNS[None, :, :] * half[:, None, :]       # (n_obj, 8, 3)
    local_h = np.concatenate([local, np.ones((n_obj, 8, 1))], axis=-1)      # (n_obj, 8, 4)
    # world[f,o,k,i] = sum_j pose[f,o,i,j] * local_h[o,k,j]
    world = np.einsum("foij,okj->foki", pose, local_h)[..., :3]             # (F, n_obj, 8, 3)
    corners = np.transpose(world, (1, 0, 2, 3)).astype(np.float32)          # (n_obj, F, 8, 3)
    return {"corners": torch.from_numpy(corners), "names": names}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", type=Path, required=True,
                    help="raw context npz root (holds obj_pose / obj_local_*)")
    ap.add_argument("--out_root", type=Path, required=True,
                    help="precomputed dataset root (the dir that CONTAINS precomputed/)")
    ap.add_argument("--max_chunks", type=int, default=0, help="0 = all encoded chunks")
    ap.add_argument("--shard", type=str, default="0/1", help='"i/N" round-robin shard')
    args = ap.parse_args()

    pre = args.out_root / "precomputed"
    latents_dir = pre / "latents" / "videos"
    obj_dir = pre / "objbbox" / "videos"
    obj_dir.mkdir(parents=True, exist_ok=True)

    latents = sorted(glob.glob(str(latents_dir / "*_target.pt")))
    if not latents:
        print(f"[error] no latents under {latents_dir} — run the encode first", file=sys.stderr)
        return 1
    if args.max_chunks > 0:
        latents = latents[: args.max_chunks]

    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    work = [Path(p) for k, p in enumerate(latents) if k % shard_n == shard_i]

    print(f"[objbbox] {len(latents)} encoded chunks | shard {shard_i}/{shard_n} -> {len(work)} to consider")
    done = skipped = failed = empty = 0
    for i, lp in enumerate(work):
        cid = cid_from_latents(lp)
        out = obj_dir / f"{cid}_target.pt"
        if out.exists():
            skipped += 1
            continue
        npz = npz_for_cid(cid, args.src_root)
        try:
            if not npz.exists():
                raise FileNotFoundError(f"raw npz missing: {npz}")
            payload = build_one(npz)
            if payload["corners"].numel() == 0:
                empty += 1
            tmp = out.with_suffix(".pt.tmp")
            torch.save(payload, tmp)
            tmp.replace(out)  # atomic
            done += 1
        except Exception as exc:
            print(f"  [WARN] {cid}: {exc!r}", file=sys.stderr)
            failed += 1
        if (i + 1) % 1000 == 0:
            print(f"  ...{i + 1}/{len(work)} (wrote {done}, skipped {skipped}, "
                  f"empty {empty}, failed {failed})", flush=True)

    print(f"[objbbox] done. wrote={done} skipped={skipped} empty={empty} "
          f"failed={failed} -> {obj_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
