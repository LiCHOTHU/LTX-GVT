"""Enumerate the (episode, chunk_idx) universe of training chunks for the GVT
v2v dataset build, filtered by camera-calibration availability + IoU quality.

Reads droid_100 RLDS shards (only on-disk DROID source today), filters to episodes
that have IoU-calibrated ext1/ext2 + wrist intrinsics, splits each episode into
33-frame chunks (overlap-tail), and writes a manifest:

  out_root/_universe.json   (atomic write)

The manifest is the single source of truth for the resumable builder:
  - the watchdog reads len(_universe.json) for total chunk count
  - the orchestrator iterates manifest entries to know what chunks to build

Manifest schema (list of dicts):
  {
    "cid": "IPRL_w026bb9b_2023-04-20-23h-40m-21s_c00",  # used for _done/<cid>
    "ep_id": "IPRL+w026bb9b+2023-04-20-23h-40m-21s",
    "ep_short": "IPRL+w026bb9b+2023-04-20-23h-40m-21s",
    "ep_index": 17,                # RLDS index, for stable lookup
    "chunk_idx": 0,
    "start": 0,                    # frame index in episode
    "end": 33,
    "n_steps": 89,                 # full episode length
    "iou": 0.92,                   # IoU calibration quality
    "institution": "IPRL",
  }

Run-once script (or re-run any time droid_100 RLDS contents change). Idempotent.

Usage:
  python enumerate_universe.py --out_root /storage/cedar/.../gvt_dataset_full
  python enumerate_universe.py --out_root <root> --min_iou 0.85 --chunk_len 33

Env vars honoured (consistent with the rest of the pipeline):
  DROID_CALIB_DIR, DROID_RLDS_DIR

This script does TF only — runs without GPU, fast (~20 s for 100 episodes).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "packages" / "ltx-action-cond" / "src"))

DEFAULT_OUT_ROOT = Path("/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/gvt_dataset_full")
CALIB_DIR = Path(os.environ.get("DROID_CALIB_DIR", "/storage/project/r-agarg35-0/lwang831/droid/calibration"))
RLDS_DIR = Path(os.environ.get("DROID_RLDS_DIR", "/storage/project/r-agarg35-0/lwang831/droid/droid_100/1.0.0"))


def short_id(ep_id: str) -> str:
    """Mirror wrist_demo.short_id so source dir names match the existing context/ layout."""
    return re.sub(r"[^A-Za-z0-9_+-]", "_", ep_id)[:40]


def cid_for(ep_short: str, chunk_idx: int) -> str:
    """Stable chunk ID used in _done/ markers. Mirrors list_chunks()."""
    return f"{ep_short.replace('+', '_')}_c{chunk_idx:02d}"


def chunk_starts(T: int, chunk: int) -> list[int]:
    """Non-overlapping windows + final overlap-tail to cover T fully. Same recipe
    as chunk_context.chunk_starts."""
    if T < chunk:
        return []
    starts = list(range(0, T - chunk + 1, chunk))
    if starts[-1] + chunk < T:
        starts.append(T - chunk)
    return starts


def _atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--min_iou", type=float, default=0.85)
    parser.add_argument("--chunk_len", type=int, default=33)
    parser.add_argument("--print_only", action="store_true",
                        help="Don't write _universe.json; just enumerate to stdout.")
    parser.add_argument("--max_episodes", type=int, default=0,
                        help="If >0, limit RLDS scan to the first N episodes (smoke-test).")
    args = parser.parse_args()

    # TF import deferred so --help works on nodes without TF.
    import tensorflow_datasets as tfds  # noqa: E402
    from ltx_action_cond.droid import (  # noqa: E402
        CalibrationIndex, find_iou_calibrated_episodes,
    )

    print(f"[enum] CALIB_DIR = {CALIB_DIR}")
    print(f"[enum] RLDS_DIR  = {RLDS_DIR}")
    print(f"[enum] min_iou   = {args.min_iou}")
    print(f"[enum] chunk_len = {args.chunk_len}")

    calib = CalibrationIndex.load(CALIB_DIR)
    builder = tfds.builder_from_directory(str(RLDS_DIR))
    ds = builder.as_dataset(split="train")
    if args.max_episodes > 0:
        ds = ds.take(args.max_episodes)
        print(f"[enum] SMOKE TEST: limited to first {args.max_episodes} episodes")

    cands = find_iou_calibrated_episodes(ds, calib, min_quality=args.min_iou)
    print(f"[enum] IoU-calibrated candidates >= {args.min_iou}: {len(cands)}")

    # Apply the same wrist-capable filter as wrist_demo.pick_episodes: must have
    # ext1/ext2/wrist serials + non-zero intrinsics.
    good = []
    for c in cands:
        sers = calib.camera_serials[c.episode_id]
        ext1 = sers.get("ext1_cam_serial")
        ext2 = sers.get("ext2_cam_serial")
        wrist = sers.get("wrist_cam_serial")
        intr = calib.intrinsics[c.episode_id]
        if not (ext1 and ext2 and wrist):
            continue
        if not all(s in intr and intr[s].get("width", 0) > 0 for s in (ext1, ext2, wrist)):
            continue
        good.append(c)
    print(f"[enum] after wrist-capable filter: {len(good)} / {len(cands)} survive")

    universe = []
    skipped_short = 0
    for c in sorted(good, key=lambda x: -x.quality):
        T = int(c.n_steps)
        starts = chunk_starts(T, args.chunk_len)
        if not starts:
            skipped_short += 1
            continue
        ep_short = short_id(c.episode_id)
        institution = c.episode_id.split("+")[0]
        for ci, st in enumerate(starts):
            universe.append({
                "cid": cid_for(ep_short, ci),
                "ep_id": c.episode_id,
                "ep_short": ep_short,
                "ep_index": int(c.ep_index),
                "chunk_idx": ci,
                "start": int(st),
                "end": int(st + args.chunk_len),
                "n_steps": T,
                "iou": float(c.quality),
                "institution": institution,
            })

    n_eps = len({u["ep_id"] for u in universe})
    print(f"[enum] universe: {len(universe)} chunks across {n_eps} episodes "
          f"(skipped {skipped_short} eps with T < {args.chunk_len})")

    if args.print_only:
        print(json.dumps(universe[:5], indent=2))
        print(f"... ({len(universe)} total)")
        return 0

    manifest_path = args.out_root / "_universe.json"
    _atomic_write_json(manifest_path, universe)
    print(f"[enum] wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
