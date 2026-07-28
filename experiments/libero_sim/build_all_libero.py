#!/usr/bin/env python
"""Resumable driver: build LIBERO-90 context for EVERY demo of EVERY task.

Walks all 90 hdf5 files, enumerates their real demo_N keys (~50 each, ~4500
total), and runs build_libero_context.py (--mode both) per demo into a flat
<out_root>/<ep_id>/ tree. Resume is marker-based:

  <out_root>/_done/<ep_id>    written on rc==0  (skip next time)
  <out_root>/_failed/<ep_id>  written on rc!=0  (skip + recorded, don't loop forever)
  <out_root>/_total           the enumerated work count (for the monitor)

A killed/preempted job just restarts the loop and skips anything already
marked — the in-flight demo re-runs from scratch (its partial output is
overwritten). Safe to run under monitor_libero_build.sh.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build_libero_context.py"
DATASET_DEFAULT = "/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/LIBERO-datasets/libero_90"
OUT_DEFAULT = "/storage/scratch1/8/lwang831/libero90_context"


def ep_id_for(hdf5: str, demo: int) -> str:
    # Full, untruncated, collision-free (task stem + demo index).
    return f"{Path(hdf5).stem}_demo_{demo}"


def enumerate_work(dataset: str) -> list[tuple[str, int]]:
    files = sorted(glob.glob(os.path.join(dataset, "*.hdf5")))
    work: list[tuple[str, int]] = []
    for f in files:
        with h5py.File(f, "r") as h:
            demos = sorted(int(k.split("_")[1]) for k in h["data"].keys()
                           if k.startswith("demo_"))
        for dnum in demos:
            work.append((f, dnum))
    return work


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET_DEFAULT)
    ap.add_argument("--out_root", default=OUT_DEFAULT)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--max_demos", type=int, default=0, help="stop after building N (0=all); for smoke tests")
    ap.add_argument("--num_shards", type=int, default=1,
                    help="split the enumerated work into N disjoint shards (parallel workers)")
    ap.add_argument("--shard", type=int, default=0,
                    help="which shard this worker owns, 0..num_shards-1 (work index %% num_shards == shard)")
    args = ap.parse_args()

    if args.num_shards < 1:
        ap.error("--num_shards must be >= 1")
    if not (0 <= args.shard < args.num_shards):
        ap.error(f"--shard must be in [0, {args.num_shards - 1}], got {args.shard}")

    out = Path(args.out_root)
    done_dir = out / "_done"
    fail_dir = out / "_failed"
    done_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)

    work = enumerate_work(args.dataset)
    (out / "_total").write_text(str(len(work)))
    n_total = len(work)
    # Disjoint sharding: each worker owns a stable, non-overlapping slice of the
    # enumerated list (index %% num_shards == shard). Markers are global, so the
    # _done/_failed skip still applies within a shard; partitioning the index just
    # guarantees two workers never pick the same pending demo at the same time.
    if args.num_shards > 1:
        work = [wd for i, wd in enumerate(work) if i % args.num_shards == args.shard]
    n_done0 = len(list(done_dir.glob("*")))
    n_fail0 = len(list(fail_dir.glob("*")))
    print(f"[driver] shard {args.shard}/{args.num_shards}: {len(work)}/{n_total} demos | "
          f"already done={n_done0} failed={n_fail0} -> {out}", flush=True)

    built = 0
    for i, (f, dnum) in enumerate(work):
        eid = ep_id_for(f, dnum)
        if (done_dir / eid).exists() or (fail_dir / eid).exists():
            continue
        print(f"[driver] ({i + 1}/{len(work)}) building {eid}", flush=True)
        cmd = [sys.executable, "-u", str(BUILD),
               "--hdf5", f, "--demo", str(dnum), "--ep_id", eid,
               "--out_root", str(out), "--res", str(args.res), "--fps", str(args.fps),
               "--mode", "both"]
        t0 = time.time()
        rc = subprocess.run(cmd).returncode
        dt = time.time() - t0
        if rc == 0:
            (done_dir / eid).write_text(f"{dt:.0f}s")
            built += 1
            print(f"[driver] OK   {eid}  ({dt:.0f}s)  built_this_run={built}", flush=True)
        else:
            (fail_dir / eid).write_text(f"rc={rc} dt={dt:.0f}s")
            print(f"[driver] FAIL {eid}  rc={rc}", flush=True)
        if args.max_demos and built >= args.max_demos:
            print(f"[driver] reached --max_demos={args.max_demos}; stopping.", flush=True)
            break

    nd = len(list(done_dir.glob("*")))
    nf = len(list(fail_dir.glob("*")))
    print(f"[driver] pass complete. done={nd} failed={nf} total={len(work)} "
          f"({nd + nf}/{len(work)} resolved)", flush=True)


if __name__ == "__main__":
    main()
