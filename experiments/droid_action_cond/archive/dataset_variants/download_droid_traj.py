#!/usr/bin/env python
"""Download raw-DROID trajectory.h5 + metadata json for our context episodes.

Episode IDs (= raw-DROID uuids, `{lab}+{hash}+{ts}`) are the directory names
under the context dataset root, or come from the enumeration universe json.
The raw scene folder is resolved via episode_id_to_path.json from the DROID
calibration release (exact, handles scenes filed under a different date folder
than the episode timestamp); episodes absent from the map fall back to the
public GCS list API (matchGlob on metadata_<uuid>.json under
robotics/droid_raw/1.0.1/<lab>/{success,failure}/<date>/). trajectory.h5 +
metadata are fetched over plain HTTPS. Resumable: episodes with both files
present are skipped. Failures are logged and retried once at the end.

Output layout: <OUT>/<ep>/trajectory.h5, <OUT>/<ep>/metadata.json
Env: TRAJ_OUT (default /storage/project/r-agarg35-0/lwang831/droid/raw_traj)
     TRAJ_CONTEXT_ROOT, TRAJ_UNIVERSE (chunk-universe json; overrides context
     root as the episode source), TRAJ_WORKERS (default 8)
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CONTEXT = Path(os.environ.get(
    "TRAJ_CONTEXT_ROOT", "/storage/scratch1/8/lwang831/gvt_dataset_full/scratch/context"))
UNIVERSE = os.environ.get("TRAJ_UNIVERSE", "")
OUT = Path(os.environ.get(
    "TRAJ_OUT", "/storage/project/r-agarg35-0/lwang831/droid/raw_traj"))
WORKERS = int(os.environ.get("TRAJ_WORKERS", "8"))
BUCKET = "https://storage.googleapis.com/gresearch/"
LIST_API = "https://storage.googleapis.com/storage/v1/b/gresearch/o?"
EP_TO_PATH = Path(os.environ.get(
    "DROID_CALIB_DIR", "/storage/project/r-agarg35-0/lwang831/droid/calibration"
)) / "episode_id_to_path.json"


def gcs_list(prefix: str, glob: str) -> list[str]:
    url = LIST_API + urllib.parse.urlencode(
        {"prefix": prefix, "matchGlob": glob, "fields": "items(name)"})
    with urllib.request.urlopen(url, timeout=60) as r:
        return [it["name"] for it in json.load(r).get("items", [])]


def fetch(name: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(
            BUCKET + urllib.parse.quote(name), timeout=120) as r, \
            open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(dest)


EP_PATH_MAP: dict[str, str] = (
    json.loads(EP_TO_PATH.read_text()) if EP_TO_PATH.exists() else {})


def download_episode(ep: str) -> str:
    try:
        return _download_episode(ep)
    except Exception as e:  # transient network errors -> retried at the end
        return f"ERR {type(e).__name__}: {e}"


def _download_episode(ep: str) -> str:
    d = OUT / ep
    traj, meta = d / "trajectory.h5", d / "metadata.json"
    if traj.exists() and meta.exists():
        return "skip"
    if ep in EP_PATH_MAP:
        scene = f"robotics/droid_raw/1.0.1/{EP_PATH_MAP[ep]}"
        meta_name = f"{scene}/metadata_{ep}.json"
    else:  # fallback: search by date folder derived from the timestamp
        lab, _, ts = ep.split("+")
        date = ts[:10]
        meta_name = None
        for split in ("success", "failure"):
            hits = gcs_list(f"robotics/droid_raw/1.0.1/{lab}/{split}/{date}/",
                            f"**metadata_{ep}.json")
            if hits:
                meta_name = hits[0]
                break
        if meta_name is None:
            return "NOT_FOUND"
        scene = meta_name.rsplit("/", 1)[0]
    d.mkdir(parents=True, exist_ok=True)
    fetch(meta_name, meta)
    fetch(f"{scene}/trajectory.h5", traj)
    return "ok"


def run(eps: list[str]) -> list[str]:
    failed = []
    done = 0
    with ThreadPoolExecutor(WORKERS) as pool:
        for ep, status in zip(eps, pool.map(download_episode, eps)):
            done += 1
            if status not in ("ok", "skip"):
                failed.append(ep)
                print(f"FAIL {ep}: {status}", flush=True)
            if done % 25 == 0:
                print(f"[{done}/{len(eps)}] ({len(failed)} failed)", flush=True)
    return failed


def main():
    if UNIVERSE:
        chunks = json.loads(Path(UNIVERSE).read_text())
        eps = sorted({c["ep_short"] for c in chunks})
    else:
        eps = sorted(p.name for p in CONTEXT.iterdir()
                     if (p / "data.npz").exists())
    print(f"{len(eps)} episodes -> {OUT}", flush=True)
    failed = run(eps)
    if failed:
        print(f"retrying {len(failed)} failures...", flush=True)
        failed = run(failed)
    (OUT / "_failed.json").write_text(json.dumps(failed, indent=1))
    n_ok = sum(1 for ep in eps if (OUT / ep / "trajectory.h5").exists())
    print(f"done: {n_ok}/{len(eps)} episodes downloaded, "
          f"{len(failed)} permanent failures", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
