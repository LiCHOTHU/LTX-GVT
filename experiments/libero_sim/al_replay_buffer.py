#!/usr/bin/env python3
# ruff: noqa: T201
"""Bounded replay buffer for the closed-loop active-learning monitor.

The buffer is the PERTURB SEED: each round the `perturb` policy draws its base
trajectory from here and adds noise (gen_libero_play.py --perturb-source). It is a
FIFO of episode references (NOT the precomputed latents — the raw states/actions a
perturb rollout needs), capped at --cap entries so it forgets stale-hard regions
and keeps perturbation moving (the bounded-buffer decision).

A buffer ENTRY is one episode: {"hdf5", "demo", "task", "round"}.
  hdf5  : absolute path to a LIBERO-format file (raw libero_90 demo, or a kept play
          episode produced in a previous round)
  demo  : group key inside it ("demo_0", ...)
  task  : task key (scene+task identity); perturb only replays within the same task
          because an action sequence is only valid in the scene it was recorded in
  round : when it entered (0 = GT seed); FIFO evicts the lowest round first

Subcommands:
  seed    : fill the buffer with PER_TASK random GT demos from every libero_90 task
            (stratified coverage) -> manifest.json, round 0.
  append  : add the episodes kept this round (a dir of play HDF5s) and evict down to
            --cap by round order (oldest first). GT seed entries bleed out naturally.
  stats   : print buffer composition (entries, tasks covered, per-round counts).

gen_libero_play.py imports task_key_of() + bases_for_task() so the perturb source
contract lives in ONE place.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py


def task_key_of(stem_or_path: str) -> str:
    """Task identity from a file stem: strip a trailing _demo or _play_<policy>.

    'KITCHEN_SCENE10_close_the_top_drawer_demo'          -> 'KITCHEN_SCENE10_close_the_top_drawer'
    'KITCHEN_SCENE10_close_the_top_drawer_play_perturb'  -> same task key
    """
    stem = Path(stem_or_path).stem
    stem = re.sub(r"_play_(reach|perturb|random|circle|gt)$", "", stem)
    stem = re.sub(r"_demo$", "", stem)
    return stem


def _demo_keys(hdf5: Path) -> list[str]:
    with h5py.File(hdf5, "r") as h:
        return sorted(h["data"].keys(), key=lambda k: int(k.split("_")[1]))


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def save_manifest(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(entries, indent=2))
    tmp.rename(path)


def bases_for_task(entries: list[dict], task_key: str) -> list[dict]:
    """Buffer entries whose task matches — the legal perturb bases for this scene."""
    return [e for e in entries if e["task"] == task_key]


# --------------------------------------------------------------------------- seed
def cmd_seed(args: argparse.Namespace) -> None:
    libero90 = Path(args.libero90)
    files = sorted(libero90.glob("*.hdf5"))
    if not files:
        raise SystemExit(f"No task HDF5s under {libero90}")
    import random

    rng = random.Random(args.seed)
    entries: list[dict] = []
    for f in files:
        keys = _demo_keys(f)
        k = min(args.per_task, len(keys))
        for dk in rng.sample(keys, k):
            entries.append({"hdf5": str(f.resolve()), "demo": dk,
                            "task": task_key_of(f.stem), "round": 0})
    save_manifest(Path(args.manifest), entries)
    print(f"[seed] {len(entries)} GT entries from {len(files)} tasks "
          f"({args.per_task}/task) -> {args.manifest}")


def _resolve_episode(stem: str, kept_dir: Path) -> tuple[Path, str] | None:
    """Episode stem '<task>_play_<policy>_demo_N' -> (play hdf5 path, 'demo_N')."""
    m = re.search(r"_(demo_\d+)$", stem)
    if not m:
        return None
    demo = m.group(1)
    play_stem = stem[: m.start()]            # '<task>_play_<policy>'
    f = kept_dir / f"{play_stem}.hdf5"
    return (f, demo) if f.exists() else None


# --------------------------------------------------------------------------- append
def cmd_append(args: argparse.Namespace) -> None:
    manifest = Path(args.manifest)
    entries = load_manifest(manifest)
    kept_dir = Path(args.kept_dir)
    have = {(e["hdf5"], e["demo"]) for e in entries}
    added = 0

    # Pairs to add: either the exact kept episodes (--picked) or every demo of every
    # hdf5 in kept_dir. --picked is what the monitor uses (only the selected half).
    pairs: list[tuple[Path, str]] = []
    if args.picked:
        picked_path = Path(args.picked)
        picked_stems = (
            [s.strip() for s in picked_path.read_text().splitlines() if s.strip()]
            if picked_path.exists() else []
        )
        if not picked_stems:
            print(f"[append] no picked episodes ({args.picked}) — nothing to add this round.")
        for stem in picked_stems:
            r = _resolve_episode(stem, kept_dir)
            if r is None:
                print(f"[append] WARN cannot resolve picked episode {stem} under {kept_dir}")
            else:
                pairs.append(r)
    else:
        for f in sorted(kept_dir.glob("*.hdf5")):
            pairs.extend((f, dk) for dk in _demo_keys(f))

    for f, dk in pairs:
        key = (str(f.resolve()), dk)
        if key in have:
            continue
        entries.append({"hdf5": str(f.resolve()), "demo": dk,
                        "task": task_key_of(f.stem), "round": args.round})
        have.add(key)
        added += 1
    # FIFO evict: keep the newest --cap by (round, stable order). Oldest round first out.
    before = len(entries)
    if before > args.cap:
        # stable sort by round ascending, then drop the front overflow
        order = sorted(range(len(entries)), key=lambda i: entries[i]["round"])
        drop = set(order[: before - args.cap])
        entries = [e for i, e in enumerate(entries) if i not in drop]
    save_manifest(manifest, entries)
    print(f"[append] round {args.round}: +{added} entries, evicted "
          f"{max(0, before - args.cap)}, buffer now {len(entries)}/{args.cap}")


# --------------------------------------------------------------------------- stats
def cmd_stats(args: argparse.Namespace) -> None:
    entries = load_manifest(Path(args.manifest))
    tasks = {e["task"] for e in entries}
    by_round: dict[int, int] = {}
    for e in entries:
        by_round[e["round"]] = by_round.get(e["round"], 0) + 1
    print(f"[stats] {len(entries)} entries, {len(tasks)} tasks covered")
    for r in sorted(by_round):
        print(f"          round {r}: {by_round[r]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Bounded replay buffer (perturb seed) for AL")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("seed", help="seed with PER_TASK GT demos from every task")
    s.add_argument("--libero90", required=True, help="dir of raw libero_90 task HDF5s")
    s.add_argument("--manifest", required=True)
    s.add_argument("--per-task", type=int, default=3)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_seed)

    a = sub.add_parser("append", help="add this round's kept play episodes + FIFO evict")
    a.add_argument("--manifest", required=True)
    a.add_argument("--kept-dir", required=True, help="this round's PLAY dir (holds the hdf5s)")
    a.add_argument("--picked", default=None,
                   help="file of kept episode stems (from al_select_and_grow --picked-out); "
                        "only these are ingested. Omit -> every demo in --kept-dir.")
    a.add_argument("--round", type=int, required=True)
    a.add_argument("--cap", type=int, default=270)
    a.set_defaults(func=cmd_append)

    t = sub.add_parser("stats", help="print buffer composition")
    t.add_argument("--manifest", required=True)
    t.set_defaults(func=cmd_stats)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
