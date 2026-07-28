#!/usr/bin/env python3
# ruff: noqa: T201
"""PSNR(pred vs GT) for strategic vs random AL arms across expert/perturb/circle.

Reads the pred|gt|ctx panel mp4s under al_eval_step<STEP>/<arm>/<domain>/, splits
the pred and gt panels (eval_prope_metrics), scores PSNR/SSIM per episode (dropping
the first N conditioning frames), and averages per domain. The two arms share the
IDENTICAL gt panel + H.264 codec, so the comparison is fair (codec noise common-mode).
Pairs episodes by the EP_<stem> key so only matched episodes are compared."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval_prope_metrics import load, split_panels

DOMAINS = ["expert", "perturb", "circle"]
ARMS = ["strategic", "random"]
TILE_W = 256  # [agentview | wrist]


def _psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean per-frame PSNR (dB) over (T,H,W,3) uint8 stacks, vectorized."""
    a = pred.astype(np.float64)
    b = gt.astype(np.float64)
    mse = ((a - b) ** 2).reshape(a.shape[0], -1).mean(axis=1)
    mse = np.maximum(mse, 1e-12)  # avoid log(0) on identical frames
    return float(np.mean(10.0 * np.log10((255.0**2) / mse)))


def score_all(pred: np.ndarray, gt: np.ndarray, drop_first: int) -> dict:
    if drop_first > 0:
        pred, gt = pred[drop_first:], gt[drop_first:]
    return {
        "all": {"psnr": _psnr(pred, gt)},
        "agentview": {"psnr": _psnr(pred[:, :, :TILE_W], gt[:, :, :TILE_W])},
        "wrist": {"psnr": _psnr(pred[:, :, TILE_W:], gt[:, :, TILE_W:])},
    }


def stem_of(p: Path) -> str:
    return p.stem.split("_EP_", 1)[1] if "_EP_" in p.stem else p.stem


def index(d: Path) -> dict[str, Path]:
    return {stem_of(f): f for f in sorted(d.glob("*.mp4"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="al_eval_step<STEP>/ dir")
    ap.add_argument("--drop-first", type=int, default=8)
    ap.add_argument("--views", nargs="+", default=["all", "agentview", "wrist"])
    args = ap.parse_args()

    results: dict = {}
    for dom in DOMAINS:
        idx = {arm: index(args.root / arm / dom) for arm in ARMS}
        shared = sorted(set(idx["strategic"]) & set(idx["random"]))
        per_ep = []
        for stem in shared:
            row = {"episode": stem}
            for arm in ARMS:
                pred, gt = split_panels(load(idx[arm][stem]))
                row[arm] = score_all(pred, gt, args.drop_first)
            per_ep.append(row)
        # per-domain mean over episodes, per view
        agg = {}
        for view in args.views:
            for arm in ARMS:
                agg.setdefault(view, {})[arm] = {
                    "psnr": float(np.mean([e[arm][view]["psnr"] for e in per_ep])),
                }
        results[dom] = {"n_episodes": len(shared), "per_view": agg, "per_episode": per_ep}

    (args.root / "psnr_arms.json").write_text(json.dumps(results, indent=2))

    # ---- table ----
    print(f"\nStrategic vs Random — PSNR (pred vs GT), first {args.drop_first} frames dropped")
    print(f"root: {args.root}\n")
    for view in args.views:
        print(f"=== view: {view} ===")
        print(f"{'domain':<10} {'n':>3}  {'strategic':>10}  {'random':>10}  {'Δ(strat-rand)':>14}  winner")
        for dom in DOMAINS:
            n = results[dom]["n_episodes"]
            s = results[dom]["per_view"][view]["strategic"]["psnr"]
            r = results[dom]["per_view"][view]["random"]["psnr"]
            d = s - r
            win = "strategic" if d > 0 else "random"
            print(f"{dom:<10} {n:>3}  {s:>10.3f}  {r:>10.3f}  {d:>+14.3f}  {win}")
        print()
    print(f"Wrote {args.root}/psnr_arms.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
