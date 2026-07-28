#!/usr/bin/env python
"""Score the GT+play eval (eval_gtplay.sbatch outputs), grouped by GT vs play.

Reuses eval_prope_metrics' panel-split + scoring, but pairs the full-stem episode
mp4s (`<tag>_<act|noact>_EP_<stem>.mp4`) across prope/noprope and reports PSNR/SSIM
SEPARATELY for the on-manifold GT task demos (`..._demo_<n>`) and the off-manifold
play episodes (`..._play_{reach,perturb,random}_demo_<n>`). Saves a .txt + .json.

Usage:
  python score_gtplay.py --root outputs/eval_gtplay_step54000 [--drop-first 8]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# panel split + scoring live in the sibling metrics script
from eval_prope_metrics import load, score_all, split_panels


def index(d: Path, tag: str) -> dict:
    """(cond, stem) -> mp4 path, from '<tag>_<act|noact>_EP_<stem>.mp4'."""
    out = {}
    for f in sorted(d.glob(f"{tag}_*_EP_*.mp4")):
        name = f.stem
        cond = "act" if "_act_" in name else "noact"
        stem = name.split("_EP_", 1)[1]
        out[(cond, stem)] = f
    return out


def is_play(stem: str) -> bool:
    return "_play_" in stem


def play_kind(stem: str) -> str:
    for k in ("reach", "perturb", "random"):
        if f"_play_{k}_" in stem:
            return k
    return "gt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="dir holding eval_prope_episode/ and eval_noprope_episode/")
    ap.add_argument("--drop-first", type=int, default=8)
    args = ap.parse_args()

    P = index(args.root / "eval_prope_episode", "prope")
    NP = index(args.root / "eval_noprope_episode", "noprope")
    keys = sorted(set(P) & set(NP))
    if not keys:
        print(f"No paired episodes under {args.root}")
        return 1

    rows = []
    for cond, stem in keys:
        pp, pg = split_panels(load(P[(cond, stem)]))
        npp, ng = split_panels(load(NP[(cond, stem)]))
        n = min(len(pg), len(ng))
        gt_diff = float(np.abs(pg[:n].astype(np.int16) - ng[:n].astype(np.int16)).mean())
        rows.append({
            "cond": cond, "stem": stem,
            "group": "play" if is_play(stem) else "gt",
            "kind": play_kind(stem),
            "prope": score_all(pp, pg, args.drop_first),
            "noprope": score_all(npp, ng, args.drop_first),
            "gt_panel_match": gt_diff,
        })

    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out(f"{'='*86}")
    out(f"GT+play eval  (pred-vs-GT, first {args.drop_first} frames dropped)   root={args.root}")
    out(f"{'='*86}")
    for r in rows:
        out(f"\n### [{r['group']}/{r['kind']}] {r['stem'][:64]}  [{r['cond']}]   "
            f"(gt-panel |Δ|={r['gt_panel_match']:.2f}/255)")
        for view in ("all", "agentview", "wrist"):
            p, n = r["prope"][view], r["noprope"][view]
            dp = p["psnr"] - n["psnr"]; ds = p["ssim"] - n["ssim"]
            win = "PRoPE" if dp > 0 else "noPRoPE"
            out(f"  {view:10s}  PRoPE[PSNR {p['psnr']:6.2f} SSIM {p['ssim']:.4f}]  "
                f"noPRoPE[PSNR {n['psnr']:6.2f} SSIM {n['ssim']:.4f}]  "
                f"ΔPSNR {dp:+5.2f} ΔSSIM {ds:+.4f} -> {win}")

    # ---- aggregates: ACT only, split GT vs play ----
    for group in ("gt", "play"):
        sub = [r for r in rows if r["group"] == group and r["cond"] == "act"]
        if not sub:
            continue
        out(f"\n{'='*86}")
        out(f"AGGREGATE — {group.upper()} (mean over ACT episodes, n={len(sub)})")
        out(f"{'='*86}")
        for view in ("all", "agentview", "wrist"):
            pp = float(np.mean([r["prope"][view]["psnr"] for r in sub]))
            npp = float(np.mean([r["noprope"][view]["psnr"] for r in sub]))
            ps = float(np.mean([r["prope"][view]["ssim"] for r in sub]))
            ns = float(np.mean([r["noprope"][view]["ssim"] for r in sub]))
            out(f"  {view:10s}  PRoPE PSNR {pp:6.2f} SSIM {ps:.4f}   "
                f"noPRoPE PSNR {npp:6.2f} SSIM {ns:.4f}   "
                f"ΔPSNR {pp-npp:+5.2f} ΔSSIM {ps-ns:+.4f}")

    txt = args.root / "scores_gtplay.txt"
    js = args.root / "scores_gtplay.json"
    txt.write_text("\n".join(lines) + "\n")
    js.write_text(json.dumps(rows, indent=2))
    out(f"\nSaved: {txt}\n       {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
