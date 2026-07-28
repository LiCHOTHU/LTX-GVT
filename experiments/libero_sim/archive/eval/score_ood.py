#!/usr/bin/env python
"""Score the held-out OOD eval (eval_ood.sbatch outputs), split rand vs gt+noise.

Pairs the full-stem episode mp4s (`<tag>_act_EP_<stem>.mp4`) across prope/noprope
and reports PSNR/SSIM SEPARATELY for the two OOD action conditions:
  * random  (`..._play_random_demo_N`)   = uniform-random actions
  * perturb (`..._play_perturb_demo_N`)  = GT actions + OU noise
Reuses eval_prope_metrics' panel-split + scoring. Saves a .txt + .json.

Usage:
  python score_ood.py --root outputs/eval_ood_step54000 [--drop-first 8]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

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


def play_kind(stem: str) -> str:
    for k in ("random", "perturb", "reach", "circle"):
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
            "kind": play_kind(stem),
            "prope": score_all(pp, pg, args.drop_first),
            "noprope": score_all(npp, ng, args.drop_first),
            "gt_panel_match": gt_diff,
        })

    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out(f"{'='*86}")
    out(f"OOD eval  (pred-vs-GT, first {args.drop_first} frames dropped)   root={args.root}")
    out(f"{'='*86}")
    for r in sorted(rows, key=lambda r: (r["kind"], r["stem"])):
        out(f"\n### [{r['kind']}] {r['stem'][:70]}  [{r['cond']}]   "
            f"(gt-panel |Δ|={r['gt_panel_match']:.2f}/255)")
        for view in ("all", "agentview", "wrist"):
            p, n = r["prope"][view], r["noprope"][view]
            dp = p["psnr"] - n["psnr"]; ds = p["ssim"] - n["ssim"]
            win = "PRoPE" if dp > 0 else "noPRoPE"
            out(f"  {view:10s}  PRoPE[PSNR {p['psnr']:6.2f} SSIM {p['ssim']:.4f}]  "
                f"noPRoPE[PSNR {n['psnr']:6.2f} SSIM {n['ssim']:.4f}]  "
                f"ΔPSNR {dp:+5.2f} ΔSSIM {ds:+.4f} -> {win}")

    # ---- aggregates: ACT only, split by OOD condition ----
    for kind in ("gt", "random", "perturb", "circle"):
        sub = [r for r in rows if r["kind"] == kind and r["cond"] == "act"]
        if not sub:
            continue
        label = {"gt": "GT (on-manifold real-demo replay = reconstruction reference)",
                 "random": "RANDOM (uniform-random actions)",
                 "perturb": "PERTURB (GT actions + OU noise)",
                 "circle": "CIRCLE (scripted descend + draw circle)"}[kind]
        out(f"\n{'='*86}")
        out(f"AGGREGATE — {label} (mean over ACT episodes, n={len(sub)})")
        out(f"{'='*86}")
        for view in ("all", "agentview", "wrist"):
            pp = float(np.mean([r["prope"][view]["psnr"] for r in sub]))
            npp = float(np.mean([r["noprope"][view]["psnr"] for r in sub]))
            ps = float(np.mean([r["prope"][view]["ssim"] for r in sub]))
            ns = float(np.mean([r["noprope"][view]["ssim"] for r in sub]))
            out(f"  {view:10s}  PRoPE PSNR {pp:6.2f} SSIM {ps:.4f}   "
                f"noPRoPE PSNR {npp:6.2f} SSIM {ns:.4f}   "
                f"ΔPSNR {pp-npp:+5.2f} ΔSSIM {ps-ns:+.4f}")

    txt = args.root / "scores_ood.txt"
    js = args.root / "scores_ood.json"
    txt.write_text("\n".join(lines) + "\n")
    js.write_text(json.dumps(rows, indent=2))
    out(f"\nSaved: {txt}\n       {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
