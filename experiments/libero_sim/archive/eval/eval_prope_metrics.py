#!/usr/bin/env python
"""Quantitative PRoPE vs non-PRoPE comparison from the stacked eval mp4s.

The eval videos from infer_action_cond.py are vertical stacks (pred|gt|ctx) where
each panel is 256x512 = [agentview | wrist] 1x2 tiles (see _stack_and_save).
We split out pred + gt and score pred-against-gt with PSNR + SSIM, overall and
per-view. The comparison is fair: prope and noprope share the IDENTICAL gt panel
and the same H.264 codec, so codec noise is common-mode and the ranking is valid.

Caveat: the first latent frame of each chunk is the clean observation (conditioning),
so it inflates both runs equally; we report full-sequence numbers (common-mode) and
also drop the first 8 pixel frames as a prediction-only view.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

PANEL_H = 256  # each panel (pred/gt/ctx) is 256 tall
TILE_W = 256   # each view tile is 256 wide; [agentview | wrist]


def split_panels(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T,H,W,3) stacked pred|gt|ctx -> (pred, gt), each (T,256,512,3)."""
    pred = v[:, 0:PANEL_H]
    gt = v[:, PANEL_H : 2 * PANEL_H]
    return pred, gt


def score(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Mean PSNR/SSIM over frames between two (T,H,W,3) uint8 stacks."""
    ps, ss = [], []
    for a, b in zip(pred, gt):
        ps.append(psnr(b, a, data_range=255))
        ss.append(ssim(b, a, channel_axis=2, data_range=255))
    return {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)), "frames": len(ps)}


def score_all(pred: np.ndarray, gt: np.ndarray, drop_first: int) -> dict:
    if drop_first > 0:
        pred, gt = pred[drop_first:], gt[drop_first:]
    out = {"all": score(pred, gt)}
    out["agentview"] = score(pred[:, :, :TILE_W], gt[:, :, :TILE_W])
    out["wrist"] = score(pred[:, :, TILE_W:], gt[:, :, TILE_W:])
    return out


def load(path: Path) -> np.ndarray:
    return np.asarray(iio.imread(path, plugin="pyav"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).parent / "outputs")
    ap.add_argument("--drop-first", type=int, default=8,
                    help="drop first N frames (conditioning) for the pred-only view")
    args = ap.parse_args()

    prope_dir = args.root / "eval_prope_episode"
    noprope_dir = args.root / "eval_noprope_episode"

    # Pair files by the EP_<episode> stem, across {act,noact} x {prope,noprope}.
    def index(d: Path, tag: str) -> dict:
        out = {}
        for f in sorted(d.glob(f"{tag}_*_EP_*.mp4")):
            # <tag>_<act|noact>_EP_<stem>.mp4
            name = f.stem
            cond = "act" if "_act_" in name else "noact"
            stem = name.split("_EP_", 1)[1]
            out[(cond, stem)] = f
        return out

    P = index(prope_dir, "prope")
    NP = index(noprope_dir, "noprope")
    keys = sorted(set(P) & set(NP))
    if not keys:
        print("No paired episodes found.")
        return 1

    rows = []
    for cond, stem in keys:
        pv = load(P[(cond, stem)])
        nv = load(NP[(cond, stem)])
        pp, pg = split_panels(pv)
        npp, ng = split_panels(nv)
        # sanity: GT panels must match (same source episode)
        n = min(len(pg), len(ng))
        gt_diff = float(np.abs(pg[:n].astype(np.int16) - ng[:n].astype(np.int16)).mean())
        ps = score_all(pp, pg, args.drop_first)
        ns = score_all(npp, ng, args.drop_first)
        rows.append((cond, stem, ps, ns, gt_diff))

    # ---- report ----
    def fmt(s):
        return f"PSNR {s['psnr']:6.2f}  SSIM {s['ssim']:.4f}"

    print(f"\n{'='*78}\nPRoPE vs non-PRoPE  (pred-vs-GT, first {args.drop_first} frames dropped)\n{'='*78}")
    for cond, stem, ps, ns, gt_diff in rows:
        print(f"\n### {stem}  [{cond}]   (gt-panel match: mean|Δ|={gt_diff:.2f}/255)")
        for view in ("all", "agentview", "wrist"):
            p, n = ps[view], ns[view]
            dpsnr = p["psnr"] - n["psnr"]
            dssim = p["ssim"] - n["ssim"]
            win = "PRoPE" if (dpsnr > 0) else "noPRoPE"
            print(f"  {view:10s}  PRoPE[{fmt(p)}]  noPRoPE[{fmt(n)}]  "
                  f"ΔPSNR {dpsnr:+5.2f} ΔSSIM {dssim:+.4f}  -> {win}")

    # ---- aggregate over act episodes (the model of interest) ----
    print(f"\n{'='*78}\nAGGREGATE (mean over episodes, ACT only)\n{'='*78}")
    for view in ("all", "agentview", "wrist"):
        ap_ = [ps for c, s, ps, ns, g in rows if c == "act"]
        an_ = [ns for c, s, ps, ns, g in rows if c == "act"]
        if not ap_:
            continue
        pp_psnr = np.mean([x[view]["psnr"] for x in ap_])
        np_psnr = np.mean([x[view]["psnr"] for x in an_])
        pp_ssim = np.mean([x[view]["ssim"] for x in ap_])
        np_ssim = np.mean([x[view]["ssim"] for x in an_])
        print(f"  {view:10s}  PRoPE PSNR {pp_psnr:6.2f} SSIM {pp_ssim:.4f}   "
              f"noPRoPE PSNR {np_psnr:6.2f} SSIM {np_ssim:.4f}   "
              f"ΔPSNR {pp_psnr-np_psnr:+5.2f} ΔSSIM {pp_ssim-np_ssim:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
