#!/usr/bin/env python3
# ruff: noqa: T201
"""Correlation proof: does the cheap latent loss track the expensive pixel error?

Reads ``combined.json`` from al_pixel_score.py and reports Spearman + Pearson between
per-chunk latent denoising loss and pixel error (−PSNR and 1−SSIM), overall and per
view, plus per-kind means and a scatter. Copies the worst/best latent-loss chunk
clips for eyeballing face validity. CPU-only — re-runnable without the GPU pass.

The signal is valid if high latent-loss chunks are also the low-PSNR (visibly wrong)
chunks, i.e. corr(latent_loss, PSNR) is clearly NEGATIVE / corr(latent_loss, 1−SSIM)
clearly POSITIVE, with Spearman ρ around 0.5+.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def _avg_rank(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties shared) — for Spearman without a scipy dependency."""
    a = np.asarray(a, dtype=float)
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_avg_rank(x), _avg_rank(y))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", required=True, help="combined.json from al_pixel_score.py")
    ap.add_argument("--out-dir", default=None, help="defaults to combined.json's dir")
    ap.add_argument("--top-k", type=int, default=5, help="worst/best example clips to copy")
    args = ap.parse_args()

    combined = Path(args.combined)
    out_dir = Path(args.out_dir) if args.out_dir else combined.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(combined.read_text())
    per_chunk = data["per_chunk"]
    cids = list(per_chunk.keys())
    if len(cids) < 3:
        raise SystemExit(f"Need >=3 chunks for correlation, got {len(cids)}")

    def col(key):
        return np.array([per_chunk[c][key] for c in cids], dtype=float)

    latent = col("latent_loss")
    psnr_all = col("psnr_all")
    ssim_all = col("ssim_all")
    err_psnr = -psnr_all          # higher = more pixel error
    err_ssim = 1.0 - ssim_all     # higher = more pixel error

    lines: list[str] = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 78)
    out(f"Active-learning signal validity — latent loss vs pixel error  (n={len(cids)})")
    out(f"  source: {combined}")
    out("=" * 78)
    out("\nGoal: high latent-loss == high pixel error. So expect:")
    out("  corr(latent, PSNR)   NEGATIVE   (more loss -> lower PSNR)")
    out("  corr(latent, 1-SSIM) POSITIVE   (more loss -> lower SSIM)")
    out("  Spearman |rho| ~ 0.5+ is a usable ranking signal.\n")

    for name, arr, expect in [
        ("PSNR_all (expect rho<0)", psnr_all, "neg"),
        ("1-SSIM_all (expect rho>0)", err_ssim, "pos"),
        ("-PSNR_all err (expect rho>0)", err_psnr, "pos"),
    ]:
        sp = _spearman(latent, arr)
        pe = _pearson(latent, arr)
        verdict = "OK" if ((expect == "neg" and sp < -0.3) or (expect == "pos" and sp > 0.3)) else "WEAK"
        out(f"  latent vs {name:30s}  Spearman={sp:+.3f}  Pearson={pe:+.3f}   [{verdict}]")

    # Per-view (latent is combined; pixel split into agentview/wrist)
    out("\nPer-view (latent loss is whole-frame; pixel split into tiles):")
    for view in ("agentview", "wrist"):
        pv = col(f"psnr_{view}")
        out(f"  latent vs PSNR_{view:9s}  Spearman={_spearman(latent, pv):+.3f}  Pearson={_pearson(latent, pv):+.3f}")

    # Per-kind means (sanity: off-manifold kinds should have higher loss AND lower PSNR)
    out("\nPer-kind means (sanity — random/circle harder than perturb/gt):")
    kinds: dict[str, list[int]] = {}
    for idx, c in enumerate(cids):
        kinds.setdefault(per_chunk[c]["kind"], []).append(idx)
    out(f"  {'kind':8s} {'n':>4s} {'latent_loss':>12s} {'PSNR_all':>9s} {'SSIM_all':>9s}")
    for k, idxs in sorted(kinds.items()):
        ii = np.array(idxs)
        out(f"  {k:8s} {len(idxs):4d} {latent[ii].mean():12.4f} {psnr_all[ii].mean():9.2f} {ssim_all[ii].mean():9.4f}")

    # ---- scatter ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        palette = {"perturb": "tab:blue", "random": "tab:red", "circle": "tab:green",
                   "reach": "tab:orange", "gt": "tab:gray"}
        for k, idxs in sorted(kinds.items()):
            ii = np.array(idxs)
            ax.scatter(latent[ii], psnr_all[ii], s=18, alpha=0.7,
                       c=palette.get(k, "black"), label=f"{k} (n={len(idxs)})")
        ax.set_xlabel("cheap latent denoising loss (AL signal)")
        ax.set_ylabel("pixel PSNR (all views)  — lower = more wrong")
        ax.set_title(f"latent loss vs pixel PSNR  (Spearman={_spearman(latent, psnr_all):+.3f}, n={len(cids)})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "corr_scatter.png", dpi=130)
        out(f"\nWrote {out_dir/'corr_scatter.png'}")
    except Exception as ex:  # noqa: BLE001
        out(f"\n[warn] scatter skipped: {ex}")

    # ---- worst/best example clips ----
    order = sorted(cids, key=lambda c: per_chunk[c]["latent_loss"])
    low, high = order[: args.top_k], order[-args.top_k:][::-1]
    for label, group in [("examples_high", high), ("examples_low", low)]:
        d = out_dir / label
        d.mkdir(parents=True, exist_ok=True)
        out(f"\n{label} (latent loss {'highest' if 'high' in label else 'lowest'}):")
        for c in group:
            src = Path(per_chunk[c]["mp4"])
            tag = f"loss{per_chunk[c]['latent_loss']:.3f}_psnr{per_chunk[c]['psnr_all']:.1f}"
            if src.exists():
                shutil.copy(src, d / f"{tag}__{c[:50]}.mp4")
            out(f"  {per_chunk[c]['latent_loss']:.4f}  PSNR={per_chunk[c]['psnr_all']:5.2f}  {c[:60]}")

    (out_dir / "corr_report.txt").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_dir/'corr_report.txt'}")


if __name__ == "__main__":
    main()
