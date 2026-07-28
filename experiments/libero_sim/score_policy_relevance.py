#!/usr/bin/env python3
# ruff: noqa: T201
"""v1 policy-relevance (occupancy proxy) for the `decaware` active-learning arm.

Re-weights the object-curiosity grade score by an OCCUPANCY PROXY: how close a
candidate episode's OBJECT trajectory is to the on-task manifold. For v1 the manifold
is the pool's own `reach` bucket (object-directed reaches — the most policy-relevant
motion in the mix); an off-task `random` flail lands far from it and is down-weighted.

    score_combined(ep) = max(grade_score(ep), 0) * occupancy(ep)
    occupancy(ep)      = exp(-lambda * d(ep) / median_d)     in (0, 1]
    d(ep)              = min L2 distance from feat(ep) to the reach-bucket features

`feat(ep)` = per-episode mean (over its chunks) of the object-center trajectory
(mean over objects, resampled to F_FIX frames, flattened). CHEAP: reads only the
already-built `objbbox/` corners on disk — no sim, no policy asset, no proprio.
max(.,0) clamps: an episode the current model is already worse-than-reference on
(negative progress) is not a learnable pick, so occupancy can't rescue it.

Output: a json identical to the grade json but with `by_episode` replaced by the
combined score (per-episode `grade_score`/`occupancy`/`policy_rel` kept for inspection),
which al_select_and_grow.py reads unchanged. UPGRADE PATH: swap the reach-manifold for
the FM-policy occupancy + action-sensitivity (see memory fm_bc_policy_asset) — same I/O.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

F_FIX = 16  # frames the per-episode object trajectory is resampled to


def _chunk_feat(corners: torch.Tensor) -> np.ndarray | None:
    """(n_obj, F, 8, 3) object corners -> flat (F_FIX*3,) mean-object center trajectory."""
    if corners is None or corners.numel() == 0 or corners.shape[0] == 0:
        return None
    c = corners.float().numpy()                 # (n_obj, F, 8, 3)
    center = c.mean(axis=2).mean(axis=0)        # mean over 8 corners, then over objects -> (F, 3)
    f = center.shape[0]
    if f < 1:
        return None
    idx = np.linspace(0, f - 1, F_FIX).round().astype(int)
    return center[idx].reshape(-1)              # (F_FIX*3,)


def _episode_features(data_root: Path, ep_chunks: dict[str, list[str]]) -> dict[str, np.ndarray]:
    """{episode: mean over its chunks of _chunk_feat}. Chunks with no objbbox are skipped."""
    feats: dict[str, np.ndarray] = {}
    for ep, cids in ep_chunks.items():
        vs = []
        for cid in cids:
            f = data_root / "objbbox" / "videos" / f"{cid}.pt"
            if not f.exists():
                continue
            d = torch.load(f, map_location="cpu", weights_only=False)
            v = _chunk_feat(d.get("corners") if isinstance(d, dict) else d)
            if v is not None:
                vs.append(v)
        if vs:
            feats[ep] = np.stack(vs).mean(0)
    return feats


def main() -> None:
    ap = argparse.ArgumentParser(description="v1 policy-relevance re-weighting of the grade score")
    ap.add_argument("--grade-json", required=True, help="objbbox/latent score json (has by_episode + per_chunk)")
    ap.add_argument("--data-root", required=True, help="precomputed/ root with objbbox/videos/*.pt")
    ap.add_argument("--out", required=True, help="combined json (by_episode = grade * occupancy)")
    ap.add_argument("--manifold-kind", default="reach", help="candidate bucket that defines the on-task manifold")
    ap.add_argument("--lambda-occ", type=float, default=2.0, help="occupancy decay (times median distance)")
    args = ap.parse_args()

    grade = json.loads(Path(args.grade_json).read_text())
    by_ep = grade.get("by_episode", {})
    per_chunk = grade.get("per_chunk", {})
    if not by_ep:
        raise SystemExit(f"grade json {args.grade_json} has no by_episode")

    # episode -> its chunk ids, and episode -> kind (from per_chunk records)
    ep_chunks: dict[str, list[str]] = {}
    ep_kind: dict[str, str] = {}
    for cid, rec in per_chunk.items():
        ep = rec.get("episode", cid)
        ep_chunks.setdefault(ep, []).append(cid)
        ep_kind.setdefault(ep, rec.get("kind", "?"))

    feats = _episode_features(Path(args.data_root), ep_chunks)
    manifold = [feats[e] for e in feats if ep_kind.get(e) == args.manifold_kind]

    occ: dict[str, float] = {}
    if not manifold:
        print(f"[policy_rel] no '{args.manifold_kind}' episodes with features -> occupancy=1 for all "
              f"(policy relevance is a no-op this round)")
        occ = {e: 1.0 for e in by_ep}
    else:
        M = np.stack(manifold)
        dist: dict[str, float] = {}
        for e in by_ep:
            if e in feats:
                dist[e] = float(np.sqrt(((M - feats[e]) ** 2).sum(axis=1)).min())
            else:
                dist[e] = float("nan")
        finite = [d for d in dist.values() if np.isfinite(d)]
        med = float(np.median(finite)) if finite else 1.0
        med = med if med > 1e-9 else 1.0
        for e, d in dist.items():
            occ[e] = float(np.exp(-args.lambda_occ * d / med)) if np.isfinite(d) else 0.0

    # combined = max(grade, 0) * occupancy. Keep the pieces per-episode for inspection.
    combined: dict[str, float] = {}
    detail: dict[str, dict] = {}
    for e, g in by_ep.items():
        gpos = max(float(g), 0.0)
        combined[e] = gpos * occ.get(e, 0.0)
        detail[e] = {"grade_score": float(g), "occupancy": occ.get(e, 0.0),
                     "policy_rel": combined[e], "kind": ep_kind.get(e, "?")}

    out_payload = dict(grade)
    out_payload["by_episode"] = combined
    out_payload["by_episode_grade"] = by_ep
    out_payload["policy_rel_detail"] = detail
    out_payload.setdefault("meta", {})["policy_rel"] = {
        "version": "v1_gt_manifold", "manifold_kind": args.manifold_kind, "lambda_occ": args.lambda_occ,
        "n_manifold": len(manifold),
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out_payload, indent=2))

    ranked = sorted(detail.items(), key=lambda kv: -kv[1]["policy_rel"])[:8]
    print(f"[policy_rel] wrote {outp} (manifold={args.manifold_kind}, n={len(manifold)})")
    print("[policy_rel] top combined (grade x occupancy):")
    for e, d in ranked:
        print(f"   {d['policy_rel']:.4f} = grade {d['grade_score']:.4f} x occ {d['occupancy']:.3f}  "
              f"[{d['kind']:7s}] {e[:48]}")


if __name__ == "__main__":
    main()
