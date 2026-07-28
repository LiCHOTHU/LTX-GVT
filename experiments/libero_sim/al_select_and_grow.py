#!/usr/bin/env python3
# ruff: noqa: T201
"""Active-learning selection + dataset growth (pool-based, one round).

Pick K candidate EPISODES from the pool that are not yet in the training set, then
symlink ALL their precomputed chunk files (every modality) into the growing training
root and append them to the added-manifest. Selection modes:

  strategic : highest mean latent denoising loss (the AL signal) — "where the model
              is most wrong" (reads by_episode from score_chunk_latent_loss's json).
  random    : uniform random (the control arm) — seeded for reproducibility.

Growth = symlinks, so it is cheap and the glob-based PrecomputedDataset picks the new
files up on the NEXT training (re)submit. The manifest makes it idempotent: an episode
is never added twice across rounds.

Used by monitor_active_finetune_libero.sh between training rounds.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch

# Every per-chunk modality the trainer may consume. cameras/ only exists for PRoPE
# pools; missing modalities for a chunk are skipped (not an error).
MODALITIES = ["latents", "reference_latents", "conditions", "actions", "cameras"]


def _episode_embedding(pool_root: Path, chunk_ids: list[str]) -> torch.Tensor | None:
    """Fixed-length (per-channel mean) latent embedding for an episode, averaged over its
    chunks. Used by the diversity selector (#1) so the kept batch covers latent space
    instead of being 20 near-duplicate high-loss clips (BatchBALD / core-set)."""
    vecs = []
    for cid in chunk_ids:
        f = pool_root / "latents" / "videos" / f"{cid}.pt"
        if not f.exists():
            continue
        d = torch.load(f, map_location="cpu", weights_only=False)
        t = d["latents"] if isinstance(d, dict) and "latents" in d else d
        t = t.float()
        vecs.append(t.mean(dim=tuple(range(1, t.ndim))))  # collapse all but channel dim
    if not vecs:
        return None
    return torch.stack(vecs).mean(0)


def _greedy_kcenter(embs: list[torch.Tensor], k: int) -> list[int]:
    """Greedy k-center (farthest-point) over episode embeddings; index 0 is the anchor
    (the highest-score episode, since the caller passes a score-sorted list). Returns the
    selected indices: hardest first, then the points that best spread out the coverage."""
    n = len(embs)
    selected = [0]
    mind = [float((embs[i] - embs[0]).pow(2).sum()) for i in range(n)]
    while len(selected) < min(k, n):
        nxt = max((i for i in range(n) if i not in selected), key=lambda i: mind[i])
        selected.append(nxt)
        for i in range(n):
            d = float((embs[i] - embs[nxt]).pow(2).sum())
            if d < mind[i]:
                mind[i] = d
    return selected


def episodes_in(root: Path) -> dict[str, list[str]]:
    """{episode_stem: [chunk_id, ...]} from <root>/latents/videos/*.pt."""
    eps: dict[str, list[str]] = {}
    for f in sorted((root / "latents" / "videos").glob("*.pt")):
        cid = f.stem  # <episode>_chunk_NN_target
        m = re.match(r"^(.*)_chunk_\d+_target$", cid)
        ep = m.group(1) if m else cid
        eps.setdefault(ep, []).append(cid)
    return eps


def main() -> None:
    ap = argparse.ArgumentParser(description="Select K pool episodes and grow the training root")
    ap.add_argument("--pool-root", required=True, help="candidate precomputed/ root")
    ap.add_argument("--grow-root", required=True, help="growing training precomputed/ root (symlink farm)")
    ap.add_argument("--manifest", required=True, help="text file of already-added episode stems")
    ap.add_argument("--k", type=int, required=True, help="episodes to add this round")
    ap.add_argument("--mode", choices=["strategic", "random"], required=True)
    ap.add_argument("--latent-json", default=None, help="score json (required for strategic)")
    ap.add_argument("--seed", type=int, default=0, help="random-mode seed")
    ap.add_argument("--diverse", action="store_true",
                    help="strategic only: take the top --oversample by score, then greedy "
                         "k-center over their latents to keep a DIVERSE k (#1, avoids the "
                         "BatchBALD redundancy trap of pure top-k). Default off == legacy top-k.")
    ap.add_argument("--oversample", type=int, default=0,
                    help="candidate pool size for --diverse (0 => 3*k).")
    ap.add_argument("--picked-out", default=None,
                    help="write the picked episode stems (one/line) — the replay buffer "
                         "ingests these as this round's kept set")
    args = ap.parse_args()

    pool_root = Path(args.pool_root)
    grow_root = Path(args.grow_root)
    manifest = Path(args.manifest)

    pool = episodes_in(pool_root)
    if not pool:
        raise SystemExit(f"No episodes under {pool_root}/latents/videos/")
    added = set()
    if manifest.exists():
        added = {ln.strip() for ln in manifest.read_text().splitlines() if ln.strip()}
    avail = [e for e in pool if e not in added]
    if not avail:
        print(f"[grow] pool EXHAUSTED ({len(added)} episodes already added) — nothing to add.")
        # Still emit an (empty) picked-out so the downstream replay-buffer append step
        # has a file to read instead of crashing on FileNotFoundError. Nothing to add
        # this round is a valid state (single-scene pools saturate once every episode
        # identity has been selected); the loop should train on, not die.
        if args.picked_out:
            Path(args.picked_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.picked_out).write_text("")
        return

    if args.mode == "strategic":
        if not args.latent_json:
            raise SystemExit("strategic mode requires --latent-json")
        by_ep = json.loads(Path(args.latent_json).read_text()).get("by_episode", {})
        missing = [e for e in avail if e not in by_ep]
        if missing:
            print(f"[grow] WARN {len(missing)} available episodes missing from score json "
                  f"(scored a different pool?) — they rank last.")
        avail.sort(key=lambda e: by_ep.get(e, float("-inf")), reverse=True)
        if args.diverse:
            m = args.oversample if args.oversample > 0 else 3 * args.k
            cand = avail[:m]
            embs, kept = [], []
            for e in cand:
                emb = _episode_embedding(pool_root, pool[e])
                if emb is not None:
                    embs.append(emb)
                    kept.append(e)
            if len(kept) <= args.k:
                picked = kept or avail[: args.k]
                print(f"[grow] diverse: only {len(kept)} embeddable candidates <= k; keeping all.")
            else:
                idx = _greedy_kcenter(embs, args.k)
                picked = [kept[i] for i in idx]
                print(f"[grow] strategic+diverse: k-center {args.k} of top-{m} by score")
        else:
            picked = avail[: args.k]
        ranked_preview = [(e, round(by_ep.get(e, float("nan")), 4)) for e in picked]
        print(f"[grow] strategic picked (score): {ranked_preview}")
    else:
        rng = random.Random(args.seed)
        rng.shuffle(avail)
        picked = avail[: args.k]
        print(f"[grow] random {args.k} (seed={args.seed}): {picked}")

    # ---- symlink every chunk of each picked episode into the growing root ----
    n_links = 0
    for ep in picked:
        for cid in pool[ep]:
            for mod in MODALITIES:
                src = pool_root / mod / "videos" / f"{cid}.pt"
                if not src.exists():
                    continue
                dst = grow_root / mod / "videos" / f"{cid}.pt"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    dst.symlink_to(src.resolve())
                    n_links += 1

    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a") as f:
        for ep in picked:
            f.write(ep + "\n")

    if args.picked_out:
        Path(args.picked_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.picked_out).write_text("\n".join(picked) + ("\n" if picked else ""))

    remaining = len(avail) - len(picked)
    print(f"[grow] added {len(picked)} episodes ({n_links} symlinks) -> {grow_root}")
    print(f"[grow] manifest now {len(added) + len(picked)} episodes; pool remaining ~{remaining}")


if __name__ == "__main__":
    main()
