#!/usr/bin/env python3
# ruff: noqa: T201
"""Cheap per-chunk "how wrong is the model" scorer for active learning.

This is the active-learning signal: instead of a full 30-step rollout + VAE decode
(~8-10 s/chunk) it runs ONE forward pass at a fixed noise level and computes the
*exact* teacher-forced denoising loss the model trains on (velocity-MSE on the
target tokens). ~400x cheaper, and train-consistent because it reuses the real
``VideoToVideoStrategy.prepare_training_inputs`` + ``compute_loss`` rather than a
re-derivation. Model loading, chunk access, PRoPE plumbing and the action projector
are all imported from ``infer_action_cond`` so there is zero logic drift.

For a STABLE, comparable ranking across chunks (a single random sigma+noise is too
noisy) the loss is averaged over a small fixed sigma grid x a few fixed seeds, and
the first target frame is always conditioned (deterministic mask).

Output: ``latent_loss.json`` (+ ``.csv``) with per-chunk loss, per-sigma breakdown,
per-kind and per-episode aggregates -> a ranking from which the AL loop selects.

Example:
    python experiments/libero_sim/score_chunk_latent_loss.py \
        --lora-path .../lora_weights_step_30000.safetensors \
        --action-projector .../action_projector_step_30000.safetensors \
        --data-root .../libero90_ood_precomputed/precomputed \
        --use-prope --prope-proj-dim 64 --prope-image-width 256 --prope-image-height 256 \
        --out experiments/libero_sim/outputs/al_corr_proof_step30000/latent_loss.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

from ltx_core.model.transformer.enable_prope import enable_prope_on_model
from ltx_core.text_encoders.gemma import convert_to_additive_mask
from ltx_trainer.model_loader import load_embeddings_processor, load_model
from ltx_trainer.training_strategies.base_strategy import DEFAULT_FPS
from ltx_trainer.training_strategies.video_to_video import VideoToVideoConfig, VideoToVideoStrategy

# Reuse the proven loaders / chunk access from the inference script (same dir on PYTHONPATH).
from infer_action_cond import (  # noqa: E402
    _default_action_projector_path,
    _list_chunk_ids,
    _load_chunk,
    load_action_projector,
    load_lora_weights,
)

# Prefer the fast scratch1-staged copy of the 22B transformer (cold-loads in a fraction
# of the ~25 min it takes from cedar0); fall back to cedar0 if scratch1 was purged
# (scratch1 auto-purges after ~60 days). GRADE loads only the transformer (no text encoder).
_STAGED_BASE = "/storage/scratch1/8/lwang831/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
_CEDAR_BASE = "/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
DEFAULT_BASE = _STAGED_BASE if Path(_STAGED_BASE).exists() else _CEDAR_BASE


class _FixedSigmaSampler:
    """Duck-typed TimestepSampler that returns a constant sigma (so we control the
    noise level instead of sampling it). prepare_training_inputs only calls .sample_for."""

    def __init__(self, sigma: float):
        self.sigma = sigma

    def sample_for(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0],), self.sigma, device=x.device, dtype=x.dtype)


def _kind_of(chunk_id: str) -> str:
    """OOD policy kind from the stem (matches gen_libero_play naming)."""
    for k in ("random", "perturb", "reach", "circle", "gt"):
        if f"_play_{k}_" in chunk_id:
            return k
    return "gt"  # GT replay stems carry no _play_<k>_ tag


def _episode_of(chunk_id: str) -> str:
    """Episode stem (drop the trailing _chunk_NN_target)."""
    m = re.match(r"^(.*)_chunk_\d+_target$", chunk_id)
    return m.group(1) if m else chunk_id


def _build_batch(chunk: dict, embeddings_processor, *, use_prope: bool,
                 data_root: Path, chunk_id: str, device, dtype) -> dict:
    """Assemble one-sample batch in the exact shape PrecomputedDataset yields, with
    text embeddings already pushed through the connectors (as the trainer does before
    calling the strategy)."""
    tgt, ref, cond = chunk["target"], chunk["reference"], chunk["conditions"]

    def _meta(d, key):
        return torch.tensor([int(d[key])])

    batch = {
        "latents": {
            "latents": tgt["latents"].unsqueeze(0).to(device=device, dtype=dtype),
            "num_frames": _meta(tgt, "num_frames"),
            "height": _meta(tgt, "height"),
            "width": _meta(tgt, "width"),
            "fps": torch.tensor([float(tgt.get("fps", DEFAULT_FPS))]),
        },
        "ref_latents": {
            "latents": ref["latents"].unsqueeze(0).to(device=device, dtype=dtype),
            "num_frames": _meta(ref, "num_frames"),
            "height": _meta(ref, "height"),
            "width": _meta(ref, "width"),
        },
        "actions": {"latents": chunk["actions"]["latents"].unsqueeze(0).to(device=device, dtype=dtype)},
    }

    # Text: pre-connector Gemma features -> connector (block 3), exactly as trainer._training_step.
    vfeat = cond["video_prompt_embeds"].unsqueeze(0).to(device=device, dtype=dtype)
    afeat = cond.get("audio_prompt_embeds")
    afeat = afeat.unsqueeze(0).to(device=device, dtype=dtype) if afeat is not None else None
    mask = cond["prompt_attention_mask"].unsqueeze(0).to(device=device)
    add_mask = convert_to_additive_mask(mask, vfeat.dtype)
    video_ctx, _audio_ctx, _ctx_mask = embeddings_processor.create_embeddings(vfeat, afeat, add_mask)
    # Training passes the ORIGINAL binary attention mask as context_mask (the connector
    # preserves the 1024-token length), so we mirror that for a faithful loss.
    batch["conditions"] = {"video_prompt_embeds": video_ctx, "prompt_attention_mask": mask}

    if use_prope:
        cam = torch.load(data_root / "cameras" / "videos" / f"{chunk_id}.pt",
                         map_location="cpu", weights_only=False)
        batch["cameras"] = {
            "viewmats": cam["viewmats"].unsqueeze(0).to(device=device),
            "Ks": cam["Ks"].unsqueeze(0).to(device=device),
        }
    return batch


def build_strategy(action_projector, *, use_prope: bool, prope_image_width: int,
                   prope_image_height: int, first_frame_conditioning_p: float = 1.0):
    """Construct the real VideoToVideoStrategy and attach what the trainer attaches
    post-construction (acceleration config + action projector). Reused by the pixel pass."""
    strategy = VideoToVideoStrategy(VideoToVideoConfig(
        first_frame_conditioning_p=first_frame_conditioning_p,
        reference_latents_dir="reference_latents",
    ))
    strategy._acceleration_config = SimpleNamespace(
        use_action_cond=True, use_prope=use_prope,
        prope_image_width=prope_image_width, prope_image_height=prope_image_height,
        use_warp_latent=False,
    )
    strategy._action_projector = action_projector
    return strategy


@torch.inference_mode()
def score_chunk(strategy, transformer, batch, *, sigma_grid, seeds, device, dtype) -> dict:
    """Average teacher-forced velocity-MSE over a fixed sigma grid x seeds -> stable score.

    Also returns the mean target-velocity energy (mean of target^2) so the caller can
    form a *magnitude-normalized* loss (loss / energy). The velocity target is
    ``noise - target_latents`` (independent of sigma), so high-motion / high-energy
    chunks carry larger targets and therefore larger raw MSE regardless of how well the
    model predicts them -- normalizing removes that "busy != wrong" confound (gap #3)."""
    per_sigma: dict[str, float] = {}
    all_vals: list[float] = []
    energies: list[float] = []
    for sigma in sigma_grid:
        sampler = _FixedSigmaSampler(sigma)
        vals = []
        for seed in seeds:
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            mi = strategy.prepare_training_inputs(batch, sampler)
            with torch.autocast(device_type=device.type, dtype=dtype):
                pred, _ = transformer(video=mi.video, audio=mi.audio, perturbations=None)
            loss = strategy.compute_loss(pred, None, mi)  # [B]
            vals.append(float(loss.mean().float().item()))
            energies.append(float(mi.video_targets.float().pow(2).mean().item()))
        per_sigma[f"{sigma:.2f}"] = sum(vals) / len(vals)
        all_vals.extend(vals)
    return {
        "loss": sum(all_vals) / len(all_vals),
        "per_sigma": per_sigma,
        "energy": sum(energies) / len(energies),
    }


def _lora_state(path: str | Path) -> dict:
    """Load a LoRA safetensors checkpoint into the in-memory PEFT key layout (mirrors
    infer_action_cond.load_lora_weights' prefix handling)."""
    sd = load_file(str(path))
    return {k.replace("diffusion_model.", "", 1): v for k, v in sd.items()}


def swap_member(transformer, strategy, lora_path: str | Path, proj_path: str | Path,
                *, dtype, device) -> None:
    """Hot-swap the PEFT adapter weights + action projector to a different checkpoint
    WITHOUT reloading the 22B base. Used for the learning-progress baseline (gap #2) and
    the ensemble-disagreement members (gap #4): same rank/target-modules, so we just
    overwrite the adapter state in place."""
    set_peft_model_state_dict(transformer.get_base_model(), _lora_state(lora_path))
    strategy._action_projector = load_action_projector(proj_path, dtype).to(device)


def _pick_metric(rec: dict, metric: str) -> float:
    """Selection score for a chunk. Falls back to raw loss if the requested metric is
    unavailable (e.g. 'progress' with no baseline checkpoint yet, early rounds)."""
    val = rec.get(metric)
    return rec["loss"] if val is None else val


def main() -> None:
    p = argparse.ArgumentParser(description="Cheap per-chunk denoising-loss scorer (active-learning signal)")
    p.add_argument("--base-checkpoint", default=DEFAULT_BASE)
    p.add_argument("--lora-path", required=True)
    p.add_argument("--action-projector", default=None, help="Defaults to action_projector_step_<same>.safetensors")
    p.add_argument("--data-root", required=True, help="precomputed/ root (holds latents/ reference_latents/ ...)")
    p.add_argument("--out", required=True, help="Output latent_loss.json (.csv written alongside)")
    p.add_argument("--indices", default=None, help="Comma-separated chunk indices (default: all)")
    p.add_argument("--max-chunks", type=int, default=0, help="0 = all")
    p.add_argument("--sigma-grid", default="0.2,0.4,0.6,0.8", help="Fixed noise levels to average over")
    p.add_argument("--seeds", default="0,1,2", help="Noise seeds to average over (per sigma)")
    p.add_argument("--first-frame-conditioning-p", type=float, default=1.0,
                   help="Deterministic when 0 or 1; 1.0 = always condition the first target frame (inference setup)")
    p.add_argument("--use-prope", action="store_true")
    p.add_argument("--prope-proj-dim", type=int, default=64)
    p.add_argument("--prope-image-width", type=int, default=256)
    p.add_argument("--prope-image-height", type=int, default=256)
    # ---- AL signal refinements (all optional; default == legacy raw-loss behavior) ----
    p.add_argument("--score-metric", default="loss",
                   choices=["loss", "loss_norm", "progress", "progress_norm", "disagreement"],
                   help="Which per-chunk metric drives selection (by_episode aggregates this). "
                        "loss=legacy raw velocity-MSE; loss_norm=magnitude-normalized (#3); "
                        "progress[_norm]=loss drop vs --baseline-lora (#2, RHO-LOSS/learning-progress); "
                        "disagreement=variance across --ensemble-loras (#4, Plan2Explore).")
    p.add_argument("--baseline-lora", default=None,
                   help="Older checkpoint for learning-progress: progress = base_loss - current_loss (#2).")
    p.add_argument("--baseline-projector", default=None, help="Defaults to projector next to --baseline-lora.")
    p.add_argument("--ensemble-loras", default=None,
                   help="Comma-separated extra checkpoints; disagreement = variance of loss across "
                        "{current} + these (#4). Each needs an action_projector beside it.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16
    data_root = Path(args.data_root)
    sigma_grid = [float(s) for s in args.sigma_grid.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    all_ids = _list_chunk_ids(data_root)
    if not all_ids:
        raise SystemExit(f"No chunks under {data_root}/latents/videos/")
    if args.indices is not None:
        idxs = [int(x) for x in args.indices.split(",") if x.strip() != ""]
        all_ids = [all_ids[i] for i in idxs]
    if args.max_chunks > 0:
        all_ids = all_ids[: args.max_chunks]
    print(f"Scoring {len(all_ids)} chunks | sigmas={sigma_grid} seeds={seeds} prope={args.use_prope}")

    # ---- Load model ONCE (no VAE, no Gemma LLM) — mirror infer_action_cond ordering ----
    components = load_model(
        checkpoint_path=args.base_checkpoint, device="cpu", dtype=dtype,
        with_video_vae_encoder=False, with_video_vae_decoder=False,
        with_audio_vae_decoder=False, with_vocoder=False, with_text_encoder=False,
    )
    if args.use_prope:
        n = enable_prope_on_model(components.transformer, proj_dim=args.prope_proj_dim,
                                  image_width=args.prope_image_width, image_height=args.prope_image_height)
        print(f"PRoPE enabled: swapped {n} attn1 modules")
    transformer = load_lora_weights(components.transformer, args.lora_path)
    transformer = transformer.to(device=device, dtype=dtype).eval()
    embeddings_processor = load_embeddings_processor(args.base_checkpoint, device=device, dtype=dtype)

    proj_path = Path(args.action_projector) if args.action_projector else _default_action_projector_path(Path(args.lora_path))
    if proj_path is None:
        raise SystemExit("No action projector found next to the LoRA; pass --action-projector.")
    action_projector = load_action_projector(proj_path, dtype).to(device)

    # ---- Build the strategy and attach what the trainer attaches post-construction ----
    strategy = build_strategy(
        action_projector, use_prope=args.use_prope,
        prope_image_width=args.prope_image_width, prope_image_height=args.prope_image_height,
        first_frame_conditioning_p=args.first_frame_conditioning_p,
    )

    # ---- Build member list: current + optional progress-baseline + optional ensemble ----
    members: list[tuple[str, str, str]] = [("current", str(args.lora_path), str(proj_path))]
    if args.baseline_lora:
        bproj = args.baseline_projector or _default_action_projector_path(Path(args.baseline_lora))
        if bproj is None:
            print(f"[warn] --baseline-lora {args.baseline_lora} has no projector; skipping progress.")
        else:
            members.append(("baseline", str(args.baseline_lora), str(bproj)))
    for j, ep in enumerate(p for p in (args.ensemble_loras or "").split(",") if p.strip()):
        eproj = _default_action_projector_path(Path(ep))
        if eproj is None:
            print(f"[warn] ensemble lora {ep} has no projector; skipping it.")
            continue
        members.append((f"ens{j}", ep, str(eproj)))
    print(f"Members: {[m[0] for m in members]}  score_metric={args.score_metric}")

    # ---- Score every chunk under each member (member 0 already loaded; rest hot-swapped) ----
    member_scores: dict[str, dict[str, dict]] = {name: {} for name, _, _ in members}
    t0 = time.perf_counter()
    for mi_idx, (name, lpath, ppath) in enumerate(members):
        if mi_idx > 0:
            swap_member(transformer, strategy, lpath, ppath, dtype=dtype, device=device)
            print(f"[member {name}] swapped -> {Path(lpath).name}")
        for i, cid in enumerate(all_ids):
            chunk = _load_chunk(data_root, cid)
            batch = _build_batch(chunk, embeddings_processor, use_prope=args.use_prope,
                                 data_root=data_root, chunk_id=cid, device=device, dtype=dtype)
            member_scores[name][cid] = score_chunk(strategy, transformer, batch,
                                                   sigma_grid=sigma_grid, seeds=seeds, device=device, dtype=dtype)
            if (i + 1) % 10 == 0 or i + 1 == len(all_ids):
                rate = (i + 1) / (time.perf_counter() - t0)
                print(f"  [{name}] [{i+1}/{len(all_ids)}] {cid[:50]} "
                      f"loss={member_scores[name][cid]['loss']:.4f} ({rate:.1f} chunks/s)")

    # ---- Derive per-chunk metrics; 'score' is what selection ranks by ----
    ens_names = ["current"] + [n for n in member_scores if n.startswith("ens")]
    results: dict[str, dict] = {}
    for cid in all_ids:
        sc = member_scores["current"][cid]
        energy = max(sc["energy"], 1e-8)
        loss = sc["loss"]
        rec: dict = {
            "loss": loss, "per_sigma": sc["per_sigma"], "energy": energy,
            "loss_norm": loss / energy, "kind": _kind_of(cid), "episode": _episode_of(cid),
        }
        if "baseline" in member_scores:
            bl = member_scores["baseline"][cid]["loss"]
            rec["base_loss"] = bl
            rec["progress"] = bl - loss                 # positive = model is improving here (learnable)
            rec["progress_norm"] = (bl - loss) / energy
        if len(ens_names) >= 2:
            losses = [member_scores[n][cid]["loss"] for n in ens_names]
            mu = sum(losses) / len(losses)
            rec["disagreement"] = sum((x - mu) ** 2 for x in losses) / len(losses)
        rec["score"] = _pick_metric(rec, args.score_metric)
        results[cid] = rec

    # ---- Aggregates (by_episode aggregates the chosen 'score' -> the ranking the selector reads) ----
    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    by_kind: dict[str, list[float]] = {}
    by_ep: dict[str, list[float]] = {}
    by_ep_loss: dict[str, list[float]] = {}
    for cid, sc in results.items():
        by_kind.setdefault(sc["kind"], []).append(sc["loss"])
        by_ep.setdefault(sc["episode"], []).append(sc["score"])
        by_ep_loss.setdefault(sc["episode"], []).append(sc["loss"])
    kind_agg = {k: {"mean": _mean(v), "n": len(v)} for k, v in sorted(by_kind.items())}
    ep_agg = {e: _mean(v) for e, v in sorted(by_ep.items())}
    ep_loss_agg = {e: _mean(v) for e, v in sorted(by_ep_loss.items())}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {"data_root": str(data_root), "lora": str(args.lora_path), "n_chunks": len(results),
                 "sigma_grid": sigma_grid, "seeds": seeds, "score_metric": args.score_metric,
                 "members": [m[0] for m in members],
                 "first_frame_conditioning_p": args.first_frame_conditioning_p, "use_prope": args.use_prope},
        "per_chunk": results, "by_kind": kind_agg,
        "by_episode": ep_agg,            # the chosen metric — selector ranks by this
        "by_episode_loss": ep_loss_agg,  # raw-loss diagnostic (always present)
    }
    out.write_text(json.dumps(payload, indent=2))
    with out.with_suffix(".csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id", "kind", "episode", "score", "loss", "loss_norm", "progress", "disagreement"])
        for cid, sc in sorted(results.items(), key=lambda kv: -kv[1]["score"]):
            w.writerow([cid, sc["kind"], sc["episode"], f"{sc['score']:.6f}", f"{sc['loss']:.6f}",
                        f"{sc['loss_norm']:.6f}", f"{sc.get('progress', float('nan')):.6f}",
                        f"{sc.get('disagreement', float('nan')):.6f}"])

    print(f"\nWrote {out} and {out.with_suffix('.csv')} (metric={args.score_metric})")
    print("Per-kind mean latent-loss (sanity: random/circle should exceed perturb/gt):")
    for k, a in kind_agg.items():
        print(f"  {k:8s} n={a['n']:3d}  mean={a['mean']:.4f}")


if __name__ == "__main__":
    main()
