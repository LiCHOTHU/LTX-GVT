#!/usr/bin/env python3
# ruff: noqa: T201
"""Proof reference pass: per chunk compute BOTH signals in ONE model load.

For the active-learning correlation proof we need, per chunk:
  - the CHEAP latent denoising loss (the candidate AL signal), and
  - the EXPENSIVE pixel error from a full 30-step rollout + decode (the ground truth
    "is the model visibly wrong"), via the exact infer path.

Doing both in one model load avoids a second ~25-min cold load and guarantees the two
numbers are a matched pair per chunk. Reuses ``_render_chunk`` (infer_action_cond),
``score_all`` (eval_prope_metrics), and ``score_chunk``/``build_strategy`` (the cheap
scorer) so there is no logic drift. Per-chunk pred|gt mp4s are saved with the FULL
chunk id (infer's multi-chunk mode truncates to cid[:55], which collides across chunks
of one episode) so al_corr_proof can pull worst/best examples.

Output: ``combined.json`` (+ ``.csv``) with per-chunk {latent_loss, psnr/ssim per view}.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.model.transformer.enable_prope import enable_prope_on_model
from ltx_trainer.model_loader import load_embeddings_processor, load_model

from eval_prope_metrics import score_all
from infer_action_cond import (
    DEFAULT_BASE,
    _default_action_projector_path,
    _list_chunk_ids,
    _load_chunk,
    _render_chunk,
    _stack_and_save,
    load_action_projector,
    load_lora_weights,
)
from score_chunk_latent_loss import _build_batch, _episode_of, _kind_of, build_strategy, score_chunk


def _to_frames(v: torch.Tensor) -> "torch.Tensor":
    """[C,F,H,W] float in [0,1] -> (F,H,W,C) uint8 numpy (what score_all expects)."""
    x = (v.clamp(0, 1) * 255.0).round().to(torch.uint8)
    return x.permute(1, 2, 3, 0).cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser(description="Per-chunk latent-loss + pixel-error (one model load)")
    p.add_argument("--base-checkpoint", default=DEFAULT_BASE)
    p.add_argument("--lora-path", required=True)
    p.add_argument("--action-projector", default=None)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", required=True, help="Holds combined.json/.csv and chunk_mp4s/")
    p.add_argument("--indices", default=None)
    p.add_argument("--max-chunks", type=int, default=0)
    p.add_argument("--num-inference-steps", type=int, default=30)
    p.add_argument("--drop-first", type=int, default=8, help="frames dropped for the pred-only PSNR/SSIM")
    p.add_argument("--sigma-grid", default="0.2,0.4,0.6,0.8")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--first-frame-conditioning-p", type=float, default=1.0)
    p.add_argument("--use-prope", action="store_true")
    p.add_argument("--prope-proj-dim", type=int, default=64)
    p.add_argument("--prope-image-width", type=int, default=256)
    p.add_argument("--prope-image-height", type=int, default=256)
    p.add_argument("--seed", type=int, default=42, help="rollout noise seed")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    mp4_dir = out_dir / "chunk_mp4s"
    mp4_dir.mkdir(parents=True, exist_ok=True)
    sigma_grid = [float(s) for s in args.sigma_grid.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    all_ids = _list_chunk_ids(data_root)
    if not all_ids:
        raise SystemExit(f"No chunks under {data_root}/latents/videos/")
    if args.indices is not None:
        all_ids = [all_ids[int(i)] for i in args.indices.split(",") if i.strip() != ""]
    if args.max_chunks > 0:
        all_ids = all_ids[: args.max_chunks]
    print(f"Scoring {len(all_ids)} chunks (latent + pixel) | steps={args.num_inference_steps} prope={args.use_prope}")

    # ---- Load model ONCE (transformer + LoRA + PRoPE + VAE decoder; no text encoder: CFG off) ----
    components = load_model(
        checkpoint_path=args.base_checkpoint, device="cpu", dtype=dtype,
        with_video_vae_encoder=False, with_video_vae_decoder=True,
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

    strategy = build_strategy(action_projector, use_prope=args.use_prope,
                              prope_image_width=args.prope_image_width, prope_image_height=args.prope_image_height,
                              first_frame_conditioning_p=args.first_frame_conditioning_p)
    patchifier = VideoLatentPatchifier(patch_size=1)

    # args namespace for _render_chunk (rollout): condition first frame, GT panel on, no CFG, no context.
    render_args = SimpleNamespace(
        num_inference_steps=args.num_inference_steps, guidance_scale=1.0,
        negative_prompt="", seed=args.seed, no_first_frame=False,
        include_gt=True, include_context=False, device=args.device,
        use_prope=args.use_prope, prope_proj_dim=args.prope_proj_dim,
        prope_image_width=args.prope_image_width, prope_image_height=args.prope_image_height,
    )

    results: dict[str, dict] = {}
    t0 = time.perf_counter()
    for i, cid in enumerate(all_ids):
        # ---- expensive: full rollout -> pred/gt -> per-view pixel metrics + save mp4 ----
        panels = _render_chunk(
            components=components, transformer=transformer, embeddings_processor=embeddings_processor,
            action_projector=action_projector, patchifier=patchifier,
            data_root=data_root, chunk_id=cid, args=render_args, use_cfg=False,
        )
        _stack_and_save(panels, mp4_dir / f"{cid}.mp4", render_args)
        px = score_all(_to_frames(panels["pred"]), _to_frames(panels["gt"]), args.drop_first)

        # ---- cheap: latent denoising loss (the AL signal) ----
        chunk = _load_chunk(data_root, cid)
        batch = _build_batch(chunk, embeddings_processor, use_prope=args.use_prope,
                             data_root=data_root, chunk_id=cid, device=device, dtype=dtype)
        lat = score_chunk(strategy, transformer, batch, sigma_grid=sigma_grid, seeds=seeds, device=device, dtype=dtype)

        results[cid] = {
            "latent_loss": lat["loss"], "per_sigma": lat["per_sigma"],
            "psnr_all": px["all"]["psnr"], "ssim_all": px["all"]["ssim"],
            "psnr_agentview": px["agentview"]["psnr"], "ssim_agentview": px["agentview"]["ssim"],
            "psnr_wrist": px["wrist"]["psnr"], "ssim_wrist": px["wrist"]["ssim"],
            "kind": _kind_of(cid), "episode": _episode_of(cid),
            "mp4": str(mp4_dir / f"{cid}.mp4"),
        }
        rate = (i + 1) / (time.perf_counter() - t0)
        print(f"  [{i+1}/{len(all_ids)}] {cid[:55]} loss={lat['loss']:.4f} "
              f"PSNR={px['all']['psnr']:.2f} SSIM={px['all']['ssim']:.4f} ({rate:.2f} ch/s)")

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {"data_root": str(data_root), "lora": str(args.lora_path), "n_chunks": len(results),
                 "num_inference_steps": args.num_inference_steps, "drop_first": args.drop_first,
                 "sigma_grid": sigma_grid, "seeds": seeds, "use_prope": args.use_prope},
        "per_chunk": results,
    }
    (out_dir / "combined.json").write_text(json.dumps(payload, indent=2))
    with (out_dir / "combined.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id", "kind", "episode", "latent_loss", "psnr_all", "ssim_all",
                    "psnr_agentview", "psnr_wrist"])
        for cid, r in sorted(results.items(), key=lambda kv: -kv[1]["latent_loss"]):
            w.writerow([cid, r["kind"], r["episode"], f"{r['latent_loss']:.6f}",
                        f"{r['psnr_all']:.4f}", f"{r['ssim_all']:.4f}",
                        f"{r['psnr_agentview']:.4f}", f"{r['psnr_wrist']:.4f}"])
    print(f"\nWrote {out_dir/'combined.json'} and combined.csv  ({len(results)} chunks)")


if __name__ == "__main__":
    main()
