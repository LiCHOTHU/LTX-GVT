#!/usr/bin/env python3
# ruff: noqa: T201
"""Per AL round, render the SELECTED (largest-signal) vs REJECTED (smallest-signal)
chunk as GT|pred panels, using THAT round's checkpoint.

Goal: *see* the selection decision at each training phase of the strategic arm. For
round N we load the checkpoint the loop actually scored with (step 30000 + N*1000),
read round N's candidate scores from latent_loss_rN.json, and pick:

  * SELECTED  = highest-`score` chunk  -> its episode was added to training this round.
  * REJECTED  = lowest-`score`  chunk  -> its episode was NOT added.

`score` is the EXACT v2 selection signal the loop used that round: raw teacher-forced
latent loss for r<=3 (baseline not yet available), magnitude-normalized learning-progress
(progress_norm) from r4 on. For each pick we run the full rollout, stack pred|gt, and
also report raw latent loss + pixel PSNR so the visible gap is on the panel.

Reuses al_pixel_score / infer_action_cond / score_chunk_latent_loss verbatim (one 22B
load; PEFT hot-swap of the adapter + action projector per round). Inference only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from peft import set_peft_model_state_dict

from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.model.transformer.enable_prope import enable_prope_on_model
from ltx_trainer.model_loader import load_embeddings_processor, load_model

from al_pixel_score import _to_frames
from eval_prope_metrics import score_all
from infer_action_cond import (
    DEFAULT_BASE,
    _load_chunk,
    _render_chunk,
    _stack_and_save,
    load_action_projector,
    load_lora_weights,
)
from score_chunk_latent_loss import (
    _build_batch,
    _episode_of,
    _kind_of,
    _lora_state,
    build_strategy,
    score_chunk,
)


def _parse_rounds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _rank_chunks(score_json: Path, present: set[str]) -> list[tuple[str, dict]]:
    """[(chunk_id, rec)] sorted by the v2 selection `score` desc, restricted to chunks
    that actually exist in the round's pool on disk."""
    per_chunk = json.loads(score_json.read_text())["per_chunk"]
    items = [(cid, rec) for cid, rec in per_chunk.items() if cid in present]
    items.sort(key=lambda kv: kv[1].get("score", kv[1]["loss"]), reverse=True)
    return items


def main() -> None:
    p = argparse.ArgumentParser(description="Render selected vs rejected GT|pred pair per AL round")
    p.add_argument("--base-checkpoint", default=DEFAULT_BASE)
    p.add_argument("--ckpt-dir", required=True, help="arm's checkpoints/ dir (per-round LoRA + projector)")
    p.add_argument("--state-dir", required=True, help="alr_<arm>/ dir holding latent_loss_rN.json")
    p.add_argument("--pool-template", required=True,
                   help="round pool root with {r} placeholder, e.g. .../libero90_ALR_strategic_round{r}/precomputed")
    p.add_argument("--rounds", default="0-8", help="e.g. '0-8' or '0,3,6'")
    p.add_argument("--start-step", type=int, default=30000)
    p.add_argument("--step-per-round", type=int, default=1000)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-inference-steps", type=int, default=30)
    p.add_argument("--drop-first", type=int, default=8)
    p.add_argument("--sigma-grid", default="0.2,0.4,0.6,0.8")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--pairs-per-round", type=int, default=4,
                   help="how many selected/rejected pairs to render per round (top-k / bottom-k)")
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
    ckpt_dir = Path(args.ckpt_dir)
    state_dir = Path(args.state_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sigma_grid = [float(s) for s in args.sigma_grid.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rounds = _parse_rounds(args.rounds)

    def lora_for(step: int) -> Path:
        return ckpt_dir / f"lora_weights_step_{step}.safetensors"

    def proj_for(step: int) -> Path:
        return ckpt_dir / f"action_projector_step_{step}.safetensors"

    # ---- Load base ONCE; wrap with round-0 LoRA via PEFT, then hot-swap per round. ----
    components = load_model(
        checkpoint_path=args.base_checkpoint, device="cpu", dtype=dtype,
        with_video_vae_encoder=False, with_video_vae_decoder=True,
        with_audio_vae_decoder=False, with_vocoder=False, with_text_encoder=False,
    )
    if args.use_prope:
        n = enable_prope_on_model(components.transformer, proj_dim=args.prope_proj_dim,
                                  image_width=args.prope_image_width, image_height=args.prope_image_height)
        print(f"PRoPE enabled: swapped {n} attn1 modules")
    first_step = args.start_step + rounds[0] * args.step_per_round
    transformer = load_lora_weights(components.transformer, lora_for(first_step))
    print("[setup] lora loaded; moving transformer to device...", flush=True)
    transformer = transformer.to(device=device, dtype=dtype).eval()
    print("[setup] transformer on device; loading embeddings processor...", flush=True)
    embeddings_processor = load_embeddings_processor(args.base_checkpoint, device=device, dtype=dtype)
    print("[setup] embeddings processor loaded; loading action projector...", flush=True)
    action_projector = load_action_projector(proj_for(first_step), dtype).to(device)
    print("[setup] action projector loaded; building strategy...", flush=True)
    strategy = build_strategy(action_projector, use_prope=args.use_prope,
                              prope_image_width=args.prope_image_width,
                              prope_image_height=args.prope_image_height,
                              first_frame_conditioning_p=args.first_frame_conditioning_p)
    patchifier = VideoLatentPatchifier(patch_size=1)
    print("[setup] strategy built; entering round loop.", flush=True)

    render_args = SimpleNamespace(
        num_inference_steps=args.num_inference_steps, guidance_scale=1.0,
        negative_prompt="", seed=args.seed, no_first_frame=False,
        include_gt=True, include_context=False, device=args.device,
        use_prope=args.use_prope, prope_proj_dim=args.prope_proj_dim,
        prope_image_width=args.prope_image_width, prope_image_height=args.prope_image_height,
    )

    summary: list[dict] = []
    for rnd in rounds:
        step = args.start_step + rnd * args.step_per_round
        lora_p, proj_p = lora_for(step), proj_for(step)
        score_json = state_dir / f"latent_loss_r{rnd}.json"
        pool_root = Path(args.pool_template.format(r=rnd))
        if not (lora_p.exists() and score_json.exists() and pool_root.exists()):
            print(f"[round {rnd}] SKIP (missing lora/json/pool): "
                  f"lora={lora_p.exists()} json={score_json.exists()} pool={pool_root.exists()}")
            continue

        # hot-swap to THIS round's checkpoint (adapter + projector for both render & latent score)
        print(f"[round {rnd}] swapping adapter+projector to step {step}...", flush=True)
        set_peft_model_state_dict(transformer.get_base_model(), _lora_state(lora_p))
        action_projector = load_action_projector(proj_p, dtype).to(device)
        strategy._action_projector = action_projector

        present = {f.stem for f in (pool_root / "latents" / "videos").glob("*.pt")}
        ranked = _rank_chunks(score_json, present)
        if len(ranked) < 2:
            print(f"[round {rnd}] SKIP (only {len(ranked)} scored chunks present)")
            continue

        k = max(1, min(args.pairs_per_round, len(ranked) // 2))
        picks = {}
        for i in range(k):
            picks[f"selected{i}"] = ranked[i]
            picks[f"rejected{i}"] = ranked[-(i + 1)]
        rdir = out_dir / f"round{rnd}_step{step}"
        rdir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== round {rnd} (step {step}) | {len(ranked)} candidate chunks =====")

        round_rec: dict = {"round": rnd, "step": step}
        for tag, (cid, rec) in picks.items():
            panels = _render_chunk(
                components=components, transformer=transformer,
                embeddings_processor=embeddings_processor, action_projector=action_projector,
                patchifier=patchifier, data_root=pool_root, chunk_id=cid,
                args=render_args, use_cfg=False,
            )
            short = cid[:60]
            mp4 = rdir / f"{tag}__score{rec.get('score', rec['loss']):.4f}__{short}.mp4"
            _stack_and_save(panels, mp4, render_args)
            px = score_all(_to_frames(panels["pred"]), _to_frames(panels["gt"]), args.drop_first)

            # cheap latent loss with this round's model (matches the scorer's number)
            chunk = _load_chunk(pool_root, cid)
            batch = _build_batch(chunk, embeddings_processor, use_prope=args.use_prope,
                                 data_root=pool_root, chunk_id=cid, device=device, dtype=dtype)
            lat = score_chunk(strategy, transformer, batch, sigma_grid=sigma_grid,
                              seeds=seeds, device=device, dtype=dtype)

            info = {
                "tag": tag, "chunk_id": cid, "episode": _episode_of(cid), "kind": _kind_of(cid),
                "select_score": rec.get("score", rec["loss"]),
                "raw_latent_loss": rec.get("loss"),
                "progress_norm": rec.get("progress_norm"),
                "rescored_latent_loss": lat["loss"],
                "psnr_all": px["all"]["psnr"], "ssim_all": px["all"]["ssim"],
                "psnr_agentview": px["agentview"]["psnr"], "psnr_wrist": px["wrist"]["psnr"],
                "mp4": str(mp4),
            }
            round_rec[tag] = info
            print(f"  [{tag:8s}] score={info['select_score']:.4f} rawloss={info['raw_latent_loss']:.4f} "
                  f"PSNR={info['psnr_all']:.2f} {info['kind']:7s} {short}")
        summary.append(round_rec)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    # human-readable
    def _ord(t: str):
        return (0 if t.startswith("selected") else 1,
                int("".join(c for c in t if c.isdigit()) or 0))
    lines = ["round  step    tag        select_score  raw_loss  PSNR    kind     episode"]
    for r in summary:
        for tag in sorted((t for t in r if t not in ("round", "step")), key=_ord):
            i = r.get(tag)
            if not i:
                continue
            lines.append(f"{r['round']:<6d} {r['step']:<7d} {tag:<10s} "
                         f"{i['select_score']:>12.4f}  {i['raw_latent_loss']:>8.4f}  "
                         f"{i['psnr_all']:>6.2f}  {i['kind']:<7s}  {i['episode'][:55]}")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nWrote {out_dir}/summary.json + summary.txt and per-round mp4 pairs.")


if __name__ == "__main__":
    main()
