#!/usr/bin/env python3
# ruff: noqa: T201
"""Head-to-head held-out eval: N AL arms at a MATCHED step.

All arms of the single-scene object-centric AL run trained on the SAME 30-episode
tomato_sauce scene, so the only fair question is: at a matched checkpoint, does a
strategic arm predict the OOD (perturb + random action) rollouts of that scene more
faithfully than the random arm? Arms are passed as repeatable `--arm NAME=DIR`;
deltas are reported against `--baseline` (default: random).

MATCHED STEP MATTERS: arms that trained further would win on step count alone, so
always eval at a step every arm reached (the 4-arm run: 46000).

Eval set = one round's candidate POOL that was never symlinked into training. The
grow-root froze on round-0 content (symlink-skip on existing names), while every later
round re-seeds fresh rollouts of the same scene -> round>=4 pools are genuine held-out
content for BOTH arms. We use ONE fixed pool for both checkpoints (the controlled
variable is the LoRA), restricted to the off-manifold `perturb` + `random` episodes.

Loads the 22B base ONCE; PEFT hot-swaps each arm's adapter + action projector. Reuses
infer_action_cond / eval_prope_metrics verbatim. Inference only.

Output: arm_compare.json + arm_compare.txt — per-arm mean PSNR/SSIM (all / agentview /
wrist), split by condition (perturb / random), and the strategic-minus-random delta.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
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
    _render_chunk,
    _stack_and_save,
    load_action_projector,
    load_lora_weights,
)
from score_chunk_latent_loss import _lora_state, build_strategy


def _kind_of(cid: str) -> str:
    for k in ("perturb", "random", "reach", "circle"):
        if f"_play_{k}_" in cid:
            return k
    return "gt"


def _episode_of(cid: str) -> str:
    m = re.match(r"^(.*)_chunk_\d+_target$", cid)
    return m.group(1) if m else cid


def _chunk_idx(cid: str) -> int:
    m = re.search(r"_chunk_(\d+)_target$", cid)
    return int(m.group(1)) if m else 0


def _eval_chunks(pool_root: Path, kinds: set[str], max_eps: int,
                 max_chunks: int) -> list[str]:
    """Chunk ids under the pool, filtered to `kinds`, capped per-episode and per-kind."""
    all_ids = sorted(f.stem for f in (pool_root / "latents" / "videos").glob("*.pt"))
    by_ep: dict[str, list[str]] = {}
    for cid in all_ids:
        if _kind_of(cid) in kinds:
            by_ep.setdefault(_episode_of(cid), []).append(cid)
    # deterministic episode order, cap episodes per kind
    eps = sorted(by_ep, key=lambda e: (_kind_of(e + "_chunk_00_target"), e))
    per_kind: dict[str, int] = {}
    out: list[str] = []
    for ep in eps:
        k = _kind_of(ep + "_chunk_00_target")
        if per_kind.get(k, 0) >= max_eps:
            continue
        per_kind[k] = per_kind.get(k, 0) + 1
        chunks = sorted(by_ep[ep], key=_chunk_idx)[:max_chunks]
        out.extend(chunks)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="N-arm held-out PSNR/SSIM eval at a matched step")
    p.add_argument("--base-checkpoint", default=DEFAULT_BASE)
    p.add_argument("--arm", action="append", default=[], metavar="NAME=DIR",
                   help="repeatable: arm name and its checkpoint dir (e.g. decaware=/path/ckpts)")
    p.add_argument("--baseline", default="random", help="arm name deltas are computed against")
    p.add_argument("--strategic-ckpt-dir", help="legacy 2-arm form (equivalent to --arm strategic=DIR)")
    p.add_argument("--random-ckpt-dir", help="legacy 2-arm form (equivalent to --arm random=DIR)")
    p.add_argument("--step", type=int, required=True, help="matched step for ALL arms")
    p.add_argument("--eval-pool", required=True,
                   help="held-out precomputed/ root (a round>=4 pool, untrained content)")
    p.add_argument("--kinds", default="perturb,random", help="episode kinds to eval")
    p.add_argument("--max-eps-per-kind", type=int, default=10)
    p.add_argument("--max-chunks-per-ep", type=int, default=4)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-inference-steps", type=int, default=30)
    p.add_argument("--drop-first", type=int, default=8)
    p.add_argument("--first-frame-conditioning-p", type=float, default=1.0)
    p.add_argument("--use-prope", action="store_true")
    p.add_argument("--prope-proj-dim", type=int, default=64)
    p.add_argument("--prope-image-width", type=int, default=256)
    p.add_argument("--prope-image-height", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-mp4", action="store_true", help="also save pred|gt panels")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_root = Path(args.eval_pool)
    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}

    arms: dict[str, Path] = {}
    for spec in args.arm:
        name, _, d = spec.partition("=")
        if not d:
            raise SystemExit(f"--arm expects NAME=DIR, got: {spec}")
        arms[name] = Path(d)
    if args.strategic_ckpt_dir:
        arms["strategic"] = Path(args.strategic_ckpt_dir)
    if args.random_ckpt_dir:
        arms["random"] = Path(args.random_ckpt_dir)
    if not arms:
        raise SystemExit("no arms given: pass --arm NAME=DIR (repeatable)")
    if args.baseline not in arms:
        raise SystemExit(f"--baseline {args.baseline!r} not among arms {sorted(arms)}")
    for arm, d in arms.items():
        for f in (f"lora_weights_step_{args.step}.safetensors",
                  f"action_projector_step_{args.step}.safetensors"):
            if not (d / f).exists():
                raise SystemExit(f"[{arm}] missing {f} under {d}")

    chunks = _eval_chunks(pool_root, kinds, args.max_eps_per_kind, args.max_chunks_per_ep)
    if not chunks:
        raise SystemExit(f"No {kinds} chunks under {pool_root}/latents/videos/")
    n_ep = len({_episode_of(c) for c in chunks})
    print(f"[eval] {len(chunks)} chunks over {n_ep} held-out episodes "
          f"({sorted(kinds)}) from {pool_root}", flush=True)

    # ---- load base ONCE; wrap with strategic LoRA (PEFT), hot-swap per arm ----
    components = load_model(
        checkpoint_path=args.base_checkpoint, device="cpu", dtype=dtype,
        with_video_vae_encoder=False, with_video_vae_decoder=True,
        with_audio_vae_decoder=False, with_vocoder=False, with_text_encoder=False,
    )
    if args.use_prope:
        n = enable_prope_on_model(components.transformer, proj_dim=args.prope_proj_dim,
                                  image_width=args.prope_image_width,
                                  image_height=args.prope_image_height)
        print(f"[setup] PRoPE enabled: {n} attn1 modules", flush=True)
    boot = next(iter(arms))  # any arm: the adapter is hot-swapped per arm below
    first_lora = arms[boot] / f"lora_weights_step_{args.step}.safetensors"
    transformer = load_lora_weights(components.transformer, first_lora)
    print("[setup] lora wrapped; moving transformer to device...", flush=True)
    transformer = transformer.to(device=device, dtype=dtype).eval()
    embeddings_processor = load_embeddings_processor(args.base_checkpoint, device=device, dtype=dtype)
    action_projector = load_action_projector(
        arms[boot] / f"action_projector_step_{args.step}.safetensors", dtype).to(device)
    strategy = build_strategy(action_projector, use_prope=args.use_prope,
                              prope_image_width=args.prope_image_width,
                              prope_image_height=args.prope_image_height,
                              first_frame_conditioning_p=args.first_frame_conditioning_p)
    patchifier = VideoLatentPatchifier(patch_size=1)  # noqa: F841  (parity w/ render script)
    print("[setup] ready; starting arm loop.", flush=True)

    render_args = SimpleNamespace(
        num_inference_steps=args.num_inference_steps, guidance_scale=1.0,
        negative_prompt="", seed=args.seed, no_first_frame=False,
        include_gt=True, include_context=False, device=args.device,
        use_prope=args.use_prope, prope_proj_dim=args.prope_proj_dim,
        prope_image_width=args.prope_image_width, prope_image_height=args.prope_image_height,
    )

    # per-arm, per-chunk records
    records: dict[str, list[dict]] = {a: [] for a in arms}
    for arm, ckdir in arms.items():
        lora_p = ckdir / f"lora_weights_step_{args.step}.safetensors"
        proj_p = ckdir / f"action_projector_step_{args.step}.safetensors"
        print(f"\n===== ARM {arm} (step {args.step}) =====", flush=True)
        set_peft_model_state_dict(transformer.get_base_model(), _lora_state(lora_p))
        action_projector = load_action_projector(proj_p, dtype).to(device)
        strategy._action_projector = action_projector

        for i, cid in enumerate(chunks):
            panels = _render_chunk(
                components=components, transformer=transformer,
                embeddings_processor=embeddings_processor, action_projector=action_projector,
                patchifier=patchifier, data_root=pool_root, chunk_id=cid,
                args=render_args, use_cfg=False,
            )
            px = score_all(_to_frames(panels["pred"]), _to_frames(panels["gt"]), args.drop_first)
            rec = {
                "chunk_id": cid, "episode": _episode_of(cid), "kind": _kind_of(cid),
                "psnr_all": px["all"]["psnr"], "ssim_all": px["all"]["ssim"],
                "psnr_agentview": px["agentview"]["psnr"], "ssim_agentview": px["agentview"]["ssim"],
                "psnr_wrist": px["wrist"]["psnr"], "ssim_wrist": px["wrist"]["ssim"],
            }
            records[arm].append(rec)
            if args.save_mp4:
                mdir = out_dir / "panels" / arm
                mdir.mkdir(parents=True, exist_ok=True)
                # keep the chunk-distinguishing tail: cid[:70] collided across chunks
                _stack_and_save(panels, mdir / f"{cid[-70:]}.mp4", render_args)
            print(f"  [{arm:9s}] {i + 1:3d}/{len(chunks)} {rec['kind']:7s} "
                  f"PSNR={rec['psnr_all']:.2f} (agent={rec['psnr_agentview']:.2f} "
                  f"wrist={rec['psnr_wrist']:.2f}) SSIM={rec['ssim_all']:.3f}", flush=True)

    # ---------- aggregate ----------
    def agg(recs: list[dict], field: str, kind: str | None = None) -> float | None:
        vals = [r[field] for r in recs if (kind is None or r["kind"] == kind)]
        return round(st.mean(vals), 4) if vals else None

    summary: dict = {"step": args.step, "eval_pool": str(pool_root),
                     "kinds": sorted(kinds), "n_chunks": len(chunks), "arms": {}}
    for arm in arms:
        recs = records[arm]
        block: dict = {}
        for cond in ["overall", *sorted(kinds)]:
            k = None if cond == "overall" else cond
            block[cond] = {
                "n": len([r for r in recs if (k is None or r["kind"] == k)]),
                "psnr_all": agg(recs, "psnr_all", k),
                "ssim_all": agg(recs, "ssim_all", k),
                "psnr_agentview": agg(recs, "psnr_agentview", k),
                "psnr_wrist": agg(recs, "psnr_wrist", k),
            }
        summary["arms"][arm] = block
    # per-arm deltas vs the baseline arm (higher PSNR/SSIM = better)
    deltas: dict = {a: {} for a in arms if a != args.baseline}
    for arm in deltas:
        for cond in ["overall", *sorted(kinds)]:
            s, r = summary["arms"][arm][cond], summary["arms"][args.baseline][cond]
            deltas[arm][cond] = {
                m: (None if s[m] is None or r[m] is None else round(s[m] - r[m], 4))
                for m in ("psnr_all", "ssim_all", "psnr_agentview", "psnr_wrist")
            }
    summary["baseline"] = args.baseline
    summary["delta_vs_baseline"] = deltas
    summary["per_chunk"] = records

    (out_dir / "arm_compare.json").write_text(json.dumps(summary, indent=2))

    lines = [
        f"Held-out arm eval @ step {args.step}   pool={pool_root.name}   "
        f"n={len(chunks)} chunks / {n_ep} eps   kinds={sorted(kinds)}",
        "",
        f"{'cond':10s} {'arm':10s} {'n':>3s} {'PSNR_all':>9s} {'SSIM_all':>9s} "
        f"{'PSNR_agent':>11s} {'PSNR_wrist':>11s}",
    ]
    for cond in ["overall", *sorted(kinds)]:
        for arm in arms:
            b = summary["arms"][arm][cond]
            lines.append(f"{cond:10s} {arm:10s} {b['n']:>3d} "
                         f"{_f(b['psnr_all']):>9s} {_f(b['ssim_all']):>9s} "
                         f"{_f(b['psnr_agentview']):>11s} {_f(b['psnr_wrist']):>11s}")
        for arm, dd in deltas.items():
            d = dd[cond]
            lines.append(f"{cond:10s} {'Δ ' + arm[:8]:10s} {'':>3s} "
                         f"{_f(d['psnr_all']):>9s} {_f(d['ssim_all']):>9s} "
                         f"{_f(d['psnr_agentview']):>11s} {_f(d['psnr_wrist']):>11s}")
        lines.append("")
    txt = "\n".join(lines)
    (out_dir / "arm_compare.txt").write_text(txt + "\n")
    print("\n" + txt)
    print(f"\nWrote {out_dir}/arm_compare.json + arm_compare.txt")


def _f(x: float | None) -> str:
    return "-" if x is None else f"{x:.3f}"


if __name__ == "__main__":
    main()
