#!/usr/bin/env python3
# ruff: noqa: T201
"""Object-centric PIXEL curiosity scorer for active learning.

Replaces the teacher-forced *latent* loss (score_chunk_latent_loss.py) with an
OBJECT-CENTRIC pixel-space signal, to fix its failure mode: the latent loss counts
the whole frame, so the wrist camera's "looking around" (large background motion)
can dominate the score even when the model handles the actual objects fine. Here the
curiosity is the model's pixel error measured ONLY inside the projected 3D bounding
boxes of the task objects — so it rewards selecting clips where the model is wrong
about the OBJECTS, not where the camera swept.

Per chunk:
  1. run the real action-conditioned rollout + VAE decode (reuse infer_action_cond
     ._render_chunk) -> pred, gt pixel videos [C,F,H,W], a 2-view hconcat
     [agentview | wrist], each tile 256x256.
  2. load the object 3D-box corners (objbbox/ stream) and the cameras (viewmats/Ks),
     resample both to the decoded frame count.
  3. project each object's 8 corners into agentview (left tile) and wrist (right tile);
     the 2D axis-aligned hull of the projection is that object's pixel mask (objects
     fully behind a camera are skipped -> wrist look-around with no object in view
     contributes nothing, which is the whole point).
  4. curiosity = mean |pred - gt| over the masked (object) pixels.

Output mirrors score_chunk_latent_loss.py's json (per_chunk + by_episode), so
al_select_and_grow.py reads `by_episode` unchanged. The selector ranks DESCENDING,
so higher curiosity == "more wrong about the objects" == selected first.

Example:
    python experiments/libero_sim/score_chunk_objbbox_pixel.py \
        --lora-path .../lora_weights_step_30000.safetensors \
        --action-projector .../action_projector_step_30000.safetensors \
        --data-root .../round0/precomputed \
        --use-prope --prope-proj-dim 64 --prope-image-width 256 --prope-image-height 256 \
        --num-inference-steps 20 --out .../latent_loss_r0.json
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from peft import set_peft_model_state_dict

from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.model.transformer.enable_prope import enable_prope_on_model
from ltx_trainer.model_loader import load_embeddings_processor, load_model

# Reuse the proven model loaders + the full rollout/decode path from inference.
from infer_action_cond import (  # noqa: E402
    _default_action_projector_path,
    _list_chunk_ids,
    _render_chunk,
    load_action_projector,
    load_lora_weights,
)
# Identical episode/kind parsing as the latent scorer (handles the _target suffix);
# _lora_state reused for the PEFT hot-swap to the frozen reference (reducible signal).
from score_chunk_latent_loss import _episode_of, _kind_of, _lora_state  # noqa: E402

_STAGED_BASE = "/storage/scratch1/8/lwang831/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
_CEDAR_BASE = "/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
DEFAULT_BASE = _STAGED_BASE if Path(_STAGED_BASE).exists() else _CEDAR_BASE

TILE_W = TILE_H = 256  # decoded video is [agentview 256 | wrist 256], each 256 tall


def _project_boxes(corners_world: np.ndarray, viewmat: np.ndarray, K: np.ndarray) -> list[tuple]:
    """Project objects' 8 world corners into one camera -> per-object 2D AABB in tile
    pixels. corners_world (n_obj,8,3); viewmat (4,4) world->cam; K (3,3). Objects with
    ANY corner at/behind the image plane are skipped (not cleanly visible)."""
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    boxes = []
    for o in range(corners_world.shape[0]):
        Xc = corners_world[o] @ R.T + t          # (8,3) camera frame
        if (Xc[:, 2] <= 1e-4).any():
            continue
        uv = Xc @ K.T                            # (8,3)
        u, v = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]
        u0, u1 = int(np.floor(u.min())), int(np.ceil(u.max()))
        v0, v1 = int(np.floor(v.min())), int(np.ceil(v.max()))
        u0, v0 = max(0, u0), max(0, v0)
        u1, v1 = min(TILE_W, u1), min(TILE_H, v1)
        if u1 > u0 and v1 > v0:
            boxes.append((u0, v0, u1, v1))
    return boxes


def _interaction_frames(cw: np.ndarray, motion_eps: float, dilate: int) -> np.ndarray:
    """Per-(decoded)-frame boolean: is the target object being INTERACTED WITH?

    In deterministic LIBERO an object only moves when acted on, so GT object motion
    (centroid displacement between consecutive frames > motion_eps, in world metres)
    is a clean interaction signal. Dilated by +-`dilate` frames so the approach/grasp
    ONSET (contact just before the object visibly moves) is included. This is the whole
    point of the thesis: we grade prediction of the object's RESPONSE to interaction,
    NOT free-space / perturbation motion that never touches it, NOR a static object."""
    n_obj, f_dec = cw.shape[0], cw.shape[1]
    if n_obj == 0 or f_dec < 2:
        return np.zeros(f_dec, dtype=bool)
    centroids = cw.mean(axis=2)                              # (n_obj, f_dec, 3)
    speed = np.linalg.norm(np.diff(centroids, axis=1), axis=2).max(axis=0)  # (f_dec-1,)
    moving = np.zeros(f_dec, dtype=bool)
    moving[1:] = speed > motion_eps
    for _ in range(max(0, dilate)):                          # symmetric temporal dilation
        d = moving.copy()
        d[1:] |= moving[:-1]; d[:-1] |= moving[1:]
        moving = d
    return moving


def score_chunk_pixels(render: dict, corners: torch.Tensor, viewmats: torch.Tensor,
                       Ks: torch.Tensor, motion_eps: float = 1e-3, gate_dilate: int = 2) -> dict:
    """Mean |pred-gt| inside the projected object boxes, plus per-view splits.

    Adds an INTERACTION-GATED variant (`loss_interact`): the same object-region error
    but restricted to frames where the object is actually being interacted with (GT in
    motion, see _interaction_frames). `loss` (ungated) is kept byte-identical so the AL
    loop's `by_episode` ranking is unaffected; the eval reads `loss_interact`."""
    pred = render["pred"].float().numpy()        # [C,F,H,W] in [0,1]
    gt = render["gt"].float().numpy()
    diff = np.abs(pred - gt).mean(axis=0)        # (F,H,W) per-pixel mean over channels
    f_dec, H, W = diff.shape

    if corners.numel() == 0 or corners.shape[0] == 0:
        return {"loss": 0.0, "agentview": 0.0, "wrist": 0.0, "vis_frac": 0.0, "n_obj": 0,
                "loss_interact": 0.0, "interact_frac": 0.0, "obj_moved": False}

    cw = corners.numpy()                         # (n_obj, F_raw, 8, 3)
    vm = viewmats.numpy()                        # (V, F_raw, 4, 4)
    ks = Ks.numpy()                              # (V, 3, 3)
    f_raw = min(cw.shape[1], vm.shape[1])
    idx = np.linspace(0, f_raw - 1, f_dec).round().astype(int)  # raw -> decoded frame map
    cw, vm = cw[:, idx], vm[:, idx]

    interact = _interaction_frames(cw, motion_eps, gate_dilate)  # (f_dec,) bool

    # (sum_err, n_px) accumulators: combined + per tile (agentview=0 / wrist=1),
    # plus an interaction-gated combined accumulator ("int").
    acc = {"all": [0.0, 0], "agentview": [0.0, 0], "wrist": [0.0, 0], "int": [0.0, 0]}
    vis_frames = 0
    vis_interact_frames = 0
    for f in range(f_dec):
        mask = np.zeros((H, W), dtype=bool)
        for v, (x_off, name) in enumerate([(0, "agentview"), (TILE_W, "wrist")]):
            tmask = np.zeros((H, W), dtype=bool)
            for (u0, v0, u1, v1) in _project_boxes(cw[:, f], vm[v, f], ks[v]):
                tmask[v0:v1, x_off + u0:x_off + u1] = True
            if tmask.any():
                vals = diff[f][tmask]
                acc[name][0] += float(vals.sum()); acc[name][1] += int(vals.size)
            mask |= tmask
        if mask.any():
            vis_frames += 1
            vals = diff[f][mask]
            acc["all"][0] += float(vals.sum()); acc["all"][1] += int(vals.size)
            if interact[f]:
                vis_interact_frames += 1
                acc["int"][0] += float(vals.sum()); acc["int"][1] += int(vals.size)

    def _mean(k):
        return acc[k][0] / acc[k][1] if acc[k][1] > 0 else 0.0

    return {"loss": _mean("all"), "agentview": _mean("agentview"), "wrist": _mean("wrist"),
            "vis_frac": vis_frames / max(f_dec, 1), "n_obj": int(corners.shape[0]),
            "loss_interact": _mean("int"),
            "interact_frac": vis_interact_frames / max(vis_frames, 1),
            "obj_moved": bool(interact.any())}


def main() -> None:
    p = argparse.ArgumentParser(description="Object-centric pixel curiosity scorer (active-learning signal)")
    p.add_argument("--base-checkpoint", default=DEFAULT_BASE)
    p.add_argument("--lora-path", required=True)
    p.add_argument("--action-projector", default=None)
    p.add_argument("--data-root", required=True, help="precomputed/ root (latents/ cameras/ objbbox/ ...)")
    p.add_argument("--out", required=True, help="Output json (.csv written alongside)")
    p.add_argument("--indices", default=None, help="Comma-separated chunk indices (default: all)")
    p.add_argument("--max-chunks", type=int, default=0, help="0 = all")
    p.add_argument("--num-inference-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--motion-eps", type=float, default=1e-3,
                   help="object centroid displacement (world m/frame) above which the "
                        "object counts as being INTERACTED WITH (interaction-gated metric)")
    p.add_argument("--gate-dilate", type=int, default=2,
                   help="dilate interaction frames by +-N to include the contact/grasp onset")
    # --- reducible (RHO-style) signal: score against a frozen REFERENCE checkpoint. ---
    # metric=loss     : raw object-region error (cur_err) — the old max-loss signal.
    # metric=progress : base_err - cur_err  (learning progress vs the frozen reference;
    #                   HIGH where the current model has learnt the objects the frozen
    #                   reference could not — i.e. the LEARNABLE frontier, not irreducible noise).
    # metric=progress_norm : (base_err - cur_err) / (base_err + eps)  — fraction of the
    #                   reference's object-error the current model has removed (bounded).
    p.add_argument("--score-metric", choices=["loss", "progress", "progress_norm"], default="loss")
    p.add_argument("--baseline-lora", default=None,
                   help="frozen reference LoRA (e.g. step-30000); required for progress metrics")
    p.add_argument("--baseline-projector", default=None,
                   help="frozen reference action projector (defaults to sibling of --baseline-lora)")
    p.add_argument("--use-prope", action="store_true")
    p.add_argument("--prope-proj-dim", type=int, default=64)
    p.add_argument("--prope-image-width", type=int, default=256)
    p.add_argument("--prope-image-height", type=int, default=256)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16
    data_root = Path(args.data_root)

    all_ids = _list_chunk_ids(data_root)
    if not all_ids:
        raise SystemExit(f"No chunks under {data_root}/latents/videos/")
    if args.indices is not None:
        idxs = [int(x) for x in args.indices.split(",") if x.strip() != ""]
        all_ids = [all_ids[i] for i in idxs]
    if args.max_chunks > 0:
        all_ids = all_ids[: args.max_chunks]
    print(f"Scoring {len(all_ids)} chunks (object-bbox pixel curiosity) | "
          f"steps={args.num_inference_steps} prope={args.use_prope}")

    # ---- Load model ONCE, WITH the VAE decoder (we need pixels). No CFG -> no text encoder. ----
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
    patchifier = VideoLatentPatchifier(patch_size=1)

    # args namespace _render_chunk reads off (include_gt so we get the GT to diff against).
    render_args = SimpleNamespace(
        device=args.device, num_inference_steps=args.num_inference_steps,
        guidance_scale=1.0, negative_prompt="", seed=args.seed, no_first_frame=False,
        include_gt=True, include_context=False, use_prope=args.use_prope,
        prope_proj_dim=args.prope_proj_dim,
        prope_image_width=args.prope_image_width, prope_image_height=args.prope_image_height,
    )

    reducible = args.score_metric != "loss"
    if reducible and not args.baseline_lora:
        raise SystemExit(f"--score-metric {args.score_metric} requires --baseline-lora (the frozen reference)")

    def _score_all(proj, tag: str) -> dict[str, dict]:
        """Object-region error for every chunk under the CURRENTLY-loaded adapter + `proj`."""
        out: dict[str, dict] = {}
        t0 = time.perf_counter()
        for i, cid in enumerate(all_ids):
            render = _render_chunk(
                components=components, transformer=transformer,
                embeddings_processor=embeddings_processor, action_projector=proj,
                patchifier=patchifier, data_root=data_root, chunk_id=cid,
                args=render_args, use_cfg=False,
            )
            cam = torch.load(data_root / "cameras" / "videos" / f"{cid}.pt", map_location="cpu", weights_only=False)
            obj = torch.load(data_root / "objbbox" / "videos" / f"{cid}.pt", map_location="cpu", weights_only=False)
            out[cid] = score_chunk_pixels(render, obj["corners"], cam["viewmats"], cam["Ks"],
                                          motion_eps=args.motion_eps, gate_dilate=args.gate_dilate)
            if (i + 1) % 5 == 0 or i + 1 == len(all_ids):
                rate = (i + 1) / (time.perf_counter() - t0)
                print(f"  [{tag} {i+1}/{len(all_ids)}] {cid[:44]} err={out[cid]['loss']:.4f} "
                      f"({rate*60:.1f} chunks/min)")
        return out

    # ---- Pass 1: current model. Pass 2 (only if reducible): frozen reference via PEFT hot-swap. ----
    cur = _score_all(action_projector, "cur")
    base: dict[str, dict] = {}
    if reducible:
        bproj_path = Path(args.baseline_projector) if args.baseline_projector \
            else _default_action_projector_path(Path(args.baseline_lora))
        if bproj_path is None:
            raise SystemExit("No baseline projector next to --baseline-lora; pass --baseline-projector.")
        print(f"Reducible: hot-swap to frozen reference {Path(args.baseline_lora).name} for pass 2")
        set_peft_model_state_dict(transformer.get_base_model(), _lora_state(Path(args.baseline_lora)))
        base_proj = load_action_projector(bproj_path, dtype).to(device)
        base = _score_all(base_proj, "ref")

    results: dict[str, dict] = {}
    for cid in all_ids:
        sc = dict(cur[cid])
        sc["kind"], sc["episode"] = _kind_of(cid), _episode_of(cid)
        if reducible:
            base_err = base[cid]["loss"]
            prog = base_err - sc["loss"]              # >0 == current beats the frozen reference
            sc["base_loss"] = base_err
            sc["progress"] = prog
            sc["score"] = prog if args.score_metric == "progress" else prog / (base_err + 1e-6)
            # interaction-gated reducible signal: object-response error removed since the ancestor
            base_int = base[cid].get("loss_interact", 0.0)
            sc["base_loss_interact"] = base_int
            sc["progress_interact"] = base_int - sc.get("loss_interact", 0.0)
        else:
            sc["score"] = sc["loss"]                  # raw object-region error (max-loss signal)
        results[cid] = sc

    # ---- Aggregates: by_episode is what the selector ranks by (descending) ----
    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    by_ep: dict[str, list[float]] = {}
    by_kind: dict[str, list[float]] = {}
    for cid, sc in results.items():
        by_ep.setdefault(sc["episode"], []).append(sc["score"])
        by_kind.setdefault(sc["kind"], []).append(sc["loss"])
    ep_agg = {e: _mean(v) for e, v in sorted(by_ep.items())}
    kind_agg = {k: {"mean": _mean(v), "n": len(v)} for k, v in sorted(by_kind.items())}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {"data_root": str(data_root), "lora": str(args.lora_path),
                 "n_chunks": len(results), "signal": "objbbox_pixel",
                 "score_metric": args.score_metric, "baseline_lora": args.baseline_lora,
                 "num_inference_steps": args.num_inference_steps, "use_prope": args.use_prope},
        "per_chunk": results, "by_kind": kind_agg, "by_episode": ep_agg,
    }
    out.write_text(json.dumps(payload, indent=2))
    with out.with_suffix(".csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id", "kind", "episode", "score", "agentview", "wrist", "vis_frac", "n_obj"])
        for cid, sc in sorted(results.items(), key=lambda kv: -kv[1]["score"]):
            w.writerow([cid, sc["kind"], sc["episode"], f"{sc['score']:.6f}",
                        f"{sc['agentview']:.6f}", f"{sc['wrist']:.6f}",
                        f"{sc['vis_frac']:.3f}", sc["n_obj"]])

    print(f"\nWrote {out} and {out.with_suffix('.csv')} (signal=objbbox_pixel)")
    print("Per-kind mean object-curiosity (sanity: random/circle should exceed perturb/gt):")
    for k, a in kind_agg.items():
        print(f"  {k:8s} n={a['n']:3d}  mean={a['mean']:.4f}")


if __name__ == "__main__":
    main()
