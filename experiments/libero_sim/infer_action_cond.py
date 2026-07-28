#!/usr/bin/env python3
# ruff: noqa: T201
"""Action-conditioned inference for the GVT v2v IC-LoRA + action-cond model.

The stock ``packages/ltx-trainer/scripts/inference.py`` supports T2V / I2V / V2V + LoRA
but has NO action conditioning and expects an mp4 reference video (it VAE-encodes it).
Our model adds Stream-2 action tokens (ActionMLPProjector) and is trained on *precomputed*
latents (reference context tile + target + per-frame actions + Gemma text features). This
script reproduces the exact training-time conditioning at sampling time:

    p(context, text, observation, actions) -> target_video
      context     = reference_latents/<id>.pt   (clean 2x2 tile, timestep 0)
      text        = conditions/<id>.pt          (pre-connector Gemma features -> connector)
      observation = first latent frame of the target kept clean (optional)
      actions     = actions/<id>.pt (33,8) -> resample to latent rate -> ActionMLPProjector
      target      = denoised from noise

It reads straight from the precomputed dataset, so it needs NO VAE encoder and NO Gemma
LLM (with guidance_scale=1.0, the default). Only the transformer (+LoRA+action projector),
the embeddings-processor connector, and the video VAE decoder are loaded.

Example:
    python experiments/libero_sim/infer_action_cond.py \
        --lora-path /storage/.../libero90_v2v_ic_lora/checkpoints/lora_weights_step_19000.safetensors \
        --index 0 --include-gt --include-context \
        --output /storage/.../tmp/rollout_step19000.mp4
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import torch
from einops import rearrange
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import CFGGuider
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.conditioning.types.action_cond import ACTION_SPATIAL_SENTINEL, ActionMLPProjector
from ltx_core.model.transformer.enable_prope import enable_prope_on_model
from ltx_core.model.transformer.model import X0Model
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
from ltx_core.text_encoders.gemma import convert_to_additive_mask
from ltx_core.types import LatentState, SpatioTemporalScaleFactors, VideoLatentShape
from ltx_trainer.model_loader import load_embeddings_processor, load_model
from ltx_trainer.training_strategies.base_strategy import DEFAULT_FPS
from ltx_trainer.training_strategies.video_to_video import VideoToVideoStrategy
from ltx_trainer.video_utils import save_video

VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()

# Pre-staged LTX-2.3 base checkpoint + Gemma dir (see memory: server_paths).
# Prefer the scratch1 copy (Lustre, ~130 MB/s) over cedar0 (pathologically slow: a
# 43 GB mmap load page-faults for 40+ min). Fall back to cedar0 if scratch1 is purged.
_SCRATCH_BASE = "/storage/scratch1/8/lwang831/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
_CEDAR_BASE = "/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
DEFAULT_BASE = _SCRATCH_BASE if os.path.exists(_SCRATCH_BASE) else _CEDAR_BASE
DEFAULT_DATA_ROOT = "/storage/scratch1/8/lwang831/libero90_precomputed/precomputed"
DEFAULT_SRC_ROOT = "/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/LIBERO-datasets/libero_90"
CHUNK_LEN = 33  # matches build_libero_context.CHUNK_LEN


def chunk_starts(T: int, chunk: int) -> list[int]:
    """Non-overlapping windows of length `chunk`; final window shift-overlapped to T-chunk.
    Mirrors experiments/droid_action_cond/chunk_context.chunk_starts."""
    if T < chunk:
        return []
    starts = list(range(0, T - chunk + 1, chunk))
    if starts[-1] + chunk < T:
        starts.append(T - chunk)
    return starts


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def _extract_lora_target_modules(state_dict: dict[str, torch.Tensor]) -> list[str]:
    pattern = re.compile(r"(.+)\.lora_[AB]\.")
    return sorted({m.group(1) for k in state_dict if (m := pattern.match(k))})


def load_lora_weights(transformer: torch.nn.Module, lora_path: str | Path) -> torch.nn.Module:
    """Wrap transformer with PEFT and load LoRA adapters (ComfyUI-format safetensors)."""
    print(f"Loading LoRA from {lora_path} ...")
    state_dict = load_file(str(lora_path))
    state_dict = {k.replace("diffusion_model.", "", 1): v for k, v in state_dict.items()}
    target_modules = _extract_lora_target_modules(state_dict)
    if not target_modules:
        raise ValueError(f"No LoRA target modules found in {lora_path}")
    rank = next(v.shape[0] for k, v in state_dict.items() if "lora_A" in k and v.ndim == 2)
    print(f"  {len(target_modules)} target modules, rank {rank}")
    transformer = get_peft_model(
        transformer,
        LoraConfig(r=rank, lora_alpha=rank, target_modules=target_modules, lora_dropout=0.0, init_lora_weights=True),
    )
    set_peft_model_state_dict(transformer.get_base_model(), state_dict)
    return transformer


def load_action_projector(path: str | Path, dtype: torch.dtype) -> ActionMLPProjector:
    """Reconstruct the ActionMLPProjector from a saved state dict (shapes infer the dims)."""
    state = load_file(str(path))
    w_in = state["net.0.weight"]  # (mid, action_dim)
    w_out = state["net.2.weight"]  # (hidden, mid)
    mid, action_dim = w_in.shape
    hidden_dim = w_out.shape[0]
    print(f"Loading action projector from {path} (action_dim={action_dim}, mid={mid}, hidden={hidden_dim})")
    proj = ActionMLPProjector(action_dim=action_dim, hidden_dim=hidden_dim, mid_dim=mid)
    proj.load_state_dict(state)
    return proj.to(dtype=dtype).eval()


def _default_action_projector_path(lora_path: Path) -> Path | None:
    m = re.search(r"step_(\d+)", lora_path.name)
    if not m:
        return None
    cand = lora_path.parent / f"action_projector_step_{m.group(1)}.safetensors"
    return cand if cand.exists() else None


# ---------------------------------------------------------------------------
# Precomputed-data access
# ---------------------------------------------------------------------------
def _list_chunk_ids(data_root: Path) -> list[str]:
    files = sorted((data_root / "latents" / "videos").glob("*.pt"))
    return [f.stem for f in files]


def _load_chunk(data_root: Path, chunk_id: str) -> dict:
    def _load(sub: str) -> dict:
        return torch.load(data_root / sub / "videos" / f"{chunk_id}.pt", map_location="cpu", weights_only=False)

    return {
        "target": _load("latents"),
        "reference": _load("reference_latents"),
        "conditions": _load("conditions"),
        "actions": _load("actions"),
    }


# ---------------------------------------------------------------------------
# Positions (matching base_strategy._get_video_positions / action_cond.py)
# ---------------------------------------------------------------------------
def _video_positions(
    patchifier: VideoLatentPatchifier, frames: int, height: int, width: int, fps: float, device, dtype
) -> torch.Tensor:
    from ltx_core.components.patchifiers import get_pixel_coords

    latent_coords = patchifier.get_patch_grid_bounds(
        output_shape=VideoLatentShape(frames=frames, height=height, width=width, batch=1, channels=128),
        device=device,
    )
    pos = get_pixel_coords(latent_coords=latent_coords, scale_factors=VIDEO_SCALE_FACTORS, causal_fix=True).to(dtype)
    pos[:, 0, ...] = pos[:, 0, ...] / fps
    return pos


def _action_positions(n: int, device, dtype) -> torch.Tensor:
    t_idx = torch.arange(n, device=device, dtype=dtype)
    time_pos = torch.stack([t_idx, t_idx + 1.0], dim=-1)
    sentinel = torch.full((n, 2), fill_value=ACTION_SPATIAL_SENTINEL, device=device, dtype=dtype)
    pos = torch.stack([time_pos, sentinel, sentinel], dim=0)  # (3, n, 2)
    return pos.unsqueeze(0).contiguous()  # (1, 3, n, 2)


# ---------------------------------------------------------------------------
# PRoPE per-token cameras (mirrors VideoToVideoStrategy at sampling time)
# ---------------------------------------------------------------------------
from types import SimpleNamespace  # noqa: E402


def _build_prope_cameras(
    *, data_root, chunk_id, positions, frames, height, width,
    ref_seq, tgt_seq, action_seq, args, device, dtype,
):
    """Load cameras/<id>.pt and expand to one (viewmat, K) per token, EXACTLY as the
    PRoPE training strategy does — reusing VideoToVideoStrategy._build_per_token_cameras
    so there is zero logic drift between train and inference. Returns (viewmats, Ks)."""
    cam = torch.load(data_root / "cameras" / "videos" / f"{chunk_id}.pt",
                     map_location="cpu", weights_only=False)
    batch = {"cameras": {
        "viewmats": cam["viewmats"].unsqueeze(0),  # (1, V, F, 4, 4)
        "Ks": cam["Ks"].unsqueeze(0),              # (1, V, 3, 3)
    }}
    # Shim carrying just the three attributes the method reads off `self`.
    shim = SimpleNamespace(
        _acceleration_config=SimpleNamespace(
            use_prope=True,
            prope_image_width=args.prope_image_width,
            prope_image_height=args.prope_image_height,
        ),
        _N_VIEWS=VideoToVideoStrategy._N_VIEWS,
        _SPATIAL_SCALE=VideoToVideoStrategy._SPATIAL_SCALE,
    )
    return VideoToVideoStrategy._build_per_token_cameras(
        shim,
        batch=batch,
        positions=positions,
        latent_height=height,
        latent_width=width,
        num_frames=frames,
        ref_seq_len=ref_seq,
        target_seq_len=tgt_seq,
        action_seq_len=action_seq,
        device=device,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------
def _decode(vae_decoder, latent_bcfhw: torch.Tensor, device) -> torch.Tensor:
    vae_decoder.to(device)
    latent = latent_bcfhw.to(device=device, dtype=torch.bfloat16)
    tiling = TilingConfig(
        spatial_config=SpatialTilingConfig(tile_size_in_pixels=192, tile_overlap_in_pixels=64),
        temporal_config=TemporalTilingConfig(tile_size_in_frames=48, tile_overlap_in_frames=24),
    )
    chunks = [c for c in vae_decoder.tiled_decode(latent, tiling_config=tiling)]
    video = torch.cat(chunks, dim=2)
    video = ((video + 1.0) / 2.0).clamp(0.0, 1.0)
    vae_decoder.to("cpu")
    return video[0].float().cpu()  # [C, F, H, W]


def _unpatchify(patchifier: VideoLatentPatchifier, latent_bseqc: torch.Tensor, frames, height, width) -> torch.Tensor:
    return patchifier.unpatchify(
        latent_bseqc,
        output_shape=VideoLatentShape(batch=1, channels=128, frames=frames, height=height, width=width),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@torch.no_grad()
def _render_chunk(
    *,
    components,
    transformer,
    embeddings_processor,
    action_projector,  # None => action stream ablated
    patchifier,
    data_root: Path,
    chunk_id: str,
    args,
    use_cfg: bool,
) -> dict:
    """Run one action-conditioned (or ablated) rollout for `chunk_id`.

    Returns decoded {'pred','gt','ctx','fps'} tensors ([C,F,H,W], gt/ctx may be None),
    with `ctx` already resized to the prediction's spatial dims. Saving is the caller's job.
    """
    device = torch.device(args.device)
    dtype = torch.bfloat16
    chunk = _load_chunk(data_root, chunk_id)

    # ---- Dimensions (these are LATENT dims in the precomputed files) ----
    tgt = chunk["target"]
    frames, height, width = int(tgt["num_frames"]), int(tgt["height"]), int(tgt["width"])
    fps = float(tgt.get("fps", DEFAULT_FPS))
    ref = chunk["reference"]
    ref_frames, ref_height, ref_width = int(ref["num_frames"]), int(ref["height"]), int(ref["width"])

    target_latent = tgt["latents"].unsqueeze(0).to(device=device, dtype=dtype)  # (1,128,F,H,W)
    ref_latent = ref["latents"].unsqueeze(0).to(device=device, dtype=dtype)

    target_patched = patchifier.patchify(target_latent)  # (1, tgt_seq, 128)
    ref_patched = patchifier.patchify(ref_latent)  # (1, ref_seq, 128)
    ref_seq = ref_patched.shape[1]
    tgt_seq = target_patched.shape[1]

    # ---- Text context: pre-connector Gemma features -> connector (block 3) ----
    cond = chunk["conditions"]
    vfeat = cond["video_prompt_embeds"].unsqueeze(0).to(device=device, dtype=dtype)  # (1,1024,4096)
    afeat = cond.get("audio_prompt_embeds")
    afeat = afeat.unsqueeze(0).to(device=device, dtype=dtype) if afeat is not None else None
    mask = cond["prompt_attention_mask"].unsqueeze(0).to(device=device)  # (1,1024)
    add_mask = convert_to_additive_mask(mask, vfeat.dtype)
    video_ctx, _audio_ctx, ctx_mask = embeddings_processor.create_embeddings(vfeat, afeat, add_mask)

    neg_ctx = None
    if use_cfg:
        te = components.text_encoder.to(device)
        neg_hs, neg_mask = te.encode(args.negative_prompt)
        neg_out = embeddings_processor.process_hidden_states(neg_hs, neg_mask)
        neg_ctx = neg_out.video_encoding
        te.model.to("cpu")

    # ---- Positions ----
    ref_pos = _video_positions(patchifier, ref_frames, ref_height, ref_width, fps, device, dtype)
    tgt_pos = _video_positions(patchifier, frames, height, width, fps, device, dtype)
    positions = torch.cat([ref_pos, tgt_pos], dim=2)

    # ---- denoise_mask: ref clean(0), target denoise(1) except first frame, action clean(0) ----
    ref_mask = torch.zeros(1, ref_seq, 1, device=device, dtype=torch.float32)
    tgt_mask = torch.ones(1, tgt_seq, 1, device=device, dtype=torch.float32)
    if not args.no_first_frame:
        first_frame_tokens = height * width
        tgt_mask[:, :first_frame_tokens] = 0.0  # observation: keep first latent frame clean
    combined_clean = torch.cat([ref_patched, target_patched], dim=1)
    combined_mask = torch.cat([ref_mask, tgt_mask], dim=1)

    # ---- Action tokens (Stream 2): resample (F_pixel,8) -> (F_latent,8) -> project -> append ----
    action_seq = 0
    if action_projector is not None:
        actions_raw = chunk["actions"]["latents"].unsqueeze(0).to(device=device, dtype=dtype)  # (1, F_pixel, 8)
        f_pixel = actions_raw.shape[1]
        if f_pixel >= frames:
            src_idx = torch.linspace(0, f_pixel - 1, steps=frames, device=device).round().long()
            actions_rs = actions_raw[:, src_idx, :]
        else:
            actions_rs = torch.nn.functional.interpolate(
                actions_raw.transpose(1, 2), size=frames, mode="nearest"
            ).transpose(1, 2)
        action_tokens = action_projector(actions_rs)  # (1, F_latent, 128)
        action_seq = action_tokens.shape[1]
        combined_clean = torch.cat([combined_clean, action_tokens], dim=1)
        combined_mask = torch.cat([combined_mask, torch.zeros(1, action_seq, 1, device=device, dtype=torch.float32)], dim=1)
        positions = torch.cat([positions, _action_positions(action_seq, device, dtype)], dim=2)

    # ---- PRoPE per-token cameras (only when the PRoPE checkpoint is loaded) ----
    prope_viewmats, prope_Ks = (None, None)
    if getattr(args, "use_prope", False):
        prope_viewmats, prope_Ks = _build_prope_cameras(
            data_root=data_root, chunk_id=chunk_id, positions=positions,
            frames=frames, height=height, width=width,
            ref_seq=ref_seq, tgt_seq=tgt_seq, action_seq=action_seq,
            args=args, device=device, dtype=dtype,
        )

    clean_state = LatentState(
        latent=combined_clean,
        denoise_mask=combined_mask,
        positions=positions,
        clean_latent=combined_clean,
    )

    # ---- Noise the denoise tokens, run the denoising loop ----
    generator = torch.Generator(device=device).manual_seed(args.seed)
    state = GaussianNoiser(generator=generator)(latent_state=clean_state, noise_scale=1.0)

    scheduler = LTX2Scheduler()
    sigmas = scheduler.execute(steps=args.num_inference_steps).to(device).float()
    stepper = EulerDiffusionStep()
    cfg = CFGGuider(args.guidance_scale)
    x0_model = X0Model(transformer)

    print(f"  [{chunk_id[:50]}] ref_seq={ref_seq} tgt_seq={tgt_seq} action_seq={action_seq} steps={args.num_inference_steps} cfg={args.guidance_scale} action={'on' if action_projector is not None else 'OFF'}")
    _TIME = os.environ.get("GVT_TIME", "") not in ("", "0")
    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
    if _TIME:
        _sync(); _t_dit0 = time.perf_counter()
    with torch.autocast(device_type=device.type, dtype=dtype):
        for step_idx, sigma in enumerate(sigmas[:-1]):
            video = Modality(
                enabled=True,
                latent=state.latent,
                sigma=sigma.repeat(state.latent.shape[0]),
                timesteps=sigma * state.denoise_mask,
                positions=state.positions,
                context=video_ctx,
                context_mask=ctx_mask,
                viewmats=prope_viewmats,
                Ks=prope_Ks,
            )
            denoised, _ = x0_model(video=video, audio=None, perturbations=None)
            if cfg.enabled() and neg_ctx is not None:
                neg = x0_model(video=Modality(
                    enabled=True, latent=state.latent, sigma=sigma.repeat(state.latent.shape[0]),
                    timesteps=sigma * state.denoise_mask, positions=state.positions, context=neg_ctx, context_mask=None,
                    viewmats=prope_viewmats, Ks=prope_Ks,
                ), audio=None, perturbations=None)[0]
                denoised = denoised + cfg.delta(denoised, neg)
            # keep conditioning tokens clean
            denoised = denoised * state.denoise_mask + clean_state.latent.float() * (1 - state.denoise_mask)
            state = LatentState(
                latent=stepper.step(sample=video.latent, denoised_sample=denoised, sigmas=sigmas, step_index=step_idx),
                denoise_mask=state.denoise_mask,
                positions=state.positions,
                clean_latent=state.clean_latent,
            )

    if _TIME:
        _sync(); _t_dit = time.perf_counter() - _t_dit0

    # ---- Decode predicted target ----
    if _TIME:
        _sync(); _t = time.perf_counter()
    pred_latent = state.latent[:, ref_seq : ref_seq + tgt_seq]
    pred_bcfhw = _unpatchify(patchifier, pred_latent, frames, height, width)
    pred_video = _decode(components.video_vae_decoder, pred_bcfhw, device)  # [C,F,H,W]
    if _TIME:
        _sync(); _t_dec_pred = time.perf_counter() - _t

    gt_video = None
    _t_dec_gt = _t_dec_ctx = 0.0
    if args.include_gt:
        if _TIME:
            _sync(); _t = time.perf_counter()
        gt_bcfhw = _unpatchify(patchifier, target_patched, frames, height, width)
        gt_video = _decode(components.video_vae_decoder, gt_bcfhw, device)
        if _TIME:
            _sync(); _t_dec_gt = time.perf_counter() - _t
    ctx_video = None
    if args.include_context:
        if _TIME:
            _sync(); _t = time.perf_counter()
        ctx_bcfhw = _unpatchify(patchifier, ref_patched, ref_frames, ref_height, ref_width)
        ctx_video = _decode(components.video_vae_decoder, ctx_bcfhw, device)
        if _TIME:
            _sync(); _t_dec_ctx = time.perf_counter() - _t
        if ctx_video.shape[2:] != pred_video.shape[2:]:
            ctx_video = torch.nn.functional.interpolate(
                rearrange(ctx_video, "c f h w -> f c h w"), size=pred_video.shape[2:], mode="bilinear", align_corners=False
            )
            ctx_video = rearrange(ctx_video, "f c h w -> c f h w")

    if _TIME:
        print(f"  [TIME] DiT({args.num_inference_steps} steps)={_t_dit:6.2f}s  "
              f"decode_pred={_t_dec_pred:5.2f}s  decode_gt={_t_dec_gt:5.2f}s  "
              f"decode_ctx={_t_dec_ctx:5.2f}s  | per-step DiT={_t_dit/max(args.num_inference_steps,1)*1000:5.1f}ms")

    return {"pred": pred_video, "gt": gt_video, "ctx": ctx_video, "fps": fps}


def _stack_and_save(panels_dict: dict, out_path: Path, args) -> None:
    """Stack pred|gt|ctx vertically (along H) and save one mp4."""
    panels = [panels_dict["pred"]]
    if args.include_gt and panels_dict["gt"] is not None:
        panels.append(panels_dict["gt"])
    if args.include_context and panels_dict["ctx"] is not None:
        panels.append(panels_dict["ctx"])
    out_video = torch.cat(panels, dim=2) if len(panels) > 1 else panels[0]  # vertical (H)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(video_tensor=out_video, output_path=out_path, fps=float(panels_dict["fps"]), video_format="CFHW")
    layout = "pred" + ("|gt" if args.include_gt else "") + ("|ctx" if args.include_context else "")
    print(f"  ✓ {out_path}  ({layout}, vertical)  frames={out_video.shape[1]}")


def _episode_chunk_ids(all_ids: list[str], episode_stem: str) -> list[str]:
    """All chunk ids '<episode_stem>_chunk_NN_target' for one episode, ordered by NN."""
    import re as _re
    pat = _re.compile(rf"^{_re.escape(episode_stem)}_chunk_(\d+)_target$")
    matched = [(int(m.group(1)), cid) for cid in all_ids if (m := pat.match(cid))]
    return [cid for _, cid in sorted(matched)]


def _episode_length(episode_stem: str, src_root: Path) -> int | None:
    """Recover the source demo's frame count T (so chunk_starts pins the overlaps)."""
    try:
        import h5py
        task, demo = episode_stem.rsplit("_demo_", 1)
        f = src_root / f"{task}.hdf5"
        if not f.exists():
            return None
        with h5py.File(str(f), "r") as h:
            return int(h["data"][f"demo_{demo}"]["actions"].shape[0])
    except Exception as ex:  # noqa: BLE001
        print(f"  [warn] could not read T from source ({ex}); concatenating chunks without overlap trim")
        return None


@torch.no_grad()
def _render_episode(*, episode_stem, all_ids, use_actions, out_path, src_root, render_fn, args) -> None:
    """Stitch every chunk of one episode along time into a single pred|gt|ctx video.

    Each 33-frame window is conditioned independently (its own context tile + actions +
    clean first frame) — i.e. teacher-forced per chunk, NOT an autoregressive rollout.
    The only inter-chunk overlap (chunk_starts' final shifted window) is trimmed exactly
    using the source episode length T.
    """
    chunk_ids = _episode_chunk_ids(all_ids, episode_stem)
    if not chunk_ids:
        raise SystemExit(f"No chunks found for episode '{episode_stem}'")
    T = _episode_length(episode_stem, src_root)
    starts = chunk_starts(T, CHUNK_LEN) if T is not None else None
    if starts is not None and len(starts) != len(chunk_ids):
        print(f"  [warn] chunk_starts gave {len(starts)} windows but {len(chunk_ids)} chunks on disk; no trim")
        starts = None
    print(f"  episode '{episode_stem[:60]}': {len(chunk_ids)} chunks, T={T}, actions={'on' if use_actions else 'OFF'}")

    pred_parts, gt_parts, ctx_parts = [], [], []
    fps = float(DEFAULT_FPS)
    for i, cid in enumerate(chunk_ids):
        r = render_fn(cid, use_actions)
        fps = float(r["fps"])
        f_dec = r["pred"].shape[1]  # decoded pixel frames in this window (== CHUNK_LEN normally)
        # overlap (in window frames) of this chunk with the previous one; only the last is ever > 0
        skip_win = 0 if (starts is None or i == 0) else max(0, (starts[i - 1] + CHUNK_LEN) - starts[i])
        skip = round(skip_win * f_dec / CHUNK_LEN)
        sl = slice(skip, None)
        pred_parts.append(r["pred"][:, sl])
        if r["gt"] is not None:
            gt_parts.append(r["gt"][:, sl])
        if r["ctx"] is not None:
            ctx_parts.append(r["ctx"][:, sl])

    stitched = {
        "pred": torch.cat(pred_parts, dim=1),
        "gt": torch.cat(gt_parts, dim=1) if gt_parts else None,
        "ctx": torch.cat(ctx_parts, dim=1) if ctx_parts else None,
        "fps": fps,
    }
    _stack_and_save(stitched, out_path, args)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description="Action-conditioned inference from precomputed LIBERO/DROID chunks")
    p.add_argument("--base-checkpoint", default=DEFAULT_BASE)
    p.add_argument("--text-encoder-path", default=None, help="Gemma dir; only needed for guidance_scale>1 (CFG)")
    p.add_argument("--lora-path", required=True)
    p.add_argument("--action-projector", default=None, help="Defaults to action_projector_step_<same>.safetensors")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--chunk-id", default=None, help="File stem under latents/videos/ (without .pt)")
    p.add_argument("--index", type=int, default=0, help="Chunk index if --chunk-id not given")
    p.add_argument("--indices", default=None, help="Comma-separated chunk indices; renders all in ONE model load.")
    p.add_argument("--episode", default=None,
                   help="Episode stem (e.g. '<TASK>_demo_demo_0'); stitches ALL its chunks into one full-episode video.")
    p.add_argument("--src-root", default=DEFAULT_SRC_ROOT, help="LIBERO demo dir (to read episode length T for overlap trim).")
    p.add_argument("--output", default=None, help="Output mp4 (single-chunk mode). Use --out-dir for multi-chunk.")
    p.add_argument("--out-dir", default=None, help="Output dir for multi-chunk / both-condition renders.")
    p.add_argument("--tag", default="eval", help="Filename prefix for multi-render outputs.")
    p.add_argument("--num-inference-steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=1.0, help=">1 needs --text-encoder-path for the negative")
    p.add_argument("--negative-prompt", default="worst quality, inconsistent motion, blurry, jittery, distorted")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-first-frame", action="store_true", help="Do NOT condition on the first target frame")
    p.add_argument("--use-prope", action="store_true",
                   help="Load a PRoPE checkpoint: swap attn1 to PRoPE attention and feed per-token cameras.")
    p.add_argument("--prope-proj-dim", type=int, default=64, help="Must match training (prope_proj_dim).")
    p.add_argument("--prope-image-width", type=int, default=256, help="Per-tile pixel width K is calibrated at.")
    p.add_argument("--prope-image-height", type=int, default=256, help="Per-tile pixel height K is calibrated at.")
    p.add_argument("--no-actions", action="store_true", help="Ablate the action stream (sanity check)")
    p.add_argument("--also-no-actions", action="store_true", help="Additionally render the action-ablated rollout per chunk")
    p.add_argument("--include-gt", action="store_true", help="Stack GT target under the prediction")
    p.add_argument("--include-context", action="store_true", help="Stack the context tile under the prediction")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16
    data_root = Path(args.data_root)

    # ---- Resolve the chunk list ----
    all_ids = _list_chunk_ids(data_root)
    if not all_ids:
        raise SystemExit(f"No chunks under {data_root}/latents/videos/")

    episode_mode = args.episode is not None
    episode_stems = [s for s in (args.episode.split(",") if episode_mode else []) if s.strip()]
    if not episode_mode:
        if args.chunk_id is not None:
            chunk_ids = [args.chunk_id]
        elif args.indices is not None:
            idxs = [int(x) for x in args.indices.split(",") if x.strip() != ""]
            chunk_ids = [all_ids[i] for i in idxs]
        else:
            chunk_ids = [all_ids[args.index]]
        multi = (args.indices is not None) or args.also_no_actions
        if not multi and args.output is None and args.out_dir is None:
            raise SystemExit("Pass --output (single chunk) or --out-dir (multi-chunk / --also-no-actions).")
        print(f"Rendering {len(chunk_ids)} chunk(s); also_no_actions={args.also_no_actions}")
    else:
        if args.out_dir is None:
            raise SystemExit("--episode requires --out-dir.")
        print(f"Rendering {len(episode_stems)} full episode(s); also_no_actions={args.also_no_actions}")

    use_cfg = args.guidance_scale != 1.0
    if use_cfg and args.text_encoder_path is None:
        raise SystemExit("guidance_scale>1 requires --text-encoder-path to encode the negative prompt")

    # ---- Load models ONCE (no VAE encoder; text encoder only if CFG) ----
    components = load_model(
        checkpoint_path=args.base_checkpoint,
        device="cpu",
        dtype=dtype,
        with_video_vae_encoder=False,
        with_video_vae_decoder=True,
        with_audio_vae_decoder=False,
        with_vocoder=False,
        with_text_encoder=use_cfg,
        text_encoder_path=args.text_encoder_path if use_cfg else None,
    )
    # PRoPE must be enabled BEFORE the LoRA wrap so adapters land on the swapped attn1
    # modules — exactly the order the trainer uses (enable_prope_on_model then get_peft_model).
    if args.use_prope:
        n_swapped = enable_prope_on_model(
            components.transformer,
            proj_dim=args.prope_proj_dim,
            image_width=args.prope_image_width,
            image_height=args.prope_image_height,
        )
        print(f"PRoPE enabled: swapped {n_swapped} attn1 modules "
              f"(proj_dim={args.prope_proj_dim}, img={args.prope_image_width}x{args.prope_image_height})")
    transformer = load_lora_weights(components.transformer, args.lora_path)
    transformer = transformer.to(device=device, dtype=dtype).eval()
    embeddings_processor = load_embeddings_processor(args.base_checkpoint, device=device, dtype=dtype)

    proj_path = Path(args.action_projector) if args.action_projector else _default_action_projector_path(Path(args.lora_path))
    action_projector = None
    need_actions = (not args.no_actions) or args.also_no_actions
    if need_actions:
        if proj_path is None:
            raise SystemExit(
                "No action projector found next to the LoRA. The checkpoint predates action-projector "
                "persistence (see trainer.py fix). Pass --action-projector explicitly, or --no-actions to ablate."
            )
        action_projector = load_action_projector(proj_path, dtype).to(device)

    patchifier = VideoLatentPatchifier(patch_size=1)
    out_dir = Path(args.out_dir) if args.out_dir else None

    # one chunk-render closure over the loaded model (action_projector gated per call)
    def render_fn(cid: str, use_actions: bool) -> dict:
        return _render_chunk(
            components=components,
            transformer=transformer,
            embeddings_processor=embeddings_processor,
            action_projector=action_projector if use_actions else None,
            patchifier=patchifier,
            data_root=data_root,
            chunk_id=cid,
            args=args,
            use_cfg=use_cfg,
        )

    # ---- Episode mode: stitch every chunk of each episode into one full-episode video ----
    if episode_mode:
        src_root = Path(args.src_root)
        conds = [True] if args.no_actions is False else [False]
        if args.also_no_actions and True in conds:
            conds.append(False)
        for stem in episode_stems:
            for ua in conds:
                # Use the FULL stem (not stem[:55]) so episodes that share a long
                # task-name prefix but differ only in suffix (GT '..._demo_0' vs
                # play '..._play_reach_demo_0' / '_play_perturb_' / '_play_random_')
                # do not collide to one filename. Stems (~110 chars) stay well under
                # the 255-byte filename limit. eval_prope_metrics splits on '_EP_'.
                op = out_dir / f"{args.tag}_{'act' if ua else 'noact'}_EP_{stem}.mp4"
                print(f"[episode] {stem[:70]}  actions={'on' if ua else 'OFF'}")
                _render_episode(
                    episode_stem=stem, all_ids=all_ids, use_actions=ua,
                    out_path=op, src_root=src_root, render_fn=render_fn, args=args,
                )
        return

    # ---- Single / multi-chunk mode ----
    jobs: list[tuple[str, bool, Path]] = []
    for cid in chunk_ids:
        primary_actions = not args.no_actions
        if out_dir is not None:
            op = out_dir / f"{args.tag}_{'act' if primary_actions else 'noact'}_{cid[:55]}.mp4"
        else:
            op = Path(args.output)
        jobs.append((cid, primary_actions, op))
        if args.also_no_actions and primary_actions:
            if out_dir is not None:
                op2 = out_dir / f"{args.tag}_noact_{cid[:55]}.mp4"
            else:
                op2 = Path(args.output).with_name(Path(args.output).stem + "_noact.mp4")
            jobs.append((cid, False, op2))

    for i, (cid, use_actions, op) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] chunk={cid[:70]}  actions={'on' if use_actions else 'OFF'}")
        panels = render_fn(cid, use_actions)
        _stack_and_save(panels, op, args)


if __name__ == "__main__":
    main()
