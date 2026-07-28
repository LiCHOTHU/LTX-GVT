"""Video-to-video training strategy for IC-LoRA.
This strategy implements training with reference video conditioning where:
- Reference latents (clean) are concatenated with target latents (noised)
- Video coordinates handle both reference and target sequences
- Loss is computed only on the target portion
"""

from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from ltx_core.model.transformer.modality import Modality
from ltx_trainer import logger
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)


class VideoToVideoConfig(TrainingStrategyConfigBase):
    """Configuration for video-to-video (IC-LoRA) training strategy."""

    name: Literal["video_to_video"] = "video_to_video"

    first_frame_conditioning_p: float = Field(
        default=0.1,
        description="Probability of conditioning on the first frame during training",
        ge=0.0,
        le=1.0,
    )

    reference_latents_dir: str = Field(
        default="reference_latents",
        description="Directory name for latents of reference videos",
    )


class VideoToVideoStrategy(TrainingStrategy):
    """Video-to-video training strategy for IC-LoRA.
    This strategy implements training with reference video conditioning where:
    - Reference latents (clean) are concatenated with target latents (noised)
    - Video coordinates handle both reference and target sequences
    - Loss is computed only on the target portion
    Attributes:
        reference_downscale_factor: The inferred downscale factor of reference videos.
            This is computed from the first batch and cached for metadata export.
    """

    config: VideoToVideoConfig
    reference_downscale_factor: int | None

    def __init__(self, config: VideoToVideoConfig):
        """Initialize strategy with configuration.
        Args:
            config: Video-to-video configuration
        """
        super().__init__(config)
        self.reference_downscale_factor = None  # Will be inferred from first batch

    def get_data_sources(self) -> dict[str, str]:
        """IC-LoRA training requires latents, conditions, and reference latents.

        When acceleration.use_action_cond is enabled, also pull an ``actions/<id>.pt``
        file per sample. Each file holds a tensor of shape (F_pixel, action_dim).
        """
        sources = {
            "latents": "latents",
            "conditions": "conditions",
            self.config.reference_latents_dir: "ref_latents",
        }
        accel = getattr(self, "_acceleration_config", None)
        if accel is not None and getattr(accel, "use_action_cond", False):
            sources["actions"] = "actions"
        # PRoPE needs per-sample cameras. Each ``cameras/<id>.pt`` holds
        # {"viewmats": (V, F, 4, 4) world-to-camera, "Ks": (V, 3, 3)} for the V views in
        # canonical order [agentview, wrist] (matching encode's _hconcat2: left tile =
        # agentview, right tile = wrist). viewmats are per-frame (the full trajectory);
        # Ks are time-invariant, at per-tile pixel resolution (prope_image_width × height).
        if accel is not None and getattr(accel, "use_prope", False):
            sources["cameras"] = "cameras"
        return sources

    def prepare_training_inputs(  # noqa: PLR0915
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare inputs for IC-LoRA training with reference videos."""
        # Get pre-encoded latents - dataset provides uniform non-patchified format [B, C, F, H, W]
        latents = batch["latents"]
        target_latents = latents["latents"]
        ref_latents = batch["ref_latents"]["latents"]

        # Get dimensions
        num_frames = latents["num_frames"][0].item()
        height = latents["height"][0].item()
        width = latents["width"][0].item()

        ref_latents_info = batch["ref_latents"]
        ref_frames = ref_latents_info["num_frames"][0].item()
        ref_height = ref_latents_info["height"][0].item()
        ref_width = ref_latents_info["width"][0].item()

        # Infer reference downscale factor from dimension ratios
        # This allows training with downscaled reference videos for efficiency
        reference_downscale_factor = self._infer_reference_downscale_factor(
            target_height=height,
            target_width=width,
            ref_height=ref_height,
            ref_width=ref_width,
        )

        # Cache the scale factor for metadata export (only on first batch)
        if self.reference_downscale_factor is None:
            self.reference_downscale_factor = reference_downscale_factor
        elif self.reference_downscale_factor != reference_downscale_factor:
            raise ValueError(
                f"Inconsistent reference downscale factor across batches. "
                f"First batch had factor={self.reference_downscale_factor}, "
                f"but current batch has factor={reference_downscale_factor}. "
                f"All training samples must use the same reference/target resolution ratio."
            )

        # Optional MosaicMem-style warp-latent on the reference latents.
        # `_acceleration_config` is attached by the trainer after strategy construction.
        # With identity cameras + flat depth the warp is the identity transform — used
        # for end-to-end wiring verification before real cameras/depth are plumbed in.
        accel = getattr(self, "_acceleration_config", None)
        if accel is not None and getattr(accel, "use_warp_latent", False):
            from ltx_core.conditioning.warp_latent import warp_latent

            b, _, _, hL, wL = ref_latents.shape
            dev, dt = ref_latents.device, ref_latents.dtype
            eye_K = torch.eye(3, device=dev, dtype=dt).unsqueeze(0).expand(b, 3, 3).contiguous()
            eye_T = torch.eye(4, device=dev, dtype=dt).unsqueeze(0).expand(b, 4, 4).contiguous()
            flat_depth = torch.ones(b, hL, wL, device=dev, dtype=dt)
            ref_latents = warp_latent(
                ref_latents, src_K=eye_K, src_T=eye_T, tgt_K=eye_K, tgt_T=eye_T, depth=flat_depth
            )

        # Capture target latent grid (F, H, W) before patchify so a pixel-space loss
        # (Perceptual Flow Matching) can unpatchify + decode the target tokens.
        _, _, _fL, _hL, _wL = target_latents.shape
        video_latent_grid = (int(_fL), int(_hL), int(_wL))

        # Patchify latents: [B, C, F, H, W] -> [B, seq_len, C]
        target_latents = self._video_patchifier.patchify(target_latents)
        ref_latents = self._video_patchifier.patchify(ref_latents)

        # Handle FPS
        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values found in the batch. Found: {fps.tolist()}, using the first one: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        # Get text embeddings (already processed by embedding connectors in trainer)
        # Video-to-video uses only video embeddings
        conditions = batch["conditions"]
        prompt_embeds = conditions["video_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size = target_latents.shape[0]
        ref_seq_len = ref_latents.shape[1]
        target_seq_len = target_latents.shape[1]
        device = target_latents.device
        dtype = target_latents.dtype

        # Create conditioning mask
        # Reference tokens are always conditioning (timestep=0)
        ref_conditioning_mask = torch.ones(batch_size, ref_seq_len, dtype=torch.bool, device=device)

        # Target tokens: check for first frame conditioning
        target_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=target_seq_len,
            height=height,
            width=width,
            device=device,
            first_frame_conditioning_p=self.config.first_frame_conditioning_p,
        )

        # Combined conditioning mask
        conditioning_mask = torch.cat([ref_conditioning_mask, target_conditioning_mask], dim=1)

        # Sample noise and sigmas for target
        sigmas = timestep_sampler.sample_for(target_latents)
        noise = torch.randn_like(target_latents)
        sigmas_expanded = sigmas.view(-1, 1, 1)

        # Apply noise to target
        noisy_target = (1 - sigmas_expanded) * target_latents + sigmas_expanded * noise

        # For first frame conditioning in target, use clean latents
        target_conditioning_mask_expanded = target_conditioning_mask.unsqueeze(-1)
        noisy_target = torch.where(target_conditioning_mask_expanded, target_latents, noisy_target)

        # Targets for loss computation
        targets = noise - target_latents

        # Concatenate reference (clean) and target (noisy)
        combined_latents = torch.cat([ref_latents, noisy_target], dim=1)

        # Create per-token timesteps
        timesteps = self._create_per_token_timesteps(conditioning_mask, sigmas.squeeze())

        # Generate positions for reference and target separately, then concatenate
        ref_positions = self._get_video_positions(
            num_frames=ref_frames,
            height=ref_height,
            width=ref_width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

        # Scale reference positions to match target coordinate space
        # This maps ref positions from (0, ref_H, ref_W) to (0, target_H, target_W)
        # Position tensor shape: [B, 3, seq_len, 2] where dim 1 is (time, height, width)
        if reference_downscale_factor != 1:
            ref_positions = ref_positions.clone()
            ref_positions[:, 1, ...] *= reference_downscale_factor  # height axis
            ref_positions[:, 2, ...] *= reference_downscale_factor  # width axis
            # Time axis (index 0) remains unchanged

        target_positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

        # Concatenate positions along sequence dimension
        positions = torch.cat([ref_positions, target_positions], dim=2)

        # Action-cond Stream 2: project the per-frame action vector and append the
        # resulting tokens onto combined_latents (MosaicMem-style concat). The projector
        # is owned by the trainer and attached to the strategy at init time.
        action_seq_len = 0
        if accel is not None and getattr(accel, "use_action_cond", False) and getattr(self, "_action_projector", None) is not None:
            actions_raw = batch["actions"]["latents"] if isinstance(batch["actions"], dict) else batch["actions"]
            actions_raw = actions_raw.to(device=device, dtype=dtype)
            if actions_raw.dim() != 3:
                raise ValueError(f"actions must be (B, F_pixel, action_dim), got {tuple(actions_raw.shape)}")
            # Resample from pixel-frame rate to latent-frame rate (one token per latent frame).
            # LTX's temporal compression is 8x with the (F % 8 == 1) frame constraint,
            # so F_latent = (F_pixel - 1) // 8 + 1.  Use uniform-stride decimation: indices
            # [0, scale, 2*scale, ...].
            f_pixel = actions_raw.shape[1]
            f_latent = num_frames  # 'num_frames' here is already the latent count
            # Map latent frame index -> source pixel index, evenly distributed across the clip.
            if f_pixel >= f_latent:
                src_idx = torch.linspace(0, f_pixel - 1, steps=f_latent, device=device).round().long()
                actions_resampled = actions_raw[:, src_idx, :]
            else:
                # Source already shorter than latent — repeat last sample
                actions_resampled = torch.nn.functional.interpolate(
                    actions_raw.transpose(1, 2), size=f_latent, mode="nearest"
                ).transpose(1, 2)
            action_tokens = self._action_projector(actions_resampled)  # (B, F_latent, hidden_dim)
            action_seq_len = action_tokens.shape[1]
            # Append onto the sequence.
            combined_latents = torch.cat([combined_latents, action_tokens], dim=1)
            # Per-token timesteps: clean (0) for action tokens.
            action_timesteps = torch.zeros(batch_size, action_seq_len, device=device, dtype=timesteps.dtype)
            timesteps = torch.cat([timesteps, action_timesteps], dim=1)
            # Positions: time = latent-frame index (in seconds-ish units, but the model
            # uses these as RoPE keys not physical time); spatial = a fixed negative sentinel
            # to stay positionally out-of-band vs real video patches (see action_cond.py).
            t_idx = torch.arange(action_seq_len, device=device, dtype=positions.dtype)
            action_time_pos = torch.stack([t_idx, t_idx + 1.0], dim=-1)
            sentinel_pos = torch.full((action_seq_len, 2), fill_value=-1.0, device=device, dtype=positions.dtype)
            action_positions = torch.stack([action_time_pos, sentinel_pos, sentinel_pos], dim=0)
            action_positions = action_positions.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
            positions = torch.cat([positions, action_positions], dim=2)

        # PRoPE per-token cameras. Expand the per-view cameras (from the "cameras" data
        # source) to one (viewmat, K) per token of `combined_latents`, mapping each video
        # token to its tile via `positions`. Returns (None, None) when use_prope is off or
        # the dataset carries no cameras — the Attention layer then auto-falls back to the
        # vanilla full-head RoPE path, so use_prope=True stays numerically safe.
        prope_viewmats, prope_Ks = self._build_per_token_cameras(
            batch=batch,
            positions=positions,
            latent_height=height,
            latent_width=width,
            num_frames=num_frames,
            ref_seq_len=ref_seq_len,
            target_seq_len=target_seq_len,
            action_seq_len=action_seq_len,
            device=device,
            dtype=dtype,
        )

        # Create video Modality
        video_modality = Modality(
            enabled=True,
            latent=combined_latents,
            sigma=sigmas,
            timesteps=timesteps,
            positions=positions,
            context=prompt_embeds,
            context_mask=prompt_attention_mask,
            viewmats=prope_viewmats,
            Ks=prope_Ks,
        )

        # Loss mask: only compute loss on non-conditioning target tokens
        # Reference tokens: all False (no loss)
        # Target tokens: True where not conditioning
        # Action tokens: all False (clean conditioning, never penalised)
        ref_loss_mask = torch.zeros(batch_size, ref_seq_len, dtype=torch.bool, device=device)
        target_loss_mask = ~target_conditioning_mask
        action_loss_mask = torch.zeros(batch_size, action_seq_len, dtype=torch.bool, device=device)
        video_loss_mask = torch.cat([ref_loss_mask, target_loss_mask, action_loss_mask], dim=1)

        return ModelInputs(
            video=video_modality,
            audio=None,
            video_targets=targets,
            audio_targets=None,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=None,
            ref_seq_len=ref_seq_len,
            video_latent_grid=video_latent_grid,
        )

    # Latent -> pixel spatial compression (see SpatioTemporalScaleFactors.default()).
    _SPATIAL_SCALE = 32
    # Canonical view order for the 2-view LIBERO layout, matching encode's
    # _hconcat2(agentview, wrist): left tile = agentview (0), right tile = wrist (1).
    _N_VIEWS = 2

    def _build_per_token_cameras(
        self,
        *,
        batch: dict[str, Any],
        positions: Tensor,
        latent_height: int,
        latent_width: int,
        num_frames: int,
        ref_seq_len: int,
        target_seq_len: int,
        action_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Expand per-frame, per-view cameras to one (viewmat, K) per token of ``combined_latents``.

        Two alignment axes per video token:
          * VIEW (which camera) from the width pixel coordinate in ``positions``: the
            canvas is agentview|wrist left|right (encode's _hconcat2), so a token whose
            width-midpoint is in the left half -> agentview (0), right half -> wrist (1).
            This holds for BOTH the reference and target blocks (same L|R layout).
          * FRAME (which pose along the trajectory) from the token's latent-frame index,
            recovered by patch ordering ``(f, h, w)`` within each block. The stored
            per-frame poses (V, F_src, 4, 4) are decimated to the ``num_frames`` latent
            frames with the SAME ``linspace(0, F_src-1, num_frames).round()`` used to
            resample the action tokens, so camera / action / video stay on one grid.

        Appended action tokens (and any token we can't confidently place) get an identity
        camera (``P = I``) so the projective block is a no-op there. Returns ``(None, None)``
        when PRoPE is off or the batch carries no ``cameras`` source (-> vanilla RoPE
        fallback in attention, so use_prope=True is safe before cameras flow).

        Args:
            positions: (B, 3, T, 2) coordinate grid for the *full* sequence (ref+target+action).
            latent_height/latent_width: target latent dims (the view split is at full_w/2).
            num_frames: latent frame count (== target latent frames).
            ref_seq_len/target_seq_len: token counts of the reference / target blocks.
            action_seq_len: number of action-cond tokens appended at the end of the sequence.
        """
        accel = getattr(self, "_acceleration_config", None)
        if accel is None or not getattr(accel, "use_prope", False):
            return None, None
        cams = batch.get("cameras") if isinstance(batch, dict) else None
        if cams is None:
            return None, None
        if not (isinstance(cams, dict) and "viewmats" in cams and "Ks" in cams):
            raise ValueError("cameras source must be a dict with 'viewmats' (B,V,F,4,4) and 'Ks' (B,V,3,3)")

        viewmats_src = cams["viewmats"].to(device=device, dtype=dtype)  # (B, V, F_src, 4, 4)
        Ks_src = cams["Ks"].to(device=device, dtype=dtype)  # (B, V, 3, 3)
        # Accept a static per-view schema (B, V, 4, 4) too -> treat as a single frame.
        if viewmats_src.dim() == 4 and tuple(viewmats_src.shape[-2:]) == (4, 4):
            viewmats_src = viewmats_src.unsqueeze(2)  # (B, V, 1, 4, 4)
        if viewmats_src.dim() != 5 or tuple(viewmats_src.shape[-2:]) != (4, 4):
            raise ValueError(f"cameras.viewmats must be (B, V, F, 4, 4), got {tuple(viewmats_src.shape)}")
        if Ks_src.dim() != 4 or tuple(Ks_src.shape[-2:]) != (3, 3):
            raise ValueError(f"cameras.Ks must be (B, V, 3, 3), got {tuple(Ks_src.shape)}")
        B, V, F_src = viewmats_src.shape[:3]
        if V < self._N_VIEWS:
            raise ValueError(f"PRoPE expects >= {self._N_VIEWS} views [agentview, wrist], got V={V}")
        T = positions.shape[2]

        # Decimate the stored per-frame poses to the latent-frame count, EXACTLY as the
        # action tokens are resampled (linspace endpoints, rounded) -> shared temporal grid.
        if F_src >= num_frames:
            cam_idx = torch.linspace(0, F_src - 1, steps=num_frames, device=device).round().long()
        else:
            cam_idx = torch.arange(num_frames, device=device).clamp(max=F_src - 1)
        viewmats_lat = viewmats_src[:, :, cam_idx]  # (B, V, num_frames, 4, 4)

        # VIEW per token from the width pixel midpoint (left half -> agentview, right -> wrist).
        n_video = T - action_seq_len
        w_mid = (positions[:, 2, :, 0] + positions[:, 2, :, 1]) * 0.5  # (B, T)
        half_w = latent_width * self._SPATIAL_SCALE * 0.5
        left = w_mid < half_w
        is_video = torch.zeros(B, T, dtype=torch.bool, device=device)
        is_video[:, :n_video] = True
        view_idx = torch.full((B, T), -1, dtype=torch.long, device=device)
        view_idx[is_video & left] = 0  # agentview (left tile)
        view_idx[is_video & ~left] = 1  # wrist (right tile)

        # FRAME per token from patch ordering (f, h, w) within each block. A block whose
        # token count isn't divisible by num_frames is left as identity (view -1) rather
        # than risk a misaligned pose.
        frame_idx = torch.zeros(B, T, dtype=torch.long, device=device)
        tok = torch.arange(T, device=device)
        if ref_seq_len > 0:
            if ref_seq_len % num_frames == 0:
                hw_ref = ref_seq_len // num_frames
                m = tok < ref_seq_len
                frame_idx[:, m] = (tok[m] // hw_ref)
            else:
                view_idx[:, :ref_seq_len] = -1
        tgt_end = ref_seq_len + target_seq_len
        if target_seq_len > 0:
            if target_seq_len % num_frames == 0:
                hw_tgt = target_seq_len // num_frames
                m = (tok >= ref_seq_len) & (tok < tgt_end)
                frame_idx[:, m] = ((tok[m] - ref_seq_len) // hw_tgt)
            else:
                view_idx[:, ref_seq_len:tgt_end] = -1
        frame_idx = frame_idx.clamp_(0, num_frames - 1)

        # Identity camera for -1 tokens: viewmat = I and a K that normalises to identity,
        # so P = lift(K_norm) @ I = I. K_norm = I  <=>  K = diag(IW, IH, 1) with cx=IW/2, cy=IH/2.
        iw = float(accel.prope_image_width)
        ih = float(accel.prope_image_height)
        eye_vm = torch.eye(4, device=device, dtype=dtype)
        ident_k = torch.tensor(
            [[iw, 0.0, 0.5 * iw], [0.0, ih, 0.5 * ih], [0.0, 0.0, 1.0]], device=device, dtype=dtype
        )
        viewmats = eye_vm.view(1, 1, 4, 4).expand(B, T, 4, 4).clone()
        Ks = ident_k.view(1, 1, 3, 3).expand(B, T, 3, 3).clone()

        valid = view_idx >= 0  # (B, T)
        if bool(valid.any()):
            b_idx, t_idx = valid.nonzero(as_tuple=True)
            v_sel = view_idx[b_idx, t_idx]
            f_sel = frame_idx[b_idx, t_idx]
            viewmats[b_idx, t_idx] = viewmats_lat[b_idx, v_sel, f_sel]
            Ks[b_idx, t_idx] = Ks_src[b_idx, v_sel]
        return viewmats, Ks

    def compute_loss(
        self,
        video_pred: Tensor,
        _audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute masked loss only on target portion. Returns [B,]."""
        # Extract target portion of prediction. Slice by the target tensor's seq_len
        # so any tokens appended *after* the target (e.g. action-cond tokens) are
        # excluded from the loss without breaking on a shape mismatch.
        ref_seq_len = inputs.ref_seq_len
        target_seq_len = inputs.video_targets.shape[1]
        target_pred = video_pred[:, ref_seq_len:ref_seq_len + target_seq_len, :]

        # Get target portion of loss mask (same slice)
        target_loss_mask = inputs.video_loss_mask[:, ref_seq_len:ref_seq_len + target_seq_len]

        # Compute per-element loss [B,]
        loss = (target_pred - inputs.video_targets).pow(2)
        loss_mask = target_loss_mask.unsqueeze(-1).float()
        masked = loss.mul(loss_mask)
        return masked.mean(dim=[-2, -1]) / loss_mask.mean(dim=[-2, -1]).clamp(min=1e-8)

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        """Get metadata for checkpoint files."""
        metadata: dict[str, Any] = {}
        # Always include reference_downscale_factor for IC-LoRAs so inference
        # pipelines know the expected scale factor for reference videos.
        if self.reference_downscale_factor is not None:
            metadata["reference_downscale_factor"] = self.reference_downscale_factor
        return metadata

    @staticmethod
    def _infer_reference_downscale_factor(
        target_height: int,
        target_width: int,
        ref_height: int,
        ref_width: int,
    ) -> int:
        """Infer the reference downscale factor from target and reference dimensions."""
        # If dimensions match, no scaling needed
        if target_height == ref_height and target_width == ref_width:
            return 1

        # Calculate scale factors for each dimension
        if target_height % ref_height != 0 or target_width % ref_width != 0:
            raise ValueError(
                f"Target dimensions ({target_height}x{target_width}) must be exact multiples "
                f"of reference dimensions ({ref_height}x{ref_width})"
            )

        scale_h = target_height // ref_height
        scale_w = target_width // ref_width

        if scale_h != scale_w:
            raise ValueError(
                f"Reference scale must be uniform. Got height scale {scale_h} and width scale {scale_w}. "
                f"Target: {target_height}x{target_width}, Reference: {ref_height}x{ref_width}"
            )

        if scale_h < 1:
            raise ValueError(
                f"Reference dimensions ({ref_height}x{ref_width}) cannot be larger than "
                f"target dimensions ({target_height}x{target_width})"
            )

        return scale_h
