"""Post-hoc PRoPE enablement on a loaded `LTXModel`.

PRoPE adds **zero learnable parameters** — the `PropeAttention` module is pure
geometry (3D RoPE coeffs + projective camera matrices, both computed at forward
time). So we can swap each transformer block's `attn1` module for a PRoPE-enabled
version **after** the stock checkpoint has been loaded, with identical state-dict
keys.

This keeps the model-builder / configurator stack untouched: the trainer flips
`use_prope: true` in its config, and after `load_transformer()` we call
`enable_prope_on_model(...)` to upgrade every block's video self-attention. The
RoPE hyper-parameters (max_pos / theta / rope_type / grid flags) are read off the
model's own `video_args_preprocessor` so the PRoPE 3D-RoPE block matches the
model's native positional-encoding conventions exactly.

When `use_prope=True` but no `viewmats` (or no `positions`) are supplied at forward
time, the `Attention` layer falls back to the vanilla full-head RoPE path — so this
swap is safe even before the dataloader carries cameras.
"""

from __future__ import annotations

import torch

from ltx_core.model.transformer.attention import Attention
from ltx_core.model.transformer.transformer_args import (
    MultiModalTransformerArgsPreprocessor,
    TransformerArgsPreprocessor,
)


def _simple_preprocessor(model: torch.nn.Module) -> TransformerArgsPreprocessor:
    """Return the plain TransformerArgsPreprocessor that holds the RoPE hyper-params."""
    pre = getattr(model, "video_args_preprocessor", None)
    if pre is None:
        raise AttributeError("enable_prope_on_model expects model.video_args_preprocessor")
    if isinstance(pre, MultiModalTransformerArgsPreprocessor):
        return pre.simple_preprocessor
    if isinstance(pre, TransformerArgsPreprocessor):
        return pre
    raise TypeError(f"unexpected video_args_preprocessor type: {type(pre)}")


def enable_prope_on_model(
    model: torch.nn.Module,
    *,
    proj_dim: int,
    image_width: int,
    image_height: int,
) -> int:
    """Replace each transformer block's `attn1` (video self-attention) with a
    PRoPE-enabled `Attention` carrying byte-identical weights.

    Args:
        model: An `LTXModel` instance (or any module exposing `.transformer_blocks`
            and `.video_args_preprocessor`).
        proj_dim: Number of per-head channels reserved for the projective camera
            block (multiple of 4). The remaining `head_dim - proj_dim` channels keep
            the native 3D RoPE.
        image_width: Per-tile pixel width (used to normalise K).
        image_height: Per-tile pixel height.

    Returns:
        Number of `attn1` modules that were swapped.
    """
    if not hasattr(model, "transformer_blocks"):
        raise AttributeError("enable_prope_on_model expects a model with .transformer_blocks")

    pre = _simple_preprocessor(model)
    max_pos = list(pre.max_pos)
    theta = pre.positional_embedding_theta
    use_middle = pre.use_middle_indices_grid
    double_prec = pre.double_precision_rope

    swapped = 0
    for block in model.transformer_blocks:
        old: Attention = block.attn1
        new = Attention(
            query_dim=old.to_q.in_features,
            context_dim=old.to_k.in_features,
            heads=old.heads,
            dim_head=old.dim_head,
            rope_type=old.rope_type,
            attention_function=old.attention_function,
            apply_gated_attention=old.to_gate_logits is not None,
            use_prope=True,
            prope_proj_dim=proj_dim,
            prope_image_width=image_width,
            prope_image_height=image_height,
            prope_max_pos=max_pos,
            prope_theta=theta,
            prope_use_middle_indices_grid=use_middle,
            prope_double_precision_rope=double_prec,
        )
        # `prope` is a parameter-free module, so `old.state_dict()` only contains keys
        # for to_q/to_k/to_v/to_out/q_norm/k_norm (+ optional gate). `strict=False` lets
        # the new module accept those exact keys; nothing PRoPE-specific is persisted.
        missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
        unexpected = [k for k in unexpected if not k.startswith("prope.")]
        if unexpected:
            raise RuntimeError(f"Unexpected keys when swapping attn1: {unexpected}")
        # Move to the same device/dtype as the original to avoid mixed-device errors.
        ref_param = next(old.parameters())
        new = new.to(device=ref_param.device, dtype=ref_param.dtype)
        block.attn1 = new
        swapped += 1
    return swapped
