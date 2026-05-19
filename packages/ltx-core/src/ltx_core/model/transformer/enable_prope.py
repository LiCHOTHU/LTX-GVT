"""Post-hoc PRoPE enablement on a loaded `LTXModel`.

PRoPE adds **zero learnable parameters** — the `PropeDotProductAttention` module
is pure geometry (projection matrices precomputed from per-tile cameras). So we
can swap each transformer block's `attn1` module for a PRoPE-enabled version
**after** the stock checkpoint has been loaded, with identical state-dict keys.

This keeps the model-builder / configurator stack untouched: the trainer flips
`use_prope: true` in its config, and after `load_transformer()` we call
`enable_prope_on_model(...)` to upgrade every block's video self-attention.

When `use_prope=True` but no `viewmats`/`Ks` are supplied at forward time, the
Attention layer falls back to the vanilla RoPE path — so this swap is safe even
before the dataloader has been extended to provide camera matrices.
"""

from __future__ import annotations

import torch

from ltx_core.model.transformer.attention import Attention


def enable_prope_on_model(
    model: torch.nn.Module,
    *,
    patches_x: int,
    patches_y: int,
    image_width: int,
    image_height: int,
) -> int:
    """Replace each transformer block's `attn1` (video self-attention) with a
    PRoPE-enabled `Attention` carrying byte-identical weights.

    Args:
        model: An `LTXModel` instance (or any module exposing `.transformer_blocks`).
        patches_x: Per-tile patch grid width.
        patches_y: Per-tile patch grid height.
        image_width: Per-tile pixel width (used to normalise K).
        image_height: Per-tile pixel height.

    Returns:
        Number of `attn1` modules that were swapped.
    """
    if not hasattr(model, "transformer_blocks"):
        raise AttributeError("enable_prope_on_model expects a model with .transformer_blocks")

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
            prope_patches_x=patches_x,
            prope_patches_y=patches_y,
            prope_image_width=image_width,
            prope_image_height=image_height,
        )
        # `prope` is a buffer-only module (zero learnable params), so `old.state_dict()`
        # only contains keys for to_q/to_k/to_v/to_out/q_norm/k_norm (+ optional gate).
        # `strict=False` lets the new module accept those exact keys without complaining
        # that PRoPE buffers are missing from `old`.
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
