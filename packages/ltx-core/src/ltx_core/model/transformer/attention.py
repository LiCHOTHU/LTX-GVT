from enum import Enum
from typing import Protocol

import torch

from ltx_core.model.transformer.prope import PropeAttention
from ltx_core.model.transformer.rope import LTXRopeType, apply_rotary_emb

memory_efficient_attention = None
flash_attn_interface = None
try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    memory_efficient_attention = None
try:
    # FlashAttention3 and XFormersAttention cannot be used together
    if memory_efficient_attention is None:
        import flash_attn_interface
except ImportError:
    flash_attn_interface = None


class AttentionCallable(Protocol):
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor: ...


class PytorchAttention(AttentionCallable):
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))

        if mask is not None:
            # add a batch dimension if there isn't already one
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a heads dimension if there isn't already one
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out


class XFormersAttention(AttentionCallable):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory_efficient_attention is None:
            raise RuntimeError("XFormersAttention was selected but `xformers` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        # xformers expects [B, M, H, K]
        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            # add a singleton batch dimension
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a singleton heads dimension
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            # pad to a multiple of 8
            pad = 8 - mask.shape[-1] % 8
            # the xformers docs says that it's allowed to have a mask of shape (1, Nq, Nk)
            # but when using separated heads, the shape has to be (B, H, Nq, Nk)
            # in flux, this matrix ends up being over 1GB
            # here, we create a mask with the same batch/head size as the input mask (potentially singleton or full)
            mask_out = torch.empty(
                [mask.shape[0], mask.shape[1], q.shape[1], mask.shape[-1] + pad], dtype=q.dtype, device=q.device
            )

            mask_out[..., : mask.shape[-1]] = mask
            # doesn't this remove the padding again??
            mask = mask_out[..., : mask.shape[-1]]
            mask = mask.expand(b, heads, -1, -1)

        out = memory_efficient_attention(q.to(v.dtype), k.to(v.dtype), v, attn_bias=mask, p=0.0)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class FlashAttention3(AttentionCallable):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if flash_attn_interface is None:
            raise RuntimeError("FlashAttention3 was selected but `FlashAttention3` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            raise NotImplementedError("Mask is not supported for FlashAttention3")

        out = flash_attn_interface.flash_attn_func(q.to(v.dtype), k.to(v.dtype), v)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class AttentionFunction(Enum):
    PYTORCH = "pytorch"
    XFORMERS = "xformers"
    FLASH_ATTENTION_3 = "flash_attention_3"
    DEFAULT = "default"

    def to_callable(self) -> AttentionCallable:
        """Resolve to a concrete callable. Use this at module init time so that
        torch.compile can trace through the attention call without graph breaks."""
        if self is AttentionFunction.PYTORCH:
            return PytorchAttention()
        elif self is AttentionFunction.XFORMERS:
            return XFormersAttention()
        elif self is AttentionFunction.FLASH_ATTENTION_3:
            return FlashAttention3()
        else:
            # Default behavior: XFormers if installed else - PyTorch
            return XFormersAttention() if memory_efficient_attention is not None else PytorchAttention()


class Attention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        attention_function: AttentionCallable | AttentionFunction = AttentionFunction.DEFAULT,
        apply_gated_attention: bool = False,
        use_prope: bool = False,
        prope_proj_dim: int | None = None,
        prope_image_width: int | None = None,
        prope_image_height: int | None = None,
        prope_max_pos: list[int] | None = None,
        prope_theta: float = 10000.0,
        prope_use_middle_indices_grid: bool = False,
        prope_double_precision_rope: bool = False,
    ) -> None:
        super().__init__()
        self.rope_type = rope_type
        self.attention_function = (
            attention_function.to_callable()
            if isinstance(attention_function, AttentionFunction)
            else attention_function
        )

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head

        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)

        # Optional per-head gating
        if apply_gated_attention:
            self.to_gate_logits = torch.nn.Linear(query_dim, heads, bias=True)
        else:
            self.to_gate_logits = None

        self.to_out = torch.nn.Sequential(torch.nn.Linear(inner_dim, query_dim, bias=True), torch.nn.Identity())

        # PRoPE — opt-in camera-relative positional encoding for GVT multi-view video.
        # When use_prope=True AND per-token cameras are supplied at forward time, the
        # head is split block-diagonally: the first `dim_head - prope_proj_dim` dims get
        # the native 3D RoPE (time/height/width, preserving temporal position), and the
        # last `prope_proj_dim` dims get the per-token projective camera transform. See
        # prope.PropeAttention. When viewmats is None at forward time the layer falls back
        # to the vanilla full-head RoPE path, so use_prope=True is safe before cameras flow.
        self.use_prope = use_prope
        if use_prope:
            for name, val in {
                "prope_proj_dim": prope_proj_dim,
                "prope_image_width": prope_image_width,
                "prope_image_height": prope_image_height,
            }.items():
                if val is None:
                    raise ValueError(f"{name} is required when use_prope=True")
            self.prope = PropeAttention(
                head_dim=dim_head,
                proj_dim=prope_proj_dim,
                num_heads=heads,
                max_pos=prope_max_pos if prope_max_pos is not None else [20, 2048, 2048],
                positional_embedding_theta=prope_theta,
                rope_type=rope_type,
                use_middle_indices_grid=prope_use_middle_indices_grid,
                double_precision_rope=prope_double_precision_rope,
                image_width=prope_image_width,
                image_height=prope_image_height,
            )
        else:
            self.prope = None

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
        viewmats: torch.Tensor | None = None,
        Ks: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Multi-head attention with optional RoPE, perturbation masking, and per-head gating.
        When ``perturbation_mask`` is all zeros, the expensive query/key path
        (linear projections, RMSNorm, RoPE) is skipped entirely and only the
        value projection is used as a pass-through.
        Args:
            x: Query input tensor of shape ``(B, T, query_dim)``.
            context: Key/value context tensor of shape ``(B, S, context_dim)``.
                Falls back to ``x`` (self-attention) when *None*.
            mask: Optional attention mask. Interpretation depends on the attention
                backend (additive bias for xformers/PyTorch SDPA).
            pe: Rotary positional embeddings applied to both ``q`` and ``k``.
            k_pe: Separate rotary positional embeddings for ``k`` only. When
                *None*, ``pe`` is reused for keys.
            perturbation_mask: Optional mask in ``[0, 1]`` that
                blends the attention output with the raw value projection:
                ``out = attn_out * mask + v * (1 - mask)``.
                **1** keeps the full attention output, **0** bypasses attention
                and passes the value projection through unchanged.
                *None* or all-ones means standard attention; all-zeros skips
                the query/key path entirely for efficiency.
            all_perturbed: Whether all perturbations are active for this block.
        Returns:
            Output tensor of shape ``(B, T, query_dim)``.
        """
        context = x if context is None else context
        use_attention = not all_perturbed

        v = self.to_v(context)

        if not use_attention:
            out = v
        else:
            q = self.to_q(x)
            k = self.to_k(context)

            q = self.q_norm(q)
            k = self.k_norm(k)

            # PRoPE path: split the head into a 3D-RoPE block + a per-token projective
            # camera block (see prope.PropeAttention). Requires per-token cameras AND the
            # raw positions grid; falls back to vanilla full-head RoPE when either is absent
            # (so use_prope=True is a no-op until the dataloader carries cameras).
            if self.use_prope and viewmats is not None and positions is not None:
                b, t, _ = q.shape
                # Reshape (B, T, H*D) -> (B, H, T, D) for PRoPE.
                q4 = q.view(b, t, self.heads, self.dim_head).transpose(1, 2)
                k4 = k.view(b, t, self.heads, self.dim_head).transpose(1, 2)
                v4 = v.view(b, t, self.heads, self.dim_head).transpose(1, 2)
                attn_mask_4d = mask
                if attn_mask_4d is not None:
                    if attn_mask_4d.ndim == 2:
                        attn_mask_4d = attn_mask_4d.unsqueeze(0)
                    if attn_mask_4d.ndim == 3:
                        attn_mask_4d = attn_mask_4d.unsqueeze(1)
                o4 = self.prope(
                    q4, k4, v4, viewmats=viewmats, Ks=Ks, positions=positions, attn_mask=attn_mask_4d
                )
                out = o4.transpose(1, 2).reshape(b, t, self.heads * self.dim_head)
            else:
                if pe is not None:
                    q = apply_rotary_emb(q, pe, self.rope_type)
                    k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)

                out = self.attention_function(q, k, v, self.heads, mask)  # (B, T, H*D)

            if perturbation_mask is not None:
                out = out * perturbation_mask + v * (1 - perturbation_mask)

        # Apply per-head gating if enabled
        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)  # (B, T, H)
            b, t, _ = out.shape
            # Reshape to (B, T, H, D) for per-head gating
            out = out.view(b, t, self.heads, self.dim_head)
            # Apply gating: 2 * sigmoid(x) so that zero-init gives identity (2 * 0.5 = 1.0)
            gates = 2.0 * torch.sigmoid(gate_logits)  # (B, T, H)
            out = out * gates.unsqueeze(-1)  # (B, T, H, D) * (B, T, H, 1)
            # Reshape back to (B, T, H*D)
            out = out.view(b, t, self.heads * self.dim_head)

        return self.to_out(out)
