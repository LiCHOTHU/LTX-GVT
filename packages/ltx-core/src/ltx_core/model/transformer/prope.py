# MIT License
#
# Copyright (c) Authors of
# "Cameras as Relative Positional Encoding" https://arxiv.org/pdf/2507.10496
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""PRoPE for **video**: 3D-RoPE  ⊕  per-token projective camera encoding.

Why this differs from the reference PRoPE port
-----------------------------------------------
The original "Cameras as Relative Positional Encoding" kernel was written for
multi-view *images*: it assumes ``seqlen == cameras * patches_x * patches_y`` and
splits ``head_dim`` as ``[projective | x-RoPE | y-RoPE]`` — a purely 2D spatial
RoPE with **no temporal axis**. Dropping that kernel in as a replacement for
LTX's native **3D** RoPE (time, height, width) therefore *throws away temporal
positional information* — fatal for a video model, and it cannot be reshaped to
``cameras * px * py`` once reference/target/action tokens are concatenated.

This module fixes both problems by splitting ``head_dim`` into two contiguous,
block-diagonal sub-spaces (they must be disjoint — RoPE rotations and the
projective transform each only preserve their own relative-pose identity, so
mixing them on shared coordinates corrupts both):

    head_dim = [ 0 : d_rope )  -> native LTX 3D RoPE  (time, height, width)
               [ d_rope : D )  -> projective camera transform  (per token)

* The RoPE block uses the *exact* LTX split-RoPE machinery (``precompute_freqs_cis``
  / ``apply_rotary_emb``), just sized to ``d_rope`` instead of the full head, so
  temporal + spatial position is preserved.
* The projective block is the GTA/PRoPE "cameras as relative PE" trick: each
  token carries its own camera ``(viewmat, K)``, the query gets ``Pᵀ``, the
  key/value get ``P⁻¹``, the output gets ``P``. Then for a query token in camera
  ``a`` and key token in camera ``b`` the logit contribution is
  ``qᵀ (P_a P_b⁻¹) k`` with ``P_a P_b⁻¹ = image_a ← image_b`` — i.e. it depends
  only on the *relative* camera geometry, and is invariant to a global world-frame
  change. Tokens with no real camera (action tokens, blank tiles, padding) pass
  an identity ``viewmat`` (+ identity ``K``) so ``P = I`` and the projective block
  degrades gracefully to a plain dot product on those dims.

Cameras are supplied **per token** as ``viewmats (B, T, 4, 4)`` (world-to-camera)
and optional ``Ks (B, T, 3, 3)`` (pinhole intrinsics). This is layout-agnostic:
the caller maps each token (reference / target / action; any tile; any frame) to
its camera however it likes — typically by reusing the same ``positions`` grid the
RoPE block consumes. When ``viewmats is None`` the module is never invoked
(``Attention`` falls back to the vanilla RoPE path).
"""

from __future__ import annotations

import torch

from ltx_core.model.transformer.rope import (
    LTXRopeType,
    apply_rotary_emb,
    generate_freq_grid_np,
    generate_freq_grid_pytorch,
    precompute_freqs_cis,
)


class PropeAttention(torch.nn.Module):
    """3D-RoPE ⊕ per-token projective attention transforms (zero learnable params).

    Construction fixes the head geometry + the LTX RoPE hyper-parameters (so the
    ``d_rope`` block matches the model's native RoPE conventions). ``forward`` takes
    ``q/k/v`` already shaped ``(B, num_heads, T, head_dim)`` plus per-token cameras
    and the raw ``positions`` grid, and returns the attended output in the same
    shape — the caller handles the ``to_q/to_k/to_v`` projections and ``to_out``.
    """

    def __init__(
        self,
        head_dim: int,
        proj_dim: int,
        num_heads: int,
        *,
        max_pos: list[int],
        positional_embedding_theta: float = 10000.0,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        use_middle_indices_grid: bool = False,
        double_precision_rope: bool = False,
        image_width: int = 1,
        image_height: int = 1,
    ) -> None:
        super().__init__()
        if proj_dim % 4 != 0:
            raise ValueError(f"proj_dim must be a multiple of 4 (homogeneous 4-vectors), got {proj_dim}")
        if proj_dim < 4 or proj_dim >= head_dim:
            raise ValueError(f"proj_dim must satisfy 4 <= proj_dim < head_dim ({head_dim}), got {proj_dim}")
        d_rope = head_dim - proj_dim
        if d_rope % 2 != 0:
            raise ValueError(f"d_rope (= head_dim - proj_dim = {d_rope}) must be even for split-RoPE")

        self.head_dim = head_dim
        self.proj_dim = proj_dim
        self.d_rope = d_rope
        self.num_heads = num_heads
        self.max_pos = list(max_pos)
        self.theta = positional_embedding_theta
        self.rope_type = rope_type
        self.use_middle_indices_grid = use_middle_indices_grid
        self.double_precision_rope = double_precision_rope
        self.image_width = image_width
        self.image_height = image_height

    # ---- RoPE block -------------------------------------------------------
    def _rope_coeffs(self, positions: torch.Tensor, out_dtype: torch.dtype):
        """Build LTX split-RoPE (cos, sin) for the ``d_rope`` sub-block.

        ``positions`` is the model's coordinate grid ``(B, n_axes, T, 2)`` — the
        same tensor the full-head RoPE consumes — so the temporal + spatial axes
        carried here are identical to the native path, only the per-head width is
        ``d_rope`` instead of ``head_dim``.
        """
        freq_grid_generator = generate_freq_grid_np if self.double_precision_rope else generate_freq_grid_pytorch
        return precompute_freqs_cis(
            positions,
            dim=self.num_heads * self.d_rope,
            out_dtype=out_dtype,
            theta=self.theta,
            max_pos=self.max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.num_heads,
            rope_type=self.rope_type,
            freq_grid_generator=freq_grid_generator,
        )

    # ---- Projective block -------------------------------------------------
    def _camera_projections(
        self, viewmats: torch.Tensor, Ks: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-token ``P`` (image←world) and ``P_inv`` (world←image), each ``(B, T, 4, 4)``.

        With intrinsics: ``P = lift(K_norm) @ viewmat``. Without: the GTA variant
        ``P = viewmat`` (camera←world). ``viewmat`` is world-to-camera.
        """
        if Ks is not None:
            B, T = Ks.shape[:2]
            Ks_norm = torch.zeros_like(Ks)
            Ks_norm[..., 0, 0] = Ks[..., 0, 0] / self.image_width
            Ks_norm[..., 1, 1] = Ks[..., 1, 1] / self.image_height
            Ks_norm[..., 0, 2] = Ks[..., 0, 2] / self.image_width - 0.5
            Ks_norm[..., 1, 2] = Ks[..., 1, 2] / self.image_height - 0.5
            Ks_norm[..., 2, 2] = 1.0
            P = _lift_K(Ks_norm) @ viewmats
            P_inv = _invert_SE3(viewmats) @ _lift_K(_invert_K(Ks_norm))
        else:
            P = viewmats
            P_inv = _invert_SE3(viewmats)
        return P, P_inv

    def forward(
        self,
        q: torch.Tensor,  # (B, num_heads, T, head_dim)
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        viewmats: torch.Tensor,  # (B, T, 4, 4) world-to-camera, per token
        Ks: torch.Tensor | None,  # (B, T, 3, 3) intrinsics, per token (or None)
        positions: torch.Tensor,  # (B, n_axes, T, 2) RoPE coordinate grid
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, H, T, D = q.shape
        if D != self.head_dim:
            raise ValueError(f"head_dim mismatch: q has {D}, module configured for {self.head_dim}")
        if viewmats.shape != (B, T, 4, 4):
            raise ValueError(f"viewmats must be (B, T, 4, 4) = {(B, T, 4, 4)}, got {tuple(viewmats.shape)}")
        if Ks is not None and Ks.shape != (B, T, 3, 3):
            raise ValueError(f"Ks must be (B, T, 3, 3) = {(B, T, 3, 3)}, got {tuple(Ks.shape)}")

        dr = self.d_rope
        # --- split into the two block-diagonal sub-spaces ---
        q_rope, q_proj = q[..., :dr], q[..., dr:]
        k_rope, k_proj = k[..., :dr], k[..., dr:]
        v_rope, v_proj = v[..., :dr], v[..., dr:]

        # --- RoPE block: native LTX 3D RoPE on q/k (v is never roped) ---
        cos, sin = self._rope_coeffs(positions, q.dtype)
        q_rope = apply_rotary_emb(q_rope, (cos, sin), self.rope_type)
        k_rope = apply_rotary_emb(k_rope, (cos, sin), self.rope_type)

        # --- Projective block: per-token camera transforms ---
        # Compute matrices in fp32 for numerical stability, then cast to q dtype.
        P, P_inv = self._camera_projections(viewmats.float(), None if Ks is None else Ks.float())
        P = P.to(q.dtype)
        P_inv = P_inv.to(q.dtype)
        q_proj = _apply_projmat(q_proj, P.transpose(-1, -2))  # query  <- Pᵀ
        k_proj = _apply_projmat(k_proj, P_inv)  # key    <- P⁻¹
        v_proj = _apply_projmat(v_proj, P_inv)  # value  <- P⁻¹

        q2 = torch.cat([q_rope, q_proj], dim=-1)
        k2 = torch.cat([k_rope, k_proj], dim=-1)
        v2 = torch.cat([v_rope, v_proj], dim=-1)

        out = torch.nn.functional.scaled_dot_product_attention(
            q2, k2, v2, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )

        # --- undo the projective frame on the output block ---
        o_rope, o_proj = out[..., :dr], out[..., dr:]
        o_proj = _apply_projmat(o_proj, P)  # output <- P
        return torch.cat([o_rope, o_proj], dim=-1)


# ----------------------------------------------------------------------------
# geometry helpers (per-token batched 4x4 / 3x3)
# ----------------------------------------------------------------------------
def _apply_projmat(feats: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Apply a per-token 4x4 ``matrix`` to ``feats`` treated as groups of homogeneous 4-vectors.

    feats:  (B, num_heads, T, d_proj)  with d_proj % 4 == 0
    matrix: (B, T, 4, 4)               one transform per token
    out_i = sum_j matrix_ij feat_j, applied independently to each 4-vector group.
    """
    B, H, T, d = feats.shape
    f = feats.reshape(B, H, T, d // 4, 4)
    out = torch.einsum("btij,bhtkj->bhtki", matrix, f)
    return out.reshape(B, H, T, d)


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert (..., 4, 4) SE(3) matrices."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    """Lift (..., 3, 3) intrinsics to homogeneous (..., 4, 4)."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device, dtype=Ks.dtype)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    """Invert (..., 3, 3) pinhole intrinsics (no skew)."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out
