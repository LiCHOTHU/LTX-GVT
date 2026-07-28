"""Integration check: the Attention layer's PRoPE path + vanilla fallback (CPU).

Verifies that `Attention(use_prope=True)` runs end-to-end (to_q/k/v -> reshape ->
PropeAttention -> to_out) with per-token cameras, and that it falls back to the
vanilla RoPE path when viewmats/positions are absent. Run:
    python packages/ltx-core/tests/test_prope_attention_integration.py
"""

import torch

from ltx_core.model.transformer.attention import Attention
from ltx_core.model.transformer.prope import _invert_SE3
from ltx_core.model.transformer.rope import LTXRopeType, precompute_freqs_cis

torch.manual_seed(0)
B, T, HEADS, DH = 2, 10, 4, 32
QDIM = HEADS * DH
PROJ = 16
MAXPOS = [20, 2048, 2048]


def _positions():
    p = torch.rand(B, 3, T, 1) * torch.tensor(MAXPOS, dtype=torch.float32).view(1, 3, 1, 1)
    return p.expand(B, 3, T, 2).contiguous()


def _viewmats():
    A = torch.randn(B * T, 3, 3)
    Q, _ = torch.linalg.qr(A)
    Q[:, :, 0] *= torch.sign(torch.linalg.det(Q)).unsqueeze(-1)
    vm = torch.eye(4).repeat(B * T, 1, 1)
    vm[:, :3, :3] = Q
    vm[:, :3, 3] = 0.2 * torch.randn(B * T, 3)
    return vm.reshape(B, T, 4, 4)


def test_attention_prope_forward():
    attn = Attention(
        query_dim=QDIM, heads=HEADS, dim_head=DH, use_prope=True,
        prope_proj_dim=PROJ, prope_image_width=256, prope_image_height=256,
        prope_max_pos=MAXPOS,
    )
    x = torch.randn(B, T, QDIM)
    Ks = torch.eye(3).repeat(B, T, 1, 1)
    Ks[..., 0, 0] = 200.0
    Ks[..., 1, 1] = 200.0
    Ks[..., 0, 2] = 128.0
    Ks[..., 1, 2] = 128.0
    out = attn(x, viewmats=_viewmats(), Ks=Ks, positions=_positions())
    assert out.shape == (B, T, QDIM), out.shape
    assert torch.isfinite(out).all()
    print("[OK] Attention(use_prope=True) forward with per-token cameras ->", tuple(out.shape))


def test_attention_prope_fallback_when_no_cameras():
    """use_prope=True but viewmats=None -> must run the vanilla full-head RoPE path
    (numerically identical to use_prope=False), so flipping the flag is safe before
    cameras flow."""
    common = dict(query_dim=QDIM, heads=HEADS, dim_head=DH)
    a_prope = Attention(
        use_prope=True, prope_proj_dim=PROJ, prope_image_width=256,
        prope_image_height=256, prope_max_pos=MAXPOS, **common,
    )
    a_vanilla = Attention(use_prope=False, **common)
    a_vanilla.load_state_dict(a_prope.state_dict(), strict=False)  # share weights

    x = torch.randn(B, T, QDIM)
    pe = precompute_freqs_cis(_positions(), dim=QDIM, out_dtype=torch.float32,
                              max_pos=MAXPOS, num_attention_heads=HEADS, rope_type=LTXRopeType.SPLIT)
    out_p = a_prope(x, pe=pe, viewmats=None)  # no cameras -> fallback
    out_v = a_vanilla(x, pe=pe)
    err = (out_p - out_v).abs().max().item()
    assert err < 1e-6, f"fallback path diverges from vanilla (err={err})"
    print(f"[OK] use_prope=True + no cameras == vanilla RoPE (max err {err:.2e})")


if __name__ == "__main__":
    test_attention_prope_forward()
    test_attention_prope_fallback_when_no_cameras()
    print("\nAttention PRoPE integration checks passed.")
