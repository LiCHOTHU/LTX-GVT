"""CPU verification for the 3D-RoPE ⊕ projective PRoPE kernel (no GPU needed).

Run: python packages/ltx-core/tests/test_prope_kernel.py
(or via pytest). Each check asserts a mathematical property the kernel must obey.
"""

import torch

from ltx_core.model.transformer.prope import PropeAttention, _invert_SE3
from ltx_core.model.transformer.rope import LTXRopeType, apply_rotary_emb

torch.manual_seed(0)

B, H, T, D = 2, 4, 12, 32  # small head_dim=32 for speed
PROJ = 16  # -> d_rope = 16
MAXPOS = [20, 2048, 2048]


def _module():
    return PropeAttention(
        head_dim=D, proj_dim=PROJ, num_heads=H, max_pos=MAXPOS,
        rope_type=LTXRopeType.SPLIT, image_width=256, image_height=256,
    ).double()


def _rand_qkv():
    return (torch.randn(B, H, T, D, dtype=torch.float64) for _ in range(3))


def _rand_positions(seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    # (B, 3, T, 2): axes = (time, height, width); [...,0]=[...,1] (no middle-grid)
    p = torch.rand(B, 3, T, 1, dtype=torch.float64) * torch.tensor(MAXPOS, dtype=torch.float64).view(1, 3, 1, 1)
    return p.expand(B, 3, T, 2).contiguous()


def _rand_se3(n):
    """Random batched SE(3) (n, 4, 4)."""
    A = torch.randn(n, 3, 3, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A)
    # ensure det +1
    Q[:, :, 0] *= torch.sign(torch.linalg.det(Q)).unsqueeze(-1)
    out = torch.eye(4, dtype=torch.float64).repeat(n, 1, 1)
    out[:, :3, :3] = Q
    out[:, :3, 3] = 0.2 * torch.randn(n, 3, dtype=torch.float64)
    return out


def _rand_intrinsics(n):
    K = torch.eye(3, dtype=torch.float64).repeat(n, 1, 1)
    K[:, 0, 0] = 200 + 50 * torch.rand(n)  # fx
    K[:, 1, 1] = 200 + 50 * torch.rand(n)  # fy
    K[:, 0, 2] = 128 + 10 * torch.randn(n)  # cx
    K[:, 1, 2] = 128 + 10 * torch.randn(n)  # cy
    return K


def test_shape_preserved():
    m = _module()
    q, k, v = _rand_qkv()
    vm = _rand_se3(B * T).reshape(B, T, 4, 4)
    out = m(q.double(), k.double(), v.double(), viewmats=vm, Ks=None, positions=_rand_positions())
    assert out.shape == (B, H, T, D), out.shape
    print("[OK] shape preserved", tuple(out.shape))


def test_identity_cameras_reduce_to_partial_rope():
    """viewmats = I, Ks = None  ->  P = P_inv = I, so the projective block is a no-op.
    Output must equal a plain attention where RoPE is applied to the first d_rope dims
    of q/k only (and nothing to the projective dims)."""
    m = _module()
    q, k, v = (t.double() for t in _rand_qkv())
    pos = _rand_positions(seed=1)
    eye = torch.eye(4, dtype=torch.float64).expand(B, T, 4, 4).contiguous()
    out = m(q, k, v, viewmats=eye, Ks=None, positions=pos)

    # Reference: rope on [:d_rope], identity on the rest.
    dr = m.d_rope
    cos, sin = m._rope_coeffs(pos, torch.float64)
    qr = torch.cat([apply_rotary_emb(q[..., :dr], (cos, sin), m.rope_type), q[..., dr:]], dim=-1)
    kr = torch.cat([apply_rotary_emb(k[..., :dr], (cos, sin), m.rope_type), k[..., dr:]], dim=-1)
    ref = torch.nn.functional.scaled_dot_product_attention(qr, kr, v)
    err = (out - ref).abs().max().item()
    assert err < 1e-9, err
    print(f"[OK] identity cameras reduce to partial-RoPE attention (max err {err:.2e})")


def test_temporal_rope_is_active():
    """The RELATIVE time structure of the tokens must affect the output — this is the
    whole point of the fix vs. the old 2D (x,y-only) kernel, which had no time axis.
    (A *uniform* time shift correctly cancels by RoPE's relative-position property, so we
    compare distinct per-token times against a flat/constant time axis instead.)"""
    m = _module()
    q, k, v = (t.double() for t in _rand_qkv())
    vm = _rand_se3(B * T).reshape(B, T, 4, 4)

    pos_ramp = _rand_positions(seed=2)
    pos_ramp[:, 0, :, :] = torch.arange(T, dtype=torch.float64).view(1, T, 1)  # distinct times
    out_ramp = m(q, k, v, viewmats=vm, Ks=None, positions=pos_ramp)

    pos_flat = pos_ramp.clone()
    pos_flat[:, 0, :, :] = 0.0  # all tokens share the same time (no temporal info)
    out_flat = m(q, k, v, viewmats=vm, Ks=None, positions=pos_flat)
    diff = (out_ramp - out_flat).abs().max().item()
    assert diff > 1e-4, f"relative time had no effect (diff={diff}) — RoPE is not 3D!"
    print(f"[OK] relative temporal structure affects output (max diff {diff:.3e}) -> 3D RoPE is live")


def test_gauge_invariance_no_intrinsics():
    """Applying a global world-frame change G to every camera must leave the FULL
    output unchanged (the output P-transform undoes the gauge)."""
    m = _module()
    q, k, v = (t.double() for t in _rand_qkv())
    pos = _rand_positions(seed=3)
    vm = _rand_se3(B * T).reshape(B, T, 4, 4)
    out0 = m(q, k, v, viewmats=vm, Ks=None, positions=pos)

    G = _rand_se3(1)  # single global SE(3), (1, 4, 4)
    Ginv = _invert_SE3(G)  # analytic SE(3) inverse (avoids generic-inverse roundoff)
    vm_g = vm @ Ginv.view(1, 1, 4, 4)
    out1 = m(q, k, v, viewmats=vm_g, Ks=None, positions=pos)
    rel = (out0 - out1).abs().max().item() / out0.abs().max().item()
    assert rel < 1e-6, f"not gauge invariant (rel err={rel})"
    print(f"[OK] gauge invariant under global SE(3), no intrinsics (rel err {rel:.2e})")


def test_gauge_invariance_with_intrinsics():
    m = _module()
    q, k, v = (t.double() for t in _rand_qkv())
    pos = _rand_positions(seed=4)
    vm = _rand_se3(B * T).reshape(B, T, 4, 4)
    Ks = _rand_intrinsics(B * T).reshape(B, T, 3, 3)
    out0 = m(q, k, v, viewmats=vm, Ks=Ks, positions=pos)

    G = _rand_se3(1)
    Ginv = _invert_SE3(G)
    vm_g = vm @ Ginv.view(1, 1, 4, 4)
    out1 = m(q, k, v, viewmats=vm_g, Ks=Ks, positions=pos)
    rel = (out0 - out1).abs().max().item() / out0.abs().max().item()
    assert rel < 1e-5, f"not gauge invariant with intrinsics (rel err={rel})"
    print(f"[OK] gauge invariant under global SE(3), with intrinsics (rel err {rel:.2e})")


if __name__ == "__main__":
    test_shape_preserved()
    test_identity_cameras_reduce_to_partial_rope()
    test_temporal_rope_is_active()
    test_gauge_invariance_no_intrinsics()
    test_gauge_invariance_with_intrinsics()
    print("\nAll PRoPE kernel checks passed.")
