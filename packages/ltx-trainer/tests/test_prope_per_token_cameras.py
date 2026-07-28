"""Verify VideoToVideoStrategy._build_per_token_cameras maps tokens -> tiles -> cameras
correctly (CPU). Run: python packages/ltx-trainer/tests/test_prope_per_token_cameras.py
"""

from types import SimpleNamespace

import torch

from ltx_trainer.training_strategies.video_to_video import VideoToVideoConfig, VideoToVideoStrategy

# 4x4 latent grid (1 frame) -> pixel 128x128, tile split at 64. Per-tile pixel = 64.
LH = LW = 4
SCALE = 32
TILE_PX = (LH * SCALE) // 2  # 64
N_SPATIAL = LH * LW  # 16 video tokens
N_ACTION = 2
T = N_SPATIAL + N_ACTION


def _strategy():
    s = VideoToVideoStrategy(VideoToVideoConfig())
    s._acceleration_config = SimpleNamespace(
        use_prope=True, prope_image_width=TILE_PX, prope_image_height=TILE_PX
    )
    return s


def _positions():
    """(1, 3, T, 2): video tokens in row-major (r, c) order, then action sentinels."""
    pos = torch.zeros(1, 3, T, 2)
    i = 0
    for r in range(LH):
        for c in range(LW):
            hmid = r * SCALE + SCALE / 2  # patch midpoint in pixels
            wmid = c * SCALE + SCALE / 2
            pos[0, 1, i, :] = hmid
            pos[0, 2, i, :] = wmid
            pos[0, 0, i, :] = 0.0
            i += 1
    # action tokens: spatial sentinel -1 (as in prepare_training_inputs)
    pos[0, 1, N_SPATIAL:, :] = -1.0
    pos[0, 2, N_SPATIAL:, :] = -1.0
    return pos


def _cameras():
    """Distinguishable per-view cameras: view i -> translation (i+1)."""
    vm = torch.eye(4).repeat(1, 3, 1, 1)
    for i in range(3):
        vm[0, i, :3, 3] = float(i + 1)
    Ks = torch.eye(3).repeat(1, 3, 1, 1)
    for i in range(3):
        Ks[0, i, 0, 0] = 100.0 + i
        Ks[0, i, 1, 1] = 100.0 + i
    return {"viewmats": vm, "Ks": Ks}


def test_quadrant_to_view_mapping():
    s = _strategy()
    pos = _positions()
    vm, Ks = s._build_per_token_cameras(
        batch={"cameras": _cameras()}, positions=pos, latent_height=LH, latent_width=LW,
        action_seq_len=N_ACTION, device=torch.device("cpu"), dtype=torch.float32,
    )
    assert vm.shape == (1, T, 4, 4) and Ks.shape == (1, T, 3, 3)

    eye = torch.eye(4)
    ident_k = torch.tensor([[TILE_PX, 0, TILE_PX / 2], [0, TILE_PX, TILE_PX / 2], [0, 0, 1.0]])

    i = 0
    counts = {"ext1": 0, "ext2": 0, "wrist": 0, "blank": 0}
    for r in range(LH):
        for c in range(LW):
            top, left = r < LH // 2, c < LW // 2
            if top and left:  # ext1 (view 0)
                assert torch.allclose(vm[0, i, :3, 3], torch.ones(3) * 1.0), (r, c)
                counts["ext1"] += 1
            elif top and not left:  # ext2 (view 1)
                assert torch.allclose(vm[0, i, :3, 3], torch.ones(3) * 2.0), (r, c)
                counts["ext2"] += 1
            elif not top and not left:  # wrist (view 2)
                assert torch.allclose(vm[0, i, :3, 3], torch.ones(3) * 3.0), (r, c)
                counts["wrist"] += 1
            else:  # bottom-left blank -> identity
                assert torch.allclose(vm[0, i], eye), (r, c)
                assert torch.allclose(Ks[0, i], ident_k), (r, c)
                counts["blank"] += 1
            i += 1

    # action tokens -> identity
    for j in range(N_SPATIAL, T):
        assert torch.allclose(vm[0, j], eye), j
        assert torch.allclose(Ks[0, j], ident_k), j

    assert counts == {"ext1": 4, "ext2": 4, "wrist": 4, "blank": 4}, counts
    print(f"[OK] quadrant->view mapping correct {counts}; action+blank -> identity")


def test_returns_none_without_cameras():
    s = _strategy()
    vm, Ks = s._build_per_token_cameras(
        batch={}, positions=_positions(), latent_height=LH, latent_width=LW,
        action_seq_len=N_ACTION, device=torch.device("cpu"), dtype=torch.float32,
    )
    assert vm is None and Ks is None
    print("[OK] no 'cameras' in batch -> (None, None) (vanilla RoPE fallback)")


if __name__ == "__main__":
    test_quadrant_to_view_mapping()
    test_returns_none_without_cameras()
    print("\nPer-token camera builder checks passed.")
