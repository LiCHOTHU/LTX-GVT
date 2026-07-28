"""Isolated memory feasibility smoke test for Perceptual Flow Matching (PFM).

The ONLY open risk in adding PFM to the 22B GVT trainer is memory: PFM must decode
the predicted clean latent x_hat0 back to pixels through the (frozen) VAE decoder,
inside the autograd graph, and run a perceptual net (DINOv2) on the result — every
step. This script measures the *incremental* cost of exactly that op, in isolation
from the transformer, so we know the per-frame decode+backprop budget on an H200
BEFORE wiring it into trainer.py.

It does NOT test correctness of the loss — only whether decode(x_hat0)->DINOv2->backward
fits, and how peak memory scales with the number of decoded latent frames and with
gradient checkpointing on the decode.
"""

import os
import torch

# --- offline hub: use the pre-staged torch-hub cache (dinov2 weights + repo present) ---
os.environ.setdefault("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from ltx_trainer.model_loader import load_video_vae_decoder  # noqa: E402
from ltx_core.types import VideoLatentShape  # noqa: E402
from ltx_core.components.patchifiers import VideoLatentPatchifier  # noqa: E402  (unused but sanity import)

LTX_CKPT = "/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
PRECOMP = "/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/libero90_combined_precomputed/precomputed"
HUB_DINO = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")

DEV = "cuda"
DT = torch.bfloat16


def load_one_latent() -> torch.Tensor:
    """Return one target latent as [1, 128, F, H, W] on GPU (bf16)."""
    f = next(e.path for e in os.scandir(os.path.join(PRECOMP, "latents", "videos")) if e.name.endswith(".pt"))
    d = torch.load(f, map_location="cpu", weights_only=False)
    lat = d["latents"]  # [128, F, H, W]
    print(f"[data] {os.path.basename(f)}  latents={tuple(lat.shape)} {lat.dtype}")
    return lat.unsqueeze(0).to(DEV, DT)


def load_dino():
    """DINOv2 ViT-S/14 from the local hub cache, weights from the cached .pth (offline)."""
    m = torch.hub.load(HUB_DINO, "dinov2_vits14", source="local", pretrained=True)
    return m.to(DEV, DT).eval().requires_grad_(False)


# ImageNet norm for DINOv2 input (expects ~224, /14 patches)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def dino_feats(dino, pix: torch.Tensor) -> torch.Tensor:
    """pix: [F,3,H,W] in [0,1] -> DINOv2 CLS features [F, D]. Kept in the graph."""
    x = torch.nn.functional.interpolate(pix, size=(224, 224), mode="bilinear", align_corners=False)
    x = (x - _MEAN.to(x)) / _STD.to(x)
    return dino(x)  # [F, D]


def run(decoder, dino, x0: torch.Tensor, n_lat_frames: int, ckpt: bool) -> tuple[float, str]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # leaf we differentiate wrt (stands in for the transformer's x_hat0 prediction)
    x = x0[:, :, :n_lat_frames].clone().detach().requires_grad_(True)
    try:
        if ckpt:
            pix = torch.utils.checkpoint.checkpoint(decoder, x, use_reentrant=False)
        else:
            pix = decoder(x)  # [1,3,Fp,Hp,Wp]
        pix = pix.float().clamp(0, 1) if pix.dtype != torch.float32 else pix.clamp(0, 1)
        frames = pix[0].permute(1, 0, 2, 3)  # [Fp,3,Hp,Wp]
        feats = dino_feats(dino, frames.to(DT))
        loss = feats.float().pow(2).mean()  # dummy target=0; only exercises the graph
        loss.backward()
        peak = torch.cuda.max_memory_allocated() / 1e9
        return peak, f"pix={tuple(pix.shape)} feats={tuple(feats.shape)} grad_ok={x.grad is not None}"
    except RuntimeError as e:
        return -1.0, f"FAIL {type(e).__name__}: {str(e)[:160]}"


def main():
    print(f"[gpu] {torch.cuda.get_device_name(0)}  total={torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
    print("[load] VAE decoder from LTX-2.3 checkpoint (this reads the 46GB safetensors)...")
    decoder = load_video_vae_decoder(LTX_CKPT, device=DEV, dtype=DT).eval().requires_grad_(False)
    print("[load] DINOv2 ViT-S/14 (offline hub cache)...")
    dino = load_dino()
    x0 = load_one_latent()
    F = x0.shape[2]
    print(f"\n{'n_lat':>5} {'pix_frames':>10} {'ckpt':>5} {'peak_GB':>8}  detail")
    for n in sorted({1, 2, min(3, F), F}):
        for ckpt in (False, True):
            peak, detail = run(decoder, dino, x0, n, ckpt)
            pf = 8 * (n - 1) + 1
            print(f"{n:>5} {pf:>10} {str(ckpt):>5} {peak:>8.2f}  {detail}")
    print("\n[done] Compare peak_GB against H200 80GB minus the transformer training footprint.")


if __name__ == "__main__":
    main()
