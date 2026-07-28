# GVT — Generative Virtual Teleoperation

A research fork of [Lightricks/LTX-2](./ltx_readme.md) that turns LTX-2 from a text+image-to-video model into an **action-conditioned, multi-view world model** for robotic manipulation. The model takes a *text instruction*, an *initial image*, and a *robot action stream* (TIA2V) and generates the resulting video — so the operator effectively "teleoperates in a generated world."

![3-view action-conditioned prior on a DROID episode](docs/media/three_view_action_cond.png)

<sub>Real DROID episode (CLVR lab). **Rows:** exterior cam 1 / exterior cam 2 / wrist cam. **Columns:** GT video frame | Franka rendered from the action stream only | rendered-on-GT sanity overlay. Middle column = the conditioning channel the model is trained on. <a href="docs/media/three_view_action_cond.mp4">▶ Watch the 12 s clip</a></sub>

> Upstream LTX-2 docs are preserved in [`ltx_readme.md`](./ltx_readme.md). This file only describes the GVT-specific work. Day-to-day experiments, measurements, and dropped directions live in [`DEVLOG.md`](./DEVLOG.md).

## What's new on top of LTX-2

- **TIA2V conditioning** — Text + Initial-image + Action → Video. The action stream is a sequence of Franka commanded joint angles + gripper, drawn from the [DROID](https://droid-dataset.github.io/) dataset.
- **Multi-view output** — two fixed exterior cameras (ZED 2 stereo rig) + one wrist camera (ZED Mini), tiled 2×2 (1024×576, 512×288 per tile; bottom-left is blank). Vanilla spatial attention covers cross-view consistency for free; an opt-in [PRoPE](#multi-view-geometry-optional-prope-attention) flag turns on geometry-aware attention when needed (see [Design tricks](#design-tricks)).
- **A rendered action prior** — every action step is materialised as a per-frame *visual prior*: run forward kinematics on the commanded joints, project the Franka skeleton (Stage 1) or render the textured Franka mesh (Stage 2) into the calibrated camera, and feed that image as an extra conditioning channel. This grounds the model on *where the arm should be* at every timestep, in image space.

## Context generation — uses *only* TIA2V model inputs

The strongest property of this design is that **the action-conditioning context is a deterministic function of inputs the model already receives at inference time** — no peek at the ground-truth video. Concretely:

```
                    ┌──────────────────────────────────────────────────────────┐
TIA2V inputs ─────► │                                                          │
  Text  T           │   Action  ─► FK (Franka Panda, Modified DH) ─► (9, 3)    │
  Image I (frame 0) │   (joints+ │                                  link pts   │
  Action A          │   gripper) │                                  in base    │
                    │            └──► (camera K, extrinsic) ────────────┐      │
                    │                                                   ▼      │
                    │                                          project_points  │
                    │                                                   │      │
                    │                                                   ▼      │
                    │                                       skeleton draw  OR  │
                    │                                       mesh rasterise     │
                    │                                                   │      │
Conditioning ◄──────┤                                          tiled prior     │
channel             │                                          (1×3 strip)     │
                    └──────────────────────────────────────────────────────────┘
```

That is: `context = render(FK(A), K, T_cam_to_base)`. The text `T` and initial image `I` do not enter the prior; only the action `A` and the (known) camera setup do. At training time we pre-render this offline once per episode; at inference time the same function runs on the user-supplied action and target-camera setup. **The GT video frames are never consulted to build the prior** — that is what makes the trained model deployable beyond DROID.

### How it's pre-computed (training data side)

The training-side pipeline lives in `experiments/droid_action_cond/`. It runs in two passes because TensorFlow (which reads DROID's RLDS shards) and pyrender (EGL/CUDA-backed) cannot share a process — they segfault on shutdown.

| Step | Script | What it produces |
|---|---|---|
| 1. Pick an IoU-calibrated DROID episode | `pick_episode.py` | `outputs/episode.npz` with `cmd_joint_position (T,7)`, `cmd_gripper_position (T,)`, `K_rls`, `cam2base`, plus GT frames for later evaluation only |
| 2a. Render skeleton prior (Stage 1) | `build_prior.py` | `outputs/02_context/ctx_NNNN.png` — **the actual model input** |
| 2b. Render mesh prior (Stage 2)     | `build_prior.py` | `outputs/04_context_mesh/` (flat-color mask) and `outputs/06_context_photo/` (textured Franka) |
| 3. Scale across episodes/cameras    | `demo_more_episodes.py`, `headline_diverse.py`, `headline_both_cams.py` | Two-pass extract/render across many labs and camera serials |
| 4. Wrist-camera calibration         | `calibrate_wrist_vggt_v2.py`, `wrist_calib_validate.py`, `wrist_demo.py` | Recover `T_cam_to_hand` for the (uncalibrated) wrist cam, then render wrist views |

The `prior_report.json` per episode records the flange-vs-Cartesian pixel error at `t=0` — a one-number sanity check that FK + extrinsic agree with the recorded EE target. Single-digit pixels is what we expect on a 320×180 RLDS frame.

### How the context is rendered per view

Each view in the 2×2 tile uses a *different* renderer because the fixed exteriors and the wrist camera have fundamentally different relationships to the robot. Both pipelines consume only the TIA2V inputs (action stream + known camera intrinsics/extrinsics + the first GT frame, which is the model's input image `I` anyway).

| View | Camera pose | Renderer | What gets drawn | File |
|---|---|---|---|---|
| ext1, ext2 | **Static** `cam2base` from the April 2025 DROID calibration | `FrankaMeshRenderer(mode="photo")` — pyrender rasterisation of the Franka FER visual meshes | The Franka arm itself (textured mesh) on black background. Mesh pose comes from `fk_urdf(joints[t])` per frame. Gripper finger displacement = `(1 − gripper) × 0.04 m` mirrored | `packages/ltx-action-cond/src/ltx_action_cond/mesh_rendering.py` |
| wrist | **Time-varying** `cam2base_wrist[t] = T_hand_to_base(joints[t]) @ T_cam_to_hand` where `T_cam_to_hand` is the per-episode **VGGT-calibrated** mount from `outputs/wrist_calib_vggt/<ep>.json` (NOT the nominal mount) | Scene plane-warp + splat from GT wrist[0], reprojected through the wrist pose at frame t | A reprojection of the scene that the wrist camera saw at t=0, viewed from where the wrist points at frame t. By construction t=0 matches GT wrist[0] exactly. **We do NOT render the Franka mesh in the wrist view** — the camera is rigidly attached to the gripper, so painting the gripper on itself adds nothing | `packages/ltx-action-cond/src/ltx_action_cond/wrist_render.py`: `reconstruct_scene_from_plane(wristf[0], K_wrist, cam2base_wrist[0], plane_z=0.0)` → `render_wrist_scene_splat(scene, cam2base_wrist[t], K_wrist, …)` |

The wrist recipe is the load-bearing piece: anchoring on GT wrist[0] (rather than ext1[0]) and using the VGGT-calibrated `T_cam_to_hand` (rather than the nominal mount transform) are both required for the rendered view to track the GT wrist video. The reference implementation lives in `experiments/droid_action_cond/wrist_video_from_wrist0.py` and is mirrored exactly in any later demo that visualises the wrist context (e.g. `motion_flow_demo.py`).

> Per-step render-time profiling, the dropped warped-noise direction, the motion-flow scaffold, and the cross-lab sanity-check demos all live in [`DEVLOG.md`](DEVLOG.md).

### Action representation

Per-step action vector: **7-D commanded joints + 1-D commanded gripper** (`action_dict.joint_position`, `action_dict.gripper_position`). The top-level RLDS Cartesian `action` is *not* used to build the prior — it is only used as a calibration sanity check (does FK's flange land where the Cartesian target says?). Gripper convention follows DROID: `0 = open`, `1 = closed`; mesh-rendered finger displacement is `q_finger = (1 − gripper) × 0.04 m`, mirrored per finger.

## Repository layout (GVT additions)

```
packages/ltx-action-cond/         # Reusable library
  src/ltx_action_cond/
    kinematics.py                 # Franka Panda FK (Modified DH, no URDF needed for keypoints)
    calibration.py                # Parse the April 2025 DROID calibration JSONs
    projection.py                 # 3D base-frame points -> image pixels
    rendering.py                  # Stage 1: draw skeleton on a PIL image
    mesh_rendering.py             # Stage 2: pyrender Franka meshes (mask + photoreal modes)
    wrist_render.py               # Synthetic wrist-cam render via plane-warp + FK
    droid.py                      # RLDS episode loading + match to calibration map

experiments/droid_action_cond/    # End-to-end runner scripts (see table above)

docs/media/                       # README assets (still + video)
ltx_readme.md                     # Upstream LTX-2 README
```

Public API:

```python
from ltx_action_cond import (
    fk, fk_urdf,                  # Franka forward kinematics
    project_points,               # 3D base-frame -> 2D pixel
    draw_skeleton,                # Stage 1 overlay
    extrinsic_6dof_to_matrix,
    intrinsics_to_K, rescale_K,   # DROID calibration helpers
)
from ltx_action_cond.mesh_rendering import FrankaMeshRenderer  # Stage 2
```

## Design tricks

Three load-bearing tricks make this stack work:

1. **Joint-space conditioning, not Cartesian.** The prior is rendered from commanded joints, not the top-level Cartesian RLDS action. Joint commands are what the controller actually executes; the Cartesian field is the high-level *target* and is one IK step removed from physical truth. Cartesian survives only as a sanity check.

2. **Action-as-render, not action-as-vector.** Instead of feeding the action as a raw 8-D vector concatenated to the latents (which the model would have to learn to relate to image space), we *render* it: FK → project → image-space skeleton/mesh. The action arrives in the same coordinate system as the image, so the model only has to learn "match the pose hint," not "translate joint angles to pixel locations." This is also why the Stage 2 mesh prior is preferred over the Stage 1 skeleton — it occupies the same image footprint the GT arm does.

3. **Multi-view via tiling, not architecture.** For 3-view (two exteriors + wrist) consistency we tile the views into a **2×2 grid** — `[top-left: cam1 | top-right: cam2 | bottom-left: blank | bottom-right: wrist]` — at 1024×576 total (each tile 512×288, the smallest 16:9 size with each tile divisible by the VAE's ×32 spatial factor). The model's existing spatial self-attention handles inter-view consistency for free; the blank quadrant is supervised to constant black (or loss-masked). The action prior is tiled with the same layout, each tile rendered with its own `K` and extrinsic. Cost: ~25% of tokens spent on the blank quadrant; benefit: square output that matches LTX-2's training distribution and a natural slot for a 4th view (depth, second wrist, …) later. If cross-view geometric drift appears, opt into [PRoPE](#multi-view-geometry-optional-prope-attention) for explicit camera awareness.

## Training: feeding action into LTX-2

Given a preprocessed sample of `(initial_image, prompt, action, context_video)` — i.e. the standard TIA2V input bundle — there are two distinct conditioning streams the GVT fine-tune introduces, and both reach the transformer via **token concatenation** along the sequence axis (MosaicMem-style). No channel concat, no architectural surgery beyond a small projector.

### How LTX-2 already supports token-concat conditioning

LTX-2's transformer treats every input as a single token sequence. The `Modality.latent` tensor is documented as shape `(B, T, D)` where `T` is the **total** number of tokens — *noisy + conditioning*. Each `ConditioningItem.apply_to` appends its own tokens to the end of that sequence, contributing matching `positions` `(B, 3, T_new, 2)` and `denoise_mask` `(B, T_new, 1)` entries, plus a slice of the 2-D self-attention mask `(B, T, T)`. Self-attention then mixes everything in one pass. This is the same pattern IC-LoRA and LipDub already use for their reference-video inputs.

### Stream 1 — Context (rendered) video → existing IC-LoRA path, zero new code

The `context_video` (the tiled 3-view Franka render produced by `build_prior.py`) is image-shaped, so it goes through the **existing** `VideoConditionByReferenceLatent` (`packages/ltx-core/src/ltx_core/conditioning/types/reference_video_cond.py`) — the same conditioning class that powers `ICLoraPipeline`'s pose-control and motion-track-control LoRAs.

```
context_video  ─►  VAE encode  ─►  patchify  ─►  reference tokens
                                                       │
                                                       ▼
                            torch.cat onto the noisy-video latent sequence
                            (with `downscale_factor` and RoPE positions
                             produced by `get_pixel_coords`)
```

Training the model to use this stream is just **training a new IC-LoRA** with the standard upstream trainer — the prior video is the "control" input. No conditioning-code change is needed.

### Stream 2 — Raw action vector → new `VideoConditionByActionTokens`

The raw `(F_pixel, 8)` action stream (7 Franka joints + 1 gripper) has no pixel representation, so it cannot ride the reference-video path. GVT adds a focused new conditioning class at `packages/ltx-core/src/ltx_core/conditioning/types/action_cond.py`:

```python
from ltx_core.conditioning import ActionMLPProjector, VideoConditionByActionTokens

# 1) Resample the action to the latent frame rate (one action sample per latent frame).
#    The dataloader is responsible for this; F_latent matches the target video.
action_resampled = ...  # (B, F_latent, 8)

# 2) Project per-frame action vectors into transformer tokens.
projector = ActionMLPProjector(action_dim=8, hidden_dim=D_HIDDEN)
action_tokens = projector(action_resampled)  # (B, F_latent, D_hidden)

# 3) Add as a conditioning item — appended to the same token sequence as the
#    noisy video latent and the IC-LoRA reference tokens.
conditionings.append(VideoConditionByActionTokens(tokens=action_tokens, strength=1.0))
```

`VideoConditionByActionTokens.apply_to` does three things, matching the contract of every other `ConditioningItem`:

1. **Concatenates** the projected action tokens to `latent_state.latent` and `latent_state.clean_latent`.
2. **Builds RoPE positions** `(B, 3, F_latent, 2)`: the temporal axis matches the video latent's frame timestamps (computed with the same `scale_factors.time / fps` and causal-fix convention used by `get_pixel_coords`); the height/width axes are pinned to a fixed negative sentinel `-1.0` so the action tokens are positionally distinguishable from any real video patch.
3. **Updates `denoise_mask` and `attention_mask`** — `denoise_mask = 1 - strength` (so by default the action tokens are kept clean, never denoised); the self-attention mask is grown to `(B, T+F_latent, T+F_latent)` via the existing `update_attention_mask` helper so all video tokens attend to all action tokens and vice versa.

The small projector (`ActionMLPProjector`: 2-layer SiLU MLP) lives outside the frozen base checkpoint and is trained from scratch — naturally folded into a LoRA fine-tune alongside the attention adapters.

### Putting the two streams together in one training step

For a batch sample `(image_0, prompt, action, context_video)`:

| Stream | Tokens come from | Conditioning class | Status |
|---|---|---|---|
| Text prompt | Gemma text encoder → cross-attention context | existing | unchanged |
| Initial image | First-frame keyframe latent | `VideoConditionByLatentIndex` / `VideoConditionByKeyframeIndex` | unchanged |
| Context (rendered) video | VAE encode + patchify | `VideoConditionByReferenceLatent` (IC-LoRA) | unchanged |
| **Raw action vector** | `ActionMLPProjector(action)` | **`VideoConditionByActionTokens` (new)** | **new in GVT** |

All four streams append tokens to the same sequence; self-attention mixes them in one transformer forward pass. The training loss is computed only on the noisy video token slice (positions `[0:N_noisy]`) — every conditioning item contributes a `denoise_mask` entry of `0` so its tokens are excluded from the loss target.

The smoke test in this commit verifies that, given a video latent of `(B=2, T_video=208, D=128)` and an action of `(2, 13, 8)`, the result has `latent=(2, 221, 128)`, `positions=(2, 3, 221, 2)`, correct denoise mask values for both `strength=1.0` and `strength=0.0`, monotonic temporal positions starting at `0.0`, and spatial sentinel positions at `-1.0`.

## Multi-view geometry: optional PRoPE attention

For the 3-view (cam1 / cam2 / wrist) tiled output, vanilla spatial self-attention has no idea that the four tiles correspond to *different camera frustums* — it sees them as one 2D image. To get geometry-aware cross-view consistency for free, GVT vendors [PRoPE](https://arxiv.org/abs/2507.10496) ("Cameras as Relative Positional Encoding") and makes it an **opt-in flag**. The vanilla LTX-2 training and inference paths are byte-for-byte unchanged when PRoPE is off.

### 3D RoPE ⊕ projective — *not* a replacement of RoPE

The naive port did 2D `(x, y)` spatial RoPE that **replaced** the model's native 3D RoPE — silently discarding the temporal axis. That is wrong: LTX-2's positional encoding (time, height, width) carries real signal we must keep. The correct construction splits each attention head's `head_dim=128` into two **disjoint** sub-spaces and runs them block-diagonally:

```
head_dim = 128
┌─────────────────────────────┬──────────────────────────┐
│  [0 : d_rope)               │  [d_rope : 128)          │
│  native 3D RoPE             │  per-token projective    │
│  (time, height, width)      │  camera transform (PRoPE)│
│  — untouched                │  — zero learnable params │
└─────────────────────────────┴──────────────────────────┘
        d_rope = head_dim - proj_dim
```

Disjoint sub-spaces are mandatory: mixing the projective block into the RoPE block corrupts both relative identities. With `proj_dim = 0` the module is bit-identical to vanilla 3D RoPE.

### Why PRoPE and not RayRoPE

[RayRoPE](https://arxiv.org/abs/2601.15275) is the obvious cousin, but it (a) replaces `scaled_dot_product_attention` itself (breaks FlashAttention compatibility) and (b) introduces a learned per-token depth head (pure cold-start parameters that the LTX-2 checkpoint cannot help). PRoPE is geometry-only — **zero learnable parameters** — and applies as a Q/K/V/O transform around standard SDPA, so we can:
- load the stock LTX-2 checkpoint into the PRoPE-enabled module unchanged;
- keep FlashAttention/xformers kernels;
- adapt the spatial Q/K behaviour with a LoRA during fine-tune instead of a full retrain.

### Files

| Path | What it is |
|---|---|
| `packages/ltx-core/src/ltx_core/model/transformer/prope.py` | `PropeAttention`: builds its own `d_rope`-sized 3D RoPE from `positions`, applies the projective block on `[d_rope:128)`. |
| `packages/ltx-core/src/ltx_core/model/transformer/attention.py` | Guarded PRoPE branch in `Attention.forward` (reshape → `PropeAttention` → `to_out`). |
| `packages/ltx-core/src/ltx_core/model/transformer/enable_prope.py` | `enable_prope_on_model` — post-hoc swap of each block's `attn1` after checkpoint load. |

### Turning it on

`Attention` gains PRoPE constructor flags, all defaulted so existing callsites are unchanged:

```python
Attention(
    query_dim=..., heads=..., dim_head=...,
    use_prope=True,                # default False — toggle PRoPE on
    prope_proj_dim=64,             # projective sub-space width (mult. of 4, < head_dim);
                                   #   the remaining head_dim - proj_dim stays 3D RoPE
    prope_image_width=IMG_W,       # per-tile pixel width  (used to normalise K)
    prope_image_height=IMG_H,      # per-tile pixel height
)
```

At forward time the per-**token** camera tensors and RoPE `positions` are passed in:

```python
attn(x, viewmats=(B, T, 4, 4), Ks=(B, T, 3, 3), positions=(B, 3, T, 2))
```

Note these are **per-token** (one camera per sequence position), not per-view — the dataloader expands the 3 physical views to per-token cameras by tile membership. If `viewmats`/`positions` are `None` (or `use_prope=False`), the call takes the vanilla full-head RoPE path with no overhead, so flipping `--use_prope` is safe before cameras flow.

### Dataloader contract

When `use_prope` is on, `VideoToVideoStrategy.get_data_sources()` adds a `cameras` source. Each sample provides:

```
cameras/<id>.pt = {"viewmats": (V, 4, 4), "Ks": (V, 3, 3)}   # V = 3, order [ext1, ext2, wrist]
```

`Ks` are in per-tile pixel resolution. `_build_per_token_cameras()` then maps every token to its tile by `positions` (TL=ext1, TR=ext2, BR=wrist, BL=blank) and scatters the matching camera; the blank tile and the action tokens get an **identity** camera (`P = I`, a no-op in attention) and are loss-masked downstream.

### Verified properties

CPU property + integration tests (`packages/ltx-core/tests/`, `packages/ltx-trainer/tests/`) confirm:
1. **Identity cameras reduce to partial 3D RoPE** (err 0) — the projective block is an exact no-op when `P = I`.
2. **Temporal structure is live** — distinct per-token times change the output (diff 1.3); a uniform time shift cancels (relative encoding), proving native 3D RoPE survives.
3. **Gauge invariance** — a global world-frame change leaves attention logits unchanged (rel err 6e-8 without intrinsics, 1.3e-7 with).
4. **Fallback** — `use_prope=True` with no cameras is numerically identical to vanilla RoPE (err 0).
5. **Per-token camera builder** — quadrant→view mapping is exact; action + blank tokens → identity; missing `cameras` → `(None, None)`.

### What's left to wire up

The model and dataloader paths are complete end-to-end. The remaining piece is **data generation**: the dataset build (`chunk_context` / `build_dataset_resumable`) does not yet emit `cameras/<id>.pt`. The values are derivable from `K1, K2, K_wrist`, `cam2base_1/2`, `T_cam_to_hand` and per-frame FK — a per-frame wrist camera is a high-value refinement over a static one. Then train a Q/K LoRA to absorb the positional-encoding shift.

## Wrist-camera calibration (separate concern)

DROID does NOT reliably ship a wrist-cam extrinsic. We recover the constant mount `T_cam_to_hand` ourselves:

1. **VGGT relative pose** — feed `(ext1_t0, wrist_t0, wrist_t_far)` to [VGGT](https://github.com/facebookresearch/vggt) and use the FK-known hand motion `|hand[t_far] − hand[0]|` as a metric scale anchor. (Older v1 used `(ext1_t0, wrist_t0)` only and assumed `|t_wrist→ext1| ≈ |t_hand→ext1|`, which breaks when the mount offset is comparable to the camera distance — TRI ended up 19 cm "behind" the hand origin.)
2. **Validate** — render `wrist[0]` from a plane-warp of `ext1[0]` using the recovered `T_cam_to_hand` and visually overlay against GT (`wrist_calib_validate.py`):

   ![wrist calib single episode](docs/media/wrist_calib_vggt.png)
   <sub>CLVR · GT wrist (left) vs wrist rendered from ext1 plane-warp using the VGGT-recovered `T_cam_to_hand` (right). <a href="docs/media/wrist_calib_vggt.mp4">▶ time-evolution clip</a></sub>

   Grid of `wrist_calib_validate.py` outputs across all 5 calibration episodes (one row per lab):

   ![wrist calib grid](docs/media/wrist_calib_grid.png)

3. **Render full wrist videos** for each episode (`wrist_demo.py`, `wrist_video_*.py`).

A fully independent fallback — for cameras that aren't in DROID at all — is the ChArUco generator at `calibration/generate_charuco_5x7_A4.py`: writes a print-ready A4 board (`DICT_5X5_100`, 35 mm squares, 26 mm markers). Print at 100% / "Actual size" and verify a square edge with a ruler before using.

## Environment

The GVT code reuses the upstream LTX-2 environment plus the `droid` extra of `ltx-action-cond`:

```bash
uv sync --frozen
uv pip install -e packages/ltx-action-cond[droid]
```

Hardware: a single RTX 5090 (32 GB) is the assumed budget for both rendering and LTX-2 inference. Pyrender is currently the throughput bottleneck of the prior; the planned acceleration path is Warp/Newton for full-DROID-scale dataset construction.

## Status

- ✅ Stage 1 (skeleton) + Stage 2 (mesh) prior on calibrated DROID-100 exterior episodes
- ✅ Two-camera prior with rig-inferred extrinsic for the uncalibrated exterior
- ✅ Wrist-cam `T_cam_to_hand` via VGGT with FK-anchored scale
- ⏳ Tiled 3-view dataset assembly (GT mp4 + action-prior mp4 sidecar) for LTX-2 fine-tuning
- ⏳ LTX-2 fine-tune with the tiled action-conditioning channel
