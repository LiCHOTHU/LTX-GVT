# GVT — Generative Virtual Teleoperation

A research fork of [Lightricks/LTX-2](./ltx_readme.md) that turns LTX-2 from a text+image-to-video model into an **action-conditioned, multi-view world model** for robotic manipulation. The model takes a *text instruction*, an *initial image*, and a *robot action stream* and generates a video of the robot executing that action — so the operator effectively "teleoperates in a generated world."

> Upstream LTX-2 docs are preserved in [`ltx_readme.md`](./ltx_readme.md). This file describes only the GVT-specific work.

## What's new on top of LTX-2

- **TIA2V conditioning** — Text + Initial-image + Action → Video. The action stream is a sequence of Franka commanded joint angles + gripper, drawn from the [DROID](https://droid-dataset.github.io/) dataset.
- **Multi-view output** — two fixed exterior cameras (ZED 2 stereo rig) + one wrist camera (ZED Mini). Views are tiled into a single canvas so the model's existing spatial attention enforces cross-view consistency without any architecture change (the "tiling trick" — see [Design tricks](#design-tricks)).
- **A render-based action prior** — every action step is materialised as a per-frame *visual prior*: run forward kinematics on the commanded joints, project the Franka skeleton (Stage 1) or render the textured Franka mesh (Stage 2) into the calibrated camera, and feed that as an extra conditioning channel. This grounds the model on *where the arm should be* at every timestep, in image space.

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

experiments/droid_action_cond/    # End-to-end runner scripts (see below)

ltx_readme.md                     # Upstream LTX-2 README
```

The `ltx-action-cond` package is imported as `ltx_action_cond` and exposes the small surface area needed to build the prior:

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

## How we capture the context (prior pipeline)

Each DROID episode is turned into a parallel `(GT video, action vector, per-frame visual prior)` triple. The pipeline lives entirely in `experiments/droid_action_cond/` and runs in two stages because TensorFlow (used to read DROID's RLDS shards) and pyrender (EGL/CUDA-backed) cannot share a process — they segfault on shutdown.

### Step 1 — pick an episode

`pick_episode.py` walks the DROID-100 calibration index, keeps only episodes whose published *IoU* (extrinsic confidence) clears a threshold, picks one fixed exterior camera, and writes everything downstream needs:

```
outputs/
  meta.json           # episode_id, camera_serial/role, n_steps, IoU
  episode.npz         # cmd_joint_position (T,7), cmd_gripper_position (T,),
                      # action (T,7), gt_frames (T,H,W,3),
                      # K_rls (3,3) rescaled to RLDS 320x180, cam2base (4,4)
  01_gt/ovr_NNNN.png  # GT frames
```

Action representation used everywhere downstream: **7-D commanded joints + 1-D commanded gripper**. Cartesian EE (the top-level RLDS `action`) is kept only as a sanity check — we compare the FK-derived flange pixel against the projected Cartesian target at `t=0` to verify the extrinsic.

### Step 2 — build the per-frame prior

`build_prior.py` consumes `episode.npz` and writes both stages of the action-conditioning prior:

| Output dir | Stage | Source | What it is |
|---|---|---|---|
| `02_context/ctx_NNNN.png` | 1 | `fk(joints[t]) → project_points → draw_skeleton` | **Skeleton on black — this is the actual model input** |
| `03_overlay/ovr_NNNN.png` | 1 | same, drawn onto GT | Sanity overlay (visually confirms FK + extrinsic) |
| `04_context_mesh/cmsh_NNNN.png` | 2a | `FrankaMeshRenderer(mode="mask")` | Per-link flat-color mesh on black |
| `05_overlay_mesh/omsh_NNNN.png` | 2a | mask blended on GT | Sanity overlay |
| `06_context_photo/cpho_NNNN.png` | 2b | `FrankaMeshRenderer(mode="photo")` | Textured Franka render on black |
| `07_overlay_photo/opho_NNNN.png` | 2b | photoreal blended on GT | Sanity overlay |
| `prior_report.json` | — | — | Geometry stats, render ms/frame, flange-vs-cart pixel error |

The mesh renderer (`mesh_rendering.py`) drives finger displacement from the commanded gripper signal: `q_finger = (1 − gripper_position) · 0.04 m`, mirrored per finger (DROID convention: 0 = open, 1 = closed). When the commanded gripper saturates short of mechanical close, `droid.resolve_gripper_signal` falls back to the observed signal *for visualization only*; the model input is always commanded.

### Step 3 — scale up

- `demo_more_episodes.py` and `headline_diverse.py` repeat the pipeline across many IoU-calibrated episodes spanning multiple labs / camera serials, in a two-pass `--extract` (TF) / `--render` (pyrender) flow.
- `headline_both_cams.py` handles the case where only ONE of the rig's two exterior cameras has a calibrated extrinsic in a given episode: it estimates the rig geometry `T_other_to_own` from *sibling episodes* (same lab, same camera pair) and composes it with the calibrated camera's extrinsic to recover the second view.
- `extra_videos.py` and `make_summary.py` produce headline videos and a human-readable `outputs/` tree for sharing.

### Step 4 — the wrist camera

DROID does NOT reliably ship a wrist-camera extrinsic. We recover the constant mount `T_cam_to_hand` ourselves:

1. `calibrate_wrist_vggt.py` — feed `(ext1_t0, wrist_t0)` to [VGGT](https://github.com/facebookresearch/vggt) to get the relative wrist-to-ext1 pose. VGGT returns translations *up to scale*.
2. `wrist_calib_scale_iprl.py` / `calibrate_wrist_vggt_v2.py` — apply two scale-fixing strategies:
   - **Focal-anchored:** `scale = K_true_fx / K_vggt_fx` (rotation is scale-invariant; we only scale the translation).
   - **FK-anchored (v2, preferred):** feed VGGT a *third* image `wrist[t_far]` and use the FK-known hand motion `|hand[t_far] − hand[0]|` as a direct metric anchor. Avoids the failure mode where the wrist mount offset is comparable to the hand-to-camera distance.
3. `wrist_calib_validate.py` — render `wrist[0]` from a plane-warp of `ext1[0]` using the recovered `T_cam_to_hand` and visually compare to the GT wrist frame.
4. `wrist_demo.py` / `wrist_video_iprl.py` / `wrist_video_from_wrist0.py` — produce per-episode wrist videos by warping the scene plane through the FK-driven wrist trajectory.

## Design tricks

Three load-bearing tricks make this stack work:

1. **Joint-space conditioning, not Cartesian.** We render the prior from the 7-D commanded joints, not the top-level Cartesian RLDS action. Joint commands are what the controller actually executes; the Cartesian field is only the high-level target and is one IK step removed from physical truth. We keep Cartesian around purely as a calibration sanity-check (flange-vs-cart pixel error at `t=0`).

2. **Action-as-render conditioning.** Instead of feeding the action as a raw 8-D vector concatenated to the latents (which the model has to learn to relate to image space), we *render* it: FK → project → image-space skeleton/mesh. The action arrives in the same coordinate system as the image, so the model only has to learn "match the pose hint," not "translate joint angles to pixel locations." This is why the Stage 2 mesh prior is preferred over the Stage 1 skeleton — it occupies the same image footprint the GT arm does.

3. **Multi-view via tiling, not architecture.** For 3-view (two exteriors + wrist) consistency we tile the views into a single frame (1×3 strip) rather than adding cross-view attention. The model's existing spatial self-attention then handles inter-view consistency for free. The action prior is tiled with the same layout, each tile rendered with that view's own `K` and extrinsic. Cost: the model has to learn that tile seams are scene discontinuities — feasible with finetuning. Benefit: zero architectural change to LTX-2.

## Calibration board (separate utility)

The wrist-camera intrinsics can also be calibrated from scratch using `calibration/generate_charuco_5x7_A4.py`, which writes a print-ready A4 ChArUco PDF (`DICT_5X5_100`, 35 mm squares, 26 mm markers). Print at 100% / "Actual size" — do not let the printer "fit to page" — and verify a square edge with a ruler before using. The `calibration/` folder is `.gitignore`d.

## Environment

The GVT code reuses the upstream LTX-2 environment (`uv sync --frozen`) plus the `droid` extra of `ltx-action-cond`:

```bash
uv sync --frozen
uv pip install -e packages/ltx-action-cond[droid]
```

Hardware: the GVT renders + LTX-2 inference both target a single RTX 5090 (32 GB). Pyrender is currently the throughput bottleneck of the prior; the planned acceleration path is Warp/Newton for full-DROID-scale dataset construction.

## Status

- ✅ Stage 1 + Stage 2 prior on calibrated DROID-100 exterior episodes
- ✅ Two-camera headline (one calibrated + one rig-inferred)
- ✅ Wrist-cam `T_cam_to_hand` via VGGT (FK-anchored scale)
- ⏳ Dataset assembly (action + multi-view tiled GT + tiled prior) for LTX-2 fine-tuning
- ⏳ LTX-2 fine-tune with the tiled action-conditioning channel
