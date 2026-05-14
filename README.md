# GVT — Generative Virtual Teleoperation

A research fork of [Lightricks/LTX-2](./ltx_readme.md) that turns LTX-2 from a text+image-to-video model into an **action-conditioned, multi-view world model** for robotic manipulation. The model takes a *text instruction*, an *initial image*, and a *robot action stream* (TIA2V) and generates the resulting video — so the operator effectively "teleoperates in a generated world."

![3-view action-conditioned prior on a DROID episode](docs/media/three_view_action_cond.png)

<sub>Real DROID episode (CLVR lab). **Rows:** exterior cam 1 / exterior cam 2 / wrist cam. **Columns:** GT video frame | Franka rendered from the action stream only | rendered-on-GT sanity overlay. Middle column = the conditioning channel the model is trained on. <a href="docs/media/three_view_action_cond.mp4">▶ Watch the 12 s clip</a></sub>

> Upstream LTX-2 docs are preserved in [`ltx_readme.md`](./ltx_readme.md). This file only describes the GVT-specific work.

## What's new on top of LTX-2

- **TIA2V conditioning** — Text + Initial-image + Action → Video. The action stream is a sequence of Franka commanded joint angles + gripper, drawn from the [DROID](https://droid-dataset.github.io/) dataset.
- **Multi-view output** — two fixed exterior cameras (ZED 2 stereo rig) + one wrist camera (ZED Mini). Views are tiled into a single canvas so the model's existing spatial attention enforces cross-view consistency without any architecture change (the "tiling trick" — see [Design tricks](#design-tricks)).
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

### Fixed-camera prior across diverse labs

The exterior-camera prior generalises across the DROID institutions (different camera serials, mounts, table layouts) because the only per-episode quantities are the published `K` and `T_cam_to_base`; the FK and the renderer are identical everywhere. Two strips below — `[GT | Franka rendered from action only | rendered-on-GT overlay]` — built with `headline_diverse.py`:

![RAIL lab](docs/media/headline_diverse_rail.png)
<sub>RAIL · <a href="docs/media/headline_diverse_rail.mp4">▶ 24 s clip</a></sub>

![AUTOLab](docs/media/headline_diverse_autolab.png)
<sub>AUTOLab · <a href="docs/media/headline_diverse_autolab.mp4">▶ 6 s clip</a></sub>

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

3. **Multi-view via tiling, not architecture.** For 3-view (two exteriors + wrist) consistency we tile the views into a single frame (1×3 strip) rather than adding cross-view attention. The model's existing spatial self-attention then handles inter-view consistency for free. The action prior is tiled with the same layout, each tile rendered with its own `K` and extrinsic. Cost: the model has to learn that tile seams are scene discontinuities — feasible with finetuning. Benefit: zero architectural change to LTX-2.

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
