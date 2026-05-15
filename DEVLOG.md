# GVT Devlog

Research notes, experiments, measurements, and dropped directions. The stable methodology — what GVT *is* and what's actually wired into the pipeline — lives in [`README.md`](README.md). This file is where things go that are useful to remember but aren't part of the current method: characterisations, dead ends, sanity-check demos, and the reasoning behind decisions.

Entries are in reverse chronological order.

---

## 2026-05-15 — Per-step render-time analysis

Profiled the per-frame rendering cost of the two context renderers (fixed-cam Franka mesh via pyrender vs wrist plane-warp scene splat) to identify the bottleneck. Measured on the RTX 5090 with 320×180 RLDS frames, averaged over 50 frames, scene cloud = 57 600 points at `sample_stride=1`. The wrist renderer is **3.4× slower than a single fixed-camera renderer** despite touching fewer primitives — the architectural cost of point-cloud splat vs hardware mesh rasterisation.

**Fixed camera — `FrankaMeshRenderer.render()` (mesh via pyrender / EGL):**

| Sub-step | Time/frame | Share |
|---|---|---|
| `fk_urdf` (FK on 7 joints) | 0.08 ms | 0.9% |
| `pyrender.Scene()` construction | 0.02 ms | 0.3% |
| add 9 link meshes to scene | 0.41 ms | 4.6% |
| finger pose calc + add 2 finger meshes | 0.09 ms | 1.0% |
| `IntrinsicsCamera` + scene.add | 0.04 ms | 0.5% |
| 2 directional lights + scene.add | 0.09 ms | 1.0% |
| **`OffscreenRenderer.render(scene)`** — GPU rasterise + framebuffer read-back | **8.10 ms** | **91.7%** |
| **Total** | **8.84 ms** | |

The `OffscreenRenderer.render` step dominates. The GPU triangle rasterisation of ~few-thousand-triangle Franka meshes is sub-millisecond on the 5090; the 8 ms is spent on (a) pyrender's per-call Python-side scene compilation (no cached GL state) and (b) the synchronous framebuffer **read-back via PCIe**. The per-call latency dominates, not the rasterisation work.

**Wrist camera — `render_wrist_scene_splat()` (57 600-point CPU splat):**

| Sub-step | Time/frame | Share |
|---|---|---|
| `inv(T_wrist_to_base)` 4×4 | 0.03 ms | 0.1% |
| homogenize + base→wrist transform (4×N matmul) | 0.26 ms | 0.9% |
| `z > 0` front filter | 0.94 ms | 3.1% |
| `K @ Pc` + divide by z | 0.14 ms | 0.5% |
| round to int pixels | 0.06 ms | 0.2% |
| in-bounds mask + filter | 0.47 ms | 1.6% |
| `argsort(-z)` (z-buffer order) | 0.82 ms | 2.7% |
| allocate output rgb + depth buffers | 0.01 ms | 0.0% |
| **disk paint loop (29 du,dv offsets)** — Python/NumPy scatter, ~1.7 M pixel writes | **27.19 ms** | **90.9%** |
| **Total** | **29.93 ms** | |

All the geometry sub-steps are sub-millisecond — the cost is entirely in painting 57 600 disks via a Python double-loop over the (du, dv) offsets within a 3-pixel radius, each iteration doing one vectorised NumPy scatter across all points. ~29 iterations × Python-loop overhead × 57 600-element ops = 27 ms.

**Per-frame total for the three views**: `8.84 + 8.84 + 29.93 ≈ 47.6 ms`, so single-process throughput tops out at ~21 fps. End-to-end `wrist_demo.py` measures ~10 fps wall-clock; the gap comes from per-frame 3× upscaling (`Image.LANCZOS`), label-strip drawing, frame stacking, and `imageio` h.264 encoding of 4 mp4s per episode.

**Optimisation knobs (in order of leverage):**

1. **`sample_stride=2`** in `reconstruct_scene_from_plane` → 4× fewer scene points → wrist splat drops to ~7 ms. One-line change, near-zero visual quality cost since each pixel is already covered by a 3-px disk.
2. **Vectorise the disk paint loop** — replace the 29 `du,dv` iterations with a single `np.add.at` over a precomputed disk-offset table, or move to torch `scatter_add_` on GPU → wrist splat drops to ~3–5 ms. Biggest win.
3. **Batch the fixed-cam renders** — share a single pyrender scene + camera across consecutive frames (only mesh poses change) to amortise the Python-side compilation overhead. Could halve the fixed-cam 8 ms.
4. **Drop the 3× upscale + 4-video encode** in `wrist_demo.py` for fast-preview runs. The upscale is purely cosmetic; native 320×180 is fine for verification.

Reproduce with `/tmp/profile_render.py` against any episode dir under `experiments/droid_action_cond/outputs/wrist_demo/<ep>/data.npz`.

---

## 2026-05-15 — Warped-noise (Go-with-the-Flow): explored, dropped

A Go-with-the-Flow style warped-noise direction was explored as a way to bake the robot's motion into the diffusion noise seed, then **deliberately abandoned** in favour of pure in-context learning (IC-LoRA on the rendered Franka prior video + raw action-token conditioning via `VideoConditionByActionTokens`). The motion-flow library, the noise-warping kernel, and the visualisation demo are kept in the tree for record/reference; they are not wired into the trainer and we don't plan to revisit them.

### What's in the tree

| Path | What it is | Status |
|---|---|---|
| `packages/ltx-core/src/ltx_core/conditioning/warped_noise.py` | Self-contained PyTorch noise-warping kernel (`apply_warped_noise(noise, flow)`), plus `regaussianize`. Backward-gather + regaussianise variant, smoke-tested at LTX-2's latent shape `(B, 128, F, H, W)`. | Built, smoke-tested, **not used at training time** |
| `packages/ltx-action-cond/src/ltx_action_cond/motion_flow.py` | Analytical per-view flow generator: `flow_from_joints_for_view` (arm flow from FK projection) + `scene_plane_flow_for_view` (camera-egomotion flow under a plane assumption) + `flow_to_rgb` (HSV visualisation). | Built, **circle-splat approximation** for arm flow — see issues below |
| `experiments/droid_action_cond/motion_flow_demo.py` | 5-episode visualiser: 3-row × 4-col grid per episode, rows = `[ext1, ext2, wrist]`, cols = `[GT, rendered context, rendered-on-GT, motion flow]`. | Built, output mp4s under `outputs/motion_flow_demo/<ep>/` |

### How the flow was built

Two components, combined per view:

1. **Arm flow** — for each transition `(t, t+1)`, run FK on both joint vectors, project the 9 Franka link keypoints through the camera, and take the per-keypoint pixel velocity. Each projected keypoint is then **splatted as a disk of radius 18 px**: every pixel within that radius inherits the nearest keypoint's velocity.
2. **Background flow** — for each pixel, back-project through the camera at frame `t` onto a plane `z = plane_z` in base coordinates, forward-project through the camera at frame `t+1`, take the difference. For a fixed `cam2base` (the exterior cameras) this is identically zero, so the result is arm-only flow. For the wrist camera (time-varying `cam2base = T_hand_to_base(joints[t]) @ T_cam_to_hand` with the VGGT-calibrated mount) this captures the dominant motion in the view — the scene sliding past as the camera rides on the gripper.

Arm flow overrides background flow on pixels within the disk radius; otherwise the background value wins. Visualisation uses HSV with hue = direction, saturation = magnitude normalised by per-view 99th percentile.

### Why dropped — the two unsolved problems

**1. The disk splatter is the wrong sparse-to-dense kernel.** The arm flow in the demo shows up as colored *circles* per joint (literally — the splat kernel is a disk). Two failure modes follow:

- Pixels covered by the arm but >18 px from any projected keypoint get zero flow (no arm-motion-prior where you most want it).
- Pixels in the background that happen to be within 18 px of a projected keypoint get *fake arm motion* — the noise would warp along arm trajectories on empty space.

The standard fix borrows from rigged-character animation: render the Franka mesh foreground mask (we already produce this in `mesh_rendering.py` mode=`"mask"`), then assign each arm pixel either (a) the projected velocity of the *link* that covers it (cheap, per-link rigid motion), or (b) the true per-pixel correspondence via the depth buffer (back-project pixel→3D point on link surface→transform with the new link frame→reproject). Option (b) gives arm-shaped flow regions with correct within-link rotation; it's what we'd ship if we revived this.

The closest precedent in the literature is the **Gaussian-filtered trajectory splatter** used by DragNUWA / DragAnything / MotionCtrl ([arXiv 2308.08089](https://arxiv.org/abs/2308.08089), [2403.07420](https://arxiv.org/abs/2403.07420), [2312.03641](https://arxiv.org/abs/2312.03641)) — strictly better than hard disks, but those same papers acknowledge it "limits fine-grained motion controllability and often fails to disentangle object and camera motions."

**2. Even a perfect Franka mesh flow misses scene motion induced by the robot.** This is the deeper issue. The robot doesn't just move the arm — it moves *things*. Concretely:

- **Fixed cam, grasping.** A grasped object travels with the gripper at the gripper's pixel velocity. The mesh-mask flow assigns it *zero* because the object isn't part of the Franka rig — we have no 3D model of "the apple on the table." Wrong flow on exactly the pixels that matter most for the task.
- **Wrist cam, grasping.** Opposite failure. The object moves *with* the wrist camera, so in the wrist frame it should be nearly *static* in pixel coords. But our plane-warp egomotion treats it like background and assigns it a sliding flow. Spurious flow on the held object.
- **Drawer / lever opening.** A non-rigid scene element with no model at all; the flow generator has no information to recover its 2D motion.

There is no model-only fix. The clean options all require either GT video (and a flow estimator like RAFT) at training time, a per-episode 3D scene reconstruction, or per-object tracking — none of which we have wired up. Go-with-the-Flow itself sidesteps this by running RAFT on the *GT* video; that's fine at training but the same flow isn't available at inference, so any pipeline depending on it has a train/test distribution shift that needs separate handling.

### Decision

The warped-noise direction is dropped, with the scaffold preserved, until either (a) we add a real flow estimator (RAFT or equivalent) to the training pipeline and accept the train/test gap, or (b) we add per-episode object/scene tracking. Neither is on the near-term roadmap. The action conditioning we already have — IC-LoRA on the rendered prior + raw action tokens — is the path forward for the first GVT fine-tune. EPiC (arXiv 2505.21876) and GEN3C (CVPR 2025 Highlight) publish the same "anchor video" recipe we're using, validating that an explicit geometric anchor is sufficient without warped noise.

Sources:
- [Go-with-the-Flow (arXiv 2501.08331)](https://arxiv.org/abs/2501.08331) — original warped-noise paper, uses RAFT on RGB video
- [DragNUWA (arXiv 2308.08089)](https://arxiv.org/abs/2308.08089), [DragAnything (arXiv 2403.07420)](https://arxiv.org/abs/2403.07420), [MotionCtrl (arXiv 2312.03641)](https://arxiv.org/abs/2312.03641) — sparse-trajectory-to-dense-motion patterns considered as alternatives to the disk splat
- [EPiC (arXiv 2505.21876)](https://arxiv.org/abs/2505.21876), [GEN3C (CVPR 2025 Highlight)](https://research.nvidia.com/labs/toronto-ai/GEN3C/) — the "anchor video" recipe we ended up using instead

---

## 2026-05-13 — Cross-lab sanity: rendered prior generalises across DROID institutions

The exterior-camera prior generalises across the DROID institutions (different camera serials, mounts, table layouts) because the only per-episode quantities are the published `K` and `T_cam_to_base`; the FK and the renderer are identical everywhere. Two strips below — `[GT | Franka rendered from action only | rendered-on-GT overlay]` — built with `headline_diverse.py`:

![RAIL lab](docs/media/headline_diverse_rail.png)
<sub>RAIL · <a href="docs/media/headline_diverse_rail.mp4">▶ 24 s clip</a></sub>

![AUTOLab](docs/media/headline_diverse_autolab.png)
<sub>AUTOLab · <a href="docs/media/headline_diverse_autolab.mp4">▶ 6 s clip</a></sub>

Same calibration code, same FK, same pyrender setup — no per-lab tuning was needed. The April-2025 DROID calibration release is reliable enough that the rendered prior tracks the GT video across labs we'd never inspected before.
