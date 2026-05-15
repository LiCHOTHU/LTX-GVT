# GVT Devlog

Research notes, experiments, measurements, and dropped directions. The stable methodology — what GVT *is* and what's actually wired into the pipeline — lives in [`README.md`](README.md). This file is where things go that are useful to remember but aren't part of the current method: characterisations, dead ends, sanity-check demos, and the reasoning behind decisions.

Entries are in reverse chronological order.

---

## 2026-05-15 — Server transfer: pre-staged assets and environment

Moving from the local laptop (RTX 5090, 32 GB) to the training server. Everything needed for inference + fine-tuning is already on-server outside the `LTX-2/` workspace, so wiping/re-cloning the workspace doesn't touch the heavy stuff.

### Pre-staged model checkpoints

Location: `/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/`

```
LTX-2.3/
  ├── ltx-2.3-22b-distilled-1.1.safetensors           43 GB   ← TI2V baseline (used for inference)
  ├── ltx-2.3-22b-dev.safetensors                     43 GB   ← full non-distilled, higher quality
  ├── ltx-2.3-22b-distilled-lora-384-1.1.safetensors  7.1 GB  ← official distillation LoRA
  └── ltx-2.3-spatial-upscaler-x2-1.1.safetensors     950 MB  ← 2× spatial upsampler
gemma-3-12b-it-qat-q4_0-unquantized/                  23 GB   ← Gemma text encoder
```

Total ~117 GB. Sufficient to run inference end-to-end with no re-download.

### Pre-staged DROID dataset

Location: `/storage/project/r-agarg35-0/lwang831/droid_hf/` — 374 GB, untouched, project-storage. Read-only training-data root; do not re-download from HF.

### Server environment

Activate the `ltx` conda env before any Python work — the working server-side environment is conda, not the laptop's `.venv`:

```bash
source $(conda info --base)/etc/profile.d/conda.sh && conda activate ltx
```

The laptop uses `.venv` (Python 3.13, uv-managed). Different machine, different convention. The local verify run earlier today hit missing-package errors (`pycollada`, `pip` itself) because `.venv` was incomplete; conda `ltx` does not have that problem.

### Wiring this knowledge

The same information is now in two persistent memory files so future sessions on the server load them automatically:
- `memory/server_paths.md` (reference) — checkpoint + DROID paths
- `memory/server_env.md` (feedback) — conda `ltx` on server, `.venv` on laptop

Use the absolute server paths directly; do NOT symlink into the workspace.

---

## 2026-05-15 — Decision: plane homography is THE default wrist-context renderer

After the splat-vs-homography side-by-side on the 5 verification episodes (`experiments/droid_action_cond/outputs/verify_splat_gpu/<ep>/wrist.mp4`, 4-panel `GT | GPU splat | CPU splat | homography`), **plane homography is locked in as the project's default and ultimate wrist-camera context renderer**. The decision is captured in the `gvt-wrist-render-recipe` memory (updated this date) — future sessions should not reconsider this for the flat-tabletop DROID workspace.

### Why homography wins on this data

The DROID workspace is approximately a single horizontal plane. Under that assumption, *both* renderers use the same math up to a point — but homography then takes the closed-form path while the splat discretises:

| Property | Plane homography | Splat (CPU loop / GPU vec) |
|---|---|---|
| Output density | every pixel computed (100% coverage) | sparse — disk gaps when wrist zooms in |
| Sampling | bilinear sub-pixel | nearest-pixel disk paint |
| Inputs needed | wrist[0], K, joints, `T_cam_to_hand` | + a SceneCloud (which is itself built from the same plane assumption) |
| Per-frame cost | ~6 ms (pure NumPy) | 8 ms GPU / 30 ms CPU baseline |
| GPU dependency | none | yes for the fast variant |
| Behaviour on non-planar scenes | wrong (same as splat-with-plane-cloud) | correct *if* given a real 3D cloud |

For our case the homography is a strict win on every dimension. The splat would only beat homography if we fed it a **non-planar** scene cloud — stereo-triangulated, VGGT/MVS — which we don't and aren't planning to.

### Wiring state after this entry

- `experiments/droid_action_cond/wrist_demo.py:264-269` — already used homography; unchanged.
- `experiments/droid_action_cond/motion_flow_demo.py:197-208` — **switched from `render_wrist_scene_splat_gpu` → `render_wrist_via_plane_homography`** in this entry's commit. The 2026-05-15 splat-optimisation entry below describes the previously wired splat path (now removed from this hot loop). The scene-cloud build and `_scene_cache` plumbing are gone.
- `experiments/droid_action_cond/wrist_video_from_wrist0.py` — already used homography; unchanged.

### What stays in the tree (retained for future non-planar work)

The three splat entry points in `packages/ltx-action-cond/src/ltx_action_cond/wrist_render.py` — `render_wrist_scene_splat` (baseline), `render_wrist_scene_splat_vec` (CPU vec), `render_wrist_scene_splat_gpu` (CUDA vec) — stay where they are. They are now the **generic-3D-cloud renderer** for future work (e.g. if we ever wire in stereo triangulation via `reconstruct_scene`, or a VGGT/MVS point cloud). They are no longer on any default code path.

The `2026-05-15 — Wrist-splat optimisation` entry below remains valuable as a reference for how to vectorise sparse-to-dense scatter on GPU, even though the optimised renderer is no longer the production wrist path.

---

## 2026-05-15 — Wrist-splat optimisation: 21.9× speedup via vectorised GPU path

Acting on the per-frame profile below, attacked the wrist `render_wrist_scene_splat` 27 ms bottleneck four ways and benchmarked head-to-head on the same scene + pose trajectory (CLVR+13759f6e, H=180 W=320, 57 600 scene pts at stride=1, radius=3 disk, 50 frames per variant, 5090).

Two new renderer entry points added alongside the original for the keepers:
- `render_wrist_scene_splat_vec` (`packages/ltx-action-cond/src/ltx_action_cond/wrist_render.py:432`) — NumPy fused-broadcast variant
- `render_wrist_scene_splat_gpu` (`packages/ltx-action-cond/src/ltx_action_cond/wrist_render.py:497`) — torch/CUDA, scene tensors resident on GPU via a per-episode `_scene_cache` dict

### Results

| Variant | ms/frame | Speedup | mean \|Δ\| | SSIM | PSNR | Verdict |
|---|---|---|---|---|---|---|
| [1] baseline — CPU, 29-iter py loop, stride=1 | 30.23 | 1.00× | 0 | 1.0000 | ∞ | reference |
| [2] stride=2 (4× sparser scene, same loop) | 7.82 | **3.9×** | 5.7 | 0.9754 | 22.7 | ✅ |
| [3] vectorised disk — CPU NumPy | 37.28 | 0.81× | 2.1 | 0.9931 | 28.2 | ❌ slower |
| [4] vectorised + torch on GPU | 5.58 | **5.4×** | 8.6 | 0.9649 | 21.1 | ✅ |
| [5] **stride=2 + vec GPU (stacked)** | **1.38** | **21.9×** | 9.5 | 0.9557 | 20.1 | ✅ best |

Comparison strip with per-variant diff images (bottom row = `|Δ| vs baseline` brightened 4×): saved during the bench to `/tmp/bench_splat_strip.png`; the texture, geometry, and labels in the top row are visually identical across all five variants — the visible diffs live exclusively at object-edge boundaries where the disk-overlap tie-breaker chooses a different one of two near-equally-close points. PSNR > 20 dB, SSIM > 0.95 across the board. The high "frac diff pixels" number (60–80%) is misleading on its own — `mean |Δ| ≈ 2–9 / 255` says each diff pixel is only off by a handful of 8-bit levels.

### Algorithm: the trick that mattered

The vectorised path replaces the 29-iteration `(du, dv)` loop with a single broadcast → mask → scatter:

```python
# offs: (D, 2) -- 29 valid disk offsets within radius r
NU = iu[:, None] + offs[None, :, 0]   # (N, D)
NV = iv[:, None] + offs[None, :, 1]
m  = (NU >= 0) & (NU < W) & (NV >= 0) & (NV < H)
flat = (NV * W + NU)[m]               # row-major: all D px of pt 0, then pt 1, ...
rgb[flat]   = cols_broadcast[m]
depth[flat] = z_broadcast[m]
```

The non-obvious correctness piece: **sort the N points by depth descending *before* the (N, D) explosion**, not after. Then the row-major flatten of `[m]` preserves back-to-front order across rows, and the destination assignment `rgb[flat] = vals` resolves duplicate indices via NumPy / torch's last-write-wins semantics → the closer point wins exactly where it should.

Naive version (sort *after* explosion, i.e. on the 1.7 M post-broadcast array) is correct but ~3× slower than baseline — both because the argsort cost dominates and because the 5 MB intermediate kills L2.

### The surprise: NumPy vectorisation *lost* on CPU

The prediction was vec-CPU = 2–4 ms (budgeting only Python dispatch overhead). The measurement is 37 ms — **slower than the 30 ms baseline**. NumPy can't fuse the broadcast → boolean-mask → fancy-gather chain, so the (57 600, 29) intermediate materialises as a 5 MB uint8 colour array + 13 MB int32 index array *per frame*; memory traffic now exceeds L2 capacity, and the gather/scatter become bandwidth-bound. The original 29-iter loop's per-iter ops are ~700 KB each — they fit in L2, so each iteration runs at ALU rate and the Python dispatch overhead is genuinely modest. *That loop is already well-tuned for the cache hierarchy*; only operator-fusion via Numba / Cython / a C extension could beat it on CPU.

Moral: dispatch overhead is the *first* thing to attack, but it's not always the dominant cost. Memory bandwidth gates the alternative, and "vectorising harder" can lose to a small loop that fits in cache.

### Why GPU won

The same fused expression on CUDA pays a similar memory-traffic cost (still materialises the (N, D) intermediates), but the 5090's effective scatter/gather bandwidth (~1.5 TB/s HBM vs ~50 GB/s host DDR) is ~30× higher, so the bandwidth-bound regime is much faster. Cost decomposition for [4]:

- scene `(Ph, cols)` cached once per episode → one upload, zero recurring transfer
- per-frame: `T_b2w` 4×4 + `K` 3×3 upload (negligible), all matmul / broadcast / mask / argsort / scatter on device, final `(H, W, 3)` u8 → CPU (~173 KB at PCIe = ~17 µs)
- projection deliberately kept in float64 to match the baseline's iu/iv rounding exactly (no silent drift from fp32)

The 5.58 ms / frame is dominated by the per-frame argsort + the final CPU readback; both are unavoidable without changing the API.

### Quality verification

Diff metrics computed against the baseline frame at t=50, plus a saved 4800×1136 comparison strip with the 5 renders on top and 5 |Δ| maps on the bottom. Sources of the residual diffs:

- **[2] stride=2**: 4× sparser cloud → different disk-overlap geometry at edges (but every screen pixel is still covered by *some* point's 3-px disk, so no holes appear). Coverage gap goes to zero, edge texture pattern differs slightly.
- **[3] vec CPU**: `argsort` tie-break permutation differs from baseline's default quicksort. Highest SSIM (0.9931) of any variant.
- **[4] / [5] GPU**: torch `argsort` on CUDA is not order-deterministic for tied keys; combined with `index_put_` for duplicate indices being thread-order-dependent, near-tied-depth disk-overlap pixels pick a different close-tie winner. Spatial pattern of differences is still confined to object edges (visible in the diff image).

None of the variants alter *what* is rendered or *where* things are — only which of two near-equally-close points colours a given boundary pixel.

### Status

Two new entry points live in the tree: `render_wrist_scene_splat_vec` and `render_wrist_scene_splat_gpu` (`packages/ltx-action-cond/src/ltx_action_cond/wrist_render.py:432, :497`).

**Wired** into `experiments/droid_action_cond/motion_flow_demo.py` (the only splat hot loop in the codebase — `wrist_demo.py` and the production context-render pipeline use the entirely different `render_wrist_via_plane_homography` algorithm, no point cloud, no disk). The demo's render path now calls `render_wrist_scene_splat_gpu` with `sample_stride=2` + a per-episode `_scene_cache` dict. Smoke-tested end-to-end on CLVR+13759f6e (T=186): **2.84 ms/frame** in the real demo loop vs ~30 ms/frame baseline.

**Not wired** into `wrist_calib_validate.py` / `wrist_calib_scale_iprl.py` — those render a single frame per episode (t=0 only), so they'd pay the ~1–2 s lazy `import torch` cost for a per-call saving of ~25 ms. Net loss; staying on the CPU baseline there is correct.

Reproduce with `/tmp/bench_splat.py` against any episode under `experiments/droid_action_cond/outputs/context/<ep>/data.npz`.

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
