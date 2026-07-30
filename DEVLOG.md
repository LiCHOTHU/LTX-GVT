# GVT Devlog

Research notes, experiments, measurements, and dropped directions. The stable methodology — what GVT *is* and what's actually wired into the pipeline — lives in [`README.md`](README.md). This file is where things go that are useful to remember but aren't part of the current method: characterisations, dead ends, sanity-check demos, and the reasoning behind decisions.

Entries are in reverse chronological order.

---

## 2026-07-28 — Perceptual Flow Matching loss for the tomato_sauce arms + a silent MSE-regression bug, and a repo cleanup

Three things happened over 2026-07-27/28: (1) we swapped the trainer's latent-space velocity MSE for a **Perceptual Flow Matching (PFM)** loss on the `tomato_sauce` active-learning arms, (2) we caught and fixed a bug where PFM was *silently* dropping back to MSE on every watchdog resubmit, and (3) we archived ~95 dead/superseded files out of `experiments/libero_sim` and `experiments/droid_action_cond` without disturbing the live run.

### 1. PFM loss (opt-in, env-gated)

Motivation: the standard flow-matching objective regresses the velocity `v = ε − x₀` in VAE-latent space with masked MSE, which is mean-seeking (blurry, weak on few-step generation). **Perceptual Flow Matching** ([arXiv 2607.03524](https://arxiv.org/abs/2607.03524)) instead supervises the *recovered clean prediction* in a frozen perceptual feature space, shifting the regression toward mode-seeking (sharper). Matrix-Game 3.5 — a PRoPE video world model — uses PFM during its causal adaptation, which is a direct precedent for our setup. (Their repo is inference-only; this is implemented from the paper, not copied.)

What it computes, per training step, when `PFM_ENABLE=1`:
1. Recover the clean latent from the model's predicted velocity: `x̂₀ = x_t − σ·v_pred` (and the GT `x₀ = x_t − σ·v_true`), on the target tokens only (conditioning tokens are pinned to GT).
2. Unpatchify both back to the `(B, 128, F, H, W)` latent grid.
3. Decode both through the **frozen VAE decoder, in the autograd graph, on GPU**.
4. Match the two in **DINOv2 ViT-S/14** feature space (imagenet-normalised, 224-interpolated): `‖φ(x̂₀_px) − φ(x₀_px)‖²`.

`PFM_MSE_WEIGHT` lets you blend the old latent MSE back in (`0.0` = pure replacement, which is what the arms run). `PFM_FRAMES` optionally decodes a frame subset. It's gated by env, matching the existing `AL_V2` flag pattern, so `PFM_ENABLE=0` (default) is byte-identical to the old trainer.

| File | Change |
|---|---|
| `packages/ltx-trainer/src/ltx_trainer/trainer.py` | `_pfm_enabled` / `_ensure_pfm` / `_dino_feats` / `_perceptual_loss`; perceptual term added at the loss site in `_training_step`; VAE decoder kept **on GPU** (not CPU-offloaded) when PFM is on |
| `.../training_strategies/base_strategy.py` | `ModelInputs.video_latent_grid: (F,H,W)` — needed to unpatchify `x̂₀` |
| `.../training_strategies/video_to_video.py` | populates `video_latent_grid` before patchify |
| `experiments/libero_sim/configs/active_learning_tomato_sauce.env` | `PFM_*` block |

Perceptual model is DINOv2 ViT-S/14 loaded from the **offline** torch-hub cache (`~/.cache/torch/hub/`), no download. Validated before launch: isolated memory smoke = full 33-frame decode + DINOv2 + backward is ~6 GB extra on an H200; a 2-step end-to-end smoke on the real 22B model trained through the full path with no OOM. Cost: **~2× slower per step** (~0.18 vs ~0.34 steps/s) — the full-clip decode + DINOv2 runs every microbatch.

### 2. The silent MSE-regression bug (and the real fix)

**Symptom.** After relaunch, the arms *looked* like they were training with PFM, but weren't: no `PFM: VAE decoder kept on GPU` banner in the logs, and steady-state **2.8 s/step = the MSE speed**, not PFM's ~5.5 s/step. Only the very first job launched right after the watchdog started had the banner; every subsequent watchdog *resubmit* had regressed to plain MSE — while still correctly carrying the other `.env` knobs (`SCORER=objbbox`, the custom `poke:10` mix).

**Root cause.** The watchdog sources the arm config **once at startup** (`set -a; source cfg; set +a`) and relies on `sbatch --export=ALL` to carry those vars into each job. `PFM_ENABLE` was reaching the *job's* environment inconsistently across resubmits (a stale/partial submitting-env → `--export=ALL` chain), so `train.py` saw `PFM_ENABLE` unset and ran the default (MSE). The failure is silent because the loss switch is a no-op fallback, not an error.

**Fix — make the job source its config itself, not trust the export chain.** The arm sbatch now re-sources the config `.env` at *job* time, so every knob (`PFM_*`, `SCORER`, mix, …) is authoritative from the file regardless of what the submitting process had. The watchdog passes its config path as `AL_CONFIG` (both `export AL_CONFIG` and explicitly in the `--export` list). `VERSIONS` is set per-arm via `--export` and is *not* in the `.env`, so it survives the re-source.

- `experiments/libero_sim/run_active_learning.sbatch` — sources `${AL_CONFIG}` after conda activate; logs `[sbatch] sourcing arm config: …`
- `experiments/libero_sim/monitor_active_tomato_sauce.sh` — `export AL_CONFIG="$CONFIG"` + `--export=ALL,VERSIONS=…,AL_CONFIG=…`

**Operational lesson (already knew, re-confirmed):** a *running* watchdog does **not** re-source `.env` edits — you must restart it. When the watchdog process later died on its own, the arms were left queued with no babysitter; the recovery was to cancel the stale (PENDING, zero-compute-lost) MSE jobs and restart the watchdog fresh, which resubmitted all four with PFM, resuming from their latest checkpoints (rand 38000, maxloss/decaware 35250, rho_div 34000).

> **Caveat on the resulting run.** Because PFM was activated mid-schedule (resume, not clean restart from step 30000), steps 30000→~35–38k were MSE-trained and the remainder is PFM — a mixed trajectory. This is a "does PFM sharpen the WM" run, **not** a clean cross-arm AL ablation. A pure-PFM-from-30000 restart is available if the comparison needs to be airtight; it was deferred to avoid discarding the warm-started checkpoints.

### 3. Repo cleanup (experiments/ decluttered, live pipeline untouched)

`experiments/libero_sim` and `experiments/droid_action_cond` had accumulated ~40 and ~30 top-level scripts. We mapped the **live `tomato_sauce` dependency set** (17 files in `libero_sim` + 2 in `droid_action_cond` — the latter, `build_dataset_resumable.py` and `chunk_context.py`, are pulled into the LIBERO run via `sys.path.insert`) and left every one of them in place, then moved everything else into a structured `archive/`:

- `libero_sim/archive/{eval,builds,al_experiments,smoke,bench,infer,misc}/` (63 files)
- `droid_action_cond/archive/{wrist_calib,wrist_compare,demos,dataset_variants}/` (33 files)

Guiding rule: **the running experiment must not break**, so the import-coupled live pipeline stays at top level (moving a file the loop subprocess-calls by path, or a sibling it imports, would break an in-flight round). Verified post-move that every subprocess/import target the live loop references still resolves. Also deleted 8 stray `*.out` logs from the repo root and `libero_sim/`. A deeper functional reorg of the live pipeline itself (into `build/`, `scoring/`, … subdirs) is deferred — it would require moving import-coupled files and a coordinated restart, and the payoff didn't justify the breakage risk while the arms are running.

---

## 2026-07-08 — Active learning result: object-single AL is a null, and why (pivot to decision-aware WM)

The object-centric single-scene active-learning experiment came back **negative-to-null**, and the diagnosis reframes the whole direction. Recording the result, the evidence, and the literature-backed conclusion so we don't re-run this configuration.

### The experiment

Two arms resumed from the step-30000 PRoPE checkpoint and trained 30000→50000 on one scene (`LIVING_ROOM_SCENE1`, "pick the tomato sauce and put it in the basket"). Each round proposes 30 candidate chunks (mix reach:10 / perturb:14 / random:6) and keeps `SELECT_K=8`:
- **strategic** (`act_obj`) — the 8 highest teacher-forced latent-loss chunks (pure max-loss top-K, `AL_V2=0`, `SCORER=objbbox`).
- **random** (`rand_obj`) — 8 uniformly.

Both reached the matched 50000-step target (strategic `COMPLETED`; random ran to 49500+ under repeated embers preemption + `--requeue`).

### Result — strategic does NOT beat random

Head-to-head held-out eval (`eval_arms_local.py`, step-46250 matched pair, held-out perturb+random chunks, PSNR/SSIM on pred-vs-GT rollouts):

| Metric | strategic | random | Δ (strat−rand) | significance |
|---|---:|---:|---:|---|
| PSNR (all) | 24.23 | 24.60 | **−0.37 dB** | t=−2.46, p≈0.017, random won 37/64 chunks |
| SSIM (all) | 0.900 | 0.899 | +0.001 | t=+0.95 (tied) |

Strategic is *slightly worse* on the one quality metric that was the point of the experiment. A statistical wash, tilting to random.

### Why — three stacked confounds (all evidenced)

1. **Pool saturation.** Only ~30 distinct episode identities in the scene; both arms had added essentially all of them by round ~3. After that the training diets are near-identical, so *selection has nothing left to act on.* The proof: the two arms' **training-loss curves are superimposable** (within ±0.002 at every 2000-step bin) and both **plateau at ~0.15 by step 44000**, flat-to-slightly-rising after. Same learning, same path — the wash is mechanical, not noise.
2. **Undefined exploration support → wrong signal.** Pure max-loss top-K over `reach ∪ perturb ∪ random` preferentially grabs the *least-predictable* random-action rollouts. That inflates average MSE/PSNR (it over-invests in the hardest-to-render data) without improving structure (SSIM tied) — textbook greedy hard-example-mining pathology.
3. **The ~0.15 floor is not aleatoric.** LIBERO sim is deterministic, so the loss floor is **partial-observability + VAE-reconstruction + LoRA capacity (misspecification)**, not environment noise. Data selection cannot lower a capacity/representation floor — which is exactly why both arms hit the same 0.15 regardless of diet. Not undertraining (converged/plateaued); if anything mildly over-trained on 30 identities.

### Literature check (verified deep-research pass, 2026-07-08)

A 107-agent, adversarially-verified review (25 sources, 23/25 claims confirmed; note `[[data_sampling_litreview]]`) confirms the failure was predictable and names the fixes:
- **RHO-LOSS** (Mindermann et al., ICML 2022): greedy max-loss selects "noisy (not learnable) or less task-relevant" points; the fix is *reducible* loss = training loss − a holdout/reference model's loss, which subtracts the irreducible component. **Direct antidote to our max-loss run.**
- **BatchBALD / core-set** (Kirsch NeurIPS 2019; Sener & Savarese ICLR 2018): naive top-K by any per-point score "acquires similar and redundant points, sometimes performing worse than random." Our `SELECT_K=8` top-K is that pathology. Fix = diverse batch (**BADGE**, Ash ICLR 2020 — tuning-free uncertainty×diversity).
- **MacKay 1992 / value-equivalence** (Farahmand 2017, Grimm 2020): information/accuracy criteria are "the right answer to the wrong question" under misspecification; target a *region of interest* / decision-relevant error, not global accuracy.

### The pivot (committed)

Stop optimizing for a globally-accurate, general action-conditioned world model — it's the wrong objective *and* unreachable with a LoRA on one scene. Target a **decision-aware / value-aware world model** accurate on the **policy-relevant distribution**, for eventual model-based-RL co-training with the trained LIBERO FlowMatchingPolicy (0.80 BC success, lives in the separate `imitation` repo — see `[[fm_bc_policy_asset]]`). New AL signal, when it re-enters: **reducible-loss + diverse batch, aimed at the policy occupancy.** Phased eval plan (T1 single-step policy-action consistency → T2 action-divergence vs horizon → T3 closed-loop task success). Open problem the verified theory does not cover: active learning against a **non-stationary occupancy** that moves as the policy improves.

Assets: `experiments/libero_sim/eval_arms_local.py` (the held-out arm comparison), `outputs/obj_single_arm_eval_step46250/arm_compare.{json,txt}` (the deliverable). Memory: `[[al_null_decision_aware_pivot]]`, `[[data_sampling_litreview]]`, `[[fm_bc_policy_asset]]`.

---

## 2026-06-14 — End-to-end timing: context build vs DiT vs decode (active-learning budget)

Profiled the full LIBERO per-chunk / per-episode cost ahead of the active-learning loop, so we know where the wall-clock goes. Instrumentation is `GVT_TIME`-gated (no-op in production) in `infer_action_cond.py` and `build_libero_context.py`. Bench jobs: `bench_infer_time.sbatch` (9957781, inference split on the real eval code path, step 80000, PRoPE, 256×512×33, 30 steps), `bench_render_split.sbatch` / `bench_render_only.sbatch` (9957931 / 9957941, FK render split from a cached capture npz). All on gpu-h200.

**Two regimes, and the key framing:** "context building" is *offline precompute*, not part of inference. The 22B model is never resident during context build; inference loads precomputed latents/conditions/cameras off disk and never touches the VAE encoder or Gemma. They were conflated before — keep them separate.

**Full pipeline per chunk (and ×7 ≈ one T=200 episode):**

| Phase | s/chunk | s/episode | Bottleneck |
|---|---:|---:|---|
| Context: sim rollout | ~0.9 | ~6 | — |
| Context: **FK render** | ~14.7 | **~103** | `fuse_scene_cloud` |
| Context: VAE-encode + Gemma | ~8.2 | ~57 | encode batch |
| Inference: **DiT (30 steps)** | ~4.75 | ~33 | the model (158 ms/step) |
| Inference: VAE decode (pred) | ~1.0 | ~7 | — |

Inference deploy cost = DiT + 1 decode ≈ **5.75 s/chunk** (DiT ~83%); eval cost (3 decodes for pred|gt|ctx) ≈ 7.8 s/chunk; first chunk pays a one-time ~20 s CUDA/cuDNN warmup.

**FK context render split** — capture pass (MuJoCo/robosuite, clean in-situ numbers):

| Stage | Time |
|---|---:|
| env setup (scene compile + `reset_from_xml_string`) | 4.8 s |
| per-frame loop: 200 fr × 3 cam × (RGB+depth+seg) | 21.5 s (36 ms/frame-cam) |
| npz save (100 MB compressed) | 9.8 s |

Render pass (pyrender) — **relative ordering is robust; absolutes ran ~3× hot** on a CPU-contended node (`fuse`/`save` are CPU-bound numpy), so trust the production episode total ~103 s, not these per-stage seconds:

| Stage | rel. | Used in training? |
|---|---:|---|
| **`fuse_scene_cloud`** (rebuild 3-cam point cloud, per chunk) | **~75%** | yes (→ `wrist_context_fused`) |
| `wrist_splat` | ~11% | yes |
| `fixed_mesh` (2 fixed cams) | ~11% | only agentview kept |
| `depth_reproject` (single-view) | small | **no — `wrist_context_single`, debug** |
| `wrist_mesh` (gripper) | small | yes |
| save (npz-compress + `headline.mp4`) | heavy | partly debug |

Encoder reads only `agentview_frames`, `wrist_frames`, `agentview_context_robot`, `wrist_context_fused`, `joints`, `gripper` (verified in `encode_libero_precomputed.py`). Everything else rendered/saved is unused.

**Active-learning implication:** per candidate episode ≈ **~200 s**, of which **~80% is context building** (render ~103 + encode ~57); the model is only ~40 s. Speed work belongs in the build, not inference. Priority levers: (1) **`fuse_scene_cloud`** — caches the static-scene cloud across chunks / subsample points; (2) pure wins — drop `depth_reproject`/`wrist_context_single`, stop writing `headline.mp4`+`clean_plate`, use uncompressed `np.savez`; (3) frontview is not a training view but its RGB-D feeds the wrist fused cloud, so dropping it is a speed↔wrist-fill tradeoff (A/B it); (4) raise VAE encode `--batch_size`; (5) shard candidates across GPUs (build is embarrassingly parallel).

**Update (same day) — `fuse` bottleneck root-caused and fixed.** The "`fuse_scene_cloud` ~75%" number was misleading: the geometry is ~90 ms; the cost was **repeated zlib decompression**. `render()` read the intermediate with `np.load` (a lazy `NpzFile` that does *not* cache), so each `d[key][s]` re-inflated the full array — ~100× per episode of CPU-bound zlib, which ballooned under node contention (hence 10 s on a clean node, 84 s on a busy one). Two edits in `build_libero_context.py`: intermediate `np.savez_compressed`→`np.savez` (scratch, no point compressing), and `np.load(...)`→`dict(np.load(...))` to materialize once. Measured on a clean H200, render-only on a cached capture npz: **`fuse` 9.99 s → 0.09 s (~110×)**, render-compute 35.1 s → 26.2 s. Materializing exposed a **latent aliasing bug**: `_label()` (debug `headline.mp4`) did `np.ascontiguousarray(img)` — a no-op on an already-contiguous `gt[t]` view — then `cv2.putText` drew the label *in place*, corrupting the shared source frames; harmless under the old copy-per-access `NpzFile`, but it poisoned the one overlapping chunk (start 167 over 165) once arrays persisted. Fixed `_label` to copy (`np.array(img, dtype=np.uint8)`). After the fix, output is **bit-identical** to baseline across 139/140 arrays; the single residual diff is **1 pixel ±1** on the discarded `frontview`, proven to be EGL rasterization nondeterminism (two identical-code runs differ identically). `save` (compressed final per-chunk `data.npz`, ~19.7 s) is now the largest render-pass term and the next lever. Next bottleneck after `fuse` is now `wrist_splat`/`fixed_mesh` (~11–12 s each).

**Safe-first production trim (`GVT_DEBUG`, default off).** Gated everything the downstream chain never reads behind `GVT_DEBUG=1`: `encode_libero_precomputed` reads only agentview+wrist `frames`/`context`, and `build_libero_cameras` only `VIEWS=("agentview","wrist")` `K`/`cam2world`. So in production we now (a) skip rendering+saving `frontview` entirely — it stays a *capture/fuse source* (read from `d` inside `fuse_scene_cloud`) but its robot-mesh render and arrays are dropped; (b) skip `depth_reproject`/`wrist_context_single` (comparison-only); (c) skip `*_clean_plate` save; (d) skip `headline.mp4`. Render-only A/B on the same cached capture npz, clean H200: `fixed_mesh` 11.74→**5.86 s** (frontview half gone), `depth_reproject` 1.61→**0 s**, `save` 19.61→**8.24 s** (no debug arrays + no mp4), render_compute_total 26.18→**19.01 s**; whole render pass **~46.5 s → ~28 s (~40% faster)**. Disk **13.3 MB → 7.0 MB per chunk** (~47%). Verified: the 13 downstream-consumed array types are **bit-for-bit identical** to baseline across all 7 chunks (91 instances); only debug keys dropped. `GVT_DEBUG=1` restores the full baseline key set + `headline.mp4`. `wrist_splat` (~11.8 s) is now the #1 render cost — the GPU splat path (`render_wrist_scene_splat_gpu`) is the next lever, then `save`.

---

## 2026-05-27 — LIBERO context data builder (sim-exact cameras, two priors)

The DROID context pipeline synthesizes a geometric prior the hard way: VGGT to
estimate the wrist mount, FK to propagate it, a Franka mesh render over an
inpainted plate for the fixed views, and a plane-homography warp for the wrist.
LIBERO is a *simulator*, so every per-frame camera pose (and depth) is exact —
the entire calibration burden vanishes. New builder `experiments/libero_sim/
build_libero_context.py` mirrors the DROID per-chunk structure (33-frame windows,
overlap-tail, anchored to each chunk's own t0) for the 3 views agentview +
frontview + robot0_eye_in_hand.

**Two priors per chunk per view, for comparison** (chosen because sim uniquely
gives both exact poses *and* exact depth):
- `*_context_homography` — reuses the DROID `render_wrist_via_plane_homography`
  kernel, fed sim's exact cam→world at t0 (source) and frame-t (target). Fixed
  cams → near-static; wrist → warped moving prior. Plane anchored at the wrist
  t0 depth median (LIBERO table z ≈ 0.900).
- `*_context_depth` — forward 3D reprojection of the chunk-t0 frame using the
  exact t0 depth buffer + exact cameras, z-buffered scatter (near wins).

**What the comparison shows** (KITCHEN_SCENE1 demo_0, 189 frames → 6 chunks):
depth-reproj is near-perfect on the fixed cams (coverage 0.96–0.99, MAE-vs-GT
~3.6, i.e. only the moving arm differs) and geometrically correct on the wrist
(coverage 0.76; holes are honest disocclusion as the camera sweeps). The plane
homography is fine on the table plane but warps off-plane content (walls, tall
objects) wrong/black — frontview MAE 26 vs depth's 3.7. Depth reprojection is
clearly the better scaffold; homography is the cheap fallback.

Cameras (per-frame cam→world + K per view) are saved alongside, so these chunks
double as the `cameras/<id>.pt` source for PRoPE (order [agentview, frontview,
wrist] = [ext1, ext2, wrist]). Output: `experiments/droid_action_cond/outputs/
libero_context_chunks/<id>/chunk_NN/{data.npz, info.json, headline.mp4}`.
headline.mp4 stacks GT | homography | depth across the 3 views.

Prereq: the LIBERO render fix (robosuite EGL byref patch + author asset-path
localizer) from the same day — without it the env can't render at all.

---

## 2026-05-27 — PRoPE fix: 3D RoPE ⊕ projective (not a RoPE replacement)

The PRoPE port wired in on 2026-05-19 was wrong in a way that would have silently degraded training. It did 2D `(x, y)` spatial RoPE that **replaced** the model's native 3D RoPE — discarding the temporal axis entirely. PRoPE is supposed to *add* a camera-relative term, not overwrite position encoding. The PRoPE branch in `attention.py` never called `apply_rotary_emb` at all.

### The correct construction

Split each attention head's `head_dim=128` into two **disjoint** sub-spaces, run block-diagonally:

- `[0 : d_rope)` → native 3D RoPE (time, height, width), **untouched**
- `[d_rope : 128)` → per-token projective camera transform (PRoPE), `d_rope = head_dim - proj_dim`

Disjoint is mandatory — mixing the projective block into the RoPE block corrupts both relative-pose identities. The projective block applies `Pᵀ` to Q, `P⁻¹` to K/V, `P` to the output, where `P = lift(K_norm) @ viewmat` (image←world); the attention logit becomes `qᵀ(P_a P_b⁻¹)k`, i.e. relative camera geometry. With `proj_dim = 0` (or identity cameras) the module is bit-identical to vanilla 3D RoPE.

A subtlety drove the design: LTX's split-style RoPE pairs dim `i` with `i+64`, so slicing the *global* `pe=(cos,sin)` to a `d_rope` sub-block is messy. Resolved by having `PropeAttention` build its **own** `d_rope`-sized 3D RoPE directly from `positions` — self-contained, no global `pe` plumbing changes.

### Per-token, not per-view

PRoPE operates on `viewmats (B,T,4,4)` / `Ks (B,T,3,3)` — one camera per **token**. The dataloader expands the 3 physical views (ext1, ext2, wrist) to per-token cameras by tile membership: `_build_per_token_cameras()` maps each token to its quadrant via `positions` (TL=ext1, TR=ext2, BR=wrist, BL=blank) and scatters the matching camera. Blank-tile and action tokens get an identity camera (`P = I`, no-op).

### Data flow now complete on both ends

- **Model:** `positions` plumbed `TransformerArgs` → block → `Attention.forward` → `PropeAttention`. `Attention` constructor takes `prope_proj_dim` (replaces the old `prope_patches_x/y`), `prope_image_width/height`, `prope_max_pos`, `prope_theta`.
- **Dataloader:** `get_data_sources()` adds a `cameras` source when `use_prope`; contract `cameras/<id>.pt = {viewmats: (V,4,4), Ks: (V,3,3)}`, order `[ext1, ext2, wrist]`, `Ks` in per-tile pixel resolution.

### Verification (CPU property + integration tests, all passing)

`packages/ltx-core/tests/`, `packages/ltx-trainer/tests/`:
1. Identity cameras → exact partial 3D RoPE (err 0) — projective block is a true no-op when `P=I`.
2. Temporal structure live — distinct per-token times change output (diff 1.3); uniform time shift cancels (relative encoding), proving 3D RoPE survives. (The first "temporal" test failed at diff 1e-12 because I shifted *all* tokens uniformly — that correctly cancels; not a kernel bug.)
3. Gauge invariance — global world-frame change leaves logits unchanged (rel err 6e-8 no-K, 1.3e-7 with-K). Required switching to a *relative* error threshold + analytic `_invert_SE3`; the absolute error was pure float roundoff scaled by translation magnitude (confirmed by ~10× drop when shrinking translations 5×).
4. Fallback — `use_prope=True` with no cameras == vanilla RoPE (err 0), so flipping the flag is safe before cameras flow.
5. Per-token camera builder — quadrant→view mapping exact ({ext1:4, ext2:4, wrist:4, blank:4} on a 4×4 grid); action+blank → identity; missing `cameras` → `(None, None)`.

### What's left

Data generation: the dataset build (`chunk_context` / `build_dataset_resumable`) does not yet emit `cameras/<id>.pt`. Derivable from `K1, K2, K_wrist`, `cam2base_1/2`, `T_cam_to_hand` + per-frame FK. A per-frame wrist camera (vs. static at chunk-t=0) is a high-value refinement. Then a Q/K LoRA to absorb the positional-encoding shift.

---

## 2026-05-20 — Full-pipeline 4-rung ladder + universe enumeration

The orchestrator from 2026-05-19 only knew how to do the *second half* (tile → process_dataset → atomic commit) over a hand-staged set of pre-rendered chunks. With ~16 chunks staged it looked like a 16-chunk universe; the watchdog was happily reporting "done." It wasn't doing the whole DROID set.

Now the orchestrator runs the **full** pipeline per chunk:

| Rung | Subprocess | Produces | Atomicity |
|---|---|---|---|
| A — extract | `wrist_demo.py --mode extract` (TF) | `outputs/context/<ep>/data.npz` + info.json | per-ep tmp dir → `_atomic_replace_dir` |
| A' — augment | `chunk_context.py --mode augment` (TF) | adds `cmd_gripper_position` to data.npz | in-place; recovered next launch if interrupted |
| B — VGGT calibrate | `calibrate_wrist_vggt.py` (Torch) | `outputs/wrist_calib_vggt/<ep>.json` | per-ep tmp JSON → `os.replace` |
| C — chunk render | `chunk_context.py --mode render` (EGL+pyrender) | `outputs/context_chunks_wrist_new/<ep>/chunk_NN/` | per-chunk tmp dir → `_atomic_replace_dir` |
| D — tile + encode | (as before) `process_dataset.py` → atomic commit on cedar | unchanged |

Each rung is a *separate subprocess*: TF + EGL cannot share a process; VGGT wants its own torch ctx; the trainer hogs the GPU. That isolation is now structural rather than incidental.

### Universe manifest (`_universe.json`)

`enumerate_universe.py` is the new single source of truth for "what chunks exist." Reads droid_100 RLDS (currently the only on-disk DROID source), filters by IoU ≥ 0.85 calibration quality + wrist-capable cameras, splits each surviving episode into 33-frame windows (overlap-tail), and atomically writes the manifest to `out_root/_universe.json`. Watchdog reads `len(_universe.json)` every tick so adding more episodes (or lowering the IoU floor) doesn't require restarting the watchdog.

Current scale: **71 chunks across 12 episodes** at IoU ≥ 0.85 (vs the 16 chunks / 3 episodes that the old hand-staged set was building).

### New env-var hooks (so the orchestrator can stage each rung into a tmp dir)

- `WRIST_EXTRACT_OUT` — redirects `wrist_demo.py`'s episode-level output root.
- `VGGT_EPS_FILTER` — comma-separated short-id whitelist for `calibrate_wrist_vggt.py`.
- (existing hooks `CHUNK_EPS` / `CHUNK_OUT_ROOT` from chunk_context.py reused for rungs A'+C.)

### `_atomic_replace_dir` — directory-level rename that doesn't trash a good dir before the new one lands

```python
backup = dst.with_name(f".{dst.name}.bak.{pid}")
os.replace(dst, backup)         # move existing aside (atomic)
try:    os.replace(src, dst)    # swap in new (atomic)
except: os.replace(backup, dst); raise   # rollback if rename fails
shutil.rmtree(backup, ignore_errors=True)
```

Same-FS guarantee: every staging dir is colocated with its destination's parent.

### Per-rung "is this already done?" predicates

The orchestrator never re-runs work whose output is already on disk:

- `_episode_extracted(ep_short)` — `data.npz` exists AND has both extract keys *and* the augment-pass-added `cmd_gripper_position`.
- `_vggt_done(ep_short)` — calib JSON loads and has `T_cam_to_hand`.
- `_chunk_rendered(ep_short, chunk_idx)` — chunk's `data.npz` has all 8 frame arrays + cmd_* + intrinsics.

`ensure_upstream_for(pending)` runs A→A'→B→C and returns the subset of `pending` whose rung-C output now exists. Anything that failed gets retried next launch (probably on a different node).

### Smoke test (single chunk, A100 40GB)

`--only_cids AUTOLab_84bd5053_..._c00 --batch_size 1 --skip_upstream --load_text_encoder_in_8bit` → **229 s wall, 1/1 committed, no staging leftovers, all 4 .pt files load and have the expected shapes.** Initial run on a V100 16GB OOMed at LTX-2.3 load — V100 is fine for the calibration/render rungs but not for D. The sbatch's `gpu-v100,gpu-a100,gpu-h100` partition list lets slurm route the job onto a big-enough GPU.

The `OSError: Device or resource busy` lines that show up at the end of each process_dataset stage are multiprocessing pool finalizers tearing down `pymp-*` tmp dirs on cedar after the stage already wrote its outputs successfully — noise, not failure.

### Watchdog re-armed

Tick 1 after the smoke: `Progress: 17/71 chunks done. [MISS] gvt_data not in queue — submitting one sbatch.` Submitted `8739748`. The full DROID-100/IoU-0.85 subset is now being built unattended.

---

## 2026-05-19 — Dataset build pipeline: tiled v2v IC-LoRA + resumable orchestrator + Slurm watchdog

End-to-end preprocessed dataset pipeline for the GVT trainer. Goal: take the per-chunk `data.npz` files we already render (`outputs/context_chunks_wrist_new/<ep>/chunk_NN/data.npz`), produce trainer-ready `latents/conditions/reference_latents/actions` on disk, and survive Slurm preemption arbitrarily many times without data corruption.

### Three scripts landed

| File | Role |
|---|---|
| `experiments/droid_action_cond/build_v2v_dataset.py` | One-shot builder for a hand-picked chunk list. Lanczos-upscales each view to 512×288, tiles `[ext1 \| ext2 / blank \| wrist]` into a 1024×576 MP4 pair (target + reference), dumps actions, then invokes the trainer's `process_dataset.py` to VAE+Gemma encode, plus the conditions-fix-up move. Used for the single-chunk + single-episode smoke tests. |
| `experiments/droid_action_cond/build_dataset_resumable.py` | The production orchestrator. Same per-chunk work but wrapped in: atomic tmp+rename writes, `_done/<id>` marker files, hash-sharded parallelism, corruption detection on resume, automatic Slurm-array shard inference. |
| `experiments/droid_action_cond/build_dataset_resumable.sbatch` | Slurm wrapper. Targets `gpu-v100,gpu-a100,gpu-h100` (preprocessing fits in ~10 GB with 8-bit Gemma — no L40S needed), `--qos=embers --requeue` for preemption, log dir at `/storage/project/.../outputs/gvt/logs/`. |
| `experiments/droid_action_cond/monitor_dataset_build.sh` | Watchdog. Every 5 min checks `squeue` for the build job; if missing, resubmits the array. Auto-exits when `_done/` count matches total chunk count. |

### Why the atomic + marker design

Slurm preemption can hit at any instruction. The contract: **a chunk is either fully committed or not committed at all** — never half. The pattern:

1. Build target.mp4 + reference.mp4 + the action `.pt` in a staging dir (`staging/batch_<id>/`). All file writes use `_atomic_tmp_path` (same-dir hidden file, real extension preserved) + `os.replace` (POSIX atomic rename on same FS).
2. Invoke `process_dataset.py` on the staging dataset.json → produces VAE-encoded `.pt` files in `staging/.../precomputed/`.
3. Verify each chunk's 4 staged `.pt` files: exists, size > 1024 bytes, **and `torch.load` succeeds** (catches truncation that POSIX rename can't).
4. Atomic-rename each of the 4 files into final `precomputed/<source>/videos/<id>_target.pt`.
5. Last operation: `touch _done/<id>`. Only after this does any consumer treat the chunk as done.

If we crash anywhere in 1–4, the next launch sees no marker → enumerates the chunk as pending → `clean_partial_outputs()` wipes leftovers → rebuilds.

### Tripwire: the MP4 atomic-write extension trap

First version of `_atomic_tmp_path` used `path.with_suffix(path.suffix + ".tmp.PID")`, yielding `foo.mp4.tmp.12345`. Imageio's plugin dispatcher sniffs the trailing extension to pick a codec — for `.12345` it falls through to the TIFF plugin, which then explodes on the `fps` kwarg. The 2-chunk smoke surfaced this immediately. Fix: rotate the suffix order to `.foo.tmp.12345.mp4` so `.mp4` stays last and the dispatch is correct.

### Corruption recovery — explicitly tested

We injected two failure modes into a freshly-committed test directory:

1. Truncated a committed `latents/.../<id>.pt` to `"garbage"` (8 bytes) while leaving its marker in place.
2. Removed an `<id>`'s marker plus its conditions+actions `.pt` files but left the latents+reference files (= what a kill-mid-commit produces).

On the next run:
- For (1), `chunk_is_done()` calls `_output_looks_valid()` which does a `torch.load` and catches the `UnpicklingError` → logs `[REVIVE]` → wipes the marker and all 4 partials → next batch rebuilds, post-commit verifies all 4 `.pt` files load with the correct shapes.
- For (2), the marker-absent state enumerates the chunk as pending. Before the rebuild's renames, `clean_partial_outputs()` removes the leftover `.pt` files so partials never coexist with the fresh writes.

Verified: both chunks come back to a fully-loadable state with the right tensor shapes (`latents: (128, 5, 18, 32)`, `conditions: (1024, 4096)`, `actions: (33, 8)`).

### Slurm parallelism via hash-shard

`build_dataset_resumable.py --shard i/N` selects chunks where `sha1(chunk_id)[:8] % N == i`. No coordination between workers — atomic file writes + per-chunk markers mean two workers can't corrupt each other's output even if they ever raced on the same chunk (which the hash-mod assignment prevents anyway). The sbatch auto-detects `SLURM_ARRAY_TASK_ID/COUNT` so `sbatch --array=0-7` spawns 8 stable shards with zero hand-holding.

### Timing (CLVR episode, 6 chunks, warm cache)

| Stage | Wall time | Per-chunk |
|---|---|---|
| Tile MP4s + action dump + dataset.json | 7.8 s | 1.3 s |
| `process_dataset.py` (VAE encode 12 videos + Gemma embed 6 captions) | 29.9 s | 5.0 s |
| **Total** | **37.7 s** | **6.3 s** |

Cold-load first-run on a fresh node: ~5–8 min for the model + VAE shards off `cedar0` (paid once per Slurm job, not per episode).

### GPU sizing — preprocessing does NOT need L40S

Process_dataset.py loads Gemma + LTX VAE only (not the 22B transformer — that's training-only). With `--load_text_encoder_in_8bit` the budget drops to **~10 GB**, well within V100-32G / A10-24G / A100-40G. The sbatch's partition list `gpu-v100,gpu-a100,gpu-h100` is intentional: short queues, lots of slots. Training still wants ≥32 GB and we'll target L40S there.

### Disk and dataset scale

The per-chunk artifact for our 1024×576×33 tile layout breaks down as:

| Source | Size per chunk | What it is |
|---|---|---|
| `latents/<id>_target.pt` | ~740 KB | `(128, 5, 18, 32)` bf16 |
| `reference_latents/<id>_target.pt` | ~740 KB | same |
| `conditions/<id>_target.pt` | **~8 MB** | Gemma `(1024, 4096)` bf16 + masks + audio embeds |
| `actions/<id>_target.pt` | ~1 KB | `(33, 8)` fp32 |
| **Total per chunk** | **~9 MB** | |

So an episode of 6 chunks lands at ~55 MB. CLVR built today is 83 MB on disk including the raw source MP4s (which are debug-only and can be dropped to save ~10 MB/episode).

The conditions file is the disk-bloat outlier: it's the same Gemma embedding for every chunk of a given episode, redundantly stored. A caption-dedup post-process (hash-keyed symlinks) would cut this from 8 MB × N chunks to 8 MB × N unique captions — ~85% disk reduction at scale. Documented as a follow-up, not implemented today.

### Outputs

- `/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/gvt_dataset_full/` (default sbatch out_root, intended for the real build) — empty until first sbatch run.
- `experiments/droid_action_cond/outputs/v2v_dataset_single/` — 1-chunk reference dataset, fully VAE-encoded for inspection.
- `experiments/droid_action_cond/outputs/v2v_dataset_clvr/` — full CLVR-episode (6-chunk) dataset, with the warm-cache timing measurement.

### DROID scale framing (what's actually addressable)

The HF mirror at `/storage/project/r-agarg35-0/lwang831/droid_hf/` has the **full** DROID — 92,233 episodes, 27M frames, ~500 hours. DROID-100 is the curated quality subset. For our method the real-world constraint is per-episode VGGT wrist calibration (~30 s GPU each), not raw data availability:

| Tier | Episodes | Chunks | Calibration cost |
|---|---|---|---|
| Today | 5 | 40 | done |
| DROID-100 full | 100 | ~800 | ~1 hr more VGGT |
| 10% of DROID | ~9k | ~80k | ~75 hr single-GPU, ~10 hr on 8 GPUs |
| All DROID | ~92k | ~750k | ~30 days single-GPU, ~4 days on 8 GPUs |

PRoPE is depth-free — only warp-latent needs depth, and depth comes free as a byproduct of VGGT (the model returns per-image depth maps alongside the cameras; we currently throw them away in `calibrate_wrist_vggt_v2.py`). Capturing depth needs ~10 LoC of new save-output code, not a new model.

### What's left (NOT done in this entry)

- **DROID-HF parquet → per-episode `data.npz`** — `chunk_context.py` currently works on already-extracted episodes. Extending coverage to fresh DROID episodes needs the extract step. Existing helpers in `packages/ltx-action-cond/src/ltx_action_cond/droid.py` work against TF/RLDS, not the LeRobot parquet format on disk.
- **Cameras + depth dumps** for PRoPE/warp-latent (items #6, #8–10 from the punch list).
- **Caption dedup** to cut conditions/ disk by 85%.
- **Strategy-side wiring** to read `batch["cameras"]` and `batch["depths"]` once they exist.

### Wiring state after this entry

- `experiments/droid_action_cond/build_v2v_dataset.py` — new, one-shot builder used for smoke verification.
- `experiments/droid_action_cond/build_dataset_resumable.py` — new, production orchestrator.
- `experiments/droid_action_cond/build_dataset_resumable.sbatch` — new, Slurm wrapper.
- `experiments/droid_action_cond/monitor_dataset_build.sh` — new, watchdog.

---

## 2026-05-19 — PRoPE + Warp Latent + Action-Cond wired through the trainer

Three opt-in conditioning paths landed on the trainer side, all verified loading the stock LTX-2.3 22B distilled checkpoint on an L40S. The kernels for two of these (PRoPE attention, raw-action MLP projector) had been sitting in `ltx-core` from earlier commits — wiring them into `ltx-trainer` was the missing piece. The third (MosaicMem-style warp-latent) is new code.

### What landed

| Feature | Flag (`acceleration.*`) | New params | Verified on L40S |
|---|---|---|---|
| **PRoPE** — Cameras as Relative Positional Encoding | `use_prope: true` + `prope_patches_x/y` + `prope_image_width/height` *(superseded 2026-05-27 → `prope_proj_dim`; that port replaced 3D RoPE — see entry above)* | **0** (pure geometry; checkpoint loads byte-identical) | ✅ 48 blocks swapped, 28.60 GB peak, 0.83 s/step |
| **Warp Latent** — MosaicMem Eq. 2 cross-view feature alignment | `use_warp_latent: true` | 0 | ✅ 28.59 GB peak, 0.83 s/step |
| **Action-Cond** — `ActionMLPProjector` + token concat | `use_action_cond: true` + `action_dim: 8` | **4,512** (8 → 128 MLP, trains from scratch in the LoRA fine-tune) | ✅ 28.60 GB peak, 0.84 s/step |
| **All three + IC-LoRA simultaneously** | all `true` | 4,512 | ✅ Loss=0.1519 after 2 steps, LoRA saves clean |

Baseline (no flags) for comparison: 28.59 GB peak, 0.83 s/step, Loss=0.1495. All flags-on adds the projector params and shifts step time by ~1%.

### Architecture choices worth recording

**Post-hoc PRoPE enablement** (`ltx_core/model/transformer/enable_prope.py`) — instead of threading `use_prope` through `SingleGPUModelBuilder` → `LTXModelConfigurator` → `LTXModel.__init__` → `_init_transformer_blocks` (5 layers of config plumbing), we let the stock model load first, then walk `model.transformer_blocks` and swap each `attn1` for a PRoPE-enabled `Attention` carrying byte-identical weights. PRoPE adds zero learnable params, so `new.load_state_dict(old.state_dict(), strict=False)` is exact for every key except the PRoPE module's internal precomputed RoPE buffers (filtered with `strict=False`). This keeps the configurator stack untouched.

**Action projector lives outside the transformer** — `ActionMLPProjector(action_dim → 128)` is a separate `nn.Module` on the trainer, not a submodule of the transformer. Its params are added to `_trainable_params` explicitly. The projector outputs `patchify_proj.in_features` (== 128, the VAE channel count), **not** `inner_dim` (4096): the action tokens get concat'd onto the pre-`patchify_proj` sequence so they ride the same lift-to-hidden-dim path as the video latents. First combined-smoke pass failed with `Expected size 128 but got size 4096` — that was the lesson.

**Modality + TransformerArgs got two new optional fields**: `viewmats: (B, C, 4, 4)` and `Ks: (B, C, 3, 3)`. Both default `None`; `TransformerArgsPreprocessor.prepare()` propagates them; the video `attn1` call now passes them through. When `viewmats is None` the `Attention` layer's existing guard auto-falls back to vanilla RoPE, so `use_prope=True` is safe to set even before the dataloader carries cameras.

**Warp-latent** (`ltx_core/conditioning/warp_latent.py`) implements MosaicMem Eq. 2 — per-pixel `(u', v') = Π(K_tgt T_tgt T_src⁻¹ K_src⁻¹ (u, v, D))`, then `F.grid_sample(mode='bilinear')`. Differentiable, no learnable params. Identity cameras + flat depth produce the identity transform (smoke-verified). One bf16 quirk: `torch.linalg.inv` doesn't support bf16, so the geometry runs in fp32 and only the sampling grid is cast back to the source dtype before `grid_sample`.

### Scope honesty

All three features are **wired** but the smoke runs them in their *degenerate* mode:

- PRoPE: `viewmats=None` → auto-fallback to vanilla path.
- Warp: identity `(K, T)` + flat depth → identity output.
- Action-cond: actions are read and projected, but they're just (cmd_joints, cmd_gripper) from `data.npz` with no validation of whether the model actually learns from them in 2 steps.

The verification proves: (a) the stock 22B checkpoint loads cleanly with all flags on, (b) all four code paths execute without shape mismatches or NaN, (c) LoRA fine-tune saves and resumes. It does **not** prove the features improve generation. That needs (i) real per-tile cameras for PRoPE, (ii) Depth Anything V3 maps for warp-latent, and (iii) longer training runs with held-out evaluation.

### What's left to make each feature actually compute

| Feature | What it needs from preprocessing |
|---|---|
| PRoPE | `cameras/<id>.pt = {viewmats: (4, 4, 4), Ks: (4, 3, 3)}` per chunk — directly derivable from `data.npz`'s `K1, K2, K_wrist, cam2base_1, cam2base_2, T_cam_to_hand` plus FK at chunk-t=0 |
| Warp Latent | `depths/<id>.pt = {wrist_t0, ext1_t0, ext2_t0: (288, 512)}` per chunk — needs a DAv3 pass at chunk anchor frames (~0.2 s/chunk on L40S) |
| Action-Cond | `actions/<id>.pt = {latents: (33, 8)}` per chunk — trivial dump from `data.npz` (already prototyped for the smoke chunks) |

Trainer-side glue once those land: extend `VideoToVideoStrategy.get_data_sources()` to include `"cameras"` and `"depths"`, then in `prepare_training_inputs` replace the `batch.get("viewmats")` / identity-depth placeholders with `batch["cameras"]` and `batch["depths"]`. No further model-internals changes.

### Wiring state after this entry

- `packages/ltx-core/src/ltx_core/model/transformer/modality.py` — `Modality` gained `viewmats`, `Ks` fields.
- `packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py` — `TransformerArgs` mirrors them; `TransformerArgsPreprocessor.prepare()` propagates.
- `packages/ltx-core/src/ltx_core/model/transformer/transformer.py` — `attn1(...)` call passes `viewmats=video.viewmats, Ks=video.Ks`.
- `packages/ltx-core/src/ltx_core/model/transformer/enable_prope.py` — **new**. Post-hoc PRoPE swap helper.
- `packages/ltx-core/src/ltx_core/conditioning/warp_latent.py` — **new**. MosaicMem Eq. 2 implementation.
- `packages/ltx-trainer/src/ltx_trainer/config.py` — `AccelerationConfig` gained `use_prope`, `prope_patches_x/y` *(→ `prope_proj_dim`, 2026-05-27)*, `prope_image_width/height`, `use_warp_latent`, `use_action_cond`, `action_dim`.
- `packages/ltx-trainer/src/ltx_trainer/trainer.py` — owns the `ActionMLPProjector`, calls `enable_prope_on_model` after the dtype cast and before quantization, adds projector params to `_trainable_params`, prepares the projector under accelerate.
- `packages/ltx-trainer/src/ltx_trainer/training_strategies/video_to_video.py` — adds `"actions"` to `get_data_sources()` when action-cond is on; in `prepare_training_inputs` reads `batch["actions"]`, resamples to `F_latent`, projects, and concats onto `combined_latents` with clean timesteps + sentinel spatial positions + zero loss-mask. Warp-latent path branches on `use_warp_latent` and runs on the ref latents. PRoPE viewmats/Ks read from `batch` (None for now). `compute_loss` slices the target window by `inputs.video_targets.shape[1]` so action tokens don't break MSE shape.

---

## 2026-05-18 — Robotiq 2F-85 gripper + gripper-aware wrist context

Two related rendering fixes after side-by-side eyeballing on the chunked outputs. The first replaced the wrong end-effector model; the second fixed a wrist-context bug that treated the gripper as a world-frame object.

### Robotiq 2F-85 swap

DROID does not use the standard Franka Hand on most episodes — the bulk of the dataset uses a **Robotiq 2F-85** parallel-jaw mounted on the Franka flange via an adapter plate. DROID's only mount spec is `T_hand = identity` at the flange with CoM `[0, 0, 0.057]`; the flange-to-2F-85-base offset isn't published, so we set it manually.

- New `packages/ltx-action-cond/src/ltx_action_cond/robotiq_kinematics.py`: hardcoded URDF tree from [a-price/robotiq_arg85_description](https://github.com/a-price/robotiq_arg85_description). 9 links, 6 unique STL meshes (knuckle/finger pairs share meshes). Drive joint `finger_joint ∈ [0, 0.725 rad]`; mimic relationships encoded in the joint table (axis sign + multiplier). `robotiq_link_T(theta)` returns per-link 4×4 transforms in the Robotiq base frame.
- `FrankaMeshRenderer` gained `gripper: Literal["franka_hand", "robotiq_2f85"]` plus `robotiq_mount_yaw` and `robotiq_mount_z_offset`. Default yaw is `1.5π` — the value that matched GT after a brief two-iteration calibration loop on CLVR ext1/ext2. Default color is `(0.01, 0.01, 0.01)` per the URDF spec (real Robotiq is black plastic).
- DROID's `gripper_position` is normalized `[0=open, 1=closed]`; mapped via `gripper_position_to_theta(g) = g · 0.725`.

Coverage: the same mount and color work across all 5 chunked episodes (IPRL, CLVR, AUTOLab×2, TRI). No per-episode tuning.

### Wrist context: the gripper was being treated as a world-frame object

After the gripper swap, the wrist-camera context exposed a separate bug. The old pipeline took the raw GT `wrist0_img` — which *contains the gripper* sitting in the lower-middle of the frame — and warped the whole thing via a plane homography to the per-frame wrist pose. The gripper pixels got dragged across the image like table texture: the gripper visually "floats" in world coordinates instead of staying glued to the camera.

The physical fact this missed: the wrist camera and the gripper share a **rigid frame**. When the fingers aren't moving, the gripper's pixel-space appearance is *literally invariant* — only finger open/close ever changes it.

New pipeline, opt-in via `WRIST_CONTEXT_MODE=gripper_aware` (now the default; `simple` falls back to the legacy single-warp path):

1. **Once per chunk** — render a gripper-only mesh at the chunk-t=0 wrist camera pose → gripper mask. TELEA-inpaint the gripper out of `wrist0_img` → `clean_bg_wrist0` (gripper-free scene plate).
2. **Per frame** — warp `clean_bg_wrist0` via plane homography to frame `t`'s wrist pose (no gripper present, so nothing smears). Then:
   - **`|Δgripper| < 0.03`** (common case — DROID chunks hold `cmd_g ≈ 1.00` or `0.00` for many consecutive frames): paste the real `wrist0_img` gripper pixels back at the same pixel location using `mask_0`. Zero render cost, perfect photorealism.
   - **else**: render the Robotiq mesh at the per-frame wrist pose with the current `cmd_gripper[t]` and composite over the plate.

A new `gripper_only=True` flag on `FrankaMeshRenderer.render()` skips the arm links — they're mostly out of the wrist FOV and would look wrong against the homography-warped plate.

The legacy single-warp path is kept (not deleted) per user instruction — both modes are accessible.

### Cost budget

- Per chunk: one extra wrist render at t=0 + one TELEA inpaint. Amortised over 33 frames, negligible.
- Per frame: an `np.where` for the common branch, or one extra mesh render for the rare finger-motion branch.
- Total render count for a typical chunk goes from 66 (2 ext renders × 33 frames) to ~67–70.

### EGL detour worth remembering

Mid-iteration on PACE the renderer started failing with `GL_OUT_OF_MEMORY` on the first `glBufferData` — GPU empty (0 MiB used, 81 GB free), but every one of the 10 EGL devices enumerated by `pyrender.platforms.egl.query_devices()` failed identically. Cause: Slurm allocated GPU index 6 (`SLURM_STEP_GPUS=6`), cgroup-restricting access to one specific `/dev/dri/renderDN`. CUDA honors the visibility mask; pyrender's `eglGetDisplay(EGL_DEFAULT_DISPLAY)` does not — it enumerates all host DRI nodes and picks one we can't allocate against. Error 1285 is what NVIDIA's EGL driver returns in that case, *not* actual memory exhaustion.

Tried `EGL_DEVICE_ID`, explicit vendor pinning, and sweeping every EGL device in the pyrender backend — all failed. The fix was to get a different node. Resubmitted to an L40S, EGL came up clean, render finished in minutes.

Lesson: on multi-GPU HPC nodes, EGL `GL_OUT_OF_MEMORY` with empty VRAM is almost always a Slurm-cgroup / EGL-default-display mismatch, not a real allocation failure. Re-allocate before debugging deeper.

### Outputs

- `experiments/droid_action_cond/outputs/context_chunks/` — 40 chunks (5 episodes), Robotiq + legacy wrist mode.
- `experiments/droid_action_cond/outputs/context_chunks_wrist_new/` — 16 chunks (IPRL + CLVR + TRI), Robotiq + new gripper-aware wrist mode. `TRI+52ca9b6a/chunk_06` is the best stress test — `cmd_g` transitions 1.00→0.00 so both branches fire.

### Wiring state after this entry

- `packages/ltx-action-cond/src/ltx_action_cond/robotiq_kinematics.py` — new module.
- `packages/ltx-action-cond/src/ltx_action_cond/mesh_rendering.py:render()` — `gripper_only` flag added; `gripper` / `robotiq_mount_*` ctor args.
- `experiments/droid_action_cond/chunk_context.py` — `WRIST_CONTEXT_MODE` env (`gripper_aware` default, `simple` for legacy).

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
