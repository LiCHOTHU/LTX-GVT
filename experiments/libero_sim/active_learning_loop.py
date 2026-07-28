#!/usr/bin/env python
"""active_learning_loop.py — the WHOLE closed-loop active-learning run, in ONE Python
process, on ONE GPU.

This replaces the bash orchestration (the old monitor_active_replay_libero.sh stage
machine + the do_round() body of run_active_replay_single.sbatch). The loop control,
config generation, selection bookkeeping and resume logic all live here in Python; the
two heavyweight stages (BUILD and TRAIN) are still launched as subprocesses because each
is its own standalone program (the build pipeline spins up the LIBERO sim; train.py runs
under accelerate with its own model load) — they cannot become in-process function calls,
but they are driven from this one readable loop.

For each round r = 0 .. MAX_ROUNDS-1 and each version (strategic / random):
    1. BUILD  ~40 practice clips   (gen_libero_play -> build_all -> encode -> cameras)
    2. SCORE  them (strategic only) (score_chunk_latent_loss -> per-clip "how wrong")
    3. KEEP   20 (strategic=hardest, random=random) -> growing train set + replay buffer
    4. TRAIN  +ROUND_STEPS steps    (resumes from the version's latest checkpoint)
Start = step 30000, end = step 30000 + MAX_ROUNDS*ROUND_STEPS = 80000.

RESUME: every finished stage drops a marker on disk (_build_done, latent_loss_r{r}.json,
_selected_r{r}); training auto-resumes from its last checkpoint. So a kill at any point
(preemption / walltime) loses < ROUND_STEPS steps. The thin keepalive monitor
(monitor_active_learning.sh) just resubmits the sbatch; this loop picks up where it left
off. There is NO self-resubmit and NO child-job coordination here.

This script does ZERO orchestration magic — read it top to bottom and it is the algorithm.

Env knobs (all optional; defaults match the prior bash):
    VERSIONS, START_STEP, ROUND_STEPS, MAX_ROUNDS, SELECT_K, BUFFER_CAP, SEED_PER_TASK,
    PER_TASK_MIX, NUM_SHARDS, RANDOM_SEED, SIGMA_GRID, SEEDS, RES, FPS, GEN_STEPS,
    OUT_BASE, START_CKPT_DIR, TEMPLATE_CONFIG, LIBERO90, BASE_ROOT, GROW_BASE, STATE_BASE,
    DRY_RUN (1 = print every command + file op, run nothing heavy).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

# --------------------------------------------------------------------------- paths
REPO = Path("/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT")
LSIM = REPO / "experiments" / "libero_sim"
TRAIN_PY = REPO / "packages" / "ltx-trainer" / "scripts" / "train.py"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


OUT_BASE = Path(_env("OUT_BASE", "/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/outputs/gvt"))
START_CKPT_DIR = Path(_env("START_CKPT_DIR", str(OUT_BASE / "libero90_v2v_ic_lora_prope" / "checkpoints")))
TEMPLATE_CONFIG = Path(_env("TEMPLATE_CONFIG", str(REPO / "packages/ltx-trainer/configs/ltx2_v2v_ic_lora_libero_prope.yaml")))
LIBERO90 = Path(_env("LIBERO90", "/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/LIBERO-datasets/libero_90"))
BASE_ROOT = _env("BASE_ROOT", "/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/libero90_combined_precomputed/precomputed")
GROW_BASE = Path(_env("GROW_BASE", "/storage/scratch1/8/lwang831"))
STATE_BASE = Path(_env("STATE_BASE", "/storage/project/r-agarg35-0/lwang831/alr_state"))

# --------------------------------------------------------------------------- knobs
VERSIONS = _env("VERSIONS", "strategic random").split()
START_STEP = int(_env("START_STEP", "30000"))
ROUND_STEPS = int(_env("ROUND_STEPS", "1000"))
MAX_ROUNDS = int(_env("MAX_ROUNDS", "50"))            # 30000 + 50*1000 = 80000
SELECT_K = int(_env("SELECT_K", "20"))                # keep half of ~40 candidates
BUFFER_CAP = int(_env("BUFFER_CAP", "270"))
SEED_PER_TASK = int(_env("SEED_PER_TASK", "3"))
PER_TASK_MIX = _env("PER_TASK_MIX", "reach:7,perturb:10,random:3")
NUM_SHARDS = int(_env("NUM_SHARDS", "45"))
RANDOM_SEED = int(_env("RANDOM_SEED", "20260615"))
SIGMA_GRID = _env("SIGMA_GRID", "0.2,0.4,0.6,0.8")
SEEDS = _env("SEEDS", "0,1,2")
RES = int(_env("RES", "256"))
FPS = int(_env("FPS", "15"))
GEN_STEPS = int(_env("GEN_STEPS", "200"))
DRY_RUN = _env("DRY_RUN", "0") != "0"

# --- v2 signal refinements (post lit-review). All OFF by default so a requeue of the
# --- legacy run is byte-identical; a FRESH launch sets AL_V2=1 to enable them. ---
AL_V2 = _env("AL_V2", "0") != "0"
SCORE_METRIC = _env("SCORE_METRIC", "progress_norm")  # used only when AL_V2 (#2/#3)
DIVERSE = _env("DIVERSE", "1" if AL_V2 else "0") != "0"  # k-center keep rule (#1)
OVERSAMPLE = int(_env("OVERSAMPLE", "0"))               # 0 => 3*SELECT_K
BASELINE_LAG = int(_env("BASELINE_LAG", "3"))           # rounds back for learning-progress baseline (#2)
ENSEMBLE_LAGS = _env("ENSEMBLE_LAGS", "")               # e.g. "1,2,3" rounds back for disagreement (#4)

# --- object-centric pixel signal (single-scene experiment). SCORER=objbbox replaces the
# --- whole-frame latent loss with model pixel error measured ONLY inside the projected
# --- object 3D boxes (so wrist look-around doesn't dominate). Needs the objbbox build step. ---
SCORER = _env("SCORER", "latent")                       # "latent" | "objbbox"
GRADE_STEPS = int(_env("GRADE_STEPS", "20"))            # rollout steps for the objbbox pixel scorer

# --- 4-arm ablation (obj-single). Reducible = learning-progress vs a FROZEN reference ckpt
# --- (BASELINE_STEP, default the step-30000 start). Pool made arm-INDEPENDENT via GT-seeded
# --- perturb so all arms see byte-identical candidates (POOL_PERTURB_SOURCE=gt). ---
REDUCIBLE_METRIC = _env("REDUCIBLE_METRIC", "progress_norm")  # progress | progress_norm
BASELINE_STEP = int(_env("BASELINE_STEP", "30000"))          # frozen reference checkpoint step
POOL_PERTURB_SOURCE = _env("POOL_PERTURB_SOURCE", "buffer")  # "gt" (arm-independent) | "buffer" (legacy)
POLICY_REL_LAMBDA = float(_env("POLICY_REL_LAMBDA", "2.0"))  # decaware occupancy decay

# Each arm = (mode, metric, baseline, diverse, policy_rel). The arm key IS the version string
# (arm_out/state_dir/etc. key on it), so VERSIONS carries the arm. Legacy strategic/random are
# preserved EXACTLY (their do_round branch is unchanged) so the old 2-arm run stays byte-identical.
ARM_CONFIGS = {
    "strategic": {"mode": "strategic"},   # legacy — handled by the untouched branch
    "random":    {"mode": "random"},      # control — handled by the untouched branch
    "maxloss":   {"mode": "strategic", "metric": "loss",           "baseline": False, "diverse": False, "policy_rel": False},
    "rho_div":   {"mode": "strategic", "metric": REDUCIBLE_METRIC, "baseline": True,  "diverse": True,  "policy_rel": False},
    "decaware":  {"mode": "strategic", "metric": REDUCIBLE_METRIC, "baseline": True,  "diverse": True,  "policy_rel": True},
}

FINAL_STEP = START_STEP + MAX_ROUNDS * ROUND_STEPS


def seed_salt(v: str) -> int:
    """Per-arm salt so multiple random arms (random, random_b, ...) draw DIFFERENT seeds
    (#5 — beats the single-seed 'statistical wash'). 'random' keeps salt 0 == legacy."""
    return 0 if v == "random" else sum(ord(c) for c in v) * 1009


# --------------------------------------------------------------------------- helpers
def say(*a: object) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def run(cmd: list[str]) -> None:
    """Run a stage subprocess (the heavy work). Echo always; execute unless DRY_RUN."""
    say("RUN:", " ".join(str(c) for c in cmd))
    if DRY_RUN:
        return
    subprocess.run([str(c) for c in cmd], check=True)


def arm_out(v: str) -> Path:
    return OUT_BASE / f"libero90_ALR_{v}"


def add_root(v: str) -> Path:
    return GROW_BASE / f"libero90_ALR_{v}_add" / "precomputed"


def state_dir(v: str) -> Path:
    return STATE_BASE / f"alr_{v}"


def round_dir(v: str, r: int) -> Path:
    return GROW_BASE / f"libero90_ALR_{v}_round{r}"


def latest_step(out: Path) -> int:
    """Highest checkpoint step under an arm's output dir (0 if none)."""
    steps = [
        int(m.group(1))
        for p in out.rglob("lora_weights_step_*.safetensors")
        if (m := re.search(r"step_(\d+)", p.name))
    ]
    return max(steps) if steps else 0


def find_one(root: Path, name: str) -> str | None:
    for p in root.rglob(name):
        return str(p)
    return None


# Fast scratch1-staged base model (transformer + gemma) with cedar0 cold-storage fallback.
# The template config points at scratch1; but if staging hasn't finished (the `.staged_ok`
# marker is written only after BOTH files are fully copied) we fall back to cedar0 so a
# rerun is always safe and simply cold-loads from cedar0 until the staged copy is ready.
_STAGED_OK = Path("/storage/scratch1/8/lwang831/LTX-2.3/.staged_ok")
_STAGED_MODEL = "/storage/scratch1/8/lwang831/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
_STAGED_TENC = "/storage/scratch1/8/lwang831/gemma-3-12b-it-qat-q4_0-unquantized"
_CEDAR_MODEL = "/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
_CEDAR_TENC = "/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/gemma-3-12b-it-qat-q4_0-unquantized"


def base_model_paths() -> tuple[str, str]:
    """(model_path, text_encoder_path).

    Text encoder ALWAYS loads from the durable cedar0 copy: scratch1 auto-purges small files
    (tokenizer.model) by access time, which silently crash-loops every arm at "Loading text
    encoder..." while the `.staged_ok` marker still claims it is staged. Cedar is slower to
    cold-load but can't be purged out from under a run. The big transformer checkpoint still
    prefers the fast scratch1 copy when it is actually present on disk (not just marked)."""
    model_path = _STAGED_MODEL if (_STAGED_OK.exists() and Path(_STAGED_MODEL).exists()) else _CEDAR_MODEL
    return model_path, _CEDAR_TENC


def write_config(al_out: Path, add: Path, steps: int, cfg_out: Path) -> None:
    """Per-round training config: clone the template, point it at this arm's checkpoint +
    its [BASE_ROOT, additions] data roots, and set the step target. Real YAML edit (no sed):
    the multi-root PrecomputedDataset concatenates the two preprocessed_data_root entries."""
    cfg = yaml.safe_load(TEMPLATE_CONFIG.read_text())
    cfg["model"]["load_checkpoint"] = str(al_out)
    model_path, tenc_path = base_model_paths()
    cfg["model"]["model_path"] = model_path
    cfg["model"]["text_encoder_path"] = tenc_path
    cfg["data"]["preprocessed_data_root"] = [BASE_ROOT, str(add)]
    cfg["optimization"]["steps"] = steps
    cfg["output_dir"] = str(al_out)
    if DRY_RUN:
        say(f"WRITE config -> {cfg_out} (load_checkpoint={al_out}, steps={steps}, roots=[BASE, {add}])")
        return
    cfg_out.write_text(yaml.safe_dump(cfg, sort_keys=False))


# --------------------------------------------------------------------------- stages
def init_version(v: str) -> None:
    """Idempotent: symlink the step-30000 start checkpoint into the arm's output dir and
    seed the replay buffer. Safe to re-enter on resume."""
    al_out, add, state = arm_out(v), add_root(v), state_dir(v)
    buf = state / "buffer.json"
    for d in (al_out / "checkpoints", al_out / "rounds", add, state):
        if not DRY_RUN:
            d.mkdir(parents=True, exist_ok=True)
    for suf in (
        f"lora_weights_step_{START_STEP}.safetensors",
        f"action_projector_step_{START_STEP}.safetensors",
        f"training_state_step_{START_STEP}.pt",
    ):
        src = START_CKPT_DIR / suf
        dst = al_out / "checkpoints" / suf
        if src.exists() and not dst.exists():
            say(f"[{v}] ln -s {src} -> {dst}")
            if not DRY_RUN:
                dst.symlink_to(src)
    if not buf.exists():
        say(f"[{v}] seed replay buffer ({SEED_PER_TASK}/task)")
        run([sys.executable, "-u", LSIM / "al_replay_buffer.py", "seed",
             "--libero90", LIBERO90, "--manifest", buf,
             "--per-task", SEED_PER_TASK, "--seed", RANDOM_SEED])
    added = state / "added.txt"
    if not added.exists() and not DRY_RUN:
        added.touch()


def do_round(v: str, r: int) -> None:
    """One round for one version: build -> (grade) -> keep+buffer -> train. Fully resumable."""
    al_out, add, state = arm_out(v), add_root(v), state_dir(v)
    buf = state / "buffer.json"
    manifest = state / "added.txt"
    rd = round_dir(v, r)
    play, ctx, pre = rd / "play", rd / "context", rd / "precomputed"
    json_path = state / f"latent_loss_r{r}.json"
    picked = state / f"picked_r{r}.txt"
    target = START_STEP + (r + 1) * ROUND_STEPS

    step = latest_step(al_out)
    if step >= target:
        say(f"[{v}] r{r} already at step {step} >= {target}; skip.")
        return

    # ---- 1) make ~40 practice clips ----
    # Each sub-step has its OWN done-marker so a preemption mid-build resumes efficiently:
    # if the action/play trajectories were already sampled (1a) we do NOT re-sample, and
    # likewise we skip already-rendered (1b) / already-encoded (1c) / already-camera'd (1d)
    # work. Only the single sub-step that was interrupted re-runs.
    if not (pre / "_build_done").exists():
        shard = r % NUM_SHARDS
        say(f"[{v}] r{r} BUILD candidates (shard {shard}/{NUM_SHARDS}, perturb<-buffer)")
        if not DRY_RUN:
            for d in (play, ctx, pre):
                d.mkdir(parents=True, exist_ok=True)
        # 1a) sample action/play trajectories (policy rollouts) — skip if already sampled.
        # POOL_PERTURB_SOURCE=gt seeds perturb off the GT demo (arm-INDEPENDENT -> identical
        # pools across arms, the clean-ablation guarantee); "buffer" is the legacy per-arm path.
        if not (play / "_play_done").exists():
            psrc = ["--perturb-source", str(buf)] if (POOL_PERTURB_SOURCE == "buffer" and buf.exists()) else []
            run([sys.executable, "-u", LSIM / "gen_libero_play.py",
                 "--dataset", LIBERO90, "--out_dir", play,
                 "--mix", PER_TASK_MIX, "--steps", GEN_STEPS,
                 "--seed", RANDOM_SEED + 7919 * r,
                 "--num_shards", NUM_SHARDS, "--shard", shard, *psrc])
            if not DRY_RUN:
                (play / "_play_done").touch()
        else:
            say(f"[{v}] r{r} BUILD 1a skip (play already sampled)")
        # 1b) render context videos from the sampled trajectories — skip if already rendered
        if not (ctx / "_ctx_done").exists():
            run([sys.executable, "-u", LSIM / "build_all_libero.py",
                 "--dataset", play, "--out_root", ctx, "--res", RES, "--fps", FPS, "--max_demos", 0])
            if not DRY_RUN:
                (ctx / "_ctx_done").touch()
        else:
            say(f"[{v}] r{r} BUILD 1b skip (context already rendered)")
        # 1c) VAE/text encode -> precomputed (also per-chunk resumable internally) — skip if done
        if not (pre / "_encode_done").exists():
            run([sys.executable, "-u", LSIM / "encode_libero_precomputed.py",
                 "--src_root", ctx, "--out_root", rd, "--shard", "0/1",
                 "--batch_size", 4, "--max_chunks", 0, "--load_text_encoder_in_8bit"])
            if not DRY_RUN:
                (pre / "_encode_done").touch()
        else:
            say(f"[{v}] r{r} BUILD 1c skip (latents already encoded)")
        # 1d) camera / PRoPE tensors — skip if done
        if not (pre / "_cameras_done").exists():
            run([sys.executable, "-u", LSIM / "build_libero_cameras.py",
                 "--src_root", ctx, "--out_root", rd, "--shard", "0/1", "--max_chunks", 0])
            if not DRY_RUN:
                (pre / "_cameras_done").touch()
        else:
            say(f"[{v}] r{r} BUILD 1d skip (cameras already built)")
        # 1e) object 3D-bbox corners — only the objbbox pixel scorer needs them — skip if done
        if SCORER == "objbbox" and not (pre / "_objbbox_done").exists():
            run([sys.executable, "-u", LSIM / "build_libero_objbbox.py",
                 "--src_root", ctx, "--out_root", rd, "--shard", "0/1", "--max_chunks", 0])
            if not DRY_RUN:
                (pre / "_objbbox_done").touch()
        if not DRY_RUN:
            (pre / "_build_done").touch()

    # ---- 2) grade (strategic / smart version only) ----
    if v == "strategic" and not json_path.exists():
        lora = find_one(al_out, f"lora_weights_step_{step}.safetensors")
        proj = find_one(al_out, f"action_projector_step_{step}.safetensors")
        say(f"[{v}] r{r} GRADE clips with ckpt step {step} (scorer={SCORER})")
        if SCORER == "objbbox":
            # object-centric pixel curiosity: full rollout + decode, error masked to the
            # projected object boxes. No progress/baseline regime (object masking already
            # removes the 'busy frame != wrong' confound that motivated progress_norm).
            run([sys.executable, "-u", LSIM / "score_chunk_objbbox_pixel.py",
                 "--lora-path", lora, "--action-projector", proj, "--data-root", pre,
                 "--use-prope", "--prope-proj-dim", 64,
                 "--prope-image-width", 256, "--prope-image-height", 256,
                 "--num-inference-steps", GRADE_STEPS, "--out", json_path])
        else:
            grade_extra: list = []
            if AL_V2:
                grade_extra += ["--score-metric", SCORE_METRIC]
                rounds = al_out / "rounds"
                # learning-progress baseline: a checkpoint BASELINE_LAG rounds back (#2). Both the
                # LoRA and its action projector are archived under rounds/ (see round 4 below).
                bstep = step - BASELINE_LAG * ROUND_STEPS
                bl = find_one(rounds, f"lora_weights_step_{bstep}.safetensors")
                bp = find_one(rounds, f"action_projector_step_{bstep}.safetensors")
                if bl and bp:
                    grade_extra += ["--baseline-lora", bl, "--baseline-projector", bp]
                elif SCORE_METRIC.startswith("progress"):
                    say(f"[{v}] r{r} GRADE: no step-{bstep} baseline yet -> scorer falls back to raw loss")
                # ensemble disagreement: older checkpoints (#4)
                if ENSEMBLE_LAGS:
                    eloras = [e for lag in ENSEMBLE_LAGS.split(",")
                              if (e := find_one(rounds, f"lora_weights_step_{step - int(lag) * ROUND_STEPS}.safetensors"))]
                    if eloras:
                        grade_extra += ["--ensemble-loras", ",".join(eloras)]
            run([sys.executable, "-u", LSIM / "score_chunk_latent_loss.py",
                 "--lora-path", lora, "--action-projector", proj, "--data-root", pre,
                 "--use-prope", "--prope-proj-dim", 64,
                 "--prope-image-width", 256, "--prope-image-height", 256,
                 "--sigma-grid", SIGMA_GRID, "--seeds", SEEDS, "--out", json_path, *grade_extra])

    # ---- 3) keep 20 (strategic=hardest, random=random) + replay buffer ----
    #        (LEGACY 2-arm path only; the 4-arm ablation arms are handled in the block below)
    if v in ("strategic", "random") and not (state / f"_selected_r{r}").exists():
        kind = "hardest" if v == "strategic" else "random"
        say(f"[{v}] r{r} KEEP {SELECT_K} ({kind}) -> train set + replay buffer")
        if v == "strategic":
            sel_extra = (["--diverse"] + (["--oversample", OVERSAMPLE] if OVERSAMPLE > 0 else [])) if DIVERSE else []
            run([sys.executable, "-u", LSIM / "al_select_and_grow.py",
                 "--pool-root", pre, "--grow-root", add, "--manifest", manifest,
                 "--k", SELECT_K, "--mode", "strategic",
                 "--latent-json", json_path, "--picked-out", picked, *sel_extra])
        else:
            run([sys.executable, "-u", LSIM / "al_select_and_grow.py",
                 "--pool-root", pre, "--grow-root", add, "--manifest", manifest,
                 "--k", SELECT_K, "--mode", "random",
                 "--seed", RANDOM_SEED + r + seed_salt(v), "--picked-out", picked])
        run([sys.executable, "-u", LSIM / "al_replay_buffer.py", "append",
             "--manifest", buf, "--kept-dir", play, "--picked", picked,
             "--round", r + 1, "--cap", BUFFER_CAP])
        if not DRY_RUN:
            (state / f"_selected_r{r}").touch()

    # ---- 2'/3') 4-ARM ABLATION arms (maxloss / rho_div / decaware): grade -> [policy-rel] -> keep.
    #            Identical pool + identical decode; the ONLY difference between these arms is the
    #            score metric (raw vs reducible), the batch rule (top-k vs diverse), and policy-rel. ----
    if v in ("maxloss", "rho_div", "decaware"):
        cfg = ARM_CONFIGS[v]
        lora = find_one(al_out, f"lora_weights_step_{step}.safetensors")
        proj = find_one(al_out, f"action_projector_step_{step}.safetensors")
        # grade with the arm's metric (reducible arms subtract the FROZEN reference checkpoint)
        if not json_path.exists():
            base_args: list = []
            if cfg["baseline"]:
                base_args = ["--baseline-lora", START_CKPT_DIR / f"lora_weights_step_{BASELINE_STEP}.safetensors",
                             "--baseline-projector", START_CKPT_DIR / f"action_projector_step_{BASELINE_STEP}.safetensors"]
            say(f"[{v}] r{r} GRADE step {step} scorer={SCORER} metric={cfg['metric']} baseline={cfg['baseline']}")
            if SCORER == "objbbox":
                run([sys.executable, "-u", LSIM / "score_chunk_objbbox_pixel.py",
                     "--lora-path", lora, "--action-projector", proj, "--data-root", pre,
                     "--use-prope", "--prope-proj-dim", 64,
                     "--prope-image-width", 256, "--prope-image-height", 256,
                     "--num-inference-steps", GRADE_STEPS,
                     "--score-metric", cfg["metric"], *base_args, "--out", json_path])
            else:
                extra = ["--score-metric", cfg["metric"], *base_args] if cfg["metric"] != "loss" else []
                run([sys.executable, "-u", LSIM / "score_chunk_latent_loss.py",
                     "--lora-path", lora, "--action-projector", proj, "--data-root", pre,
                     "--use-prope", "--prope-proj-dim", 64,
                     "--prope-image-width", 256, "--prope-image-height", 256,
                     "--sigma-grid", SIGMA_GRID, "--seeds", SEEDS, "--out", json_path, *extra])
        # policy relevance (decaware only): reweight the grade by GT-manifold occupancy
        select_json = json_path
        if cfg["policy_rel"]:
            pr_json = state / f"policy_rel_r{r}.json"
            if not pr_json.exists():
                say(f"[{v}] r{r} POLICY-REL reweight (occupancy x reducible)")
                run([sys.executable, "-u", LSIM / "score_policy_relevance.py",
                     "--grade-json", json_path, "--data-root", pre,
                     "--lambda-occ", POLICY_REL_LAMBDA, "--out", pr_json])
            select_json = pr_json
        # keep SELECT_K (diverse batch if configured) + replay buffer
        if not (state / f"_selected_r{r}").exists():
            say(f"[{v}] r{r} KEEP {SELECT_K} (strategic diverse={cfg['diverse']}) -> train + buffer")
            sel_extra = (["--diverse"] + (["--oversample", OVERSAMPLE] if OVERSAMPLE > 0 else [])) if cfg["diverse"] else []
            run([sys.executable, "-u", LSIM / "al_select_and_grow.py",
                 "--pool-root", pre, "--grow-root", add, "--manifest", manifest,
                 "--k", SELECT_K, "--mode", "strategic",
                 "--latent-json", select_json, "--picked-out", picked, *sel_extra])
            run([sys.executable, "-u", LSIM / "al_replay_buffer.py", "append",
                 "--manifest", buf, "--kept-dir", play, "--picked", picked,
                 "--round", r + 1, "--cap", BUFFER_CAP])
            if not DRY_RUN:
                (state / f"_selected_r{r}").touch()

    # ---- 4) train +ROUND_STEPS (auto-resumes from al_out's latest ckpt) ----
    step = latest_step(al_out)
    if step < target:
        cfg = state / f"config_r{r}.yaml"
        write_config(al_out, add, target, cfg)
        say(f"[{v}] r{r} TRAIN step {step} -> {target}")
        run([sys.executable, "-u", TRAIN_PY, cfg, "--disable-progress-bars"])

    # ---- archive this round's ckpt for the quality-vs-#clips curve, AND so later rounds
    # ---- can use it as a learning-progress baseline / ensemble member (#2/#4). The
    # ---- action projector is archived too because checkpoints/ only keeps the last
    # ---- keep_last_n steps, but a baseline BASELINE_LAG rounds back outlives that window.
    for stem in (f"lora_weights_step_{target}.safetensors", f"action_projector_step_{target}.safetensors"):
        src = find_one(al_out / "checkpoints", stem)
        if src:
            dst = al_out / "rounds" / Path(src).name
            if not dst.exists() and not DRY_RUN:
                shutil.copy2(src, dst)
    say(f"[{v}] r{r} done.")


def all_done() -> bool:
    return all(latest_step(arm_out(v)) >= FINAL_STEP for v in VERSIONS)


# --------------------------------------------------------------------------- main
def main() -> int:
    say("=== single-process active-learning loop (python) ===")
    say(f"versions={VERSIONS}  start={START_STEP}  final={FINAL_STEP}  rounds={MAX_ROUNDS}x{ROUND_STEPS}")
    say(f"scorer={SCORER}  select_k={SELECT_K}  mix={PER_TASK_MIX}  num_shards={NUM_SHARDS}  grade_steps={GRADE_STEPS}")
    say(f"JobID={os.environ.get('SLURM_JOB_ID', '<none>')}  Node={os.environ.get('SLURM_NODELIST', '<none>')}  DRY_RUN={DRY_RUN}")

    for v in VERSIONS:
        init_version(v)

    for r in range(MAX_ROUNDS):
        for v in VERSIONS:
            do_round(v, r)

    if all_done():
        say(f"=== ALL DONE: every version reached step {FINAL_STEP}. Per-round ckpts under <out>/rounds/. ===")
        return 0
    say(f"=== loop returned but not all versions at {FINAL_STEP} (check logs above) ===")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
