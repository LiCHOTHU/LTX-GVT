#!/usr/bin/env bash
# Resumable v2v IC-LoRA dataset builder for GVT / DROID.
#
# Single-worker. Use monitor_dataset_build.sh to keep one instance alive across
# preemption — the orchestrator skips chunks that already have _done/<id> markers,
# so each relaunch picks up where it left off.
#
# Preemption-safe: --requeue + per-chunk atomic markers in the Python orchestrator
# mean a killed job re-runs only the in-flight batch on restart.
#
# Memory budget: Phase D needs ~16+ GB to load Gemma-3-12B (8-bit) + LTX-2.3 VAE.
# V100 16GB has been empirically observed to OOM during Gemma load (the 8-bit
# model + working set just barely overflows 16GB). A100 40GB and H100 80GB are
# safe; that's what we restrict to. (Phases A-C don't need much GPU but live in
# the same job, so they get an over-provisioned card. Acceptable tradeoff.)

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# L40S (48 GB) has plenty of memory for Phase D's Gemma-3-12B 8-bit encoder.
# Uses the dedicated ideas_l40s account so it doesn't compete with A100 queue.
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=128G
#SBATCH --job-name=gvt_data
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --requeue
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.err

set -euo pipefail

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT"
# Streaming per-episode builder. Replaces the batched 4-phase pipeline (which
# OOM'd at full-DROID scale). One process loads VGGT+VAE+Gemma+pyrender once,
# walks TFDS once, and commits _done markers chunk-by-chunk per episode.
SCRIPT="${WORKDIR}/experiments/droid_action_cond/build_streaming.py"
CONDA_ENV="ltx"

# Built data is WRITTEN to scratch1 (cedar0 hit 100% full). The DROID RLDS source
# (DROID_RLDS_DIR) + model checkpoints stay on cedar as read-only inputs.
OUT_ROOT="${OUT_ROOT:-/storage/scratch1/8/lwang831/gvt_dataset_full}"
MAX_CHUNKS="${MAX_CHUNKS:-0}"   # 0 = no limit
MAX_EPS="${MAX_EPS:-0}"         # 0 = run until walltime/preempt

# Source RLDS: full DROID-1.0.1 by default. The watchdog passes this through.
# droid.py / enumerate_universe.py both honour this env var.
export DROID_RLDS_DIR="${DROID_RLDS_DIR:-/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/droid_full/1.0.1}"

mkdir -p "$(dirname "${SLURM_SUBMIT_DIR:-$WORKDIR}/logs")" /storage/project/r-agarg35-0/lwang831/outputs/gvt/logs

cd "${WORKDIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Per the memory note: use /storage/project tmpdir, not /tmp (limited size + EBUSY surprises).
export TMPDIR="/storage/project/r-agarg35-0/lwang831/tmp"
# Scrub the .bashrc-leaked HF cache pointers BEFORE re-exporting. .bashrc has
# `HF_HOME=${XDG_CACHE_HOME}/huggingface`; on compute nodes XDG_CACHE_HOME is
# unset, so the var expands to "/huggingface" (unwritable). Same for the
# *_CACHE siblings. Unset, then re-export to a writable project path.
unset HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export MPLCONFIGDIR="${TMPDIR}/mpl"
# pyrender headless on GPU nodes (rung 3 of the ladder).
export PYOPENGL_PLATFORM=egl
# Push VGGT to GPU so its weights don't sit in CPU RAM competing with Gemma's
# CPU-side load staging (~12 GB transient during Phase D model load).
export VGGT_DEVICE="${VGGT_DEVICE:-cuda}"
mkdir -p "${TMPDIR}" "${HF_HOME}" "${MPLCONFIGDIR}"

echo "================ Job Info ================"
echo "JobID         : ${SLURM_JOB_ID:-N/A}"
echo "Node          : ${SLURM_NODELIST:-N/A}"
echo "OutRoot       : ${OUT_ROOT}"
echo "DROID_RLDS_DIR: ${DROID_RLDS_DIR}"
echo "MaxEps        : ${MAX_EPS}"
echo "MaxChunks     : ${MAX_CHUNKS}"
echo "HF_HOME       : ${HF_HOME}"
echo "HF_HUB_CACHE  : ${HF_HUB_CACHE}"
echo "TMPDIR        : ${TMPDIR}"
echo "Start         : $(date)"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

python -V
which python

# Stamp-file guard: refuse to run if the universe wasn't enumerated against
# the same RLDS dir we're about to load from. Without this check the builder
# would happily look up ep_index values that map to *different* episodes in
# the wrong dataset, silently producing garbage. The CPU enumeration sbatch
# writes this stamp on success. The watchdog (monitor_dataset_build.sh)
# routes Phase 1 there before falling through to this Phase 2.
STAMP="${OUT_ROOT}/_universe.rlds_root"
if [[ ! -f "${OUT_ROOT}/_universe.json" ]] || [[ ! -f "${STAMP}" ]] \
   || [[ "$(cat "${STAMP}" 2>/dev/null)" != "${DROID_RLDS_DIR}" ]]; then
    echo "[error] _universe.json is missing or stale (RLDS stamp mismatch)." >&2
    echo "        expected stamp: ${DROID_RLDS_DIR}" >&2
    echo "        actual stamp  : $(cat "${STAMP}" 2>/dev/null || echo '<missing>')" >&2
    echo "        Use monitor_dataset_build.sh to bootstrap the universe via the" >&2
    echo "        CPU enumeration sbatch, then this build sbatch will pick up." >&2
    exit 2
fi

python -u "${SCRIPT}" \
    --out_root "${OUT_ROOT}" \
    --max_chunks "${MAX_CHUNKS}" \
    --max_eps "${MAX_EPS}" \
    --load_text_encoder_in_8bit

echo "Finished at $(date)"
