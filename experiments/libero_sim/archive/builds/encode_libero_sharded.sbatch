#!/usr/bin/env bash
# Parallel (sharded) LIBERO-90 PREPROCESSING — array-job variant that encodes the
# raw context chunks into the trainer's precomputed/ schema (VAE latents of the
# target + reference 2x2 tiles, Gemma caption embeds, action tensors). Each array
# task owns a disjoint, stable-hash slice of the ~22.5k chunks
# (sha1(cid) %% num_shards == shard), so N tasks encode N chunks at once and the
# single-worker wall time drops ~N-fold.
#
# Submit with an array sized to the shard count, e.g. 8 parallel workers:
#     sbatch --array=0-7 encode_libero_sharded.sbatch
# The shard identity + count are derived from the array automatically
# (SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT). Per-task logs use %A_%a.
#
# RESUME / SAFETY: _done/<cid> markers are global and shards are disjoint, so two
# tasks never encode the same chunk. A preempted task relaunches (--requeue) and
# skips finished chunks. Use monitor_libero_encode.sh to keep the array alive
# across preemption / the 8 h walltime (a single shard's ~2.8k chunks can exceed
# one window).
#
# Memory: process_dataset.py loads the LTX-2.3 VAE + Gemma-3-12B (8-bit). An L40S
# (48 GB) holds both comfortably; that's what we target.

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=128G
#SBATCH --job-name=lib_enc
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --requeue
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%A_%a.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%A_%a.err

set -euo pipefail

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT"
ENCODE="${WORKDIR}/experiments/libero_sim/encode_libero_precomputed.py"
CONDA_ENV="ltx"

# Raw context (input) + precomputed (output) both live on scratch1, siblings of
# each other (auto-purges ~60 days — move the finished precomputed/ somewhere
# durable before training at scale).
SRC_ROOT="${SRC_ROOT:-/storage/scratch1/8/lwang831/libero90_context}"
OUT_ROOT="${OUT_ROOT:-/storage/scratch1/8/lwang831/libero90_precomputed}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_CHUNKS="${MAX_CHUNKS:-0}"   # 0 = all chunks in this shard

# Shard identity from the array task; count from the array size (overridable).
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
if [[ -z "${NUM_SHARDS:-}" ]]; then
    if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
        NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT}"
    elif [[ -n "${SLURM_ARRAY_TASK_MAX:-}" && -n "${SLURM_ARRAY_TASK_MIN:-}" ]]; then
        NUM_SHARDS=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
    else
        NUM_SHARDS=1
    fi
fi

cd "${WORKDIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Env hygiene mirrors the dataset builder + finetune-verify sbatch.
export TMPDIR="/storage/project/r-agarg35-0/lwang831/tmp"
unset HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export MPLCONFIGDIR="${TMPDIR}/mpl"
export PYTHONPATH="${WORKDIR}/packages/ltx-trainer/src:${PYTHONPATH:-}"
mkdir -p "${TMPDIR}" "${HF_HOME}" "${MPLCONFIGDIR}" "${OUT_ROOT}" \
         /storage/project/r-agarg35-0/lwang831/outputs/gvt/logs

echo "================ LIBERO preprocess (sharded) ================"
echo "JobID    : ${SLURM_ARRAY_JOB_ID:-N/A}_${SLURM_ARRAY_TASK_ID:-N/A} | Node: ${SLURM_NODELIST:-N/A}"
echo "Shard    : ${SHARD} / ${NUM_SHARDS}"
echo "SrcRoot  : ${SRC_ROOT}"
echo "OutRoot  : ${OUT_ROOT}"
echo "Start    : $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
python -V

python -u "${ENCODE}" \
    --src_root "${SRC_ROOT}" \
    --out_root "${OUT_ROOT}" \
    --shard "${SHARD}/${NUM_SHARDS}" \
    --batch_size "${BATCH_SIZE}" \
    --max_chunks "${MAX_CHUNKS}" \
    --load_text_encoder_in_8bit

echo "Finished at $(date)"
