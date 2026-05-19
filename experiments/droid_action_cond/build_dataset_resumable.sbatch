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
# Memory budget: ~10 GB GPU with --load_text_encoder_in_8bit (Gemma 8-bit + VAE bf16).
# V100/A100/H100 partitions all comfortably fit; no need to pin to L40S.

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-v100,gpu-a100,gpu-h100
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=gvt_data
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.err

set -euo pipefail

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT"
SCRIPT="${WORKDIR}/experiments/droid_action_cond/build_dataset_resumable.py"
CONDA_ENV="ltx"

SRC_ROOT="${SRC_ROOT:-${WORKDIR}/experiments/droid_action_cond/outputs/context_chunks_wrist_new}"
OUT_ROOT="${OUT_ROOT:-/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/gvt_dataset_full}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_CHUNKS="${MAX_CHUNKS:-0}"   # 0 = no limit

mkdir -p "$(dirname "${SLURM_SUBMIT_DIR:-$WORKDIR}/logs")" /storage/project/r-agarg35-0/lwang831/outputs/gvt/logs

cd "${WORKDIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Per the memory note: use /storage/project tmpdir, not /tmp (limited size + EBUSY surprises).
export TMPDIR="/storage/project/r-agarg35-0/lwang831/tmp"
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export MPLCONFIGDIR="${TMPDIR}/mpl"
mkdir -p "${TMPDIR}" "${HF_HOME}" "${MPLCONFIGDIR}"

echo "================ Job Info ================"
echo "JobID       : ${SLURM_JOB_ID:-N/A}"
echo "Node        : ${SLURM_NODELIST:-N/A}"
echo "SrcRoot     : ${SRC_ROOT}"
echo "OutRoot     : ${OUT_ROOT}"
echo "BatchSize   : ${BATCH_SIZE}"
echo "MaxChunks   : ${MAX_CHUNKS}"
echo "Start       : $(date)"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

python -V
which python

python -u "${SCRIPT}" \
    --src_root "${SRC_ROOT}" \
    --out_root "${OUT_ROOT}" \
    --batch_size "${BATCH_SIZE}" \
    --max_chunks "${MAX_CHUNKS}" \
    --load_text_encoder_in_8bit

echo "Finished at $(date)"
