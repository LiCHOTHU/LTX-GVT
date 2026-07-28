#!/usr/bin/env bash
# infer_libero.sbatch — action-conditioned (or ablated) inference for the GVT
# v2v IC-LoRA model on a precomputed LIBERO chunk. Renders pred|gt|ctx panels to
# experiments/libero_sim/outputs/. H100 (80 GB) holds the bf16 22B transformer.
#
# Override knobs via --export:
#   LORA   path to lora_weights_step_*.safetensors (default: step_19000)
#   IDX    chunk index (default 0)
#   STEPS  denoising steps (default 30)
#   EXTRA  extra flags, e.g. "--no-actions" or "--action-projector <path>"
#   OUT    output mp4

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH -p gpu-h100
#SBATCH --time=00:40:00
#SBATCH --mem=96G
#SBATCH --job-name=lib_inf
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.err

set -euo pipefail

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT"
CKPTS="/storage/project/r-agarg35-0/lwang831/outputs/gvt/libero90_v2v_ic_lora/checkpoints"
LORA="${LORA:-${CKPTS}/lora_weights_step_19000.safetensors}"
IDX="${IDX:-0}"
STEPS="${STEPS:-30}"
EXTRA="${EXTRA:---no-actions}"
OUT="${OUT:-${WORKDIR}/experiments/libero_sim/outputs/infer_step19000.mp4}"

cd "${WORKDIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ltx

export TMPDIR="/storage/project/r-agarg35-0/lwang831/tmp"
unset HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export MPLCONFIGDIR="${TMPDIR}/mpl"
export PYTHONPATH="${WORKDIR}/packages/ltx-trainer/src:${WORKDIR}/packages/ltx-core/src:${PYTHONPATH:-}"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}" "$(dirname "${OUT}")"

echo "============== LIBERO inference =============="
echo "JobID : ${SLURM_JOB_ID:-N/A} | Node: ${SLURM_NODELIST:-N/A}"
echo "LoRA  : ${LORA}"
echo "Chunk : index ${IDX} | steps ${STEPS} | extra ${EXTRA}"
echo "Out   : ${OUT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u experiments/libero_sim/infer_action_cond.py \
    --lora-path "${LORA}" \
    --index "${IDX}" \
    --num-inference-steps "${STEPS}" \
    --include-gt --include-context \
    --output "${OUT}" \
    ${EXTRA}

echo "Done at $(date)"
