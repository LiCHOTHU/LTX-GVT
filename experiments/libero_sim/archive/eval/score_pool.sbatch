#!/usr/bin/env bash
# score_pool.sbatch — score a candidate POOL with one checkpoint, producing the
# per-episode latent-loss ranking the active-learning selector consumes.
# Thin GPU wrapper around score_chunk_latent_loss.py. Parametrised via --export:
#   LORA   (required)  lora_weights_step_*.safetensors of the arm's latest ckpt
#   PROJ   (required)  action_projector_step_*.safetensors beside it
#   DATA   (required)  pool precomputed/ root to score
#   OUT    (required)  output latent_loss json path
#   USE_PROPE (default 1), SIGMA_GRID, SEEDS  optional overrides
# This is an EVAL job (no training). The active-learning watchdog submits it between
# rounds for the strategic arm only.
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH -p gpu-h200
#SBATCH --time=01:30:00
#SBATCH --mem-per-gpu=128G
#SBATCH --job-name=al_score
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --requeue
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.err

set -euo pipefail
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT"
LSIM="${WORKDIR}/experiments/libero_sim"
USER_LIBERO="/storage/home/hcoda1/8/lwang831/workspace/LIBERO"

: "${LORA:?set LORA=lora_weights_step_*.safetensors}"
: "${PROJ:?set PROJ=action_projector_step_*.safetensors}"
: "${DATA:?set DATA=pool precomputed/ root}"
: "${OUT:?set OUT=output latent_loss.json path}"
USE_PROPE="${USE_PROPE:-1}"
SIGMA_GRID="${SIGMA_GRID:-0.2,0.4,0.6,0.8}"
SEEDS="${SEEDS:-0,1,2}"

cd "${WORKDIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ltx
export PYTHONPATH="${USER_LIBERO}:${LSIM}:${WORKDIR}/packages/ltx-trainer/src:${WORKDIR}/packages/ltx-core/src:${PYTHONPATH:-}"
export TMPDIR="/storage/project/r-agarg35-0/lwang831/tmp"
export MPLCONFIGDIR="${TMPDIR}/mpl"
unset HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}" "$(dirname "${OUT}")"

PROPE_FLAGS=""
[[ "${USE_PROPE}" == "1" ]] && PROPE_FLAGS="--use-prope --prope-proj-dim 64 --prope-image-width 256 --prope-image-height 256"

echo "=== score_pool: $(basename "${LORA}") on ${DATA} -> ${OUT} ==="
echo "Start: $(date)"
# shellcheck disable=SC2086
python -u "${LSIM}/score_chunk_latent_loss.py" \
    --lora-path "${LORA}" --action-projector "${PROJ}" \
    --data-root "${DATA}" ${PROPE_FLAGS} \
    --sigma-grid "${SIGMA_GRID}" --seeds "${SEEDS}" \
    --out "${OUT}"
echo "=== DONE $(date) ==="
