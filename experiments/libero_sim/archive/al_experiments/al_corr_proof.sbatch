#!/usr/bin/env bash
# al_corr_proof.sbatch — Active-learning Phase 0: validate the cheap latent-loss signal.
# Stage A (al_pixel_score.py): ONE 22B model load; per chunk compute the cheap latent
#   denoising loss (the AL signal) AND the expensive full-rollout pixel PSNR/SSIM (ground
#   truth "is the model wrong"). Step-30000 PRoPE checkpoint, over the existing OOD set.
# Stage B (al_corr_proof.py): Spearman/Pearson + scatter + worst/best example clips.
# This is an EVAL job (no training).
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH -p gpu-h200
#SBATCH --time=02:00:00
#SBATCH --mem-per-gpu=128G
#SBATCH --job-name=al_corr
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --requeue
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.err

set -euo pipefail
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT"
LSIM="${WORKDIR}/experiments/libero_sim"
USER_LIBERO="/storage/home/hcoda1/8/lwang831/workspace/LIBERO"
STEP=30000
CK="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/outputs/gvt/libero90_v2v_ic_lora_prope/checkpoints"
DATA="/storage/scratch1/8/lwang831/libero90_ood_precomputed/precomputed"
OUT="${LSIM}/outputs/al_corr_proof_step${STEP}"

cd "${WORKDIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ltx
export PYTHONPATH="${USER_LIBERO}:${LSIM}:${WORKDIR}/packages/ltx-trainer/src:${WORKDIR}/packages/ltx-core/src:${PYTHONPATH:-}"
export TMPDIR="/storage/project/r-agarg35-0/lwang831/tmp"
export MPLCONFIGDIR="${TMPDIR}/mpl"
unset HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${OUT}" "${TMPDIR}" "${MPLCONFIGDIR}"

echo "=== AL corr proof (step ${STEP}, PRoPE) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Start: $(date)"

# Stage A — combined per-chunk latent loss + pixel error (one model load).
python -u "${LSIM}/al_pixel_score.py" \
    --lora-path "${CK}/lora_weights_step_${STEP}.safetensors" \
    --action-projector "${CK}/action_projector_step_${STEP}.safetensors" \
    --data-root "${DATA}" \
    --use-prope --prope-proj-dim 64 --prope-image-width 256 --prope-image-height 256 \
    --num-inference-steps 30 --drop-first 8 \
    --sigma-grid 0.2,0.4,0.6,0.8 --seeds 0,1,2 \
    --out-dir "${OUT}"

# Stage B — correlation + scatter + examples (CPU).
python -u "${LSIM}/al_corr_proof.py" --combined "${OUT}/combined.json" --out-dir "${OUT}"

echo "=== DONE $(date) ==="
echo "Deliverables under ${OUT}: combined.json/.csv, corr_report.txt, corr_scatter.png, examples_{high,low}/"
