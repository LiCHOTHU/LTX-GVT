#!/usr/bin/env bash
# Run render_select_pairs on the local interactive H100 (no SLURM batch, no 2h cap).
set -euo pipefail
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT"
LSIM="${WORKDIR}/experiments/libero_sim"
USER_LIBERO="/storage/home/hcoda1/8/lwang831/workspace/LIBERO"

ARM="${ARM:-strategic}"
ROUNDS="${ROUNDS:-0-8}"
CK="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/outputs/gvt_alv2/libero90_ALR_${ARM}/checkpoints"
STATE="/storage/project/r-agarg35-0/lwang831/alr_state_v2/alr_${ARM}"
POOL_TMPL="/storage/scratch1/8/lwang831/alv2/libero90_ALR_${ARM}_round{r}/precomputed"
OUT="${LSIM}/outputs/al_select_pairs_${ARM}"

cd "${WORKDIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ltx
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${USER_LIBERO}:${LSIM}:${WORKDIR}/packages/ltx-trainer/src:${WORKDIR}/packages/ltx-core/src:${PYTHONPATH:-}"
export TMPDIR="/storage/project/r-agarg35-0/lwang831/tmp"
export MPLCONFIGDIR="${TMPDIR}/mpl"
unset HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${OUT}" "${TMPDIR}" "${MPLCONFIGDIR}"

echo "=== render select/reject pairs | arm=${ARM} rounds=${ROUNDS} | LOCAL H100 ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Start: $(date)"

python -u "${LSIM}/render_select_pairs.py" \
    --base-checkpoint "/storage/scratch1/8/lwang831/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors" \
    --ckpt-dir "${CK}" \
    --state-dir "${STATE}" \
    --pool-template "${POOL_TMPL}" \
    --rounds "${ROUNDS}" \
    --use-prope --prope-proj-dim 64 --prope-image-width 256 --prope-image-height 256 \
    --num-inference-steps 30 --drop-first 8 \
    --sigma-grid 0.2,0.4,0.6,0.8 --seeds 0,1,2 \
    --out-dir "${OUT}"

echo "=== DONE $(date) ==="
