#!/usr/bin/env bash
# bench_infer_time.sbatch — micro-benchmark the per-chunk INFERENCE split
# (DiT denoise loop vs VAE decode) on the EXACT eval code path, GVT_TIME=1.
# Runs a few single chunks in ONE 22B model load against the already-built
# combined precomputed root. PRoPE mode (matches the running eval).
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH -p gpu-h200
#SBATCH --time=00:40:00
#SBATCH --mem-per-gpu=128G
#SBATCH --job-name=bench_t
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --requeue
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.err

set -euo pipefail
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT"
LSIM="${WORKDIR}/experiments/libero_sim"
USER_LIBERO="/storage/home/hcoda1/8/lwang831/workspace/LIBERO"
DATA="/storage/scratch1/8/lwang831/libero90_alleval_precomputed/precomputed"
CK_PROPE="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/outputs/gvt/libero90_v2v_ic_lora_prope/checkpoints"
STEP=80000
OUTDIR="${LSIM}/outputs/bench_infer_time"

cd "${WORKDIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ltx
export PYTHONPATH="${USER_LIBERO}:${WORKDIR}/packages/ltx-trainer/src:${WORKDIR}/packages/ltx-core/src:${PYTHONPATH:-}"
export TMPDIR="/storage/project/r-agarg35-0/lwang831/tmp"
export MPLCONFIGDIR="${TMPDIR}/mpl"
unset HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export GVT_TIME=1
mkdir -p "${OUTDIR}" "${TMPDIR}" "${MPLCONFIGDIR}"

echo "=== per-chunk inference timing (DiT vs decode), step ${STEP}, PRoPE ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Start: $(date)"

# 5 single chunks in ONE model load; --include-gt/--include-context => 3 decodes/chunk
# (matches the eval). The first chunk also pays lazy CUDA/cuDNN warmup; report
# the later chunks as steady-state.
python -u "${LSIM}/infer_action_cond.py" \
    --lora-path "${CK_PROPE}/lora_weights_step_${STEP}.safetensors" \
    --action-projector "${CK_PROPE}/action_projector_step_${STEP}.safetensors" \
    --data-root "${DATA}" \
    --use-prope --prope-proj-dim 64 --prope-image-width 256 --prope-image-height 256 \
    --indices 0,1,2,3,4 \
    --include-gt --include-context --num-inference-steps 30 \
    --tag bench --out-dir "${OUTDIR}"

echo "=== DONE $(date) ==="
