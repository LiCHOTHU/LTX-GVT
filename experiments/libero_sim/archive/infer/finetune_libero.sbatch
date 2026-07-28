#!/usr/bin/env bash
# finetune_libero.sbatch — REAL LTX-2.3 v2v IC-LoRA + action-cond finetune on the
# full 22,482-chunk LIBERO-90 precomputed dataset. Factorization:
#     p(context, text, observation, actions) -> target_video
# context = reference_latents (rendered context priors); NO PRoPE (vanilla 3D RoPE).
#
# Resume-safe under embers preemption: the config's model.load_checkpoint points
# at output_dir, so on --requeue train.py auto-loads the latest step_*.safetensors
# + minimal training state and continues. First run = empty dir -> fresh start.
# A single 8 h window won't reach steps:30000; rely on requeue (and/or a keepalive
# watchdog) to carry it across windows — checkpoints land every 1000 steps.
#
# H200 (141 GB): the full bf16 22B transformer + Gemma + LoRA + grad-ckpt fit with
# headroom (same node class the smoke verify used).

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH -p gpu-h200
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --job-name=lib_ft
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.err

set -euo pipefail

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT"
# CONFIG + OUTDIR are overridable so the SAME sbatch drives both the no-PRoPE and the
# PRoPE run (the watchdog passes them via --export). OUTDIR must match the chosen
# config's output_dir / load_checkpoint so auto-resume reloads the right checkpoints.
CONFIG="${CONFIG:-${WORKDIR}/packages/ltx-trainer/configs/ltx2_v2v_ic_lora_libero.yaml}"
OUTDIR="${OUTDIR:-/storage/project/r-agarg35-0/lwang831/outputs/gvt/libero90_v2v_ic_lora}"
TRAIN="${WORKDIR}/packages/ltx-trainer/scripts/train.py"
CONDA_ENV="ltx"

cd "${WORKDIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Same env hygiene as the encode + smoke-verify jobs.
export TMPDIR="/storage/project/r-agarg35-0/lwang831/tmp"
unset HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export MPLCONFIGDIR="${TMPDIR}/mpl"
export PYTHONPATH="${WORKDIR}/packages/ltx-trainer/src:${PYTHONPATH:-}"
mkdir -p "${TMPDIR}" "${HF_HOME}" "${MPLCONFIGDIR}" \
         /storage/project/r-agarg35-0/lwang831/outputs/gvt/logs \
         "${OUTDIR}"

echo "================ LIBERO-90 v2v IC-LoRA finetune ================"
echo "JobID : ${SLURM_JOB_ID:-N/A} | Node: ${SLURM_NODELIST:-N/A}"
echo "Config: ${CONFIG}"
echo "Data  : /storage/scratch1/8/lwang831/libero90_precomputed/precomputed"
echo "Out   : ${OUTDIR}"
echo "Start : $(date)"
echo "==============================================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
python -V

python -u "${TRAIN}" "${CONFIG}" --disable-progress-bars

echo "Finished at $(date)"
