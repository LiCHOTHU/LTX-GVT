#!/usr/bin/env bash
# smoke_finetune.sbatch — one-step IC-LoRA + action-cond training run to PROVE
# the scratch1 GVT/DROID dataset can be finetuned on LTX-2.3 end-to-end.
#
# H200 (141 GB) so the full bf16 22B transformer + Gemma load with zero memory
# pressure — a clean yes/no on the data, not a memory-tuning exercise.

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH -p gpu-h200
#SBATCH --time=01:00:00
#SBATCH --mem=128G
#SBATCH --job-name=gvt_smoke
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.err

set -euo pipefail

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT"
CONFIG="${WORKDIR}/packages/ltx-trainer/configs/ltx2_v2v_ic_lora_smoke.yaml"
TRAIN="${WORKDIR}/packages/ltx-trainer/scripts/train.py"
CONDA_ENV="ltx"

cd "${WORKDIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Same env hygiene as the dataset builder (see build_dataset_resumable.sbatch).
export TMPDIR="/storage/project/r-agarg35-0/lwang831/tmp"
unset HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export MPLCONFIGDIR="${TMPDIR}/mpl"
mkdir -p "${TMPDIR}" "${HF_HOME}" "${MPLCONFIGDIR}"

# Make the trainer package importable.
export PYTHONPATH="${WORKDIR}/packages/ltx-trainer/src:${PYTHONPATH:-}"

echo "================ Smoke Finetune ================"
echo "JobID : ${SLURM_JOB_ID:-N/A}"
echo "Node  : ${SLURM_NODELIST:-N/A}"
echo "Config: ${CONFIG}"
echo "Start : $(date)"
echo "==============================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
python -V
which python

python -u "${TRAIN}" "${CONFIG}" --disable-progress-bars

echo "Finished at $(date)"
