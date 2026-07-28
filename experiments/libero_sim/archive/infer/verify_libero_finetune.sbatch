#!/usr/bin/env bash
# verify_libero_finetune.sbatch — PROVE the LIBERO context data finetunes LTX-2.3
# end-to-end. Two steps in one H200 job:
#   1. ENCODE a handful of LIBERO chunks -> precomputed/ (VAE latents + Gemma
#      embeds + action tensors) via encode_libero_precomputed.py.
#   2. TRAIN one IC-LoRA + action-cond step on them (steps:1, validation off).
# A clean exit with a finite loss == the LIBERO dataset is trainable.
#
# H200 (141 GB) so the full bf16 22B transformer + Gemma load with no memory
# pressure for BOTH the encode (process_dataset.py) and the train step.

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH -p gpu-h200
#SBATCH --time=01:30:00
#SBATCH --mem=128G
#SBATCH --job-name=lib_smk
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.err

set -euo pipefail

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT"
LSIM="${WORKDIR}/experiments/libero_sim"
CONFIG="${WORKDIR}/packages/ltx-trainer/configs/ltx2_v2v_ic_lora_libero_smoke.yaml"
TRAIN="${WORKDIR}/packages/ltx-trainer/scripts/train.py"
ENCODE="${LSIM}/encode_libero_precomputed.py"
CONDA_ENV="ltx"

SRC_ROOT="${SRC_ROOT:-/storage/scratch1/8/lwang831/libero90_context}"
PRE_ROOT="${PRE_ROOT:-/storage/scratch1/8/lwang831/libero90_precomputed_smoke}"
N_CHUNKS="${N_CHUNKS:-4}"

cd "${WORKDIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Same env hygiene as the dataset builder + DROID smoke run.
export TMPDIR="/storage/project/r-agarg35-0/lwang831/tmp"
unset HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export MPLCONFIGDIR="${TMPDIR}/mpl"
mkdir -p "${TMPDIR}" "${HF_HOME}" "${MPLCONFIGDIR}" \
         /storage/project/r-agarg35-0/lwang831/outputs/gvt/logs
export PYTHONPATH="${WORKDIR}/packages/ltx-trainer/src:${PYTHONPATH:-}"

echo "================ LIBERO Finetune Verify ================"
echo "JobID : ${SLURM_JOB_ID:-N/A} | Node: ${SLURM_NODELIST:-N/A}"
echo "Src   : ${SRC_ROOT}"
echo "Pre   : ${PRE_ROOT}   (N_CHUNKS=${N_CHUNKS})"
echo "Config: ${CONFIG}"
echo "Start : $(date)"
echo "======================================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
python -V

echo; echo "########## STEP 1/2: encode ${N_CHUNKS} LIBERO chunks -> precomputed ##########"
python -u "${ENCODE}" \
    --src_root "${SRC_ROOT}" \
    --out_root "${PRE_ROOT}" \
    --max_chunks "${N_CHUNKS}" \
    --batch_size "${N_CHUNKS}"

echo; echo "########## STEP 2/2: one IC-LoRA + action-cond training step ##########"
python -u "${TRAIN}" "${CONFIG}" --disable-progress-bars

echo; echo "Finished at $(date)"
echo "If you see a finite training loss above with no traceback, LIBERO is trainable. ✔"
