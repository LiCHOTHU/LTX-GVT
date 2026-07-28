#!/bin/bash
#SBATCH --job-name=clean_interact
#SBATCH --account=gts-agarg35
#SBATCH --partition=gpu-a100
#SBATCH -C A100-80GB
#SBATCH --qos=inferno
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
# Render the full interaction family (gt / reach / poke / random) on the clean
# tomato_sauce-ONLY scene, to eyeball the object's behavior under each interaction
# type before committing to a coverage dataset. See gen_clean_interact.py.
set -euo pipefail
WORKDIR=/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT
USER_LIBERO=/storage/home/hcoda1/8/lwang831/workspace/LIBERO
cd "${WORKDIR}"
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ltx

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="${USER_LIBERO}:${PYTHONPATH:-}"
export TMPDIR="/storage/project/r-agarg35-0/lwang831/tmp"
export MPLCONFIGDIR="${TMPDIR}/mpl"
unset HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}"

echo "Start: $(date)  Node: ${SLURM_NODELIST:-N/A}"
nvidia-smi --query-gpu=name --format=csv,noheader || true
python -V

python -u experiments/libero_sim/gen_clean_interact.py \
  --behaviors "gt,reach,poke,random" \
  --episodes 2 --steps 160 --res 256 \
  --out-dir "${WORKDIR}/experiments/libero_sim/outputs/clean_interact"

echo "=== clean_interact DONE $(date) ==="
echo "Videos: ${WORKDIR}/experiments/libero_sim/outputs/clean_interact/"
