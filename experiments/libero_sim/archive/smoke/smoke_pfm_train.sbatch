#!/usr/bin/env bash
# smoke_pfm_train.sbatch — 2-step end-to-end trainer run with PFM_ENABLE=1 (pure replace).
# Verifies the full PFM path (transformer forward -> x_hat0 -> unpatchify -> decode -> DINOv2
# -> backward) trains without crash/OOM on the real 22B model. inferno = fast + no preempt.
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH -p gpu-h200
#SBATCH --time=01:00:00
#SBATCH --mem=128G
#SBATCH --job-name=pfm_train_smoke
#SBATCH --qos=inferno
#SBATCH --account=gts-agarg35
#SBATCH --open-mode=append
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.err

set -uo pipefail
REPO="/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT"
LSIM="${REPO}/experiments/libero_sim"
USER_LIBERO="/storage/home/hcoda1/8/lwang831/workspace/LIBERO"
cd "${REPO}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ltx
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="${USER_LIBERO}:${LSIM}:${REPO}/packages/ltx-trainer/src:${REPO}/packages/ltx-core/src:${PYTHONPATH:-}"
export TORCH_HOME="${HOME}/.cache/torch" HF_HUB_OFFLINE=1

# --- PFM: pure replacement (latent MSE weight 0), full-clip decode, DINOv2 ---
export PFM_ENABLE=1 PFM_WEIGHT=1.0 PFM_MSE_WEIGHT=0.0 PFM_FRAMES=0

python -u "${REPO}/packages/ltx-trainer/scripts/train.py" \
  /storage/project/r-agarg35-0/lwang831/tmp/pfm_smoke_train.yaml --disable-progress-bars
echo "SMOKE_EXIT=$?"
