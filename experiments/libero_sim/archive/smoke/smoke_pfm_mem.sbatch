#!/usr/bin/env bash
# smoke_pfm_mem.sbatch — isolated GPU memory feasibility test for PFM (decode+DINOv2+backward).
# Short, non-preemptible (inferno) so it schedules fast and we get an answer quickly.
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH -p gpu-h200
#SBATCH --time=00:30:00
#SBATCH --mem=128G
#SBATCH --job-name=pfm_smoke
#SBATCH --qos=inferno
#SBATCH --account=gts-agarg35
#SBATCH --open-mode=append
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.err

set -uo pipefail
REPO="/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT"
LSIM="${REPO}/experiments/libero_sim"
CONDA_ENV="ltx"
cd "${REPO}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${LSIM}:${REPO}/packages/ltx-trainer/src:${REPO}/packages/ltx-core/src:${PYTHONPATH:-}"
export TORCH_HOME="${HOME}/.cache/torch" HF_HUB_OFFLINE=1

python -u "${LSIM}/smoke_pfm_mem.py"
