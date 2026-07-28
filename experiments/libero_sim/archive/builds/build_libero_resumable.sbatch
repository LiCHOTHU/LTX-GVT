#!/usr/bin/env bash
# Resumable LIBERO-90 context builder (all tasks, all demos).
#
# Single-worker, marker-resumable: build_all_libero.py skips any demo with a
# _done/ or _failed/ marker, so a preempted/timed-out job just relaunches and
# picks up where it left off. Use monitor_libero_build.sh to keep one alive.
#
# LIBERO wrist context uses the SIMULATOR'S GROUND-TRUTH DEPTH (camera_depths=
# True) fused across 3 cameras -> real depth+multiview reprojection (unlike
# DROID, where depth had to be estimated). Fixed cams use the DROID mesh-over-
# clean-plate method.

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=64G
#SBATCH --job-name=lib90
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --requeue
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.err

set -euo pipefail

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT"
LSIM="${WORKDIR}/experiments/libero_sim"
DRIVER="${LSIM}/build_all_libero.py"
CONDA_ENV="ltx"
# LIBERO is the user's clone; PEP 660 editable install is broken, so put the
# package parent on PYTHONPATH explicitly (see libero-render-fix memory).
USER_LIBERO="/storage/home/hcoda1/8/lwang831/workspace/LIBERO"

# Written to scratch1 (cedar0 is full). Auto-purges after ~60 days.
OUT_ROOT="${OUT_ROOT:-/storage/scratch1/8/lwang831/libero90_context}"
DATASET="${DATASET:-/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/LIBERO-datasets/libero_90}"
RES="${RES:-256}"
FPS="${FPS:-15}"
MAX_DEMOS="${MAX_DEMOS:-0}"   # 0 = build everything

cd "${WORKDIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Headless GL: mujoco (capture) + pyrender (render), both EGL.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="${USER_LIBERO}:${PYTHONPATH:-}"
export TMPDIR="/storage/project/r-agarg35-0/lwang831/tmp"
export MPLCONFIGDIR="${TMPDIR}/mpl"
# Scrub .bashrc-leaked unwritable HF cache pointers (same fix as DROID build).
unset HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${OUT_ROOT}" "${TMPDIR}" "${MPLCONFIGDIR}" \
         /storage/project/r-agarg35-0/lwang831/outputs/gvt/logs

echo "================ LIBERO build ================"
echo "JobID    : ${SLURM_JOB_ID:-N/A} | Node: ${SLURM_NODELIST:-N/A}"
echo "OutRoot  : ${OUT_ROOT}"
echo "Dataset  : ${DATASET}"
echo "Res/FPS  : ${RES}/${FPS}   MaxDemos: ${MAX_DEMOS}"
echo "Start    : $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
python -V

python -u "${DRIVER}" \
    --dataset "${DATASET}" \
    --out_root "${OUT_ROOT}" \
    --res "${RES}" \
    --fps "${FPS}" \
    --max_demos "${MAX_DEMOS}"

echo "Finished at $(date)"
