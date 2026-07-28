#!/usr/bin/env bash
# Generate LIBERO-90 *play* episodes (recipe #3 demo-state init + #4 clamped
# object-relative play). Writes LIBERO-format HDF5s into OUT_DIR; the existing
# build_all_libero.py -> build_libero_context.py -> encode pipeline then consumes
# them UNCHANGED (point its --dataset at OUT_DIR).
#
# Sharded over TASKS (one HDF5 per source task). Submit an array sized to the
# shard count; NUM_SHARDS is derived from the array size.
#     sbatch --array=0-7 gen_libero_play.sbatch
#
# Smoke test first (1 task, 2 episodes, 60 steps):
#     MAX_TASKS=1 EPISODES_PER_TASK=2 STEPS=60 sbatch --array=0-0 gen_libero_play.sbatch

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=04:00:00
#SBATCH --mem-per-gpu=32G
#SBATCH --job-name=lib_play
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --requeue
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%A_%a.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%A_%a.err

set -euo pipefail

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT"
LSIM="${WORKDIR}/experiments/libero_sim"
GEN="${LSIM}/gen_libero_play.py"
CONDA_ENV="ltx"
USER_LIBERO="/storage/home/hcoda1/8/lwang831/workspace/LIBERO"

OUT_DIR="${OUT_DIR:-/storage/scratch1/8/lwang831/libero90_play}"
DATASET="${DATASET:-/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/LIBERO-datasets/libero_90}"
# Play-dominant mix (2026-06-08): reach 36 / perturb 18 / random 6 per task.
MIX="${MIX:-reach:36,perturb:18,random:6}"
STEPS="${STEPS:-200}"
SEED="${SEED:-0}"
MAX_TASKS="${MAX_TASKS:-0}"

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
if [[ -z "${NUM_SHARDS:-}" ]]; then
    if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
        NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT}"
    elif [[ -n "${SLURM_ARRAY_TASK_MAX:-}" && -n "${SLURM_ARRAY_TASK_MIN:-}" ]]; then
        NUM_SHARDS=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
    else
        NUM_SHARDS=1
    fi
fi

cd "${WORKDIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="${USER_LIBERO}:${PYTHONPATH:-}"
export TMPDIR="/storage/project/r-agarg35-0/lwang831/tmp"
export MPLCONFIGDIR="${TMPDIR}/mpl"
unset HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${OUT_DIR}" "${TMPDIR}" "${MPLCONFIGDIR}" \
         /storage/project/r-agarg35-0/lwang831/outputs/gvt/logs

echo "================ LIBERO play-data generation ================"
echo "JobID    : ${SLURM_ARRAY_JOB_ID:-N/A}_${SLURM_ARRAY_TASK_ID:-N/A} | Node: ${SLURM_NODELIST:-N/A}"
echo "Shard    : ${SHARD} / ${NUM_SHARDS}"
echo "OutDir   : ${OUT_DIR}"
echo "Mix      : ${MIX} | steps=${STEPS}"
echo "Start    : $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
python -V

python -u "${GEN}" \
    --dataset "${DATASET}" \
    --out_dir "${OUT_DIR}" \
    --mix "${MIX}" \
    --steps "${STEPS}" \
    --seed "${SEED}" \
    --max_tasks "${MAX_TASKS}" \
    --num_shards "${NUM_SHARDS}" \
    --shard "${SHARD}"

echo "Finished at $(date)"
