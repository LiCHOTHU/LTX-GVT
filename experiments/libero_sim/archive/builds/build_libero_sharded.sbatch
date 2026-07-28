#!/usr/bin/env bash
# Parallel (sharded) LIBERO-90 context builder — array-job variant of
# build_libero_resumable.sbatch. Each array task is one worker that owns a
# disjoint slice of the enumerated demos (work index %% num_shards == shard),
# so N tasks build N demos at once and the ~77h serial wall time drops to ~77/N.
#
# Submit with an array sized to the shard count, e.g. 8 parallel workers:
#     sbatch --array=0-7 build_libero_sharded.sbatch
# NUM_SHARDS is derived from the array size automatically (override with the
# NUM_SHARDS env var if you submit a partial array). Per-task logs use %A_%a so
# the workers never clobber each other's output.
#
# RESUME / SAFETY: the _done/_failed markers are global and shards are disjoint,
# so workers never pick the same pending demo. A preempted task just relaunches
# (--requeue) and skips finished demos. Before launching this, retire the single
# unsharded worker so it doesn't race the shards on the same pending demo:
#     - stop the watchdog (kill the monitor_libero_build.sh nohup), then
#     - scancel the running `lib90` job.
# (Disjoint shards are safe among THEMSELVES; only the legacy single worker,
# which walks the whole list, can collide with them.)

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=64G
#SBATCH --job-name=lib90s
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --requeue
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%A_%a.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%A_%a.err

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

# Shard identity from the array task; NUM_SHARDS from the array size (overridable).
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

echo "================ LIBERO build (sharded) ================"
echo "JobID    : ${SLURM_ARRAY_JOB_ID:-N/A}_${SLURM_ARRAY_TASK_ID:-N/A} | Node: ${SLURM_NODELIST:-N/A}"
echo "Shard    : ${SHARD} / ${NUM_SHARDS}"
echo "OutRoot  : ${OUT_ROOT}"
echo "Dataset  : ${DATASET}"
echo "Res/FPS  : ${RES}/${FPS}"
echo "Start    : $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
python -V

python -u "${DRIVER}" \
    --dataset "${DATASET}" \
    --out_root "${OUT_ROOT}" \
    --res "${RES}" \
    --fps "${FPS}" \
    --num_shards "${NUM_SHARDS}" \
    --shard "${SHARD}"

echo "Finished at $(date)"
