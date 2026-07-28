#!/usr/bin/env bash
# Enumerate the (episode, chunk_idx) universe for the FULL DROID-1.0.1 RLDS.
#
# CPU-only — pure TFDS iteration + JSON join against calibration. No GPU.
# Wallclock: ~2.7 h sequentially (95,658 episodes × ~20 s/100 eps observed
# during smoke). embers caps at 8 h, which is the wallclock set below.
# embers + --requeue means a preempted run is harmless: the manifest is
# rewritten atomically at the end, and partial intermediate state isn't
# persisted, so a requeue restarts the scan from scratch — but at ~2.7 h
# the chance of preemption mid-run is low.
#
# After the manifest is written this script also writes a stamp file recording
# which RLDS dir it was built against — both monitor_dataset_build_full.sh and
# build_dataset_resumable_full.sbatch check this stamp to refuse stale runs.

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --job-name=gvt_enum
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.err

set -euo pipefail

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT"
SCRIPT="${WORKDIR}/experiments/droid_action_cond/enumerate_universe.py"
CONDA_ENV="ltx"

OUT_ROOT="${OUT_ROOT:-/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/gvt_dataset_full}"
export DROID_RLDS_DIR="${DROID_RLDS_DIR:-/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/droid_full/1.0.1}"
export DROID_CALIB_DIR="${DROID_CALIB_DIR:-/storage/project/r-agarg35-0/lwang831/droid/calibration}"

# Optional override; default 0 = scan everything.
MAX_EPISODES="${MAX_EPISODES:-0}"

mkdir -p /storage/project/r-agarg35-0/lwang831/outputs/gvt/logs "${OUT_ROOT}"

cd "${WORKDIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export TMPDIR="/storage/project/r-agarg35-0/lwang831/tmp"
unset HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export MPLCONFIGDIR="${TMPDIR}/mpl"
mkdir -p "${TMPDIR}" "${HF_HOME}" "${MPLCONFIGDIR}"

echo "================ Job Info ================"
echo "JobID         : ${SLURM_JOB_ID:-N/A}"
echo "Node          : ${SLURM_NODELIST:-N/A}"
echo "OutRoot       : ${OUT_ROOT}"
echo "DROID_RLDS_DIR: ${DROID_RLDS_DIR}"
echo "DROID_CALIB   : ${DROID_CALIB_DIR}"
echo "MaxEpisodes   : ${MAX_EPISODES} (0 = all)"
echo "Start         : $(date)"
echo "=========================================="

EXTRA_ARGS=()
if [[ "${MAX_EPISODES}" != "0" ]]; then
    EXTRA_ARGS+=(--max_episodes "${MAX_EPISODES}")
fi

python -u "${SCRIPT}" --out_root "${OUT_ROOT}" "${EXTRA_ARGS[@]}"

# Stamp the RLDS source so the build sbatch can detect stale universes.
STAMP="${OUT_ROOT}/_universe.rlds_root"
echo "${DROID_RLDS_DIR}" > "${STAMP}"
echo "[stamp] wrote ${STAMP} -> ${DROID_RLDS_DIR}"

echo "Finished at $(date)"
