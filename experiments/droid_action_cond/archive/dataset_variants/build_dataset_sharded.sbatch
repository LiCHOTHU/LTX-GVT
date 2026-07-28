#!/usr/bin/env bash
# Parallel (sharded) GVT / DROID dataset builder — array-job variant of
# build_dataset_resumable.sbatch. Each array task is one worker that owns a
# disjoint, hash-stable slice of the enumerated universe (shard_filter_universe:
# hash(cid) %% num_shards == shard), so N tasks build N chunks at once and the
# single-worker wall time drops ~N-fold.
#
# Submit with an array sized to the shard count, e.g. 8 parallel workers:
#     sbatch --array=0-7 build_dataset_sharded.sbatch
# The shard identity and count are derived from the array automatically
# (SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT). Per-task logs use %A_%a so the
# workers never clobber each other's output.
#
# RESUME / SAFETY: the _done/<cid> markers are global and shards are disjoint
# (stable hash partition), so workers never pick the same pending chunk. A
# preempted task just relaunches (--requeue) and skips finished chunks. Before
# launching this, retire the single unsharded worker so it doesn't race the
# shards on the same pending chunk:
#     - stop the watchdog (kill the monitor_dataset_build.sh nohup), then
#     - scancel the running `gvt_data` job.
# (Disjoint shards are safe among THEMSELVES; only the legacy single worker,
# which walks the whole universe, can collide with them.) Use
# monitor_dataset_sharded.sh to keep the array alive across preemption.
#
# Memory: Phase D loads Gemma-3-12B (8-bit) + LTX-2.3 VAE — needs an L40S (48 GB)
# or larger; V100 16 GB OOMs. Each shard is one such GPU job.

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=128G
#SBATCH --job-name=gvt_datas
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --requeue
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%A_%a.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%A_%a.err

set -euo pipefail

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT"
# Streaming per-episode builder. One process loads VGGT+VAE+Gemma+pyrender once,
# walks TFDS once, and commits _done markers chunk-by-chunk per episode. It honours
# --shard "i/N" (and auto-infers it from the Slurm array) to build a disjoint slice.
SCRIPT="${WORKDIR}/experiments/droid_action_cond/build_streaming.py"
CONDA_ENV="ltx"

# Built data is WRITTEN to cedar project share (scratch1 hit its 1M-inode quota;
# cedar0 has cleared its earlier full state, ~12 TB free). The DROID RLDS source
# (DROID_RLDS_DIR) + model checkpoints stay on cedar as read-only inputs.
OUT_ROOT="${OUT_ROOT:-/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/gvt_dataset_full}"
MAX_CHUNKS="${MAX_CHUNKS:-0}"   # 0 = no limit
MAX_EPS="${MAX_EPS:-0}"         # 0 = run until walltime/preempt

# Source RLDS: full DROID-1.0.1 by default. The watchdog passes this through.
export DROID_RLDS_DIR="${DROID_RLDS_DIR:-/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/droid_full/1.0.1}"

# Shard identity from the array task; count from the array size (overridable).
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

# Per the memory note: use /storage/project tmpdir, not /tmp.
export TMPDIR="/storage/project/r-agarg35-0/lwang831/tmp"
# Scrub the .bashrc-leaked HF cache pointers BEFORE re-exporting (same fix as the
# single-worker sbatch): .bashrc expands HF_HOME to an unwritable "/huggingface"
# on compute nodes. Unset, then re-export to a writable project path.
unset HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export MPLCONFIGDIR="${TMPDIR}/mpl"
export PYOPENGL_PLATFORM=egl
export VGGT_DEVICE="${VGGT_DEVICE:-cuda}"
# Production context recipe (2026-06-07): official per-frame wrist extrinsics
# from raw trajectory.h5 (mirrored-mount Rz(180) handled in the loader; VGGT is
# lazy fallback only) + raw wrist0 plane-warp with NO ext-camera fill.
export WRIST_CONTEXT_MODE="${WRIST_CONTEXT_MODE:-simple}"
export RAW_TRAJ_DIR="${RAW_TRAJ_DIR:-/storage/project/r-agarg35-0/lwang831/droid/raw_traj}"
mkdir -p "${TMPDIR}" "${HF_HOME}" "${MPLCONFIGDIR}" \
         /storage/project/r-agarg35-0/lwang831/outputs/gvt/logs

echo "================ GVT/DROID build (sharded) ================"
echo "JobID         : ${SLURM_ARRAY_JOB_ID:-N/A}_${SLURM_ARRAY_TASK_ID:-N/A} | Node: ${SLURM_NODELIST:-N/A}"
echo "Shard         : ${SHARD} / ${NUM_SHARDS}"
echo "OutRoot       : ${OUT_ROOT}"
echo "DROID_RLDS_DIR: ${DROID_RLDS_DIR}"
echo "Start         : $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
python -V

# Stamp-file guard: refuse to run if the universe wasn't enumerated against the
# same RLDS dir we're about to load from (would silently produce garbage).
STAMP="${OUT_ROOT}/_universe.rlds_root"
if [[ ! -f "${OUT_ROOT}/_universe.json" ]] || [[ ! -f "${STAMP}" ]] \
   || [[ "$(cat "${STAMP}" 2>/dev/null)" != "${DROID_RLDS_DIR}" ]]; then
    echo "[error] _universe.json is missing or stale (RLDS stamp mismatch)." >&2
    echo "        expected stamp: ${DROID_RLDS_DIR}" >&2
    echo "        actual stamp  : $(cat "${STAMP}" 2>/dev/null || echo '<missing>')" >&2
    echo "        Bootstrap the universe via the CPU enumeration sbatch first" >&2
    echo "        (monitor_dataset_sharded.sh routes Phase 1 there)." >&2
    exit 2
fi

python -u "${SCRIPT}" \
    --out_root "${OUT_ROOT}" \
    --shard "${SHARD}/${NUM_SHARDS}" \
    --max_chunks "${MAX_CHUNKS}" \
    --max_eps "${MAX_EPS}" \
    --load_text_encoder_in_8bit

echo "Finished at $(date)"
