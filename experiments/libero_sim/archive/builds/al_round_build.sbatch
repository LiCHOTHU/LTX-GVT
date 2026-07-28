#!/usr/bin/env bash
# al_round_build.sbatch — build ONE active-learning round's candidate pool.
#
# Closed-loop AL generates fresh candidates every round (not a static pool), so this
# runs the full LIBERO pipeline on a small task slice into per-round/arm scratch dirs:
#   1. gen_libero_play.py            -> ~40 play episodes, mix reach:7/perturb:10/random:3
#                                       per task x TASKS_PER_ROUND tasks. PERTURB draws its
#                                       base trajectory from the arm's REPLAY BUFFER
#                                       (--perturb-source) + noise; reach/random unchanged.
#   2. build_all_libero.py           -> context (state-replay -> joints -> FK render)
#   3. encode_libero_precomputed.py  -> precomputed/ (latents/ref/cond/actions)
#   4. build_libero_cameras.py       -> precomputed/cameras (PRoPE)
#
# The monitor (monitor_active_replay_libero.sh) submits this once per round/arm, then
# scores PRE/precomputed and selects the hardest half. EVAL/BUILD only — no training.
#
# Task slice: NUM_SHARDS/SHARD pick TASKS_PER_ROUND tasks; SHARD = round % NUM_SHARDS
# rotates the slice through all tasks over successive rounds (coverage spread).
#
# Required env: PLAY CTX PRE SEED SHARD     Optional: PERTURB_SOURCE MIX NUM_SHARDS STEPS

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=04:00:00
#SBATCH --mem-per-gpu=128G
#SBATCH --job-name=al_build
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --requeue
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.err

set -euo pipefail

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT"
LSIM="${WORKDIR}/experiments/libero_sim"
CONDA_ENV="ltx"
USER_LIBERO="/storage/home/hcoda1/8/lwang831/workspace/LIBERO"

LIBERO90="${LIBERO90:-/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/LIBERO-datasets/libero_90}"
: "${PLAY:?set PLAY=per-round play dir}"
: "${CTX:?set CTX=per-round context dir}"
: "${PRE:?set PRE=per-round precomputed dir}"
: "${SEED:?set SEED}"
: "${SHARD:?set SHARD (= round % NUM_SHARDS)}"
PERTURB_SOURCE="${PERTURB_SOURCE:-}"
# Per-task mix at the 35/50/15 ratio (7:10:3 = 20/task); TASKS_PER_ROUND=2 -> ~40/round.
# The monitor pipe-encodes commas (sbatch --export separator); translate them back.
MIX="${MIX:-reach:7,perturb:10,random:3}"
MIX="${MIX//|/,}"
NUM_SHARDS="${NUM_SHARDS:-45}"     # 90 tasks / 2 per round
STEPS="${STEPS:-200}"
RES="${RES:-256}"
FPS="${FPS:-15}"

cd "${WORKDIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="${USER_LIBERO}:${LSIM}:${WORKDIR}/packages/ltx-trainer/src:${WORKDIR}/packages/ltx-core/src:${PYTHONPATH:-}"
export TMPDIR="/storage/project/r-agarg35-0/lwang831/tmp"
export MPLCONFIGDIR="${TMPDIR}/mpl"
unset HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${PLAY}" "${CTX}" "${PRE}" "${TMPDIR}" "${MPLCONFIGDIR}" \
         /storage/project/r-agarg35-0/lwang831/outputs/gvt/logs

PSRC_FLAG=""
[[ -n "${PERTURB_SOURCE}" && -f "${PERTURB_SOURCE}" ]] && PSRC_FLAG="--perturb-source ${PERTURB_SOURCE}"

echo "================ AL round build ================"
echo "JobID  : ${SLURM_JOB_ID:-N/A} | Node: ${SLURM_NODELIST:-N/A}"
echo "Seed   : ${SEED}  Shard: ${SHARD}/${NUM_SHARDS}  Mix: ${MIX}"
echo "Perturb-source: ${PERTURB_SOURCE:-<none, GT fallback>}"
echo "Play   : ${PLAY}"
echo "Pre    : ${PRE}/precomputed"
echo "Start  : $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

echo; echo "---- [1/4] generate candidates (perturb <- replay buffer) ----"
# shellcheck disable=SC2086
python -u "${LSIM}/gen_libero_play.py" \
    --dataset "${LIBERO90}" --out_dir "${PLAY}" \
    --mix "${MIX}" --steps "${STEPS}" --seed "${SEED}" \
    --num_shards "${NUM_SHARDS}" --shard "${SHARD}" ${PSRC_FLAG}

echo; echo "---- [2/4] build context ----"
python -u "${LSIM}/build_all_libero.py" \
    --dataset "${PLAY}" --out_root "${CTX}" \
    --res "${RES}" --fps "${FPS}" --max_demos 0

echo; echo "---- [3/4] encode -> precomputed ----"
python -u "${LSIM}/encode_libero_precomputed.py" \
    --src_root "${CTX}" --out_root "${PRE}" \
    --shard "0/1" --batch_size 4 --max_chunks 0 \
    --load_text_encoder_in_8bit

echo; echo "---- [4/4] build PRoPE cameras ----"
python -u "${LSIM}/build_libero_cameras.py" \
    --src_root "${CTX}" --out_root "${PRE}" \
    --shard "0/1" --max_chunks 0

touch "${PRE}/_build_done"
echo; echo "=== AL round build DONE at $(date) ==="
