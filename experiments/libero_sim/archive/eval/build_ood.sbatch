#!/usr/bin/env bash
# build_ood.sbatch — one-shot, resumable build of a FRESH held-out OOD eval set.
#
# Mirrors the production LIBERO pipeline EXACTLY, just on a fresh seed + a small
# task slice, into dedicated scratch dirs so nothing collides with the training
# corpus:
#   1. gen_libero_play.py     -> 10 tasks x {perturb:1, random:1} play HDF5s
#                                (perturb = GT actions + OU noise; random = rand)
#   2. build_all_libero.py    -> context (state-replay -> joints -> FK render)
#   3. encode_libero_precomputed.py -> precomputed/ (latents/ref/cond/actions)
#   4. build_libero_cameras.py      -> precomputed/cameras (for PRoPE)
#
# Every stage is marker-resumable (_gen_done / _done), so a preempted --requeue
# run picks up where it left off. ~20 episodes / ~120 chunks fits one 8h window.
#
#   sbatch build_ood.sbatch
#
# Knobs (defaults shown): SEED=20260612  TASKS=10  STEPS=200
#                         MIX=perturb:1,random:1

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=128G
#SBATCH --job-name=ood_build
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
PLAY="${PLAY:-/storage/scratch1/8/lwang831/libero90_ood_play}"
CTX="${CTX:-/storage/scratch1/8/lwang831/libero90_ood_context}"
PRE="${PRE:-/storage/scratch1/8/lwang831/libero90_ood_precomputed}"
SEED="${SEED:-20260612}"
TASKS="${TASKS:-10}"
STEPS="${STEPS:-200}"
MIX="${MIX:-perturb:1,random:1}"
RES="${RES:-256}"
FPS="${FPS:-15}"

cd "${WORKDIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Headless GL for mujoco (capture) + pyrender (render).
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# Union of the gen/build (LIBERO) and encode (trainer/core) PYTHONPATHs.
export PYTHONPATH="${USER_LIBERO}:${WORKDIR}/packages/ltx-trainer/src:${WORKDIR}/packages/ltx-core/src:${PYTHONPATH:-}"
export TMPDIR="/storage/project/r-agarg35-0/lwang831/tmp"
export MPLCONFIGDIR="${TMPDIR}/mpl"
unset HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${PLAY}" "${CTX}" "${PRE}" "${TMPDIR}" "${MPLCONFIGDIR}" \
         /storage/project/r-agarg35-0/lwang831/outputs/gvt/logs

echo "================ OOD eval-set build ================"
echo "JobID  : ${SLURM_JOB_ID:-N/A} | Node: ${SLURM_NODELIST:-N/A}"
echo "Seed   : ${SEED}  Tasks: ${TASKS}  Steps: ${STEPS}  Mix: ${MIX}"
echo "Play   : ${PLAY}"
echo "Ctx    : ${CTX}"
echo "Pre    : ${PRE}/precomputed"
echo "Start  : $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

echo; echo "---- [1/4] generate fresh OOD play episodes ----"
# Diversity: take every GEN_STRIDE-th task (shard 0 of GEN_STRIDE) so the ~10
# tasks span many scenes instead of the first-N alphabetical cluster (all SCENE1/10).
# 90 tasks / stride 9 = 10 spread tasks.
python -u "${LSIM}/gen_libero_play.py" \
    --dataset "${LIBERO90}" --out_dir "${PLAY}" \
    --mix "${MIX}" --steps "${STEPS}" --seed "${SEED}" \
    --max_tasks 0 --num_shards "${GEN_STRIDE:-9}" --shard 0

echo; echo "---- [2/4] build context (capture + FK render) ----"
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

echo; echo "=== OOD build DONE at $(date) ==="
echo "data-root for inference: ${PRE}/precomputed"
