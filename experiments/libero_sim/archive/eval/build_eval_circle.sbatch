#!/usr/bin/env bash
# build_eval_circle.sbatch — full scripted "descend + draw circle" OOD case,
# build AND eval in ONE resumable H200 job (gpu-h200, account gts-agarg35, embers).
#
#   1. gen_libero_play.py  --mix circle:1   (10 spread scenes; +2 extra candidates
#                                            so >=10 succeed even if a scene's BDDL
#                                            is broken, e.g. LIVING_ROOM_SCENE4)
#   2. build_all_libero.py        (capture state-replay -> joints -> FK render)
#   3. encode_libero_precomputed  (-> latents/ref/cond/actions)
#   4. build_libero_cameras       (PRoPE cameras)
#   5. infer_action_cond.py x2    (PRoPE + non-PRoPE, step 54000, save mp4s)
#   6. score_ood.py               (CIRCLE aggregate, PRoPE vs non-PRoPE)
#
# Every build stage is marker-resumable (_gen skip-if-exists / _done), so it
# REUSES the ~9 episodes already built+encoded on scratch1 and only processes the
# new scene(s). Submit:  sbatch experiments/libero_sim/build_eval_circle.sbatch

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH -p gpu-h200
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=128G
#SBATCH --job-name=circle_eval
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --requeue
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.err

set -euo pipefail

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT"
LSIM="${WORKDIR}/experiments/libero_sim"
CONDA_ENV="ltx"
USER_LIBERO="/storage/home/hcoda1/8/lwang831/workspace/LIBERO"

LIBERO90="${LIBERO90:-/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/LIBERO-datasets/libero_90}"
PLAY="${PLAY:-/storage/scratch1/8/lwang831/libero90_circle_play}"
CTX="${CTX:-/storage/scratch1/8/lwang831/libero90_circle_context}"
PRE="${PRE:-/storage/scratch1/8/lwang831/libero90_circle_precomputed}"
SEED="${SEED:-20260613}"
STEPS="${STEPS:-200}"
MIX="${MIX:-circle:1}"
RES="${RES:-256}"
FPS="${FPS:-15}"

STEP="${STEP:-54000}"
DATA="${PRE}/precomputed"
OUTROOT="${WORKDIR}/experiments/libero_sim/outputs/eval_circle_step${STEP}"
CK_PROPE="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/outputs/gvt/libero90_v2v_ic_lora_prope/checkpoints"
CK_NOPROPE="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/outputs/gvt/libero90_v2v_ic_lora/checkpoints"

cd "${WORKDIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="${USER_LIBERO}:${WORKDIR}/packages/ltx-trainer/src:${WORKDIR}/packages/ltx-core/src:${PYTHONPATH:-}"
export TMPDIR="/storage/project/r-agarg35-0/lwang831/tmp"
export MPLCONFIGDIR="${TMPDIR}/mpl"
unset HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export CIRCLE_RADIUS_M="${CIRCLE_RADIUS_M:-0.14}"
mkdir -p "${PLAY}" "${CTX}" "${PRE}" "${TMPDIR}" "${MPLCONFIGDIR}" "${OUTROOT}" \
         /storage/project/r-agarg35-0/lwang831/outputs/gvt/logs

echo "================ circle-OOD build+eval (gpu-h200) ================"
echo "JobID : ${SLURM_JOB_ID:-N/A} | Node: ${SLURM_NODELIST:-N/A}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
echo "Seed ${SEED} Steps ${STEPS} Mix ${MIX} R=${CIRCLE_RADIUS_M}m  Step ${STEP}"
echo "Out  : ${OUTROOT}"
echo "Start: $(date)"

echo; echo "---- [1/6] generate scripted circle play episodes (>=10 scenes) ----"
# shard 0 of 9 -> the 10 canonical spread scenes (reused if already generated;
# LIVING_ROOM_SCENE4 fails on a LIBERO BDDL geom bug, leaving 9).
python -u "${LSIM}/gen_libero_play.py" \
    --dataset "${LIBERO90}" --out_dir "${PLAY}" \
    --mix "${MIX}" --steps "${STEPS}" --seed "${SEED}" \
    --max_tasks 0 --num_shards 9 --shard 0
# shard 1 -> 2 extra distinct scenes so the eval covers a FULL 10+ (margin for fails).
python -u "${LSIM}/gen_libero_play.py" \
    --dataset "${LIBERO90}" --out_dir "${PLAY}" \
    --mix "${MIX}" --steps "${STEPS}" --seed "${SEED}" \
    --max_tasks 2 --num_shards 9 --shard 1

echo; echo "---- [2/6] build context (capture + FK render) ----"
python -u "${LSIM}/build_all_libero.py" \
    --dataset "${PLAY}" --out_root "${CTX}" \
    --res "${RES}" --fps "${FPS}" --max_demos 0

echo; echo "---- [3/6] encode -> precomputed ----"
python -u "${LSIM}/encode_libero_precomputed.py" \
    --src_root "${CTX}" --out_root "${PRE}" \
    --shard "0/1" --batch_size 4 --max_chunks 0 \
    --load_text_encoder_in_8bit

echo; echo "---- [4/6] build PRoPE cameras ----"
python -u "${LSIM}/build_libero_cameras.py" \
    --src_root "${CTX}" --out_root "${PRE}" \
    --shard "0/1" --max_chunks 0

EPS="$(ls "${DATA}/actions/videos" | sed -E 's/_chunk_[0-9]+_target\.pt$//' | sort -u | paste -sd, -)"
NEPS="$(echo "${EPS}" | tr ',' '\n' | wc -l)"
echo; echo "Episodes in eval: ${NEPS}"
echo "${EPS}" | tr ',' '\n' | sed 's/_demo_play_circle_demo_0//'

run_eval () {  # $1=mode  $2=ckdir  $3=propeflags
  local mode="$1" ckdir="$2" flags="$3"
  local outdir="${OUTROOT}/eval_${mode}_episode"
  mkdir -p "${outdir}"
  echo; echo "---- [5/6] inference: ${mode} (step ${STEP}, ${NEPS} eps) ----"
  python -u "${LSIM}/infer_action_cond.py" \
    --lora-path "${ckdir}/lora_weights_step_${STEP}.safetensors" \
    --action-projector "${ckdir}/action_projector_step_${STEP}.safetensors" \
    --data-root "${DATA}" \
    ${flags} \
    --episode "${EPS}" \
    --include-gt --include-context --num-inference-steps 30 \
    --tag "${mode}" --out-dir "${outdir}"
}

run_eval prope   "${CK_PROPE}"   "--use-prope --prope-proj-dim 64 --prope-image-width 256 --prope-image-height 256"
run_eval noprope "${CK_NOPROPE}" ""

echo; echo "---- [6/6] score ----"
python -u "${LSIM}/score_ood.py" --root "${OUTROOT}" || echo "[warn] scoring failed (videos saved)"

echo; echo "=== circle-OOD build+eval DONE at $(date) ==="
echo "Videos: ${OUTROOT}/eval_{prope,noprope}_episode/*.mp4"
