#!/usr/bin/env bash
# eval_comprehensive.sbatch — ONE resumable H200 job that quantitatively evaluates
# BOTH finetunes (PRoPE + non-PRoPE) at the FINISHED step 80000 across all FOUR
# action conditions:  GT | perturb | random | circle.
#
#   A. build the missing GT held-out set (gen gt:1 -> context -> encode -> cameras)
#      (perturb+random already at libero90_ood_precomputed; circle at libero90_circle_precomputed)
#   B. symlink-union the 3 precomputed roots into ONE combined root
#      (so the 22B model loads ONCE per mode, not 3x)
#   C. infer_action_cond.py x2   (PRoPE + non-PRoPE, step 80000, save pred|gt|ctx mp4s)
#   D. score_ood.py              (GT / RANDOM / PERTURB / CIRCLE aggregates, PRoPE vs non-PRoPE)
#
# Every build stage is marker-resumable, so a requeue/re-run reuses finished work.
# Submit:  sbatch experiments/libero_sim/eval_comprehensive.sbatch

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH -p gpu-h200
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=128G
#SBATCH --job-name=eval_all
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
SEED="${SEED:-20260612}"          # == the OOD set's seed (same stride-9 scenes)
STEPS="${STEPS:-200}"
RES="${RES:-256}"
FPS="${FPS:-15}"
STEP="${STEP:-80000}"             # the FINISHED finetunes

# --- existing (reused) precomputed roots ---
PRE_OOD="${PRE_OOD:-/storage/scratch1/8/lwang831/libero90_ood_precomputed}"        # random + perturb
PRE_CIRCLE="${PRE_CIRCLE:-/storage/scratch1/8/lwang831/libero90_circle_precomputed}"  # circle
# --- GT set built here ---
PLAY_GT="${PLAY_GT:-/storage/scratch1/8/lwang831/libero90_gt_play}"
CTX_GT="${CTX_GT:-/storage/scratch1/8/lwang831/libero90_gt_context}"
PRE_GT="${PRE_GT:-/storage/scratch1/8/lwang831/libero90_gt_precomputed}"
# --- combined symlink union ---
COMBINED="${COMBINED:-/storage/scratch1/8/lwang831/libero90_alleval_precomputed}"
DATA="${COMBINED}/precomputed"

OUTROOT="${WORKDIR}/experiments/libero_sim/outputs/eval_all_step${STEP}"
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
mkdir -p "${PLAY_GT}" "${CTX_GT}" "${PRE_GT}" "${COMBINED}" "${TMPDIR}" "${MPLCONFIGDIR}" "${OUTROOT}" \
         /storage/project/r-agarg35-0/lwang831/outputs/gvt/logs

echo "================ comprehensive 4-condition eval (gpu-h200) ================"
echo "JobID : ${SLURM_JOB_ID:-N/A} | Node: ${SLURM_NODELIST:-N/A}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
echo "Seed ${SEED} Steps ${STEPS} Step ${STEP}"
echo "Reuse: ${PRE_OOD} (rand+perturb), ${PRE_CIRCLE} (circle)"
echo "Build: ${PRE_GT} (gt)"
echo "Union: ${DATA}"
echo "Out  : ${OUTROOT}"
echo "Start: $(date)"

echo; echo "---- [A/D] build GT held-out set (real-demo replay, gt:1) ----"
python -u "${LSIM}/gen_libero_play.py" \
    --dataset "${LIBERO90}" --out_dir "${PLAY_GT}" \
    --mix "gt:1" --steps "${STEPS}" --seed "${SEED}" \
    --max_tasks 0 --num_shards 9 --shard 0
python -u "${LSIM}/build_all_libero.py" \
    --dataset "${PLAY_GT}" --out_root "${CTX_GT}" \
    --res "${RES}" --fps "${FPS}" --max_demos 0
python -u "${LSIM}/encode_libero_precomputed.py" \
    --src_root "${CTX_GT}" --out_root "${PRE_GT}" \
    --shard "0/1" --batch_size 4 --max_chunks 0 \
    --load_text_encoder_in_8bit
python -u "${LSIM}/build_libero_cameras.py" \
    --src_root "${CTX_GT}" --out_root "${PRE_GT}" \
    --shard "0/1" --max_chunks 0

echo; echo "---- [B/D] symlink-union the 3 precomputed roots -> ${DATA} ----"
for src in "${PRE_OOD}/precomputed" "${PRE_CIRCLE}/precomputed" "${PRE_GT}/precomputed"; do
  if [ ! -d "${src}" ]; then echo "[warn] missing ${src}, skipping"; continue; fi
  for moddir in "${src}"/*/videos; do
    [ -d "${moddir}" ] || continue
    rel="${moddir#${src}/}"                 # e.g. actions/videos
    dst="${DATA}/${rel}"
    mkdir -p "${dst}"
    for f in "${moddir}"/*.pt; do
      [ -e "${f}" ] || continue
      ln -sfn "${f}" "${dst}/$(basename "${f}")"
    done
  done
done
for m in actions latents reference_latents conditions cameras; do
  printf "  %-18s %5s files\n" "${m}/videos" "$(ls "${DATA}/${m}/videos" 2>/dev/null | wc -l)"
done

EPS="$(ls "${DATA}/actions/videos" | sed -E 's/_chunk_[0-9]+_target\.pt$//' | sort -u | paste -sd, -)"
NEPS="$(echo "${EPS}" | tr ',' '\n' | grep -c . || true)"
echo; echo "Episodes in eval: ${NEPS}"
echo "  by kind:"
for k in gt random perturb circle; do
  c="$(echo "${EPS}" | tr ',' '\n' | grep -c "_play_${k}_" || true)"
  printf "    %-8s %3s\n" "${k}" "${c}"
done

run_eval () {  # $1=mode  $2=ckdir  $3=propeflags
  local mode="$1" ckdir="$2" flags="$3"
  local outdir="${OUTROOT}/eval_${mode}_episode"
  mkdir -p "${outdir}"
  echo; echo "---- [C/D] inference: ${mode} (step ${STEP}, ${NEPS} eps) ----"
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

echo; echo "---- [D/D] score ----"
python -u "${LSIM}/score_ood.py" --root "${OUTROOT}" || echo "[warn] scoring failed (videos saved)"

echo; echo "=== comprehensive eval DONE at $(date) ==="
echo "Videos: ${OUTROOT}/eval_{prope,noprope}_episode/*.mp4"
echo "Scores: ${OUTROOT}/scores_ood.txt"
