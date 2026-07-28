#!/usr/bin/env bash
# eval_arms.sbatch — eval ONE active-learning arm (strategic|random) at a matched
# step across THREE eval sets, rendering pred|gt|ctx full-episode rollouts:
#   1. expert   — LIBERO expert-trajectory action-conditional (on-manifold GT replay)
#   2. perturb  — expert trajectory + OU perturbation (off-manifold)
#   3. circle   — scripted end-effector circle trajectory (synthetic)
# One 22B base load per eval set (3 loads total); inference only (no training).
#
# Submit per arm at a matched step (both arms retain step 72000):
#   sbatch --export=ALL,ARM=strategic,STEP=72000 eval_arms.sbatch
#   sbatch --export=ALL,ARM=random,STEP=72000    eval_arms.sbatch
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH -p gpu-a100
#SBATCH --constraint=A100-80GB
#SBATCH --time=04:00:00
#SBATCH --mem-per-gpu=128G
#SBATCH --job-name=eval_arm
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --requeue
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.err

set -euo pipefail
WD=/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT
cd "$WD"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate ltx
export TMPDIR=/storage/project/r-agarg35-0/lwang831/tmp
unset HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache; export HF_HUB_CACHE="$HF_HOME/hub"
export MPLCONFIGDIR="$TMPDIR/mpl"
export PYTHONPATH="$WD/packages/ltx-trainer/src:$WD/packages/ltx-core/src:${PYTHONPATH:-}"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"

ARM="${ARM:-strategic}"
STEP="${STEP:-72000}"
BASE_CKPT="${BASE_CKPT:-/storage/scratch1/8/lwang831/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors}"
CKDIR="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/outputs/gvt_alv2/libero90_ALR_${ARM}/checkpoints"
LORA="$CKDIR/lora_weights_step_${STEP}.safetensors"
PROJ="$CKDIR/action_projector_step_${STEP}.safetensors"
PROPE_FLAGS="--use-prope --prope-proj-dim 64 --prope-image-width 256 --prope-image-height 256"
OUTROOT="$WD/experiments/libero_sim/outputs/al_eval_step${STEP}/${ARM}"

GT=/storage/scratch1/8/lwang831/libero90_gt_precomputed/precomputed
OOD=/storage/scratch1/8/lwang831/libero90_ood_precomputed/precomputed
CIRCLE=/storage/scratch1/8/lwang831/libero90_circle_precomputed/precomputed

# stems present in a precomputed set (optionally filtered by a grep pattern), joined by commas
stems () { # $1=data-root  $2=optional grep pattern
  local f; f="$(ls "$1/actions/videos" | sed -E 's/_chunk_[0-9]+_target\.pt$//' | sort -u)"
  [ -n "${2:-}" ] && f="$(echo "$f" | grep -E "$2")"
  echo "$f" | paste -sd, -
}

EPS_EXPERT="$(stems "$GT")"
EPS_PERTURB="$(stems "$OOD" 'perturb')"
EPS_CIRCLE="$(stems "$CIRCLE")"

echo "============== AL arm eval =============="
echo "JobID : ${SLURM_JOB_ID:-N/A} | Node: ${SLURM_NODELIST:-N/A}"
echo "ARM   : ${ARM}  STEP : ${STEP}"
echo "LoRA  : $LORA"
echo "Out   : $OUTROOT"
echo "episodes: expert=$(echo "$EPS_EXPERT" | tr ',' '\n' | wc -l)  perturb=$(echo "$EPS_PERTURB" | tr ',' '\n' | wc -l)  circle=$(echo "$EPS_CIRCLE" | tr ',' '\n' | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Start: $(date)"
[ -f "$LORA" ] || { echo "MISSING LoRA: $LORA"; exit 1; }

run_eval () { # $1=tag/subdir  $2=data-root  $3=episode-csv
  local tag="$1" data="$2" eps="$3"
  local out="$OUTROOT/$tag"; mkdir -p "$out"
  echo "----- [$tag] $(echo "$eps" | tr ',' '\n' | wc -l) episodes -> $out -----"
  python -u experiments/libero_sim/infer_action_cond.py \
    --base-checkpoint "$BASE_CKPT" \
    --lora-path "$LORA" --action-projector "$PROJ" \
    --data-root "$data" $PROPE_FLAGS \
    --episode "$eps" \
    --include-gt --include-context --num-inference-steps 30 \
    --tag "${ARM}_${tag}" --out-dir "$out"
}

run_eval expert  "$GT"     "$EPS_EXPERT"
run_eval perturb "$OOD"    "$EPS_PERTURB"
run_eval circle  "$CIRCLE" "$EPS_CIRCLE"

echo "=== ${ARM} step ${STEP} 3-way eval DONE at $(date) ==="
echo "Videos under: $OUTROOT/{expert,perturb,circle}"
