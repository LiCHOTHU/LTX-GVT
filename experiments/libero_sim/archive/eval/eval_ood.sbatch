#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=128G
#SBATCH --job-name=ood_eval
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.err
#
# eval_ood.sbatch — render pred|gt|ctx rollouts for ONE model (PRoPE or non-PRoPE)
# on the FRESH held-out OOD set built by build_ood.sbatch. Episodes (10 rand +
# 10 gt+noise) are enumerated automatically from the precomputed data root, so
# nothing is hardcoded. Score with score_ood.py afterward.
#
# Submit twice (matched step):
#   sbatch --export=ALL,MODE=prope   eval_ood.sbatch
#   sbatch --export=ALL,MODE=noprope eval_ood.sbatch
#
# Knobs: MODE={prope|noprope} (default prope), STEP (default 54000 = latest step
# BOTH finetunes currently retain LoRA for), DATA (precomputed dir).

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

MODE="${MODE:-prope}"
STEP="${STEP:-54000}"
DATA="${DATA:-/storage/scratch1/8/lwang831/libero90_ood_precomputed/precomputed}"
OUTROOT="$WD/experiments/libero_sim/outputs/eval_ood_step${STEP}"

# Enumerate every episode stem present in the OOD precomputed set (10 play_random
# + 10 play_perturb), strip the _chunk_NN_target.pt suffix, dedup, join with commas.
EPS="$(ls "$DATA/actions/videos" | sed -E 's/_chunk_[0-9]+_target\.pt$//' | sort -u | paste -sd, -)"
NEPS="$(echo "$EPS" | tr ',' '\n' | wc -l)"

if [ "$MODE" = "prope" ]; then
  CKDIR=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/outputs/gvt/libero90_v2v_ic_lora_prope/checkpoints
  PROPE_FLAGS="--use-prope --prope-proj-dim 64 --prope-image-width 256 --prope-image-height 256"
  OUTDIR="$OUTROOT/eval_prope_episode"; TAG=prope
else
  CKDIR=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/outputs/gvt/libero90_v2v_ic_lora/checkpoints
  PROPE_FLAGS=""
  OUTDIR="$OUTROOT/eval_noprope_episode"; TAG=noprope
fi
mkdir -p "$OUTDIR"

echo "============== OOD eval =============="
echo "JobID : ${SLURM_JOB_ID:-N/A} | Node: ${SLURM_NODELIST:-N/A}"
echo "MODE  : ${MODE}  STEP : ${STEP}  episodes: ${NEPS}"
echo "LoRA  : $CKDIR/lora_weights_step_${STEP}.safetensors"
echo "Data  : $DATA"
echo "Out   : $OUTDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u experiments/libero_sim/infer_action_cond.py \
  --lora-path "$CKDIR/lora_weights_step_${STEP}.safetensors" \
  --action-projector "$CKDIR/action_projector_step_${STEP}.safetensors" \
  --data-root "$DATA" \
  $PROPE_FLAGS \
  --episode "$EPS" \
  --include-gt --include-context --num-inference-steps 30 \
  --tag "$TAG" --out-dir "$OUTDIR"

echo "=== ${MODE} step ${STEP} OOD eval DONE at $(date) ==="
