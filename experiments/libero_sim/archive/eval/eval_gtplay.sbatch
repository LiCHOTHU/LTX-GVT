#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=128G
#SBATCH --job-name=ev_gtpl
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.err
#
# eval_gtplay.sbatch — render full-episode pred|gt|ctx rollouts for ONE model
# (PRoPE or non-PRoPE) on a balanced GT + play/rand episode set, so we can score
# PSNR/SSIM on both the on-manifold (GT task demos) and off-manifold (play) data.
#
# Submit twice (same matched step for a fair head-to-head):
#   sbatch --export=ALL,MODE=prope   eval_gtplay.sbatch
#   sbatch --export=ALL,MODE=noprope eval_gtplay.sbatch
#
# Env knobs: MODE={prope|noprope} (default prope), STEP (default 54000 — the
# latest checkpoint step BOTH runs still retain LoRA weights for).
# Dedicated L40s = full 46 GB for the ~44 GB bf16 22B (no offload).

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
DATA=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/libero90_combined_precomputed/precomputed
OUTROOT="$WD/experiments/libero_sim/outputs/eval_gtplay_step${STEP}"

# Balanced eval set: 2 GT (on-manifold task demos) + 4 play (off-manifold).
# Play covers all three policies; SCENE10 carries reach/perturb/random so the
# GT-vs-play contrast is within the SAME task/scene.
GT1="KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it_demo_demo_0"
GT2="KITCHEN_SCENE7_open_the_microwave_demo_demo_42"
PL1="KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it_demo_play_reach_demo_0"
PL2="KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it_demo_play_perturb_demo_0"
PL3="KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it_demo_play_random_demo_0"
PL4="KITCHEN_SCENE7_open_the_microwave_demo_play_reach_demo_0"
EPS="${GT1},${GT2},${PL1},${PL2},${PL3},${PL4}"

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

echo "============== GT+play eval =============="
echo "JobID : ${SLURM_JOB_ID:-N/A} | Node: ${SLURM_NODELIST:-N/A}"
echo "MODE  : ${MODE}  STEP : ${STEP}"
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
  --also-no-actions --include-gt --include-context --num-inference-steps 30 \
  --tag "$TAG" --out-dir "$OUTDIR"

echo "=== ${MODE} step ${STEP} DONE at $(date) ==="
