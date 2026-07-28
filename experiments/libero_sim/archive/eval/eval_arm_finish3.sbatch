#!/bin/bash
#SBATCH --job-name=eval_fin3
#SBATCH --account=gts-agarg35
#SBATCH --partition=gpu-a100
#SBATCH -C A100-80GB
#SBATCH --qos=embers
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
# Finish the 3 circle episodes strategic missed after preemption.
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ltx
WD=/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT
cd "$WD"

ARM=strategic
STEP=80000
BASE_CKPT=/storage/scratch1/8/lwang831/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors
CKDIR="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/outputs/gvt_alv2/libero90_ALR_${ARM}/checkpoints"
LORA="$CKDIR/lora_weights_step_${STEP}.safetensors"
PROJ="$CKDIR/action_projector_step_${STEP}.safetensors"
PROPE_FLAGS="--use-prope --prope-proj-dim 64 --prope-image-width 256 --prope-image-height 256"
CIRCLE=/storage/scratch1/8/lwang831/libero90_circle_precomputed/precomputed
OUT="$WD/experiments/libero_sim/outputs/al_eval_step${STEP}/${ARM}/circle"
mkdir -p "$OUT"

EPS="LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket_demo_play_circle_demo_0,LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_demo_play_circle_demo_0,STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy_demo_play_circle_demo_0"

echo "Start: $(date)  Node: ${SLURM_NODELIST:-N/A}"
python -u experiments/libero_sim/infer_action_cond.py \
  --base-checkpoint "$BASE_CKPT" \
  --lora-path "$LORA" --action-projector "$PROJ" \
  --data-root "$CIRCLE" $PROPE_FLAGS \
  --episode "$EPS" \
  --include-gt --include-context --num-inference-steps 30 \
  --tag "${ARM}_circle" --out-dir "$OUT"
echo "DONE $(date)"
