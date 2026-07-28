#!/bin/bash
#SBATCH --job-name=eval_4arm
#SBATCH --account=gts-agarg35
#SBATCH --partition=gpu-a100
#SBATCH -C A100-80GB
#SBATCH --qos=embers
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=%x_%j.out
# LATEST single-task AL run: gvt_obj_single_4arm (LIVING_ROOM_SCENE1 pick tomato_sauce).
# 4 acquisition arms {random, maxloss, decaware, rho_div}, matched step 46000
# (decaware's max). Eval ONLY on the training task, across all held-out trajectory
# kinds (reach / perturb / random) from the round16 pool. Renders pred|gt|ctx.
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ltx
WD=/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT
cd "$WD"

STEP=46000
BASE_CKPT=/storage/scratch1/8/lwang831/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors
PROPE_FLAGS="--use-prope --prope-proj-dim 64 --prope-image-width 256 --prope-image-height 256"
POOL=/storage/scratch1/8/lwang831/obj_single_4arm/libero90_ALR_random_round16/precomputed
OUTROOT="$WD/experiments/libero_sim/outputs/single_task_4arm_step${STEP}"

STEM=LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket_demo_play
# two episodes per trajectory kind (all held out, all the single training task)
EPS="${STEM}_reach_demo_0,${STEM}_reach_demo_1,${STEM}_perturb_demo_0,${STEM}_perturb_demo_1,${STEM}_random_demo_0,${STEM}_random_demo_1"

echo "Start: $(date)  Node: ${SLURM_NODELIST:-N/A}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for ARM in random maxloss rho_div decaware; do
  CKDIR=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/outputs/gvt_obj_single_4arm/libero90_ALR_${ARM}/checkpoints
  LORA="$CKDIR/lora_weights_step_${STEP}.safetensors"
  PROJ="$CKDIR/action_projector_step_${STEP}.safetensors"
  OUT="$OUTROOT/${ARM}"; mkdir -p "$OUT"
  [ -f "$LORA" ] || { echo "MISSING LoRA: $LORA — skipping $ARM"; continue; }
  echo "===== ARM $ARM step $STEP -> $OUT ====="
  python -u experiments/libero_sim/infer_action_cond.py \
    --base-checkpoint "$BASE_CKPT" \
    --lora-path "$LORA" --action-projector "$PROJ" \
    --data-root "$POOL" $PROPE_FLAGS \
    --episode "$EPS" \
    --include-gt --include-context --num-inference-steps 30 \
    --tag "st4_${ARM}" --out-dir "$OUT"
done
echo "=== single-task 4-arm eval DONE $(date) ==="
echo "Videos: $OUTROOT/{random,maxloss,rho_div,decaware}/"
