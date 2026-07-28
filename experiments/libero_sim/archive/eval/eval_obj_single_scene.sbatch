#!/bin/bash
#SBATCH --job-name=eval_objsc
#SBATCH --account=gts-agarg35
#SBATCH --partition=gpu-a100
#SBATCH -C A100-80GB
#SBATCH --qos=embers
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
# Single-scene AL eval (the run we actually trained per-scene): gvt_obj_single,
# strategic vs random, matched step 46250, on the TRAINING scene only
# (LIVING_ROOM_SCENE1 pick_up_the_tomato_sauce), across ALL trajectory kinds
# (reach / perturb / random) held out in the round4 pool. Renders pred|gt|ctx.
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ltx
WD=/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT
cd "$WD"

STEP=46250
BASE_CKPT=/storage/scratch1/8/lwang831/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors
PROPE_FLAGS="--use-prope --prope-proj-dim 64 --prope-image-width 256 --prope-image-height 256"
# Both arms evaluated on the SAME held-out pool (strategic's round4) so episodes are identical.
POOL=/storage/scratch1/8/lwang831/libero90_ALR_strategic_round4/precomputed
OUTROOT="$WD/experiments/libero_sim/outputs/obj_single_scene_step${STEP}"

# One representative held-out episode per trajectory kind (all from the training scene).
STEM=LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket_demo_play
EPS="${STEM}_reach_demo_0,${STEM}_reach_demo_1,${STEM}_perturb_demo_0,${STEM}_perturb_demo_1,${STEM}_perturb_demo_2,${STEM}_random_demo_0,${STEM}_random_demo_1"

echo "Start: $(date)  Node: ${SLURM_NODELIST:-N/A}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for ARM in strategic random; do
  CKDIR=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/outputs/gvt_obj_single/libero90_ALR_${ARM}/checkpoints
  LORA="$CKDIR/lora_weights_step_${STEP}.safetensors"
  PROJ="$CKDIR/action_projector_step_${STEP}.safetensors"
  OUT="$OUTROOT/${ARM}"; mkdir -p "$OUT"
  [ -f "$LORA" ] || { echo "MISSING LoRA: $LORA"; exit 1; }
  echo "===== ARM $ARM step $STEP -> $OUT ====="
  python -u experiments/libero_sim/infer_action_cond.py \
    --base-checkpoint "$BASE_CKPT" \
    --lora-path "$LORA" --action-projector "$PROJ" \
    --data-root "$POOL" $PROPE_FLAGS \
    --episode "$EPS" \
    --include-gt --include-context --num-inference-steps 30 \
    --tag "objsc_${ARM}" --out-dir "$OUT"
done
echo "=== single-scene obj eval DONE $(date) ==="
echo "Videos: $OUTROOT/{strategic,random}/"
