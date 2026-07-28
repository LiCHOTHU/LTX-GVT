#!/bin/bash
#SBATCH --job-name=eval_objreg
#SBATCH --account=gts-agarg35
#SBATCH --partition=gpu-a100
#SBATCH -C A100-80GB
#SBATCH --qos=embers
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
# OBJECT-REGION eval matching the committed thesis: "learned physics" = correct
# pixel prediction of the target object's response to interactions. Scores mean
# |pred-gt| ONLY inside the projected 3D object bbox (score_chunk_objbbox_pixel.py),
# NOT full-frame. Latest single-task run gvt_obj_single_4arm, matched step 46000,
# held-out round16 pool (all trajectory kinds). Resume-safe: skips arms already done.
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ltx
WD=/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT
cd "$WD"

STEP=46000
POOL=/storage/scratch1/8/lwang831/obj_single_4arm/libero90_ALR_random_round16/precomputed
PROPE_FLAGS="--use-prope --prope-proj-dim 64 --prope-image-width 256 --prope-image-height 256"
OUTDIR="$WD/experiments/libero_sim/outputs/objregion_4arm_step${STEP}"
mkdir -p "$OUTDIR"

echo "Start: $(date)  Node: ${SLURM_NODELIST:-N/A}"; nvidia-smi --query-gpu=name --format=csv,noheader || true

for ARM in random maxloss rho_div decaware; do
  CKDIR=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/outputs/gvt_obj_single_4arm/libero90_ALR_${ARM}/checkpoints
  LORA="$CKDIR/lora_weights_step_${STEP}.safetensors"
  PROJ="$CKDIR/action_projector_step_${STEP}.safetensors"
  # -- (A) absolute object-region error at step 46000 --
  OUT_ABS="$OUTDIR/${ARM}_objloss.json"
  if [ -f "$OUT_ABS" ]; then echo "skip $ARM abs (exists)"; else
    echo "===== [$ARM] absolute object-region error (step $STEP) ====="
    python -u experiments/libero_sim/score_chunk_objbbox_pixel.py \
      --lora-path "$LORA" --action-projector "$PROJ" \
      --data-root "$POOL" $PROPE_FLAGS \
      --num-inference-steps 30 --score-metric loss \
      --out "$OUT_ABS"
  fi
  # -- (B) reducible object error learned since the shared step-30000 ancestor --
  BASE_LORA="$CKDIR/lora_weights_step_30000.safetensors"
  BASE_PROJ="$CKDIR/action_projector_step_30000.safetensors"
  OUT_PROG="$OUTDIR/${ARM}_objprogress.json"
  if [ -f "$OUT_PROG" ]; then echo "skip $ARM progress (exists)"; else
    echo "===== [$ARM] reducible object error vs step-30000 ====="
    python -u experiments/libero_sim/score_chunk_objbbox_pixel.py \
      --lora-path "$LORA" --action-projector "$PROJ" \
      --baseline-lora "$BASE_LORA" --baseline-projector "$BASE_PROJ" \
      --data-root "$POOL" $PROPE_FLAGS \
      --num-inference-steps 30 --score-metric progress \
      --out "$OUT_PROG"
  fi
done
echo "=== object-region 4-arm eval DONE $(date) ==="
echo "JSONs: $OUTDIR/*_objloss.json  (+ *_objprogress.json)"
