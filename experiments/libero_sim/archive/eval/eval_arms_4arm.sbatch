#!/usr/bin/env bash
# eval_arms_4arm.sbatch — head-to-head held-out eval of the 4 object-centric AL arms.
#
#   rand_obj|random      control (random keep)
#   maxloss_obj|maxloss  raw object-region loss, top-K   (reproduces the Jul-8 null)
#   rho_div_obj|rho_div  reducible loss + diverse batch
#   decaware_obj|decaware reducible x policy-relevance + diverse   (proposed)
#
# MATCHED STEP = 46000. The arms stopped at different steps (random/maxloss 50000,
# rho_div 48000, decaware 46000) because the last two crash-looped on the scratch1
# inode quota. Evaluating them where they stopped would hand the control 4000 extra
# training steps and bias the test against the proposed method, so we compare at the
# largest step EVERY arm reached. All arms have lora+projector at 46000 (= rounds 0-15).
#
# EVAL POOL = round-16 candidates. At step 46000 no arm had trained on round>=16 yet,
# so it is genuinely held out for all four. Pools are byte-identical across arms
# (GT-seeded perturb + same per-round seed), so any arm's round-16 pool serves as the
# one shared eval set; we take random's. Restricted to off-manifold perturb+random.
#
# Inference only; loads the 22B base once and hot-swaps each arm's adapter.
#   sbatch eval_arms_4arm.sbatch
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH -p gpu-h200
#SBATCH --time=06:00:00
#SBATCH --mem=128G
#SBATCH --job-name=eval4arm
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.err

set -uo pipefail
WD=/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT
LSIM="$WD/experiments/libero_sim"
cd "$WD"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate ltx
export TMPDIR=/storage/project/r-agarg35-0/lwang831/tmp
unset HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache; export HF_HUB_CACHE="$HF_HOME/hub"
export MPLCONFIGDIR="$TMPDIR/mpl"
export PYTHONPATH="$LSIM:$WD/packages/ltx-trainer/src:$WD/packages/ltx-core/src:${PYTHONPATH:-}"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"

STEP="${STEP:-46000}"
B=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/outputs/gvt_obj_single_4arm
POOL="${POOL:-/storage/scratch1/8/lwang831/obj_single_4arm/libero90_ALR_random_round16/precomputed}"
# results live on cedar (scratch1 is out of inodes and auto-purges)
OUT="${OUT:-$B/evals/step${STEP}}"
PROPE_FLAGS="--use-prope --prope-proj-dim 64 --prope-image-width 256 --prope-image-height 256"

echo "============== 4-ARM held-out eval =============="
echo "JobID : ${SLURM_JOB_ID:-N/A} | Node: ${SLURM_NODELIST:-N/A}"
echo "STEP  : $STEP (matched across all 4 arms)"
echo "POOL  : $POOL  (round-16 = held out for every arm at step $STEP)"
echo "OUT   : $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Start : $(date)"

for a in random maxloss rho_div decaware; do
  f="$B/libero90_ALR_$a/checkpoints/lora_weights_step_${STEP}.safetensors"
  [ -f "$f" ] || { echo "MISSING $a LoRA @ $STEP: $f"; exit 1; }
done
[ -d "$POOL/latents/videos" ] || { echo "MISSING eval pool: $POOL"; exit 1; }
mkdir -p "$OUT"

python -u "$LSIM/eval_arms_local.py" \
  --base-checkpoint /storage/scratch1/8/lwang831/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors \
  --arm random="$B/libero90_ALR_random/checkpoints" \
  --arm maxloss="$B/libero90_ALR_maxloss/checkpoints" \
  --arm rho_div="$B/libero90_ALR_rho_div/checkpoints" \
  --arm decaware="$B/libero90_ALR_decaware/checkpoints" \
  --baseline random \
  --step "$STEP" \
  --eval-pool "$POOL" \
  --kinds perturb,random \
  --max-eps-per-kind 10 --max-chunks-per-ep 4 \
  --num-inference-steps 30 --drop-first 8 \
  $PROPE_FLAGS \
  --out-dir "$OUT"

echo "=== 4-arm eval DONE at $(date) ==="
echo "Results: $OUT/arm_compare.{json,txt}"
