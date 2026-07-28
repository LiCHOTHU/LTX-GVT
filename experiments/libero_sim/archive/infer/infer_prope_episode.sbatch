#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=128G
#SBATCH --job-name=prope_ep
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.err
#
# Re-run the PRoPE episode eval with the FIXED inference path (--use-prope:
# swaps attn1 -> PRoPE attention + feeds per-token cameras). The prior eval was
# garbage because the script ran the PRoPE LoRA under vanilla 3D RoPE with no
# cameras. Dedicated L40s = full 46 GB for the ~44 GB bf16 22B (no offload).

set -euo pipefail
WD=/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT
cd "$WD"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate ltx
export TMPDIR=/storage/project/r-agarg35-0/lwang831/tmp
unset HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache; export HF_HUB_CACHE="$HF_HOME/hub"
export MPLCONFIGDIR="$TMPDIR/mpl"
export PYTHONPATH="$WD/packages/ltx-trainer/src:$WD/packages/ltx-core/src:${PYTHONPATH:-}"

CK=/storage/project/r-agarg35-0/lwang831/outputs/gvt/libero90_v2v_ic_lora_prope/checkpoints
python -u experiments/libero_sim/infer_action_cond.py \
  --lora-path "$CK/lora_weights_step_30000.safetensors" \
  --use-prope --prope-proj-dim 64 --prope-image-width 256 --prope-image-height 256 \
  --episode "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it_demo_demo_0,KITCHEN_SCENE7_open_the_microwave_demo_demo_42" \
  --also-no-actions --include-gt --include-context --num-inference-steps 30 \
  --tag prope --out-dir "$WD/experiments/libero_sim/outputs/eval_prope_episode"
echo "=== prope_ep DONE ==="
