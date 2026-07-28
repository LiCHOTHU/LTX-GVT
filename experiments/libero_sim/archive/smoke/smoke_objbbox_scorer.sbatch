#!/usr/bin/env bash
# smoke_objbbox_scorer.sbatch — fast END-TO-END smoke of the object-centric pixel scorer
# on a big GPU (the V100 used for the geometry smoke can't hold the 22B). Reuses the
# already-built single-scene context under $SM/context, finishes the precompute streams
# (encode + cameras + objbbox), then runs score_chunk_objbbox_pixel on a HANDFUL of chunks
# with few rollout steps. NO training. Confirms the 22B rollout+decode+masked-diff path
# produces finite, varied per-chunk curiosity + a ranking.
#
#   sbatch experiments/libero_sim/smoke_objbbox_scorer.sbatch
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH -p gpu-a100
#SBATCH --constraint=A100-80GB
#SBATCH --time=01:00:00
#SBATCH --mem-per-gpu=128G
#SBATCH --job-name=smoke_obj
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --requeue
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.err

set -euo pipefail
WD=/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT
cd "$WD"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate ltx
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="/storage/home/hcoda1/8/lwang831/workspace/LIBERO:$WD/experiments/libero_sim:$WD/packages/ltx-trainer/src:$WD/packages/ltx-core/src:${PYTHONPATH:-}"
export TMPDIR=/storage/project/r-agarg35-0/lwang831/tmp
unset HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache; export HF_HUB_CACHE="$HF_HOME/hub"
export MPLCONFIGDIR="$TMPDIR/mpl"; mkdir -p "$TMPDIR" "$MPLCONFIGDIR"

SM=/storage/project/r-agarg35-0/lwang831/tmp/objbbox_smoke
CKDIR=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/outputs/gvt/libero90_v2v_ic_lora_prope/checkpoints
LORA="$CKDIR/lora_weights_step_30000.safetensors"
PROJ="$CKDIR/action_projector_step_30000.safetensors"

echo "============== objbbox scorer smoke =============="
echo "JobID: ${SLURM_JOB_ID:-N/A}  Node: ${SLURM_NODELIST:-N/A}  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
[ -d "$SM/context" ] || { echo "MISSING $SM/context — run the build smoke first"; exit 1; }

echo "----- encode (VAE + text encoder) -----"
python -u experiments/libero_sim/encode_libero_precomputed.py \
  --src_root "$SM/context" --out_root "$SM" --shard 0/1 --batch_size 4 \
  --max_chunks 0 --load_text_encoder_in_8bit
echo "----- cameras + objbbox -----"
python -u experiments/libero_sim/build_libero_cameras.py --src_root "$SM/context" --out_root "$SM" --shard 0/1 --max_chunks 0
python -u experiments/libero_sim/build_libero_objbbox.py --src_root "$SM/context" --out_root "$SM" --shard 0/1 --max_chunks 0

echo "----- grade (object-centric pixel curiosity, few steps, few chunks) -----"
python -u experiments/libero_sim/score_chunk_objbbox_pixel.py \
  --lora-path "$LORA" --action-projector "$PROJ" --data-root "$SM/precomputed" \
  --use-prope --prope-proj-dim 64 --prope-image-width 256 --prope-image-height 256 \
  --num-inference-steps 8 --max-chunks 6 --out "$SM/smoke_score.json"

echo "----- ranking (csv) -----"
cat "$SM/smoke_score.csv"
echo "=== objbbox scorer smoke DONE $(date) ==="
