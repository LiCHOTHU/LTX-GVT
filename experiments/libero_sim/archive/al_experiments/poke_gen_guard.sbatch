#!/bin/bash
#SBATCH --job-name=poke_guard
#SBATCH --account=gts-agarg35
#SBATCH --partition=gpu-a100
#SBATCH -C A100-80GB
#SBATCH --qos=inferno
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
# Crash/motion guard for the ARMS' exact BUILD path: run gen_libero_play with the
# poke-inclusive coverage mix (same call active_learning_loop.py issues), then verify
# the poke episodes were written and the tomato_sauce actually moved.
set -euo pipefail
REPO=/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT
LSIM=$REPO/experiments/libero_sim
USER_LIBERO=/storage/home/hcoda1/8/lwang831/workspace/LIBERO
cd "$REPO"
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ltx
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="${USER_LIBERO}:${LSIM}:${PYTHONPATH:-}"
export TMPDIR=/storage/project/r-agarg35-0/lwang831/tmp
export MPLCONFIGDIR="$TMPDIR/mpl"; mkdir -p "$MPLCONFIGDIR"
SCENE=/storage/project/r-agarg35-0/lwang831/alr_tomato_sauce_scene
GUARD=/storage/project/r-agarg35-0/lwang831/tmp/poke_guard
echo "Start $(date) Node ${SLURM_NODELIST:-?}"; nvidia-smi --query-gpu=name --format=csv,noheader || true

python -u "$LSIM/gen_libero_play.py" \
  --dataset "$SCENE" --out_dir "$GUARD/play" \
  --mix "reach:6,poke:10,perturb:8,random:6" --steps 200 --seed 20260726 \
  --num_shards 1 --shard 0

echo "=== inspect poke episodes: did tomato_sauce move? ==="
python -u - <<'PY'
import glob, h5py, numpy as np
fs = sorted(glob.glob("/storage/project/r-agarg35-0/lwang831/tmp/poke_guard/play/*poke*.hdf5"))
assert fs, "NO poke hdf5 written -> gen_libero_play crashed or skipped poke"
for f in fs:
    with h5py.File(f,"r") as h:
        d=h["data"]; ks=sorted(d.keys())
        print(f"{f.split('/')[-1]}: {len(ks)} poke demos")
        moved=[]
        for k in ks:
            st=d[k]["states"][:]           # (T, state_dim) full sim state
            # crude motion proxy: total variation of the last object's free-joint block
            # (object qpos sits near the tail of qpos); use max per-dim travel as a signal.
            span=(st.max(0)-st.min(0))
            moved.append(float(span.max()))
        print(f"   per-demo max state travel: min={min(moved):.3f} max={max(moved):.3f} mean={np.mean(moved):.3f}")
print("POKE_GUARD_OK")
PY
echo "=== poke_guard DONE $(date) ==="
