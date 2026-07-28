#!/usr/bin/env bash
# bench_render_split.sbatch — measure the exact FK-context-render split for ONE
# LIBERO episode (capture: MuJoCo RGB+depth+seg per frame/cam ; render: pyrender
# mesh + wrist splat). GVT_TIME=1 prints [TIME-capture] / [TIME-render] breakdowns.
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH -p gpu-h200
#SBATCH --time=00:20:00
#SBATCH --mem-per-gpu=64G
#SBATCH --job-name=bench_r
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --requeue
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.err

set -euo pipefail
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT"
LSIM="${WORKDIR}/experiments/libero_sim"
USER_LIBERO="/storage/home/hcoda1/8/lwang831/workspace/LIBERO"
H5="/storage/scratch1/8/lwang831/libero90_gt_play/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_demo_play_gt.hdf5"
OUT="/storage/scratch1/8/lwang831/libero90_bench_render"

cd "${WORKDIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ltx
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="${USER_LIBERO}:${LSIM}:${WORKDIR}/packages/ltx-action-cond/src:${WORKDIR}/packages/ltx-core/src:${PYTHONPATH:-}"
export TMPDIR="/storage/project/r-agarg35-0/lwang831/tmp"
export MPLCONFIGDIR="${TMPDIR}/mpl"
export GVT_TIME=1
mkdir -p "${OUT}" "${TMPDIR}" "${MPLCONFIGDIR}"

echo "=== FK context render split (1 episode, 3 cams, res 256) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
echo "Start: $(date)"

# /usr/bin/time -v gives the wall clock of the WHOLE --mode both run (= capture
# subprocess + render subprocess + 2x interpreter startup). The [TIME-*] lines
# inside give the per-pass internal split; the gap to wall = subprocess/import overhead.
/usr/bin/time -v python -u "${LSIM}/build_libero_context.py" \
    --hdf5 "${H5}" --demo 0 --ep_id bench_render_split \
    --out_root "${OUT}" --res 256 --fps 15 --mode both 2>&1 | \
    grep -E "TIME-|capture\]|render\]|chunk |done\]|Elapsed \(wall|Maximum resident" || true

echo "=== DONE $(date) ==="
