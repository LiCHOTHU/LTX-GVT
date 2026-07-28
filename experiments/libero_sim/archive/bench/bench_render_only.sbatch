#!/usr/bin/env bash
# bench_render_only.sbatch — recover the full [TIME-render] per-stage split using
# the already-captured npz (no grep filter; full stdout to log).
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH -p gpu-h200
#SBATCH --time=00:15:00
#SBATCH --mem-per-gpu=64G
#SBATCH --job-name=bench_ro
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --requeue
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.err

set -euo pipefail
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT"
LSIM="${WORKDIR}/experiments/libero_sim"
USER_LIBERO="/storage/home/hcoda1/8/lwang831/workspace/LIBERO"
NPZ="/storage/project/r-agarg35-0/lwang831/tmp/libero_capture_bench_render_split.npz"
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

echo "=== FK render pass split (render-only, cached npz) ==="
echo "Start: $(date)"
python -u "${LSIM}/build_libero_context.py" \
    --hdf5 dummy --demo 0 --ep_id bench_render_only \
    --out_root "${OUT}" --res 256 --fps 15 --mode render --inter "${NPZ}"
echo "=== DONE $(date) ==="
