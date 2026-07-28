#!/usr/bin/env bash
# render_test.sbatch — verify (1) pyrender still renders after the PyOpenGL 3.1.7
# upgrade (protects the DROID build), and (2) we can replay a LIBERO demo and
# render 3 views (2 fixed + 1 wrist) to mp4.

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=00:25:00
#SBATCH --mem=32G
#SBATCH --job-name=libero_render
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.err

set -euo pipefail

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT"
LSIM="${WORKDIR}/experiments/libero_sim"
# LIBERO is the user's clone; PEP 660 editable install is broken (empty finder MAPPING),
# so we put the package's parent dir on PYTHONPATH explicitly.
USER_LIBERO="/storage/home/hcoda1/8/lwang831/workspace/LIBERO"
OUT="/storage/scratch1/8/lwang831/libero_replay_test"
DATASET="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/LIBERO-datasets/libero_90"
DEMO="KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it_demo.hdf5"

cd "${WORKDIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ltx

# Headless GL backends.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TMPDIR="/storage/project/r-agarg35-0/lwang831/tmp"
export MPLCONFIGDIR="${TMPDIR}/mpl"
export PYTHONPATH="${USER_LIBERO}:${PYTHONPATH:-}"
mkdir -p "${OUT}" "${TMPDIR}" "${MPLCONFIGDIR}"

echo "================ render_test ================"
echo "JobID : ${SLURM_JOB_ID:-N/A} | Node: ${SLURM_NODELIST:-N/A}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -V

echo "=== [1/2] pyrender post-upgrade smoke render (DROID build safety) ==="
python - <<'PY'
import numpy as np, trimesh, pyrender
m = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
scene = pyrender.Scene()
scene.add(pyrender.Mesh.from_trimesh(m))
cam = pyrender.PerspectiveCamera(yfov=np.pi/3.0)
import numpy as np
pose = np.eye(4); pose[2, 3] = 0.8
scene.add(cam, pose=pose)
scene.add(pyrender.DirectionalLight(intensity=3.0), pose=pose)
r = pyrender.OffscreenRenderer(128, 128)
color, _ = r.render(scene)
r.delete()
assert color.shape == (128, 128, 3) and color.sum() > 0, "pyrender produced empty frame"
print(f"[OK] pyrender rendered {color.shape}, nonzero pixels={int((color.sum(-1)>0).sum())}")
PY

echo "=== [2/2] LIBERO replay -> 3-view render ==="
python -u "${LSIM}/replay_libero.py" \
    --hdf5 "${DATASET}/${DEMO}" \
    --demos 0 \
    --out_dir "${OUT}" \
    --res 256 \
    --fps 20 \
    --save_per_view

echo "=== outputs ==="
ls -lh "${OUT}"
echo "Finished at $(date)"
