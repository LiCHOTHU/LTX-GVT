#!/usr/bin/env bash
# run_active_learning.sbatch — ONE GPU job running ONE arm of the active-learning ablation.
#
# This is a THIN wrapper: it only sets up the environment and then runs ONE Python
# program (active_learning_loop.py), which IS the loop (build -> grade -> keep -> train,
# rounds 0..49, step 30000 -> 80000, in-process on this one GPU). No bash loop, no child
# jobs, no marker-file coordination in here — all of that lives in the Python file.
#
# WHICH ARM this job trains is set by the VERSIONS env var (the loop iterates only over
# VERSIONS). The monitor submits two of these jobs, one per arm:
#   strategic ->  sbatch --job-name=act_ltx  --export=ALL,VERSIONS=strategic  run_active_learning.sbatch
#   random    ->  sbatch --job-name=rand_ltx --export=ALL,VERSIONS=random     run_active_learning.sbatch
# Each arm has its own checkpoint dir (libero90_ALR_<arm>) and tracks its own steps, so
# the two are a clean strategic-vs-random comparison. If VERSIONS is unset, the loop's
# default ("strategic random") would run BOTH arms in one job — the monitor always sets it.
#
# It does NOT resubmit itself. The thin keepalive monitor (monitor_active_learning.sh)
# keeps each arm's job alive: it checks every 15 min and resubmits if the job died or was
# preempted. On resubmit the Python loop resumes from disk (finished stages leave markers;
# training auto-resumes from that arm's last checkpoint).
#
# Submit one arm by hand for a smoke test, or let the monitor own both:
#   sbatch --job-name=act_ltx --export=ALL,VERSIONS=strategic run_active_learning.sbatch
# Dry check (no GPU, prints every command):
#   VERSIONS=strategic DRY_RUN=1 python experiments/libero_sim/active_learning_loop.py

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH -p gpu-h200
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --job-name=act_ltx
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.err

set -uo pipefail

REPO="/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT"
LSIM="${REPO}/experiments/libero_sim"
USER_LIBERO="/storage/home/hcoda1/8/lwang831/workspace/LIBERO"
CONDA_ENV="ltx"

cd "${REPO}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Re-source the arm's config at JOB time so every knob (SCORER, PER_TASK_MIX, PFM_*, ...) is
# authoritative from the .env, not from whatever the (possibly stale) submitting watchdog process
# had exported. The watchdog passes its config path via AL_CONFIG. VERSIONS is set per-arm via
# --export and is NOT in the .env, so it survives this re-source. Belt-and-suspenders vs the
# --export=ALL chain silently dropping vars (observed: PFM_ENABLE lost on watchdog resubmit).
if [[ -n "${AL_CONFIG:-}" && -f "${AL_CONFIG}" ]]; then
  echo "[sbatch] sourcing arm config: ${AL_CONFIG}"
  set -a; source "${AL_CONFIG}"; set +a
else
  echo "[sbatch] WARNING: AL_CONFIG unset or missing (${AL_CONFIG:-<unset>}); relying on --export env only"
fi

# Env union of build (LIBERO sim render) + score + train needs.
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="${USER_LIBERO}:${LSIM}:${REPO}/packages/ltx-trainer/src:${REPO}/packages/ltx-core/src:${PYTHONPATH:-}"
export TMPDIR="/storage/project/r-agarg35-0/lwang831/tmp"
export MPLCONFIGDIR="${TMPDIR}/mpl"
unset HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}" "${HF_HOME}" /storage/project/r-agarg35-0/lwang831/outputs/gvt/logs

echo "================ active-learning loop (one GPU job) ================"
echo "Arm   : VERSIONS=${VERSIONS:-<unset: defaults to BOTH>}"
echo "JobID : ${SLURM_JOB_ID:-N/A} | Node: ${SLURM_NODELIST:-N/A}"
echo "Start : $(date)"
echo "==================================================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

exec python -u "${LSIM}/active_learning_loop.py"
