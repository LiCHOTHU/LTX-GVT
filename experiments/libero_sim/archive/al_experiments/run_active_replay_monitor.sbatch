#!/usr/bin/env bash
# run_active_replay_monitor.sbatch — run the active-learning MONITOR as a SLURM job.
#
# This is the "monitor style" loop, kept alive the right way. The monitor
# (monitor_active_replay_libero.sh) is the babysitter: it submits the BUILD (l40s),
# SCORE + TRAIN (h200) jobs for each round and coordinates them through marker files.
# Its ONLY weakness was that running it with plain `bash` on a login node killed it the
# moment the terminal closed. Running it as THIS small CPU job fixes that:
#
#   * It lives inside SLURM, so it survives your SSH disconnect.
#   * cpu-small allows up to 21 days of walltime — the whole 30000->80000 run fits in
#     one allocation, no self-resubmit needed.
#   * If embers preempts it, --requeue restarts it; the monitor re-reads its on-disk
#     state and continues (it is idempotent — init returns early once seeded).
#
# It uses ZERO GPU itself; it only submits GPU jobs and runs the cheap select/buffer
# python inline. Training still happens in the dedicated h200 train jobs, marching
# 30000 -> 80000 in ROUND_STEPS increments (now 1000/round x 50 rounds).
#
# Submit once:   sbatch experiments/libero_sim/run_active_replay_monitor.sbatch
# Watch:         tail -f on this job's .out log (path printed at startup), or squeue.
# Stop all:      scancel THIS job id, then scancel any al_* child jobs it launched.

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -p cpu-small
#SBATCH --time=7-00:00:00
#SBATCH --mem=8G
#SBATCH --job-name=al_mon
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

# Same env the monitor's inline python (replay-buffer seed/append, select) expects.
export PYTHONPATH="${USER_LIBERO}:${LSIM}:${REPO}/packages/ltx-trainer/src:${REPO}/packages/ltx-core/src:${PYTHONPATH:-}"
export TMPDIR="/storage/project/r-agarg35-0/lwang831/tmp"
export MPLCONFIGDIR="${TMPDIR}/mpl"
unset HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}" /storage/project/r-agarg35-0/lwang831/outputs/gvt/logs

echo "================ active-learning monitor (in SLURM) ================"
echo "JobID : ${SLURM_JOB_ID:-N/A} | Node: ${SLURM_NODELIST:-N/A}"
echo "Start : $(date)"
echo "Loop  : monitor_active_replay_libero.sh  (30000 -> 80000, 1000/round x 50)"
echo "==================================================================="

# Run the monitor in the foreground; its sbatch submissions + 5-min ticks proceed here.
exec bash "${LSIM}/monitor_active_replay_libero.sh"
