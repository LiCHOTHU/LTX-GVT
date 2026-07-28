#!/usr/bin/env bash
# SLURM wrapper around download_droid_full.sh.
#
# CPU-only job: pure network I/O, no GPU needed. Long wall-clock because
# embers can preempt — `gsutil rsync` is idempotent so a preempted+requeued
# job just resumes where it left off.

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --job-name=droid_dl
#SBATCH --qos=embers
#SBATCH --requeue
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.err

set -euo pipefail

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT"
SCRIPT="${WORKDIR}/experiments/droid_action_cond/download_droid_full.sh"

mkdir -p /storage/project/r-agarg35-0/lwang831/outputs/gvt/logs

echo "================ Job Info ================"
echo "JobID  : ${SLURM_JOB_ID:-N/A}"
echo "Node   : ${SLURM_NODELIST:-N/A}"
echo "Start  : $(date)"
echo "=========================================="

# Reuse the standalone shell script — single source of truth.
exec bash "${SCRIPT}"
