#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -p cpu-small
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --job-name=gvt_move
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.err
#
# Move the DROID build output scratch1 -> cedar project share on a CPU node.
# rsync is idempotent: re-running resumes (already-copied files skipped on
# size+mtime), so this is safe across requeue/preemption. Does NOT delete the
# scratch source -- that is a separate, manual step after counts verify.

set -uo pipefail

SRC=/storage/scratch1/8/lwang831/gvt_dataset_full
DST=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/gvt_dataset_full

echo "=== gvt_move START $(date) on $(hostname) ==="
echo "SRC=${SRC}"
echo "DST=${DST}"

mkdir -p "${DST}"

# -a archive, partial-resume, human stats. No per-file progress (log stays small).
rsync -a --partial --info=stats2 "${SRC}/" "${DST}/"
RC=$?
echo "rsync exit code: ${RC}"

echo "=== verify counts ==="
SRC_DONE=$(find "${SRC}/_done" -type f 2>/dev/null | wc -l)
DST_DONE=$(find "${DST}/_done" -type f 2>/dev/null | wc -l)
SRC_LAT=$(find "${SRC}/precomputed/latents" -type f 2>/dev/null | wc -l)
DST_LAT=$(find "${DST}/precomputed/latents" -type f 2>/dev/null | wc -l)
SRC_ALL=$(find "${SRC}" 2>/dev/null | wc -l)
DST_ALL=$(find "${DST}" 2>/dev/null | wc -l)
echo "  _done   : src=${SRC_DONE}  dst=${DST_DONE}"
echo "  latents : src=${SRC_LAT}  dst=${DST_LAT}"
echo "  allfiles: src=${SRC_ALL}  dst=${DST_ALL}"

if [[ "${RC}" -eq 0 && "${SRC_DONE}" == "${DST_DONE}" && "${SRC_LAT}" == "${DST_LAT}" && "${SRC_DONE}" -gt 0 ]]; then
  echo "VERIFY OK -- counts match. Safe to delete scratch source to free inodes:"
  echo "  rm -rf ${SRC}"
else
  echo "VERIFY MISMATCH or rsync error -- do NOT delete scratch source; inspect."
fi
echo "=== gvt_move DONE $(date) ==="
