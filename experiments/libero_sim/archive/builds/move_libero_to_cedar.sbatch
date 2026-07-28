#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -p cpu-small
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --job-name=lib_move
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.err
#
# TRUE MOVE of the LIBERO GT + rand/play datasets scratch1 -> cedar project share.
# For each dir: rsync (resumable) -> verify file counts match -> ONLY THEN rm -rf the
# source. A mismatch or rsync error leaves the source intact (nothing is deleted). This
# is what frees scratch1 inodes (currently 999,488 / 1,000,000).

set -uo pipefail

SRC_BASE=/storage/scratch1/8/lwang831
DST_BASE=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w

# ALL LIBERO data: GT (precomputed + rendered context) and rand/play (raw hdf5 +
# rendered context). Override the dir list via env (MOVE_DIRS="a b c") so each big
# dir can run as its own parallel job and fit inside the 8 h embers walltime — a
# single job covering every dir times out (510G+272G alone exhausts 8 h).
if [[ -n "${MOVE_DIRS:-}" ]]; then
  read -r -a DIRS <<< "${MOVE_DIRS}"
else
  DIRS=(libero90_play_context libero90_precomputed libero90_context libero90_play)
fi

echo "=== lib_move START $(date) on $(hostname) ==="

for d in "${DIRS[@]}"; do
  SRC="${SRC_BASE}/${d}"
  DST="${DST_BASE}/${d}"
  echo
  echo "########## ${d} ##########"
  if [[ ! -d "${SRC}" ]]; then
    echo "  [skip] source missing: ${SRC}"
    continue
  fi
  echo "  rsync ${SRC}/ -> ${DST}/"
  mkdir -p "${DST}"
  rsync -a --partial "${SRC}/" "${DST}/"
  RC=$?
  SRC_N=$(find "${SRC}" 2>/dev/null | wc -l)
  DST_N=$(find "${DST}" 2>/dev/null | wc -l)
  echo "  rsync rc=${RC}  src_files=${SRC_N}  dst_files=${DST_N}"
  if [[ "${RC}" -eq 0 && "${SRC_N}" -eq "${DST_N}" && "${SRC_N}" -gt 0 ]]; then
    echo "  VERIFY OK -> deleting source ${SRC}"
    rm -rf "${SRC}"
    echo "  deleted. rc=$?"
  else
    echo "  VERIFY MISMATCH / rsync error -> SOURCE KEPT (no delete) for ${d}"
  fi
done

echo
echo "=== scratch1 quota after move ==="
lfs quota -u "$(whoami)" /storage/scratch1 2>/dev/null | sed -n '1,3p' || true
echo "=== lib_move DONE $(date) ==="
