#!/usr/bin/env bash
# monitor_libero_encode.sh — keepalive watchdog for the PARALLEL LIBERO preprocess
# (raw context chunks -> trainer precomputed/ via encode_libero_sharded.sbatch).
#
# Keeps the `lib_enc` ARRAY alive: if no lib_enc task is queued/running and chunks
# remain un-encoded, it resubmits the array with NUM_SHARDS tasks. Disjoint
# stable-hash shards + global _done/<cid> markers mean a resubmitted array safely
# resumes where it left off (needed because a single shard's ~2.8k chunks can
# exceed the 8 h embers walltime, and embers is preemptible).
#
# "Total" is counted live from the source tree (one data.npz per chunk); "done"
# from the _done markers in OUT_ROOT. Exits 0 once done >= total.
#
# Env overrides (defaults shown):
#   ENCODE_SBATCH  ./encode_libero_sharded.sbatch
#   JOB_NAME       lib_enc   (must match --job-name in the sbatch)
#   NUM_SHARDS     8         (array width: --array=0-(NUM_SHARDS-1))
#   SLEEP_INTERVAL_MIN  10
#   SRC_ROOT       /storage/scratch1/8/lwang831/libero90_context
#   OUT_ROOT       /storage/scratch1/8/lwang831/libero90_precomputed
#
# Usage:
#   nohup bash monitor_libero_encode.sh > /storage/project/r-agarg35-0/lwang831/tmp/libero_encode_watchdog.log 2>&1 &
set -uo pipefail   # NO -e: transient squeue/find hiccups must not kill the watchdog.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENCODE_SBATCH="${ENCODE_SBATCH:-${SCRIPT_DIR}/encode_libero_sharded.sbatch}"
JOB_NAME="${JOB_NAME:-lib_enc}"
NUM_SHARDS="${NUM_SHARDS:-8}"
SLEEP_INTERVAL_MIN="${SLEEP_INTERVAL_MIN:-10}"
SRC_ROOT="${SRC_ROOT:-/storage/scratch1/8/lwang831/libero90_context}"
OUT_ROOT="${OUT_ROOT:-/storage/scratch1/8/lwang831/libero90_precomputed}"

[[ -f "${ENCODE_SBATCH}" ]] || { echo "[error] sbatch not found: ${ENCODE_SBATCH}" >&2; exit 1; }
mkdir -p "${OUT_ROOT}/_done"
export SRC_ROOT OUT_ROOT NUM_SHARDS

_job_active() {
    local name="$1" active
    active="$(squeue -u "${USER}" -h -o '%j' 2>/dev/null || true)"
    printf '%s\n' "${active}" | awk -v j="${name}" '$0==j {f=1} END {exit !f}'
}

echo "▶ Monitoring PARALLEL LIBERO preprocess"
echo "   user:       ${USER}"
echo "   sbatch:     ${ENCODE_SBATCH}  (job: ${JOB_NAME}, ${NUM_SHARDS} shards)"
echo "   src_root:   ${SRC_ROOT}"
echo "   out_root:   ${OUT_ROOT}"
echo "   interval:   ${SLEEP_INTERVAL_MIN}m"
echo

tick=0
while true; do
    tick=$((tick + 1))
    echo "============== tick ${tick} @ $(date '+%Y-%m-%d %H:%M:%S') =============="
    total="$(find "${SRC_ROOT}"/*/chunk_*/data.npz -maxdepth 0 2>/dev/null | wc -l)"
    done_count="$(find "${OUT_ROOT}/_done" -type f 2>/dev/null | wc -l)"
    echo "Progress: done=${done_count}/${total} chunks encoded  (${NUM_SHARDS} shards)"

    if (( total > 0 && done_count >= total )); then
        echo "[DONE] all ${total} chunks encoded. Stopping."
        exit 0
    fi

    if _job_active "${JOB_NAME}"; then
        echo "[OK]   ${JOB_NAME} array active in queue."
    else
        echo "[MISS] ${JOB_NAME} not in queue — submitting array --array=0-$((NUM_SHARDS - 1))."
        if sbatch --export=ALL --array=0-$((NUM_SHARDS - 1)) "${ENCODE_SBATCH}"; then
            echo "[OK]   submission accepted."
        else
            echo "[ERR]  sbatch returned non-zero; will retry next tick."
        fi
    fi

    echo "Sleeping ${SLEEP_INTERVAL_MIN}m"
    sleep "${SLEEP_INTERVAL_MIN}m"
done
