#!/usr/bin/env bash
# monitor_libero_sharded.sh — keepalive watchdog for the PARALLEL LIBERO-90 build.
#
# Sharded analogue of monitor_libero_build.sh. Instead of keeping ONE serial
# `lib90` worker alive, it keeps the `lib90s` ARRAY job alive: if no lib90s task
# is queued/running and work remains, it submits the array
# (build_libero_sharded.sbatch) with NUM_SHARDS tasks. Disjoint shards + global
# _done/_failed markers mean a resubmitted array safely resumes where it left off.
# Exits 0 once every enumerated demo is resolved (done + failed >= total).
#
# IMPORTANT: do NOT run this alongside the old monitor_libero_build.sh — the
# serial `lib90` worker walks the whole list and would race the shards. Kill the
# old watchdog (and scancel any lib90) before starting this one.
#
# Env overrides (defaults shown):
#   BUILD_SBATCH   ./build_libero_sharded.sbatch
#   JOB_NAME       lib90s   (must match --job-name in the sbatch)
#   NUM_SHARDS     8        (array width: --array=0-(NUM_SHARDS-1))
#   SLEEP_INTERVAL_MIN  10
#   OUT_ROOT       /storage/scratch1/8/lwang831/libero90_context
#
# Usage:
#   nohup bash monitor_libero_sharded.sh > /storage/project/r-agarg35-0/lwang831/tmp/libero_sharded_watchdog.log 2>&1 &
set -uo pipefail   # NO -e: transient squeue hiccups must not kill the watchdog.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_SBATCH="${BUILD_SBATCH:-${SCRIPT_DIR}/build_libero_sharded.sbatch}"
JOB_NAME="${JOB_NAME:-lib90s}"
NUM_SHARDS="${NUM_SHARDS:-8}"
SLEEP_INTERVAL_MIN="${SLEEP_INTERVAL_MIN:-10}"
OUT_ROOT="${OUT_ROOT:-/storage/scratch1/8/lwang831/libero90_context}"

[[ -f "${BUILD_SBATCH}" ]] || { echo "[error] sbatch not found: ${BUILD_SBATCH}" >&2; exit 1; }
mkdir -p "${OUT_ROOT}"
export OUT_ROOT NUM_SHARDS

# True if ANY task of the named array job is queued/running.
_job_active() {
    local name="$1" active
    active="$(squeue -u "${USER}" -h -o '%j' 2>/dev/null || true)"
    printf '%s\n' "${active}" | awk -v j="${name}" '$0==j {f=1} END {exit !f}'
}

echo "▶ Monitoring PARALLEL LIBERO-90 build"
echo "   user:       ${USER}"
echo "   sbatch:     ${BUILD_SBATCH}  (job: ${JOB_NAME}, ${NUM_SHARDS} shards)"
echo "   out_root:   ${OUT_ROOT}"
echo "   interval:   ${SLEEP_INTERVAL_MIN}m"
echo

tick=0
while true; do
    tick=$((tick + 1))
    echo "============== tick ${tick} @ $(date '+%Y-%m-%d %H:%M:%S') =============="
    total="$(cat "${OUT_ROOT}/_total" 2>/dev/null || echo 0)"
    done_count="$(find "${OUT_ROOT}/_done" -type f 2>/dev/null | wc -l)"
    fail_count="$(find "${OUT_ROOT}/_failed" -type f 2>/dev/null | wc -l)"
    resolved=$((done_count + fail_count))
    echo "Progress: done=${done_count} failed=${fail_count} resolved=${resolved}/${total}"

    if (( total > 0 && resolved >= total )); then
        echo "[DONE] all ${total} demos resolved (done=${done_count} failed=${fail_count}). Stopping."
        exit 0
    fi

    if _job_active "${JOB_NAME}"; then
        echo "[OK]   ${JOB_NAME} array active in queue."
    else
        echo "[MISS] ${JOB_NAME} not in queue — submitting array --array=0-$((NUM_SHARDS - 1))."
        if sbatch --export=ALL --array=0-$((NUM_SHARDS - 1)) "${BUILD_SBATCH}"; then
            echo "[OK]   submission accepted."
        else
            echo "[ERR]  sbatch returned non-zero; will retry next tick."
        fi
    fi

    echo "Sleeping ${SLEEP_INTERVAL_MIN}m"
    sleep "${SLEEP_INTERVAL_MIN}m"
done
