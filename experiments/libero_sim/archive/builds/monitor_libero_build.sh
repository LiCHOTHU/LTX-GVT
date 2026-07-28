#!/usr/bin/env bash
# monitor_libero_build.sh — keepalive watchdog for the LIBERO-90 context build.
#
# Keeps exactly ONE build_libero_resumable.sbatch alive: if the job isn't
# queued/running, submit one; if it is, wait. The driver skips _done/_failed
# markers, so each relaunch resumes. Exits 0 once every enumerated demo is
# resolved (done + failed >= total).
#
# Env overrides (defaults shown):
#   BUILD_SBATCH        ./build_libero_resumable.sbatch
#   BUILD_JOB_NAME      libero_data   (must match --job-name in the sbatch)
#   SLEEP_INTERVAL_MIN  10
#   OUT_ROOT            /storage/scratch1/8/lwang831/libero90_context
#
# Usage:
#   nohup bash monitor_libero_build.sh > /storage/project/r-agarg35-0/lwang831/tmp/libero_watchdog.log 2>&1 &

set -uo pipefail   # NO -e: transient squeue hiccups must not kill the watchdog.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_SBATCH="${BUILD_SBATCH:-${SCRIPT_DIR}/build_libero_resumable.sbatch}"
BUILD_JOB_NAME="${BUILD_JOB_NAME:-lib90}"
SLEEP_INTERVAL_MIN="${SLEEP_INTERVAL_MIN:-10}"
OUT_ROOT="${OUT_ROOT:-/storage/scratch1/8/lwang831/libero90_context}"

[[ -f "${BUILD_SBATCH}" ]] || { echo "[error] sbatch not found: ${BUILD_SBATCH}" >&2; exit 1; }
mkdir -p "${OUT_ROOT}"
export OUT_ROOT

_job_active() {
    local name="$1" active
    active="$(squeue -u "${USER}" -h -o '%j' 2>/dev/null || true)"
    printf '%s\n' "${active}" | awk -v j="${name}" '$0==j {f=1} END {exit !f}'
}

echo "▶ Monitoring LIBERO-90 context build"
echo "   user:       ${USER}"
echo "   sbatch:     ${BUILD_SBATCH}  (job: ${BUILD_JOB_NAME})"
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

    if _job_active "${BUILD_JOB_NAME}"; then
        echo "[OK]   ${BUILD_JOB_NAME} active in queue."
    else
        echo "[MISS] ${BUILD_JOB_NAME} not in queue — submitting one sbatch."
        if sbatch --export=ALL "${BUILD_SBATCH}"; then
            echo "[OK]   submission accepted."
        else
            echo "[ERR]  sbatch returned non-zero; will retry next tick."
        fi
    fi

    echo "Sleeping ${SLEEP_INTERVAL_MIN}m"
    sleep "${SLEEP_INTERVAL_MIN}m"
done
