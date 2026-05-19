#!/usr/bin/env bash
# monitor_dataset_build.sh — keep one GVT dataset-build sbatch alive.
#
# Loop: every SLEEP_INTERVAL_MIN minutes, check `squeue` for a job named
# ${JOB_NAME} belonging to the current user.
#   • If it's in the queue (PENDING / RUNNING / REQUEUED) → skip.
#   • Else, if every chunk under SRC_ROOT already has _done/<id> in OUT_ROOT
#     → exit 0, build is complete.
#   • Else → submit one sbatch (no array). The resumable orchestrator picks up
#     where the previous run left off via the _done/ markers.
#
# Single instance only — never submits while one is still queued/running.
#
# Env overrides (defaults shown):
#   SBATCH_SCRIPT       path to build_dataset_resumable.sbatch
#   JOB_NAME            must match --job-name in the sbatch (default: gvt_data)
#   SLEEP_INTERVAL_MIN  polling cadence in minutes (default: 15)
#   SRC_ROOT, OUT_ROOT  same as the sbatch
#
# Usage:
#   bash monitor_dataset_build.sh                       # foreground
#   SLEEP_INTERVAL_MIN=10 bash monitor_dataset_build.sh  # 10-min cadence
#   nohup bash monitor_dataset_build.sh > /tmp/gvt_watchdog.log 2>&1 &
#
# Stop with Ctrl-C. Idempotent — re-running is harmless.

# NOTE: NO `-e`. A transient `squeue` failure must not kill the watchdog.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SBATCH_SCRIPT="${SBATCH_SCRIPT:-${SCRIPT_DIR}/build_dataset_resumable.sbatch}"
JOB_NAME="${JOB_NAME:-gvt_data}"
SLEEP_INTERVAL_MIN="${SLEEP_INTERVAL_MIN:-15}"
SRC_ROOT="${SRC_ROOT:-${SCRIPT_DIR}/outputs/context_chunks_wrist_new}"
OUT_ROOT="${OUT_ROOT:-/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/gvt_dataset_full}"

if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
    echo "[error] sbatch script not found: ${SBATCH_SCRIPT}" >&2
    exit 1
fi

TOTAL_CHUNKS="$(find "${SRC_ROOT}" -name data.npz 2>/dev/null | wc -l)"

echo "▶ Monitoring GVT dataset build (single-instance, no array)"
echo "   user:        ${USER}"
echo "   job:         ${JOB_NAME}"
echo "   sbatch:      ${SBATCH_SCRIPT}"
echo "   src_root:    ${SRC_ROOT}   (${TOTAL_CHUNKS} chunks)"
echo "   out_root:    ${OUT_ROOT}"
echo "   interval:    ${SLEEP_INTERVAL_MIN}m"
echo

# Export so the sbatch (which honours these env vars) picks the same paths.
export SRC_ROOT OUT_ROOT

tick=0
while true; do
    tick=$((tick + 1))
    now="$(date '+%Y-%m-%d %H:%M:%S')"

    # Single squeue per tick (slow on busy controllers; cache the result).
    active="$(squeue -u "${USER}" -h -o '%j' 2>/dev/null || true)"

    # Membership check via awk (reference pattern).
    if printf '%s\n' "${active}" | awk -v j="${JOB_NAME}" '$0==j {found=1} END {exit !found}'; then
        in_queue=1
    else
        in_queue=0
    fi

    done_count="$(find "${OUT_ROOT}/_done" -type f 2>/dev/null | wc -l)"

    echo "============== tick ${tick} @ ${now} =============="
    echo "Progress: ${done_count}/${TOTAL_CHUNKS} chunks done"

    if (( TOTAL_CHUNKS > 0 && done_count >= TOTAL_CHUNKS )); then
        echo "[DONE] all ${TOTAL_CHUNKS} chunks committed. Stopping watchdog."
        exit 0
    fi

    if (( in_queue )); then
        echo "[OK]   ${JOB_NAME} active in queue."
    else
        echo "[MISS] ${JOB_NAME} not in queue — submitting one sbatch."
        if sbatch --job-name="${JOB_NAME}" --export=ALL "${SBATCH_SCRIPT}"; then
            echo "[OK]   submission accepted."
        else
            echo "[ERR]  sbatch returned non-zero; will retry on next tick."
        fi
    fi

    next_tick="$(date -d "+${SLEEP_INTERVAL_MIN} minutes" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "in ${SLEEP_INTERVAL_MIN}m")"
    echo "Sleeping ${SLEEP_INTERVAL_MIN}m  (next tick at ${next_tick})"
    sleep "${SLEEP_INTERVAL_MIN}m"
done
