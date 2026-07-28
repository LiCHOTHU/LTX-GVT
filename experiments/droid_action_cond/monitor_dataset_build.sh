#!/usr/bin/env bash
# monitor_dataset_build.sh — single-command watchdog for the GVT dataset build.
#
# Two phases, both keepalive:
#
#   Phase 1 (enumeration): if _universe.json is missing OR its stamp file
#     doesn't match the configured RLDS root, keep one CPU sbatch alive that
#     runs enumerate_universe.py against the full DROID-1.0.1 RLDS.
#     ~2.7 h wallclock. The watchdog re-submits if preempted (the enumerator
#     overwrites atomically at the end, so a re-run is harmless).
#
#   Phase 2 (build): once the universe is fresh, the loop keeps one GPU
#     sbatch alive that produces chunks, skipping anything already in _done/.
#     Episodes that fail the IoU + wrist-capable filter are already absent
#     from _universe.json, so the builder never tries to build them.
#     Progress carries over from the prior droid_100 build (78 chunks done)
#     via cid matching — the cid is institution+hash+timestamp, stable across
#     DROID releases.
#
# Single instance per phase (never submits while one is queued/running).
# Same polling cadence for both phases. Exits 0 only when ALL chunks are done.
#
# Env overrides (defaults shown):
#   ENUM_SBATCH         enumerate_universe_full.sbatch
#   BUILD_SBATCH        build_dataset_resumable.sbatch
#   ENUM_JOB_NAME       gvt_enum   (must match --job-name in ENUM_SBATCH)
#   BUILD_JOB_NAME      gvt_data   (must match --job-name in BUILD_SBATCH)
#   SLEEP_INTERVAL_MIN  15
#   OUT_ROOT            /storage/cedar/.../gvt_dataset_full
#   DROID_RLDS_DIR      /storage/cedar/.../droid_full/1.0.1
#
# Usage:
#   bash monitor_dataset_build.sh
#   nohup bash monitor_dataset_build.sh > /tmp/gvt_watchdog.log 2>&1 &

# NO `-e`. Transient squeue failures must not kill the watchdog.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENUM_SBATCH="${ENUM_SBATCH:-${SCRIPT_DIR}/enumerate_universe_full.sbatch}"
BUILD_SBATCH="${BUILD_SBATCH:-${SCRIPT_DIR}/build_dataset_resumable.sbatch}"
ENUM_JOB_NAME="${ENUM_JOB_NAME:-gvt_enum}"
BUILD_JOB_NAME="${BUILD_JOB_NAME:-gvt_data}"
SLEEP_INTERVAL_MIN="${SLEEP_INTERVAL_MIN:-15}"
# Built data WRITES to scratch1 (cedar0 hit 100% full). This is exported to both
# sbatches below, so it overrides their defaults — keep it on scratch1 or the build
# crashes ENOSPC at Phase D. DROID_RLDS_DIR stays on cedar (read-only source).
OUT_ROOT="${OUT_ROOT:-/storage/scratch1/8/lwang831/gvt_dataset_full}"
DROID_RLDS_DIR="${DROID_RLDS_DIR:-/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/droid_full/1.0.1}"

UNIVERSE_JSON="${OUT_ROOT}/_universe.json"
RLDS_STAMP="${OUT_ROOT}/_universe.rlds_root"

for f in "${ENUM_SBATCH}" "${BUILD_SBATCH}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[error] sbatch script not found: ${f}" >&2
        exit 1
    fi
done

mkdir -p "${OUT_ROOT}"

# Pass through to both sbatches so they target the same root + RLDS as the watchdog.
export OUT_ROOT
export DROID_RLDS_DIR

_job_active() {
    local name="$1"
    local active
    active="$(squeue -u "${USER}" -h -o '%j' 2>/dev/null || true)"
    printf '%s\n' "${active}" | awk -v j="${name}" '$0==j {found=1} END {exit !found}'
}

_count_universe() {
    python3 -c "import json; print(len(json.load(open('${UNIVERSE_JSON}'))))" 2>/dev/null || echo 0
}

_universe_fresh() {
    [[ -f "${UNIVERSE_JSON}" ]] || return 1
    [[ -f "${RLDS_STAMP}" ]] || return 1
    [[ "$(cat "${RLDS_STAMP}" 2>/dev/null)" == "${DROID_RLDS_DIR}" ]] || return 1
    return 0
}

echo "▶ Monitoring GVT dataset build (full DROID-1.0.1)"
echo "   user:        ${USER}"
echo "   enum sbatch: ${ENUM_SBATCH}    (job: ${ENUM_JOB_NAME})"
echo "   build sbatch:${BUILD_SBATCH}   (job: ${BUILD_JOB_NAME})"
echo "   out_root:    ${OUT_ROOT}"
echo "   rlds source: ${DROID_RLDS_DIR}"
echo "   interval:    ${SLEEP_INTERVAL_MIN}m"
echo

tick=0
while true; do
    tick=$((tick + 1))
    now="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "============== tick ${tick} @ ${now} =============="

    if ! _universe_fresh; then
        if [[ ! -f "${UNIVERSE_JSON}" ]]; then
            why="universe missing"
        elif [[ ! -f "${RLDS_STAMP}" ]]; then
            why="stamp missing (universe predates full-RLDS workflow)"
        else
            why="stamp mismatch (stale: $(cat "${RLDS_STAMP}" 2>/dev/null))"
        fi
        echo "Phase 1 (enumeration) — ${why}"

        if _job_active "${ENUM_JOB_NAME}"; then
            echo "[OK]   ${ENUM_JOB_NAME} active in queue — waiting for it to stamp the universe."
        else
            echo "[MISS] ${ENUM_JOB_NAME} not in queue — submitting one sbatch."
            if sbatch --job-name="${ENUM_JOB_NAME}" --export=ALL "${ENUM_SBATCH}"; then
                echo "[OK]   submission accepted."
            else
                echo "[ERR]  sbatch returned non-zero; will retry on next tick."
            fi
        fi
    else
        done_count="$(find "${OUT_ROOT}/_done" -type f 2>/dev/null | wc -l)"
        total_chunks="$(_count_universe)"
        echo "Phase 2 (build) — Progress: ${done_count}/${total_chunks} chunks done"

        if (( total_chunks > 0 && done_count >= total_chunks )); then
            echo "[DONE] all ${total_chunks} chunks committed. Stopping watchdog."
            exit 0
        fi

        if _job_active "${BUILD_JOB_NAME}"; then
            echo "[OK]   ${BUILD_JOB_NAME} active in queue."
        else
            echo "[MISS] ${BUILD_JOB_NAME} not in queue — submitting one sbatch."
            if sbatch --job-name="${BUILD_JOB_NAME}" --export=ALL "${BUILD_SBATCH}"; then
                echo "[OK]   submission accepted."
            else
                echo "[ERR]  sbatch returned non-zero; will retry on next tick."
            fi
        fi
    fi

    next_tick="$(date -d "+${SLEEP_INTERVAL_MIN} minutes" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "in ${SLEEP_INTERVAL_MIN}m")"
    echo "Sleeping ${SLEEP_INTERVAL_MIN}m  (next tick at ${next_tick})"
    sleep "${SLEEP_INTERVAL_MIN}m"
done
