#!/usr/bin/env bash
# monitor_libero_rand_sharded.sh — keepalive watchdog for the PARALLEL LIBERO-90
# *random-play* dataset (the off-manifold play split added per the Interactive
# World Simulator paper). Two phases, both sharded arrays:
#
#   Phase 1 (generate): if the play HDF5s aren't all generated yet, keep the
#     `lib_play` array (gen_libero_play.sbatch) alive. Each task emits 3 play
#     HDF5s (<stem>_play_{reach,perturb,random}.hdf5) into PLAY_DIR; the generator
#     writes per-task _gen_done/_gen_failed markers + _total_tasks so a resubmitted
#     array resumes and this watchdog can detect completion. Done when
#     gen_done + gen_failed >= total_tasks.
#
#   Phase 2 (context build): once generation is complete, keep the `lib_rndb`
#     array (build_libero_sharded.sbatch, pointed at PLAY_DIR -> PLAY_CTX) alive.
#     Reuses the SAME build_all_libero.py -> build_libero_context.py pipeline as
#     the demo build (the play HDF5s are byte-compatible LIBERO format), with
#     global _done/_failed markers + _total. Exits 0 once every play chunk is
#     resolved (done + failed >= total).
#
# ENCODE is a SEPARATE step (like the demo flow): after Phase 2, run
# monitor_libero_encode.sh with SRC_ROOT=PLAY_CTX to fold the play chunks into the
# precomputed set the finetune reads. This watchdog stops at the context build.
#
# IMPORTANT: the play build uses a DISTINCT job name (lib_rndb) and DISTINCT
# DATASET/OUT_ROOT, so it is safe to run ALONGSIDE the demo build (lib90s) and the
# DROID build (gvt_datas) — they share only the embers backfill pool, not markers.
#
# Env overrides (defaults shown):
#   GEN_SBATCH          gen_libero_play.sbatch
#   BUILD_SBATCH        build_libero_sharded.sbatch
#   GEN_JOB_NAME        lib_play   (submitted with --job-name=GEN_JOB_NAME)
#   BUILD_JOB_NAME      lib_rndb   (submitted with --job-name=BUILD_JOB_NAME)
#   NUM_SHARDS          16         (array width for BOTH phases: --array=0-(N-1))
#   SLEEP_INTERVAL_MIN  10
#   SRC_DATASET         /storage/cedar/.../LIBERO-datasets/libero_90  (source demos)
#   PLAY_DIR            /storage/scratch1/8/lwang831/libero90_play         (gen out)
#   PLAY_CTX            /storage/scratch1/8/lwang831/libero90_play_context (build out)
#   MIX                 reach:36,perturb:18,random:6
#
# Usage (you submit this yourself):
#   nohup bash monitor_libero_rand_sharded.sh \
#     > /storage/project/r-agarg35-0/lwang831/tmp/libero_rand_watchdog.log 2>&1 &

# NO `-e`. Transient squeue failures must not kill the watchdog.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GEN_SBATCH="${GEN_SBATCH:-${SCRIPT_DIR}/gen_libero_play.sbatch}"
BUILD_SBATCH="${BUILD_SBATCH:-${SCRIPT_DIR}/build_libero_sharded.sbatch}"
GEN_JOB_NAME="${GEN_JOB_NAME:-lib_play}"
BUILD_JOB_NAME="${BUILD_JOB_NAME:-lib_rndb}"
NUM_SHARDS="${NUM_SHARDS:-16}"
SLEEP_INTERVAL_MIN="${SLEEP_INTERVAL_MIN:-10}"

SRC_DATASET="${SRC_DATASET:-/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/LIBERO-datasets/libero_90}"
PLAY_DIR="${PLAY_DIR:-/storage/scratch1/8/lwang831/libero90_play}"
PLAY_CTX="${PLAY_CTX:-/storage/scratch1/8/lwang831/libero90_play_context}"
MIX="${MIX:-reach:36,perturb:18,random:6}"

for f in "${GEN_SBATCH}" "${BUILD_SBATCH}"; do
    [[ -f "${f}" ]] || { echo "[error] sbatch not found: ${f}" >&2; exit 1; }
done
mkdir -p "${PLAY_DIR}" "${PLAY_CTX}"

# Pass-through env so BOTH sbatches target the same roots/mix as the watchdog.
# Phase 1 (gen) reads: DATASET(=SRC_DATASET), OUT_DIR(=PLAY_DIR), MIX, NUM_SHARDS.
# Phase 2 (build) reads: DATASET(=PLAY_DIR), OUT_ROOT(=PLAY_CTX), NUM_SHARDS.

_job_active() {
    local name="$1" active
    active="$(squeue -u "${USER}" -h -o '%j' 2>/dev/null || true)"
    printf '%s\n' "${active}" | awk -v j="${name}" '$0==j {f=1} END {exit !f}'
}

echo "▶ Monitoring PARALLEL LIBERO-90 random-play dataset (gen -> context build)"
echo "   user:        ${USER}"
echo "   gen sbatch:  ${GEN_SBATCH}    (job: ${GEN_JOB_NAME})"
echo "   build sbatch:${BUILD_SBATCH}  (job: ${BUILD_JOB_NAME}, ${NUM_SHARDS} shards)"
echo "   src demos:   ${SRC_DATASET}"
echo "   play dir:    ${PLAY_DIR}"
echo "   play ctx:    ${PLAY_CTX}"
echo "   mix:         ${MIX}"
echo "   interval:    ${SLEEP_INTERVAL_MIN}m"
echo

tick=0
while true; do
    tick=$((tick + 1))
    echo "============== tick ${tick} @ $(date '+%Y-%m-%d %H:%M:%S') =============="

    total_tasks="$(cat "${PLAY_DIR}/_total_tasks" 2>/dev/null || echo 0)"
    gen_done="$(find "${PLAY_DIR}/_gen_done" -type f 2>/dev/null | wc -l)"
    gen_fail="$(find "${PLAY_DIR}/_gen_failed" -type f 2>/dev/null | wc -l)"
    gen_resolved=$((gen_done + gen_fail))

    # ---------------------------------------------------------------- Phase 1: generate
    if (( total_tasks == 0 || gen_resolved < total_tasks )); then
        echo "Phase 1 (generate) — gen markers: done=${gen_done} failed=${gen_fail} / ${total_tasks:-?} tasks"
        if _job_active "${GEN_JOB_NAME}"; then
            echo "[OK]   ${GEN_JOB_NAME} array active — generating play HDF5s."
        else
            echo "[MISS] ${GEN_JOB_NAME} not in queue — submitting array --array=0-$((NUM_SHARDS - 1))."
            if DATASET="${SRC_DATASET}" OUT_DIR="${PLAY_DIR}" MIX="${MIX}" NUM_SHARDS="${NUM_SHARDS}" \
               sbatch --job-name="${GEN_JOB_NAME}" --export=ALL,DATASET,OUT_DIR,MIX,NUM_SHARDS \
                      --array=0-$((NUM_SHARDS - 1)) "${GEN_SBATCH}"; then
                echo "[OK]   gen submission accepted."
            else
                echo "[ERR]  gen sbatch returned non-zero; retry next tick."
            fi
        fi
        next_tick="$(date -d "+${SLEEP_INTERVAL_MIN} minutes" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "in ${SLEEP_INTERVAL_MIN}m")"
        echo "Sleeping ${SLEEP_INTERVAL_MIN}m  (next tick at ${next_tick})"
        sleep "${SLEEP_INTERVAL_MIN}m"
        continue
    fi

    # ---------------------------------------------------------------- Phase 2: context build
    total_chunks="$(cat "${PLAY_CTX}/_total" 2>/dev/null || echo 0)"
    b_done="$(find "${PLAY_CTX}/_done" -type f 2>/dev/null | wc -l)"
    b_fail="$(find "${PLAY_CTX}/_failed" -type f 2>/dev/null | wc -l)"
    b_resolved=$((b_done + b_fail))
    echo "Phase 2 (context build) — done=${b_done} failed=${b_fail} resolved=${b_resolved}/${total_chunks:-?}  (gen complete: ${gen_done} ok, ${gen_fail} failed)"

    if (( total_chunks > 0 && b_resolved >= total_chunks )); then
        echo "[DONE] all ${total_chunks} play chunks resolved (done=${b_done} failed=${b_fail})."
        echo "       Next: encode with monitor_libero_encode.sh SRC_ROOT=${PLAY_CTX}. Stopping watchdog."
        exit 0
    fi

    if _job_active "${BUILD_JOB_NAME}"; then
        echo "[OK]   ${BUILD_JOB_NAME} array active — building play context."
    else
        echo "[MISS] ${BUILD_JOB_NAME} not in queue — submitting array --array=0-$((NUM_SHARDS - 1))."
        if DATASET="${PLAY_DIR}" OUT_ROOT="${PLAY_CTX}" NUM_SHARDS="${NUM_SHARDS}" \
           sbatch --job-name="${BUILD_JOB_NAME}" --export=ALL,DATASET,OUT_ROOT,NUM_SHARDS \
                  --array=0-$((NUM_SHARDS - 1)) "${BUILD_SBATCH}"; then
            echo "[OK]   build submission accepted."
        else
            echo "[ERR]  build sbatch returned non-zero; retry next tick."
        fi
    fi

    next_tick="$(date -d "+${SLEEP_INTERVAL_MIN} minutes" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "in ${SLEEP_INTERVAL_MIN}m")"
    echo "Sleeping ${SLEEP_INTERVAL_MIN}m  (next tick at ${next_tick})"
    sleep "${SLEEP_INTERVAL_MIN}m"
done
