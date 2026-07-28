#!/usr/bin/env bash
# monitor_active_learning.sh — keepalive watchdog for the active-learning ABLATION.
#
# This is THE monitor: it runs the post-lit-review "v2" signal (AL_V2=1) — the final,
# best version. Two arms differ ONLY in how they pick which 20 of each round's ~40
# candidate trajectories to keep (same start ckpt @ step 30000, same 7:10:3 mix, same
# keep-20, same replay/training schedule, both PRoPE):
#
#   * strategic (keep the 20 hardest, v2 score) : job act_ltx -> <OUT_BASE>/libero90_ALR_strategic
#   * random    (keep 20 at random)             : job rand_ltx -> <OUT_BASE>/libero90_ALR_random
#
# v2 signal refinements (lit-review-driven; all consumed by active_learning_loop.py):
#   AL_V2=1                     -> enable the refinements below
#   SCORE_METRIC=progress_norm  -> rank by magnitude-normalized learning-progress
#                                  (reducible loss; RHO-LOSS/Schmidhuber). Auto-falls back
#                                  to raw loss for the first BASELINE_LAG (3) rounds.
#   DIVERSE (on under AL_V2)     -> greedy k-center keep rule (BatchBALD/core-set)
#   (optional) ENSEMBLE_LAGS=1,2,3 -> also add disagreement (Plan2Explore); off by default.
#
# FRESH dirs (a clean run from step 30000; the cancelled raw-loss run under
# outputs/gvt/libero90_ALR_{strategic,random} is left ON DISK, untouched):
#   OUT_BASE   = .../outputs/gvt_alv2
#   STATE_BASE = .../alr_state_v2
#   GROW_BASE  = /storage/scratch1/8/lwang831/alv2
#   START_CKPT_DIR pinned to the REAL step-30000 ckpt (gvt_alv2 has no such dir of its own).
#
# Both jobs run the SAME run_active_learning.sbatch / active_learning_loop.py, specialised
# to one arm by VERSIONS at submit time. Each arm reloads only ITS OWN latest checkpoint on
# resume, so a requeue can never cross streams. This watchdog is thin: every
# SLEEP_INTERVAL_MIN minutes, for each arm, if no job of that name is queued/running and the
# arm's latest step < TOTAL_STEPS, it resubmits; the Python loop resumes from disk.
#
# Stops once BOTH arms have a checkpoint at >= TOTAL_STEPS.
#
# Run it:
#   nohup bash monitor_active_learning.sh \
#       > /storage/project/r-agarg35-0/lwang831/tmp/al_watchdog.log 2>&1 &
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-${SCRIPT_DIR}/run_active_learning.sbatch}"

# ---- v2 experiment config (exported so --export=ALL carries it into the job) ----
export AL_V2="${AL_V2:-1}"
export SCORE_METRIC="${SCORE_METRIC:-progress_norm}"
export ENSEMBLE_LAGS="${ENSEMBLE_LAGS:-}"            # e.g. "1,2,3" to also add disagreement
export OUT_BASE="${OUT_BASE:-/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/outputs/gvt_alv2}"
export STATE_BASE="${STATE_BASE:-/storage/project/r-agarg35-0/lwang831/alr_state_v2}"
export GROW_BASE="${GROW_BASE:-/storage/scratch1/8/lwang831/alv2}"
# OUT_BASE has no source ckpt of its own; pin the start to the REAL step-30000 dir.
export START_CKPT_DIR="${START_CKPT_DIR:-/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/outputs/gvt/libero90_v2v_ic_lora_prope/checkpoints}"

TOTAL_STEPS="${TOTAL_STEPS:-80000}"
SLEEP_INTERVAL_MIN="${SLEEP_INTERVAL_MIN:-15}"
ARMS_FILTER="${ARMS_FILTER:-both}"

[[ -f "${SBATCH_SCRIPT}" ]] || { echo "[error] sbatch script not found: ${SBATCH_SCRIPT}" >&2; exit 1; }
[[ -f "${START_CKPT_DIR}/lora_weights_step_30000.safetensors" ]] \
  || { echo "[error] no step-30000 ckpt under START_CKPT_DIR=${START_CKPT_DIR}" >&2; exit 1; }

# Each arm = "job_name|version|ckpt_dir". Version MUST be strategic/random (the loop's
# selection logic keys on the exact string); the v2 signal comes from AL_V2 + fresh dirs.
ALL_RUNS=(
  "act_ltx|strategic|${OUT_BASE}/libero90_ALR_strategic"
  "rand_ltx|random|${OUT_BASE}/libero90_ALR_random"
)
RUNS=()
for r in "${ALL_RUNS[@]}"; do
  ver="$(IFS='|' read -r _n v _c <<< "$r"; echo "$v")"
  case "${ARMS_FILTER}" in
    both) RUNS+=("$r") ;;
    strategic) [[ "$ver" == "strategic" ]] && RUNS+=("$r") ;;
    random)    [[ "$ver" == "random"    ]] && RUNS+=("$r") ;;
    *) echo "[error] bad ARMS_FILTER=${ARMS_FILTER}" >&2; exit 1 ;;
  esac
done

# Highest checkpoint step under an arm's output dir (0 if none yet).
_latest_ckpt_step() {
  find "$1" -name 'lora_weights_step_*.safetensors' 2>/dev/null \
    | grep -oE 'step_[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1
}

# Count queued/running jobs with an exact name for this user.
_job_count() {
  squeue -u "${USER}" -h -o "%j" 2>/dev/null \
    | awk -v j="$1" '$0==j {c++} END {print c+0}'
}

echo "▶ Keepalive watchdog for the active-learning ablation — v2 signal (strategic vs random)"
echo "   user:        ${USER}"
echo "   sbatch:      ${SBATCH_SCRIPT}"
echo "   AL_V2:       ${AL_V2}  SCORE_METRIC=${SCORE_METRIC}  ENSEMBLE_LAGS='${ENSEMBLE_LAGS:-<off>}'"
echo "   OUT_BASE:    ${OUT_BASE}"
echo "   STATE_BASE:  ${STATE_BASE}"
echo "   GROW_BASE:   ${GROW_BASE}"
echo "   start ckpt:  ${START_CKPT_DIR}"
echo "   target:      ${TOTAL_STEPS} steps (per arm)"
echo "   interval:    ${SLEEP_INTERVAL_MIN}m"
echo "   arms:"
for r in "${RUNS[@]}"; do
  IFS='|' read -r name ver ckpt <<< "$r"
  echo "     - ${name}: version=${ver} ckpts=${ckpt}"
done
echo

while true; do
  echo "============== $(date '+%Y-%m-%d %H:%M:%S') =============="
  all_done=1

  for r in "${RUNS[@]}"; do
    IFS='|' read -r name ver ckpt <<< "$r"
    step="$(_latest_ckpt_step "${ckpt}")"; step="${step:-0}"
    running="$(_job_count "${name}")"

    if (( step >= TOTAL_STEPS )); then
      echo "[${name}] DONE  step ${step}/${TOTAL_STEPS}."
      continue
    fi
    all_done=0

    if (( running > 0 )); then
      echo "[${name}] OK    step ${step}/${TOTAL_STEPS} | ${running} job(s) in queue."
    else
      echo "[${name}] MISS  step ${step}/${TOTAL_STEPS} | resubmitting (VERSIONS=${ver}, AL_V2=${AL_V2})."
      if sbatch --job-name="${name}" \
                --export=ALL,VERSIONS="${ver}" \
                "${SBATCH_SCRIPT}"; then
        echo "[${name}] submission accepted."
      else
        echo "[${name}] sbatch non-zero; retry next tick."
      fi
    fi
  done

  if (( all_done == 1 )); then
    echo "[DONE] both arms reached ${TOTAL_STEPS}. Stopping watchdog."
    exit 0
  fi

  echo "Sleeping ${SLEEP_INTERVAL_MIN} minutes..."
  sleep "${SLEEP_INTERVAL_MIN}m"
done
