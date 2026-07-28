#!/usr/bin/env bash
# monitor_active_learning_obj_single.sh — 4-ARM object-centric single-scene active-learning
# ablation watchdog. Sources configs/active_learning_obj_single.env and keeps 4 arm jobs alive
# (one Slurm job per arm, max 4). Each arm resumes from the SAME step-30000 checkpoint and trains
# to step 50000; the arms differ ONLY in the score + keep rule, on a byte-identical candidate pool:
#
#   rand_obj|random      — random keep                            (control)
#   maxloss_obj|maxloss  — raw object-region loss, top-K          (reproduces the July-8 null)
#   rho_div_obj|rho_div  — reducible loss (vs frozen step-30000) + diverse batch
#   decaware_obj|decaware— reducible x policy-relevance + diverse batch   (proposed method)
#
# Isolates: reducible-loss (rho_div vs maxloss) | diversity (in rho_div) | decision-awareness
# (decaware vs rho_div). Resumable: each arm's loop drops per-stage markers; the watchdog only
# resubmits the sbatch when an arm's job disappears before reaching the target step.
#
# Run:
#   nohup bash monitor_active_learning_obj_single.sh \
#       > /storage/project/r-agarg35-0/lwang831/tmp/al_obj_4arm_watchdog.log 2>&1 &
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${AL_CONFIG:-${SCRIPT_DIR}/configs/active_learning_obj_single.env}"
[[ -f "$CONFIG" ]] || { echo "[error] config not found: $CONFIG" >&2; exit 1; }
# `set -a` auto-EXPORTS every var the config sets, so --export=ALL carries them into each arm job.
set -a; source "$CONFIG"; set +a

SBATCH_SCRIPT="${SBATCH_SCRIPT:-${SCRIPT_DIR}/run_active_learning.sbatch}"
[[ -f "$SBATCH_SCRIPT" ]] || { echo "[error] sbatch script not found: $SBATCH_SCRIPT" >&2; exit 1; }

# ---- single-scene dataset dir: symlink ONLY the chosen task into a clean dir ----
mkdir -p "$SCENE_DIR"
[[ -f "${SRC_LIBERO90}/${SCENE_FILE}" ]] \
  || { echo "[error] scene demo not found: ${SRC_LIBERO90}/${SCENE_FILE}" >&2; exit 1; }
ln -sf "${SRC_LIBERO90}/${SCENE_FILE}" "${SCENE_DIR}/${SCENE_FILE}"
export LIBERO90="$SCENE_DIR"

# ---- preflight: the start ckpt AND the frozen reference the reducible arms need ----
[[ -f "${START_CKPT_DIR}/lora_weights_step_${START_STEP}.safetensors" ]] \
  || { echo "[error] no step-${START_STEP} start ckpt under ${START_CKPT_DIR}" >&2; exit 1; }
[[ -f "${START_CKPT_DIR}/lora_weights_step_${BASELINE_STEP}.safetensors" ]] \
  || { echo "[error] no frozen reference step-${BASELINE_STEP} ckpt (needed by rho_div/decaware)" >&2; exit 1; }

TOTAL_STEPS="${TRAIN_TO_STEP:-50000}"
SLEEP_INTERVAL_MIN="${SLEEP_INTERVAL_MIN:-15}"

# ARMS = "job|key job|key ..."; each arm -> $OUT_BASE/libero90_ALR_<key>. Split on whitespace.
read -r -a RUNS <<< "$ARMS"

_latest_ckpt_step() {
  find "$1" -name 'lora_weights_step_*.safetensors' 2>/dev/null \
    | grep -oE 'step_[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1
}
_job_count() {
  squeue -u "${USER}" -h -o "%j" 2>/dev/null | awk -v j="$1" '$0==j {c++} END {print c+0}'
}

echo "▶ 4-ARM object-centric single-scene active learning (rand | maxloss | rho_div | decaware)"
echo "   config:   $CONFIG"
echo "   scene:    $SCENE_FILE"
echo "   scorer:   $SCORER  reducible=$REDUCIBLE_METRIC (frozen ref step $BASELINE_STEP)  perturb=$POOL_PERTURB_SOURCE"
echo "   select:   mix=$PER_TASK_MIX keep=$SELECT_K rounds=$MAX_ROUNDS target=$TOTAL_STEPS"
echo "   OUT_BASE: $OUT_BASE"
echo "   arms:"
for r in "${RUNS[@]}"; do
  IFS='|' read -r name key <<< "$r"
  echo "     - $name : version=$key  ckpts=$OUT_BASE/libero90_ALR_$key"
done
echo

while true; do
  echo "============== $(date '+%Y-%m-%d %H:%M:%S') =============="
  all_done=1
  for r in "${RUNS[@]}"; do
    IFS='|' read -r name key <<< "$r"
    ckpt="$OUT_BASE/libero90_ALR_${key}"
    step="$(_latest_ckpt_step "$ckpt")"; step="${step:-0}"
    running="$(_job_count "$name")"

    if (( step >= TOTAL_STEPS )); then
      echo "[$name] DONE  step $step/$TOTAL_STEPS."
      continue
    fi
    all_done=0

    if (( running > 0 )); then
      echo "[$name] OK    step $step/$TOTAL_STEPS | $running job(s) in queue."
    else
      echo "[$name] MISS  step $step/$TOTAL_STEPS | resubmitting (VERSIONS=$key)."
      if sbatch --job-name="$name" --export=ALL,VERSIONS="$key" "$SBATCH_SCRIPT"; then
        echo "[$name] submission accepted."
      else
        echo "[$name] sbatch non-zero; retry next tick."
      fi
    fi
  done

  if (( all_done == 1 )); then
    echo "[DONE] all ${#RUNS[@]} arms reached $TOTAL_STEPS. Stopping watchdog."
    echo "[next] evaluate the 4 arms at step $TOTAL_STEPS -> $EVAL_ROOT (eval_arms_local.py)."
    exit 0
  fi

  echo "Sleeping ${SLEEP_INTERVAL_MIN} minutes..."
  sleep "${SLEEP_INTERVAL_MIN}m"
done
