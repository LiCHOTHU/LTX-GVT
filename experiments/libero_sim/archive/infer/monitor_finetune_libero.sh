#!/usr/bin/env bash
# monitor_finetune_libero.sh — keepalive watchdog for the LTX-2.3 LIBERO-90
# v2v IC-LoRA + action-cond finetunes. Now drives BOTH runs at once:
#
#   * no-PRoPE :  job lib_ft | config ltx2_v2v_ic_lora_libero.yaml
#                 ckpts -> outputs/gvt/libero90_v2v_ic_lora
#   * PRoPE    :  job lib_fp | config ltx2_v2v_ic_lora_libero_prope.yaml
#                 ckpts -> outputs/gvt/libero90_v2v_ic_lora_prope
#
# Each run is fully independent: its own job name, its own checkpoint dir, and its
# own config whose `load_checkpoint` points at THAT dir — so auto-resume can never
# cross streams (a requeued lib_fp reloads only the PRoPE checkpoints, etc.). They
# share one sbatch (finetune_libero.sbatch), parametrised via --export CONFIG/OUTDIR
# and --job-name at submit time.
#
# Keeps each job alive across embers preemption AND the 8 h walltime TIMEOUT (neither
# of which `--requeue` recovers from for a TIMEOUT): whenever a run has no job
# queued/running and its latest checkpoint step < TOTAL_STEPS, it resubmits. The
# trainer auto-resumes from that run's latest step_*.safetensors (LoRA + the
# action_projector_*.safetensors saved beside it), so a resubmit continues cleanly
# AND the action encoder is restored — no more "checkpoint can't be inferenced".
#
# Stops once BOTH runs have a checkpoint at >= TOTAL_STEPS.
#
# NOTE: `set -uo pipefail` WITHOUT -e on purpose — a transient squeue/find failure
# must not kill a multi-day watchdog.
#
# Env overrides (defaults shown):
#   SBATCH_SCRIPT       <this dir>/finetune_libero.sbatch
#   CONFIG_DIR          <repo>/packages/ltx-trainer/configs
#   TOTAL_STEPS         80000   (must match optimization.steps in BOTH configs)
#   SLEEP_INTERVAL_MIN  5
#   RUNS_FILTER         "both" | "noprope" | "prope"  (restrict to one run if desired)
#
# Usage:
#   nohup bash monitor_finetune_libero.sh \
#       > /storage/project/r-agarg35-0/lwang831/tmp/lib_ft_watchdog.log 2>&1 &
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT"

SBATCH_SCRIPT="${SBATCH_SCRIPT:-${SCRIPT_DIR}/finetune_libero.sbatch}"
CONFIG_DIR="${CONFIG_DIR:-${REPO}/packages/ltx-trainer/configs}"
OUT_BASE="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/outputs/gvt"
TOTAL_STEPS="${TOTAL_STEPS:-80000}"
SLEEP_INTERVAL_MIN="${SLEEP_INTERVAL_MIN:-5}"
RUNS_FILTER="${RUNS_FILTER:-both}"

if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
  echo "[error] sbatch script not found: ${SBATCH_SCRIPT}" >&2
  exit 1
fi

# Each run = "job_name|config_yaml|ckpt_dir". job names kept < 7 chars (PACE limit).
ALL_RUNS=(
  "lib_ft|${CONFIG_DIR}/ltx2_v2v_ic_lora_libero.yaml|${OUT_BASE}/libero90_v2v_ic_lora"
  "lib_fp|${CONFIG_DIR}/ltx2_v2v_ic_lora_libero_prope.yaml|${OUT_BASE}/libero90_v2v_ic_lora_prope"
)
RUNS=()
for r in "${ALL_RUNS[@]}"; do
  name="${r%%|*}"
  case "${RUNS_FILTER}" in
    both) RUNS+=("$r") ;;
    noprope) [[ "$name" == "lib_ft" ]] && RUNS+=("$r") ;;
    prope)   [[ "$name" == "lib_fp" ]] && RUNS+=("$r") ;;
    *) echo "[error] bad RUNS_FILTER=${RUNS_FILTER}" >&2; exit 1 ;;
  esac
done

# Highest step number among a run's saved LoRA checkpoints (0 if none yet).
_latest_ckpt_step() {
  local ckpt_dir="$1"
  find "${ckpt_dir}" -name '*step_*.safetensors' 2>/dev/null \
    | grep -oE 'step_[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1
}

# Count queued/running jobs with an exact name for this user.
_job_count() {
  local job_name="$1"
  squeue -u "${USER}" -h -o "%j" 2>/dev/null \
    | awk -v j="${job_name}" '$0==j {c++} END {print c+0}'
}

echo "▶ Monitoring LIBERO finetunes (both PRoPE and no-PRoPE)"
echo "   user:     ${USER}"
echo "   sbatch:   ${SBATCH_SCRIPT}"
echo "   target:   ${TOTAL_STEPS} steps"
echo "   interval: ${SLEEP_INTERVAL_MIN}m"
echo "   runs:"
for r in "${RUNS[@]}"; do
  IFS='|' read -r name cfg ckpt <<< "$r"
  echo "     - ${name}: cfg=$(basename "$cfg") ckpts=${ckpt}"
done
echo

while true; do
  now="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "============== ${now} =============="
  all_done=1

  for r in "${RUNS[@]}"; do
    IFS='|' read -r name cfg ckpt <<< "$r"
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
      echo "[${name}] MISS  step ${step}/${TOTAL_STEPS} | resubmitting (resume from ${step})."
      if sbatch --job-name="${name}" \
                --export=ALL,CONFIG="${cfg}",OUTDIR="${ckpt}" \
                "${SBATCH_SCRIPT}"; then
        echo "[${name}] submission accepted."
      else
        echo "[${name}] sbatch non-zero; retry next tick."
      fi
    fi
  done

  if (( all_done == 1 )); then
    echo "[DONE] all monitored runs reached ${TOTAL_STEPS} steps. Stopping watchdog."
    exit 0
  fi

  echo "Sleeping ${SLEEP_INTERVAL_MIN} minutes..."
  sleep "${SLEEP_INTERVAL_MIN}m"
done
