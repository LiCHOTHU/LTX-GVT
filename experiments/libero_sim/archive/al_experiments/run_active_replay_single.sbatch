#!/usr/bin/env bash
# run_active_replay_single.sbatch — the WHOLE active-learning loop in ONE job, ONE GPU.
#
# No babysitter, no child jobs, no coordination. This single SLURM job does, by itself,
# for each round r = 0..MAX_ROUNDS-1 and for each model version:
#     1. make ~40 practice clips      (gen_libero_play -> build context -> encode -> cameras)
#     2. grade them (smart only)      (score_chunk_latent_loss -> per-clip "how wrong" score)
#     3. keep 20                       (smart: hardest 20 ; dumb: random 20) + replay buffer
#     4. train +ROUND_STEPS steps      (resumes from the version's latest checkpoint)
# Everything runs in-process on the one allocated GPU, in order. Start = step 30000,
# end = step 80000 (10 x 5000).
#
# Two model versions are compared (run sequentially in this one job):
#     strategic = the SMART way: keep the 20 clips the model is worst at
#     random    = the DUMB control: keep 20 clips at random
# Set VERSIONS to run just one, e.g.  --export=ALL,VERSIONS=strategic
#
# SURVIVING THE TIME LIMIT (still one job to control):
#   * Everything is resumable from disk: finished stages drop marker files; training
#     auto-resumes from its last checkpoint. So a kill at any point loses < a few hundred
#     steps, never a whole round.
#   * On preemption (embers) SLURM requeues this same job (--requeue) and it picks up.
#   * Near the walltime, SLURM sends USR1 (--signal) and the job RESUBMITS ITSELF once,
#     then exits. So there is only ever ONE job running; it just keeps relaunching until
#     it reaches step 80000. To stop everything: scancel that one job (no resubmit on
#     a user scancel). Set AUTO_RESUBMIT=0 to resubmit by hand instead.
#
# Submit (just once):   sbatch experiments/libero_sim/run_active_replay_single.sbatch
# Dry check (no GPU):   DRY_RUN=1 bash experiments/libero_sim/run_active_replay_single.sbatch

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH -p gpu-h200
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --job-name=al_loop
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs/%x_%j.err

set -uo pipefail   # NOT -e: one transient find/squeue blip must not kill a multi-day run

# ---------- locations (must match the per-version state already seeded on disk) ----------
REPO="/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT"
LSIM="${REPO}/experiments/libero_sim"
USER_LIBERO="/storage/home/hcoda1/8/lwang831/workspace/LIBERO"
SELF="${LSIM}/run_active_replay_single.sbatch"     # hardcoded so self-resubmit works from spool
CONDA_ENV="ltx"

OUT_BASE="${OUT_BASE:-/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/outputs/gvt}"
START_CKPT_DIR="${START_CKPT_DIR:-${OUT_BASE}/libero90_v2v_ic_lora_prope/checkpoints}"
TEMPLATE_CONFIG="${TEMPLATE_CONFIG:-${REPO}/packages/ltx-trainer/configs/ltx2_v2v_ic_lora_libero_prope.yaml}"
LIBERO90="${LIBERO90:-/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/LIBERO-datasets/libero_90}"
BASE_ROOT="${BASE_ROOT:-/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/libero90_combined_precomputed/precomputed}"
GROW_BASE="${GROW_BASE:-/storage/scratch1/8/lwang831}"
STATE_BASE="${STATE_BASE:-/storage/project/r-agarg35-0/lwang831/alr_state}"
LOGDIR="/storage/project/r-agarg35-0/lwang831/outputs/gvt/logs"

# ---------- loop knobs ----------
START_STEP="${START_STEP:-30000}"
ROUND_STEPS="${ROUND_STEPS:-1000}"
MAX_ROUNDS="${MAX_ROUNDS:-50}"          # 30000 + 50x1000 = 80000
SELECT_K="${SELECT_K:-20}"              # keep half of ~40 candidates
BUFFER_CAP="${BUFFER_CAP:-270}"
SEED_PER_TASK="${SEED_PER_TASK:-3}"
PER_TASK_MIX="${PER_TASK_MIX:-reach:7,perturb:10,random:3}"
NUM_SHARDS="${NUM_SHARDS:-45}"
RANDOM_SEED="${RANDOM_SEED:-20260615}"
VERSIONS="${VERSIONS:-strategic random}"   # space-separated; set to one to run only that
SIGMA_GRID="${SIGMA_GRID:-0.2,0.4,0.6,0.8}"
SEEDS="${SEEDS:-0,1,2}"
RES="${RES:-256}"; FPS="${FPS:-15}"; GEN_STEPS="${GEN_STEPS:-200}"
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"
DRY_RUN="${DRY_RUN:-0}"

FINAL_STEP=$(( START_STEP + MAX_ROUNDS * ROUND_STEPS ))

# ---------- env (union of build + score + train needs) ----------
if [[ "${DRY_RUN}" == "0" ]]; then
  cd "${REPO}"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="${USER_LIBERO}:${LSIM}:${REPO}/packages/ltx-trainer/src:${REPO}/packages/ltx-core/src:${PYTHONPATH:-}"
export TMPDIR="/storage/project/r-agarg35-0/lwang831/tmp"
export MPLCONFIGDIR="${TMPDIR}/mpl"
unset HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}" "${HF_HOME}" "${LOGDIR}"

# ---------- helpers ----------
_dry() { [[ "${DRY_RUN}" != "0" ]]; }
say()  { echo "[$(date '+%H:%M:%S')] $*"; }

# Run a (long) python stage so the USR1 walltime signal can interrupt it; echo-only in dry.
run() {
  say "RUN: $*"
  _dry && return 0
  "$@" &
  wait $!
}

_arm_out()   { echo "${OUT_BASE}/libero90_ALR_$1"; }
_add_root()  { echo "${GROW_BASE}/libero90_ALR_$1_add/precomputed"; }
_state_dir() { echo "${STATE_BASE}/alr_$1"; }
_round_dir() { echo "${GROW_BASE}/libero90_ALR_$1_round$2"; }   # version, round
_latest_step() {
  find "$1" -name 'lora_weights_step_*.safetensors' 2>/dev/null \
    | grep -oE 'step_[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1
}

_write_config() {  # al_out  add_root  steps  cfg_out
  sed -E \
    -e "s|^  load_checkpoint:.*|  load_checkpoint: \"$1\"|" \
    -e "s|^  preprocessed_data_root:.*|  preprocessed_data_root: [\"${BASE_ROOT}\", \"$2\"]|" \
    -e "s|^output_dir:.*|output_dir: \"$1\"|" \
    -e "s|^  steps:.*|  steps: $3|" \
    "${TEMPLATE_CONFIG}" > "$4"
}

init_version() {  # idempotent: ckpt symlinks + replay buffer seed
  local v="$1" al_out add state buf
  al_out="$(_arm_out "$v")"; add="$(_add_root "$v")"; state="$(_state_dir "$v")"
  buf="${state}/buffer.json"
  mkdir -p "${al_out}/checkpoints" "${al_out}/rounds" "${add}" "${state}"
  for suf in "lora_weights_step_${START_STEP}.safetensors" \
             "action_projector_step_${START_STEP}.safetensors" \
             "training_state_step_${START_STEP}.pt"; do
    [[ -f "${START_CKPT_DIR}/${suf}" && ! -e "${al_out}/checkpoints/${suf}" ]] \
      && ln -sfn "${START_CKPT_DIR}/${suf}" "${al_out}/checkpoints/${suf}"
  done
  if [[ ! -f "${buf}" ]]; then
    say "[${v}] seed replay buffer (${SEED_PER_TASK}/task)"
    _dry || python -u "${LSIM}/al_replay_buffer.py" seed \
        --libero90 "${LIBERO90}" --manifest "${buf}" \
        --per-task "${SEED_PER_TASK}" --seed "${RANDOM_SEED}"
    [[ -f "${state}/added.txt" ]] || : > "${state}/added.txt"
  fi
}

do_round() {  # version  round  -> build, (grade), select+buffer, train ; all resumable
  local v="$1" r="$2"
  local al_out add state buf manifest rd play ctx pre json picked target step
  al_out="$(_arm_out "$v")"; add="$(_add_root "$v")"; state="$(_state_dir "$v")"
  buf="${state}/buffer.json"; manifest="${state}/added.txt"
  rd="$(_round_dir "$v" "$r")"; play="${rd}/play"; ctx="${rd}/context"; pre="${rd}/precomputed"
  json="${state}/latent_loss_r${r}.json"; picked="${state}/picked_r${r}.txt"
  target=$(( START_STEP + (r + 1) * ROUND_STEPS ))

  step="$(_latest_step "${al_out}")"; step="${step:-0}"
  if (( step >= target )); then say "[${v}] r${r} already at step ${step} >= ${target}; skip."; return 0; fi

  # ---- 1) make ~40 practice clips ----
  if [[ ! -f "${pre}/_build_done" ]]; then
    say "[${v}] r${r} BUILD candidates (shard $(( r % NUM_SHARDS ))/${NUM_SHARDS}, perturb<-buffer)"
    mkdir -p "${play}" "${ctx}" "${pre}"
    local psrc=(); [[ -f "${buf}" ]] && psrc=(--perturb-source "${buf}")
    run python -u "${LSIM}/gen_libero_play.py" \
        --dataset "${LIBERO90}" --out_dir "${play}" \
        --mix "${PER_TASK_MIX}" --steps "${GEN_STEPS}" \
        --seed "$(( RANDOM_SEED + 7919*r ))" \
        --num_shards "${NUM_SHARDS}" --shard "$(( r % NUM_SHARDS ))" "${psrc[@]}"
    run python -u "${LSIM}/build_all_libero.py" \
        --dataset "${play}" --out_root "${ctx}" --res "${RES}" --fps "${FPS}" --max_demos 0
    run python -u "${LSIM}/encode_libero_precomputed.py" \
        --src_root "${ctx}" --out_root "${rd}" \
        --shard "0/1" --batch_size 4 --max_chunks 0 --load_text_encoder_in_8bit
    run python -u "${LSIM}/build_libero_cameras.py" \
        --src_root "${ctx}" --out_root "${rd}" --shard "0/1" --max_chunks 0
    _dry || touch "${pre}/_build_done"
  fi

  # ---- 2) grade (smart version only) ----
  if [[ "${v}" == "strategic" && ! -f "${json}" ]]; then
    local lora proj
    lora="$(find "${al_out}" -name "lora_weights_step_${step}.safetensors" 2>/dev/null | head -1)"
    proj="$(find "${al_out}" -name "action_projector_step_${step}.safetensors" 2>/dev/null | head -1)"
    say "[${v}] r${r} GRADE clips with ckpt step ${step}"
    run python -u "${LSIM}/score_chunk_latent_loss.py" \
        --lora-path "${lora}" --action-projector "${proj}" --data-root "${pre}" \
        --use-prope --prope-proj-dim 64 --prope-image-width 256 --prope-image-height 256 \
        --sigma-grid "${SIGMA_GRID}" --seeds "${SEEDS}" --out "${json}"
  fi

  # ---- 3) keep 20 (smart=hardest, dumb=random) + replay buffer ----
  if [[ ! -f "${state}/_selected_r${r}" ]]; then
    say "[${v}] r${r} KEEP ${SELECT_K} ($([[ "$v" == strategic ]] && echo hardest || echo random))"
    if [[ "${v}" == "strategic" ]]; then
      run python -u "${LSIM}/al_select_and_grow.py" \
          --pool-root "${pre}" --grow-root "${add}" --manifest "${manifest}" \
          --k "${SELECT_K}" --mode strategic --latent-json "${json}" --picked-out "${picked}"
    else
      run python -u "${LSIM}/al_select_and_grow.py" \
          --pool-root "${pre}" --grow-root "${add}" --manifest "${manifest}" \
          --k "${SELECT_K}" --mode random --seed "$(( RANDOM_SEED + r ))" --picked-out "${picked}"
    fi
    run python -u "${LSIM}/al_replay_buffer.py" append \
        --manifest "${buf}" --kept-dir "${play}" --picked "${picked}" \
        --round "$(( r + 1 ))" --cap "${BUFFER_CAP}"
    _dry || touch "${state}/_selected_r${r}"
  fi

  # ---- 4) train +ROUND_STEPS (auto-resumes from al_out's latest ckpt) ----
  step="$(_latest_step "${al_out}")"; step="${step:-0}"
  if (( step < target )); then
    local cfg="${state}/config_r${r}.yaml"
    _write_config "${al_out}" "${add}" "${target}" "${cfg}"
    say "[${v}] r${r} TRAIN step ${step} -> ${target}"
    run python -u "${REPO}/packages/ltx-trainer/scripts/train.py" "${cfg}" --disable-progress-bars
  fi

  # ---- archive this round's ckpt for the quality-vs-#clips curve ----
  local src; src="$(find "${al_out}/checkpoints" -name "lora_weights_step_${target}.safetensors" 2>/dev/null | head -1)"
  [[ -n "${src}" && ! -f "${al_out}/rounds/$(basename "${src}")" ]] && { _dry || cp -f "${src}" "${al_out}/rounds/"; }
  say "[${v}] r${r} done."
}

all_done() {
  local v s
  for v in ${VERSIONS}; do
    s="$(_latest_step "$(_arm_out "$v")")"; s="${s:-0}"
    (( s < FINAL_STEP )) && return 1
  done
  return 0
}

maybe_resubmit() {
  if all_done; then say "All versions reached step ${FINAL_STEP}. Nothing to resubmit."; return; fi
  if [[ "${AUTO_RESUBMIT}" != "1" ]]; then
    say "Work remains but AUTO_RESUBMIT=0. Continue by hand: sbatch ${SELF}"; return
  fi
  say "Work remains -> resubmitting self to continue (one job at a time)."
  _dry || sbatch "${SELF}"
}

on_timeout() { echo; say "[signal] walltime approaching — saving point is on disk; continuing later."; maybe_resubmit; exit 0; }
trap on_timeout USR1

# ============================ main ============================
say "=== single-job active-learning loop ==="
say "versions=[${VERSIONS}]  start=${START_STEP}  final=${FINAL_STEP}  rounds=${MAX_ROUNDS}x${ROUND_STEPS}"
say "JobID=${SLURM_JOB_ID:-<none>}  Node=${SLURM_NODELIST:-<none>}  DRY_RUN=${DRY_RUN}"
_dry || nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

for v in ${VERSIONS}; do init_version "${v}"; done

for (( r=0; r<MAX_ROUNDS; r++ )); do
  for v in ${VERSIONS}; do
    do_round "${v}" "${r}"
  done
done

if all_done; then
  say "=== ALL DONE: every version reached step ${FINAL_STEP}. Per-round ckpts under <out>/rounds/. ==="
else
  say "=== loop returned but not all versions are at ${FINAL_STEP} (check logs above) ==="
  maybe_resubmit
fi
