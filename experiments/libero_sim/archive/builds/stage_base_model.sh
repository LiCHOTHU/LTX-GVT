#!/usr/bin/env bash
# stage_base_model.sh — copy the 22B base transformer + Gemma text encoder from cedar0
# cold storage onto fast scratch1, so the AL loop's grade/train stages cold-load in a
# fraction of the ~25 min it takes from cedar0.
#
# Why: every round the 22B model is cold-loaded TWICE (grade loads the transformer;
# train loads transformer + gemma), and again on every embers preemption/requeue. Reading
# from scratch1 (parallel Lustre) instead of cedar0 removes most of that wall-clock.
#
# Re-run this if scratch1 was purged (it auto-purges after ~60 days). It is idempotent:
# rsync only copies what's missing/changed. The config + score script auto-fall back to
# the cedar0 originals if the scratch1 copy is absent, so nothing breaks if you forget.
#
#   bash experiments/libero_sim/stage_base_model.sh
set -euo pipefail

SRC="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w"
DST="/storage/scratch1/8/lwang831"

mkdir -p "${DST}/LTX-2.3"

echo "[stage] transformer (44G) -> ${DST}/LTX-2.3/ ..."
rsync -ah --info=progress2 \
  "${SRC}/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors" \
  "${DST}/LTX-2.3/"

echo "[stage] gemma text encoder (23G) -> ${DST}/ ..."
rsync -ah --info=progress2 \
  "${SRC}/gemma-3-12b-it-qat-q4_0-unquantized" \
  "${DST}/"

# rsync writes each file to a temp name and renames on completion, so the final paths
# never expose a partial file. Touch a marker only after BOTH are fully staged.
touch "${DST}/LTX-2.3/.staged_ok"

echo "[stage] done. Staged paths:"
ls -laLh "${DST}/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
du -sh "${DST}/gemma-3-12b-it-qat-q4_0-unquantized"
