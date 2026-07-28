#!/usr/bin/env bash
# Resumable downloader for the FULL DROID-1.0.1 RLDS into cedar.
#
# Source : gs://gresearch/robotics/droid/1.0.1/  (public Google Cloud bucket, no auth)
#          2048 train shards, ~1 GB each (~1.7 TB total) + dataset_info.json + features.json
# Dest   : /storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/droid_full/1.0.1/  (overridable via DEST=...)
#
# RESUMABILITY
#   Uses `gsutil -m rsync` which checks each destination file's size + CRC32C hash
#   against the bucket and only re-downloads files that are missing/different.
#   Safe to Ctrl-C at any moment and re-run — the worst-case re-work is
#   parallelism × shard_size (~16 GB) for files that were mid-download when
#   interrupted. Already-finished shards are skipped instantly.
#
# USAGE
#   bash download_droid_full.sh                     # foreground (login node OK for short test)
#   nohup bash download_droid_full.sh > dl.log 2>&1 &   # background, survives logout
#   sbatch download_droid_full.sbatch               # slurm CPU job (preferred for full run)
#
# ENV (all optional)
#   DEST                target dir (default cedar droid_full/1.0.1)
#   SRC                 source bucket prefix
#   PARALLEL_PROC       gsutil parallel_process_count (default 16)
#   PARALLEL_THREAD     gsutil parallel_thread_count   (default 4)

set -euo pipefail

DEST="${DEST:-/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/droid_full/1.0.1}"
SRC="${SRC:-gs://gresearch/robotics/droid/1.0.1}"
PARALLEL_PROC="${PARALLEL_PROC:-16}"
PARALLEL_THREAD="${PARALLEL_THREAD:-4}"
EXPECTED_SHARDS=2048

# --- conda env (gsutil installed here on first run) ---
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ltx

# pip cache often points at an unwritable path on PACE — redirect to our scratch
export PIP_CACHE_DIR="${TMPDIR:-/storage/project/r-agarg35-0/lwang831/tmp}/pip-cache"
mkdir -p "${PIP_CACHE_DIR}"

if ! command -v gsutil >/dev/null 2>&1; then
    echo "[setup] gsutil not found in ltx env — pip-installing…"
    python -m pip install --no-warn-script-location gsutil
    if ! command -v gsutil >/dev/null 2>&1; then
        echo "[setup] FATAL: gsutil install ran but binary still not on PATH"
        echo "        PATH=${PATH}"
        exit 1
    fi
    echo "[setup] gsutil installed at $(which gsutil)"
fi

# Anonymous public-bucket access (gresearch is world-readable). Without this,
# gsutil tries an OAuth handshake against a non-existent service-account file
# and fails. The minimal .boto declares no creds → gsutil falls back to anon.
export BOTO_CONFIG="${TMPDIR:-/storage/project/r-agarg35-0/lwang831/tmp}/anon.boto"
if [[ ! -f "${BOTO_CONFIG}" ]]; then
    cat > "${BOTO_CONFIG}" <<'EOF'
[Boto]
https_validate_certificates = True
EOF
fi

# gsutil's -m flag uses Python multiprocessing which creates pymp-XXX scratch dirs
# in TMPDIR. On the project Lustre mount these can hit "Device or resource busy"
# (EBUSY) on rmtree at shutdown. Use node-local /tmp instead — pymp dirs are
# tiny coordination files, not the actual downloaded shards.
export TMPDIR_GSUTIL="/tmp/gsutil-${USER}-$$"
mkdir -p "${TMPDIR_GSUTIL}"
trap 'rm -rf "${TMPDIR_GSUTIL}"' EXIT

mkdir -p "${DEST}"

echo "================ DROID-1.0.1 download =================="
echo "src         : ${SRC}"
echo "dest        : ${DEST}"
echo "parallelism : ${PARALLEL_PROC} procs × ${PARALLEL_THREAD} threads"
echo "free disk   : $(df -h "${DEST}" | tail -1 | awk '{print $4}')"
echo "start       : $(date)"
echo "========================================================"

# gsutil rsync: idempotent (size+CRC32C check), parallel via -m, continue on errors.
# TMPDIR is overridden inline so Python multiprocessing's pymp-* dirs land on
# node-local /tmp (not Lustre, which EBUSYs on rmtree).
TMPDIR="${TMPDIR_GSUTIL}" gsutil \
    -o "GSUtil:parallel_process_count=${PARALLEL_PROC}" \
    -o "GSUtil:parallel_thread_count=${PARALLEL_THREAD}" \
    -m rsync -r -C "${SRC}" "${DEST}"

echo
echo "================ Verify ================"
shards_local="$(find "${DEST}" -maxdepth 1 -name 'droid_101-train.tfrecord-*' | wc -l)"
echo "shards local : ${shards_local} / ${EXPECTED_SHARDS}"
du -sh "${DEST}"
echo "finished at $(date)"

if [[ "${shards_local}" -lt "${EXPECTED_SHARDS}" ]]; then
    echo "WARNING: shard count below expected. Re-run this script to resume — already-downloaded shards will be skipped."
    exit 2
fi
echo "[done] all ${EXPECTED_SHARDS} shards present."
