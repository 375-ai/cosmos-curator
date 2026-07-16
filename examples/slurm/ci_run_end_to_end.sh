#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[%s] %s\n' "$(date -Ins)" "$*"
}

required_vars=(
  S3_INPUT_VIDEO_PATH
  SLURM_E2E_OUTPUT_CLIP_PATH
  SLURM_E2E_OUTPUT_DEDUP_PATH
  SLURM_E2E_OUTPUT_DATASET_PATH
)
for var in "${required_vars[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    echo "Missing required environment variable: ${var}" >&2
    exit 1
  fi
done

SLURM_E2E_S3_PROFILE_NAME=${SLURM_E2E_S3_PROFILE_NAME:-default}
SLURM_E2E_RUN_LANCE=${SLURM_E2E_RUN_LANCE:-0}
SLURM_E2E_OUTPUT_CLIP_LANCE_PATH=${SLURM_E2E_OUTPUT_CLIP_LANCE_PATH:-${SLURM_E2E_OUTPUT_CLIP_PATH}_lance}
SLURM_E2E_OUTPUT_DATASET_LANCE_PATH=${SLURM_E2E_OUTPUT_DATASET_LANCE_PATH:-${SLURM_E2E_OUTPUT_DATASET_PATH}_lance}

export AWS_SHARED_CREDENTIALS_FILE="${AWS_SHARED_CREDENTIALS_FILE:-/creds/s3_creds}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-west-2}"
export MODEL_WEIGHTS_PREFIX="${MODEL_WEIGHTS_PREFIX:-/config/models}"
export PYTHONUNBUFFERED=1

# If the CI job downloaded a locally-built cosmos-xenna wheel artifact (via
# the optional build_xenna_wheels matrix job, gated by the
# 'use-local-xenna-build' MR label), install it over the pixi-resolved
# cosmos-xenna in the default env. No-op when the helper or artifact is
# absent, so this stays safe for end-user copies of this example.
if [[ -x /config/project/.gitlab/scripts/install_local_xenna_into_pixi.sh ]]; then
  WHEEL_DIR="/config/project/cosmos-xenna/target/wheels" \
    bash /config/project/.gitlab/scripts/install_local_xenna_into_pixi.sh default
fi

run_split() {
  log "Running split pipeline -> ${SLURM_E2E_OUTPUT_CLIP_PATH}"
  python -m cosmos_curator.pipelines.video.run_pipeline split \
    --input-video-path "${S3_INPUT_VIDEO_PATH}" \
    --output-clip-path "${SLURM_E2E_OUTPUT_CLIP_PATH}" \
    --input-s3-profile-name "${SLURM_E2E_S3_PROFILE_NAME}" \
    --output-s3-profile-name "${SLURM_E2E_S3_PROFILE_NAME}" \
    --limit 1 \
    --execution-mode STREAMING
  log "Split pipeline completed"
}

run_dedup() {
  log "Running dedup pipeline -> ${SLURM_E2E_OUTPUT_DEDUP_PATH}"
  python -m cosmos_curator.pipelines.video.run_pipeline dedup \
    --input-embeddings-path "${SLURM_E2E_OUTPUT_CLIP_PATH}" \
    --output-path "${SLURM_E2E_OUTPUT_DEDUP_PATH}" \
    --input-s3-profile-name "${SLURM_E2E_S3_PROFILE_NAME}" \
    --output-s3-profile-name "${SLURM_E2E_S3_PROFILE_NAME}" \
    --eps-to-extract 0.01
  log "Dedup pipeline completed"
}

run_shard() {
  log "Running shard pipeline -> ${SLURM_E2E_OUTPUT_DATASET_PATH}"
  python -m cosmos_curator.pipelines.video.run_pipeline shard \
    --input-clip-path "${SLURM_E2E_OUTPUT_CLIP_PATH}" \
    --output-dataset-path "${SLURM_E2E_OUTPUT_DATASET_PATH}" \
    --input-semantic-dedup-path "${SLURM_E2E_OUTPUT_DEDUP_PATH}" \
    --input-semantic-dedup-s3-profile-name "${SLURM_E2E_S3_PROFILE_NAME}" \
    --input-s3-profile-name "${SLURM_E2E_S3_PROFILE_NAME}" \
    --output-s3-profile-name "${SLURM_E2E_S3_PROFILE_NAME}" \
    --annotation-version v0 \
    --semantic-dedup-epsilon 0.01
  log "Shard pipeline completed"
}

run_split_lance() {
  log "Running Lance split pipeline -> ${SLURM_E2E_OUTPUT_CLIP_LANCE_PATH}"
  python -m cosmos_curator.pipelines.video.run_pipeline split \
    --input-video-path "${S3_INPUT_VIDEO_PATH}" \
    --output-clip-path "${SLURM_E2E_OUTPUT_CLIP_LANCE_PATH}" \
    --input-s3-profile-name "${SLURM_E2E_S3_PROFILE_NAME}" \
    --output-s3-profile-name "${SLURM_E2E_S3_PROFILE_NAME}" \
    --limit 1 \
    --execution-mode STREAMING \
    --upload-clip-info-in-lance
  log "Lance split pipeline completed"
}

run_shard_lance() {
  log "Running Lance shard pipeline -> ${SLURM_E2E_OUTPUT_DATASET_LANCE_PATH}"
  python -m cosmos_curator.pipelines.video.run_pipeline shard \
    --input-clip-path "${SLURM_E2E_OUTPUT_CLIP_LANCE_PATH}" \
    --output-dataset-path "${SLURM_E2E_OUTPUT_DATASET_LANCE_PATH}" \
    --input-s3-profile-name "${SLURM_E2E_S3_PROFILE_NAME}" \
    --output-s3-profile-name "${SLURM_E2E_S3_PROFILE_NAME}" \
    --annotation-version v0 \
    --metadata-input-format lance
  log "Lance shard pipeline completed"
}

run_split
run_dedup
run_shard

if [[ "${SLURM_E2E_RUN_LANCE}" == "1" ]]; then
  run_split_lance
  run_shard_lance
fi

log "SLURM end-to-end pipeline finished successfully"
