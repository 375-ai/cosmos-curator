#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Extended GPU end-to-end coverage. Runs a single pre-canned scenario config
# (selected via the SCENARIO matrix variable) so the k8s_gpu_extended job can
# fan each scenario out to its own pod and keep wall-clock close to the slowest
# single scenario instead of the sum of all of them.

set -e

# shellcheck source=common.sh
source "$(dirname "$0")/common.sh"

: "${SCENARIO:?SCENARIO must be set to a scenario name under examples/ci/ (without .json)}"

# Resolve the repo root from this script's location so the pre-canned config
# JSONs always match the commit under test (not whatever is baked into the image).
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG_PATH="${REPO_ROOT}/examples/ci/${SCENARIO}.json"

echo "=== K8s GPU Extended Pipeline Test: ${SCENARIO} ==="

cd /opt/cosmos-curator

# Report GPU availability when running on GPU-tagged pods; split_openai runs on CPU.
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

# Set up model cache (persisted via hostPath at /cache)
export NVCF_MODEL_CACHE_DIR="/cache/cosmos-models"
mkdir -p "${NVCF_MODEL_CACHE_DIR}"

# Configure S3 and NGC
setup_s3_credentials
setup_ngc_model_download
if [ "${SCENARIO}" = "split_openai" ]; then
    if [ ! -s "${CI_PROJECT_DIR}/cosmos_curator_config.yaml" ]; then
        echo "ERROR: OpenAI config was not rendered from Vault" >&2
        exit 1
    fi
    mkdir -p /cosmos_curator/config
    cp "${CI_PROJECT_DIR}/cosmos_curator_config.yaml" /cosmos_curator/config/cosmos_curator.yaml
    chmod 600 /cosmos_curator/config/cosmos_curator.yaml
    rm -f "${CI_PROJECT_DIR}/cosmos_curator_config.yaml"
fi

# Per-scenario output path so concurrent matrix pods never collide.
K8S_OUTPUT_PATH="${S3_OUTPUT_PATH}/k8s-gpu-extended/${SCENARIO}"

run_pipeline_from_config "${CONFIG_PATH}" "${K8S_OUTPUT_PATH}"

if [ "${SCENARIO}" = "split_openai" ]; then
    rm -f /cosmos_curator/config/cosmos_curator.yaml
fi

if [ "${SCENARIO}" = "split_openai" ]; then
    CAPTION_QUALITY_STATS_PATH="${K8S_OUTPUT_PATH}/caption_quality_stats.json" pixi run --as-is python - <<'PY'
import os
import sys

from cosmos_curator.core.utils.storage import storage_utils

stats_path = os.environ["CAPTION_QUALITY_STATS_PATH"]
client = storage_utils.get_storage_client(stats_path, profile_name="default")
try:
    stats = storage_utils.read_json_file(stats_path, client)
except Exception as exc:
    print(f"ERROR: Failed to read OpenAI caption quality stats from {stats_path}: {exc}", file=sys.stderr)
    raise SystemExit(1) from exc

counts = stats.get("caption_status_counts") or {}
checked = int(stats.get("caption_windows_checked") or 0)
success = int(counts.get("success") or 0)
errors = int(counts.get("error") or 0)

if checked <= 0:
    print(f"ERROR: OpenAI validation found no caption windows in {stats_path}", file=sys.stderr)
    raise SystemExit(1)
if success <= 0:
    print(f"ERROR: OpenAI validation found no successful captions in {stats_path}: {counts}", file=sys.stderr)
    raise SystemExit(1)
if errors:
    print(f"ERROR: OpenAI validation found caption errors in {stats_path}: {counts}", file=sys.stderr)
    raise SystemExit(1)

print(f"OpenAI validation passed: {success}/{checked} caption windows succeeded")
PY
fi

echo "✓ K8s GPU extended pipeline test (${SCENARIO}) completed successfully"
