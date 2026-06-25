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

# Verify GPU is available
nvidia-smi

# Set up model cache (persisted via hostPath at /cache)
export NVCF_MODEL_CACHE_DIR="/cache/cosmos-models"
mkdir -p "${NVCF_MODEL_CACHE_DIR}"

# Configure S3 and NGC
setup_s3_credentials
setup_ngc_model_download

# Per-scenario output path so concurrent matrix pods never collide.
K8S_OUTPUT_PATH="${S3_OUTPUT_PATH}/k8s-gpu-extended/${SCENARIO}"

run_pipeline_from_config "${CONFIG_PATH}" "${K8S_OUTPUT_PATH}"

echo "✓ K8s GPU extended pipeline test (${SCENARIO}) completed successfully"
