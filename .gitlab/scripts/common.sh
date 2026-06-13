#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Common functions for CI scripts
# Usage: source "$(dirname "$0")/common.sh"

# Decode base64 S3 credentials and export path
# Args: output_path (default: /tmp/s3_creds_file)
setup_s3_credentials() {
    local output_path="${1:-/tmp/s3_creds_file}"
    echo -n "$AWS_CONFIG_FILE_CONTENTS" | base64 -d > "$output_path"
    export COSMOS_S3_PROFILE_PATH="$output_path"
    echo "S3 credentials written to $output_path"
}

# Resolve branch prefix used in image and cache tags.
# MR pipelines use target branch; push/web pipelines use commit branch.
get_branch_prefix() {
    local branch_prefix
    if [ -n "${CI_MERGE_REQUEST_TARGET_BRANCH_NAME:-}" ]; then
        branch_prefix="${CI_MERGE_REQUEST_TARGET_BRANCH_NAME##*/}"
    else
        branch_prefix="${CI_COMMIT_BRANCH##*/}"
    fi
    echo "${branch_prefix}"
}

# Generate consistent image tag from CI variables
# Mirrors the compute_image_tag anchor in .gitlab-ci.yml
get_image_tag() {
    local branch_prefix
    branch_prefix="$(get_branch_prefix)"
    echo "${branch_prefix}_${CI_COMMIT_TIMESTAMP%%T*}_${CI_COMMIT_SHORT_SHA}"
}

# Compute buildx registry cache references for --cache-from / --cache-to.
# Used by .build_curator_arch template in .gitlab-ci.yml.
#
# Args: image_repo, platform
# Outputs: Exports CACHE_FROM_ARGS and CACHE_TO_ARG for use by callers.
#
# MR pipelines:  cache-mr-{IID}-{platform}  (primary) + cache-main-{platform} (fallback)
# Push pipelines: cache-{branch}-{platform}
get_cache_refs() {
    local image_repo="$1"
    local platform="$2"
    local branch_prefix
    branch_prefix="$(get_branch_prefix)"

    export CACHE_FROM_ARGS=""
    export CACHE_TO_ARG=""

    # Compression knobs shared by every --cache-to. Registry cache blobs are
    # only ever read back by BuildKit (they are never pulled as a runnable
    # image). Switching them to zstd is safe and compresses several times
    # faster than the default gzip. force-compression=false lets unchanged
    # layers be reused as-is instead of being recompressed on every export.
    local cache_compression="compression=zstd,compression-level=3,image-manifest=true,oci-mediatypes=true,force-compression=false"

    if [ -n "${CI_MERGE_REQUEST_IID:-}" ]; then
        local mr_ref="${image_repo}:cache-mr-${CI_MERGE_REQUEST_IID}-${platform}"
        local main_ref="${image_repo}:cache-main-${platform}"
        CACHE_FROM_ARGS="--cache-from type=registry,ref=${mr_ref} --cache-from type=registry,ref=${main_ref}"
        CACHE_TO_ARG="--cache-to type=registry,ref=${mr_ref},mode=min,${cache_compression}"
    else
        local branch_ref="${image_repo}:cache-${branch_prefix}-${platform}"
        CACHE_FROM_ARGS="--cache-from type=registry,ref=${branch_ref}"
        CACHE_TO_ARG="--cache-to type=registry,ref=${branch_ref},mode=max,${cache_compression}"
    fi
}

# Ensure a persistent buildx builder with the docker-container driver.
# The default "docker" driver does not support registry cache export
# (--cache-to type=registry). The docker-container driver runs BuildKit
# in a container, which supports all cache backends.
#
# The builder is kept alive across jobs so its local layer cache stays
# hot. Only layers missing from the local cache are pulled from the
# registry (--cache-from acts as a cross-runner fallback).
#
#   First job on runner:  create builder --> cold, pulls from registry
#   Subsequent jobs:      reuse builder  --> hot local cache, fast builds
#
# Must run AFTER docker login so the builder can reach the registry.
#
# Args: builder_name (default: $BUILDX_BUILDER_NAME or ci-cosmos-builder)
setup_buildx_builder() {
    local builder_name="${1:-${BUILDX_BUILDER_NAME:-ci-cosmos-builder}}"
    if ! docker buildx inspect "${builder_name}" &>/dev/null; then
        docker buildx create \
            --name "${builder_name}" \
            --driver docker-container \
            --driver-opt network=host
        echo "Created buildx builder: ${builder_name}"
    else
        echo "Reusing existing buildx builder: ${builder_name}"
    fi
    docker buildx use "${builder_name}"
}

# Canonical NGC credentials
# NGC_NVCF_ORG is set in .gitlab-ci.yml (defaults to DEV org)
# NGC_API_KEY defaults to DEV key (vault secrets not available at CI variable definition time)
: "${NGC_API_KEY:=${NVCF_KEY:-}}"
: "${NGC_ORG:=${NGC_NVCF_ORG:-${NVCF_ORG_ID:-}}}"

# Configure environment for NGC model downloads
setup_ngc_model_download() {
    export NVCF_MULTI_NODE=true
    export NGC_NVCF_API_KEY="${NGC_API_KEY}"
    export NGC_NVCF_ORG="${NGC_ORG}"
    echo "NGC model download configured (org: ${NGC_ORG})"
}

# Wait for S3 file to appear
# Args: s3_path, max_attempts (default: 10), sleep_seconds (default: 5)
wait_for_s3_file() {
    local path="$1"
    local max="${2:-10}"
    local sleep_sec="${3:-5}"

    for ((i=0; i<max; i++)); do
        if aws s3 ls "$path" &>/dev/null; then
            echo "Found: $path"
            return 0
        fi
        echo "Waiting for $path... ($((i+1))/$max)"
        sleep "$sleep_sec"
    done
    echo "ERROR: $path not found after $max attempts"
    return 1
}

# Validate JSON from S3
# Args: s3_path
validate_s3_json() {
    local path="$1"
    local content
    if ! content=$(aws s3 cp "$path" - 2>/dev/null); then
        echo "ERROR: Failed to read $path"
        return 1
    fi
    if ! jq empty <<< "$content" 2>/dev/null; then
        echo "ERROR: Invalid JSON in $path"
        return 1
    fi
    echo "$content"
}

# Standard pipeline run arguments for reduced CPU usage (fits 8-core nodes)
# Returns args string suitable for appending to pipeline command
get_reduced_cpu_pipeline_args() {
    echo "--transnetv2-frame-decode-cpus-per-worker 1 --transcode-cpus-per-worker 1 --clip-extraction-cpus-per-worker 1"
}

_SLURM_LOG_TAIL_PID=""

stop_slurm_log_tail() {
    if [[ -n "${_SLURM_LOG_TAIL_PID}" ]]; then
        kill "${_SLURM_LOG_TAIL_PID}" >/dev/null 2>&1 || true
        wait "${_SLURM_LOG_TAIL_PID}" 2>/dev/null || true
        _SLURM_LOG_TAIL_PID=""
    fi
}

start_slurm_log_tail() {
    local log_file=$1
    (
        while [[ ! -f "${log_file}" ]]; do
            sleep 5
        done
        echo "---- Streaming SLURM job log (${log_file}) ----"
        exec tail --pid="$$" -n +1 -F "${log_file}"
    ) &
    _SLURM_LOG_TAIL_PID=$!
}

wait_for_slurm_log_drain() {
    local log_file=$1
    local previous_size=""
    local current_size
    local stable_attempts=0

    for _ in {1..10}; do
        if [[ ! -f "${log_file}" ]]; then
            sleep 1
            continue
        fi

        current_size=$(stat -c %s "${log_file}" 2>/dev/null || true)
        if [[ -n "${current_size}" && "${current_size}" == "${previous_size}" ]]; then
            stable_attempts=$((stable_attempts + 1))
            if (( stable_attempts >= 2 )); then
                return 0
            fi
        else
            previous_size="${current_size}"
            stable_attempts=0
        fi

        sleep 1
    done
}

wait_for_slurm_job() {
    local job_id=$1
    local max_attempts=$2
    local sleep_seconds=$3
    local attempt=0

    while (( attempt < max_attempts )); do
        if [[ -n "$(squeue -h -j "${job_id}" 2>/dev/null)" ]]; then
            echo "[$(date -Ins)] Job ${job_id} still running..."
        else
            local state
            state=$(sacct -j "${job_id}" -o State -n | head -n 1 | tr -d ' ')
            echo "[$(date -Ins)] Job ${job_id} completed with state ${state}"
            if [[ "${state}" == COMPLETED* ]]; then
                return 0
            fi
            return 1
        fi
        sleep "${sleep_seconds}"
        attempt=$((attempt + 1))
    done

    echo "Timeout waiting for job ${job_id}" >&2
    return 1
}

monitor_slurm_job() {
    # Args: job_id, log_file, max_attempts, sleep_seconds
    local job_id=$1
    local log_file=$2
    local max_attempts=$3
    local sleep_seconds=$4

    start_slurm_log_tail "${log_file}"

    if ! wait_for_slurm_job "${job_id}" "${max_attempts}" "${sleep_seconds}"; then
        stop_slurm_log_tail
        if [[ -f "${log_file}" ]]; then
            echo "---- SLURM job log (${log_file}) ----"
            tail -n 200 "${log_file}"
        else
            echo "SLURM log file ${log_file} was not found" >&2
        fi
        return 1
    fi

    wait_for_slurm_log_drain "${log_file}"
    stop_slurm_log_tail
    if [[ -f "${log_file}" ]]; then
        echo "Collected SLURM log at ${log_file}"
    fi
}

trap stop_slurm_log_tail EXIT
