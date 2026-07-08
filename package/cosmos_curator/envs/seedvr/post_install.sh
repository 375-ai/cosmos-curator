#!/usr/bin/env bash
# Post-install for seedvr environment: flash-attn precompiled wheel.
set -euo pipefail

ARCH="$(uname -m)"
if [[ "${ARCH}" != "x86_64" ]]; then
    echo "Skipping flash-attn install on ${ARCH} (no precompiled wheel available)."
    exit 0
fi

# Precompiled flash-attn wheel for CUDA 13.0 + torch 2.11 + Python 3.13 (x86_64 only).
# Update from https://github.com/mjun0812/flash-attention-prebuild-wheels/releases
# and keep FLASH_ATTN_WHL_SHA256 in sync with the release asset digest.
FLASH_ATTN_WHL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3%2Bcu130torch2.11-cp313-cp313-linux_x86_64.whl"
FLASH_ATTN_WHL_SHA256="f4161833c5a710ab4ab409c0fbdb7a6971b46513646e07f0ff59b19cc98880b0"
FLASH_ATTN_REQ="$(mktemp)"
trap 'rm -f "${FLASH_ATTN_REQ}"' EXIT
printf '%s --hash=sha256:%s\n' "${FLASH_ATTN_WHL}" "${FLASH_ATTN_WHL_SHA256}" > "${FLASH_ATTN_REQ}"
pip install --no-cache-dir --require-hashes -r "${FLASH_ATTN_REQ}"
