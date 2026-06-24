# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fatal vLLM decode failures that should stop captioning instead of becoming window errors."""

import hashlib

_DIAGNOSTIC_HASH_CHARS = 16


class FatalVllmDecodeError(RuntimeError):
    """A model response violated decoder invariants; do not continue best-effort."""


class ReasoningOutputTruncatedError(FatalVllmDecodeError):
    """Generation hit the token limit before a complete answer could be extracted."""


def raise_if_reasoning_output_truncated(
    *,
    text: str,
    finish_reason: str | None,
    model_name: str,
) -> None:
    """Raise when a reasoning-style decode exhausts max tokens before producing a complete answer."""
    if finish_reason != "length" or "</think>" in text.casefold():
        return

    output_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:_DIAGNOSTIC_HASH_CHARS]
    msg = (
        f"{model_name} output hit max_tokens before a complete answer could be safely extracted. This is likely a "
        f"reasoning-model configuration error: disable/close thinking for captioning, or increase the max output token "
        f"limit (splitting pipeline: --captioning-max-output-tokens). output_chars={len(text)}, "
        f"output_sha256_prefix={output_hash}"
    )
    raise ReasoningOutputTruncatedError(msg)
