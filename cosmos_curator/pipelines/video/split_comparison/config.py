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
"""Input contract for split output comparison -- one ``SplitComparisonConfig``.

Pydantic v2 models with ``frozen=True``, ``strict=True``, ``extra="forbid"``:
no field coercion, no silent typos, immutable once constructed. A single config
is a self-describing audit spec -- the comparison targets (``output_a`` /
``output_b``), the storage profile, and the per-feature comparison policies.
The CLI loads one via ``--config PATH``.
"""

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_PROFILE_NAME = "default"

# Project precedent: every config model uses this triple.
#   frozen=True       -- immutability by design
#   strict=True       -- no "5" -> 5 coercion; YAML/JSON typos in value types fail loudly
#   extra="forbid"    -- typos in field names fail at load (instead of silently ignored)
_CONFIG_MODEL_CONFIG = ConfigDict(frozen=True, strict=True, extra="forbid")


class SummaryPolicy(BaseModel):
    """Knobs for :mod:`summary_compare`.

    ``token_count_abs_tolerance`` / ``token_count_rel_tolerance`` apply to the
    ``total_prompt_tokens`` and ``total_output_tokens`` summary fields. Everything
    else is compared by strict equality.
    """

    model_config = _CONFIG_MODEL_CONFIG

    token_count_abs_tolerance: float = Field(default=0.0, ge=0.0)
    token_count_rel_tolerance: float = Field(default=0.0, ge=0.0)


class ScoreTolerance(BaseModel):
    """Abs / rel tolerance for scalar per-clip score comparisons (aesthetic, motion)."""

    model_config = _CONFIG_MODEL_CONFIG

    abs_tolerance: float = Field(default=1e-6, ge=0.0)
    rel_tolerance: float = Field(default=1e-6, ge=0.0)


class CaptionPolicy(BaseModel):
    """Knobs for caption comparison: which embedding model to load and how strict the similarity check is.

    ``encode_batch_size`` is the chunk size handed to
    :func:`SentenceTransformer.encode` inside the cross-clip batched caption
    path; the optimum depends on machine and model, so it's tuneable. Default
    128 is CPU-friendly for BGE-small at audit batch sizes.
    """

    model_config = _CONFIG_MODEL_CONFIG

    model_id: str = Field(default="BAAI/bge-small-en-v1.5", min_length=1)
    min_similarity: float = Field(default=0.85, ge=0.0, le=1.0)
    encode_batch_size: int = Field(default=128, ge=1)


class SplitComparisonConfig(BaseModel):
    """Top-level configuration for a split-output comparison.

    Holds the comparison targets (``output_a`` / ``output_b``), the storage
    profile, and the per-feature comparison policies. Designed for JSON / YAML
    round-trip via :meth:`model_validate_json` / :meth:`model_dump_json`.
    """

    model_config = _CONFIG_MODEL_CONFIG

    # Comparison targets.
    output_a: str = Field(min_length=1)
    output_b: str = Field(min_length=1)

    # Storage profile used when reading both outputs.
    profile_name: str = Field(default=DEFAULT_PROFILE_NAME, min_length=1)

    # Comparison policies.
    summary: SummaryPolicy = Field(default_factory=SummaryPolicy)
    aesthetic: ScoreTolerance = Field(default_factory=ScoreTolerance)
    motion: ScoreTolerance = Field(default_factory=ScoreTolerance)
    caption: CaptionPolicy = Field(default_factory=CaptionPolicy)

    # Feature toggles.
    compare_captions: bool = True
