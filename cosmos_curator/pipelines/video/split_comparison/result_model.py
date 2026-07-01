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
"""Result-model types for split output comparison: the ``Issue`` row contract.

Every comparator emits :class:`Issue` rows that conform to the Arrow
``ISSUE_SCHEMA`` (the canonical issue contract), constructed via
:func:`make_issue`. Pairs with :mod:`...config`, which owns the input contract.
"""

import json
from typing import Any, Literal, TypedDict

import pyarrow as pa

ISSUE_SCHEMA: pa.Schema = pa.schema(
    [
        ("code", pa.dictionary(pa.int16(), pa.string())),
        ("message", pa.string()),
        ("feature", pa.string()),
        ("video", pa.string()),
        ("clip", pa.string()),
        ("output", pa.string()),
        ("field", pa.string()),
        ("details", pa.string()),  # JSON-encoded
    ],
)

# Universe of issue codes. Add new codes here when a comparator emits one.
IssueCode = Literal[
    "summary_field_mismatch",
    "summary_load_failed",
    "summary_source_layout_inconsistent",
    "summary_video_only_in_a",
    "summary_video_only_in_b",
    "summary_video_processed_state_mismatch",
    "summary_video_field_mismatch",
    "summary_clip_uuid_set_mismatch",
    "metadata_one_sided",
    "metadata_unreadable",
    "metadata_value_invalid_type",
    "metadata_value_one_sided",
    "aesthetic_score_mismatch",
    "motion_score_mismatch",
    "clip_field_mismatch",
    "caption_similarity_below_threshold",
    "clip_mp4_missing",
    "clip_mp4_unreadable",
    "clip_mp4_header_index_unavailable",
    "clip_mp4_index_mismatch",
    "clip_mp4_index_dtype_mismatch",
    "clip_mp4_metadata_mismatch",
    "clip_mp4_comparison_failed",
]


class Issue(TypedDict, total=False):
    """Row shape for the Arrow issue table.

    Carries no methods; the table is the canonical issue representation. Construct
    rows via :func:`make_issue` so keyword names are checked and ``details`` is
    JSON-encoded at the call site.
    """

    code: str
    message: str
    feature: str | None
    video: str | None
    clip: str | None
    output: str | None
    field: str | None
    details: str | None  # JSON-encoded


def make_issue(  # noqa: PLR0913 -- ISSUE_SCHEMA has 8 columns; helper mirrors them as kwargs
    code: IssueCode,
    message: str,
    *,
    feature: str | None = None,
    video: str | None = None,
    clip: str | None = None,
    output: str | None = None,
    field: str | None = None,
    details: dict[str, Any] | None = None,
) -> Issue:
    """Build a schema-compatible :class:`Issue` row.

    Keyword-only args enforce field names; ``details`` is JSON-encoded so the
    resulting row fits ``ISSUE_SCHEMA`` directly.
    """
    return Issue(
        code=code,
        message=message,
        feature=feature,
        video=video,
        clip=clip,
        output=output,
        field=field,
        details=json.dumps(details, sort_keys=True) if details is not None else None,
    )
