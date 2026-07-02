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
"""Result row and report types for caption judging."""

from collections.abc import Sequence
from typing import Literal, TypedDict

import attrs
import pyarrow as pa

from cosmos_curator.pipelines.ray_data.caption_judge.config import CaptionJudgePipelineConfig

IssueCode = Literal[
    "caption_window_not_comparable",
    "caption_judge_failed",
    "caption_judge_invalid_response",
    "caption_judge_prefers_baseline",
]

ISSUE_SCHEMA: pa.Schema = pa.schema(
    [
        ("code", pa.dictionary(pa.int16(), pa.string())),
        ("message", pa.string()),
        ("video_uuid", pa.string()),
        ("clip_uuid", pa.string()),
        ("start_ns", pa.int64()),
        ("end_ns", pa.int64()),
        ("output", pa.string()),
        ("caption_model_baseline", pa.string()),
        ("caption_model_candidate", pa.string()),
        ("winner", pa.string()),
        ("confidence", pa.float64()),
        ("reason", pa.large_string()),
        ("baseline_errors", pa.list_(pa.large_string())),
        ("candidate_errors", pa.list_(pa.large_string())),
        ("baseline_caption", pa.large_string()),
        ("candidate_caption", pa.large_string()),
        ("raw_response", pa.large_string()),
    ],
)

JUDGE_JOB_SCHEMA: pa.Schema = pa.schema(
    [
        ("video_uuid", pa.string()),
        ("clip_uuid", pa.string()),
        ("start_ns", pa.int64()),
        ("end_ns", pa.int64()),
        ("clip_location", pa.large_string()),
        ("caption_model_baseline", pa.string()),
        ("caption_model_candidate", pa.string()),
        ("baseline_caption", pa.large_string()),
        ("candidate_caption", pa.large_string()),
    ],
)


class Issue(TypedDict, total=False):
    """Schema-compatible issue row."""

    code: str
    message: str
    video_uuid: str | None
    clip_uuid: str | None
    start_ns: int | None
    end_ns: int | None
    output: str | None
    caption_model_baseline: str | None
    caption_model_candidate: str | None
    winner: str | None
    confidence: float | None
    reason: str | None
    baseline_errors: list[str] | None
    candidate_errors: list[str] | None
    baseline_caption: str | None
    candidate_caption: str | None
    raw_response: str | None


class JudgeJob(TypedDict):
    """One caption window that should be sent to the hosted judge."""

    video_uuid: str
    clip_uuid: str
    start_ns: int
    end_ns: int
    clip_location: str
    caption_model_baseline: str
    caption_model_candidate: str
    baseline_caption: str
    candidate_caption: str


def make_issue(  # noqa: PLR0913 -- mirrors the persisted issue schema.
    code: IssueCode,
    message: str,
    *,
    video_uuid: str | None = None,
    clip_uuid: str | None = None,
    start_ns: int | None = None,
    end_ns: int | None = None,
    output: str | None = None,
    caption_model_baseline: str | None = None,
    caption_model_candidate: str | None = None,
    winner: str | None = None,
    confidence: float | None = None,
    reason: str | None = None,
    baseline_errors: Sequence[str] | None = None,
    candidate_errors: Sequence[str] | None = None,
    baseline_caption: str | None = None,
    candidate_caption: str | None = None,
    raw_response: str | None = None,
) -> Issue:
    """Build one issue row."""
    return Issue(
        code=code,
        message=message,
        video_uuid=video_uuid,
        clip_uuid=clip_uuid,
        start_ns=start_ns,
        end_ns=end_ns,
        output=output,
        caption_model_baseline=caption_model_baseline,
        caption_model_candidate=caption_model_candidate,
        winner=winner,
        confidence=confidence,
        reason=reason,
        baseline_errors=list(baseline_errors) if baseline_errors is not None else None,
        candidate_errors=list(candidate_errors) if candidate_errors is not None else None,
        baseline_caption=baseline_caption,
        candidate_caption=candidate_caption,
        raw_response=raw_response,
    )


def empty_issues() -> pa.Table:
    """Return an empty issue table with the caption judge issue schema."""
    return pa.Table.from_pylist([], schema=ISSUE_SCHEMA)


@attrs.define(frozen=True)
class CaptionJudgeStats:
    """Run-level counts included in the persisted report summary."""

    clips_in_baseline: int = 0
    clips_in_candidate: int = 0
    clips_in_both: int = 0
    windows_in_baseline: int = 0
    windows_in_candidate: int = 0
    windows_in_both: int = 0
    windows_judged: int = 0


@attrs.define(frozen=True)
class Report:
    """Result of one caption judge run."""

    issues: pa.Table = attrs.field(eq=False)
    passed: bool
    stats: CaptionJudgeStats
    baseline: str
    candidate: str
    baseline_metadata: str
    candidate_metadata: str
    caption_model_baseline: str | None = None
    caption_model_candidate: str | None = None
    runtime_sec: float = 0.0
    config: CaptionJudgePipelineConfig | None = None
