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
"""Tests for summary_compare: A/B summary.json comparison emitting a pa.Table."""

import json
from collections.abc import Mapping
from typing import Any

import pyarrow as pa

from cosmos_curator.pipelines.video.split_comparison.config import SummaryPolicy
from cosmos_curator.pipelines.video.split_comparison.result_model import ISSUE_SCHEMA
from cosmos_curator.pipelines.video.split_comparison.summary_compare import compare_summaries
from cosmos_curator.pipelines.video.split_comparison.summary_schema import (
    OutputSummary,
    ProcessedVideoSummary,
    UnprocessedVideoSummary,
    VideoSummary,
)


def _processed_video(
    key: str,
    *,
    clips: tuple[str, ...] = (),
    filtered_clips: tuple[str, ...] = (),
) -> ProcessedVideoSummary:
    return ProcessedVideoSummary(
        source_video=f"/inputs/{key}",
        video_uuid=f"uuid-{key}",
        num_clip_chunks=1,
        num_total_clips=len(clips) + len(filtered_clips),
        num_clips_filtered_by_motion=0,
        num_clips_filtered_by_aesthetic=0,
        num_clips_filtered_by_qwen_classifier=0,
        num_clips_filtered_by_qwen_semantic=0,
        num_clips_filtered_by_artificial_text=0,
        num_clips_passed=len(clips),
        num_clips_transcoded=len(clips) + len(filtered_clips),
        num_clips_with_embeddings=len(clips),
        num_clips_with_caption=0,
        num_caption_windows=0,
        num_clips_with_webp=len(clips),
        clips=clips,
        filtered_clips=filtered_clips,
    )


def _unprocessed_video(key: str) -> UnprocessedVideoSummary:
    return UnprocessedVideoSummary(processed=False, source_video=f"/inputs/{key}")


def _summary(
    videos: Mapping[str, VideoSummary],
    **overrides: Any,  # noqa: ANN401 -- test fixture, fields are heterogeneous OutputSummary kwargs
) -> OutputSummary:
    """Build a minimal OutputSummary populating only the fields the comparison reads."""
    base: dict[str, Any] = {
        "videos": dict(videos),
        "num_input_videos": len(videos),
        "num_input_videos_selected": len(videos),
        "num_processed_videos": sum(1 for v in videos.values() if v.processed),
        "embedding_algorithm": "internvideo2",
        "total_video_duration": 10.0,
        "total_clip_duration": 8.0,
        "max_clip_duration": 4.0,
        "total_video_bytes": 12345,
        "num_remuxed_videos": 0,
        "total_num_clips_filtered_by_motion": 0,
        "total_num_clips_filtered_by_aesthetic": 0,
        "total_num_clips_filtered_by_qwen_classifier": 0,
        "total_num_clips_filtered_by_qwen_semantic": 0,
        "total_num_clips_filtered_by_artificial_text": 0,
        "total_num_clips_passed": 0,
        "total_num_clips_transcoded": 0,
        "total_num_clips_with_embeddings": 0,
        "total_num_clips_with_caption": 0,
        "total_num_caption_windows": 0,
        "total_num_clips_with_webp": 0,
        "total_prompt_tokens": 0,
        "total_output_tokens": 0,
    }
    base.update(overrides)
    return OutputSummary(**base)


def _codes(issues: pa.Table) -> list[str]:
    return list(issues["code"].to_pylist())


def _by_code(issues: pa.Table, code: str) -> list[dict[str, object]]:
    rows = issues.filter(pa.compute.equal(issues["code"], code)).to_pylist()
    for row in rows:
        if row.get("details") is not None:
            row["details"] = json.loads(row["details"])
    return rows


def test_compare_summaries_returns_arrow_table_with_schema() -> None:
    """Output is always a pa.Table with ISSUE_SCHEMA, even when nothing differs."""
    summary = _summary({"video.mp4": _processed_video("video.mp4")})

    table = compare_summaries(summary, summary, SummaryPolicy())

    assert table.schema == ISSUE_SCHEMA
    assert table.num_rows == 0


def test_compare_summaries_flags_top_level_field_mismatches() -> None:
    """Top-level counter fields are compared by strict equality."""
    summary_a = _summary({}, num_processed_videos=10)
    summary_b = _summary({}, num_processed_videos=9)

    table = compare_summaries(summary_a, summary_b, SummaryPolicy())

    matches = _by_code(table, "summary_field_mismatch")
    assert len(matches) == 1
    issue = matches[0]
    assert issue["field"] == "num_processed_videos"
    assert issue["details"] == {"a": 10, "b": 9}


def test_compare_summaries_flags_only_in_a_and_only_in_b_videos() -> None:
    """One-sided video keys produce summary_video_only_in_{a,b} issues."""
    summary_a = _summary(
        {
            "shared.mp4": _processed_video("shared.mp4"),
            "a-only.mp4": _processed_video("a-only.mp4"),
        },
    )
    summary_b = _summary(
        {
            "shared.mp4": _processed_video("shared.mp4"),
            "b-only.mp4": _processed_video("b-only.mp4"),
        },
    )

    table = compare_summaries(summary_a, summary_b, SummaryPolicy())

    only_a = _by_code(table, "summary_video_only_in_a")
    only_b = _by_code(table, "summary_video_only_in_b")
    assert len(only_a) == 1
    assert only_a[0]["video"] == "a-only.mp4"
    assert len(only_b) == 1
    assert only_b[0]["video"] == "b-only.mp4"


def test_compare_summaries_flags_processed_state_mismatch() -> None:
    """A video that's processed on one output and not the other reports the mismatch."""
    summary_a = _summary({"video.mp4": _processed_video("video.mp4")})
    summary_b = _summary({"video.mp4": _unprocessed_video("video.mp4")})

    table = compare_summaries(summary_a, summary_b, SummaryPolicy())

    matches = _by_code(table, "summary_video_processed_state_mismatch")
    assert len(matches) == 1
    assert matches[0]["video"] == "video.mp4"
    assert matches[0]["details"] == {"a_processed": True, "b_processed": False}


def test_compare_summaries_flags_per_video_field_mismatches() -> None:
    """Per-video processed-summary fields compare by strict equality on the typed values."""
    summary_a = _summary({"video.mp4": _processed_video("video.mp4", clips=("c1", "c2"))})
    summary_b = _summary({"video.mp4": _processed_video("video.mp4", clips=("c1",))})

    table = compare_summaries(summary_a, summary_b, SummaryPolicy())

    field_mismatches = _by_code(table, "summary_video_field_mismatch")
    # clips differ -> num_clips_passed, num_clips_transcoded, num_total_clips,
    # num_clips_with_embeddings, num_clips_with_webp all change in _processed_video helper.
    fields = sorted({issue["field"] for issue in field_mismatches})
    assert "num_clips_passed" in fields
    assert all(issue["video"] == "video.mp4" for issue in field_mismatches)


def test_compare_summaries_flags_clip_uuid_set_mismatches() -> None:
    """Per-video clip uuid set differences surface as summary_clip_uuid_set_mismatch issues."""
    summary_a = _summary({"video.mp4": _processed_video("video.mp4", clips=("c1", "c2"))})
    summary_b = _summary({"video.mp4": _processed_video("video.mp4", clips=("c1", "c3"))})

    table = compare_summaries(summary_a, summary_b, SummaryPolicy())

    matches = _by_code(table, "summary_clip_uuid_set_mismatch")
    clips_match = next(issue for issue in matches if issue["field"] == "clips")
    assert clips_match["video"] == "video.mp4"
    assert clips_match["details"] == {"only_in_a": ["c2"], "only_in_b": ["c3"]}


def test_compare_summaries_no_field_mismatch_when_summaries_match() -> None:
    """Identical summaries produce zero issues."""
    summary = _summary({"video.mp4": _processed_video("video.mp4", clips=("c1",))})

    table = compare_summaries(summary, summary, SummaryPolicy())

    assert table.num_rows == 0
    assert _codes(table) == []


def test_token_count_tolerance_absorbs_small_drift() -> None:
    """Token-count fields tolerate drift within the configured abs/rel tolerance."""
    summary_a = _summary({}, total_prompt_tokens=10_000, total_output_tokens=5_000)
    summary_b = _summary({}, total_prompt_tokens=10_050, total_output_tokens=5_025)
    policy = SummaryPolicy(token_count_abs_tolerance=0, token_count_rel_tolerance=0.01)

    table = compare_summaries(summary_a, summary_b, policy)

    assert _by_code(table, "summary_field_mismatch") == []


def test_token_count_tolerance_does_not_absorb_large_drift() -> None:
    """Drift exceeding the relative tolerance is still reported."""
    summary_a = _summary({}, total_prompt_tokens=10_000)
    summary_b = _summary({}, total_prompt_tokens=11_000)
    policy = SummaryPolicy(token_count_abs_tolerance=0, token_count_rel_tolerance=0.01)

    table = compare_summaries(summary_a, summary_b, policy)

    mismatches = _by_code(table, "summary_field_mismatch")
    assert [issue["field"] for issue in mismatches] == ["total_prompt_tokens"]


def test_token_count_tolerance_does_not_apply_to_non_token_fields() -> None:
    """Other counter fields keep strict equality even when token tolerance is loose."""
    summary_a = _summary({}, num_processed_videos=100)
    summary_b = _summary({}, num_processed_videos=101)
    policy = SummaryPolicy(token_count_abs_tolerance=10, token_count_rel_tolerance=0.5)

    table = compare_summaries(summary_a, summary_b, policy)

    mismatches = _by_code(table, "summary_field_mismatch")
    assert [issue["field"] for issue in mismatches] == ["num_processed_videos"]
