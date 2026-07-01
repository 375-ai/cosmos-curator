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
"""Tests for the Issue type, make_issue, and the Arrow issue-schema round-trip."""

import json
from typing import get_args

import pyarrow as pa

from cosmos_curator.pipelines.video.split_comparison.result_model import (
    ISSUE_SCHEMA,
    IssueCode,
    make_issue,
)


def test_make_issue_produces_row_that_fits_issue_schema() -> None:
    """A fully-populated row from make_issue lands in the Arrow table cleanly."""
    row = make_issue(
        code="caption_similarity_below_threshold",
        message="similarity 0.71 below 0.85",
        feature="captions",
        video="video.mp4",
        clip="clip-a",
        output=None,
        field="caption",
        details={"start_ns": 0, "similarity": 0.71},
    )
    table = pa.Table.from_pylist([row], schema=ISSUE_SCHEMA)

    assert table.num_rows == 1
    assert table["code"][0].as_py() == "caption_similarity_below_threshold"
    assert table["field"][0].as_py() == "caption"
    parsed = json.loads(table["details"][0].as_py())
    assert parsed == {"start_ns": 0, "similarity": 0.71}


def test_clip_field_mismatch_is_a_valid_issue_code() -> None:
    """clip_field_mismatch is registered in IssueCode and round-trips through the schema."""
    assert "clip_field_mismatch" in get_args(IssueCode)
    row = make_issue(
        code="clip_field_mismatch",
        message="Clip field 'valid' differs between outputs",
        feature="metadata_structure",
        clip="clip-a",
        field="valid",
        details={"a": True, "b": False},
    )
    table = pa.Table.from_pylist([row], schema=ISSUE_SCHEMA)

    assert table["code"][0].as_py() == "clip_field_mismatch"
    assert table["field"][0].as_py() == "valid"
    assert json.loads(table["details"][0].as_py()) == {"a": True, "b": False}


def test_make_issue_leaves_unset_fields_null_with_no_details() -> None:
    """Optional kwargs default to None; details stays null when no dict is provided."""
    row = make_issue(code="summary_field_mismatch", message="num_input_videos differs")
    table = pa.Table.from_pylist([row], schema=ISSUE_SCHEMA)

    assert table["clip"][0].as_py() is None
    assert table["video"][0].as_py() is None
    assert table["details"][0].as_py() is None


def test_make_issue_serializes_details_with_sorted_keys() -> None:
    """Details JSON is sorted-key so identical content produces identical bytes."""
    row_a = make_issue(code="aesthetic_score_mismatch", message="x", details={"a": 1, "b": 2})
    row_b = make_issue(code="aesthetic_score_mismatch", message="x", details={"b": 2, "a": 1})

    assert row_a["details"] == row_b["details"]
