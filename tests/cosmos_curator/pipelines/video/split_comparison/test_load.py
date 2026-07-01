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
"""Tests for clip-metadata loading: the unique-clip_uuid contract is enforced at the boundary."""

from pathlib import Path

import lance
import pyarrow as pa
import pytest

from cosmos_curator.pipelines.video.split_comparison.load import DEFAULT_LANCE_VERSION, load_clip_metadata


def _clip_table(clip_uuids: list[str]) -> pa.Table:
    """Build a minimal clip-metadata table with the required columns, one row per given uuid."""
    n = len(clip_uuids)
    return pa.table(
        {
            "clip_uuid": clip_uuids,
            "video_uuid": ["v"] * n,
            "aesthetic_score": [1.0] * n,
            "motion_score": [0.0] * n,
            "valid": [True] * n,
            "has_caption": [True] * n,
            "rejection_stage": [""] * n,
            "windows": [""] * n,
        }
    )


def _write_output(tmp_path: Path, table: pa.Table) -> str:
    """Persist ``table`` as an output's clip-metadata Lance dataset; return the output root."""
    root = tmp_path / "out"
    lance.write_dataset(table, str(root / "lance" / DEFAULT_LANCE_VERSION), mode="overwrite")
    return str(root)


def test_duplicate_clip_uuid_fails_loud(tmp_path: Path) -> None:
    """A duplicate clip_uuid (producer-contract violation) is rejected at load, not handled silently.

    Downstream the clip-diff join would fan out on the dup while window alignment collapses
    last-wins -- a garbled comparison with no error. The load boundary must surface it instead.
    """
    root = _write_output(tmp_path, _clip_table(["dup", "dup", "unique"]))
    with pytest.raises(ValueError, match="unique-clip_uuid contract"):
        load_clip_metadata(root, profile_name="default")


def test_unique_clip_uuids_load_cleanly(tmp_path: Path) -> None:
    """A well-formed output (all clip_uuids distinct) loads without tripping the contract check."""
    root = _write_output(tmp_path, _clip_table(["a", "b", "c"]))
    table = load_clip_metadata(root, profile_name="default")
    assert table.num_rows == 3
