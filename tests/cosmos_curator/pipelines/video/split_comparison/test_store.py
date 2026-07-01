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
"""Tests for the measurements-root store: provenance manifest contents."""

from pathlib import Path

import pyarrow as pa
import pytest

from cosmos_curator.pipelines.video.split_comparison import store
from cosmos_curator.pipelines.video.split_comparison.config import SplitComparisonConfig
from cosmos_curator.pipelines.video.split_comparison.measure.core import Measurements
from cosmos_curator.pipelines.video.split_comparison.measure.schema import (
    CLIP_MEASUREMENT_SCHEMA,
    WINDOW_MEASUREMENT_SCHEMA,
)
from cosmos_curator.pipelines.video.split_comparison.store import build_manifest

_EMPTY = Measurements(
    clip_table=pa.Table.from_pylist([], schema=CLIP_MEASUREMENT_SCHEMA),
    window_table=pa.Table.from_pylist([], schema=WINDOW_MEASUREMENT_SCHEMA),
    stats={},
)


def test_manifest_records_measure_time_provenance() -> None:
    """The manifest records the knobs that shape stored values, so a root is self-describing."""
    manifest = build_manifest(
        _EMPTY,
        config=SplitComparisonConfig(output_a="/a", output_b="/b"),
        device="cuda",
        fp16=True,
        lance_version="v0",
        created_at="2026-01-01T00:00:00Z",
        summaries_snapshotted=False,
    )
    assert manifest["device"] == "cuda"
    assert manifest["fp16"] is True
    assert manifest["summaries_snapshotted"] is False


def test_manifest_records_fp16_false() -> None:
    """fp16 is recorded faithfully (not just truthy) so fp32 roots are distinguishable."""
    manifest = build_manifest(
        _EMPTY,
        config=SplitComparisonConfig(output_a="/a", output_b="/b"),
        device="cpu",
        fp16=False,
        lance_version="v0",
        created_at="2026-01-01T00:00:00Z",
        summaries_snapshotted=True,
    )
    assert manifest["fp16"] is False


def test_snapshot_write_failure_is_non_fatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed snapshot *write* (disk full / permission) returns False instead of aborting the run."""
    src_root = tmp_path / "src"
    src_root.mkdir()
    (src_root / "summary.json").write_text("{}")  # source read succeeds

    real_open = store.smart_open.open

    def open_or_fail_on_write(uri: str, mode: str = "r", **kwargs: object) -> object:
        if "w" in mode:
            msg = "disk full"
            raise PermissionError(msg)
        return real_open(uri, mode, **kwargs)

    monkeypatch.setattr(store.smart_open, "open", open_or_fail_on_write)
    # Must not raise; returns False so the measure run continues (measurements already persisted).
    assert store._snapshot_one_summary(str(src_root), str(tmp_path / "dst"), profile_name="default") is False
