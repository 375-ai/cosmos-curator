# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the nanosecond clip metadata Lance schema and its normalization."""

from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]

from cosmos_curator.pipelines.video.read_write.clip_metadata_lance_schema import (
    _FILTERED_WINDOW_STRUCT,
    _WINDOW_STRUCT,
    CLIP_METADATA_LANCE_SCHEMA,
    CLIP_METADATA_LANCE_SCHEMA_VERSION,
    build_clip_metadata_lance_table,
    clip_metadata_row_to_lance_row,
)


def _window_struct_field_names(struct: pa.StructType) -> set[str]:
    return {struct.field(i).name for i in range(struct.num_fields)}


class TestSchemaShape:
    """The schema is nanosecond-based, not seconds/frames-based."""

    def test_schema_version(self) -> None:
        """The schema version stays 1: the prior seconds-based schema was never published."""
        assert CLIP_METADATA_LANCE_SCHEMA_VERSION == 1

    def test_clip_schema_uses_ns_not_seconds(self) -> None:
        """Clip timing fields are start_ns/end_ns/duration_ns; the *_s fields are gone."""
        names = set(CLIP_METADATA_LANCE_SCHEMA.names)
        assert {"start_ns", "end_ns", "duration_ns"} <= names
        assert not ({"span_start_s", "span_end_s", "duration_s"} & names)

    def test_window_structs_use_ns_not_frames(self) -> None:
        """Window structs expose start_ns/end_ns and no longer carry frame indices."""
        for struct in (_WINDOW_STRUCT, _FILTERED_WINDOW_STRUCT):
            names = _window_struct_field_names(struct)
            assert {"start_ns", "end_ns"} <= names
            assert not ({"start_frame", "end_frame"} & names)


class TestNormalization:
    """clip_metadata_row_to_lance_row maps the ns row contract onto the schema."""

    def _row(self) -> dict[str, Any]:
        return {
            "span_uuid": "clip-1",
            "source_video": "input/video.mp4",
            "start_ns": 1_000_000_000,
            "end_ns": 5_000_000_000,
            "windows": [{"start_ns": 0, "end_ns": 2_000_000_000, "caption_status": "success"}],
            "filtered_windows": [{"start_ns": 0, "end_ns": 1_000_000_000, "qwen_rejection_reasons": "blurry"}],
        }

    def test_clip_ns_and_duration(self) -> None:
        """Clip start/end ns pass through and duration_ns is derived from them."""
        lance_row = clip_metadata_row_to_lance_row(self._row(), video_uuid="vid-1", clip_chunk_index=0)
        assert lance_row["start_ns"] == 1_000_000_000
        assert lance_row["end_ns"] == 5_000_000_000
        assert lance_row["duration_ns"] == 4_000_000_000

    def test_window_ns(self) -> None:
        """Window and filtered-window ns bounds are carried onto the lance row."""
        lance_row = clip_metadata_row_to_lance_row(self._row(), video_uuid="vid-1", clip_chunk_index=0)
        assert lance_row["windows"][0]["start_ns"] == 0
        assert lance_row["windows"][0]["end_ns"] == 2_000_000_000
        assert lance_row["filtered_windows"][0]["start_ns"] == 0
        assert lance_row["filtered_windows"][0]["end_ns"] == 1_000_000_000

    def test_missing_ns_is_null(self) -> None:
        """Absent clip ns (e.g. an errored clip) yields null bounds and null duration."""
        lance_row = clip_metadata_row_to_lance_row({"span_uuid": "clip-2"}, video_uuid="vid-1", clip_chunk_index=0)
        assert lance_row["start_ns"] is None
        assert lance_row["end_ns"] is None
        assert lance_row["duration_ns"] is None

    def test_build_table_matches_schema(self) -> None:
        """The assembled table conforms to the canonical schema."""
        table = build_clip_metadata_lance_table([self._row()], video_uuid="vid-1", clip_chunk_index=2)
        assert table.schema.equals(CLIP_METADATA_LANCE_SCHEMA, check_metadata=True)
        assert table.to_pylist()[0]["duration_ns"] == 4_000_000_000
