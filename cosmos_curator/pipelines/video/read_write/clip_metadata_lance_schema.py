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
"""Arrow schema and normalization for clip metadata Lance output."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal, SupportsFloat, SupportsIndex, SupportsInt

import pyarrow as pa  # type: ignore[import-untyped]

CLIP_METADATA_LANCE_SCHEMA_VERSION = 1
CLIP_METADATA_LANCE_DATA_STORAGE_VERSION: Literal["2.2"] = "2.2"
_TRUE_STRINGS = frozenset({"1", "on", "true", "yes", "y"})
_FALSE_STRINGS = frozenset({"", "0", "false", "no", "n", "off"})

_STRING_MAP = pa.map_(pa.string(), pa.large_string())
_TOKEN_COUNTS_MAP = pa.map_(
    pa.string(),
    pa.struct(
        [
            pa.field("prompt_tokens", pa.int64()),
            pa.field("output_tokens", pa.int64()),
        ]
    ),
)

_WINDOW_STRUCT = pa.struct(
    [
        # Window bounds in nanoseconds, relative to the clip start (derived from the
        # clip's own per-frame PTS), not the source timeline.
        pa.field("start_ns", pa.int64()),
        pa.field("end_ns", pa.int64()),
        pa.field("caption_status", pa.string()),
        pa.field("caption_failure_reason", pa.string()),
        pa.field("captions", _STRING_MAP),
        pa.field("enhanced_captions", _STRING_MAP),
        pa.field("token_counts", _TOKEN_COUNTS_MAP),
        pa.field("flag_length_outlier", pa.bool_()),
        pa.field("flag_repetition", pa.bool_()),
        pa.field("flag_near_duplicate", pa.bool_()),
        pa.field("errors", _STRING_MAP),
    ]
)

_FILTERED_WINDOW_STRUCT = pa.struct(
    [
        # Window bounds in nanoseconds, relative to the clip start (derived from the
        # clip's own per-frame PTS), not the source timeline.
        pa.field("start_ns", pa.int64()),
        pa.field("end_ns", pa.int64()),
        pa.field("rejection_reasons", pa.large_string()),
        pa.field("errors", _STRING_MAP),
    ]
)

_MOTION_SCORE_STRUCT = pa.struct(
    [
        pa.field("global_mean", pa.float64()),
        pa.field("per_patch_min_256", pa.float64()),
    ]
)

CLIP_METADATA_LANCE_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.int32(), nullable=False),
        pa.field("clip_uuid", pa.string(), nullable=False),
        pa.field("video_uuid", pa.string(), nullable=False),
        pa.field("clip_chunk_index", pa.int32(), nullable=False),
        pa.field("source_video", pa.large_string()),
        pa.field("clip_location", pa.large_string()),
        pa.field("start_ns", pa.int64()),
        pa.field("end_ns", pa.int64()),
        pa.field("duration_ns", pa.int64()),
        pa.field("source_width", pa.int32()),
        pa.field("source_height", pa.int32()),
        pa.field("source_framerate", pa.float64()),
        pa.field("clip_width", pa.int32()),
        pa.field("clip_height", pa.int32()),
        pa.field("clip_framerate", pa.float64()),
        pa.field("clip_num_frames", pa.int64()),
        pa.field("clip_video_codec", pa.string()),
        pa.field("clip_num_bytes", pa.int64()),
        pa.field("valid", pa.bool_()),
        pa.field("has_caption", pa.bool_()),
        pa.field("caption_quality_flags_enabled", pa.bool_()),
        pa.field("total_prompt_tokens", pa.int64()),
        pa.field("total_output_tokens", pa.int64()),
        pa.field("num_caption_windows", pa.int64()),
        pa.field("motion_score", _MOTION_SCORE_STRUCT),
        pa.field("aesthetic_score", pa.float64()),
        pa.field("classification_labels", pa.list_(pa.large_string())),
        pa.field("rejection_stage", pa.string()),
        pa.field("post_production_text", pa.bool_()),
        pa.field("sam3_num_instances", pa.int64()),
        pa.field("sam3_num_events", pa.int64()),
        pa.field("errors", pa.list_(pa.large_string())),
        pa.field("windows", pa.list_(_WINDOW_STRUCT)),
        pa.field("filtered_windows", pa.list_(_FILTERED_WINDOW_STRUCT)),
        pa.field("embedding", pa.list_(pa.float32())),
        pa.field("embedding_dim", pa.int64()),
        pa.field("embedding_model_name", pa.string()),
        pa.field("embedding_model_version", pa.string()),
    ]
)


def build_clip_metadata_lance_table(
    metadata_rows: Sequence[Mapping[str, Any]],
    *,
    video_uuid: str,
    clip_chunk_index: int,
) -> pa.Table:
    """Build a schema-stable Arrow table for clip metadata Lance output."""
    lance_rows = [
        clip_metadata_row_to_lance_row(row, video_uuid=video_uuid, clip_chunk_index=clip_chunk_index)
        for row in metadata_rows
    ]
    return pa.Table.from_pylist(lance_rows, schema=CLIP_METADATA_LANCE_SCHEMA)


def clip_metadata_row_to_lance_row(
    row: Mapping[str, Any],
    *,
    video_uuid: str,
    clip_chunk_index: int,
) -> dict[str, Any]:
    """Normalize the legacy JSON metadata row into the Lance metadata contract."""
    start_ns = _int_or_none(row.get("start_ns"))
    end_ns = _int_or_none(row.get("end_ns"))
    embedding = _float_list_or_none(row.get("embedding"))
    return {
        "schema_version": CLIP_METADATA_LANCE_SCHEMA_VERSION,
        "clip_uuid": _required_string(row, "span_uuid"),
        "video_uuid": video_uuid,
        "clip_chunk_index": int(clip_chunk_index),
        "source_video": _string_or_none(row.get("source_video")),
        "clip_location": _string_or_none(row.get("clip_location")),
        "start_ns": start_ns,
        "end_ns": end_ns,
        "duration_ns": _duration_ns(start_ns, end_ns),
        "source_width": _int_or_none(row.get("width_source")),
        "source_height": _int_or_none(row.get("height_source")),
        "source_framerate": _float_or_none(row.get("framerate_source")),
        "clip_width": _int_or_none(row.get("width")),
        "clip_height": _int_or_none(row.get("height")),
        "clip_framerate": _float_or_none(row.get("framerate")),
        "clip_num_frames": _int_or_none(row.get("num_frames")),
        "clip_video_codec": _string_or_none(row.get("video_codec")),
        "clip_num_bytes": _int_or_none(row.get("num_bytes")),
        "valid": _bool_or_none(row.get("valid")),
        "has_caption": _bool_or_none(row.get("has_caption")),
        "caption_quality_flags_enabled": _bool_or_none(row.get("caption_quality_flags_enabled")),
        "total_prompt_tokens": _int_or_none(row.get("total_prompt_tokens")),
        "total_output_tokens": _int_or_none(row.get("total_output_tokens")),
        "num_caption_windows": _int_or_none(row.get("num_caption_windows")),
        "motion_score": _motion_score(row.get("motion_score")),
        "aesthetic_score": _float_or_none(row.get("aesthetic_score")),
        "classification_labels": _string_list_or_none(row.get("qwen_type_classification")),
        "rejection_stage": _string_or_none(row.get("qwen_rejection_stage")),
        "post_production_text": _bool_or_none(row.get("post_production_text")),
        "sam3_num_instances": _int_or_none(row.get("sam3_num_instances")),
        "sam3_num_events": _int_or_none(row.get("sam3_num_events")),
        "errors": _string_list_or_none(row.get("errors")),
        "windows": _windows(row.get("windows")),
        "filtered_windows": _filtered_windows(row.get("filtered_windows")),
        "embedding": embedding,
        "embedding_dim": len(embedding) if embedding is not None else None,
        "embedding_model_name": _string_or_none(row.get("embedding_model_name")),
        "embedding_model_version": _string_or_none(row.get("embedding_model_version")),
    }


def _required_string(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None:
        error_msg = f"Missing required Lance metadata field {key!r}"
        raise ValueError(error_msg)
    return str(value)


def _duration_ns(start_ns: int | None, end_ns: int | None) -> int | None:
    if start_ns is None or end_ns is None:
        return None
    return end_ns - start_ns


def _motion_score(value: object) -> dict[str, float | None] | None:
    if not isinstance(value, Mapping):
        return None
    return {
        "global_mean": _float_or_none(value.get("global_mean")),
        "per_patch_min_256": _float_or_none(value.get("per_patch_min_256")),
    }


def _windows(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (bytes, str)):
        return []
    return [_window(window) for window in value if isinstance(window, Mapping)]


def _window(window: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "start_ns": _int_or_none(window.get("start_ns")),
        "end_ns": _int_or_none(window.get("end_ns")),
        "caption_status": _string_or_none(window.get("caption_status")),
        "caption_failure_reason": _string_or_none(window.get("caption_failure_reason")),
        "captions": _model_text_map(window, "_caption"),
        "enhanced_captions": _model_text_map(window, "_enhanced_caption"),
        "token_counts": _token_counts_map(window),
        "flag_length_outlier": _bool_or_none(window.get("flag_length_outlier")),
        "flag_repetition": _bool_or_none(window.get("flag_repetition")),
        "flag_near_duplicate": _bool_or_none(window.get("flag_near_duplicate")),
        "errors": _string_map(window.get("errors")),
    }


def _filtered_windows(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (bytes, str)):
        return []
    return [_filtered_window(window) for window in value if isinstance(window, Mapping)]


def _filtered_window(window: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "start_ns": _int_or_none(window.get("start_ns")),
        "end_ns": _int_or_none(window.get("end_ns")),
        "rejection_reasons": _string_or_none(window.get("qwen_rejection_reasons")),
        "errors": _string_map(window.get("errors")),
    }


def _model_text_map(window: Mapping[str, Any], suffix: str) -> list[tuple[str, str]]:
    values: dict[str, str] = {}
    for key, value in window.items():
        if not isinstance(key, str) or value is None:
            continue
        if suffix == "_caption" and key.endswith("_enhanced_caption"):
            continue
        if key.endswith(suffix):
            model = key.removesuffix(suffix)
            values[model] = str(value)
    return sorted(values.items())


def _token_counts_map(window: Mapping[str, Any]) -> list[tuple[str, dict[str, int | None]]]:
    token_counts: dict[str, dict[str, int | None]] = {}
    for key, value in window.items():
        if key.endswith("_prompt_tokens"):
            model = key.removesuffix("_prompt_tokens")
            token_counts.setdefault(model, {})["prompt_tokens"] = _int_or_none(value)
        elif key.endswith("_output_tokens"):
            model = key.removesuffix("_output_tokens")
            token_counts.setdefault(model, {})["output_tokens"] = _int_or_none(value)

    return [
        (
            model,
            {
                "prompt_tokens": counts.get("prompt_tokens"),
                "output_tokens": counts.get("output_tokens"),
            },
        )
        for model, counts in sorted(token_counts.items())
    ]


def _string_map(value: object) -> list[tuple[str, str]]:
    if not isinstance(value, Mapping):
        return []
    return sorted((str(key), str(item)) for key, item in value.items() if item is not None)


def _string_list_or_none(value: object) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (bytes, str)):
        return [str(value)]
    return [str(item) for item in value if item is not None]


def _float_list_or_none(value: object) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (bytes, str)):
        return [_float_required(value)]
    return [_float_required(item) for item in value]


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray, SupportsInt, SupportsIndex)):
        return int(value)
    return int(str(value))


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray, SupportsFloat, SupportsIndex)):
        return float(value)
    return float(str(value))


def _float_required(value: object) -> float:
    converted = _float_or_none(value)
    if converted is None:
        error_msg = "Float list values must not be None"
        raise ValueError(error_msg)
    return converted


def _bool_or_none(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
        error_msg = f"Cannot convert string {value!r} to bool"
        raise ValueError(error_msg)
    return bool(value)
