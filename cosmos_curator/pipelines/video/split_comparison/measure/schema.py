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
"""Arrow schemas for the measure phase's two durable measurement tables.

The measure phase records *facts* -- raw per-side values plus the computed
difference for every clip and every caption window -- with no thresholds
applied. Evaluation is a separate, cheap pass that reads these tables and emits
issues, so the same measurements can be re-evaluated under different policy
without recomputing diffs or re-running the GPU embedding.

Two grains, two tables:

* ``CLIP_MEASUREMENT_SCHEMA`` -- one row per clip (union of both outputs).
  Carries raw ``*_a`` / ``*_b`` values *and* the ``*_abs_diff`` / ``*_equal``
  derivations. Raw values are kept so eval can apply relative tolerance (not
  just absolute) without re-measuring.
* ``WINDOW_MEASUREMENT_SCHEMA`` -- one row per ``(clip, window, model, kind)``
  caption comparison, carrying the cosine ``similarity`` plus presence and
  text-length context. Full caption text is intentionally *not* stored -- the
  source Lance datasets still hold it, joinable by ``clip_uuid`` + bounds.
"""

import pyarrow as pa  # type: ignore[import-untyped]

MEASUREMENT_SCHEMA_VERSION = 1

CLIP_MEASUREMENT_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("clip_uuid", pa.string(), nullable=False),
        pa.field("video_uuid", pa.string()),
        pa.field("present_a", pa.bool_(), nullable=False),
        pa.field("present_b", pa.bool_(), nullable=False),
        # aesthetic_score: raw per-side + abs diff (diff null unless present on both).
        pa.field("aesthetic_a", pa.float64()),
        pa.field("aesthetic_b", pa.float64()),
        pa.field("aesthetic_abs_diff", pa.float64()),
        # motion_score.global_mean
        pa.field("motion_global_mean_a", pa.float64()),
        pa.field("motion_global_mean_b", pa.float64()),
        pa.field("motion_global_mean_abs_diff", pa.float64()),
        # motion_score.per_patch_min_256
        pa.field("motion_per_patch_min_256_a", pa.float64()),
        pa.field("motion_per_patch_min_256_b", pa.float64()),
        pa.field("motion_per_patch_min_256_abs_diff", pa.float64()),
        # equality-compared scalar fields: raw per-side + equal flag.
        pa.field("valid_a", pa.bool_()),
        pa.field("valid_b", pa.bool_()),
        pa.field("valid_equal", pa.bool_()),
        pa.field("has_caption_a", pa.bool_()),
        pa.field("has_caption_b", pa.bool_()),
        pa.field("has_caption_equal", pa.bool_()),
        pa.field("rejection_stage_a", pa.string()),
        pa.field("rejection_stage_b", pa.string()),
        pa.field("rejection_stage_equal", pa.bool_()),
        # window accounting + caption-similarity rollups for ranking / "good clip" queries.
        pa.field("num_windows_a", pa.int64()),
        pa.field("num_windows_b", pa.int64()),
        pa.field("min_caption_similarity", pa.float64()),
        pa.field("mean_caption_similarity", pa.float64()),
    ]
)

WINDOW_MEASUREMENT_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("clip_uuid", pa.string(), nullable=False),
        pa.field("video_uuid", pa.string()),
        # Window bounds in nanoseconds, relative to the clip start (matches the source schema).
        pa.field("start_ns", pa.int64()),
        pa.field("end_ns", pa.int64()),
        pa.field("model", pa.string(), nullable=False),
        # "caption" (base) vs "enhanced" caption map.
        pa.field("kind", pa.string(), nullable=False),
        pa.field("present_a", pa.bool_(), nullable=False),
        pa.field("present_b", pa.bool_(), nullable=False),
        # Text equal on both sides -> similarity 1.0 with no embedding needed.
        pa.field("identical", pa.bool_()),
        # Cosine similarity; 1.0 when identical; null when not computable (one-sided).
        pa.field("similarity", pa.float64()),
        pa.field("len_a", pa.int64()),
        pa.field("len_b", pa.int64()),
    ]
)
