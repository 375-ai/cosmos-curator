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
"""Arrow schema for the eval phase's per-clip verdict table.

Evaluation emits issues (``ISSUE_SCHEMA``) *and* this per-clip verdict, so the
question "which clips are good?" is a direct query rather than an absence of
issues to join against. ``passed`` means no issue fired for the clip under the
applied policy; the carried ``min_caption_similarity`` / ``max_score_abs_diff``
let consumers rank clips by agreement without rejoining the measurement tables.
"""

import pyarrow as pa  # type: ignore[import-untyped]

CLIP_VERDICT_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("clip_uuid", pa.string(), nullable=False),
        pa.field("video_uuid", pa.string()),
        pa.field("present_a", pa.bool_(), nullable=False),
        pa.field("present_b", pa.bool_(), nullable=False),
        pa.field("num_issues", pa.int64(), nullable=False),
        pa.field("passed", pa.bool_(), nullable=False),
        # Carried from the clip measurement row for ranking good / divergent clips.
        pa.field("min_caption_similarity", pa.float64()),
        pa.field("max_score_abs_diff", pa.float64()),
    ]
)
