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
"""Tests for the eval phase: eval output must depend on eval config, not measure history."""

import pyarrow as pa

from cosmos_curator.pipelines.video.split_comparison.config import SplitComparisonConfig
from cosmos_curator.pipelines.video.split_comparison.eval import evaluate
from cosmos_curator.pipelines.video.split_comparison.measure.core import Measurements
from cosmos_curator.pipelines.video.split_comparison.measure.schema import (
    CLIP_MEASUREMENT_SCHEMA,
    WINDOW_MEASUREMENT_SCHEMA,
)


def _measurements_with_caption_rollup() -> Measurements:
    """Build a captioned root: one two-sided clip carrying a stored min_caption_similarity rollup."""
    clip_table = pa.Table.from_pylist(
        [{"clip_uuid": "c1", "video_uuid": "v", "present_a": True, "present_b": True, "min_caption_similarity": 0.5}],
        schema=CLIP_MEASUREMENT_SCHEMA,
    )
    return Measurements(
        clip_table=clip_table,
        window_table=pa.Table.from_pylist([], schema=WINDOW_MEASUREMENT_SCHEMA),
        stats={},
    )


def _verdict(result: pa.Table) -> dict:
    return result.verdicts.to_pylist()[0]


def test_no_captions_nulls_stale_caption_rollup_in_verdict() -> None:
    """Re-eval with --no-captions must not leak the captioned root's min_caption_similarity.

    The rollup is a stored clip-table column from a captioned measure; gating window issues
    alone leaves it echoed into the verdict, so a clip ranked by it would still see caption data
    the eval was told to ignore -- and differ from a fresh --no-captions run (which has null).
    """
    result = evaluate(
        _measurements_with_caption_rollup(),
        config=SplitComparisonConfig(output_a="/a", output_b="/b", compare_captions=False),
    )
    assert _verdict(result)["min_caption_similarity"] is None


def test_captions_preserve_caption_rollup_in_verdict() -> None:
    """With captions in scope the stored rollup is carried through to the verdict unchanged."""
    result = evaluate(
        _measurements_with_caption_rollup(),
        config=SplitComparisonConfig(output_a="/a", output_b="/b", compare_captions=True),
    )
    assert _verdict(result)["min_caption_similarity"] == 0.5
