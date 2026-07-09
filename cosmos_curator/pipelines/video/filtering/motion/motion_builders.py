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
"""Stage builder for motion filtering.

Motion vectors are always exported upstream by the CameraSensor clip-extraction stage
(``ClipFrameExtractionStage`` + ``CameraSensorMotionVectorConfig``), so this builder only
constructs the scoring/filter stage that consumes ``clip.decoded_motion_data``.
"""

import attrs

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curator.pipelines.video.filtering.motion.motion_filter_stages import MotionFilterStage


@attrs.define(frozen=True)
class MotionFilterConfig:
    """Configuration for motion filtering."""

    score_only: bool = False
    global_mean_threshold: float = 0.00098
    per_patch_min_256_threshold: float = 0.000001
    score_gpus_per_worker: float = 0.5
    score_batch_size: int = 64
    verbose: bool = False
    perf_profile: bool = False


def build_motion_filter_stages(config: MotionFilterConfig) -> list[CuratorStage | CuratorStageSpec]:
    """Construct and return the motion filter stage.

    Motion vectors are supplied by the upstream CameraSensor clip-extraction stage, so only the
    scoring/filter stage is built here.
    """
    filter_stage = MotionFilterStage(
        score_only=config.score_only,
        global_mean_threshold=config.global_mean_threshold,
        per_patch_min_256_threshold=config.per_patch_min_256_threshold,
        num_gpus_per_worker=config.score_gpus_per_worker,
        batch_size=config.score_batch_size,
        verbose=config.verbose,
        log_stats=config.perf_profile,
    )
    return [filter_stage]
