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

"""Stage builders for monocular 3D scene reconstruction."""

import attrs

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curator.models.depth_anything_v2 import DepthVariant
from cosmos_curator.pipelines.video.scene3d.calibration import CameraOverrides, GroundFitParams
from cosmos_curator.pipelines.video.scene3d.lifting import CloudParams
from cosmos_curator.pipelines.video.scene3d.object_lift import ObjectLiftParams
from cosmos_curator.pipelines.video.scene3d.scene3d_stage import (
    BackgroundMode,
    ObjectDepthMode,
    Scene3DStage,
)


@attrs.define(frozen=True, kw_only=True)
class Scene3DConfig:
    """Configuration for the 3D scene reconstruction block.

    The four tuning groups are held as the same objects the stage consumes, rather
    than flattened and re-nested by the builder — matching ``SAM3TrackingConfig``,
    which nests ``SAM3QualityConfig`` the same way. That keeps one home per default
    instead of restating each one here and in the params class.
    """

    depth_variant: DepthVariant = "outdoor"
    depth_long_side: int = 700
    target_fps: float = 5.0
    max_clip_duration_s: float = 120.0

    camera: CameraOverrides = attrs.Factory(CameraOverrides)
    ground: GroundFitParams = attrs.Factory(GroundFitParams)
    cloud: CloudParams = attrs.Factory(CloudParams)
    objects: ObjectLiftParams = attrs.Factory(ObjectLiftParams)

    background_mode: BackgroundMode = "depth"
    object_depth_mode: ObjectDepthMode = "background"

    gpus_per_worker: float = 0.5
    num_workers_per_node: int = 0
    verbose: bool = False


def build_scene3d_stages(config: Scene3DConfig) -> list[CuratorStage | CuratorStageSpec]:
    """Build the 3D scene reconstruction stage block.

    Currently a single fused stage (``Scene3DStage``) that runs depth, calibrates
    the camera, builds the background cloud and lifts objects in one pass over each
    clip. Returned as a list to match the ``build_*_stages`` convention.
    """
    stage = Scene3DStage(
        depth_variant=config.depth_variant,
        depth_long_side=config.depth_long_side,
        target_fps=config.target_fps,
        max_clip_duration_s=config.max_clip_duration_s,
        camera_overrides=config.camera,
        ground_params=config.ground,
        cloud_params=config.cloud,
        object_params=config.objects,
        background_mode=config.background_mode,
        object_depth_mode=config.object_depth_mode,
        gpus_per_worker=config.gpus_per_worker,
        verbose=config.verbose,
    )
    if config.num_workers_per_node > 0:
        return [CuratorStageSpec(stage, num_workers_per_node=config.num_workers_per_node)]
    return [stage]
