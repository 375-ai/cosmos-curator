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

"""Stage builders + config for Cosmos3 video style transfer.

Every knob lives on the frozen ``StyleTransferConfig``; the CLI in
``splitting_pipeline.py`` only surfaces the fidelity-relevant subset as flags
(matching how ``SuperResolutionConfig`` is wired). Field defaults follow the
Cosmos3 transfer cookbook's example specs.
"""

import attrs

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec

# The two control types the first implementation can compute on the host with no
# extra model weights. depth/seg need DepthAnything/SAM2 and are deferred.
STYLE_TRANSFER_CONTROLS: tuple[str, ...] = ("edge", "blur")
STYLE_TRANSFER_CONTROL_PRESETS: tuple[str, ...] = ("low", "medium", "high")
# Cosmos3 output resolution buckets; the transfer worker rejects anything else.
STYLE_TRANSFER_RESOLUTIONS: tuple[str, ...] = ("256", "480", "704", "720")


@attrs.define(frozen=True)
class StyleTransferConfig:
    """Configuration for the Cosmos3 video style-transfer stage.

    Fields map onto the vLLM-Omni Cosmos3 transfer request (see
    ``Cosmos3OmniTransferModel.generate``). ``fps`` defaults to the source clip's
    own frame rate when left unset.
    """

    # Model variant: cosmos3_nano (1 GPU) or cosmos3_super (4+ GPUs).
    model_variant: str = "cosmos3_nano"

    # Prompt + negative prompt (the caption pair driving the restyle).
    prompt: str = ""
    negative_prompt: str = ""

    # Control signal and its fidelity knobs.
    control: str = "edge"
    control_guidance: float = 1.5
    edge_preset: str = "medium"
    blur_preset: str = "medium"
    # When True, compute the edge/blur control on the host (pre-computed
    # control_path); when False, hand the decoded source frames to vLLM-Omni and
    # let it derive the control on-the-fly.
    precompute_control: bool = False

    # Cosmos3 text/video safety guardrails. Default False: vLLM-Omni loads them at
    # engine build and requires the ``cosmos-guardrail`` package, which is not in any
    # env, so enabling them on the current build fails. Opt in once the package is
    # available (NVIDIA Open Model License expects guardrails on).
    guardrails: bool = False

    # Prompt CFG + seed.
    guidance: float = 3.0
    seed: int = 2026

    # Output geometry. resolution is a Cosmos3 bucket key (256/480/704/720); the
    # transfer path scales the SOURCE clip's aspect ratio to that bucket, so output
    # width/height are derived from the source (no explicit aspect ratio knob).
    # fps defaults to the source clip fps.
    resolution: str = "720"
    fps: int | None = None

    # Internal chunking (the transfer path stitches chunks itself). 93 mirrors
    # vLLM-Omni's tuned Cosmos3 per-chunk default (the NATTEN window size).
    num_video_frames_per_chunk: int = 93
    num_conditional_frames: int = 1

    # GPUs for the in-process transfer engine (Super is clamped up to 4 by the model).
    num_gpus: int = 1

    # Workers per node. 0 = auto: return a bare stage and let Xenna autoscale by GPU
    # (so nano packs one worker per GPU). >0 pins an explicit count.
    num_workers_per_node: int = 0

    # Pipeline standard flags.
    verbose: bool = False
    perf_profile: bool = False


def build_style_transfer_stages(config: StyleTransferConfig) -> list[CuratorStage | CuratorStageSpec]:
    """Construct and return the style-transfer stage.

    Args:
        config: Style-transfer configuration.

    Returns:
        A list containing the configured style-transfer stage.

    """
    from cosmos_curator.pipelines.video.style_transfer.style_transfer_stage import (  # noqa: PLC0415
        StyleTransferStage,
    )

    stage = StyleTransferStage(config=config)
    if config.num_workers_per_node > 0:
        return [CuratorStageSpec(stage, num_workers_per_node=config.num_workers_per_node)]
    return [stage]
