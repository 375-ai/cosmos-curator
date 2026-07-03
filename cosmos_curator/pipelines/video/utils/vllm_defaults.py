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

"""Video-pipeline merge policy for model-authored vLLM defaults."""

from typing import Any

import attrs

from cosmos_curator.models.vllm_model_defaults import (
    get_vllm_default_sampling_config,
    get_vllm_default_sampling_fps,
)
from cosmos_curator.pipelines.video.utils.data_model import VllmSamplingConfig, WindowConfig

_GENERIC_SAMPLING_DEFAULTS = VllmSamplingConfig()
_GENERIC_CAPTIONING_SAMPLING_FPS = WindowConfig().sampling_fps


def resolve_vllm_sampling_config(model_variant: str, requested: VllmSamplingConfig) -> VllmSamplingConfig:
    """Apply model-owned generation defaults only where the request kept generic defaults."""
    model_defaults = get_vllm_default_sampling_config(model_variant)
    if model_defaults is None:
        return requested

    resolved_values: dict[str, Any] = {}
    for field in attrs.fields(VllmSamplingConfig):
        field_name = field.name
        requested_value = getattr(requested, field_name)
        if requested_value == getattr(_GENERIC_SAMPLING_DEFAULTS, field_name):
            resolved_values[field_name] = getattr(model_defaults, field_name)

    return attrs.evolve(requested, **resolved_values)


def resolve_vllm_sampling_fps(model_variant: str, requested_fps: float) -> float:
    """Apply a model-owned captioning FPS only when the request kept the generic default."""
    model_default_fps = get_vllm_default_sampling_fps(model_variant)
    if model_default_fps is None:
        return requested_fps
    if requested_fps == _GENERIC_CAPTIONING_SAMPLING_FPS:
        return model_default_fps
    return requested_fps
