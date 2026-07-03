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

"""Lightweight model-authored defaults for vLLM variants.

This module intentionally avoids importing plugin classes because those modules
load vLLM. CLI/config code can still resolve model-owned defaults in CPU-only
test and submission environments.
"""

import attrs

from cosmos_curator.pipelines.video.utils.data_model import VllmSamplingConfig


@attrs.define(frozen=True)
class VllmModelDefaults:
    """Model-owned generation and video sampling defaults for a vLLM variant."""

    sampling_config: VllmSamplingConfig | None = None
    sampling_fps: float | None = None


_COSMOS3_OMNI_DEFAULTS = VllmModelDefaults(
    sampling_config=VllmSamplingConfig(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        min_tokens=0,
    ),
    sampling_fps=4.0,
)

_VLLM_MODEL_DEFAULTS: dict[str, VllmModelDefaults] = {
    "cosmos3_nano": _COSMOS3_OMNI_DEFAULTS,
    "cosmos3_super": _COSMOS3_OMNI_DEFAULTS,
}


def get_vllm_default_sampling_config(model_variant: str) -> VllmSamplingConfig | None:
    """Return model-authored generation defaults for a vLLM variant, if known."""
    defaults = _VLLM_MODEL_DEFAULTS.get(model_variant.lower())
    if defaults is None or defaults.sampling_config is None:
        return None
    return attrs.evolve(defaults.sampling_config)


def get_vllm_default_sampling_fps(model_variant: str) -> float | None:
    """Return model-authored captioning input FPS for a vLLM variant, if known."""
    defaults = _VLLM_MODEL_DEFAULTS.get(model_variant.lower())
    if defaults is None:
        return None
    return defaults.sampling_fps
