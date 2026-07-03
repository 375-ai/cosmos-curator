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

"""Tests for video-pipeline vLLM default resolution."""

import pytest

from cosmos_curator.models.vllm_model_defaults import get_vllm_default_sampling_config
from cosmos_curator.pipelines.video.utils.data_model import VllmSamplingConfig, WindowConfig
from cosmos_curator.pipelines.video.utils.vllm_defaults import (
    resolve_vllm_sampling_config,
    resolve_vllm_sampling_fps,
)


def test_unknown_model_keeps_requested_sampling_config() -> None:
    """Models without authored defaults keep the requested sampling config unchanged."""
    requested = VllmSamplingConfig(temperature=0.2, top_p=0.95)

    assert resolve_vllm_sampling_config("qwen", requested) is requested


@pytest.mark.parametrize("model_variant", ["cosmos3_nano", "cosmos3_super"])
def test_cosmos3_generic_sampling_defaults_resolve_to_model_defaults(model_variant: str) -> None:
    """Generic split-pipeline defaults resolve to Cosmos3 non-thinking defaults."""
    resolved = resolve_vllm_sampling_config(model_variant, VllmSamplingConfig())

    assert resolved.temperature == 0.7
    assert resolved.top_p == 0.8
    assert resolved.top_k == 20
    assert resolved.repetition_penalty == 1.0
    assert resolved.min_tokens == 0
    assert resolved.presence_penalty == 1.5
    assert resolved.frequency_penalty == 0.0
    assert resolved.min_p == 0.0
    assert resolved.max_tokens == VllmSamplingConfig().max_tokens


def test_model_sampling_defaults_return_fresh_config_instances() -> None:
    """The model-default registry should not leak its mutable stored config object."""
    first = get_vllm_default_sampling_config("cosmos3_nano")
    second = get_vllm_default_sampling_config("cosmos3_nano")

    assert first is not None
    assert second is not None
    assert first is not second

    first.temperature = 0.2

    third = get_vllm_default_sampling_config("cosmos3_nano")
    assert third is not None
    assert second.temperature == 0.7
    assert second.presence_penalty == 1.5
    assert third.temperature == 0.7


def test_cosmos3_non_default_sampling_values_override_model_defaults() -> None:
    """Only fields left at generic defaults are replaced by model-authored defaults."""
    requested = VllmSamplingConfig(
        temperature=0.2,
        top_p=0.95,
        min_tokens=7,
        max_tokens=None,
    )

    resolved = resolve_vllm_sampling_config("cosmos3_nano", requested)

    assert resolved.temperature == 0.2
    assert resolved.top_p == 0.95
    assert resolved.top_k == 20
    assert resolved.repetition_penalty == 1.0
    assert resolved.min_tokens == 7
    assert resolved.max_tokens is None


def test_unknown_model_keeps_requested_sampling_fps() -> None:
    """Models without authored FPS defaults keep the requested FPS unchanged."""
    assert resolve_vllm_sampling_fps("qwen", WindowConfig().sampling_fps) == WindowConfig().sampling_fps


@pytest.mark.parametrize("model_variant", ["cosmos3_nano", "cosmos3_super"])
def test_cosmos3_generic_sampling_fps_resolves_to_model_default(model_variant: str) -> None:
    """Generic captioning FPS resolves to the Cosmos3 recommendation."""
    assert resolve_vllm_sampling_fps(model_variant, WindowConfig().sampling_fps) == 4.0


def test_cosmos3_non_default_sampling_fps_overrides_model_default() -> None:
    """Explicit captioning FPS survives model default resolution."""
    assert resolve_vllm_sampling_fps("cosmos3_nano", 3.0) == 3.0
