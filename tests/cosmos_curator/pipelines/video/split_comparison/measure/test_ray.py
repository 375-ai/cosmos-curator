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
"""Tests for the Ray measure driver's GPU-count resolution (override path is Ray-free)."""

import pyarrow as pa
import pytest

from cosmos_curator.pipelines.video.split_comparison.config import SplitComparisonConfig
from cosmos_curator.pipelines.video.split_comparison.measure import ray as raymod
from cosmos_curator.pipelines.video.split_comparison.measure.ray import resolve_num_gpus, run
from cosmos_curator.pipelines.video.split_comparison.measure.schema import (
    CLIP_MEASUREMENT_SCHEMA,
    WINDOW_MEASUREMENT_SCHEMA,
)


@pytest.mark.parametrize("override", [0, -1])
def test_resolve_num_gpus_rejects_non_positive_override(override: int) -> None:
    """A zero/negative --num-gpus is rejected up front, not handed to the actor pool as a bad size."""
    with pytest.raises(ValueError, match="resolve_num_gpus"):
        resolve_num_gpus(override)


def test_resolve_num_gpus_returns_positive_override() -> None:
    """A positive override is returned verbatim without consulting Ray's detected count."""
    assert resolve_num_gpus(4) == 4


def _stub_run_collaborators(monkeypatch: pytest.MonkeyPatch, *, gpu_resolver: object) -> None:
    """Patch run()'s heavy Ray/Lance collaborators so its control flow can be tested in-process."""
    empty_source = pa.table({"clip_uuid": pa.array([], pa.string())})
    monkeypatch.setattr(raymod.ray, "is_initialized", lambda: True)  # skip real ray.init
    monkeypatch.setattr(raymod, "_configure_progress", lambda **_: None)
    monkeypatch.setattr(raymod, "load_both", lambda *_a, **_k: (empty_source, empty_source))
    monkeypatch.setattr(
        raymod, "clip_measurements_columnar", lambda *_a, **_k: pa.Table.from_pylist([], schema=CLIP_MEASUREMENT_SCHEMA)
    )
    monkeypatch.setattr(raymod, "merge_caption_rollups", lambda clip, _window: clip)
    monkeypatch.setattr(raymod, "measure_stats", lambda *_a, **_k: {})
    monkeypatch.setattr(
        raymod, "_measure_windows_ray", lambda *_a, **_k: pa.Table.from_pylist([], schema=WINDOW_MEASUREMENT_SCHEMA)
    )
    monkeypatch.setattr(raymod, "resolve_num_gpus", gpu_resolver)


def test_run_clip_only_does_not_require_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """A --no-captions measure must run on a GPU-less host: GPU resolution is never consulted.

    resolve_num_gpus(None) raises on a host with no GPU; a clip-only run does zero GPU work,
    so run() must not call it. We simulate the GPU-less host by making it raise.
    """
    calls: list[object] = []

    def no_gpu(override: int | None) -> int:
        calls.append(override)
        msg = "no GPUs detected by Ray; the measure requires at least one GPU"
        raise ValueError(msg)

    _stub_run_collaborators(monkeypatch, gpu_resolver=no_gpu)
    config = SplitComparisonConfig(output_a="/a", output_b="/b", compare_captions=False)
    result = run(config)  # must not raise
    assert calls == []
    assert result.window_table.num_rows == 0


def test_run_with_captions_resolves_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """With captions in scope the fan-out still resolves the GPU count for the actor pool."""
    calls: list[object] = []

    def spy(override: int | None) -> int:
        calls.append(override)
        return 2

    _stub_run_collaborators(monkeypatch, gpu_resolver=spy)
    config = SplitComparisonConfig(output_a="/a", output_b="/b", compare_captions=True)
    run(config)
    assert calls == [None]
