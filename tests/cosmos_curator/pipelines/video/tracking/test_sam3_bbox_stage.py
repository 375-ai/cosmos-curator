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

"""CPU tests for ``track_objects`` chunk/id bookkeeping.

SAM3 inference itself is GPU-only, so the model is faked: the test drives the
real chunking + clip-global id remap with a stand-in processor/model that
re-emits the same per-chunk raw object_ids in every chunk.
"""

import types

import numpy as np
import pytest
import torch

from cosmos_curator.pipelines.video.tracking import sam3_bbox_stage
from cosmos_curator.pipelines.video.tracking.sam3_bbox_stage import track_objects


class _FakeProcessor:
    """Stand-in for ``SAM3Model.processor`` re-emitting raw ids 0 and 1."""

    def init_video_session(self, *, video, inference_device, video_storage_device, dtype):  # noqa: ANN001, ANN202, ARG002
        return types.SimpleNamespace(num_frames=len(video))

    def add_text_prompt(self, session, prompt):  # noqa: ANN001, ANN202, ARG002
        return None

    def postprocess_outputs(self, session, model_outputs):  # noqa: ANN001, ANN202, ARG002
        # Two objects with the same raw per-chunk ids on every frame, so every
        # chunk reuses {0, 1} -- exactly the cross-chunk collision case.
        return {
            "object_ids": torch.tensor([0, 1]),
            "masks": [torch.ones((2, 2), dtype=torch.bool), torch.ones((2, 2), dtype=torch.bool)],
            "boxes": [torch.tensor([0.0, 0.0, 1.0, 1.0]), torch.tensor([1.0, 1.0, 2.0, 2.0])],
            "prompt_to_obj_ids": {"obj": [0, 1]},
        }


class _FakeModel:
    """Stand-in for ``SAM3Model.model`` yielding one output per chunk frame."""

    def propagate_in_video_iterator(self, *, inference_session, show_progress_bar):  # noqa: ANN001, ANN202, ARG002
        for i in range(inference_session.num_frames):
            yield types.SimpleNamespace(frame_idx=i)


class _FakeSam3:
    def __init__(self) -> None:
        self.processor = _FakeProcessor()
        self.model = _FakeModel()


def _noop(*_args: object, **_kwargs: object) -> None:
    return None


@pytest.fixture(autouse=True)
def _stub_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neuter the GPU allocator calls so ``track_objects`` runs on CPU."""
    monkeypatch.setattr(sam3_bbox_stage.torch.cuda, "empty_cache", _noop)
    monkeypatch.setattr(sam3_bbox_stage.torch.cuda, "reset_peak_memory_stats", _noop)


def test_object_ids_are_clip_global_across_chunks() -> None:
    """Reused per-chunk raw ids must map to disjoint clip-global ids."""
    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(4)]
    timestamps = [0.0, 0.5, 1.0, 1.5]

    # session_reset_s * target_fps = 2 -> two chunks of two frames each.
    per_frame_dets, instances = track_objects(
        _FakeSam3(),
        frames,
        timestamps,
        ["obj"],
        session_reset_s=1.0,
        target_fps=2.0,
    )

    chunk0_ids = {det.object_id for f in (0, 1) for det in per_frame_dets[f]}
    chunk1_ids = {det.object_id for f in (2, 3) for det in per_frame_dets[f]}

    # Within a chunk the same raw id is one stable track across frames.
    assert {det.object_id for det in per_frame_dets[0]} == {det.object_id for det in per_frame_dets[1]}
    # Across chunks the reused raw ids (0, 1) must map to disjoint global ids.
    assert chunk0_ids.isdisjoint(chunk1_ids)
    assert chunk0_ids == {0, 1}
    assert chunk1_ids == {2, 3}

    # One instance per global id, all unique.
    instance_ids = [inst["object_id"] for inst in instances]
    assert sorted(instance_ids) == [0, 1, 2, 3]
    assert len(instance_ids) == len(set(instance_ids))
