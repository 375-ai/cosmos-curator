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

"""Tests for SAM3 output JSON envelope shapes."""

from typing import Any

from cosmos_curator.pipelines.video.tracking.serialization import (
    sam3_events_envelope,
    sam3_instances_envelope,
    sam3_objects_envelope,
)


def test_objects_envelope_is_frame_list_with_timestamps() -> None:
    """``objects.json`` is a ``frames`` list, each carrying a real timestamp."""
    frames: list[dict[str, Any]] = [
        {
            "frame_idx": 0,
            "timestamp_s": 0.0,
            "detections": [{"prompt": "a car", "object_id": 1, "box_xyxy": [0, 0, 1, 1], "contours_xy": [[0, 0]]}],
        },
        {
            "frame_idx": 16,
            "timestamp_s": 0.533,
            "detections": [],
        },
    ]

    envelope = sam3_objects_envelope(frames)

    assert envelope == {"frames": frames}
    # Timestamps survive the envelope unchanged (downstream stages read them).
    assert envelope["frames"][1]["timestamp_s"] == 0.533
    assert envelope["frames"][1]["frame_idx"] == 16


def test_objects_envelope_empty() -> None:
    """An empty clip still serializes to a ``frames`` key (empty list)."""
    assert sam3_objects_envelope([]) == {"frames": []}


def test_instances_envelope() -> None:
    """``instances.json`` wraps the per-object summary list under ``instances``."""
    instances = [{"object_id": 1, "prompt": "a car", "start_time_s": 0.0, "end_time_s": 2.5, "num_frames": 5}]
    assert sam3_instances_envelope(instances) == {"instances": instances}


def test_events_envelope() -> None:
    """``events.json`` wraps the VLM event list under ``events``."""
    events = [{"start_time": 0.0, "end_time": 1.0, "description": "a car turns"}]
    assert sam3_events_envelope(events) == {"events": events}
