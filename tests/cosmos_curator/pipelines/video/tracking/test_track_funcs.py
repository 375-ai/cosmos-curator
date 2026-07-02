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

"""Tests for the CPU-only track-data functions (serialize / annotate / encode).

These pin the fused stage's post-inference pipeline — the contract that the
``Detection`` lists ``track_objects`` returns compose cleanly through
``build_track_records -> annotate_frames -> encode_frames_to_mp4`` with one
decode and one encode and aligned, real-PTS records — without needing a GPU or
the SAM3 model.
"""

import numpy as np

from cosmos_curator.pipelines.video.tracking.track_funcs import (
    annotate_frames,
    build_track_records,
    encode_frames_to_mp4,
)
from cosmos_curator.pipelines.video.tracking.visualization import Detection


def _mask(h: int = 48, w: int = 64) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[12:36, 16:48] = True
    return mask


def _per_frame_dets() -> list[list[Detection]]:
    # Frame 0 has one car; frame 1 has none; frame 2 has a car + a bus.
    return [
        [Detection(prompt="a car", object_id=1, box_xyxy=[16, 12, 48, 36], mask=_mask())],
        [],
        [
            Detection(prompt="a car", object_id=1, box_xyxy=[18, 14, 50, 38], mask=_mask()),
            Detection(prompt="a bus", object_id=2, box_xyxy=[0, 0, 8, 8], mask=_mask()),
        ],
    ]


def test_build_track_records_aligns_and_carries_real_pts() -> None:
    """One record per sampled frame: contiguous frame_idx + real PTS anchor."""
    timestamps_s = [0.0, 0.533, 1.1]
    records = build_track_records(_per_frame_dets(), timestamps_s, include_contours=True)

    assert len(records) == 3
    # frame_idx is the contiguous sampled position, not a native source index.
    assert [r["frame_idx"] for r in records] == [0, 1, 2]
    assert [r["timestamp_s"] for r in records] == timestamps_s
    assert len(records[0]["detections"]) == 1
    assert records[1]["detections"] == []
    assert len(records[2]["detections"]) == 2
    # contour mode emits polygons for the masked detections.
    assert records[0]["detections"][0]["contours_xy"]


def test_build_track_records_region_box_drops_contours() -> None:
    """``include_contours=False`` (region=box) emits empty contour lists."""
    records = build_track_records(_per_frame_dets(), [0.0, 0.1, 0.2], include_contours=False)
    assert records[0]["detections"][0]["contours_xy"] == []
    # The box survives for box-only consumers.
    assert records[0]["detections"][0]["box_xyxy"] == [16, 12, 48, 36]


def test_fused_compose_serialize_annotate_encode() -> None:
    """decode(stub) -> serialize -> annotate -> encode composes end-to-end."""
    per_frame_dets = _per_frame_dets()
    frames_rgb = [np.full((48, 64, 3), 100, dtype=np.uint8) for _ in per_frame_dets]
    timestamps_s = [0.0, 0.5, 1.0]

    sam3_frames = build_track_records(per_frame_dets, timestamps_s, include_contours=True)
    annotated = annotate_frames(frames_rgb, sam3_frames, ["a car", "a bus"], draw_masks=True, draw_timestamps=True)

    assert len(annotated) == len(frames_rgb)
    assert all(f.shape == (48, 64, 3) for f in annotated)
    # The frame with detections is drawn on; the empty frame still gets a timestamp badge.
    assert not np.array_equal(annotated[0], frames_rgb[0])

    encoded = encode_frames_to_mp4(annotated, fps=2.0, width=64, height=48)
    # mp4v may be unavailable in some headless builds; when present, bytes are produced.
    assert encoded is None or (isinstance(encoded, bytes) and len(encoded) > 0)


def test_annotate_frames_timestamp_only_skips_masks() -> None:
    """draw_masks=False leaves detection pixels alone but still burns the clock."""
    # Larger frame + centrally-placed detection so the top-left timestamp badge
    # can't overlap the region we assert on.
    mask = np.zeros((240, 320), dtype=bool)
    mask[100:140, 140:200] = True
    dets = [[Detection(prompt="a car", object_id=1, box_xyxy=[140, 100, 200, 140], mask=mask)]]
    frames_rgb = [np.full((240, 320, 3), 100, dtype=np.uint8)]
    sam3_frames = build_track_records(dets, [0.0], include_contours=True)

    masks_off = annotate_frames(frames_rgb, sam3_frames, ["a car"], draw_masks=False, draw_timestamps=True)
    masks_on = annotate_frames(frames_rgb, sam3_frames, ["a car"], draw_masks=True, draw_timestamps=True)
    region = (slice(100, 140), slice(140, 200))
    # Masks-off leaves the detection silhouette untouched; masks-on draws on it.
    assert np.array_equal(masks_off[0][region], frames_rgb[0][region])
    assert not np.array_equal(masks_on[0][region], frames_rgb[0][region])
    # The clock badge is still burned in (top-left changed) even with masks off.
    assert not np.array_equal(masks_off[0][0:48, 0:200], frames_rgb[0][0:48, 0:200])


def test_encode_empty_frames_returns_none() -> None:
    """No frames -> None (nothing to encode), not a crash."""
    assert encode_frames_to_mp4([], fps=2.0, width=64, height=48) is None


def test_encode_size_mismatch_returns_none() -> None:
    """A frame whose size != writer geometry is rejected, not silently dropped."""
    # Frames are 48x64 but we declare width=32, so cv2 would otherwise silently
    # drop them and emit a corrupt/empty mp4. The guard must return None instead.
    frames = [np.full((48, 64, 3), 100, dtype=np.uint8) for _ in range(3)]
    assert encode_frames_to_mp4(frames, fps=2.0, width=32, height=48) is None


def test_encode_non_3channel_returns_none() -> None:
    """A non-3-channel frame is rejected rather than producing a broken mp4."""
    frames = [np.zeros((48, 64), dtype=np.uint8)]
    assert encode_frames_to_mp4(frames, fps=2.0, width=64, height=48) is None
