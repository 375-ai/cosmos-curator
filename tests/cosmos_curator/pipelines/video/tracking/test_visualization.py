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

"""Unit tests for the contour-based drawing helpers used by the annotate step.

The annotate step redraws from the compact ``contours_xy`` polygons in the
track data rather than boolean masks; these tests pin that polygon-decode +
draw path (positive vs negative mode, opacity, labels) without any video I/O.
"""

import collections

import numpy as np

from cosmos_curator.pipelines.video.tracking.visualization import (
    Detection,
    draw_detections,
    draw_timestamp,
)


def _flat_frame(value: int = 128) -> np.ndarray:
    return np.full((64, 64, 3), value, dtype=np.uint8)


def _detection() -> dict:
    return {
        "prompt": "a car",
        "object_id": 7,
        "box_xyxy": [20, 20, 44, 44],
        "contours_xy": [[20, 20, 44, 20, 44, 44, 20, 44]],
    }


def test_draw_detections_positive_changes_pixels() -> None:
    """Positive mode draws a contour outline that perturbs the flat frame."""
    frame = _flat_frame()
    out = draw_detections(frame, [_detection()], ["a car"], collections.defaultdict(list))
    assert out.shape == frame.shape
    # Original is untouched (function returns a copy); output differs.
    assert np.array_equal(frame, _flat_frame())
    assert not np.array_equal(out, frame)


def test_draw_detections_opacity_fills_interior() -> None:
    """A non-zero opacity tints the interior of the contour, not just the edge."""
    centre = (32, 32)
    no_fill = draw_detections(_flat_frame(), [_detection()], ["a car"], collections.defaultdict(list), mask_opacity=0)
    filled = draw_detections(_flat_frame(), [_detection()], ["a car"], collections.defaultdict(list), mask_opacity=80)
    # The geometric centre is inside the polygon: only the filled render alters it.
    assert np.array_equal(no_fill[centre], _flat_frame()[centre])
    assert not np.array_equal(filled[centre], _flat_frame()[centre])


def test_draw_detections_negative_blurs_region() -> None:
    """Negative mode obscures inside the region (changes a textured interior)."""
    frame = _flat_frame()
    # Add interior texture so the blur has something to smear.
    frame[24:40, 24:40] = np.tile(np.arange(0, 48, 3, dtype=np.uint8), (16, 1))[:, :, None]
    before = frame.copy()
    out = draw_detections(frame, [_detection()], ["a car"], collections.defaultdict(list), mode="negative")
    assert not np.array_equal(out[24:40, 24:40], before[24:40, 24:40])


def test_draw_detections_box_only_fallback() -> None:
    """With no contour (region=box), positive mode draws the bbox rectangle."""
    frame = _flat_frame()
    det = {"prompt": "a car", "object_id": 7, "box_xyxy": [20, 20, 44, 44], "contours_xy": []}
    out = draw_detections(frame, [det], ["a car"], collections.defaultdict(list), label_style="none")
    assert not np.array_equal(out, frame)
    # The rectangle border perturbs pixels along the box edge.
    assert not np.array_equal(out[20, 20:44], _flat_frame()[20, 20:44])


def test_detection_to_json_dict_region_box_drops_contours() -> None:
    """``include_contours=False`` yields an empty ``contours_xy`` (region=box)."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:44, 20:44] = True
    det = Detection(prompt="a car", object_id=1, box_xyxy=[20, 20, 44, 44], mask=mask)
    with_contours = det.to_json_dict(include_contours=True)
    without = det.to_json_dict(include_contours=False)
    assert with_contours["contours_xy"]  # non-empty polygon
    assert without["contours_xy"] == []
    assert without["box_xyxy"] == [20, 20, 44, 44]


def test_draw_timestamp_marks_top_left() -> None:
    """The burnt-in ``t=X.XXs`` badge lands top-left and leaves bottom-right clean."""
    # Use a realistically sized frame: the badge scale is tuned for full-size
    # video, so it would overflow a 64x64 frame.
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    draw_timestamp(frame, 1.23)
    assert frame[0:48, 0:200].std() > 10.0
    # Bottom-right stays flat.
    assert frame[-30:, -30:].std() < 1.0
