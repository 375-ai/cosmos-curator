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

"""Tests for the native -> COCO / MOT track-data exporters."""

from typing import Any

from cosmos_curator.pipelines.video.tracking.exporters import to_coco_dict, to_mot_text


def _frames() -> list[dict[str, Any]]:
    return [
        {
            "frame_idx": 0,
            "timestamp_s": 0.0,
            "detections": [
                {
                    "prompt": "a car",
                    "object_id": 1,
                    "box_xyxy": [10, 20, 30, 60],
                    "contours_xy": [[10, 20, 30, 20, 30, 60]],
                },
                {"prompt": "a bus", "object_id": 2, "box_xyxy": [0, 0, 4, 4], "contours_xy": []},
            ],
        },
        {
            "frame_idx": 16,
            "timestamp_s": 0.533,
            "detections": [
                {
                    "prompt": "a car",
                    "object_id": 1,
                    "box_xyxy": [12, 22, 32, 62],
                    "contours_xy": [[12, 22, 32, 22, 32, 62]],
                },
            ],
        },
    ]


def test_coco_categories_are_stable_and_sorted() -> None:
    """Categories are the sorted distinct prompts with deterministic 1-based ids."""
    coco = to_coco_dict(_frames(), image_width=640, image_height=480, file_stem="clipA")
    assert coco["categories"] == [{"id": 1, "name": "a bus"}, {"id": 2, "name": "a car"}]


def test_coco_images_carry_geometry_and_timestamps() -> None:
    """Each frame becomes a COCO image with geometry, native frame_idx, and PTS."""
    coco = to_coco_dict(_frames(), image_width=640, image_height=480, file_stem="clipA")
    assert len(coco["images"]) == 2
    first = coco["images"][0]
    assert first["width"] == 640
    assert first["height"] == 480
    assert first["frame_idx"] == 0
    assert first["timestamp_s"] == 0.0
    assert coco["images"][1]["frame_idx"] == 16
    assert coco["images"][1]["timestamp_s"] == 0.533
    assert coco["images"][1]["file_name"] == "clipA_000016.jpg"


def test_coco_annotations_bbox_track_and_segmentation() -> None:
    """Annotations use xywh bbox, carry track_id, and keep contours as segmentation."""
    coco = to_coco_dict(_frames(), image_width=640, image_height=480, file_stem="clipA")
    anns = coco["annotations"]
    assert len(anns) == 3
    car0 = anns[0]
    assert car0["bbox"] == [10.0, 20.0, 20.0, 40.0]  # [x, y, w, h]
    assert car0["area"] == 20.0 * 40.0
    assert car0["track_id"] == 1
    assert car0["category_id"] == 2  # "a car" sorts after "a bus"
    assert car0["segmentation"] == [[10, 20, 30, 20, 30, 60]]
    assert car0["iscrowd"] == 0
    # The bus had no contour -> empty segmentation, still a valid bbox annotation.
    assert anns[1]["segmentation"] == []
    assert anns[1]["track_id"] == 2
    # image_id links annotations to images (1-based).
    assert {a["image_id"] for a in anns} == {1, 2}


def test_coco_empty_frames() -> None:
    """No frames -> empty COCO collections, not a crash."""
    coco = to_coco_dict([], image_width=10, image_height=10, file_stem="x")
    assert coco == {"images": [], "annotations": [], "categories": []}


def test_mot_rows_are_dense_framed_and_carry_object_id() -> None:
    """MOT rows are 1-based dense frames with id + xywh bbox columns."""
    text = to_mot_text(_frames())
    rows = text.strip().split("\n")
    assert len(rows) == 3
    # frame 1 has two detections, frame 2 has one.
    assert rows[0] == "1,1,10.00,20.00,20.00,40.00,1,-1,-1,-1"
    assert rows[1] == "1,2,0.00,0.00,4.00,4.00,1,-1,-1,-1"
    assert rows[2].startswith("2,1,12.00,22.00,20.00,40.00")


def test_mot_empty_frames() -> None:
    """No detections -> empty string (no trailing newline noise)."""
    assert to_mot_text([]) == ""
    assert to_mot_text([{"frame_idx": 0, "timestamp_s": 0.0, "detections": []}]) == ""
