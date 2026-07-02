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

"""Track-data exporters: native -> COCO / MOT.

The track stage emits native per-frame records (see ``serialization.py`` and
``docs/curator/design/object-tracking.md``); these helpers convert that native
shape into the two standard formats teams commonly plug existing tooling into:

- `COCO <https://cocodataset.org/#format-data>`_ object-detection JSON, with a
  per-detection ``track_id`` extension so instance identity survives.
- `MOT <https://motchallenge.net/instructions/>`_ CSV rows
  (``frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z``).

Both are pure functions over the native ``frames`` list so they're trivially
testable and reusable by any writer.
"""

from typing import Any

# Sentinel columns for MOT detection rows (no 3D / confidence info available).
_MOT_CONF = 1
_MOT_WORLD = -1


def _category_index(frames: list[dict[str, Any]]) -> dict[str, int]:
    """Map each distinct prompt to a stable 1-based COCO ``category_id``.

    Sorted so the mapping is deterministic across runs and clips.
    """
    prompts: set[str] = set()
    for frame in frames:
        for det in frame.get("detections", []):
            prompts.add(str(det.get("prompt", "")))
    return {prompt: i for i, prompt in enumerate(sorted(prompts), start=1)}


def _xywh(box_xyxy: list[float]) -> tuple[float, float, float, float]:
    """Convert ``[x1, y1, x2, y2]`` to COCO/MOT ``(x, y, w, h)``."""
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def to_coco_dict(
    frames: list[dict[str, Any]],
    *,
    image_width: int,
    image_height: int,
    file_stem: str,
) -> dict[str, Any]:
    """Convert native per-frame records to a COCO object-detection dict.

    Args:
        frames: native ``objects.json`` ``frames`` list, each
            ``{"frame_idx", "timestamp_s", "detections": [...]}``.
        image_width: frame width in pixels (for the COCO ``images`` entries).
        image_height: frame height in pixels.
        file_stem: clip identifier used to synthesize per-frame ``file_name``s.

    Returns:
        A COCO dict with ``images`` / ``annotations`` / ``categories``. Each
        annotation carries a ``track_id`` (the SAM3 ``object_id``) and the
        native ``contours_xy`` polygon as ``segmentation`` when present. Each
        image keeps the native ``frame_idx`` + real ``timestamp_s``.

    """
    categories = _category_index(frames)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 1

    for image_id, frame in enumerate(frames, start=1):
        frame_idx = int(frame.get("frame_idx", image_id - 1))
        images.append(
            {
                "id": image_id,
                "frame_idx": frame_idx,
                "file_name": f"{file_stem}_{frame_idx:06d}.jpg",
                "width": image_width,
                "height": image_height,
                "timestamp_s": frame.get("timestamp_s"),
            }
        )
        for det in frame.get("detections", []):
            x, y, w, h = _xywh(det.get("box_xyxy", [0, 0, 0, 0]))
            segmentation = det.get("contours_xy") or []
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": categories.get(str(det.get("prompt", "")), 0),
                    "track_id": det.get("object_id"),
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "segmentation": segmentation,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": cid, "name": name} for name, cid in sorted(categories.items(), key=lambda kv: kv[1])],
    }


def to_mot_text(frames: list[dict[str, Any]]) -> str:
    """Convert native per-frame records to MOT-challenge CSV rows.

    Columns: ``frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z``.
    Frame numbers are 1-based and contiguous over the sampled frames (MOT
    sequences are dense), so the native ``frame_idx`` is not used here; identity
    is carried by ``id`` (the SAM3 ``object_id``).
    """
    rows: list[str] = []
    for frame_no, frame in enumerate(frames, start=1):
        for det in frame.get("detections", []):
            x, y, w, h = _xywh(det.get("box_xyxy", [0, 0, 0, 0]))
            obj_id = int(det.get("object_id", -1))
            rows.append(
                f"{frame_no},{obj_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},"
                f"{_MOT_CONF},{_MOT_WORLD},{_MOT_WORLD},{_MOT_WORLD}"
            )
    return "\n".join(rows) + ("\n" if rows else "")
