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

"""Where the 2D tracks that become 3D cuboids come from.

The 3D lift needs nothing but per-frame boxes with stable identities, so it is
kept behind a narrow protocol rather than wired to a specific detector. Today the
only implementation reads the SAM3 tracking stage's output; a standalone detector
plugs in by satisfying :class:`DetectionSource` without ``object_lift`` changing.

Boxes are expressed in the coordinate space of the frames the detector saw, which
is not necessarily the space the depth map lives in — the source reports its own
frame size so callers can rescale.
"""

from typing import TYPE_CHECKING, Protocol

import attrs

if TYPE_CHECKING:
    from cosmos_curator.pipelines.video.utils.data_model import Clip


@attrs.define(frozen=True)
class TrackedBox:
    """One tracked 2D detection."""

    object_id: int
    label: str
    box_xyxy: tuple[float, float, float, float]


@attrs.define(frozen=True)
class FrameDetections:
    """All tracked detections for one sampled frame."""

    timestamp_s: float
    boxes: list[TrackedBox]


@attrs.define(frozen=True)
class DetectionTracks:
    """A clip's per-frame tracks plus the pixel space the boxes live in."""

    frames: list[FrameDetections]
    frame_width: int
    frame_height: int


class DetectionSource(Protocol):
    """Supplies tracked 2D boxes for a clip."""

    def tracks(self, clip: "Clip") -> DetectionTracks | None:
        """Return the clip's tracked detections, or ``None`` when unavailable."""
        ...


_BOX_LENGTH = 4


def _coerce_box(raw: object) -> tuple[float, float, float, float] | None:
    """Validate a raw ``[x1, y1, x2, y2]`` payload into an ordered, non-degenerate box."""
    if not isinstance(raw, list | tuple) or len(raw) != _BOX_LENGTH:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    left, right = (x1, x2) if x1 <= x2 else (x2, x1)
    top, bottom = (y1, y2) if y1 <= y2 else (y2, y1)
    if right - left <= 0 or bottom - top <= 0:
        return None
    return left, top, right, bottom


@attrs.define(frozen=True)
class Sam3DetectionSource:
    """Adapts ``Clip.sam3_frames`` to :class:`DetectionSource`.

    ``sam3_frames`` entries look like
    ``{"frame_idx": int, "timestamp_s": float, "detections": [{"prompt", "object_id",
    "box_xyxy", "contours_xy"}]}``; only the prompt, id and box are used here.
    """

    def tracks(self, clip: "Clip") -> DetectionTracks | None:
        """Return the clip's SAM3 tracks, or ``None`` when SAM3 did not run."""
        records = clip.sam3_frames
        if not records:
            return None
        width = clip.sam3_frame_width or 0
        height = clip.sam3_frame_height or 0
        if width <= 0 or height <= 0:
            return None

        frames: list[FrameDetections] = []
        for record in records:
            timestamp = record.get("timestamp_s")
            if timestamp is None:
                continue
            boxes: list[TrackedBox] = []
            for detection in record.get("detections", []):
                object_id = detection.get("object_id")
                box = _coerce_box(detection.get("box_xyxy"))
                if object_id is None or box is None:
                    continue
                boxes.append(TrackedBox(object_id=int(object_id), label=str(detection.get("prompt", "")), box_xyxy=box))
            frames.append(FrameDetections(timestamp_s=float(timestamp), boxes=boxes))
        if not frames:
            return None
        return DetectionTracks(frames=frames, frame_width=int(width), frame_height=int(height))
