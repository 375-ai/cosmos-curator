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

"""Pure (CPU-only) track-data functions wired by ``SAM3BBoxStage``.

These are the stateless building blocks downstream of SAM3 inference:
``build_track_records`` (serialize detections), ``annotate_frames`` (render),
and ``encode_frames_to_mp4`` (encode). They deliberately have no ``torch`` /
SAM3 dependency so they unit-test on CPU and can be recomposed by a future
annotate-only stage that loads track data from disk.
"""

import collections
import pathlib
import tempfile
from typing import Any, Literal

import cv2
import numpy as np
import numpy.typing as npt
from loguru import logger

from cosmos_curator.pipelines.video.tracking.visualization import Detection, draw_detections, draw_timestamp


def build_track_records(
    per_frame_dets: list[list[Detection]],
    timestamps_s: list[float],
    *,
    include_contours: bool = True,
) -> list[dict[str, Any]]:
    """Distill in-memory ``Detection`` lists into compact per-frame JSON records.

    One ``findContours`` pass (via ``Detection.to_json_dict``) turns each mask
    into ``contours_xy`` and drops the heavy boolean masks. Emits one record per
    sampled frame carrying the contiguous sampled position as ``frame_idx`` and
    the frame's real PTS as ``timestamp_s`` (the robust anchor), so downstream
    consumers (annotate, COCO/MOT exporters) never re-derive time from
    ``frame_idx / fps``.

    ``per_frame_dets`` and ``timestamps_s`` must be aligned 1:1 (both indexed by
    sampled-frame position).
    """
    records: list[dict[str, Any]] = []
    for frame_idx, (dets, ts) in enumerate(zip(per_frame_dets, timestamps_s, strict=True)):
        records.append(
            {
                "frame_idx": frame_idx,
                "timestamp_s": ts,
                "detections": [det.to_json_dict(include_contours=include_contours) for det in dets],
            }
        )
    return records


def annotate_frames(  # noqa: PLR0913 — each flag toggles an orthogonal overlay; a config object would hurt call sites
    frames_rgb: list[npt.NDArray[np.uint8]],
    sam3_frames: list[dict[str, Any]],
    prompts: list[str],
    *,
    draw_masks: bool = True,
    draw_timestamps: bool = True,
    label_style: Literal["id", "name", "none"] = "id",
    mask_opacity: int = 0,
    draw_trails: bool = False,
) -> list[npt.NDArray[np.uint8]]:
    """Render annotations onto decoded frames in a single pass.

    Composes the single-frame primitives (``draw_detections`` for masks/labels,
    ``draw_timestamp`` for the burnt-in clock) over the serialized track data.
    The ``draw_masks`` / ``draw_timestamps`` flags let a caller pick either
    overlay alone without an extra pass or re-decode. ``frames_rgb`` and
    ``sam3_frames`` must be aligned 1:1 (both produced from the same decode).

    Returns:
        Annotated BGR frames, ready for ``encode_frames_to_mp4``.

    """
    out_frames: list[npt.NDArray[np.uint8]] = []
    trails: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    for rgb, record in zip(frames_rgb, sam3_frames, strict=True):
        bgr: npt.NDArray[np.uint8] = np.asarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), dtype=np.uint8)
        if draw_masks:
            bgr = draw_detections(
                bgr,
                record.get("detections", []),
                prompts,
                trails,
                draw_trails=draw_trails,
                label_style=label_style,
                mask_opacity=mask_opacity,
            )
        if draw_timestamps:
            ts = record.get("timestamp_s")
            if ts is not None:
                draw_timestamp(bgr, float(ts))
        out_frames.append(bgr)
    return out_frames


def encode_frames_to_mp4(
    bgr_frames: list[npt.NDArray[np.uint8]],
    fps: float,
    width: int,
    height: int,
) -> bytes | None:
    """Encode BGR frames to an mp4 byte buffer via a temp file.

    ``cv2.VideoWriter`` needs a filesystem path, so we write to a temp file and
    read the bytes back. ``delete=False`` + explicit ``unlink`` avoids racing
    the ``VideoWriter``'s own handle on the same path; ``release()`` runs in a
    ``finally`` so the handle is freed even if a write raises.

    ``cv2.VideoWriter.write`` does NOT raise when a frame's size doesn't match
    the writer geometry — it silently drops the frame, yielding a short/corrupt
    mp4 with no signal. So we validate every frame's shape up front and bail
    loudly (return ``None``) on a mismatch rather than emit a truncated file.
    """
    if not bgr_frames:
        return None
    for i, frame in enumerate(bgr_frames):
        if frame.ndim != 3 or frame.shape[:2] != (height, width) or frame.shape[2] != 3:  # noqa: PLR2004
            logger.error(
                f"encode_frames_to_mp4: frame {i} has shape {frame.shape}, expected ({height}, {width}, 3); "
                f"skipping annotated video instead of writing a truncated/corrupt mp4"
            )
            return None
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        tmp_path = tf.name
    try:
        writer = cv2.VideoWriter(
            tmp_path,
            cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
            fps,
            (width, height),
        )
        try:
            if not writer.isOpened():
                logger.warning("encode_frames_to_mp4: cv2.VideoWriter failed to open — skipping annotated video")
                return None
            for frame in bgr_frames:
                writer.write(frame)
        except Exception:  # noqa: BLE001 — surface any encode failure as a skip, not a pipeline abort
            logger.exception("encode_frames_to_mp4: failed while writing frames — skipping annotated video")
            return None
        finally:
            writer.release()
        try:
            return pathlib.Path(tmp_path).read_bytes()
        except OSError:
            logger.exception("encode_frames_to_mp4: failed to read encoded mp4 — skipping annotated video")
            return None
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)
