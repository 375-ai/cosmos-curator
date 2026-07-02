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

"""SAM3 object-tracking stage for the splitting pipeline.

A single fused stage that decodes each clip once, runs SAM3 in pre-loaded
chunked mode (enables hotstart heuristics: phantom/duplicate removal, occlusion
handling), and — in the same pass — serializes the track data and (optionally)
renders the annotated mp4, encoding once. Annotation is folded into the stage
rather than split into separate CPU stages because (a) SAM3 masks are too large
to transport across the Ray task boundary and (b) splitting forces a re-encode
between stages; keeping it here means exactly one decode and one encode.

The work is decomposed into small, stateless functions wired by a thin
``_process_clip`` so each can be unit-tested and recomposed (e.g. a future
annotate-only stage reuses ``annotate_frames`` over loaded track data):

- ``decode_clip_at_fps`` (``sensor_decode``) — bytes -> frames + real PTS
- ``track_objects``      — frames -> per-frame ``Detection`` lists + instances
- ``build_track_records`` (``track_funcs``) — ``Detection`` lists -> ``sam3_frames`` JSON
- ``annotate_frames`` (``track_funcs``)     — frames + ``sam3_frames`` -> annotated BGR
- ``encode_frames_to_mp4`` (``track_funcs``)— annotated frames -> mp4 bytes

The CPU-only serialize/render/encode helpers live in ``track_funcs`` (no
``torch`` dependency, so they unit-test on CPU); only ``track_objects`` (model
inference) and the stage wiring live here. Frames are decoded via the shared
sensor library, so each carries its real presentation timestamp (PTS) rather
than ``frame_idx / fps``. This stage runs in the ``sam3`` pixi environment
(isolated from vLLM) and requires one full GPU.
"""

from typing import Any, Literal

import attrs
import numpy as np
import numpy.typing as npt
import torch
from loguru import logger

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.core.utils.infra.gpu_start_helper import gpu_stage_startup
from cosmos_curator.models.sam3 import SAM3Model
from cosmos_curator.pipelines.video.tracking.sensor_decode import decode_clip_at_fps
from cosmos_curator.pipelines.video.tracking.track_funcs import (
    annotate_frames,
    build_track_records,
    encode_frames_to_mp4,
)
from cosmos_curator.pipelines.video.tracking.visualization import Detection
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask


@attrs.define
class SAM3QualityConfig:
    """Optional ``Sam3VideoConfig`` attribute overrides; ``None`` = SAM3 default."""

    score_threshold_detection: float | None = None
    det_nms_thresh: float | None = None
    new_det_thresh: float | None = None
    fill_hole_area: int | None = None
    recondition_every_nth_frame: int | None = None
    recondition_on_trk_masks: bool | None = None
    high_conf_thresh: float | None = None
    high_iou_thresh: float | None = None

    def to_overrides(self) -> dict[str, Any] | None:
        """Return a dict of non-``None`` overrides or ``None`` if all are default."""
        overrides = {k: v for k, v in attrs.asdict(self).items() if v is not None}
        return overrides or None


def _postprocess_to_detections(processed: dict[str, Any], prompts: list[str]) -> list[Detection]:
    """Convert SAM3 ``postprocess_outputs`` dict to a flat list of ``Detection``."""
    obj_ids: list[int] = processed["object_ids"].tolist()
    masks = processed["masks"]
    boxes = processed["boxes"]
    p2o: dict[str, list[int]] = processed["prompt_to_obj_ids"]

    detections: list[Detection] = []
    for prompt in prompts:
        for oid in p2o.get(prompt, []):
            if oid not in obj_ids:
                continue
            idx = obj_ids.index(oid)
            mask_np = masks[idx].cpu().numpy()
            if mask_np.any():
                detections.append(
                    Detection(
                        prompt=prompt,
                        object_id=oid,
                        box_xyxy=boxes[idx].tolist(),
                        mask=mask_np,
                    )
                )
    return detections


def track_objects(  # noqa: PLR0913 — inference helper; params mirror the SAM3 + decode contract
    sam3: SAM3Model,
    frames_rgb: list[npt.NDArray[np.uint8]],
    timestamps_s: list[float],
    prompts: list[str],
    *,
    session_reset_s: float,
    target_fps: float,
) -> tuple[list[list[Detection]], list[dict[str, Any]]]:
    """Run SAM3 pre-loaded chunked inference over already-decoded frames.

    Pure inference: no decode, no drawing, no I/O. Returns per-frame in-memory
    ``Detection`` lists (carrying boolean masks) aligned 1:1 with
    ``frames_rgb``, plus per-object instance summaries.

    SAM3 assigns object_ids fresh per session (per chunk), so the same raw id
    recurs in every chunk. Each ``(chunk_idx, raw_object_id)`` is remapped to a
    clip-global id (rewritten onto the detections and instances) so no two
    unrelated tracks from different chunks ever share an id downstream. Instance
    start/end times use the frames' real sensor PTS (``timestamps_s``, aligned
    1:1 with ``frames_rgb``).

    Returns:
        ``(per_frame_dets, instances)`` where ``per_frame_dets[i]`` are the
        detections for ``frames_rgb[i]`` and ``instances`` is sorted by
        ``(start_time_s, object_id)``.

    """
    rgb_frames = frames_rgb
    per_frame_dets: list[list[Detection]] = [[] for _ in rgb_frames]
    instances_map: dict[int, dict[str, Any]] = {}
    # (chunk_idx, raw per-chunk object_id) -> clip-global id, assigned in order
    # of first appearance so ids are compact and collision-free across chunks.
    global_id_map: dict[tuple[int, int], int] = {}

    if not rgb_frames:
        return per_frame_dets, []

    chunk_size = max(1, int(session_reset_s * target_fps) if session_reset_s else len(rgb_frames))
    n_chunks = (len(rgb_frames) + chunk_size - 1) // chunk_size

    with torch.no_grad():
        session = None
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, len(rgb_frames))
            chunk_rgb = rgb_frames[start:end]

            if session is not None:
                del session
                torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            session = sam3.processor.init_video_session(
                video=chunk_rgb,
                inference_device="cuda",
                video_storage_device="cpu",
                dtype=torch.bfloat16,
            )
            for prompt in prompts:
                sam3.processor.add_text_prompt(session, prompt)

            for model_outputs in sam3.model.propagate_in_video_iterator(
                inference_session=session,
                show_progress_bar=False,
            ):
                processed = sam3.processor.postprocess_outputs(session, model_outputs)
                local_idx = model_outputs.frame_idx
                if local_idx >= len(chunk_rgb):
                    continue
                sampled_idx = start + local_idx
                frame_time_s = timestamps_s[sampled_idx]

                detections = _postprocess_to_detections(processed, prompts)
                per_frame_dets[sampled_idx] = detections

                for det in detections:
                    global_id = global_id_map.setdefault((chunk_idx, det.object_id), len(global_id_map))
                    det.object_id = global_id
                    entry = instances_map.setdefault(
                        global_id,
                        {
                            "object_id": global_id,
                            "prompt": det.prompt,
                            "start_time_s": frame_time_s,
                            "end_time_s": frame_time_s,
                            "num_frames": 0,
                        },
                    )
                    entry["end_time_s"] = frame_time_s
                    entry["num_frames"] += 1

    # Chronological by start time, then object_id for stable output.
    instances = sorted(instances_map.values(), key=lambda e: (e["start_time_s"], e["object_id"]))
    return per_frame_dets, instances


class SAM3BBoxStage(CuratorStage):
    """SAM3 object tracking stage producing per-clip track data + annotated mp4.

    Uses pre-loaded chunked inference (enables SAM3's hotstart heuristics for
    higher-quality tracks). Fused single pass: consumes ``clip.encoded_data``
    (post-transcode mp4 bytes), decodes once, and populates
    ``clip.sam3_frames`` / ``clip.sam3_instances`` (+ frame geometry) and,
    when ``write_annotated_video`` is set, ``clip.sam3_annotated_video``
    (encoded once).
    """

    def __init__(  # noqa: PLR0913  # flat config surface keeps CLI wiring straightforward
        self,
        prompts: list[str],
        *,
        target_fps: float = 10.0,
        max_clip_duration_s: float = 30.0,
        session_reset_s: float = 10.0,
        quality_config: SAM3QualityConfig | None = None,
        region: Literal["box", "contour"] = "contour",
        write_annotated_video: bool = False,
        draw_trails: bool = False,
        annotated_video_label_style: Literal["id", "name", "none"] = "id",
        annotated_video_mask_opacity: int = 0,
        gpus_per_worker: float = 1.0,
        verbose: bool = False,
    ) -> None:
        """Initialise the stage.

        Args:
            prompts: Text descriptions of objects to track.
            target_fps: Sub-sampling rate applied to clip frames before inference.
            max_clip_duration_s: Safety rail — clips longer than this are skipped
                (GPU memory scales with clip length; the memory-bank grows per
                frame inside a session).
            session_reset_s: Chunk length in seconds. The SAM3 session is re-init'd
                between chunks to bound GPU memory.
            quality_config: Optional ``Sam3VideoConfig`` tuning knobs.
            region: ``"contour"`` (default) emits per-detection polygons; ``"box"``
                emits bounding boxes only (skips contour extraction).
            write_annotated_video: If ``True``, emit an annotated mp4 per clip
                (masks + ids + timestamp + optional trails) into
                ``clip.sam3_annotated_video``.
            draw_trails: If ``True`` and ``write_annotated_video`` is on, draw
                trajectory trails.
            annotated_video_label_style: ``"id"`` (default), ``"name"`` or
                ``"none"`` — what text label to render next to each detection
                in the annotated video.
            annotated_video_mask_opacity: 0-100 opacity of the translucent mask
                fill drawn inside each detection's silhouette. ``0`` (default)
                = outline only.
            gpus_per_worker: GPU fraction (default: one full GPU).
            verbose: Extra per-clip logging.

        """
        if not prompts:
            msg = "SAM3BBoxStage requires at least one prompt"
            raise ValueError(msg)
        if annotated_video_mask_opacity < 0 or annotated_video_mask_opacity > 100:  # noqa: PLR2004
            msg = f"annotated_video_mask_opacity must be in [0, 100], got {annotated_video_mask_opacity}"
            raise ValueError(msg)
        self._prompts = prompts
        self._target_fps = target_fps
        self._max_clip_duration_s = max_clip_duration_s
        self._session_reset_s = session_reset_s
        self._quality_config = quality_config or SAM3QualityConfig()
        self._region = region
        self._write_annotated_video = write_annotated_video
        self._draw_trails = draw_trails
        self._annotated_video_label_style = annotated_video_label_style
        self._annotated_video_mask_opacity = annotated_video_mask_opacity
        self._gpus_per_worker = gpus_per_worker
        self._verbose = verbose
        # Eager construct so ``self.model`` resolves when the pipeline builder
        # probes it; weights are loaded later in ``stage_setup``.
        self._sam3_model: SAM3Model = SAM3Model()

    @property
    def conda_env_name(self) -> str:
        """Return the pixi environment name for this stage."""
        return "sam3"

    @property
    def resources(self) -> CuratorStageResource:
        """Return resource requirements."""
        return CuratorStageResource(gpus=self._gpus_per_worker)

    @property
    def model(self) -> ModelInterface:
        """Return the underlying SAM3 model wrapper (weights loaded by ``stage_setup``)."""
        return self._sam3_model

    def stage_setup(self) -> None:
        """Load SAM3 with any configured overrides and log GPU memory."""
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=True)
        self._sam3_model.setup(config_overrides=self._quality_config.to_overrides())
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=False)

    def _process_clip(self, clip: Clip) -> None:
        # Release allocator reservations between clips; without this,
        # fragmentation accumulates and clip N+1 can OOM even when clip N fit.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        if clip.duration > self._max_clip_duration_s:
            logger.warning(
                f"[SAM3BBoxStage] clip {clip.uuid}: duration {clip.duration:.1f}s exceeds "
                f"max_clip_duration_s={self._max_clip_duration_s}s — skipping"
            )
            clip.errors["sam3_bbox"] = "clip_too_long"
            return

        mp4_data = clip.encoded_data.resolve()
        if mp4_data is None:
            logger.warning(f"[SAM3BBoxStage] clip {clip.uuid}: encoded_data missing — skipping")
            clip.errors["sam3_bbox"] = "missing_encoded_data"
            return

        mp4_bytes = mp4_data.tobytes()

        try:
            # Decode ONCE; everything below operates on these frames in memory.
            decoded = decode_clip_at_fps(mp4_bytes, self._target_fps)
            per_frame_dets, instances = track_objects(
                self._sam3_model,
                decoded.frames_rgb,
                decoded.timestamps_s,
                self._prompts,
                session_reset_s=self._session_reset_s,
                target_fps=self._target_fps,
            )
            sam3_frames = build_track_records(
                per_frame_dets,
                decoded.timestamps_s,
                include_contours=self._region == "contour",
            )
        except Exception:  # noqa: BLE001
            clip.errors["sam3_bbox"] = "inference_error"
            logger.exception(f"[SAM3BBoxStage] clip {clip.uuid}: SAM3 inference failed")
            return

        clip.sam3_frames = sam3_frames
        clip.sam3_instances = instances
        clip.sam3_frame_width = decoded.width
        clip.sam3_frame_height = decoded.height

        if self._write_annotated_video and decoded.frames_rgb:
            # Track data is already committed above; a render/encode failure only
            # costs the annotated video, so record it per-clip and keep going
            # rather than aborting the whole run.
            try:
                annotated_bgr = annotate_frames(
                    decoded.frames_rgb,
                    sam3_frames,
                    self._prompts,
                    draw_masks=True,
                    draw_timestamps=True,
                    label_style=self._annotated_video_label_style,
                    mask_opacity=self._annotated_video_mask_opacity,
                    draw_trails=self._draw_trails,
                )
                # Encode ONCE at the sampling fps (frames were sampled at target_fps).
                annotated_bytes = encode_frames_to_mp4(annotated_bgr, self._target_fps, decoded.width, decoded.height)
                if annotated_bytes is not None:
                    clip.sam3_annotated_video = bytes_to_numpy(annotated_bytes)  # type: ignore[assignment]
            except Exception:  # noqa: BLE001
                clip.errors["sam3_annotated"] = "annotate_error"
                logger.exception(f"[SAM3BBoxStage] clip {clip.uuid}: annotated video render/encode failed")

        if self._verbose:
            logger.info(
                f"[SAM3BBoxStage] clip {clip.uuid}: {len(sam3_frames)} tracked frames, "
                f"{len(instances)} instances, prompts={self._prompts}"
            )

    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:  # type: ignore[override]
        """Run SAM3 on every clip of every video in ``tasks``."""
        for task in tasks:
            for video in task.videos:
                for clip in video.clips:
                    self._process_clip(clip)
        return tasks
