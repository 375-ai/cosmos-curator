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

"""Monocular 3D scene reconstruction stage.

Turns each clip into the two 3D channels the MCAP writer emits — a coloured
background point cloud and per-frame object cuboids — plus the camera
calibration those channels are expressed in.

The stage is deliberately the only file in ``scene3d`` that touches torch: the
geometry lives in pure-NumPy modules that unit-test on CPU. It runs after SAM3
(whose tracks it consumes) and before ``ClipWriterStage``, because it needs
``clip.encoded_data``, which the writer drops.

Failure is per-clip and non-fatal: a clip that cannot be reconstructed records a
tag in ``clip.errors`` and flows on with its other artefacts intact.
"""

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch
from loguru import logger

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.utils.infra.gpu_start_helper import gpu_stage_startup
from cosmos_curator.core.utils.misc.grouping import split_by_chunk_size
from cosmos_curator.models.depth_anything_v2 import DepthAnythingV2Model, DepthVariant
from cosmos_curator.pipelines.video.scene3d import lifting
from cosmos_curator.pipelines.video.scene3d.calibration import (
    Calib,
    CameraOverrides,
    GroundFitParams,
    estimate_calibration,
)
from cosmos_curator.pipelines.video.scene3d.detection_source import DetectionSource, Sam3DetectionSource
from cosmos_curator.pipelines.video.scene3d.object_lift import (
    DepthSequence,
    ObjectLiftParams,
    build_scene_objects,
)
from cosmos_curator.pipelines.video.tracking.sensor_decode import DecodedClip, decode_clip_at_fps
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask

BackgroundMode = Literal["depth", "ground-plane"]
ObjectDepthMode = Literal["background", "per-frame"]

# Depth inference is batched so a `per-frame` run does not serialise N forwards.
_DEPTH_BATCH_SIZE = 8


class Scene3DStage(CuratorStage):
    """Reconstructs a metric 3D scene per clip from monocular depth.

    Populates ``clip.scene3d_calibration``, ``clip.scene3d_background`` (the packed
    point cloud) and ``clip.scene3d_objects`` (per-frame cuboids). Objects require
    tracked 2D boxes from a :class:`DetectionSource`; without them the stage still
    produces the background cloud and calibration.
    """

    def __init__(  # noqa: PLR0913  # flat config surface keeps CLI wiring straightforward
        self,
        *,
        depth_variant: DepthVariant = "outdoor",
        depth_long_side: int = 700,
        target_fps: float = 5.0,
        max_clip_duration_s: float = 120.0,
        camera_overrides: CameraOverrides | None = None,
        ground_params: GroundFitParams | None = None,
        cloud_params: lifting.CloudParams | None = None,
        object_params: ObjectLiftParams | None = None,
        background_mode: BackgroundMode = "depth",
        object_depth_mode: ObjectDepthMode = "background",
        gpus_per_worker: float = 0.5,
        verbose: bool = False,
    ) -> None:
        """Initialise the stage.

        Args:
            depth_variant: Depth Anything V2 metric checkpoint (``outdoor``/``indoor``).
            depth_long_side: Long side frames are downscaled to before inference.
            target_fps: Sampling rate for object lifting.
            max_clip_duration_s: Skip clips longer than this.
            camera_overrides: Explicit calibration values that beat the estimate.
            ground_params: RANSAC ground-fit thresholds.
            cloud_params: Point-cloud filters and budget.
            object_params: Cuboid filtering and sizing thresholds.
            background_mode: ``depth`` back-projects the depth map; ``ground-plane``
                ray-casts pixels onto ``z = 0``.
            object_depth_mode: ``background`` uses the single median-plate depth map;
                ``per-frame`` runs depth on every sampled frame.
            gpus_per_worker: GPU fraction reserved per worker.
            verbose: Extra per-clip logging.

        """
        if depth_long_side <= 0:
            msg = f"depth_long_side must be positive, got {depth_long_side}"
            raise ValueError(msg)
        self._depth_long_side = depth_long_side
        self._target_fps = target_fps
        self._max_clip_duration_s = max_clip_duration_s
        self._camera_overrides = camera_overrides or CameraOverrides()
        self._ground_params = ground_params or GroundFitParams()
        self._cloud_params = cloud_params or lifting.CloudParams()
        self._object_params = object_params or ObjectLiftParams()
        self._background_mode: BackgroundMode = background_mode
        self._object_depth_mode: ObjectDepthMode = object_depth_mode
        # Only one source exists today; `DetectionSource` keeps `object_lift` free of
        # any SAM3 knowledge, so a second backend is a swap here rather than a rewrite.
        self._detection_source: DetectionSource = Sam3DetectionSource()
        self._gpus_per_worker = gpus_per_worker
        self._verbose = verbose
        # Eager construct so ``self.model`` resolves when the pipeline builder
        # probes it for weight downloads; weights load later in ``stage_setup``.
        self._depth_model = DepthAnythingV2Model(depth_variant)

    @property
    def resources(self) -> CuratorStageResource:
        """Return resource requirements."""
        return CuratorStageResource(cpus=2.0, gpus=self._gpus_per_worker)

    @property
    def model(self) -> ModelInterface:
        """Return the depth model wrapper (weights loaded by ``stage_setup``)."""
        return self._depth_model

    def stage_setup(self) -> None:
        """Load the depth model and log GPU memory around it."""
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=True)
        self._depth_model.setup()
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=False)

    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:  # type: ignore[override]
        """Reconstruct every clip of every video in ``tasks``."""
        for task in tasks:
            for video in task.videos:
                for clip in video.clips:
                    self._process_clip(clip)
        return tasks

    def _process_clip(self, clip: Clip) -> None:
        if clip.duration > self._max_clip_duration_s:
            logger.warning(
                f"[Scene3DStage] clip {clip.uuid}: duration {clip.duration:.1f}s exceeds "
                f"max_clip_duration_s={self._max_clip_duration_s}s — skipping"
            )
            clip.errors["scene3d"] = "clip_too_long"
            return

        mp4_data = clip.encoded_data.resolve()
        if mp4_data is None:
            logger.warning(f"[Scene3DStage] clip {clip.uuid}: encoded_data missing — skipping")
            clip.errors["scene3d"] = "missing_encoded_data"
            return

        # Release allocator reservations between clips; without this, fragmentation
        # accumulates and clip N+1 can OOM even when clip N fit. Placed after the
        # guards so a skipped clip does not pay for a synchronising empty_cache().
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            decoded = decode_clip_at_fps(mp4_data.tobytes(), self._target_fps)
        except Exception:  # noqa: BLE001
            clip.errors["scene3d"] = "decode_error"
            logger.exception(f"[Scene3DStage] clip {clip.uuid}: decode failed")
            return
        if not decoded.frames_rgb:
            clip.errors["scene3d"] = "no_frames"
            return

        source_width, source_height = decoded.width, decoded.height
        timestamps_s = list(decoded.timestamps_s)
        scaled = self._downscale_frames(decoded)
        # `scaled` is non-empty (guarded above), so the plate always builds.
        plate = lifting.background_plate(scaled, long_side=self._depth_long_side)
        assert plate is not None

        try:
            depth_m = self._depth_model.infer(plate)
            calib = estimate_calibration(
                depth_m,
                overrides=self._camera_overrides.rescaled_to(plate.shape[1], source_width),
                ground_params=self._ground_params,
            )
            points, colours = self._build_cloud(calib, depth_m, plate)
        except Exception:  # noqa: BLE001
            clip.errors["scene3d"] = "reconstruction_error"
            logger.exception(f"[Scene3DStage] clip {clip.uuid}: 3D reconstruction failed")
            return

        if calib.source == "fallback":
            # A fully overridden pose is also `estimated=False`, but that is the user's
            # choice rather than a failure, so only a rejected fit is reported.
            clip.errors["scene3d_calibration"] = "ground_fit_failed"

        clip.scene3d_calibration = calib.to_payload(width=source_width, height=source_height)
        clip.scene3d_background = lifting.pack_point_cloud(points, colours)  # type: ignore[assignment]

        try:
            clip.scene3d_objects = self._lift_objects(clip, calib, depth_m, scaled, timestamps_s)
        except Exception:  # noqa: BLE001
            # The cloud and calibration are already committed; an object-lift
            # failure only costs the cuboids, so record it and keep going.
            clip.errors["scene3d_objects"] = "object_lift_error"
            logger.exception(f"[Scene3DStage] clip {clip.uuid}: object lift failed")

        if self._verbose:
            num_entities = sum(len(record["entities"]) for record in clip.scene3d_objects or [])
            logger.info(
                f"[Scene3DStage] clip {clip.uuid}: {points.shape[0]} background points, "
                f"{len(clip.scene3d_objects or [])} object frames ({num_entities} cuboids), "
                f"camera height {calib.camera_height_m:.2f} m "
                f"(estimated={calib.estimated}, inliers={calib.ground_inlier_frac:.2f})"
            )

    def _downscale_frames(self, decoded: DecodedClip) -> list[npt.NDArray[np.uint8]]:
        """Downscale to the depth grid and release the full-resolution frames.

        ``decode_clip_at_fps`` materialises every sampled frame at source resolution;
        at 1080p and the default 120 s clip cap that is several GB held for the whole
        of ``_process_clip``, while nothing downstream reads pixels above
        ``--scene3d-depth-long-side``. In ``background`` mode only the frames that
        enter the median are worth resizing at all.
        """
        wanted = (
            decoded.frames_rgb
            if self._object_depth_mode == "per-frame"
            else lifting.select_plate_frames(decoded.frames_rgb)
        )
        scaled = [lifting.resize_long_side(frame, self._depth_long_side) for frame in wanted]
        # `decoded` is local to the caller, so dropping its frames is safe and is the
        # only way to free them before the GPU forward runs.
        decoded.frames_rgb.clear()
        return scaled

    def _build_cloud(
        self,
        calib: Calib,
        depth_m: npt.NDArray[np.float32],
        plate: npt.NDArray[np.uint8],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
        """Produce the downsampled background cloud for the configured mode."""
        if self._background_mode == "ground-plane":
            points, colours = lifting.ground_orthophoto(calib, plate, self._cloud_params)
        else:
            points, colours = lifting.backproject(calib, depth_m, plate, self._cloud_params)
        return lifting.voxel_downsample(points, colours, self._cloud_params.max_points)

    def _lift_objects(
        self,
        clip: Clip,
        calib: Calib,
        depth_m: npt.NDArray[np.float32],
        scaled: list[npt.NDArray[np.uint8]],
        timestamps_s: list[float],
    ) -> list[dict[str, Any]] | None:
        """Lift tracked 2D boxes into cuboids, or return ``None`` without tracks."""
        tracks = self._detection_source.tracks(clip)
        if tracks is None:
            return None
        depth_by_frame = None
        if self._object_depth_mode == "per-frame":
            depth_by_frame = self._infer_frame_depths(scaled, timestamps_s, depth_m.shape)
        return build_scene_objects(
            calib,
            depth_m,
            tracks,
            self._object_params,
            depth_by_frame=depth_by_frame,
        )

    def _infer_frame_depths(
        self,
        scaled: list[npt.NDArray[np.uint8]],
        timestamps_s: list[float],
        plate_shape: tuple[int, ...],
    ) -> DepthSequence | None:
        """Run depth on every sampled frame, keeping each frame's real timestamp.

        The timestamps travel with the depth maps because the detector samples at its
        own rate; ``DepthSequence`` pairs them back to track frames by time.
        """
        target = (int(plate_shape[0]), int(plate_shape[1]))
        usable = [
            (frame, float(timestamp))
            for frame, timestamp in zip(scaled, timestamps_s, strict=True)
            if frame.shape[:2] == target
        ]
        depths: list[npt.NDArray[np.float32]] = []
        for batch in split_by_chunk_size(usable, _DEPTH_BATCH_SIZE):
            depths.extend(self._depth_model.infer_batch([frame for frame, _ in batch]))
        sequence = DepthSequence.build([timestamp for _, timestamp in usable], depths)
        return sequence if sequence.depths else None
