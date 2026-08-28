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

"""Tracked 2D boxes to 3D cuboids in the map frame.

Two passes, because heading is a property of a whole trajectory rather than of a
frame:

1. Lift each detection's ground-contact pixel (bottom-centre of the box) to a
   map-frame point, accumulating one trajectory per track id.
2. Fit a single least-squares velocity per track, then emit per-frame cuboids
   using that one stable heading. Fitting per frame would let the arrow flip
   whenever two consecutive positions jitter.

Sizing combines two estimators (see :mod:`priors`): a monocular view measures an
object's width and height directly but can never see its length, so a class prior
supplies length when the label is recognised and the depth-derived extent covers
everything else.
"""

import math
from collections import defaultdict
from typing import Any, Literal

import attrs
import numpy as np
import numpy.typing as npt

from cosmos_curator.pipelines.video.scene3d import priors
from cosmos_curator.pipelines.video.scene3d.calibration import Calib
from cosmos_curator.pipelines.video.scene3d.detection_source import DetectionTracks
from cosmos_curator.pipelines.video.scene3d.lifting import DEFAULT_MAX_RANGE_M, place

DimensionMode = Literal["auto", "prior", "depth"]
DIMENSION_MODES: tuple[DimensionMode, ...] = ("auto", "prior", "depth")

# Arrow colours for the two dominant motion directions (see `_flow_reference`).
_FLOW_COLOR_A = (0.10, 0.85, 1.0, 1.0)
_FLOW_COLOR_B = (1.0, 0.30, 0.85, 1.0)
_MIN_FLOW_TRACKS = 2
_CUBE_ALPHA = 0.75
_LABEL_FONT_SIZE = 0.5
# Below this the depth-derived extent is noise rather than a measurement.
_MIN_DIMENSION_M = 0.05
# Below this a fitted velocity is numerical noise rather than motion.
_MIN_SPEED_MPS = 1e-6


@attrs.define(frozen=True)
class DepthSequence:
    """Per-frame depth maps addressed by time rather than by index.

    The detector and this stage decode independently (``--sam3-target-fps`` vs
    ``--scene3d-target-fps``), so their frame indices are unrelated. Selecting the
    depth map nearest a track frame's presentation timestamp is the only correct
    pairing; indexing by position silently misaligns whenever the two rates differ.
    """

    timestamps_s: npt.NDArray[np.float64]
    depths: list[npt.NDArray[np.float32]]

    @classmethod
    def build(cls, timestamps_s: list[float], depths: list[npt.NDArray[np.float32]]) -> "DepthSequence":
        """Pair timestamps with depth maps, truncating to the shorter of the two."""
        count = min(len(timestamps_s), len(depths))
        return cls(
            timestamps_s=np.asarray(timestamps_s[:count], dtype=np.float64),
            depths=list(depths[:count]),
        )

    def nearest(self, timestamp_s: float) -> npt.NDArray[np.float32]:
        """Return the depth map closest in time to ``timestamp_s``."""
        index = int(np.argmin(np.abs(self.timestamps_s - timestamp_s)))
        return self.depths[index]


@attrs.define(frozen=True, kw_only=True)
class ObjectLiftParams:
    """Thresholds governing which tracks survive and how cuboids are sized."""

    dimension_mode: DimensionMode = "auto"
    snap_to_ground: bool = True
    max_range_m: float = DEFAULT_MAX_RANGE_M
    min_track_points: int = 4
    min_net_displacement_m: float = 1.0
    max_speed_mps: float = 60.0


@attrs.define(frozen=True)
class _Observation:
    """One lifted detection: where it is, how big it looked, and when."""

    track_id: int
    label: str
    position_xy: npt.NDArray[np.float64]
    ground_z: float
    width_m: float
    height_m: float


def _scale_factors(tracks: DetectionTracks, calib: Calib) -> tuple[float, float]:
    """Map detector pixel coordinates onto the depth map's pixel grid."""
    if tracks.frame_width <= 0 or tracks.frame_height <= 0:
        return 1.0, 1.0
    return calib.width / tracks.frame_width, calib.height / tracks.frame_height


def _measure_extent(
    calib: Calib,
    box: tuple[float, float, float, float],
    depth_z: float,
) -> tuple[float, float]:
    """Convert a 2D box at range ``depth_z`` into metric width and height."""
    x1, y1, x2, y2 = box
    width_m = (x2 - x1) * depth_z / float(calib.K[0, 0])
    height_m = (y2 - y1) * depth_z / float(calib.K[1, 1])
    return max(width_m, _MIN_DIMENSION_M), max(height_m, _MIN_DIMENSION_M)


def _dimensions_for(
    label: str,
    observed_width_m: float,
    observed_height_m: float,
    mode: DimensionMode,
) -> tuple[priors.Dimensions, priors.Color]:
    """Resolve ``(length, width, height)`` and a display colour for one object."""
    prior = priors.prior_for(label)
    if mode == "prior" or (mode == "auto" and prior is not None):
        if prior is not None:
            return prior
        return priors.DEFAULT_DIMENSIONS, priors.DEFAULT_COLOR

    # Depth-derived: width and height are measured; length borrows the prior's
    # aspect ratio when one exists, else the object is assumed square in plan.
    if prior is not None:
        prior_dims, colour = prior
        aspect = prior_dims[0] / prior_dims[1] if prior_dims[1] > 0 else 1.0
    else:
        aspect, colour = 1.0, priors.DEFAULT_COLOR
    length_m = max(observed_width_m * aspect, _MIN_DIMENSION_M)
    return (length_m, observed_width_m, observed_height_m), colour


def _fit_velocity(
    times_s: list[float],
    positions: list[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Least-squares ``(vx, vy)`` for ``position = p0 + v * t`` over a whole track."""
    times: npt.NDArray[np.float64] = np.asarray(times_s, dtype=np.float64)
    points = np.asarray(positions, dtype=np.float64)
    design = np.vstack([times, np.ones_like(times)]).T
    # lstsq accepts a 2-D right-hand side, so x and y share one decomposition.
    velocity: npt.NDArray[np.float64] = np.linalg.lstsq(design, points[:, :2], rcond=None)[0][0]
    return velocity


def _flow_reference(velocities: list[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64] | None:
    """Principal motion axis across all moving tracks, used to two-colour the flow.

    Unsupervised and domain-free: it just finds the dominant direction of travel
    in the scene, so opposing streams get distinct arrow colours. Returns ``None``
    when too few tracks move to define an axis.
    """
    if len(velocities) < _MIN_FLOW_TRACKS:
        return None
    _, _, vt = np.linalg.svd(np.asarray(velocities, dtype=np.float64))
    return np.asarray(vt[0], dtype=np.float64)


def build_scene_objects(
    calib: Calib,
    depth_m: npt.NDArray[np.float32],
    tracks: DetectionTracks,
    params: ObjectLiftParams,
    *,
    depth_by_frame: DepthSequence | None = None,
) -> list[dict[str, Any]]:
    """Lift tracked 2D boxes into per-frame 3D cuboid records.

    Args:
        calib: Camera model matching ``depth_m``'s resolution.
        depth_m: Background-plate metric depth, used when ``depth_by_frame`` is None.
        tracks: Per-frame tracked boxes from a :class:`DetectionSource`.
        params: Filtering and sizing thresholds.
        depth_by_frame: Optional per-frame depth maps with their own timestamps, for
            scenes where a single plate is not enough. Matched to track frames by
            time, never by position: the detector and this stage sample at
            independent rates, so their frame indices do not correspond.

    Returns:
        ``[{"timestamp_s": float, "entities": [...]}]`` — one record per frame that
        produced at least one entity. Entities are plain dicts so they serialise
        straight into the MCAP writer and the JSON sidecar.

    """
    scale_u, scale_v = _scale_factors(tracks, calib)
    horizon = calib.horizon_v()

    per_frame: list[tuple[float, list[_Observation]]] = []
    trajectories: dict[int, list[tuple[float, npt.NDArray[np.float64]]]] = defaultdict(list)

    for frame in tracks.frames:
        frame_depth = depth_m if depth_by_frame is None else depth_by_frame.nearest(frame.timestamp_s)
        observations: list[_Observation] = []
        for tracked in frame.boxes:
            x1, y1, x2, y2 = tracked.box_xyxy
            scaled = (x1 * scale_u, y1 * scale_v, x2 * scale_u, y2 * scale_v)
            contact_u = (scaled[0] + scaled[2]) / 2.0
            contact_v = scaled[3]
            if contact_v <= horizon + 2:
                continue
            placed = place(calib, frame_depth, contact_u, contact_v)
            if placed is None:
                continue
            point, depth_z = placed
            position_xy = point[:2]
            if float(np.linalg.norm(position_xy - calib.t[:2])) > params.max_range_m:
                continue
            width_m, height_m = _measure_extent(calib, scaled, depth_z)
            observations.append(
                _Observation(
                    track_id=tracked.object_id,
                    label=tracked.label,
                    position_xy=position_xy,
                    ground_z=float(point[2]),
                    width_m=width_m,
                    height_m=height_m,
                )
            )
            trajectories[tracked.object_id].append((frame.timestamp_s, position_xy))
        per_frame.append((frame.timestamp_s, observations))

    headings, velocities = _fit_headings(trajectories, params)
    reference = _flow_reference(velocities)

    lifetime_ns = _lifetime_ns(tracks)
    records: list[dict[str, Any]] = []
    for timestamp_s, observations in per_frame:
        entities = [
            entity
            for observation in observations
            if (entity := _build_entity(observation, headings, reference, params, lifetime_ns)) is not None
        ]
        if entities:
            records.append({"timestamp_s": timestamp_s, "entities": entities})
    return records


def _fit_headings(
    trajectories: dict[int, list[tuple[float, npt.NDArray[np.float64]]]],
    params: ObjectLiftParams,
) -> tuple[dict[int, tuple[npt.NDArray[np.float64] | None, float]], list[npt.NDArray[np.float64]]]:
    """Fit one heading and speed per track; drop tracks too short to trust."""
    headings: dict[int, tuple[npt.NDArray[np.float64] | None, float]] = {}
    velocities: list[npt.NDArray[np.float64]] = []
    for track_id, samples in trajectories.items():
        if len(samples) < params.min_track_points:
            continue
        times = [time for time, _ in samples]
        points = [point for _, point in samples]
        velocity = _fit_velocity(times, points)
        speed = min(float(np.linalg.norm(velocity)), params.max_speed_mps)
        net = float(np.linalg.norm(np.asarray(points[-1]) - np.asarray(points[0])))
        moving = net >= params.min_net_displacement_m and float(np.linalg.norm(velocity)) > _MIN_SPEED_MPS
        direction = velocity / float(np.linalg.norm(velocity)) if moving else None
        headings[track_id] = (direction, speed)
        if direction is not None:
            velocities.append(velocity)
    return headings, velocities


def _lifetime_ns(tracks: DetectionTracks) -> int:
    """Two sampling periods, so an entity persists until its successor arrives."""
    timestamps = [frame.timestamp_s for frame in tracks.frames]
    if len(timestamps) < 2:  # noqa: PLR2004
        return 1_000_000_000
    deltas: npt.NDArray[np.float64] = np.diff(np.asarray(timestamps, dtype=np.float64))
    positive = deltas[deltas > 0]
    period = float(np.median(positive)) if positive.size else 0.5
    return int(2 * period * 1e9)


def _build_entity(
    observation: _Observation,
    headings: dict[int, tuple[npt.NDArray[np.float64] | None, float]],
    reference: npt.NDArray[np.float64] | None,
    params: ObjectLiftParams,
    lifetime_ns: int,
) -> dict[str, Any] | None:
    """Render one observation as a cuboid entity, or ``None`` for a dropped track."""
    heading = headings.get(observation.track_id)
    if heading is None:
        return None
    direction, speed = heading

    (length_m, width_m, height_m), colour = _dimensions_for(
        observation.label, observation.width_m, observation.height_m, params.dimension_mode
    )
    base_z = 0.0 if params.snap_to_ground else observation.ground_z
    centre = (
        float(observation.position_xy[0]),
        float(observation.position_xy[1]),
        base_z + height_m / 2.0,
    )

    if direction is not None:
        yaw = float(math.atan2(direction[1], direction[0]))
        flow_colour = _FLOW_COLOR_A
        if reference is not None and float(np.dot(direction, reference)) < 0:
            flow_colour = _FLOW_COLOR_B
        arrow = {
            "position": (centre[0], centre[1], centre[2] + height_m / 2.0 + 0.3),
            "yaw": yaw,
            "length": float(min(max(speed * 0.4, 1.0), 10.0)),
            "color": flow_colour,
        }
    else:
        yaw = 0.0
        arrow = None

    label = priors.match_label(observation.label) or (observation.label or "object")
    return {
        "id": f"obj_{observation.track_id}",
        "label": label,
        "track_id": int(observation.track_id),
        "lifetime_ns": lifetime_ns,
        "cube": {
            "position": centre,
            "size": (length_m, width_m, height_m),
            "yaw": yaw,
            "color": (*colour, _CUBE_ALPHA),
        },
        "arrow": arrow,
        "text": {
            "position": (centre[0], centre[1], centre[2] + height_m / 2.0 + 0.7),
            "value": f"{label} #{observation.track_id}",
            "font_size": _LABEL_FONT_SIZE,
        },
        "metadata": {"class": label, "speed_mps": f"{speed:.1f}"},
    }
