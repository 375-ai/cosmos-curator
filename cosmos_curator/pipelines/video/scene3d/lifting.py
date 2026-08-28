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

"""Pixels to metric points: background plate, back-projection, and cloud packing.

Everything here is pure NumPy (plus OpenCV for resizing) so it unit-tests on CPU
without torch or a GPU.

The background plate deserves a note. Depth is inferred once per clip from a
*temporal median* of sampled frames, which erases anything that moves and leaves
the static scene. That has two payoffs: the depth model sees an unoccluded scene,
and a lookup at an object's ground-contact pixel returns the depth of the floor
*behind* the object — which is what a ground-contact point should measure. Frames
are downscaled to the depth input size **before** stacking; medianing full 4K
frames promotes a ~500 MB uint8 stack to float64 and peaks at several GB.
"""

import attrs
import cv2
import numpy as np
import numpy.typing as npt

from cosmos_curator.pipelines.video.scene3d.calibration import Calib, pixel_rays

_MAX_PLATE_FRAMES = 20
# Relative depth-gradient ceiling; above it a sample is a "flying pixel" straddling
# a depth discontinuity, which renders as a smear trailing thin foreground objects.
DEFAULT_EDGE_THRESHOLD = 0.06
# Voxel-size bisection bounds, as fractions of the cloud's largest extent.
_VOXEL_SEARCH_FLOOR = 4096.0
_VOXEL_SEARCH_EPSILON = 8192.0
# Ground range beyond which points and objects are culled. The outdoor depth model
# saturates at 80 m, so anything far above that only admits noise.
DEFAULT_MAX_RANGE_M = 150.0
# A ray must point measurably downwards to strike the ground plane.
_DOWNWARD_EPSILON = -1e-6
# Points this far below the ground plane are depth noise, not scene geometry.
_MIN_POINT_HEIGHT_M = -3.0


@attrs.define(frozen=True, kw_only=True)
class CloudParams:
    """Filters and budget applied when turning a depth map into a point cloud."""

    stride: int = 4
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD
    max_range_m: float = DEFAULT_MAX_RANGE_M
    max_height_m: float = 30.0
    max_points: int = 200_000


def resize_long_side(image: npt.NDArray[np.uint8], long_side: int) -> npt.NDArray[np.uint8]:
    """Downscale ``image`` so its longer side is ``long_side``, keeping the aspect ratio.

    Images already at or below the target are returned unchanged — upscaling would
    only invent detail the depth model then has to undo.
    """
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= long_side:
        return image
    scale = long_side / longest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return np.asarray(cv2.resize(image, new_size, interpolation=cv2.INTER_AREA), dtype=np.uint8)


def select_plate_frames(
    frames_rgb: list[npt.NDArray[np.uint8]],
    max_frames: int = _MAX_PLATE_FRAMES,
) -> list[npt.NDArray[np.uint8]]:
    """Evenly spaced subset of frames that will be blended into the plate.

    Exposed separately so callers can downscale *only* these frames and release the
    full-resolution originals before anything expensive runs.

    The spacing spans the whole clip, endpoints included. A simple
    ``frames[::count // max_frames]`` stride collapses to 1 whenever the clip has
    fewer than ``2 * max_frames`` frames, which would median only the clip's opening
    seconds — leaving anything that moves later baked into a supposedly mover-free
    plate, and with it into the depths objects are placed against.
    """
    count = len(frames_rgb)
    if count <= max_frames:
        return list(frames_rgb)
    indices = np.linspace(0, count - 1, max_frames).round().astype(int)
    return [frames_rgb[index] for index in indices]


def background_plate(
    frames_rgb: list[npt.NDArray[np.uint8]],
    *,
    long_side: int,
    max_frames: int = _MAX_PLATE_FRAMES,
) -> npt.NDArray[np.uint8] | None:
    """Median-blend evenly spaced frames into a mover-free background plate.

    Args:
        frames_rgb: ``(H, W, 3)`` uint8 RGB frames in presentation order. Frames
            already at or below ``long_side`` pass through the resize untouched, so
            passing pre-downscaled frames costs nothing.
        long_side: Target long side; frames are downscaled before stacking.
        max_frames: Upper bound on how many frames enter the median.

    Returns:
        The ``(h, w, 3)`` uint8 plate, or ``None`` when ``frames_rgb`` is empty.

    """
    if not frames_rgb:
        return None
    scaled = [resize_long_side(frame, long_side) for frame in select_plate_frames(frames_rgb, max_frames)]
    if len(scaled) == 1:
        return scaled[0]
    target_shape = scaled[0].shape
    stack = np.stack([frame for frame in scaled if frame.shape == target_shape])
    plate: npt.NDArray[np.uint8] = np.median(stack, axis=0).astype(np.uint8)
    return plate


def backproject(
    calib: Calib,
    depth_m: npt.NDArray[np.float32],
    image_rgb: npt.NDArray[np.uint8],
    params: CloudParams,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
    """Back-project a metric depth map into a coloured map-frame point cloud.

    Args:
        calib: Camera model; must match ``depth_m``'s resolution.
        depth_m: ``(H, W)`` metric depth in metres (optical-Z).
        image_rgb: ``(H, W, 3)`` uint8 RGB, the colour source.
        params: Filters and budget.

    Returns:
        ``(points, colours)`` with points ``(N, 3)`` float32 in the map frame and
        colours ``(N, 3)`` uint8 RGB.

    """
    height, width = depth_m.shape[:2]
    # Kept in float32: the result is only compared against a scalar threshold, and
    # promoting a full depth map to float64 costs three large temporaries.
    gradient_y, gradient_x = np.gradient(depth_m)
    relative_gradient = np.hypot(gradient_x, gradient_y) / np.maximum(depth_m, 1e-3)

    rays, flat_u, flat_v = pixel_rays(calib.inverse_k(), width, height, stride=params.stride)
    z = depth_m[flat_v, flat_u].astype(np.float64)
    points_map = calib.camera_to_map((rays * z).T)

    ground_range = np.linalg.norm(points_map[:, :2] - calib.t[:2], axis=1)
    keep = (
        (z > 0)
        & np.isfinite(z)
        & (ground_range <= params.max_range_m)
        & (points_map[:, 2] > _MIN_POINT_HEIGHT_M)
        & (points_map[:, 2] < params.max_height_m)
        & (relative_gradient[flat_v, flat_u] < params.edge_threshold)
    )
    colours = image_rgb[flat_v[keep], flat_u[keep]]
    return points_map[keep].astype(np.float32), np.ascontiguousarray(colours, dtype=np.uint8)


def ground_orthophoto(
    calib: Calib,
    image_rgb: npt.NDArray[np.uint8],
    params: CloudParams,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
    """Ray-cast image pixels onto the ``z = 0`` plane, colouring the ground itself.

    Faithful for a flat dominant ground (road markings stay crisp); anything above
    the plane smears, which is unavoidable from a single view. Kept as an
    alternative to :func:`backproject` for scenes where the depth model's far field
    is less trustworthy than the flat-plane assumption.
    """
    height, width = image_rgb.shape[:2]
    horizon = calib.horizon_v()
    if int(horizon) + 3 >= height:
        empty_points: npt.NDArray[np.float32] = np.zeros((0, 3), dtype=np.float32)
        return empty_points, np.zeros((0, 3), dtype=np.uint8)

    rays, flat_u, flat_v = pixel_rays(calib.inverse_k(), width, height, stride=params.stride, v_start=int(horizon) + 3)
    rays = rays / np.linalg.norm(rays, axis=0)
    directions_map = calib.R @ rays
    direction_z = directions_map[2]
    downward = direction_z < _DOWNWARD_EPSILON
    scale = np.where(downward, -calib.t[2] / np.where(downward, direction_z, -1.0), 0.0)
    points_map = (calib.t[:, None] + scale * directions_map).T
    ground_range = np.linalg.norm(points_map[:, :2] - calib.t[:2], axis=1)
    keep = downward & (scale > 0) & (ground_range <= params.max_range_m)
    colours = image_rgb[flat_v[keep], flat_u[keep]]
    return points_map[keep].astype(np.float32), np.ascontiguousarray(colours, dtype=np.uint8)


def voxel_downsample(
    points: npt.NDArray[np.float32],
    colours: npt.NDArray[np.uint8],
    max_points: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
    """Reduce a cloud to at most ``max_points`` by keeping one point per voxel.

    Deterministic (no RNG) and spatially uniform, unlike a random subsample which
    thins dense near-field detail and sparse far-field detail equally. The voxel
    size is found by bisection on the cloud's bounding box.
    """
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, colours

    extent = points.max(axis=0) - points.min(axis=0)
    largest = float(np.max(extent))
    if largest <= 0:
        return points[:max_points], colours[:max_points]

    origin = points.min(axis=0)
    low, high = largest / _VOXEL_SEARCH_FLOOR, largest
    best: npt.NDArray[np.int64] | None = None
    while high - low >= largest / _VOXEL_SEARCH_EPSILON:
        size = (low + high) / 2.0
        index = _voxel_representatives(points, origin, size)
        if index.size <= max_points:
            best = index
            high = size
        else:
            low = size
    if best is None:
        # Bisection never landed under budget (degenerate geometry); take a
        # deterministic uniform stride instead of returning an oversized cloud.
        step = max(1, points.shape[0] // max_points)
        return points[::step][:max_points], colours[::step][:max_points]
    return points[best], colours[best]


def _voxel_representatives(
    points: npt.NDArray[np.float32],
    origin: npt.NDArray[np.float32],
    size: float,
) -> npt.NDArray[np.int64]:
    """Index of one representative point per occupied voxel.

    Voxel coordinates are folded into a single int64 key so the uniqueness test is a
    1-D sort rather than ``np.unique(..., axis=0)``, which sorts a 3-column view and
    is several times slower for identical output.
    """
    cells = np.floor((points - origin) / size).astype(np.int64)
    dims = cells.max(axis=0) + 1
    flat = (cells[:, 0] * dims[1] + cells[:, 1]) * dims[2] + cells[:, 2]
    _, index = np.unique(flat, return_index=True)
    return index


def pack_point_cloud(
    points: npt.NDArray[np.float32],
    colours: npt.NDArray[np.uint8],
) -> npt.NDArray[np.float32]:
    """Interleave points and colours into the ``foxglove.PointCloud`` byte layout.

    Produces ``(N, 7)`` float32 — ``x, y, z, red, green, blue, alpha`` — with
    colours normalised to ``0..1``. Foxglove's "RGBA (separate fields)" colour mode
    reads this unambiguously and auto-selects it, so no manual panel setting is
    needed. ``N * 28`` bytes, matching :data:`POINT_STRIDE_BYTES`.
    """
    count = points.shape[0]
    packed = np.empty((count, 7), dtype=np.float32)
    if count:
        packed[:, :3] = points
        packed[:, 3:6] = colours.astype(np.float32) / 255.0
        packed[:, 6] = 1.0
    return packed


def sample_depth(depth_m: npt.NDArray[np.float32], u: float, v: float) -> float:
    """Read the depth at a pixel, clamped to the image bounds."""
    height, width = depth_m.shape[:2]
    col = int(min(max(u, 0.0), width - 1))
    row = int(min(max(v, 0.0), height - 1))
    return float(depth_m[row, col])


def place(
    calib: Calib, depth_m: npt.NDArray[np.float32], u: float, v: float
) -> tuple[npt.NDArray[np.float64], float] | None:
    """Lift a pixel to a map-frame point using the metric depth map.

    Returns ``(point, depth_m_at_pixel)`` so callers that also need the range do not
    sample the same pixel twice, or ``None`` when the sample is missing or non-positive.
    """
    z = sample_depth(depth_m, u, v)
    if not np.isfinite(z) or z <= 0:
        return None
    ray = calib.inverse_k() @ np.array([u, v, 1.0])
    point: npt.NDArray[np.float64] = calib.R @ (ray * z) + calib.t
    return point, z
