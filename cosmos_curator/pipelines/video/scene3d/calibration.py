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

"""Camera model and self-calibration for monocular 3D scene reconstruction.

Curator has no calibration source: an arbitrary input video carries neither
intrinsics nor a camera pose, and the MCAP writer's ``/camera/camera-info`` and
``/tf-static`` records are documented placeholders. So rather than *requiring*
``K, R, t``, this module *derives* them:

- ``K`` from an assumed horizontal field of view (or an explicit focal length);
- ``R, t`` by RANSAC-fitting a ground plane to the **metric** depth point cloud,
  which yields the camera's height above the ground and its pitch/roll relative
  to it. Yaw is unobservable from one static view and is fixed so the camera
  looks along map ``+X``.

That inversion is what makes the reconstruction generic. It only works because
the depth model predicts absolute metres; with relative depth the plane fit has
no scale and the height falls out arbitrary.

Conventions (matching Foxglove, so the results drop straight into
``foxglove.CameraCalibration`` / ``foxglove.FrameTransform``):

- Camera optical frame: x right, y **down**, z forward.
  ``u = fx * x / z + cx``, ``v = fy * y / z + cy``.
- ``FrameTransform`` gives the child (camera) pose in the parent (map) frame:
  ``p_map = R @ p_cam + t``.
- The map frame is gravity-aligned with the ground at ``z = 0``, so ``t[2]`` is
  the camera height in metres.
"""

import math
from typing import Any, Literal

import attrs
import numpy as np
import numpy.typing as npt

# Camera-frame "up" is -Y (y points down), used to disambiguate the ground normal.
_CAMERA_UP = np.array([0.0, -1.0, 0.0])
_MIN_PLANE_INLIERS = 16
_RANSAC_SEED = 0
# Fixed RANSAC effort: enough hypotheses to find a dominant plane, capped so a dense
# cloud does not make the fit scale with resolution.
_RANSAC_ITERATIONS = 200
_RANSAC_MAX_SAMPLES = 20000
# Physically plausible mounting heights: below is a plane fit to a nearby object,
# above is a degenerate fit to something that is not the ground.
_MIN_CAMERA_HEIGHT_M = 0.3
_MAX_CAMERA_HEIGHT_M = 100.0
# Largest angle between the fitted normal and camera-up that still reads as ground.
# A fronto-parallel wall sits at 90 degrees, a very steep downward view at ~60.
_MAX_GROUND_TILT_DEG = 75.0

# Numerical guards: a ray shorter than this has no usable direction, a ground-ray
# z-component below it is parallel to the plane, and a point closer than the last
# is behind the image plane.
_MIN_RAY_NORM = 1e-12
_MIN_GROUND_RAY_Z = 1e-9
_MIN_FORWARD_Z = 1e-6
MAX_HFOV_DEG = 179.0
MIN_HFOV_DEG = 1.0

DEFAULT_HFOV_DEG = 60.0
DEFAULT_CAMERA_HEIGHT_M = 1.5
DEFAULT_CAMERA_TILT_DEG = 10.0
DEFAULT_CAMERA_ROLL_DEG = 0.0


# Where a camera pose came from. ``ground-fit`` was measured from the depth cloud;
# ``overridden`` was fully specified on the command line, so no fit was attempted;
# ``fallback`` means a fit was attempted and rejected.
CalibrationSource = Literal["ground-fit", "overridden", "fallback"]


@attrs.define(frozen=True, kw_only=True)
class CameraOverrides:
    """Explicit calibration values that win over the automatic estimate.

    Every field defaults to ``None``, meaning "estimate this". ``tilt_deg`` is
    positive downwards; ``roll_deg`` is a rotation about the optical axis.
    """

    focal_px: float | None = None
    hfov_deg: float = DEFAULT_HFOV_DEG
    camera_height_m: float | None = None
    camera_tilt_deg: float | None = None
    camera_roll_deg: float | None = None

    def rescaled_to(self, plate_width: int, source_width: int) -> "CameraOverrides":
        """Convert a source-resolution focal length onto the depth plate's pixel grid.

        ``focal_px`` is a property of the camera, so users give it in the source
        video's pixels while the fit runs on a downscaled plate. This is the inverse
        of :meth:`Calib.to_payload`'s rescale, which takes the estimate back the other
        way. ``hfov_deg`` needs no conversion: an angle is resolution-independent.
        """
        if self.focal_px is None or source_width <= 0 or plate_width == source_width:
            return self
        return attrs.evolve(self, focal_px=self.focal_px * (plate_width / source_width))


@attrs.define(frozen=True, kw_only=True)
class GroundFitParams:
    """Acceptance thresholds for the RANSAC ground-plane fit."""

    inlier_tol_m: float = 0.10
    min_inlier_frac: float = 0.15


@attrs.define(frozen=True)
class Calib:
    """Pinhole camera with a known pose over a gravity-aligned ground plane."""

    K: npt.NDArray[np.float64]
    width: int
    height: int
    R: npt.NDArray[np.float64]
    t: npt.NDArray[np.float64]
    # True when R/t came from an accepted ground-plane fit rather than fallbacks.
    ground_inlier_frac: float = 0.0
    source: CalibrationSource = "ground-fit"
    # Lazily filled on first use; declared so the frozen slotted class has slots.
    _horizon_v: float | None = attrs.field(default=None, init=False, eq=False, repr=False)
    _inverse_k: npt.NDArray[np.float64] | None = attrs.field(default=None, init=False, eq=False, repr=False)

    @property
    def estimated(self) -> bool:
        """True when the pose was measured from the depth cloud."""
        return self.source == "ground-fit"

    @property
    def camera_height_m(self) -> float:
        """Camera height above the ground plane, in metres."""
        return float(self.t[2])

    def inverse_k(self) -> npt.NDArray[np.float64]:
        """Return ``K``-inverse, computed once per camera."""
        cached = self._inverse_k
        if cached is None:
            cached = np.linalg.inv(self.K)
            object.__setattr__(self, "_inverse_k", cached)
        return cached

    def pixel_to_ground(self, u: float, v: float) -> npt.NDArray[np.float64] | None:
        """Project a pixel onto the ``z = 0`` ground plane.

        Args:
            u: Pixel column.
            v: Pixel row.

        Returns:
            ``(x, y, 0)`` in the map frame, or ``None`` when the ray does not
            strike the ground in front of the camera.

        """
        direction_cam = self.inverse_k() @ np.array([u, v, 1.0])
        norm = float(np.linalg.norm(direction_cam))
        if norm < _MIN_RAY_NORM:
            return None
        direction_map = self.R @ (direction_cam / norm)
        if abs(direction_map[2]) < _MIN_GROUND_RAY_Z:
            return None
        scale = -self.t[2] / direction_map[2]
        if scale <= 0:
            return None
        ground: npt.NDArray[np.float64] = self.t + scale * direction_map
        return ground

    def ground_to_pixel(self, point_map: npt.NDArray[np.float64]) -> npt.NDArray[np.float64] | None:
        """Project a map-frame point to pixel coordinates, or ``None`` if behind the camera."""
        point_cam = self.R.T @ (np.asarray(point_map, dtype=np.float64) - self.t)
        if point_cam[2] <= _MIN_FORWARD_Z:
            return None
        uvw = self.K @ point_cam
        return np.array([uvw[0] / uvw[2], uvw[1] / uvw[2]])

    def camera_to_map(self, points_cam: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Transform an ``(N, 3)`` camera-frame point array into the map frame."""
        return np.asarray(points_cam, dtype=np.float64) @ self.R.T + self.t

    def horizon_v(self) -> float:
        """Image row where ground rays become parallel to the plane.

        Bisects the centre column for the last row that still yields a ground hit.
        Cached on the instance because every consumer needs it and the search is
        pure overhead when repeated.
        """
        if self._horizon_v is not None:
            return self._horizon_v
        low, high = 0.0, float(self.height)
        for _ in range(40):
            mid = (low + high) / 2
            if self.pixel_to_ground(self.width / 2, mid) is None:
                low = mid
            else:
                high = mid
        object.__setattr__(self, "_horizon_v", high)
        return high

    def rotation_quaternion(self) -> tuple[float, float, float, float]:
        """Return ``R`` as an ``(x, y, z, w)`` quaternion for ``foxglove.FrameTransform``."""
        return rotation_to_quaternion(self.R)

    def to_payload(self, *, width: int | None = None, height: int | None = None) -> dict[str, Any]:
        """Serialize to the JSON-friendly dict carried on ``Clip.scene3d_calibration``.

        The camera model is estimated at the depth map's resolution but published
        alongside the full-resolution video on the same ``frame_id``, so
        ``width``/``height`` rescale the intrinsics to the image this calibration
        will be paired with. Without that, a viewer sees a calibration whose size and
        focal length disagree with the image stream. Pose is resolution-independent
        and is emitted unchanged.
        """
        out_width = int(width) if width else int(self.width)
        out_height = int(height) if height else int(self.height)
        scale_x = out_width / self.width
        scale_y = out_height / self.height
        fx = float(self.K[0, 0]) * scale_x
        fy = float(self.K[1, 1]) * scale_y
        cx = float(self.K[0, 2]) * scale_x
        cy = float(self.K[1, 2]) * scale_y
        return {
            "width": out_width,
            "height": out_height,
            "K": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
            "D": [0.0, 0.0, 0.0, 0.0, 0.0],
            "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "P": [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
            "translation": [float(value) for value in self.t],
            "rotation": list(self.rotation_quaternion()),
            "estimated": self.estimated,
            "ground_inlier_frac": float(self.ground_inlier_frac),
            "source": str(self.source),
        }


def rotation_to_quaternion(matrix: npt.NDArray[np.float64]) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to an ``(x, y, z, w)`` quaternion."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def intrinsics_from_fov(
    width: int, height: int, *, hfov_deg: float, focal_px: float | None = None
) -> npt.NDArray[np.float64]:
    """Build ``K`` from an assumed horizontal FOV, or from an explicit focal length.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        hfov_deg: Horizontal field of view in degrees; ignored when ``focal_px`` is given.
        focal_px: Explicit focal length in pixels.

    Returns:
        A 3x3 intrinsics matrix with a centred principal point and square pixels.

    """
    if focal_px is not None and focal_px > 0:
        focal = float(focal_px)
    else:
        if not MIN_HFOV_DEG < hfov_deg < MAX_HFOV_DEG:
            msg = f"hfov_deg must be in ({MIN_HFOV_DEG}, {MAX_HFOV_DEG}), got {hfov_deg}"
            raise ValueError(msg)
        focal = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )


def pixel_rays(
    inverse_k: npt.NDArray[np.float64],
    width: int,
    height: int,
    *,
    stride: int = 1,
    v_start: int = 0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Build camera-frame ray directions for a strided pixel grid.

    Shared by every back-projection in the package, which otherwise each rebuild the
    same meshgrid and ``K^-1 @ [u, v, 1]`` scaffolding. The rays are NOT normalised:
    their z-component is 1, so scaling one by an optical-Z depth lands the point
    directly (``P_cam = Z * ray``).

    Args:
        inverse_k: ``K``-inverse.
        width: Image width in pixels.
        height: Image height in pixels.
        stride: Pixel sampling stride.
        v_start: First row to sample, for callers that only want part of the image.

    Returns:
        ``(rays, us, vs)`` with ``rays`` shaped ``(3, N)`` and ``us``/``vs`` the
        sampled pixel coordinates.

    """
    grid_u: npt.NDArray[np.int64]
    grid_v: npt.NDArray[np.int64]
    grid_u, grid_v = np.meshgrid(np.arange(0, width, stride), np.arange(v_start, height, stride))
    flat_u = grid_u.ravel()
    flat_v = grid_v.ravel()
    rays = inverse_k @ np.stack(
        [flat_u.astype(np.float64), flat_v.astype(np.float64), np.ones(flat_u.size, dtype=np.float64)]
    )
    return rays, flat_u.astype(np.int64), flat_v.astype(np.int64)


def fit_ground_plane(
    points_cam: npt.NDArray[np.float64],
    params: GroundFitParams,
) -> tuple[npt.NDArray[np.float64], float, float] | None:
    """RANSAC-fit a plane to camera-frame points.

    Args:
        points_cam: ``(N, 3)`` candidate ground points in the camera frame.
        params: Acceptance thresholds.

    Returns:
        ``(unit_normal, offset, inlier_frac)`` such that ``normal . p + offset = 0``,
        with the normal oriented so it points away from the ground (camera-frame
        "up"). ``None`` when no plane clears the thresholds.

    """
    finite = points_cam[np.isfinite(points_cam).all(axis=1) & (points_cam[:, 2] > 0)]
    if finite.shape[0] < _MIN_PLANE_INLIERS:
        return None

    rng = np.random.default_rng(_RANSAC_SEED)
    if finite.shape[0] > _RANSAC_MAX_SAMPLES:
        finite = finite[rng.choice(finite.shape[0], _RANSAC_MAX_SAMPLES, replace=False)]

    total = finite.shape[0]
    best_inliers: npt.NDArray[np.bool_] | None = None
    best_count = 0
    for _ in range(_RANSAC_ITERATIONS):
        idx = rng.choice(total, 3, replace=False)
        a, b, c = finite[idx]
        normal = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(normal))
        if norm < _MIN_GROUND_RAY_Z:
            continue
        normal = normal / norm
        offset = -float(normal @ a)
        distances = np.abs(finite @ normal + offset)
        inliers = distances <= params.inlier_tol_m
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    inlier_frac = best_count / total
    if best_inliers is None or best_count < _MIN_PLANE_INLIERS or inlier_frac < params.min_inlier_frac:
        return None

    # Refit on the consensus set: the 3-point hypothesis is only a seed.
    consensus = finite[best_inliers]
    centroid = consensus.mean(axis=0)
    _, _, vt = np.linalg.svd(consensus - centroid, full_matrices=False)
    normal = vt[2]
    normal = normal / float(np.linalg.norm(normal))
    offset = -float(normal @ centroid)

    # Point the normal "up" in camera terms (camera +Y is down).
    if float(normal @ _CAMERA_UP) < 0:
        normal = -normal
        offset = -offset
    return normal, offset, inlier_frac


def rotation_from_ground_normal(normal_cam: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Build the camera->map rotation that maps ``normal_cam`` onto map ``+Z``.

    Map ``+X`` is the camera's forward axis projected onto the ground plane, so
    the camera always looks along ``+X``; ``+Y`` completes the right-handed frame.
    Yaw is therefore fixed by construction — a single static view cannot observe it.
    """
    up = normal_cam / float(np.linalg.norm(normal_cam))
    forward_cam = np.array([0.0, 0.0, 1.0])
    forward = forward_cam - float(forward_cam @ up) * up
    if float(np.linalg.norm(forward)) < _MIN_FORWARD_Z:
        # Camera looks straight down the normal: pick any in-plane axis.
        fallback = np.array([1.0, 0.0, 0.0])
        forward = fallback - float(fallback @ up) * up
    forward = forward / float(np.linalg.norm(forward))
    left = np.cross(up, forward)
    # Rows are the map-frame axes expressed in camera coordinates, which IS the
    # camera->map rotation: (R @ p_cam)[0] projects p_cam onto map +X. Do not
    # transpose this -- doing so inverts every pose in the pipeline.
    return np.stack([forward, left, up])


def rotation_from_angles(*, tilt_deg: float, roll_deg: float) -> npt.NDArray[np.float64]:
    """Build a camera->map rotation from a downward tilt and an optical-axis roll.

    Used for the fallback path and for CLI overrides. At ``tilt_deg = 0`` the
    camera looks along map ``+X`` with its optical axis horizontal.
    """
    tilt = math.radians(tilt_deg)
    roll = math.radians(roll_deg)
    # Ground normal in the camera frame for a camera pitched `tilt` downwards.
    normal = np.array([0.0, -math.cos(tilt), -math.sin(tilt)])
    rotation = rotation_from_ground_normal(normal)
    if abs(roll) < _MIN_RAY_NORM:
        return rotation
    cos_r, sin_r = math.cos(roll), math.sin(roll)
    roll_cam = np.array([[cos_r, -sin_r, 0.0], [sin_r, cos_r, 0.0], [0.0, 0.0, 1.0]])
    rolled: npt.NDArray[np.float64] = rotation @ roll_cam
    return rolled


def fallback_calibration(
    width: int,
    height: int,
    overrides: CameraOverrides,
    source: CalibrationSource = "fallback",
) -> Calib:
    """Build a calibration purely from CLI values (or their defaults).

    ``source`` distinguishes a pose the user fully specified (``overridden``) from
    one reached because the ground fit was rejected (``fallback``); only the latter
    is a failure worth reporting.
    """
    k = intrinsics_from_fov(width, height, hfov_deg=overrides.hfov_deg, focal_px=overrides.focal_px)
    tilt = overrides.camera_tilt_deg if overrides.camera_tilt_deg is not None else DEFAULT_CAMERA_TILT_DEG
    roll = overrides.camera_roll_deg if overrides.camera_roll_deg is not None else DEFAULT_CAMERA_ROLL_DEG
    height_m = overrides.camera_height_m if overrides.camera_height_m is not None else DEFAULT_CAMERA_HEIGHT_M
    rotation = rotation_from_angles(tilt_deg=tilt, roll_deg=roll)
    return Calib(
        K=k,
        width=width,
        height=height,
        R=rotation,
        t=np.array([0.0, 0.0, float(height_m)]),
        ground_inlier_frac=0.0,
        source=source,
    )


def estimate_calibration(
    depth_m: npt.NDArray[np.float32],
    *,
    overrides: CameraOverrides | None = None,
    ground_params: GroundFitParams | None = None,
    stride: int = 4,
) -> Calib:
    """Recover ``K, R, t`` from a metric depth map.

    ``K`` comes from the assumed FOV (or an explicit focal length). ``R`` and
    ``t`` come from a RANSAC ground-plane fit over the lower half of the image,
    which is where the ground is for any camera that is not upside down. On a
    failed or implausible fit the result falls back to
    :func:`fallback_calibration` with ``estimated=False`` so callers can report it.

    Args:
        depth_m: ``(H, W)`` metric depth in metres.
        overrides: Explicit values that take precedence over the estimate.
        ground_params: RANSAC acceptance thresholds.
        stride: Pixel sampling stride for the plane fit.

    Returns:
        The recovered :class:`Calib`.

    """
    overrides = overrides or CameraOverrides()
    ground_params = ground_params or GroundFitParams()
    height, width = depth_m.shape[:2]
    k = intrinsics_from_fov(width, height, hfov_deg=overrides.hfov_deg, focal_px=overrides.focal_px)

    # Explicit pose overrides make the fit moot; skip straight to the fallback.
    fully_overridden = (
        overrides.camera_height_m is not None
        and overrides.camera_tilt_deg is not None
        and overrides.camera_roll_deg is not None
    )
    if fully_overridden:
        return fallback_calibration(width, height, overrides, source="overridden")

    # The ground is in the lower half for any camera that is not upside down;
    # true image coordinates are kept so the rays stay consistent with K.
    points_cam = _unproject_lower_half(depth_m, np.linalg.inv(k), stride=stride)

    fit = fit_ground_plane(points_cam, ground_params)
    if fit is None:
        return fallback_calibration(width, height, overrides)

    normal, offset, inlier_frac = fit
    estimated_height = abs(offset)
    if not _MIN_CAMERA_HEIGHT_M <= estimated_height <= _MAX_CAMERA_HEIGHT_M:
        return fallback_calibration(width, height, overrides)
    # Reject planes that are really walls: their normal is near-perpendicular to
    # camera-up, which would yield a well-formed but meaningless "ground" frame.
    tilt_from_up = math.degrees(math.acos(max(-1.0, min(1.0, float(normal @ _CAMERA_UP)))))
    if tilt_from_up > _MAX_GROUND_TILT_DEG:
        return fallback_calibration(width, height, overrides)

    rotation = rotation_from_ground_normal(normal)
    if overrides.camera_tilt_deg is not None or overrides.camera_roll_deg is not None:
        # Overriding one angle must not silently zero the other: each falls back to
        # what the accepted ground fit measured, not to a default.
        tilt = overrides.camera_tilt_deg if overrides.camera_tilt_deg is not None else _tilt_from_normal(normal)
        roll = overrides.camera_roll_deg if overrides.camera_roll_deg is not None else _roll_from_normal(normal)
        rotation = rotation_from_angles(tilt_deg=tilt, roll_deg=roll)
    camera_height = overrides.camera_height_m if overrides.camera_height_m is not None else estimated_height

    return Calib(
        K=k,
        width=width,
        height=height,
        R=rotation,
        t=np.array([0.0, 0.0, float(camera_height)]),
        ground_inlier_frac=float(inlier_frac),
    )


def _unproject_lower_half(
    depth_m: npt.NDArray[np.float32],
    inverse_k: npt.NDArray[np.float64],
    *,
    stride: int,
) -> npt.NDArray[np.float64]:
    """Back-project only the lower half of the image, keeping true pixel coordinates."""
    height, width = depth_m.shape[:2]
    rays, flat_u, flat_v = pixel_rays(inverse_k, width, height, stride=stride, v_start=height // 2)
    return (rays * depth_m[flat_v, flat_u].astype(np.float64)).T


def _tilt_from_normal(normal_cam: npt.NDArray[np.float64]) -> float:
    """Downward tilt in degrees implied by a camera-frame ground normal."""
    unit = normal_cam / float(np.linalg.norm(normal_cam))
    return float(math.degrees(math.asin(max(-1.0, min(1.0, -unit[2])))))


def _roll_from_normal(normal_cam: npt.NDArray[np.float64]) -> float:
    """Optical-axis roll in degrees implied by a camera-frame ground normal.

    Inverts :func:`rotation_from_angles`, which places the normal at
    ``(-sin(roll)cos(tilt), -cos(roll)cos(tilt), -sin(tilt))``.
    """
    unit = normal_cam / float(np.linalg.norm(normal_cam))
    return float(math.degrees(math.atan2(-unit[0], -unit[1])))
