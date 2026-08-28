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
"""Tests for camera self-calibration from a metric depth map."""

import math
from typing import cast

import attrs
import numpy as np
import numpy.testing as npt
import pytest

from cosmos_curator.pipelines.video.scene3d.calibration import (
    Calib,
    CalibrationSource,
    CameraOverrides,
    GroundFitParams,
    estimate_calibration,
    intrinsics_from_fov,
    rotation_from_angles,
    rotation_to_quaternion,
)
from tests.cosmos_curator.pipelines.video.scene3d.scene_fixtures import (
    HEIGHT,
    HFOV_DEG,
    WIDTH,
    make_calib,
    render_ground_depth,
)


def _tilt_of(calib: Calib) -> float:
    """Recover the downward tilt in degrees from a camera->map rotation."""
    normal_cam = calib.R.T @ np.array([0.0, 0.0, 1.0])
    return math.degrees(math.asin(max(-1.0, min(1.0, -normal_cam[2]))))


def test_intrinsics_from_fov_matches_the_pinhole_relation() -> None:
    """A 60-degree HFOV over 640 px gives the textbook focal length."""
    k = intrinsics_from_fov(WIDTH, HEIGHT, hfov_deg=60.0)
    expected = (WIDTH / 2) / math.tan(math.radians(60.0) / 2)
    assert k[0, 0] == pytest.approx(expected)
    assert k[1, 1] == pytest.approx(expected)
    assert (k[0, 2], k[1, 2]) == (WIDTH / 2, HEIGHT / 2)


def test_intrinsics_prefers_an_explicit_focal_length() -> None:
    """--scene3d-focal-px overrides the FOV assumption."""
    k = intrinsics_from_fov(WIDTH, HEIGHT, hfov_deg=60.0, focal_px=1234.0)
    assert k[0, 0] == 1234.0


def test_intrinsics_rejects_an_impossible_fov() -> None:
    """A degenerate FOV fails loudly rather than producing a silent bad scale."""
    with pytest.raises(ValueError, match="hfov_deg must be in"):
        intrinsics_from_fov(WIDTH, HEIGHT, hfov_deg=200.0)


@pytest.mark.parametrize(
    ("camera_height_m", "tilt_deg"),
    [(8.0, 20.0), (1.6, 8.0), (15.3, 17.5), (3.0, 35.0)],
)
def test_estimate_recovers_the_true_camera_pose(camera_height_m: float, tilt_deg: float) -> None:
    """The ground fit recovers height and tilt from an exact ground-plane depth map."""
    truth = make_calib(camera_height_m=camera_height_m, tilt_deg=tilt_deg)
    estimate = estimate_calibration(render_ground_depth(truth), overrides=CameraOverrides(hfov_deg=HFOV_DEG))

    assert estimate.estimated
    assert estimate.ground_inlier_frac > 0.5
    assert estimate.camera_height_m == pytest.approx(camera_height_m, abs=0.05)
    assert _tilt_of(estimate) == pytest.approx(tilt_deg, abs=0.5)


def test_estimate_falls_back_when_no_plane_exists() -> None:
    """Unstructured depth yields no ground, so the CLI defaults take over."""
    rng = np.random.default_rng(0)
    noise = rng.uniform(1.0, 80.0, (HEIGHT, WIDTH)).astype(np.float32)

    estimate = estimate_calibration(noise, overrides=CameraOverrides())

    assert not estimate.estimated
    assert estimate.source == "fallback"
    assert estimate.camera_height_m == pytest.approx(1.5)
    assert _tilt_of(estimate) == pytest.approx(10.0, abs=1e-6)


def test_estimate_rejects_a_wall_as_ground() -> None:
    """A fronto-parallel surface fits a plane, but its normal is not up."""
    wall = np.full((HEIGHT, WIDTH), 12.0, dtype=np.float32)

    estimate = estimate_calibration(wall, overrides=CameraOverrides())

    assert not estimate.estimated


def test_estimate_falls_back_on_an_implausible_height() -> None:
    """A plane 500 m below the camera is not a mounting height."""
    truth = make_calib(camera_height_m=8.0, tilt_deg=20.0)
    estimate = estimate_calibration(
        render_ground_depth(truth, sky_depth_m=120.0) * 60.0,
        overrides=CameraOverrides(hfov_deg=HFOV_DEG),
    )
    assert not estimate.estimated


def test_cli_overrides_beat_the_estimate() -> None:
    """Explicit height/tilt/roll skip the fit entirely."""
    truth = make_calib(camera_height_m=8.0, tilt_deg=20.0)
    depth = render_ground_depth(truth)

    estimate = estimate_calibration(
        depth,
        overrides=CameraOverrides(hfov_deg=HFOV_DEG, camera_height_m=2.5, camera_tilt_deg=30.0, camera_roll_deg=0.0),
    )

    # A user-specified pose is not a failure, so it is `overridden`, not `fallback`.
    assert estimate.source == "overridden"
    assert not estimate.estimated
    assert estimate.camera_height_m == pytest.approx(2.5)
    assert _tilt_of(estimate) == pytest.approx(30.0)


def test_partial_override_keeps_the_estimated_tilt() -> None:
    """Pinning only the height still uses the fitted orientation."""
    truth = make_calib(camera_height_m=8.0, tilt_deg=20.0)

    estimate = estimate_calibration(
        render_ground_depth(truth),
        overrides=CameraOverrides(hfov_deg=HFOV_DEG, camera_height_m=9.0),
    )

    assert estimate.estimated
    assert estimate.camera_height_m == pytest.approx(9.0)
    assert _tilt_of(estimate) == pytest.approx(20.0, abs=0.5)


def test_estimate_needs_enough_inliers() -> None:
    """Raising the inlier floor above what the scene offers forces the fallback."""
    truth = make_calib(camera_height_m=8.0, tilt_deg=2.0)
    depth = render_ground_depth(truth)

    strict = estimate_calibration(
        depth,
        overrides=CameraOverrides(hfov_deg=HFOV_DEG),
        ground_params=GroundFitParams(min_inlier_frac=0.999, inlier_tol_m=0.001),
    )
    assert not strict.estimated


def test_pixel_to_ground_round_trips(calib: Calib) -> None:
    """Projecting a pixel to the ground and back returns the same pixel."""
    for u, v in [(WIDTH / 2, HEIGHT - 1), (10.0, HEIGHT - 20), (WIDTH - 10.0, HEIGHT * 0.8)]:
        ground = calib.pixel_to_ground(u, v)
        assert ground is not None
        assert ground[2] == pytest.approx(0.0, abs=1e-9)
        pixel = calib.ground_to_pixel(ground)
        assert pixel is not None
        npt.assert_allclose(pixel, [u, v], atol=1e-6)


def test_pixel_to_ground_returns_none_above_the_horizon() -> None:
    """A ray that never meets the plane has no ground point."""
    shallow = make_calib(camera_height_m=8.0, tilt_deg=2.0)
    assert shallow.pixel_to_ground(WIDTH / 2, 0.0) is None


def test_horizon_matches_the_analytic_row() -> None:
    """The bisection lands on fy*tan(-tilt) + cy, clamped to the frame."""
    shallow = make_calib(camera_height_m=8.0, tilt_deg=8.0)
    expected = shallow.K[1, 1] * math.tan(math.radians(-8.0)) + shallow.K[1, 2]
    assert shallow.horizon_v() == pytest.approx(expected, abs=0.01)
    # Tilted far enough down, the horizon leaves the frame and clamps to row 0.
    assert make_calib(tilt_deg=45.0).horizon_v() == pytest.approx(0.0, abs=0.01)


def test_horizon_is_cached(calib: Calib) -> None:
    """The 40-step bisection runs once per camera, not once per consumer."""
    first = calib.horizon_v()
    assert calib._horizon_v is not None
    assert calib.horizon_v() == first


def test_rotation_to_quaternion_round_trips() -> None:
    """Quaternion conversion preserves the rotation it describes."""
    for tilt, roll in [(0.0, 0.0), (20.0, 0.0), (35.0, 12.0), (5.0, -30.0)]:
        rotation = rotation_from_angles(tilt_deg=tilt, roll_deg=roll)
        x, y, z, w = rotation_to_quaternion(rotation)
        npt.assert_allclose(np.linalg.norm([x, y, z, w]), 1.0, atol=1e-9)
        rebuilt = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        npt.assert_allclose(rebuilt, rotation, atol=1e-9)


def test_payload_matches_the_mcap_calibration_shape(calib: Calib) -> None:
    """to_payload() emits exactly the fields the MCAP writer consumes."""
    payload = calib.to_payload()
    assert len(payload["K"]) == 9
    assert len(payload["R"]) == 9
    assert len(payload["P"]) == 12
    assert len(payload["D"]) == 5
    assert len(payload["translation"]) == 3
    assert len(payload["rotation"]) == 4
    assert payload["translation"][2] == pytest.approx(8.0)
    assert (payload["width"], payload["height"]) == (WIDTH, HEIGHT)
    # P is K with a zero fourth column, per the ROS/Foxglove convention.
    assert payload["P"][0] == payload["K"][0]
    assert payload["P"][3] == 0.0


def _roll_of(calib: Calib) -> float:
    """Recover the optical-axis roll in degrees from a camera->map rotation."""
    normal_cam = calib.R.T @ np.array([0.0, 0.0, 1.0])
    return math.degrees(math.atan2(-normal_cam[0], -normal_cam[1]))


def test_overriding_tilt_keeps_the_fitted_roll() -> None:
    """Pinning one angle must not silently zero the other on a rolled camera."""
    truth = make_calib(camera_height_m=6.0, tilt_deg=18.0, roll_deg=12.0)
    depth = render_ground_depth(truth)

    estimate = estimate_calibration(depth, overrides=CameraOverrides(hfov_deg=HFOV_DEG, camera_tilt_deg=25.0))

    assert estimate.estimated
    assert _tilt_of(estimate) == pytest.approx(25.0, abs=0.5)
    assert _roll_of(estimate) == pytest.approx(12.0, abs=0.5)


def test_overriding_roll_keeps_the_fitted_tilt() -> None:
    """The symmetric case: an explicit roll must not discard the measured tilt."""
    truth = make_calib(camera_height_m=6.0, tilt_deg=18.0, roll_deg=12.0)

    estimate = estimate_calibration(
        render_ground_depth(truth), overrides=CameraOverrides(hfov_deg=HFOV_DEG, camera_roll_deg=0.0)
    )

    assert _tilt_of(estimate) == pytest.approx(18.0, abs=0.5)
    assert _roll_of(estimate) == pytest.approx(0.0, abs=0.5)


def test_accepted_fit_reports_ground_fit_as_its_source() -> None:
    """Provenance distinguishes a measured pose from a told one and a guessed one."""
    truth = make_calib()
    assert (
        estimate_calibration(render_ground_depth(truth), overrides=CameraOverrides(hfov_deg=HFOV_DEG)).source
        == "ground-fit"
    )


def test_payload_rescales_intrinsics_to_the_published_image_size(calib: Calib) -> None:
    """Intrinsics are estimated on the depth plate but published beside the video.

    The calibration and ``/camera/image-raw`` share a frame_id, so a viewer pairs
    them; emitting plate-resolution intrinsics next to a full-resolution image makes
    the projection wrong by the downscale factor.
    """
    native = calib.to_payload()
    scaled = calib.to_payload(width=WIDTH * 3, height=HEIGHT * 3)

    assert (scaled["width"], scaled["height"]) == (WIDTH * 3, HEIGHT * 3)
    assert scaled["K"][0] == pytest.approx(native["K"][0] * 3)
    assert scaled["K"][4] == pytest.approx(native["K"][4] * 3)
    assert scaled["K"][2] == pytest.approx(native["K"][2] * 3)
    assert scaled["K"][5] == pytest.approx(native["K"][5] * 3)
    # P mirrors K, and the pose is resolution-independent.
    assert scaled["P"][0] == pytest.approx(scaled["K"][0])
    assert scaled["P"][2] == pytest.approx(scaled["K"][2])
    assert scaled["translation"] == native["translation"]
    assert scaled["rotation"] == native["rotation"]


def test_payload_reprojection_is_consistent_after_rescaling(calib: Calib) -> None:
    """A rescaled calibration projects the same world point to the same relative spot."""
    point = np.array([25.0, 4.0, 0.0])
    pixel = calib.ground_to_pixel(point)
    assert pixel is not None

    scaled = calib.to_payload(width=WIDTH * 2, height=HEIGHT * 2)
    k = np.asarray(scaled["K"], dtype=np.float64).reshape(3, 3)
    point_cam = calib.R.T @ (point - calib.t)
    uvw = k @ point_cam
    npt.assert_allclose([uvw[0] / uvw[2], uvw[1] / uvw[2]], [pixel[0] * 2, pixel[1] * 2], rtol=1e-9)


def test_focal_override_rescales_between_source_and_plate() -> None:
    """A source-resolution focal length round-trips through the plate unchanged.

    ``rescaled_to`` converts a camera property (source pixels) onto the downscaled
    grid the fit runs on; ``to_payload`` converts the estimate back. The two are
    inverses, so what the user typed is what gets published.
    """
    overrides = CameraOverrides(focal_px=1200.0)
    plate = overrides.rescaled_to(700, 1920)

    assert plate.focal_px == pytest.approx(1200.0 * 700 / 1920)
    calib = Calib(
        K=intrinsics_from_fov(700, 394, hfov_deg=60.0, focal_px=plate.focal_px),
        width=700,
        height=394,
        R=rotation_from_angles(tilt_deg=10.0, roll_deg=0.0),
        t=np.array([0.0, 0.0, 5.0]),
    )
    assert calib.to_payload(width=1920, height=1080)["K"][0] == pytest.approx(1200.0)


def test_focal_override_is_a_noop_without_a_focal_length() -> None:
    """An hfov-only override has nothing resolution-dependent to convert."""
    overrides = CameraOverrides(hfov_deg=72.0)
    assert overrides.rescaled_to(700, 1920) is overrides
    # Nor when the plate was not downscaled at all.
    with_focal = CameraOverrides(focal_px=800.0)
    assert with_focal.rescaled_to(1920, 1920) is with_focal


def test_estimated_is_derived_from_source() -> None:
    """One field decides provenance; `estimated` cannot drift out of step with it."""
    base = make_calib()
    for source, expected in [("ground-fit", True), ("overridden", False), ("fallback", False)]:
        calib = attrs.evolve(base, source=cast("CalibrationSource", source))
        assert calib.estimated is expected
        assert calib.to_payload()["estimated"] is expected


def test_inverse_k_is_cached(calib: Calib) -> None:
    """K-inverse is computed once per camera; `place` calls it once per detection."""
    first = calib.inverse_k()
    assert calib.inverse_k() is first
