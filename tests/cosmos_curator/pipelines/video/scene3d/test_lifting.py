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
"""Tests for the pixels-to-metric-points helpers."""

import numpy as np
import numpy.testing as npt
import pytest

from cosmos_curator.pipelines.video.read_write.mcap_schemas import POINT_CLOUD_STRIDE_BYTES
from cosmos_curator.pipelines.video.scene3d import lifting
from cosmos_curator.pipelines.video.scene3d.calibration import Calib
from tests.cosmos_curator.pipelines.video.scene3d.scene_fixtures import HEIGHT, WIDTH


def _rgb(value: int = 128) -> np.ndarray:
    return np.full((HEIGHT, WIDTH, 3), value, dtype=np.uint8)


def test_resize_long_side_preserves_aspect_and_never_upscales() -> None:
    """Downscaling keeps the aspect ratio; an already-small frame is untouched."""
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    resized = lifting.resize_long_side(image, 320)
    assert resized.shape == (180, 320, 3)
    # Upscaling would only invent detail the depth model has to undo.
    assert lifting.resize_long_side(image, 1280) is image


def test_background_plate_erases_movers() -> None:
    """A temporal median removes an object that never occupies the same place twice."""
    frames = []
    for index in range(9):
        frame = np.full((64, 128, 3), 40, dtype=np.uint8)
        frame[20:40, index * 10 : index * 10 + 10] = 240  # a bright square sweeping across
        frames.append(frame)

    plate = lifting.background_plate(frames, long_side=128)

    assert plate is not None
    assert plate.shape == (64, 128, 3)
    # The mover occupies each column in a minority of frames, so the median drops it.
    assert plate.max() == 40


def test_background_plate_downscales_before_stacking() -> None:
    """The plate comes out at the depth input size, which is what bounds memory."""
    frames = [np.zeros((360, 640, 3), dtype=np.uint8) for _ in range(4)]
    plate = lifting.background_plate(frames, long_side=200)
    assert plate is not None
    assert max(plate.shape[:2]) == 200


def test_background_plate_handles_empty_and_single_frame() -> None:
    """No frames means no plate; one frame is its own plate."""
    assert lifting.background_plate([], long_side=128) is None
    single = np.full((16, 16, 3), 7, dtype=np.uint8)
    plate = lifting.background_plate([single], long_side=128)
    assert plate is not None
    npt.assert_array_equal(plate, single)


def test_backproject_places_ground_pixels_on_the_plane(calib: Calib, ground_depth: np.ndarray) -> None:
    """An exact ground depth map back-projects to points at z ~ 0."""
    params = lifting.CloudParams(stride=8, max_range_m=100.0)
    points, colours = lifting.backproject(calib, ground_depth, _rgb(), params)

    assert points.shape[0] == colours.shape[0]
    assert points.shape[0] > 0
    horizon = int(calib.horizon_v())
    if horizon < HEIGHT - 40:
        # Points sampled below the horizon are genuine ground; sky pixels sit at the
        # saturation ceiling and are culled by max_range_m instead.
        ground_points = points[points[:, 2] < 1.0]
        npt.assert_allclose(ground_points[:, 2], 0.0, atol=1e-3)


def test_backproject_drops_flying_pixels_at_a_depth_step() -> None:
    """A hard depth discontinuity produces samples that are culled, not smeared."""
    calib = Calib(
        K=np.array([[300.0, 0.0, 64.0], [0.0, 300.0, 32.0], [0.0, 0.0, 1.0]]),
        width=128,
        height=64,
        R=np.eye(3),
        t=np.zeros(3),
    )
    depth = np.full((64, 128), 10.0, dtype=np.float32)
    depth[:, 64:] = 40.0  # a cliff down the middle

    permissive = lifting.CloudParams(stride=1, edge_threshold=10.0, max_range_m=1000.0, max_height_m=1000.0)
    strict = lifting.CloudParams(stride=1, edge_threshold=0.06, max_range_m=1000.0, max_height_m=1000.0)

    many, _ = lifting.backproject(calib, depth, _rgb()[:64, :128], permissive)
    few, _ = lifting.backproject(calib, depth, _rgb()[:64, :128], strict)
    assert few.shape[0] < many.shape[0]


def test_ground_orthophoto_lands_every_point_on_z_zero(calib: Calib) -> None:
    """The flat-plane backdrop is exactly planar by construction."""
    params = lifting.CloudParams(stride=8, max_range_m=100.0)
    points, colours = lifting.ground_orthophoto(calib, _rgb(), params)

    assert points.shape[0] > 0
    assert points.shape[0] == colours.shape[0]
    npt.assert_allclose(points[:, 2], 0.0, atol=1e-6)


def test_ground_orthophoto_is_empty_when_the_horizon_fills_the_frame() -> None:
    """A camera looking up has no ground rows to sample."""
    upward = Calib(
        K=np.array([[300.0, 0.0, 64.0], [0.0, 300.0, 32.0], [0.0, 0.0, 1.0]]),
        width=128,
        height=64,
        R=np.eye(3),
        t=np.array([0.0, 0.0, 5.0]),
    )
    points, colours = lifting.ground_orthophoto(upward, _rgb()[:64, :128], lifting.CloudParams())
    assert points.shape == (0, 3)
    assert colours.shape == (0, 3)


def test_voxel_downsample_respects_the_budget_and_is_deterministic() -> None:
    """The cloud lands under budget, and the same input always gives the same output."""
    rng = np.random.default_rng(0)
    points = rng.uniform(-50, 50, (50_000, 3)).astype(np.float32)
    colours = rng.integers(0, 255, (50_000, 3)).astype(np.uint8)

    first_points, first_colours = lifting.voxel_downsample(points, colours, 5_000)
    second_points, _ = lifting.voxel_downsample(points, colours, 5_000)

    assert first_points.shape[0] <= 5_000
    assert first_points.shape[0] == first_colours.shape[0]
    npt.assert_array_equal(first_points, second_points)


def test_voxel_downsample_is_a_noop_under_budget() -> None:
    """A small cloud passes through untouched."""
    points = np.zeros((10, 3), dtype=np.float32)
    colours = np.zeros((10, 3), dtype=np.uint8)
    out_points, out_colours = lifting.voxel_downsample(points, colours, 5_000)
    assert out_points is points
    assert out_colours is colours


def test_voxel_downsample_handles_a_degenerate_cloud() -> None:
    """Coincident points have no extent to bisect, so a uniform stride is used."""
    points = np.zeros((100, 3), dtype=np.float32)
    colours = np.zeros((100, 3), dtype=np.uint8)
    out_points, _ = lifting.voxel_downsample(points, colours, 10)
    assert out_points.shape[0] <= 10


def test_pack_point_cloud_matches_the_wire_layout() -> None:
    """Packing yields (N, 7) float32 with colours normalised to 0..1."""
    points = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    colours = np.array([[255, 128, 0]], dtype=np.uint8)

    packed = lifting.pack_point_cloud(points, colours)

    assert packed.shape == (1, 7)
    assert packed.dtype == np.float32
    # The packed row must match the stride the writer puts on the wire.
    assert packed.nbytes == POINT_CLOUD_STRIDE_BYTES
    npt.assert_allclose(packed[0, :3], [1.0, 2.0, 3.0])
    npt.assert_allclose(packed[0, 3:], [1.0, 128 / 255, 0.0, 1.0], atol=1e-6)


def test_pack_point_cloud_handles_an_empty_cloud() -> None:
    """An empty reconstruction packs to an empty, still well-shaped array."""
    packed = lifting.pack_point_cloud(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8))
    assert packed.shape == (0, 7)


def test_place_inverts_the_projection(calib: Calib, ground_depth: np.ndarray) -> None:
    """Lifting a ground pixel returns the point that projects back to it."""
    u, v = WIDTH / 2, HEIGHT - 5
    placed = lifting.place(calib, ground_depth, u, v)
    assert placed is not None
    point, depth_z = placed
    # place() also hands back the depth it sampled, so callers need not re-read it.
    assert depth_z == pytest.approx(lifting.sample_depth(ground_depth, u, v))
    assert point[2] == pytest.approx(0.0, abs=1e-3)
    pixel = calib.ground_to_pixel(point)
    assert pixel is not None
    npt.assert_allclose(pixel, [u, v], atol=1e-3)


def test_place_rejects_missing_depth(calib: Calib) -> None:
    """A non-positive or non-finite depth sample yields no point."""
    depth = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    assert lifting.place(calib, depth, 10.0, 10.0) is None
    depth[:] = np.nan
    assert lifting.place(calib, depth, 10.0, 10.0) is None


def test_sample_depth_clamps_to_the_image() -> None:
    """Out-of-bounds pixels read the nearest edge rather than raising."""
    depth = np.arange(HEIGHT * WIDTH, dtype=np.float32).reshape(HEIGHT, WIDTH)
    assert lifting.sample_depth(depth, -50.0, -50.0) == depth[0, 0]
    assert lifting.sample_depth(depth, WIDTH + 50.0, HEIGHT + 50.0) == depth[-1, -1]


@pytest.mark.parametrize("count", [21, 25, 30, 39, 40, 50, 100, 601])
def test_select_plate_frames_spans_the_whole_clip(count: int) -> None:
    """The plate must span the clip at every length, not just multiples of the budget.

    A ``frames[::count // max_frames]`` stride collapses to 1 below ``2 * max_frames``
    and returns only the opening frames — at 39 frames it covered just the first half,
    so anything moving in the back half survived the median and polluted the plate.
    """
    frames = [np.full((4, 4, 3), index % 256, dtype=np.uint8) for index in range(count)]

    selected = lifting.select_plate_frames(frames, max_frames=20)

    assert len(selected) == 20
    # Compare identity, not pixel value, so the assertion holds past 256 frames.
    positions = [next(i for i, f in enumerate(frames) if f is frame) for frame in selected]
    assert positions == sorted(positions)
    assert positions[0] == 0
    assert positions[-1] == count - 1


def test_select_plate_frames_handles_fewer_frames_than_the_budget() -> None:
    """A short clip contributes every frame it has."""
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
    assert len(lifting.select_plate_frames(frames, max_frames=20)) == 3


def test_background_plate_accepts_pre_downscaled_frames() -> None:
    """Pre-downscaled frames pass through the resize untouched.

    That is what lets the stage downscale once up front and release the
    full-resolution originals before anything expensive runs.
    """
    frames = [np.full((100, 200, 3), 30, dtype=np.uint8) for _ in range(5)]

    plate = lifting.background_plate(frames, long_side=700)

    assert plate is not None
    assert plate.shape == (100, 200, 3)


def test_voxel_downsample_matches_the_row_wise_reference() -> None:
    """The scalar voxel key selects the same representatives as np.unique(axis=0)."""
    rng = np.random.default_rng(1)
    points = rng.uniform(-20, 20, (20_000, 3)).astype(np.float32)
    origin = points.min(axis=0)
    size = float(np.max(points.max(axis=0) - origin)) / 64

    scalar = lifting._voxel_representatives(points, origin, size)
    cells = np.floor((points - origin) / size).astype(np.int64)
    _, reference = np.unique(cells, axis=0, return_index=True)

    npt.assert_array_equal(np.sort(scalar), np.sort(reference))
