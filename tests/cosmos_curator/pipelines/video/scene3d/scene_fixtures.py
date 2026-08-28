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
"""Synthetic-scene builders shared by the 3D reconstruction tests.

Every test here builds a scene analytically -- a known camera looking at a known
ground plane -- so the geometry can be checked against exact expected values
rather than against a model's output.
"""

import numpy as np
import numpy.typing as npt

from cosmos_curator.pipelines.video.scene3d.calibration import (
    Calib,
    intrinsics_from_fov,
    rotation_from_angles,
)

WIDTH = 640
HEIGHT = 360
HFOV_DEG = 60.0


def make_calib(*, camera_height_m: float = 8.0, tilt_deg: float = 20.0, roll_deg: float = 0.0) -> Calib:
    """Build a camera at a known height and tilt above the z=0 ground plane."""
    return Calib(
        K=intrinsics_from_fov(WIDTH, HEIGHT, hfov_deg=HFOV_DEG),
        width=WIDTH,
        height=HEIGHT,
        R=rotation_from_angles(tilt_deg=tilt_deg, roll_deg=roll_deg),
        t=np.array([0.0, 0.0, camera_height_m]),
    )


def render_ground_depth(calib: Calib, *, sky_depth_m: float = 120.0) -> npt.NDArray[np.float32]:
    """Render the exact optical-Z depth of the ground plane for every pixel.

    Rays that miss the ground (above the horizon) are filled with ``sky_depth_m``,
    which is what a real depth model does at its saturation ceiling.
    """
    us, vs = np.meshgrid(np.arange(WIDTH, dtype=np.float64), np.arange(HEIGHT, dtype=np.float64))
    rays = calib.inverse_k() @ np.stack([us.ravel(), vs.ravel(), np.ones(us.size)])
    direction_z = (calib.R @ rays)[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        # The ray's own z-component is 1 by construction, so the scale factor that
        # lands it on the plane *is* the optical-Z depth.
        depth = np.where(direction_z < -1e-9, -calib.t[2] / direction_z, np.nan)
    depth = depth.reshape(HEIGHT, WIDTH)
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, sky_depth_m)
    return np.clip(depth, 0.1, sky_depth_m).astype(np.float32)
