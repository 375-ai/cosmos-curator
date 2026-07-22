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
"""Tests for sensor-library ``EgoTrajectory``."""

from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.sensors.data.aligned_frame import AlignedFrame
from cosmos_curator.core.sensors.data.sensor_data import SensorData
from cosmos_curator.core.sensors.data.trajectory_data import EgoTrajectory


def _make_ego_trajectory(**overrides: object) -> EgoTrajectory:
    """Build a minimal valid EgoTrajectory (N=2)."""
    values: dict[str, object] = {
        "align_timestamps_ns": np.array([100, 200], dtype=np.int64),
        "sensor_timestamps_ns": np.array([90, 210], dtype=np.int64),
        "poses": np.tile(np.eye(4, dtype=np.float64), (2, 1, 1)),
        "frame": "world",
    }
    values.update(overrides)
    return EgoTrajectory(**values)


def test_ego_trajectory_satisfies_sensor_data_protocol() -> None:
    """EgoTrajectory should be structurally usable anywhere SensorData is expected."""
    sensor_data: SensorData = _make_ego_trajectory()

    np.testing.assert_array_equal(sensor_data.align_timestamps_ns, np.array([100, 200], dtype=np.int64))
    np.testing.assert_array_equal(sensor_data.sensor_timestamps_ns, np.array([90, 210], dtype=np.int64))


def test_aligned_frame_rejects_mismatched_ego_trajectory_reference_timeline() -> None:
    """AlignedFrame should reject EgoTrajectory sampled on a different reference timeline."""
    ego = _make_ego_trajectory(align_timestamps_ns=np.array([100, 300], dtype=np.int64))

    with pytest.raises(ValueError, match="align_timestamps_ns must exactly match"):
        AlignedFrame(
            align_timestamps_ns=np.array([100, 200], dtype=np.int64),
            sensor_data={"ego": cast("SensorData", ego)},
        )


def test_ego_trajectory_arrays_are_readonly() -> None:
    """EgoTrajectory should expose read-only NumPy arrays for every field."""
    ego = _make_ego_trajectory()

    for array in (ego.align_timestamps_ns, ego.sensor_timestamps_ns, ego.poses):
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = array.flat[0]


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("poses", np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)), "dtype float64"),
        ("poses", np.zeros((2, 3, 3), dtype=np.float64), r"shape \(N, 4, 4\)"),
        ("poses", np.zeros((2, 4, 4), dtype=np.float64), r"last row of each \(4, 4\)"),
        ("poses", np.full((2, 4, 4), np.nan, dtype=np.float64), "finite"),
        ("frame", "", "frame must be a non-empty string"),
    ],
)
def test_ego_trajectory_rejects_invalid_required_fields(
    field_name: str,
    value: object,
    match: str,
) -> None:
    """EgoTrajectory should validate poses and frame."""
    with pytest.raises(ValueError, match=match):
        _make_ego_trajectory(**{field_name: value})


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("sensor_timestamps_ns", np.array([90, 210, 300], dtype=np.int64)),
        ("poses", np.tile(np.eye(4, dtype=np.float64), (3, 1, 1))),
    ],
)
def test_ego_trajectory_rejects_length_mismatch(
    field_name: str,
    value: npt.NDArray[Any],
) -> None:
    """All EgoTrajectory arrays must share the row-count ``N``."""
    with pytest.raises(ValueError, match="same length"):
        _make_ego_trajectory(**{field_name: value})
