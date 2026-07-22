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
"""Trajectory data structures for cosmos_curator.core.sensors package.

Trajectories are sensor-agnostic — the same rig-to-global pose stream that
supports LiDAR motion compensation also supports radar and ultrasonic sensors.
"""

from typing import TYPE_CHECKING, Any, Protocol

import attrs
import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.utils.helpers import as_readonly_view
from cosmos_curator.core.sensors.utils.validation import (
    nondecreasing_int64_array,
    nonempty_str,
    strictly_increasing_int64_array,
)

if TYPE_CHECKING:
    AttrsAttribute = attrs.Attribute[Any]
else:
    AttrsAttribute = attrs.Attribute

_POSES_TAIL_SHAPE = (4, 4)
_POSES_BATCH_NDIM = 3
_POSES_LAST_ROW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
_POSES_LAST_ROW_TOLERANCE = 1e-9


class _HasEgoBatchFields(Protocol):
    align_timestamps_ns: npt.NDArray[np.int64]
    sensor_timestamps_ns: npt.NDArray[np.int64]
    poses: npt.NDArray[np.float64]
    frame: str


def _poses_batch(
    _instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.float64],
) -> None:
    """Validate an ``(N, 4, 4)`` ``float64`` pose batch with a homogeneous last row."""
    if value.dtype != np.float64:
        msg = f"{attribute.name} must have dtype float64, got {value.dtype}"
        raise ValueError(msg)
    if value.ndim != _POSES_BATCH_NDIM or value.shape[1:] != _POSES_TAIL_SHAPE:
        msg = f"{attribute.name} must have shape (N, 4, 4), got shape={value.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(value)):
        msg = f"{attribute.name} must contain only finite values"
        raise ValueError(msg)
    if value.shape[0] and not np.allclose(
        value[:, 3, :],
        _POSES_LAST_ROW,
        rtol=0.0,
        atol=_POSES_LAST_ROW_TOLERANCE,
    ):
        msg = f"{attribute.name} last row of each (4, 4) must equal [0, 0, 0, 1] within tolerance"
        raise ValueError(msg)


def _ego_batch_lengths(
    instance: _HasEgoBatchFields,
    _attribute: object,
    _value: object,
) -> None:
    """Validate the shared row-count ``N`` invariant for :class:`EgoTrajectory`."""
    expected_len = len(instance.align_timestamps_ns)
    lengths = {
        "align_timestamps_ns": len(instance.align_timestamps_ns),
        "sensor_timestamps_ns": len(instance.sensor_timestamps_ns),
        "poses": len(instance.poses),
    }
    if any(length != expected_len for length in lengths.values()):
        length_summary = " ".join(f"{name}={length}" for name, length in lengths.items())
        msg = f"All arrays must be the same length: {length_summary}"
        raise ValueError(msg)


@attrs.define(hash=False, frozen=True)
class EgoTrajectory:
    """Rig-to-global ego pose sampled on an alignment timeline.

    Satisfies ``SensorData`` (``cosmos_curator.core.sensors.data.sensor_data``).
    """

    __hash__ = None  # type: ignore[assignment]

    align_timestamps_ns: npt.NDArray[np.int64] = attrs.field(
        converter=as_readonly_view,
        validator=strictly_increasing_int64_array,
    )
    sensor_timestamps_ns: npt.NDArray[np.int64] = attrs.field(
        converter=as_readonly_view,
        validator=nondecreasing_int64_array,
    )
    poses: npt.NDArray[np.float64] = attrs.field(
        converter=as_readonly_view,
        validator=_poses_batch,
    )
    # frame is the last-declared field; its non-empty-string check chains with _ego_batch_lengths.
    frame: str = attrs.field(
        validator=attrs.validators.and_(nonempty_str, _ego_batch_lengths),
    )
