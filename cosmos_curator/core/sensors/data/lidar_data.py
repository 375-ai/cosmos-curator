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
"""LiDAR data structures for cosmos_curator.core.sensors package."""

from typing import TYPE_CHECKING, Any, Protocol

import attrs
import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.data.extrinsics import SensorExtrinsics
from cosmos_curator.core.sensors.utils.helpers import as_readonly_view
from cosmos_curator.core.sensors.utils.validation import (
    finite_float32_point_batch,
    nondecreasing_int64_array,
    nonempty_str,
    optional_bool_array,
    optional_finite_float32_vector,
    optional_nonempty_str,
    optional_uint8_array,
    optional_uint16_array,
    strictly_increasing_int64_array,
)

if TYPE_CHECKING:
    AttrsAttribute = attrs.Attribute[Any]
else:
    AttrsAttribute = attrs.Attribute

_REFERENCE_FRAME_UNCOMPENSATED = frozenset({"sensor", "rig"})
_REFERENCE_FRAME_COMPENSATED = frozenset({"rig", "world", "map", "odom"})


class _HasLidarBatchFields(Protocol):
    align_timestamps_ns: npt.NDArray[np.int64]
    sensor_timestamps_ns: npt.NDArray[np.int64]
    points_xyz: npt.NDArray[np.float32]
    points_timestamps_ns: npt.NDArray[np.int64]
    metadata: "LidarMetadata"
    points_intensity: npt.NDArray[np.float32] | None
    points_ring: npt.NDArray[np.uint16] | None
    points_return_index: npt.NDArray[np.uint8] | None
    points_reflectivity: npt.NDArray[np.uint16] | None
    points_ambient: npt.NDArray[np.uint16] | None
    points_validity: npt.NDArray[np.bool_] | None
    points_radial_velocity: npt.NDArray[np.float32] | None
    points_sweep_index: npt.NDArray[np.uint16] | None
    points_align_index: npt.NDArray[np.uint16] | None


def _as_optional_readonly_view(array: npt.NDArray[Any] | None) -> npt.NDArray[Any] | None:
    """Return a read-only view of ``array`` while preserving ``None``."""
    if array is None:
        return None
    return as_readonly_view(array)


def _optional_extrinsics(
    _instance: object,
    attribute: AttrsAttribute,
    value: object,
) -> None:
    """Validate optional ``SensorExtrinsics`` values (attrs boundary; ``value`` is untyped input)."""
    if value is None:
        return
    if not isinstance(value, SensorExtrinsics):
        msg = f"{attribute.name} must be a SensorExtrinsics instance, got {type(value).__name__}"
        raise ValueError(msg)  # noqa: TRY004 -- ValueError matches project convention for invalid attrs input


def _metadata_cross_fields(
    instance: "LidarMetadata",
    _attribute: object,
    _value: object,
) -> None:
    """Validate ``motion_compensated`` / ``reference_frame`` cross-field consistency."""
    if instance.motion_compensated:
        if instance.reference_frame not in _REFERENCE_FRAME_COMPENSATED:
            msg = (
                "reference_frame must be one of "
                f"{sorted(_REFERENCE_FRAME_COMPENSATED)!r} when motion_compensated=True, "
                f"got {instance.reference_frame!r}"
            )
            raise ValueError(msg)
    elif instance.reference_frame not in _REFERENCE_FRAME_UNCOMPENSATED:
        msg = (
            "reference_frame must be one of "
            f"{sorted(_REFERENCE_FRAME_UNCOMPENSATED)!r} when motion_compensated=False, "
            f"got {instance.reference_frame!r}"
        )
        raise ValueError(msg)


def _lidar_batch_lengths(
    instance: _HasLidarBatchFields,
    _attribute: object,
    _value: object,
) -> None:
    """Validate shared row-count ``N`` and point-count ``P_total`` invariants."""
    row_len = len(instance.align_timestamps_ns)
    row_lengths = {
        "align_timestamps_ns": len(instance.align_timestamps_ns),
        "sensor_timestamps_ns": len(instance.sensor_timestamps_ns),
    }
    if any(length != row_len for length in row_lengths.values()):
        length_summary = " ".join(f"{name}={length}" for name, length in row_lengths.items())
        msg = f"All row-level arrays must be the same length: {length_summary}"
        raise ValueError(msg)

    point_len = len(instance.points_xyz)
    point_lengths = {
        "points_xyz": len(instance.points_xyz),
        "points_timestamps_ns": len(instance.points_timestamps_ns),
    }
    optional_point_fields = (
        "points_intensity",
        "points_ring",
        "points_return_index",
        "points_reflectivity",
        "points_ambient",
        "points_validity",
        "points_radial_velocity",
        "points_sweep_index",
        "points_align_index",
    )
    for field_name in optional_point_fields:
        field_value = getattr(instance, field_name)
        if field_value is not None:
            point_lengths[field_name] = len(field_value)
    if any(length != point_len for length in point_lengths.values()):
        length_summary = " ".join(f"{name}={length}" for name, length in point_lengths.items())
        msg = f"All per-point arrays must be the same length: {length_summary}"
        raise ValueError(msg)

    align_index = instance.points_align_index
    # points_align_index is uint16 (validated earlier), so entries are always >= 0.
    if align_index is not None and align_index.size and align_index.max() >= row_len:
        msg = f"points_align_index entries must satisfy v < N (N={row_len}), got max={align_index.max()}"
        raise ValueError(msg)


@attrs.define(hash=False, frozen=True)
class LidarMetadata:
    """Batch-level LiDAR metadata: motion-compensation state, reference frame, and calibration."""

    __hash__ = None  # type: ignore[assignment]

    motion_compensated: bool = attrs.field(validator=attrs.validators.instance_of(bool))
    # reference_frame is the last-declared required field; its non-empty-string check chains with
    # _metadata_cross_fields so the motion_compensated/reference_frame invariant is enforced.
    reference_frame: str = attrs.field(
        validator=attrs.validators.and_(nonempty_str, _metadata_cross_fields),
    )

    extrinsics: SensorExtrinsics | None = attrs.field(
        default=None,
        validator=_optional_extrinsics,
    )
    sensor_model: str | None = attrs.field(
        default=None,
        validator=optional_nonempty_str,
    )


@attrs.define(hash=False, frozen=True)
class LidarData:
    """LiDAR points stored as structure-of-arrays batches.

    Satisfies ``SensorData`` (``cosmos_curator.core.sensors.data.sensor_data``).
    Has two independent batch dimensions: ``N`` (alignment rows, matched to the
    ``SensorData`` protocol timestamp arrays) and ``P_total`` (points). When
    ``metadata.motion_compensated`` is ``True`` each ``align_timestamps_ns[i]``
    also serves as the per-row motion-compensation reference instant.
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
    points_xyz: npt.NDArray[np.float32] = attrs.field(
        converter=as_readonly_view,
        validator=finite_float32_point_batch,
    )
    points_timestamps_ns: npt.NDArray[np.int64] = attrs.field(
        converter=as_readonly_view,
        validator=nondecreasing_int64_array,
    )
    metadata: LidarMetadata = attrs.field(
        validator=attrs.validators.instance_of(LidarMetadata),
    )

    points_intensity: npt.NDArray[np.float32] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=optional_finite_float32_vector,
    )
    points_ring: npt.NDArray[np.uint16] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=optional_uint16_array,
    )
    points_return_index: npt.NDArray[np.uint8] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=optional_uint8_array,
    )
    points_reflectivity: npt.NDArray[np.uint16] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=optional_uint16_array,
    )
    points_ambient: npt.NDArray[np.uint16] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=optional_uint16_array,
    )
    points_validity: npt.NDArray[np.bool_] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=optional_bool_array,
    )
    points_radial_velocity: npt.NDArray[np.float32] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=optional_finite_float32_vector,
    )
    points_sweep_index: npt.NDArray[np.uint16] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=optional_uint16_array,
    )
    # points_align_index is the last optional field; its _as_optional_readonly_view converter runs
    # before validators. attrs.validators.and_ then runs optional_uint16_array and
    # _lidar_batch_lengths (which also range-checks points_align_index against N when present).
    points_align_index: npt.NDArray[np.uint16] | None = attrs.field(
        default=None,
        converter=_as_optional_readonly_view,
        validator=attrs.validators.and_(
            optional_uint16_array,
            _lidar_batch_lengths,
        ),
    )
