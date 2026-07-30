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
"""GPS/GNSS data structures for the Sensor Library."""

import enum
from typing import TYPE_CHECKING, Any, Protocol

import attrs
import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.utils.helpers import as_optional_readonly_view, as_readonly_view
from cosmos_curator.core.sensors.utils.validation import (
    bool_batch,
    float64_batch,
    int64_array,
    nondecreasing_int64_array,
    strictly_increasing_int64_array,
    symmetric_psd_covariance_batch,
    uint8_array,
    uint32_array,
    uint64_array,
)

if TYPE_CHECKING:
    AttrsAttribute = attrs.Attribute[Any]
else:
    AttrsAttribute = attrs.Attribute

_VECTOR_COLUMNS = 3
_VECTOR_BATCH_VALIDATOR = float64_batch((_VECTOR_COLUMNS,))
_PER_AXIS_VALIDITY_VALIDATOR = bool_batch((_VECTOR_COLUMNS,))
_OPTIONAL_COVARIANCE_VALIDATOR = attrs.validators.optional(symmetric_psd_covariance_batch(_VECTOR_COLUMNS))
_OPTIONAL_VECTOR_BATCH_VALIDATOR = attrs.validators.optional(_VECTOR_BATCH_VALIDATOR)
_OPTIONAL_PER_AXIS_VALIDITY_VALIDATOR = attrs.validators.optional(_PER_AXIS_VALIDITY_VALIDATOR)
_MIN_LATITUDE_DEG = -90.0
_MAX_LATITUDE_DEG = 90.0
_MIN_LONGITUDE_DEG = -180.0
_MAX_LONGITUDE_DEG = 180.0
_SCALAR_VALIDITY_PAIRS = (
    ("satellites_used", "satellites_used_valid"),
    ("horizontal_accuracy_m", "horizontal_accuracy_m_valid"),
    ("vertical_accuracy_m", "vertical_accuracy_m_valid"),
    ("hdop", "hdop_valid"),
    ("vdop", "vdop_valid"),
    ("pdop", "pdop_valid"),
)


class GpsFixType(enum.IntEnum):
    """Normalized GPS/GNSS fix type values stored in ``GpsData.fix_type``."""

    NO_FIX_OR_UNKNOWN = 0
    FIX_2D = 2
    FIX_3D = 3
    DIFFERENTIAL = 4
    RTK_FLOAT = 5
    RTK_FIXED = 6
    EXTRAPOLATED = 8


_VALID_FIX_TYPES = frozenset(fix_type.value for fix_type in GpsFixType)
_VALID_FIX_TYPES_ARRAY = np.asarray(sorted(_VALID_FIX_TYPES), dtype=np.uint8)


class _HasGpsBatchFields(Protocol):
    align_timestamps_ns: npt.NDArray[np.int64]
    sensor_timestamps_ns: npt.NDArray[np.int64]
    latitude_deg: npt.NDArray[np.float64]
    longitude_deg: npt.NDArray[np.float64]
    altitude_m: npt.NDArray[np.float64]
    position_valid: npt.NDArray[np.bool_]
    position_covariance_enu_m2: npt.NDArray[np.float64] | None
    velocity_enu_m_s: npt.NDArray[np.float64] | None
    velocity_valid: npt.NDArray[np.bool_] | None
    fix_type: npt.NDArray[np.uint8] | None
    satellites_used: npt.NDArray[np.uint32] | None
    satellites_used_valid: npt.NDArray[np.bool_] | None
    horizontal_accuracy_m: npt.NDArray[np.float64] | None
    horizontal_accuracy_m_valid: npt.NDArray[np.bool_] | None
    vertical_accuracy_m: npt.NDArray[np.float64] | None
    vertical_accuracy_m_valid: npt.NDArray[np.bool_] | None
    hdop: npt.NDArray[np.float64] | None
    hdop_valid: npt.NDArray[np.bool_] | None
    vdop: npt.NDArray[np.float64] | None
    vdop_valid: npt.NDArray[np.bool_] | None
    pdop: npt.NDArray[np.float64] | None
    pdop_valid: npt.NDArray[np.bool_] | None
    host_timestamps_ns: npt.NDArray[np.int64] | None
    utc_timestamps_ns: npt.NDArray[np.int64] | None
    sequence_counter: npt.NDArray[np.uint64] | None


def _require_float64_vector(name: str, value: npt.NDArray[np.float64]) -> None:
    """Raise if ``value`` is not a 1-D ``float64`` vector."""
    if value.ndim != 1:
        msg = f"{name} must have shape (N,), got shape={value.shape}"
        raise ValueError(msg)
    if value.dtype != np.float64:
        msg = f"{name} must have dtype float64, got {value.dtype}"
        raise ValueError(msg)


def _latitude_vector(
    _instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.float64],
) -> None:
    """Validate the shape and dtype of WGS-84 latitude values in degrees."""
    _require_float64_vector(attribute.name, value)


def _longitude_vector(
    _instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.float64],
) -> None:
    """Validate the shape and dtype of WGS-84 longitude values in degrees."""
    _require_float64_vector(attribute.name, value)


def _float64_vector(
    _instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.float64],
) -> None:
    """Validate the shape and dtype of a 1-D ``float64`` array."""
    _require_float64_vector(attribute.name, value)


def _optional_scalar_validity_mask(
    _instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.bool_] | None,
) -> None:
    """Validate optional row-level validity masks with shape ``(N,)``."""
    if value is None:
        return
    if value.ndim != 1:
        msg = f"{attribute.name} must have shape (N,), got shape={value.shape}"
        raise ValueError(msg)
    if value.dtype != np.bool_:
        msg = f"{attribute.name} must have dtype bool, got {value.dtype}"
        raise ValueError(msg)


def _optional_uint8_vector(
    instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.uint8] | None,
) -> None:
    """Validate optional 1-D ``uint8`` arrays."""
    if value is None:
        return
    if value.ndim != 1:
        msg = f"{attribute.name} must have shape (N,), got shape={value.shape}"
        raise ValueError(msg)
    uint8_array(instance, attribute, value)


def _optional_fix_type(
    instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.uint8] | None,
) -> None:
    """Validate optional normalized GPS/GNSS fix type arrays."""
    if value is None:
        return
    _optional_uint8_vector(instance, attribute, value)
    if not np.all(np.isin(value, _VALID_FIX_TYPES_ARRAY)):
        msg = f"{attribute.name} must contain only valid fix type values: {sorted(_VALID_FIX_TYPES)}"
        raise ValueError(msg)


def _optional_uint32_vector(
    instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.uint32] | None,
) -> None:
    """Validate optional 1-D ``uint32`` arrays."""
    if value is None:
        return
    if value.ndim != 1:
        msg = f"{attribute.name} must have shape (N,), got shape={value.shape}"
        raise ValueError(msg)
    uint32_array(instance, attribute, value)


def _optional_int64_vector(
    instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.int64] | None,
) -> None:
    """Validate optional 1-D ``int64`` arrays."""
    if value is None:
        return
    if value.ndim != 1:
        msg = f"{attribute.name} must have shape (N,), got shape={value.shape}"
        raise ValueError(msg)
    int64_array(instance, attribute, value)


def _optional_uint64_vector(
    instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.uint64] | None,
) -> None:
    """Validate optional 1-D ``uint64`` arrays."""
    if value is None:
        return
    if value.ndim != 1:
        msg = f"{attribute.name} must have shape (N,), got shape={value.shape}"
        raise ValueError(msg)
    uint64_array(instance, attribute, value)


def _optional_float64_vector(
    _instance: object,
    attribute: AttrsAttribute,
    value: npt.NDArray[np.float64] | None,
) -> None:
    """Validate optional 1-D ``float64`` arrays."""
    if value is None:
        return
    _require_float64_vector(attribute.name, value)


def _scalar_validity_pairs(
    instance: _HasGpsBatchFields,
    _attribute: object,
    _value: object,
) -> None:
    """Require optional scalar arrays and row-level validity masks to be paired."""
    for value_name, validity_name in _SCALAR_VALIDITY_PAIRS:
        value = getattr(instance, value_name)
        validity = getattr(instance, validity_name)
        if (value is None) != (validity is None):
            msg = f"{value_name} and {validity_name} must be provided together or both be None"
            raise ValueError(msg)


def _require_finite_or_marked_invalid(
    name: str,
    values: npt.NDArray[np.float64],
    validity: npt.NDArray[np.bool_] | None,
) -> None:
    """Allow non-finite values only when their matching validity entry is false."""
    nonfinite = ~np.isfinite(values)
    if not np.any(nonfinite):
        return
    if validity is None:
        msg = f"{name} must contain only finite values when no validity mask is provided"
        raise ValueError(msg)
    if np.any(nonfinite & validity):
        msg = f"{name} non-finite values require matching validity mask entries to be false"
        raise ValueError(msg)


def _require_range_or_marked_invalid(
    name: str,
    values: npt.NDArray[np.float64],
    validity: npt.NDArray[np.bool_],
    *,
    minimum: float,
    maximum: float,
) -> None:
    """Allow out-of-range values only when their matching validity entry is false."""
    out_of_range = (values < minimum) | (values > maximum)
    if np.any(out_of_range & validity):
        msg = f"{name} values must be in [{minimum}, {maximum}] when matching validity mask entries are true"
        raise ValueError(msg)


def _require_nonnegative_or_marked_invalid(
    name: str,
    values: npt.NDArray[np.float64],
    validity: npt.NDArray[np.bool_],
) -> None:
    """Allow negative values only when their matching validity entry is false."""
    if np.any((values < 0) & validity):
        msg = f"{name} must contain only nonnegative values when matching validity mask entries are true"
        raise ValueError(msg)


def _raw_value_constraints(
    instance: _HasGpsBatchFields,
    _attribute: object,
    _value: object,
) -> None:
    """Validate raw GPS measurements against their matching validity masks."""
    _require_finite_or_marked_invalid("latitude_deg", instance.latitude_deg, instance.position_valid[:, 0])
    _require_finite_or_marked_invalid("longitude_deg", instance.longitude_deg, instance.position_valid[:, 1])
    _require_finite_or_marked_invalid("altitude_m", instance.altitude_m, instance.position_valid[:, 2])
    _require_range_or_marked_invalid(
        "latitude_deg",
        instance.latitude_deg,
        instance.position_valid[:, 0],
        minimum=_MIN_LATITUDE_DEG,
        maximum=_MAX_LATITUDE_DEG,
    )
    _require_range_or_marked_invalid(
        "longitude_deg",
        instance.longitude_deg,
        instance.position_valid[:, 1],
        minimum=_MIN_LONGITUDE_DEG,
        maximum=_MAX_LONGITUDE_DEG,
    )
    for value_name, validity_name in _SCALAR_VALIDITY_PAIRS:
        values = getattr(instance, value_name)
        validity = getattr(instance, validity_name)
        if values is None or validity is None or value_name == "satellites_used":
            continue
        _require_finite_or_marked_invalid(value_name, values, validity)
        _require_nonnegative_or_marked_invalid(value_name, values, validity)


def _batch_lengths(
    instance: _HasGpsBatchFields,
    _attribute: object,
    _value: object,
) -> None:
    """Validate shared row-count invariants across GPS/GNSS batch arrays."""
    expected_len = len(instance.align_timestamps_ns)
    lengths = {
        "align_timestamps_ns": len(instance.align_timestamps_ns),
        "sensor_timestamps_ns": len(instance.sensor_timestamps_ns),
        "latitude_deg": len(instance.latitude_deg),
        "longitude_deg": len(instance.longitude_deg),
        "altitude_m": len(instance.altitude_m),
        "position_valid": len(instance.position_valid),
    }
    optional_fields = (
        "position_covariance_enu_m2",
        "velocity_enu_m_s",
        "velocity_valid",
        "fix_type",
        "satellites_used",
        "satellites_used_valid",
        "horizontal_accuracy_m",
        "horizontal_accuracy_m_valid",
        "vertical_accuracy_m",
        "vertical_accuracy_m_valid",
        "hdop",
        "hdop_valid",
        "vdop",
        "vdop_valid",
        "pdop",
        "pdop_valid",
        "host_timestamps_ns",
        "utc_timestamps_ns",
        "sequence_counter",
    )
    for field_name in optional_fields:
        field_value = getattr(instance, field_name)
        if field_value is not None:
            lengths[field_name] = len(field_value)
    if any(length != expected_len for length in lengths.values()):
        length_summary = " ".join(f"{name}={length}" for name, length in lengths.items())
        msg = f"All arrays must be the same length: {length_summary}"
        raise ValueError(msg)


@attrs.define(hash=False, frozen=True)
class GpsData:
    """GPS/GNSS fix samples stored as structure-of-arrays batches.

    Satisfies ``SensorData`` (``cosmos_curator.core.sensors.data.sensor_data``).
    Required position fields use WGS-84 geodetic coordinates.
    Optional scalar measurements with per-sample presence must be provided as a
    value array plus matching ``*_valid`` mask, or both fields must be ``None``.
    Measurements retain raw values when their matching validity entries are
    false; valid entries satisfy their documented numeric constraints.
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
    latitude_deg: npt.NDArray[np.float64] = attrs.field(
        converter=as_readonly_view,
        validator=_latitude_vector,
    )
    longitude_deg: npt.NDArray[np.float64] = attrs.field(
        converter=as_readonly_view,
        validator=_longitude_vector,
    )
    altitude_m: npt.NDArray[np.float64] = attrs.field(
        converter=as_readonly_view,
        validator=_float64_vector,
    )
    position_valid: npt.NDArray[np.bool_] = attrs.field(
        converter=as_readonly_view,
        validator=_PER_AXIS_VALIDITY_VALIDATOR,
    )

    position_covariance_enu_m2: npt.NDArray[np.float64] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=_OPTIONAL_COVARIANCE_VALIDATOR,
    )
    velocity_enu_m_s: npt.NDArray[np.float64] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=_OPTIONAL_VECTOR_BATCH_VALIDATOR,
    )
    velocity_valid: npt.NDArray[np.bool_] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=_OPTIONAL_PER_AXIS_VALIDITY_VALIDATOR,
    )
    fix_type: npt.NDArray[np.uint8] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=_optional_fix_type,
    )
    satellites_used: npt.NDArray[np.uint32] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=_optional_uint32_vector,
    )
    satellites_used_valid: npt.NDArray[np.bool_] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=_optional_scalar_validity_mask,
    )
    horizontal_accuracy_m: npt.NDArray[np.float64] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=_optional_float64_vector,
    )
    horizontal_accuracy_m_valid: npt.NDArray[np.bool_] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=_optional_scalar_validity_mask,
    )
    vertical_accuracy_m: npt.NDArray[np.float64] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=_optional_float64_vector,
    )
    vertical_accuracy_m_valid: npt.NDArray[np.bool_] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=_optional_scalar_validity_mask,
    )
    hdop: npt.NDArray[np.float64] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=_optional_float64_vector,
    )
    hdop_valid: npt.NDArray[np.bool_] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=_optional_scalar_validity_mask,
    )
    vdop: npt.NDArray[np.float64] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=_optional_float64_vector,
    )
    vdop_valid: npt.NDArray[np.bool_] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=_optional_scalar_validity_mask,
    )
    pdop: npt.NDArray[np.float64] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=_optional_float64_vector,
    )
    pdop_valid: npt.NDArray[np.bool_] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=_optional_scalar_validity_mask,
    )
    host_timestamps_ns: npt.NDArray[np.int64] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=_optional_int64_vector,
    )
    utc_timestamps_ns: npt.NDArray[np.int64] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=_optional_int64_vector,
    )
    # sequence_counter is final so cross-field validators run after attrs has
    # set and structurally validated every required and optional GPS/GNSS array.
    sequence_counter: npt.NDArray[np.uint64] | None = attrs.field(
        default=None,
        converter=as_optional_readonly_view,
        validator=attrs.validators.and_(
            _optional_uint64_vector,
            _scalar_validity_pairs,
            _batch_lengths,
            _raw_value_constraints,
        ),
    )
