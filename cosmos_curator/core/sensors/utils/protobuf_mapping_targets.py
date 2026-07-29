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
"""Public target-row contracts for GPS and IMU protobuf mappings."""

from dataclasses import dataclass

REQUIRED_GPS_MAPPING_FIELDS = frozenset(
    {
        "sensor_timestamp_ns",
        "latitude_deg",
        "longitude_deg",
        "altitude_m",
    }
)
REQUIRED_IMU_MAPPING_FIELDS = frozenset(
    {
        "sensor_timestamp_ns",
        "angular_velocity_rad_s",
        "linear_acceleration_m_s2",
    }
)


@dataclass(frozen=True)
class GpsMappingRow:
    """One GPS row produced by a generic protobuf mapping."""

    sensor_timestamp_ns: int
    align_timestamp_ns: int
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    position_valid: tuple[bool, bool, bool] = (True, True, True)
    host_timestamp_ns: int | None = None
    hdop: float = 0.0
    vdop: float = 0.0
    pdop: float = 0.0
    horizontal_accuracy_m: float = 0.0
    vertical_accuracy_m: float = 0.0
    satellites_used: int = 0
    hdop_valid: bool = False
    vdop_valid: bool = False
    pdop_valid: bool = False
    horizontal_accuracy_m_valid: bool = False
    vertical_accuracy_m_valid: bool = False
    satellites_used_valid: bool = False


@dataclass(frozen=True)
class ImuMappingRow:
    """One IMU row produced by a generic protobuf mapping."""

    sensor_timestamp_ns: int
    align_timestamp_ns: int
    angular_velocity_rad_s: tuple[float, float, float]
    linear_acceleration_m_s2: tuple[float, float, float]
    angular_velocity_valid: tuple[bool, bool, bool] = (True, True, True)
    linear_acceleration_valid: tuple[bool, bool, bool] = (True, True, True)
    angular_velocity_bias_rad_s: tuple[float, float, float] | None = None
    linear_acceleration_bias_m_s2: tuple[float, float, float] | None = None
    angular_velocity_bias_valid: tuple[bool, bool, bool] | None = None
    linear_acceleration_bias_valid: tuple[bool, bool, bool] | None = None
    host_timestamp_ns: int | None = None
    sequence_counter: int | None = None
    temperature_c: float | None = None
    temperature_valid: bool | None = None
