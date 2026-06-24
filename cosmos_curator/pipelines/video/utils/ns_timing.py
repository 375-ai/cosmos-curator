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
"""Nanosecond timing helpers for clip metadata.

Clip and window timing is persisted as integer nanoseconds (see the clip metadata
Lance schema). These helpers convert decoded per-frame PTS, which the pipeline
carries as float seconds (``get_video_timestamps`` / ``Video.timestamps``), into
the integer-nanosecond representation used downstream.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

NS_PER_SECOND = 1_000_000_000


def seconds_to_ns(value_s: float) -> int:
    """Convert a value in seconds to integer nanoseconds, rounded to the nearest ns.

    Args:
        value_s: A time value in seconds (typically a decoded PTS).

    Returns:
        The value in nanoseconds as an ``int``.

    """
    return round(value_s * NS_PER_SECOND)


def frame_pts_bounds_ns(
    pts_ns: Sequence[int] | npt.NDArray[np.integer[Any]],
    start_frame: int,
    end_frame: int,
    *,
    relative_to_first: bool = False,
) -> tuple[int, int]:
    """Map a frame range to nanosecond bounds using per-frame PTS in nanoseconds.

    ``pts_ns`` is the monotonically increasing per-frame PTS (in integer
    nanoseconds) of the video the frames belong to. ``start_frame`` / ``end_frame``
    are inclusive indices into that array; they are clamped to the valid range so
    that window bounds derived from estimates cannot raise.

    Args:
        pts_ns: Per-frame PTS in nanoseconds, monotonically increasing.
        start_frame: Inclusive start frame index.
        end_frame: Inclusive end frame index.
        relative_to_first: When ``True``, subtract the first PTS so the bounds are
            relative to the start of ``pts_ns`` (used for clip-relative window
            bounds). When ``False``, keep the absolute timeline.

    Returns:
        ``(start_ns, end_ns)`` as integer nanoseconds.

    Raises:
        ValueError: If ``pts_ns`` is empty.

    """
    if len(pts_ns) == 0:
        error_msg = "pts_ns must not be empty"
        raise ValueError(error_msg)

    last_index = len(pts_ns) - 1
    start_index = min(max(start_frame, 0), last_index)
    end_index = min(max(end_frame, 0), last_index)

    base = int(pts_ns[0]) if relative_to_first else 0
    return (int(pts_ns[start_index]) - base, int(pts_ns[end_index]) - base)
