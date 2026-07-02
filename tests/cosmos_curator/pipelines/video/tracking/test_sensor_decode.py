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

"""Tests for the sensor-library decode helper used by the tracking stages.

These exercise the core decode contract: every sampled frame carries its
*real* presentation timestamp (PTS), and the native ``frame_idx`` maps back to
the clip's true display order, so time is never recomputed as ``frame_idx / fps``.
"""

import io
from fractions import Fraction

import av
import numpy as np
import pytest

from cosmos_curator.core.sensors.sensors.camera_sensor import CameraSensor
from cosmos_curator.pipelines.video.tracking.sensor_decode import decode_clip_at_fps

_NS_PER_SECOND = 1_000_000_000


def _encode_clip(pts_ms: list[int], *, width: int = 64, height: int = 64) -> bytes:
    """Encode an all-intra H.264 mp4 whose frames sit at the given PTS (ms).

    ``bf=0`` + ``g=1`` keeps the stream B-frame-free and fully keyframed so the
    sensor library reads exact display timestamps from the header. Frame pixel
    values increase per frame so decode order is visually checkable.
    """
    time_base = Fraction(1, 1000)
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("h264", rate=30)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.codec_context.options = {"bf": "0", "g": "1"}
        stream.codec_context.time_base = time_base
        for i, pts in enumerate(pts_ms):
            value = (i * 20) % 256
            frame = av.VideoFrame.from_ndarray(np.full((height, width, 3), value, dtype=np.uint8), format="rgb24")
            frame.pts = pts
            frame.time_base = time_base
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return buffer.getvalue()


def _real_relative_times_s(mp4_bytes: bytes) -> list[float]:
    """Ground-truth per-frame display times (s, relative to first frame, ms-rounded)."""
    sensor = CameraSensor(bytes(mp4_bytes))
    display_pts_ns = sensor.timestamps_ns
    start_ns = int(display_pts_ns[0])
    return [round((int(p) - start_ns) / _NS_PER_SECOND, 3) for p in display_pts_ns]


def test_timestamps_are_real_pts() -> None:
    """Each decoded ``timestamp_s`` is a real frame PTS, not a synthetic grid time."""
    # Deliberately uneven (variable-frame-rate) spacing.
    pts_ms = [0, 50, 175, 200, 360, 400, 600, 850, 900, 1000]
    mp4_bytes = _encode_clip(pts_ms)
    ref_times = set(_real_relative_times_s(mp4_bytes))

    decoded = decode_clip_at_fps(mp4_bytes, target_fps=5.0)

    assert decoded.frames_rgb, "expected at least one decoded frame"
    assert len(decoded.timestamps_s) == len(decoded.frames_rgb)
    # Every reported time is an actual frame PTS (member of the display set),
    # not an evenly spaced grid time.
    for ts in decoded.timestamps_s:
        assert ts in ref_times


def test_decoded_order_is_monotonic() -> None:
    """Decoded timestamps are strictly increasing."""
    pts_ms = [0, 100, 250, 400, 600, 900]
    decoded = decode_clip_at_fps(_encode_clip(pts_ms), target_fps=4.0)

    assert decoded.timestamps_s == sorted(decoded.timestamps_s)
    assert all(b > a for a, b in zip(decoded.timestamps_s, decoded.timestamps_s[1:], strict=False))


def test_vfr_timestamps_are_not_uniform_grid() -> None:
    """On a VFR clip the real timestamps drift from a uniform ``idx / fps`` grid.

    This is the regression the sensor decode fixes: the legacy
    ``frame_idx / fps`` path would have produced an evenly spaced grid, hiding
    the true frame times.
    """
    pts_ms = [0, 40, 230, 260, 470, 520, 730, 980]
    target_fps = 4.0
    decoded = decode_clip_at_fps(_encode_clip(pts_ms), target_fps=target_fps)

    # A uniform-grid assumption would place frame k at exactly k / fps.
    uniform = [round(k / target_fps, 3) for k in range(len(decoded.timestamps_s))]
    assert decoded.timestamps_s != uniform


def test_geometry_reported() -> None:
    """Frame geometry is surfaced for downstream encoding."""
    decoded = decode_clip_at_fps(_encode_clip([0, 100, 200, 300], width=80, height=48), target_fps=10.0)
    assert decoded.width == 80
    assert decoded.height == 48
    assert all(frame.shape == (48, 80, 3) for frame in decoded.frames_rgb)


@pytest.mark.parametrize("target_fps", [2.0, 5.0])
def test_frame_count_within_bounds(target_fps: float) -> None:
    """Sampling never returns more frames than the source has."""
    pts_ms = list(range(0, 1000, 50))  # 20 frames over ~1s
    decoded = decode_clip_at_fps(_encode_clip(pts_ms), target_fps=target_fps)
    assert 0 < len(decoded.frames_rgb) <= len(pts_ms)
