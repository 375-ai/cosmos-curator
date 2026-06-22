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

"""Shared fixtures for sensor tests."""

import io
import pathlib
from collections.abc import Callable

import av
import numpy as np
import pytest

# B-frame test files are NOT encoded on the fly. The ffmpeg we ship is the LGPL
# build, whose only H.264 encoder is openh264 -- and openh264 CANNOT emit
# B-frames; the encoder that can, libx264, is GPL and deliberately not bundled.
# So a ``bf=N`` encode option is silently ignored here and yields a B-frame-free
# stream (which is why AUTO would then pick FROM_HEADER and check_video_index
# would report CONSISTENT instead of HEADER_BYPASSED). Instead we read a small
# clip pre-encoded with libx264 + B-frames and checked into the repo -- decoding
# B-frames works in any ffmpeg build; only *encoding* them needs libx264.
_BFRAME_CLIP = (
    pathlib.Path(__file__).resolve().parents[2] / "pipelines" / "video" / "data" / "test_clip_10s_bframes.mp4"
)


@pytest.fixture
def h264_video() -> Callable[..., bytes]:
    """Return a factory for an H.264 MP4 with (``bframes`` > 0) or without B-frames.

    For ``bframes > 0`` it returns the checked-in libx264 B-frame clip (the
    bundled openh264 encoder can't make B-frames here -- see ``_BFRAME_CLIP``);
    the exact count is not significant, only that the stream has B-frames. For
    ``bframes == 0`` it encodes a tiny clip live (any H.264 encoder handles that).
    """

    def _make(*, bframes: int) -> bytes:
        if bframes > 0:
            return _BFRAME_CLIP.read_bytes()
        buffer = io.BytesIO()
        with av.open(buffer, mode="w", format="mp4") as container:
            stream = container.add_stream("h264", rate=30)
            stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
            stream.codec_context.options = {"bf": "0", "g": "30"}
            for i in range(30):
                frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), i, dtype=np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return buffer.getvalue()

    return _make
