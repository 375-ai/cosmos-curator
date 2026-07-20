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

"""CPU tests for the style-transfer control-signal extractors (edge / blur).

These pin the host-computable control helpers: they preserve frame geometry,
produce uint8 RGB, and reject controls that need extra model weights — no GPU
or model load required.
"""

import numpy as np
import pytest

from cosmos_curator.pipelines.video.style_transfer.control_signals import (
    HOST_COMPUTABLE_CONTROLS,
    blur_frame,
    canny_edge_frame,
    extract_control_frames,
)


def _frame(h: int = 48, w: int = 64) -> np.ndarray:
    """Build a frame with a bright rectangle so Canny produces real edges."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[12:36, 16:48] = 200
    return frame


def test_host_computable_controls_are_edge_and_blur() -> None:
    """The first implementation ships exactly the two weight-free controls."""
    assert set(HOST_COMPUTABLE_CONTROLS) == {"edge", "blur"}


def test_canny_edge_frame_preserves_geometry_and_dtype() -> None:
    """Edge map keeps (H, W, 3) uint8 and has non-zero edges on a rectangle."""
    edges = canny_edge_frame(_frame(), preset="medium")
    assert edges.shape == (48, 64, 3)
    assert edges.dtype == np.uint8
    assert edges.max() > 0


def test_blur_frame_preserves_geometry_and_reduces_detail() -> None:
    """Blur keeps geometry/dtype and softens the sharp rectangle edge."""
    src = _frame()
    blurred = blur_frame(src, preset="high")
    assert blurred.shape == src.shape
    assert blurred.dtype == np.uint8
    # A downscale/upscale blur bleeds the hard edge into neighboring pixels,
    # so the blurred frame differs from the source.
    assert not np.array_equal(blurred, src)


@pytest.mark.parametrize("control", ["edge", "blur"])
def test_extract_control_frames_aligns_one_to_one(control: str) -> None:
    """One control frame per input frame, same order/geometry."""
    frames = [_frame(), _frame(), _frame()]
    out = extract_control_frames(frames, control, preset="low")  # type: ignore[arg-type]
    assert len(out) == len(frames)
    assert all(f.shape == (48, 64, 3) for f in out)


def test_extract_control_frames_rejects_non_host_controls() -> None:
    """depth/seg need DepthAnything/SAM2 and must not be silently accepted."""
    with pytest.raises(ValueError, match="not host-computable"):
        extract_control_frames([_frame()], "depth", preset="medium")  # type: ignore[arg-type]
