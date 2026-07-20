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

"""Pure (CPU-only) spatial control-signal extractors for video style transfer.

Cosmos3 transfer follows a per-frame *control signal* derived from the source
clip: the model repaints each frame to match the text prompt while staying
true to that structure. This module implements the two control types that need
no extra model weights:

- ``edge`` -- Canny edge map (object outlines / shapes).
- ``blur`` -- downscale-then-upscale blur (rough color/composition, most
  freedom to restyle detail).

vLLM-Omni's Cosmos3 transfer path can also derive these on-the-fly from a raw
input video (Canny for ``edge``, downscale/upscale for ``blur``). We reimplement
them here so the pre-computed ``control_path`` mode is available and so the
extraction is unit-testable on CPU without loading the model. ``depth`` / ``seg``
are intentionally absent: they depend on DepthAnything / SAM2 and are out of
scope for the first implementation (see the plan's deferred controls).

Functions are stateless and torch-free (OpenCV + numpy only) so they unit-test
on CPU, mirroring ``tracking/track_funcs.py``.
"""

from typing import Literal, get_args

import cv2
import numpy as np
import numpy.typing as npt

ControlType = Literal["edge", "blur"]

# Control types this module can compute on the host without extra model weights.
HOST_COMPUTABLE_CONTROLS: tuple[ControlType, ...] = get_args(ControlType)

# Named presets mirror the cookbook spec fields (``preset_edge_threshold`` /
# ``preset_blur_strength``) so host-side extraction stays visually close to the
# model's own on-the-fly derivation.
_EDGE_THRESHOLD_PRESETS: dict[str, tuple[int, int]] = {
    "low": (50, 150),
    "medium": (100, 200),
    "high": (150, 300),
}
_BLUR_STRENGTH_PRESETS: dict[str, int] = {
    "low": 2,
    "medium": 4,
    "high": 8,
}


def canny_edge_frame(
    rgb: npt.NDArray[np.uint8],
    *,
    preset: str = "medium",
) -> npt.NDArray[np.uint8]:
    """Compute a 3-channel Canny edge map for a single RGB frame.

    Args:
        rgb: Source frame as (H, W, 3) uint8 RGB.
        preset: Threshold preset ('low' | 'medium' | 'high').

    Returns:
        (H, W, 3) uint8 RGB edge map (white edges on black), suitable for
        re-encoding as a control video.

    """
    lo, hi = _EDGE_THRESHOLD_PRESETS.get(preset, _EDGE_THRESHOLD_PRESETS["medium"])
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, lo, hi)
    return np.asarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB), dtype=np.uint8)


def blur_frame(
    rgb: npt.NDArray[np.uint8],
    *,
    preset: str = "medium",
) -> npt.NDArray[np.uint8]:
    """Compute a downscale-then-upscale blur for a single RGB frame.

    Downscaling by ``factor`` and upscaling back is the blur the Cosmos3
    transfer path uses for the ``blur`` hint (it discards fine detail while
    keeping rough color/composition), rather than a Gaussian kernel.

    Args:
        rgb: Source frame as (H, W, 3) uint8 RGB.
        preset: Blur strength preset ('low' | 'medium' | 'high'); larger
            strength downsamples more aggressively.

    Returns:
        (H, W, 3) uint8 RGB blurred frame at the original resolution.

    """
    factor = _BLUR_STRENGTH_PRESETS.get(preset, _BLUR_STRENGTH_PRESETS["medium"])
    height, width = rgb.shape[:2]
    small_w = max(1, width // factor)
    small_h = max(1, height // factor)
    small = cv2.resize(rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return np.asarray(cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR), dtype=np.uint8)


def extract_control_frames(
    frames_rgb: list[npt.NDArray[np.uint8]],
    control: str,
    *,
    preset: str = "medium",
) -> list[npt.NDArray[np.uint8]]:
    """Apply a host-computable control extractor across a list of RGB frames.

    Args:
        frames_rgb: Source frames as a list of (H, W, 3) uint8 RGB arrays.
        control: Control type ('edge' | 'blur').
        preset: Strength/threshold preset passed to the per-frame extractor.

    Returns:
        List of (H, W, 3) uint8 RGB control frames, aligned 1:1 with the input.

    Raises:
        ValueError: If ``control`` is not host-computable.

    """
    if control == "edge":
        return [canny_edge_frame(f, preset=preset) for f in frames_rgb]
    if control == "blur":
        return [blur_frame(f, preset=preset) for f in frames_rgb]
    msg = f"Control '{control}' is not host-computable; choose one of {HOST_COMPUTABLE_CONTROLS}."
    raise ValueError(msg)
