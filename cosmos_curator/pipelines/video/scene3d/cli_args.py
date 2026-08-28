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

"""Shared argparse wiring for the 3D scene reconstruction stage.

Kept in one place (like ``tracking/cli_args.py``) so the splitting pipeline and
any example pipeline surface identical flags, help text and defaults.
"""

import argparse
from collections.abc import Callable

from cosmos_curator.models.depth_anything_v2 import DEPTH_VARIANTS
from cosmos_curator.pipelines.video.scene3d.calibration import DEFAULT_HFOV_DEG, MAX_HFOV_DEG, MIN_HFOV_DEG
from cosmos_curator.pipelines.video.scene3d.object_lift import DIMENSION_MODES

BACKGROUND_MODES = ("depth", "ground-plane")
OBJECT_DEPTH_MODES = ("background", "per-frame")


def _checked[T: (int, float)](
    convert: Callable[[str], T], accept: Callable[[T], bool], what: str
) -> Callable[[str], T]:
    """Build an argparse ``type=`` callable that converts then range-checks.

    The five knobs below differ only in their conversion and their predicate, so
    they share one body rather than five copies of the same try/raise/check shape.
    """

    def parse(value: str) -> T:
        try:
            parsed = convert(value)
        except ValueError as exc:
            msg = f"expected {what}, got {value!r}"
            raise argparse.ArgumentTypeError(msg) from exc
        if not accept(parsed):
            msg = f"expected {what}, got {parsed}"
            raise argparse.ArgumentTypeError(msg)
        return parsed

    return parse


_positive_float = _checked(float, lambda v: v > 0, "a positive number")
_positive_int = _checked(int, lambda v: v > 0, "a positive integer")
# 0 means "let Xenna decide" for the worker-count knob.
_non_negative_int = _checked(int, lambda v: v >= 0, "a non-negative integer")
_unit_fraction = _checked(float, lambda v: 0.0 <= v <= 1.0, "a fraction in [0, 1]")
# Bounds live with the projection maths that enforces them, so they cannot drift.
_hfov_deg = _checked(
    float,
    lambda v: MIN_HFOV_DEG < v < MAX_HFOV_DEG,
    f"a field of view in ({MIN_HFOV_DEG}, {MAX_HFOV_DEG}) degrees",
)


def add_scene3d_args(parser: argparse.ArgumentParser, *, include_enable_flag: bool = True) -> None:
    """Register ``--scene3d*`` arguments on ``parser``.

    Args:
        parser: Argparse parser to register the arguments on.
        include_enable_flag: If True, register ``--scene3d`` / ``--no-scene3d``.
            An example pipeline that always reconstructs does not need it.

    """
    if include_enable_flag:
        parser.add_argument(
            "--scene3d",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                "Reconstruct a 3D scene per clip from monocular metric depth and add it to the "
                "output MCAP as /scene/background (foxglove.PointCloud) and /scene/objects "
                "(foxglove.SceneUpdate), with /camera/camera-info and /tf-static carrying the "
                "estimated calibration instead of placeholders. /scene/objects needs tracked "
                "boxes, so combine with --sam3; without it only the background cloud is written."
            ),
        )

    parser.add_argument(
        "--scene3d-depth-variant",
        choices=DEPTH_VARIANTS,
        default="outdoor",
        help=(
            "Depth Anything V2 metric checkpoint. 'outdoor' (default) is trained to 80 m and "
            "suits street/driving/surveillance footage; 'indoor' is trained to 20 m and is "
            "sharper in rooms and corridors."
        ),
    )
    parser.add_argument(
        "--scene3d-depth-long-side",
        type=_positive_int,
        default=700,
        help=(
            "Long side (pixels) frames are downscaled to before depth inference. Also the "
            "resolution of the median background plate, so this bounds both GPU and host "
            "memory. Raising it sharpens the far field at superlinear cost."
        ),
    )
    parser.add_argument(
        "--scene3d-target-fps",
        type=_positive_float,
        default=5.0,
        help="Frames-per-second the clip is sampled at when lifting objects into 3D.",
    )
    parser.add_argument(
        "--scene3d-max-clip-duration-s",
        type=_positive_float,
        default=120.0,
        help="Skip 3D reconstruction on clips longer than this (seconds).",
    )

    # --- camera model -------------------------------------------------------
    parser.add_argument(
        "--scene3d-hfov-deg",
        type=_hfov_deg,
        default=DEFAULT_HFOV_DEG,
        help=(
            "Assumed horizontal field of view, which sets the focal length and therefore the "
            "absolute scale of the reconstruction. 60 is a reasonable default for consumer and "
            "surveillance optics; use --scene3d-focal-px when the true value is known."
        ),
    )
    parser.add_argument(
        "--scene3d-focal-px",
        type=_positive_float,
        default=None,
        help=(
            "Explicit focal length in pixels, expressed in the *source video's* resolution. "
            "Overrides --scene3d-hfov-deg; the stage rescales it onto the depth plate."
        ),
    )
    parser.add_argument(
        "--scene3d-camera-height-m",
        type=_positive_float,
        default=None,
        help=(
            "Camera height above the ground in metres. Default (unset) estimates it from the "
            "ground-plane fit; set it to pin the scale when the mounting height is known."
        ),
    )
    parser.add_argument(
        "--scene3d-camera-tilt-deg",
        type=float,
        default=None,
        help="Downward camera tilt in degrees. Default (unset) estimates it from the ground fit.",
    )
    parser.add_argument(
        "--scene3d-camera-roll-deg",
        type=float,
        default=None,
        help="Camera roll about the optical axis in degrees. Default (unset) estimates it.",
    )
    parser.add_argument(
        "--scene3d-ground-min-inlier-frac",
        type=_unit_fraction,
        default=0.15,
        help=(
            "Fraction of sampled points that must lie on the fitted plane for the ground "
            "estimate to be accepted. Below it the stage falls back to the camera-height/tilt "
            "values above and marks the calibration as not estimated."
        ),
    )
    parser.add_argument(
        "--scene3d-ground-inlier-tol-m",
        type=_positive_float,
        default=0.10,
        help="RANSAC inlier tolerance (metres) for the ground-plane fit.",
    )

    # --- reconstruction -----------------------------------------------------
    parser.add_argument(
        "--scene3d-background-mode",
        choices=BACKGROUND_MODES,
        default="depth",
        help=(
            "How /scene/background is built. 'depth' (default) back-projects the metric depth "
            "map into a true 3D backdrop and works for any scene. 'ground-plane' ray-casts "
            "pixels onto z=0, which keeps road markings crisp on a flat dominant ground but "
            "smears everything standing above it."
        ),
    )
    parser.add_argument(
        "--scene3d-object-depth",
        choices=OBJECT_DEPTH_MODES,
        default="background",
        help=(
            "Depth source for placing objects. 'background' (default) uses the single "
            "median-plate depth map: movers are erased from the plate, so a ground-contact "
            "pixel reads the floor behind the object, and it costs one inference per clip. "
            "'per-frame' runs depth on every sampled frame — necessary when the camera pans "
            "or moves within a clip, at N times the cost."
        ),
    )
    parser.add_argument(
        "--scene3d-max-points",
        type=_positive_int,
        default=200_000,
        help="Point budget for /scene/background; the cloud is voxel-downsampled to fit.",
    )
    parser.add_argument(
        "--scene3d-max-range-m",
        type=_positive_float,
        default=150.0,
        help=(
            "Cull points and objects beyond this ground range (metres). The outdoor depth "
            "model saturates at 80 m, so values far above that only admit noise."
        ),
    )
    parser.add_argument(
        "--scene3d-dimension-mode",
        choices=DIMENSION_MODES,
        default="auto",
        help=(
            "How cuboid extents are chosen. 'auto' (default) uses a class prior when the "
            "detection label is recognised and falls back to depth-derived extents otherwise. "
            "'prior' always uses the table; 'depth' always measures from the 2D box at the "
            "object's range and needs no class knowledge."
        ),
    )
    parser.add_argument(
        "--scene3d-snap-to-ground",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Sit cuboids on the fitted ground plane (z=0) instead of the noisier per-pixel "
            "lifted elevation. Disable for scenes with real height variation."
        ),
    )
    parser.add_argument(
        "--scene3d-min-track-points",
        type=_positive_int,
        default=4,
        help="Drop tracks lifted in fewer than this many frames; they are usually detector ghosts.",
    )
    parser.add_argument(
        "--scene3d-min-net-displacement-m",
        type=_positive_float,
        default=1.0,
        help=(
            "Net travel required before a track is given a heading arrow. Below it the object "
            "is treated as stationary and drawn axis-aligned rather than with a jittery yaw."
        ),
    )
    parser.add_argument(
        "--scene3d-max-speed-mps",
        type=_positive_float,
        default=60.0,
        help="Clamp reported track speed to this, so a lift outlier cannot produce absurd values.",
    )

    # --- outputs and resources ---------------------------------------------
    parser.add_argument(
        "--scene3d-write-json",
        action="store_true",
        default=False,
        help=(
            "Also write a per-clip scene3d/<uuid>.json with the estimated calibration and the "
            "per-frame cuboids. The point cloud stays MCAP-only (it is far too large for JSON)."
        ),
    )
    parser.add_argument(
        "--scene3d-gpus-per-worker",
        type=_positive_float,
        default=0.5,
        help="Fraction of a GPU each 3D reconstruction worker reserves.",
    )
    parser.add_argument(
        "--num-scene3d-workers-per-node",
        type=_non_negative_int,
        default=0,
        help="Workers per node for the 3D reconstruction stage. 0 lets Xenna decide.",
    )
