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

"""Per-class metric size and display-colour priors for lifted 3D cuboids.

Detections arrive as free-text labels (SAM3 prompts are natural language, e.g.
``"a red car"``), so a class taxonomy alone can't cover the input. Two
complementary estimators live here and are combined by
:func:`cosmos_curator.pipelines.video.scene3d.object_lift`:

- a **prior** table keyed by normalised keyword, which is the only way to know a
  car is 4.5 m *long* when a single view shows only its width and height;
- a **depth-derived** fallback that converts the 2D box extent to metres at the
  object's measured range, which needs no class knowledge at all.

Dimensions are ``(length_along_heading, width, height)`` in metres. Longer keys
are matched first so ``"pickup truck"`` never resolves as ``"truck"``.
"""

import re
from typing import Final

Dimensions = tuple[float, float, float]
Color = tuple[float, float, float]

# (length along heading, width, height) in metres, and display RGB.
DIMENSION_PRIORS: Final[dict[str, tuple[Dimensions, Color]]] = {
    # --- vehicles ---
    "car": ((4.5, 1.8, 1.5), (0.20, 0.55, 1.00)),
    "sedan": ((4.6, 1.8, 1.5), (0.20, 0.55, 1.00)),
    "suv": ((4.8, 1.9, 1.7), (0.20, 0.85, 0.45)),
    "van": ((5.5, 2.0, 2.2), (0.20, 0.85, 0.85)),
    "minivan": ((5.0, 2.0, 1.8), (0.20, 0.85, 0.85)),
    "pickup truck": ((5.5, 2.0, 1.9), (1.00, 0.55, 0.10)),
    "box truck": ((8.0, 2.5, 3.0), (0.95, 0.85, 0.10)),
    "flatbed truck": ((9.0, 2.5, 2.0), (0.85, 0.70, 0.30)),
    "semi trailer": ((16.0, 2.5, 4.0), (0.95, 0.20, 0.20)),
    "semi": ((16.0, 2.5, 4.0), (0.95, 0.20, 0.20)),
    "truck": ((9.0, 2.5, 3.2), (0.95, 0.65, 0.10)),
    "bus": ((12.0, 2.6, 3.2), (0.95, 0.75, 0.10)),
    "train": ((25.0, 3.0, 4.0), (0.60, 0.40, 0.90)),
    "motorcycle": ((2.2, 0.9, 1.4), (0.85, 0.30, 0.85)),
    "scooter": ((1.9, 0.7, 1.2), (0.85, 0.30, 0.85)),
    "bicycle": ((1.8, 0.6, 1.1), (0.45, 0.85, 0.35)),
    "stroller": ((1.0, 0.6, 1.0), (0.45, 0.85, 0.35)),
    # --- people and animals ---
    "person": ((0.6, 0.6, 1.7), (1.00, 0.85, 0.25)),
    "pedestrian": ((0.6, 0.6, 1.7), (1.00, 0.85, 0.25)),
    "child": ((0.4, 0.4, 1.2), (1.00, 0.85, 0.25)),
    "dog": ((0.9, 0.3, 0.6), (0.90, 0.60, 0.40)),
    "cat": ((0.5, 0.2, 0.3), (0.90, 0.60, 0.40)),
    "horse": ((2.4, 0.8, 1.7), (0.90, 0.60, 0.40)),
    # --- street furniture ---
    "traffic cone": ((0.4, 0.4, 0.7), (1.00, 0.40, 0.10)),
    "traffic light": ((0.4, 0.4, 1.0), (0.30, 0.90, 0.50)),
    "stop sign": ((0.1, 0.8, 0.8), (0.90, 0.20, 0.20)),
    "fire hydrant": ((0.4, 0.4, 0.8), (0.90, 0.20, 0.20)),
    "bench": ((1.8, 0.6, 0.9), (0.60, 0.50, 0.40)),
    "trash can": ((0.6, 0.6, 1.0), (0.50, 0.50, 0.50)),
    # --- indoor ---
    "chair": ((0.6, 0.6, 0.9), (0.55, 0.75, 0.55)),
    "table": ((1.4, 0.8, 0.75), (0.55, 0.75, 0.55)),
    "desk": ((1.4, 0.7, 0.75), (0.55, 0.75, 0.55)),
    "sofa": ((2.0, 0.9, 0.8), (0.55, 0.75, 0.55)),
    "bed": ((2.0, 1.5, 0.6), (0.55, 0.75, 0.55)),
    "laptop": ((0.35, 0.25, 0.25), (0.70, 0.70, 0.80)),
    "backpack": ((0.3, 0.2, 0.5), (0.70, 0.70, 0.80)),
    "box": ((0.5, 0.5, 0.5), (0.70, 0.70, 0.80)),
}

DEFAULT_DIMENSIONS: Final[Dimensions] = (0.8, 0.8, 1.0)
DEFAULT_COLOR: Final[Color] = (0.70, 0.70, 0.70)

# Longest keys first so multi-word entries win over their single-word suffixes.
_MATCH_ORDER: Final[tuple[str, ...]] = tuple(sorted(DIMENSION_PRIORS, key=len, reverse=True))
_NON_WORD = re.compile(r"[^a-z0-9]+")


def normalize_label(label: str | None) -> str:
    """Lowercase ``label`` and collapse punctuation to single spaces."""
    return _NON_WORD.sub(" ", (label or "").lower()).strip()


def match_label(label: str | None) -> str | None:
    """Resolve a free-text label to a :data:`DIMENSION_PRIORS` key.

    Matching is substring-based on word boundaries, longest key first, so
    ``"a white pickup truck"`` resolves to ``pickup truck`` rather than ``truck``.

    Args:
        label: Free-text detection label (a SAM3 prompt, a detector class name).

    Returns:
        The matching prior key, or ``None`` when nothing matches.

    """
    text = normalize_label(label)
    if not text:
        return None
    padded = f" {text} "
    for key in _MATCH_ORDER:
        if f" {key} " in padded:
            return key
    return None


def prior_for(label: str | None) -> tuple[Dimensions, Color] | None:
    """Return ``(dimensions, colour)`` for ``label``, or ``None`` when unmatched."""
    key = match_label(label)
    return None if key is None else DIMENSION_PRIORS[key]
