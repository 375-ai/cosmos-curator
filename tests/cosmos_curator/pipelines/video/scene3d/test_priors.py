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
"""Tests for the class-prior and depth-derived sizing table."""

import pytest

from cosmos_curator.pipelines.video.scene3d import priors


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("car", "car"),
        ("a red car", "car"),
        ("A CAR.", "car"),
        ("a white pickup truck", "pickup truck"),
        ("pickup-truck", "pickup truck"),
        ("a person walking", "person"),
        ("  bus  ", "bus"),
    ],
)
def test_match_label_resolves_free_text(label: str, expected: str) -> None:
    """SAM3 prompts are natural language, so matching is keyword-based."""
    assert priors.match_label(label) == expected


def test_longer_keys_win_over_their_suffixes() -> None:
    """'pickup truck' must not collapse to the much larger 'truck' prior."""
    pickup = priors.prior_for("a pickup truck")
    truck = priors.prior_for("a truck")
    assert pickup is not None
    assert truck is not None
    assert pickup[0] != truck[0]
    assert pickup[0][0] < truck[0][0]


@pytest.mark.parametrize("label", [None, "", "   ", "a glorpsnaggle", "!!!"])
def test_unmatched_labels_return_none(label: str | None) -> None:
    """An unknown label has no prior, which is what routes it to depth-derived sizing."""
    assert priors.match_label(label) is None
    assert priors.prior_for(label) is None


def test_priors_are_physically_plausible() -> None:
    """Every entry is positive and, for vehicles, longer than it is wide."""
    for key, (dimensions, colour) in priors.DIMENSION_PRIORS.items():
        assert min(dimensions) > 0, key
        assert len(colour) == 3
        assert all(0.0 <= channel <= 1.0 for channel in colour), key


def test_normalize_label_collapses_punctuation() -> None:
    """Normalisation is what lets 'pickup-truck' and 'Pickup Truck!' agree."""
    assert priors.normalize_label("Pickup--Truck!!") == "pickup truck"
    assert priors.normalize_label(None) == ""
