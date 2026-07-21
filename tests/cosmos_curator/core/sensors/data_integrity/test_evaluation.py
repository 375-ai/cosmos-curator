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

"""Unit tests for the data-integrity evaluators.

Margin sign contract: positive = headroom (passing with room to spare); negative =
the amount by which the value is outside the acceptable region.
"""

from types import SimpleNamespace

import pytest

from cosmos_curator.core.sensors.data_integrity.evaluation import (
    EvaluationStatus,
    above_threshold,
    below_threshold,
    within_range,
)


def _m(value: float) -> SimpleNamespace:
    return SimpleNamespace(value=value)


def _value(m: SimpleNamespace) -> float:
    return m.value


def test_below_threshold_pass_has_positive_margin() -> None:
    """At or below the limit passes with positive headroom."""
    r = below_threshold(threshold=10, measurement=_m(4), accessor=_value)
    assert r.status is EvaluationStatus.PASS
    assert r.margin == 6


def test_below_threshold_fail_has_negative_margin() -> None:
    """Over the limit fails with the (negative) amount over."""
    r = below_threshold(threshold=10, measurement=_m(13), accessor=_value)
    assert r.status is EvaluationStatus.FAIL
    assert r.margin == -3


def test_above_threshold_pass_has_positive_margin() -> None:
    """At or above the floor passes with positive headroom."""
    r = above_threshold(threshold=10, measurement=_m(13), accessor=_value)
    assert r.status is EvaluationStatus.PASS
    assert r.margin == 3


def test_above_threshold_fail_has_negative_margin() -> None:
    """Under the floor fails with the (negative) amount under."""
    r = above_threshold(threshold=10, measurement=_m(7), accessor=_value)
    assert r.status is EvaluationStatus.FAIL
    assert r.margin == -3


def test_within_range_pass_margin_is_nearest_bound() -> None:
    """Inside the range passes; margin is the distance to the nearer bound."""
    r = within_range(min_value=0, max_value=10, measurement=_m(8), accessor=_value)
    assert r.status is EvaluationStatus.PASS
    assert r.margin == 2


def test_within_range_fail_below_has_negative_margin() -> None:
    """Below the range fails with a negative margin."""
    r = within_range(min_value=0, max_value=10, measurement=_m(-3), accessor=_value)
    assert r.status is EvaluationStatus.FAIL
    assert r.margin == -3


def test_within_range_fail_above_has_negative_margin() -> None:
    """Above the range must also be negative (regression: previously returned positive)."""
    r = within_range(min_value=0, max_value=10, measurement=_m(13), accessor=_value)
    assert r.status is EvaluationStatus.FAIL
    assert r.margin == -3


def test_below_threshold_at_boundary_passes() -> None:
    """The threshold is inclusive: value == threshold passes with zero margin."""
    r = below_threshold(threshold=10, measurement=_m(10), accessor=_value)
    assert r.status is EvaluationStatus.PASS
    assert r.margin == 0


def test_above_threshold_at_boundary_passes() -> None:
    """The floor is inclusive: value == threshold passes with zero margin."""
    r = above_threshold(threshold=10, measurement=_m(10), accessor=_value)
    assert r.status is EvaluationStatus.PASS
    assert r.margin == 0


def test_within_range_at_bounds_passes() -> None:
    """The range is inclusive at both ends (margin zero at a bound)."""
    lo = within_range(min_value=0, max_value=10, measurement=_m(0), accessor=_value)
    hi = within_range(min_value=0, max_value=10, measurement=_m(10), accessor=_value)
    assert lo.status is EvaluationStatus.PASS
    assert hi.status is EvaluationStatus.PASS
    assert lo.margin == 0
    assert hi.margin == 0


def test_within_range_rejects_inverted_bounds() -> None:
    """An inverted range (min > max) is empty and would fail every value; reject it as a caller bug."""
    with pytest.raises(ValueError, match="min_value <= max_value"):
        within_range(min_value=10, max_value=0, measurement=_m(5), accessor=_value)
