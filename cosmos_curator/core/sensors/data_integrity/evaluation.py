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

"""Data-integrity evaluation.

Evaluators are functions that turn a measurement into a pass/fail judgment,
keeping measurement (metrics) separate from policy (evaluation). A measurement can
be taken once and evaluated many times with different thresholds. See
``docs/curator/design/data-integrity-design.md``.

Precondition: evaluators assume the measurement is well-defined (``is_defined`` is
True) and do not check it. An undefined measurement carries ``nan``/``None`` fact
fields and must not be passed to an evaluator directly; the caller is responsible
for checking ``is_defined`` first.
"""

from collections.abc import Callable
from enum import Enum

import attrs

from cosmos_curator.core.sensors.data_integrity.metrics import Measurement


class EvaluationStatus(Enum):
    """Outcome of evaluating a measurement — always ``PASS`` or ``FAIL``.

    There is no ``UNDEFINED`` here: an undefined measurement is never evaluated
    (evaluators would have no honest margin for it). Callers must check
    ``is_defined`` and skip evaluation when it is False; see
    ``docs/curator/design/data-integrity-design.md``.
    """

    PASS = "PASS"  # noqa: S105
    FAIL = "FAIL"


@attrs.define(frozen=True)
class EvaluationResult[T: (int, float)]:
    """Result of evaluating a metric against a policy.

    Attributes:
        status: pass or fail
        margin: signed distance from the threshold — positive means headroom
            (passing with room to spare), negative means the amount by which the
            value exceeded the limit.

    """

    status: EvaluationStatus
    margin: T


def below_threshold[T: (int, float), M: Measurement](
    *,
    threshold: T,
    measurement: M,
    accessor: Callable[[M], T],
) -> EvaluationResult[T]:
    """Pass when the accessed value is at or below ``threshold`` (lower is better).

    Positive margin = headroom below the threshold; negative = amount over.
    """
    value = accessor(measurement)
    status = EvaluationStatus.PASS if value <= threshold else EvaluationStatus.FAIL
    margin = threshold - value
    return EvaluationResult(status=status, margin=margin)


def above_threshold[T: (int, float), M: Measurement](
    *,
    threshold: T,
    measurement: M,
    accessor: Callable[[M], T],
) -> EvaluationResult[T]:
    """Pass when the accessed value is at or above ``threshold`` (higher is better).

    Positive margin = headroom above the threshold; negative = amount under.
    """
    value = accessor(measurement)
    status = EvaluationStatus.PASS if value >= threshold else EvaluationStatus.FAIL
    margin = value - threshold
    return EvaluationResult(status=status, margin=margin)


def within_range[T: (int, float), M: Measurement](
    *,
    min_value: T,
    max_value: T,
    measurement: M,
    accessor: Callable[[M], T],
) -> EvaluationResult[T]:
    """Pass when the accessed value is within ``[min_value, max_value]``.

    Positive margin = distance to the nearest bound (headroom); negative = distance
    outside the range.

    Raises:
        ValueError: if ``min_value > max_value``. An inverted range is empty, so
            every value would fail with a misleading margin; that is a caller bug,
            not a measurement to judge, so it is rejected rather than evaluated.

    """
    if min_value > max_value:
        msg = f"within_range requires min_value <= max_value, got min_value={min_value}, max_value={max_value}"
        raise ValueError(msg)
    value = accessor(measurement)
    passed = min_value <= value <= max_value
    if passed:
        margin = min(value - min_value, max_value - value)
    else:
        margin = value - min_value if value < min_value else max_value - value
    status = EvaluationStatus.PASS if passed else EvaluationStatus.FAIL
    return EvaluationResult(status=status, margin=margin)
