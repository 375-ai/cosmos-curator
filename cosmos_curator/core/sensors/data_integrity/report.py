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

"""Session-level aggregation and rendering for the data-integrity tool.

The per-metric verdict comes from the shared engine
(:mod:`cosmos_curator.core.sensors.data_integrity.cli_common`): every
:class:`~cosmos_curator.core.sensors.data_integrity.cli_common.CheckResult` is
``PASS`` / ``FAIL`` / ``SKIPPED``. This module adds only what the *session* view
needs on top of that: a per-stream :class:`StreamResult` that can also be an
open/decode ``ERROR`` (a stream that could not be checked at all), and a
:class:`SessionReport` that rolls one session's streams into a single verdict.

Rendering is provided in two forms: a human-readable block via
:func:`render_text` and a machine-readable object via :func:`report_to_dict` /
:func:`to_json`.
"""

import json
from enum import Enum

import attrs

from cosmos_curator.core.sensors.data_integrity.cli_common import CheckResult, CheckStatus, ExpectedHzSource


class OverallStatus(Enum):
    """Rolled-up verdict for a stream or a session.

    ``ERROR`` is distinct from ``FAIL``: ``FAIL`` means the integrity checks ran
    and something was out of bounds, while ``ERROR`` means the stream could not
    be opened or decoded so no judgment was possible. Because "unmeasured" is the
    weaker guarantee, ``ERROR`` is the more severe of the two when several streams
    are rolled up; see :meth:`SessionReport.status`.
    """

    PASS = "PASS"  # noqa: S105
    FAIL = "FAIL"
    ERROR = "ERROR"


@attrs.define(frozen=True)
class StreamResult:
    """Integrity result for a single stream (one video / one camera).

    A stream that failed to open or decode carries ``error`` set and an empty
    ``metrics`` list; its :attr:`status` is then ``ERROR``. Otherwise the status
    is ``FAIL`` if any metric failed and ``PASS`` if none did (``SKIPPED``
    metrics do not fail the stream, but are reported).

    Attributes:
        source: the stream's source path or URI.
        codec_name: codec as reported by the sensor, or ``None`` on error.
        has_bframes: B-frame (frame-reordering) flag, or ``None`` on error.
        num_samples: number of timestamps analyzed, or ``None`` on error.
        start_ns / end_ns: first / last timestamp in nanoseconds, or ``None``.
        metrics: per-metric results (empty on error).
        error: failure message if the stream could not be opened/decoded.
        expected_hz: effective expected sample rate for this stream, or ``None``
            when unavailable or on error. Resolved per stream, since each stream
            carries its own header rate.
        expected_hz_source: where ``expected_hz`` came from, or ``None`` on error.
            Reported because it explains the rate / gap / jitter verdicts: an
            ``UNAVAILABLE`` rate is why those three checks SKIP.

    """

    source: str
    codec_name: str | None
    has_bframes: bool | None
    num_samples: int | None
    start_ns: int | None
    end_ns: int | None
    metrics: list[CheckResult]
    error: str | None = None
    expected_hz: float | None = None
    expected_hz_source: ExpectedHzSource | None = None

    @property
    def status(self) -> OverallStatus:
        """Roll the metric statuses (or an open/decode error) into one verdict."""
        if self.error is not None:
            return OverallStatus.ERROR
        if any(m.status is CheckStatus.FAIL for m in self.metrics):
            return OverallStatus.FAIL
        return OverallStatus.PASS


@attrs.define(frozen=True)
class SessionReport:
    """Integrity results for one session (all of its streams).

    A future multi-session run is simply a ``list[SessionReport]``; this type
    deliberately models a single session only.

    Attributes:
        session_path: the session path / prefix the streams were discovered under.
        streams: per-stream results, in discovery order.

    """

    session_path: str
    streams: list[StreamResult]

    @property
    def status(self) -> OverallStatus:
        """Session verdict: ``ERROR`` if any stream errored, else ``FAIL`` if any failed, else ``PASS``.

        ``ERROR`` outranks ``FAIL`` because the two are different kinds of
        statement. A ``FAIL`` is a completed measurement judged against the
        thresholds in force, so it can be re-judged if that policy changes, and
        may then pass. An ``ERROR`` stream was never measured, so no
        re-evaluation can complete it: the session's verdict stays partial until
        that stream is read again. Ranking ``ERROR`` first is what lets a
        partially measured session be recognised, and re-queued, from its verdict
        alone. Both are still counted in the report, so neither is hidden. An
        empty session (nothing discovered) is ``ERROR`` for the same reason:
        nothing was measured.
        """
        if not self.streams:
            return OverallStatus.ERROR
        statuses = {s.status for s in self.streams}
        if OverallStatus.ERROR in statuses:
            return OverallStatus.ERROR
        if OverallStatus.FAIL in statuses:
            return OverallStatus.FAIL
        return OverallStatus.PASS


def _count_by_status(streams: list[StreamResult]) -> dict[str, int]:
    counts = {status.value: 0 for status in OverallStatus}
    for stream in streams:
        counts[stream.status.value] += 1
    return counts


def report_to_dict(report: SessionReport) -> dict[str, object]:
    """Convert a :class:`SessionReport` to a plain JSON-serializable dict."""
    return {
        "session_path": report.session_path,
        "status": report.status.value,
        "num_streams": len(report.streams),
        "stream_status_counts": _count_by_status(report.streams),
        "streams": [
            {
                "source": stream.source,
                "status": stream.status.value,
                "codec_name": stream.codec_name,
                "has_bframes": stream.has_bframes,
                "num_samples": stream.num_samples,
                "start_ns": stream.start_ns,
                "end_ns": stream.end_ns,
                "expected_hz": stream.expected_hz,
                "expected_hz_source": (
                    stream.expected_hz_source.value if stream.expected_hz_source is not None else None
                ),
                "error": stream.error,
                "metrics": [
                    {
                        "name": metric.name,
                        "status": metric.status.value,
                        "reason": metric.reason,
                        "measurement": metric.measurement,
                        "evaluation": metric.evaluation,
                    }
                    for metric in stream.metrics
                ],
            }
            for stream in report.streams
        ],
    }


def to_json(report: SessionReport, *, indent: int = 2) -> str:
    """Render a :class:`SessionReport` as a machine-readable JSON string."""
    return json.dumps(report_to_dict(report), indent=indent, allow_nan=False)


_METRIC_NAME_WIDTH = 26
_METRIC_STATUS_WIDTH = 10


def _render_stream(stream: StreamResult) -> list[str]:
    lines = [f"Stream: {stream.source}"]
    if stream.error is not None:
        lines.append(f"  ERROR: {stream.error}")
        return lines
    lines.append(
        f"  codec: {stream.codec_name}   has_bframes: {str(stream.has_bframes).lower()}   "
        f"num_samples: {stream.num_samples}   start_ns: {stream.start_ns}   end_ns: {stream.end_ns}"
    )
    if stream.expected_hz_source is not None:
        # Same wording as the single-video report so the two outputs stay comparable.
        value = f"{stream.expected_hz:.3f}" if stream.expected_hz is not None else "N/A"
        lines.append(f"  expected_hz: {value} (source: {stream.expected_hz_source.value})")
    lines.append("")
    lines.extend(
        f"  {metric.name:<{_METRIC_NAME_WIDTH}}{metric.status.value:<{_METRIC_STATUS_WIDTH}}{metric.reason}"
        for metric in stream.metrics
    )
    lines.append(f"  -> {stream.status.value}")
    return lines


def render_text(report: SessionReport) -> str:
    """Render a :class:`SessionReport` as a human-readable multi-line string."""
    counts = _count_by_status(report.streams)
    lines: list[str] = []
    for stream in report.streams:
        lines.extend(_render_stream(stream))
        lines.append("")
    lines.append(f"Data-integrity report for session: {report.session_path}")
    lines.append(
        f"  streams: {len(report.streams)}   "
        + "   ".join(f"{name.lower()}: {counts[name]}" for name in (s.value for s in OverallStatus))
    )
    lines.append(f"Session overall: {report.status.value}")
    return "\n".join(lines)
