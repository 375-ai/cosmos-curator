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

"""Unit tests for the session runner (run_stream + run_session)."""

import pathlib
from collections.abc import Iterator
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

from cosmos_curator.core.sensors.data_integrity import session_runner
from cosmos_curator.core.sensors.data_integrity.cli_common import CheckResult, CheckStatus, ExpectedHzSource
from cosmos_curator.core.sensors.data_integrity.report import OverallStatus, StreamResult
from cosmos_curator.core.sensors.data_integrity.session_runner import run_session, run_stream

HZ_100_PERIOD_NS = 10_000_000  # one sample every 10 ms at 100 Hz


class _FakeSensor:
    """Minimal stand-in satisfying the shared engine's IntegritySensor surface."""

    def __init__(
        self,
        timestamps: list[int],
        *,
        has_bframes: bool = False,
        codec_name: str = "h264",
        avg_frame_rate: Fraction = Fraction(100, 1),
    ) -> None:
        self._ts = np.array(timestamps, dtype=np.int64)
        self.has_bframes = has_bframes
        self.codec_name = codec_name
        self.video_metadata = SimpleNamespace(avg_frame_rate=avg_frame_rate)

    @property
    def timestamps_ns(self) -> NDArray[np.int64]:
        return self._ts

    @property
    def start_ns(self) -> int:
        return int(self._ts[0]) if len(self._ts) else 0

    @property
    def end_ns(self) -> int:
        return int(self._ts[-1]) if len(self._ts) else 0

    def stream_timestamps(self, batch_size: int = 0) -> Iterator[NDArray[np.int64]]:
        step = batch_size or len(self._ts) or 1
        for start in range(0, len(self._ts), step):
            yield self._ts[start : start + step]


def _perfect_cadence(n: int = 10) -> list[int]:
    return [i * HZ_100_PERIOD_NS for i in range(n)]


def _statuses(result: StreamResult) -> dict[str, CheckStatus]:
    return {m.name: m.status for m in result.metrics}


def test_run_stream_perfect_cadence_all_pass() -> None:
    """A strictly-increasing, on-cadence, B-frame-free stream passes every metric."""
    sensor = _FakeSensor(_perfect_cadence(), has_bframes=False)
    result = run_stream(sensor, source="x", expected_hz=100.0)
    assert result.num_samples == 10
    assert result.status is OverallStatus.PASS
    assert set(_statuses(result).values()) == {CheckStatus.PASS}


def test_run_stream_records_user_supplied_expected_hz() -> None:
    """An explicit rate is carried onto the result so the session report can show its origin."""
    result = run_stream(_FakeSensor(_perfect_cadence()), source="x", expected_hz=100.0)
    assert result.expected_hz == 100.0
    assert result.expected_hz_source is ExpectedHzSource.USER


def test_run_stream_records_header_expected_hz() -> None:
    """With no explicit rate, the stream's own header rate is resolved and recorded."""
    sensor = _FakeSensor(_perfect_cadence(), avg_frame_rate=Fraction(100, 1))
    result = run_stream(sensor, source="x")
    assert result.expected_hz == 100.0
    assert result.expected_hz_source is ExpectedHzSource.HEADER


def test_run_stream_records_unavailable_expected_hz() -> None:
    """A stream with no usable header rate records UNAVAILABLE, explaining the SKIPs."""
    sensor = _FakeSensor(_perfect_cadence(), avg_frame_rate=Fraction(0, 1))
    result = run_stream(sensor, source="x")
    assert result.expected_hz is None
    assert result.expected_hz_source is ExpectedHzSource.UNAVAILABLE
    assert _statuses(result)["rate"] is CheckStatus.SKIPPED


def test_run_stream_bframes_fail_the_stream() -> None:
    """A frame-reordering flag fails the frame-reordering metric and the stream."""
    sensor = _FakeSensor(_perfect_cadence(), has_bframes=True)
    result = run_stream(sensor, source="x", expected_hz=100.0)
    assert _statuses(result)["frame_reordering_present"] is CheckStatus.FAIL
    assert result.status is OverallStatus.FAIL


def test_run_stream_single_sample_marks_timing_metrics_skipped() -> None:
    """One timestamp cannot define any timing metric; they are SKIPPED, not evaluated."""
    sensor = _FakeSensor([0], has_bframes=False)
    result = run_stream(sensor, source="x", expected_hz=100.0)
    statuses = _statuses(result)
    for name in ("timestamp_ordering", "rate", "timestamp_gap", "jitter"):
        assert statuses[name] is CheckStatus.SKIPPED
    # Frame reordering only needs the flag, so it is still defined and passes.
    assert statuses["frame_reordering_present"] is CheckStatus.PASS
    # Skipped metrics do not fail the stream.
    assert result.status is OverallStatus.PASS


def test_run_stream_falls_back_to_avg_frame_rate() -> None:
    """With no explicit expected_hz, the nominal avg_frame_rate drives the cadence metrics."""
    sensor = _FakeSensor(_perfect_cadence(), avg_frame_rate=Fraction(100, 1))
    result = run_stream(sensor, source="x")
    assert _statuses(result)["rate"] is CheckStatus.PASS


def test_run_stream_no_rate_leaves_cadence_metrics_skipped() -> None:
    """Without an explicit or nominal rate, cadence metrics are SKIPPED but ordering still runs."""
    sensor = _FakeSensor(_perfect_cadence(), avg_frame_rate=Fraction(0, 1))
    result = run_stream(sensor, source="x")
    statuses = _statuses(result)
    assert statuses["rate"] is CheckStatus.SKIPPED
    assert statuses["timestamp_gap"] is CheckStatus.SKIPPED
    assert statuses["jitter"] is CheckStatus.SKIPPED
    assert statuses["timestamp_ordering"] is CheckStatus.PASS


def test_run_stream_batched_matches_one_shot() -> None:
    """Streaming in small batches yields the same verdict as one batch."""
    sensor = _FakeSensor(_perfect_cadence(), has_bframes=False)
    result = run_stream(sensor, source="x", expected_hz=100.0, batch_size=3)
    assert result.num_samples == 10
    assert result.status is OverallStatus.PASS


def _canned(source: str, status: CheckStatus) -> StreamResult:
    return StreamResult(
        source=source,
        codec_name="h264",
        has_bframes=False,
        num_samples=10,
        start_ns=0,
        end_ns=1,
        metrics=[CheckResult("m", status, "m=ok", measurement=None, evaluation=None)],
    )


def test_run_session_aggregates_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_session preserves discovery order and rolls the per-stream verdicts up."""
    monkeypatch.setattr(session_runner, "discover_streams", lambda *_a, **_k: ["x", "y"])
    canned = {"x": _canned("x", CheckStatus.PASS), "y": _canned("y", CheckStatus.FAIL)}
    monkeypatch.setattr(session_runner, "_run_one_stream", lambda source, **_k: canned[source])

    report = run_session("sess")

    assert report.session_path == "sess"
    assert [s.source for s in report.streams] == ["x", "y"]
    assert report.status is OverallStatus.FAIL


def test_run_session_captures_open_errors(tmp_path: pathlib.Path) -> None:
    """A file that is not a decodable video becomes a per-stream ERROR, not a crash."""
    (tmp_path / "a.mp4").write_bytes(b"not a real video")
    report = run_session(str(tmp_path))
    assert len(report.streams) == 1
    assert report.streams[0].status is OverallStatus.ERROR
    assert report.streams[0].error is not None
    assert report.status is OverallStatus.ERROR


def test_run_session_empty_dir_is_error(tmp_path: pathlib.Path) -> None:
    """A session with no discoverable streams reports ERROR and no streams."""
    report = run_session(str(tmp_path))
    assert report.streams == []
    assert report.status is OverallStatus.ERROR


@pytest.mark.parametrize("bad_hz", [0.0, -1.0, float("nan"), float("inf")])
def test_run_stream_rejects_invalid_expected_hz(bad_hz: float) -> None:
    """run_stream fails fast on a non-positive / non-finite expected_hz."""
    sensor = _FakeSensor(_perfect_cadence())
    with pytest.raises(ValueError, match="expected_hz"):
        run_stream(sensor, source="x", expected_hz=bad_hz)


def test_run_stream_rejects_negative_batch_size() -> None:
    """run_stream fails fast on a negative batch_size."""
    sensor = _FakeSensor(_perfect_cadence())
    with pytest.raises(ValueError, match="batch_size"):
        run_stream(sensor, source="x", expected_hz=100.0, batch_size=-1)


def test_run_session_rejects_invalid_expected_hz_before_io(tmp_path: pathlib.Path) -> None:
    """run_session validates expected_hz at the boundary, even for an empty session (no I/O)."""
    with pytest.raises(ValueError, match="expected_hz"):
        run_session(str(tmp_path), expected_hz=-5.0)


def test_run_session_rejects_negative_batch_size_before_io(tmp_path: pathlib.Path) -> None:
    """run_session validates batch_size at the boundary, even for an empty session (no I/O)."""
    with pytest.raises(ValueError, match="batch_size"):
        run_session(str(tmp_path), batch_size=-1)
