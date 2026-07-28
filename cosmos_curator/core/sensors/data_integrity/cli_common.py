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

"""Shared per-stream engine for the data-integrity CLIs.

This is the single source of truth for *running all data-integrity metrics
against one video stream and judging them*. Both entry points build on it:

* the single-video ``di-check`` CLI (:mod:`cosmos_curator.core.sensors.data_integrity.cli`)
* the single-session tool (:mod:`cosmos_curator.core.sensors.data_integrity.session_runner`)

It owns the vocabulary that surrounds the kernel
(:mod:`cosmos_curator.core.sensors.data_integrity.metrics` / ``.evaluation``):
the kernel only ever judges a *well-defined* measurement as ``PASS`` / ``FAIL``,
so this module adds :class:`CheckStatus` (which also carries ``SKIPPED`` for an
undefined measurement or a missing rate prerequisite), the :class:`Thresholds`
pass/fail policy, the effective-rate resolution (:func:`resolve_expected_hz`),
and JSON-safe serialisation of measurements / evaluations.

Rendering and aggregation live with the callers: the single-video report shapes
in :mod:`.cli`, the session rollup in :mod:`.report`.
"""

import argparse
import enum
import math
import pathlib
import time
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from typing import BinaryIO, Protocol, cast

import attrs
import numpy as np
from numpy.typing import NDArray

from cosmos_curator.core.sensors.data.video import VideoMetadata
from cosmos_curator.core.sensors.data_integrity.evaluation import (
    EvaluationResult,
    EvaluationStatus,
    below_threshold,
)
from cosmos_curator.core.sensors.data_integrity.metrics import (
    FrameReorderingPresentMetric,
    JitterMetric,
    Measurement,
    RateMetric,
    TimestampGapMetric,
    TimestampOrderingMetric,
)
from cosmos_curator.core.sensors.scripts._cli_cloud import is_cloud_uri, open_cloud_source
from cosmos_curator.core.sensors.sensors.camera_sensor import CameraSensor
from cosmos_curator.core.sensors.types.types import DataSource

# Metric identifiers used verbatim in both CLIs' human and JSON reports.
NAME_ORDERING = "timestamp_ordering"
NAME_RATE = "rate"
NAME_GAP = "timestamp_gap"
NAME_JITTER = "jitter"
NAME_REORDERING = "frame_reordering_present"

# Reason attached to a rate-dependent check that has no usable expected rate.
REASON_MISSING_HZ = "skipped: --expected-hz not provided and container header lacks a nominal frame rate"


def positive_finite_float(raw: str) -> float:
    """Argparse ``type=`` for flags that must be strictly positive and finite (rejects 0, NaN, inf)."""
    msg = f"expected a positive finite number, got {raw!r}"
    try:
        value = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(msg) from None
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(msg)
    return value


def non_negative_int(raw: str) -> int:
    """Argparse ``type=`` for flags that must be a non-negative integer (rejects negatives that would wrap)."""
    msg = f"expected a non-negative integer, got {raw!r}"
    try:
        value = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(msg) from None
    if value < 0:
        raise argparse.ArgumentTypeError(msg)
    return value


def validate_expected_hz(value: float | None) -> float | None:
    """Validate an already-typed expected sample rate at a library boundary.

    The argparse ``type=`` layer (:func:`positive_finite_float`) only guards the
    CLIs; this is the value-level counterpart so direct callers of the runner /
    resolver APIs get the same fail-fast contract. ``None`` (auto-detect from the
    container header) is allowed; ``0``, negatives, ``NaN``, and ``inf`` are not,
    since they would feed a meaningless cadence into the rate/gap/jitter metrics.
    """
    if value is not None and (not math.isfinite(value) or value <= 0.0):
        msg = f"expected_hz must be a positive finite number or None, got {value!r}"
        raise ValueError(msg)
    return value


def validate_non_negative_int(name: str, value: int) -> int:
    """Validate that an already-typed integer flag (e.g. ``batch_size`` / ``limit``) is non-negative."""
    if value < 0:
        msg = f"{name} must be >= 0, got {value!r}"
        raise ValueError(msg)
    return value


class CheckStatus(enum.Enum):
    """Per-metric check status, deliberately distinct from :class:`EvaluationStatus`.

    Kernel evaluators only ever return PASS/FAIL over a well-defined measurement.
    ``SKIPPED`` lives here because it covers the cases the kernel cannot evaluate
    at all: an undefined measurement (a kernel invariant not to evaluate) or a
    missing prerequisite (no usable expected rate from either ``--expected-hz`` or
    the container header).
    """

    PASS = "PASS"  # noqa: S105
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"


class ExpectedHzSource(enum.Enum):
    """Origin of the effective expected sample rate used by rate-dependent metrics.

    Variants, in fallback priority:

    * ``USER`` -- user-supplied ``--expected-hz``; the authoritative baseline.
    * ``HEADER`` -- from :attr:`VideoMetadata.avg_frame_rate`; best-effort only,
      and not a sound basis for the rate check (see :func:`resolve_expected_hz`).
    * ``UNAVAILABLE`` -- neither is usable; rate-dependent metrics SKIP.

    Reported alongside the rate itself so a reader can tell which of those three
    situations produced a given verdict.
    """

    USER = "user"
    HEADER = "header"
    UNAVAILABLE = "unavailable"


@attrs.define(frozen=True)
class Thresholds:
    """Pass/fail policy applied by :func:`run_metrics`.

    Defaults are neutral, first-principles limits (an ideal stream is strictly
    increasing, on-cadence, gap-free), not values tuned on any dataset.

    Attributes:
        max_strict_violations: max allowed ordering violations (backward +
            duplicate steps); default 0 (require strictly increasing).
        max_rate_deviation_percent: max mean-period deviation from the expected
            cadence, in percent.
        max_gaps: max allowed inferred gaps; default 0.
        max_jitter_percent: max inter-sample jitter, in percent of the period.
        allow_frame_reordering: when False (default), a B-frame / frame-reordering
            flag fails the frame-reordering metric.

    """

    max_strict_violations: int = 0
    max_rate_deviation_percent: float = 5.0
    max_gaps: int = 0
    max_jitter_percent: float = 10.0
    allow_frame_reordering: bool = False


DEFAULT_THRESHOLDS = Thresholds()


@attrs.define(frozen=True)
class CheckResult:
    """Result of one metric on one stream.

    Attributes:
        name: metric identifier (one of the ``NAME_*`` constants).
        status: PASS / FAIL / SKIPPED for this metric on this stream.
        reason: human-readable one-line summary (value and threshold, or why it
            was skipped), shown verbatim in the human report.
        measurement: JSON-safe dict of the raw measurement, or ``None`` when the
            metric was skipped before a measurement existed.
        evaluation: JSON-safe dict of the kernel evaluation (status + margin), or
            ``None`` when the measurement was undefined / skipped.

    """

    name: str
    status: CheckStatus
    reason: str
    measurement: dict[str, object] | None
    evaluation: dict[str, object] | None


@attrs.define(frozen=True)
class ResolvedConfig:
    """Effective expected rate resolved from user args + sensor metadata.

    ``expected_hz`` is ``None`` iff ``expected_hz_source`` is ``UNAVAILABLE``; the
    invariant is enforced by :func:`resolve_expected_hz`.
    """

    expected_hz: float | None
    expected_hz_source: ExpectedHzSource


@attrs.define(frozen=True)
class VideoInfo:
    """Small snapshot of sensor-level facts shared by both reports."""

    codec_name: str
    has_bframes: bool
    num_samples: int
    start_ns: int | None
    end_ns: int | None

    def to_dict(self) -> dict[str, object]:
        """Return a plain JSON-serialisable dict of the fields."""
        return {
            "codec_name": self.codec_name,
            "has_bframes": self.has_bframes,
            "num_samples": self.num_samples,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
        }


class IntegritySensor(Protocol):  # pragma: no cover
    """Structural sensor surface :func:`run_metrics` needs (``CameraSensor`` satisfies it)."""

    @property
    def codec_name(self) -> str:
        """Video codec name (e.g. ``h264``)."""
        ...

    @property
    def has_bframes(self) -> bool:
        """Whether the stream signals frame reordering (B-frames)."""
        ...

    @property
    def start_ns(self) -> int:
        """First timestamp in nanoseconds."""
        ...

    @property
    def end_ns(self) -> int:
        """Last timestamp in nanoseconds."""
        ...

    @property
    def timestamps_ns(self) -> NDArray[np.int64]:
        """The full decoded timeline in ``int64`` nanoseconds."""
        ...

    @property
    def video_metadata(self) -> VideoMetadata:
        """Scalar stream metadata (carries the nominal ``avg_frame_rate``)."""
        ...

    def stream_timestamps(self, batch_size: int = 0) -> Iterator[NDArray[np.int64]]:
        """Yield the timeline in ``int64`` ns batches (``0`` = one batch)."""
        ...


def _json_safe(value: object) -> object:
    """Convert numpy scalars, NaN, and infinities into JSON-serialisable equivalents.

    NaN and infinity have no JSON representation; ``allow_nan=False`` in
    :func:`json.dumps` would otherwise raise, silently promoting a rare corruption
    into a hard crash on reporting. Both map to ``None`` so the report survives an
    undefined field while staying loudly wrong (rather than ``0.0``, which would
    look defined).
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def measurement_to_dict(measurement: Measurement) -> dict[str, object]:
    """Serialise a metric measurement into a JSON-safe dict."""
    raw = attrs.asdict(measurement)  # type: ignore[arg-type]  # protocol vs. attrs class
    return _json_safe(raw)  # type: ignore[return-value]


def evaluation_to_dict(result: EvaluationResult[int] | EvaluationResult[float]) -> dict[str, object]:
    """Serialise an :class:`EvaluationResult` into a JSON-safe dict."""
    return {"status": result.status.value, "margin": _json_safe(result.margin)}


def resolve_expected_hz(user_hz: float | None, sensor: IntegritySensor) -> ResolvedConfig:
    """Resolve the effective expected sample rate and record its origin.

    Priority: explicit ``user_hz`` > container header ``avg_frame_rate`` > unavailable.

    An explicit ``user_hz`` is the authoritative baseline: it states independently
    what the capture was supposed to be, which is the only way to judge whether it
    was right.

    The header value is best-effort and weaker than it looks. It comes from
    libav's ``AVStream.avg_frame_rate``, which is not guaranteed to preserve an
    encoder-declared cadence -- for MP4 it is commonly derived from the
    container's own sample-duration table, the same timing data the checked
    timestamps come from. Judged against such a value, gap and jitter still bite,
    because they measure variation *between* samples: an irregular cadence shows
    up whatever the baseline. The rate check does not, and cannot be relied on to
    catch a uniformly wrong capture rate, since a stream that ran entirely at the
    wrong speed tends to declare a header rate that matches it. Pass
    ``--expected-hz`` whenever that distinction matters.

    ``Fraction(0)`` -- the sentinel used in
    :mod:`cosmos_curator.core.sensors.utils.video` when the header lacks a usable
    rate -- collapses to ``UNAVAILABLE`` and rate-dependent metrics SKIP.

    Raises:
        ValueError: if ``user_hz`` is non-positive or non-finite (see
            :func:`validate_expected_hz`).

    """
    user_hz = validate_expected_hz(user_hz)
    if user_hz is not None:
        return ResolvedConfig(expected_hz=user_hz, expected_hz_source=ExpectedHzSource.USER)
    nominal = sensor.video_metadata.avg_frame_rate
    if nominal.numerator == 0:
        return ResolvedConfig(expected_hz=None, expected_hz_source=ExpectedHzSource.UNAVAILABLE)
    return ResolvedConfig(expected_hz=float(nominal), expected_hz_source=ExpectedHzSource.HEADER)


def video_info(sensor: IntegritySensor) -> VideoInfo:
    """Snapshot the sensor-level facts shared by both reports."""
    has_samples = bool(sensor.timestamps_ns.size)
    return VideoInfo(
        codec_name=sensor.codec_name,
        has_bframes=sensor.has_bframes,
        num_samples=int(sensor.timestamps_ns.size),
        start_ns=sensor.start_ns if has_samples else None,
        end_ns=sensor.end_ns if has_samples else None,
    )


def _skipped_missing_hz(name: str) -> CheckResult:
    """Build a SKIPPED result for a rate-dependent metric with no usable rate."""
    return CheckResult(
        name=name, status=CheckStatus.SKIPPED, reason=REASON_MISSING_HZ, measurement=None, evaluation=None
    )


def _evaluate_scalar[M: Measurement, T: (int, float)](
    *,
    name: str,
    measurement: M,
    threshold: T,
    accessor: Callable[[M], T],
    reason: str,
) -> CheckResult:
    """Package one below-threshold check as a :class:`CheckResult`.

    SKIPPED when the measurement is undefined; the ``num_samples=<N>`` /
    ``insufficient data`` distinction follows whether the measurement carries a
    ``num_samples`` field (:class:`FrameReorderingPresentMeasurement` does not).
    """
    if not measurement.is_defined:
        n = getattr(measurement, "num_samples", None)
        detail = f"num_samples={n}" if n is not None else "insufficient data"
        return CheckResult(
            name=name,
            status=CheckStatus.SKIPPED,
            reason=f"measurement undefined ({detail})",
            measurement=measurement_to_dict(measurement),
            evaluation=None,
        )
    result = below_threshold(threshold=threshold, measurement=measurement, accessor=accessor)
    status = CheckStatus.PASS if result.status is EvaluationStatus.PASS else CheckStatus.FAIL
    return CheckResult(
        name=name,
        status=status,
        reason=reason,
        measurement=measurement_to_dict(measurement),
        evaluation=evaluation_to_dict(result),
    )


def _evaluate_ordering(metric: TimestampOrderingMetric, thresholds: Thresholds) -> CheckResult:
    m = metric.measurement()
    threshold = thresholds.max_strict_violations
    return _evaluate_scalar(
        name=NAME_ORDERING,
        measurement=m,
        threshold=threshold,
        accessor=lambda x: x.strict_violation_count,
        reason=f"strict_violation_count={m.strict_violation_count} (threshold={threshold})",
    )


def _evaluate_rate(metric: RateMetric, thresholds: Thresholds) -> CheckResult:
    m = metric.measurement()
    threshold = thresholds.max_rate_deviation_percent
    return _evaluate_scalar(
        name=NAME_RATE,
        measurement=m,
        threshold=threshold,
        accessor=lambda x: x.period_deviation_percent,
        reason=f"period_deviation_percent={m.period_deviation_percent:.4f} (threshold={threshold:.4f}%)",
    )


def _evaluate_gap(metric: TimestampGapMetric, thresholds: Thresholds) -> CheckResult:
    m = metric.measurement()
    threshold = thresholds.max_gaps
    return _evaluate_scalar(
        name=NAME_GAP,
        measurement=m,
        threshold=threshold,
        accessor=lambda x: x.num_gaps,
        reason=f"num_gaps={m.num_gaps} (threshold={threshold})",
    )


def _evaluate_jitter(metric: JitterMetric, thresholds: Thresholds) -> CheckResult:
    m = metric.measurement()
    threshold = thresholds.max_jitter_percent
    return _evaluate_scalar(
        name=NAME_JITTER,
        measurement=m,
        threshold=threshold,
        accessor=lambda x: x.jitter_percent,
        reason=f"jitter_percent={m.jitter_percent:.4f} (threshold={threshold:.4f}%)",
    )


def _evaluate_reordering(metric: FrameReorderingPresentMetric, thresholds: Thresholds) -> CheckResult:
    m = metric.measurement()
    # threshold 0 fails when a reordering flag is set; 1 permits it.
    threshold = 1 if thresholds.allow_frame_reordering else 0
    return _evaluate_scalar(
        name=NAME_REORDERING,
        measurement=m,
        threshold=threshold,
        accessor=lambda x: int(bool(x.has_reordering)),
        reason=f"has_reordering={bool(m.has_reordering)} (threshold={threshold})",
    )


def run_metrics(
    sensor: IntegritySensor,
    *,
    expected_hz: float | None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    batch_size: int = 0,
    stats: dict[str, float] | None = None,
) -> tuple[list[CheckResult], VideoInfo, ResolvedConfig]:
    """Run every data-integrity metric over one open sensor and evaluate them.

    Streams the sensor's timestamps once (``stream_timestamps(batch_size)``),
    folding them into the ordering / rate / gap / jitter instruments, records the
    frame-reordering flag, then evaluates each metric against ``thresholds``. A
    metric whose measurement is undefined -- or whose expected rate is unavailable
    -- is reported ``SKIPPED`` and never handed to an evaluator.

    Pure compute: the sensor is already open, so this performs no I/O and is the
    unit both CLIs share.

    Args:
        sensor: an opened sensor exposing the :class:`IntegritySensor` surface.
        expected_hz: expected sample rate in Hz; ``None`` falls back to the
            sensor's nominal ``avg_frame_rate`` (see :func:`resolve_expected_hz`).
        thresholds: pass/fail policy (see :class:`Thresholds`).
        batch_size: window size for streaming timestamps; ``0`` = one batch. The
            kernel guarantees the streaming result matches the one-shot result.
        stats: optional out-parameter; when provided, ``stream_ms`` and
            ``evaluate_ms`` wall-clock phase timings (milliseconds) are recorded.

    Returns:
        ``(results, video_info, resolved_cfg)`` with ``results`` in the fixed
        metric order (ordering, rate, gap, jitter, reordering).

    Raises:
        ValueError: if ``expected_hz`` or ``batch_size`` is invalid (see
            :func:`validate_expected_hz` / :func:`validate_non_negative_int`).

    """
    batch_size = validate_non_negative_int("batch_size", batch_size)
    resolved_cfg = resolve_expected_hz(expected_hz, sensor)
    hz = resolved_cfg.expected_hz

    ordering = TimestampOrderingMetric()
    rate = RateMetric(expected_hz=hz) if hz is not None else None
    gap = TimestampGapMetric(expected_hz=hz) if hz is not None else None
    jitter = JitterMetric(expected_hz=hz) if hz is not None else None
    timestamp_driven = [m for m in (ordering, rate, gap, jitter) if m is not None]

    t0 = time.perf_counter()
    for window in sensor.stream_timestamps(batch_size):
        for metric in timestamp_driven:
            metric.update(timestamps_ns=window)
    reordering = FrameReorderingPresentMetric()
    reordering.update(has_reordering=sensor.has_bframes)
    if stats is not None:
        stats["stream_ms"] = (time.perf_counter() - t0) * 1000

    # Ordering (correctness of the timeline itself) -> rate/gap/jitter (need a rate
    # to judge) -> codec-level reordering. Sequenced so timeline defects read first.
    t0 = time.perf_counter()
    results: list[CheckResult] = [
        _evaluate_ordering(ordering, thresholds),
        _evaluate_rate(rate, thresholds) if rate is not None else _skipped_missing_hz(NAME_RATE),
        _evaluate_gap(gap, thresholds) if gap is not None else _skipped_missing_hz(NAME_GAP),
        _evaluate_jitter(jitter, thresholds) if jitter is not None else _skipped_missing_hz(NAME_JITTER),
        _evaluate_reordering(reordering, thresholds),
    ]
    if stats is not None:
        stats["evaluate_ms"] = (time.perf_counter() - t0) * 1000
    return results, video_info(sensor), resolved_cfg


def _as_data_source(stream: BinaryIO) -> DataSource:
    """Cast a ``BinaryIO`` from ``open_cloud_source`` to a ``DataSource``.

    ``smart_open``'s S3 / Azure readers are seekable ``io.BufferedIOBase``
    subclasses, so they satisfy the ``DataSource`` union at runtime even though
    static typing only sees ``BinaryIO`` (mirrors ``check_video_index``).
    """
    return cast("DataSource", stream)


@contextmanager
def open_source(
    source: str,
    *,
    s3_profile_name: str | None,
    azure_profile_name: str,
    endpoint_url: str | None = None,
    stream_wrapper: Callable[[BinaryIO], BinaryIO] | None = None,
) -> Generator[pathlib.Path | BinaryIO]:
    """Yield a :class:`Path` for local sources and a fresh :class:`BinaryIO` for cloud URIs.

    ``stream_wrapper``, when given, wraps the cloud :class:`BinaryIO` before it is
    yielded (for example a byte-counting reader used to report download progress).
    It is a no-op for local sources, which are handed to the sensor as a
    :class:`Path` and read directly off disk.
    """
    if is_cloud_uri(source):
        with open_cloud_source(
            source,
            s3_profile_name=s3_profile_name,
            azure_profile_name=azure_profile_name,
            endpoint_url=endpoint_url,
        ) as stream:
            yield stream_wrapper(stream) if stream_wrapper is not None else stream
    else:
        yield pathlib.Path(source)


def run_checks(  # noqa: PLR0913
    source: str,
    *,
    expected_hz: float | None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    stream_idx: int = 0,
    batch_size: int = 0,
    s3_profile_name: str | None = None,
    azure_profile_name: str = "default",
    endpoint_url: str | None = None,
    stats: dict[str, float] | None = None,
    stream_wrapper: Callable[[BinaryIO], BinaryIO] | None = None,
) -> tuple[list[CheckResult], VideoInfo, ResolvedConfig]:
    """Open ``source``, build a :class:`CameraSensor`, and run every metric on it.

    The I/O wrapper around :func:`run_metrics`: it resolves the local path or
    cloud stream, constructs the sensor, and delegates the compute. Any open /
    decode failure propagates as an exception for the caller to classify (the
    single-video CLI maps it to exit code 2; the session tool records a per-stream
    ERROR).

    Args:
        source: local path, ``s3://`` URI, or ``az://`` URI.
        expected_hz: expected sample rate for the rate-dependent metrics.
        thresholds: pass/fail policy (see :class:`Thresholds`).
        stream_idx: which video stream to open (default 0).
        batch_size: timestamps per metric update; 0 feeds the whole array at once.
        s3_profile_name: optional AWS profile forwarded to ``open_cloud_source``.
        azure_profile_name: Azure profile forwarded to ``open_cloud_source``.
        endpoint_url: optional S3 endpoint override for S3-compatible stores.
        stats: optional out-parameter; ``sensor_init_ms`` is recorded here in
            addition to the ``stream_ms`` / ``evaluate_ms`` from :func:`run_metrics`.
        stream_wrapper: optional wrapper applied to the cloud stream before the
            sensor reads it (e.g. a byte-counting reader for progress); no-op for
            local sources.

    Returns:
        The tuple returned by :func:`run_metrics`.

    """
    with open_source(
        source,
        s3_profile_name=s3_profile_name,
        azure_profile_name=azure_profile_name,
        endpoint_url=endpoint_url,
        stream_wrapper=stream_wrapper,
    ) as src:
        data: DataSource = src if isinstance(src, pathlib.Path) else _as_data_source(src)
        t0 = time.perf_counter()
        sensor = CameraSensor(data, stream_idx=stream_idx)
        if stats is not None:
            stats["sensor_init_ms"] = (time.perf_counter() - t0) * 1000
        return run_metrics(sensor, expected_hz=expected_hz, thresholds=thresholds, batch_size=batch_size, stats=stats)


def overall_status(results: list[CheckResult]) -> CheckStatus:
    """FAIL if any check failed; SKIPPED never fails the run, otherwise PASS."""
    return CheckStatus.FAIL if any(r.status is CheckStatus.FAIL for r in results) else CheckStatus.PASS
