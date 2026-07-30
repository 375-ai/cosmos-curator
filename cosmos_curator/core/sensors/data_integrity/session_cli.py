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
#

# LOCAL DEV SETUP
#
# Same environment as the single-video di-check CLI (see cli.py). Once that venv
# exists:
#
#   alias di-session="$VENV/bin/python -m cosmos_curator.core.sensors.data_integrity.session_cli"
#
# Run against one session (a clips/<uuid>/ directory or cloud prefix):
#
#   di-session --session-path /path/to/clips/<uuid>
#   di-session --session-path s3://bucket/clips/<uuid>/ --s3-profile-name my-profile --json
#   di-session --session-path s3://hyperion8/clips/<uuid>/ --endpoint-url <endpoint-url> --limit 5
#

"""Run every data-integrity metric against every stream of a single session.

A *session* is one recording -- typically a ``clips/<uuid>/`` directory (or cloud
prefix) holding one video per camera. This CLI discovers those streams, runs the
shared per-stream engine
(:mod:`cosmos_curator.core.sensors.data_integrity.cli_common`) on each, and prints
a per-stream + session-level verdict. It is the session-level companion to the
single-video ``di-check`` CLI (:mod:`.cli`) and reuses the same engine, cloud
credential flags, and reason strings.
"""

import argparse
import io
import sys
import threading
import time
from collections.abc import Callable
from typing import BinaryIO, cast

from cosmos_curator.core.sensors.data_integrity.cli_common import (
    available_cpu_count,
    non_negative_int,
    positive_finite_float,
    positive_int,
)
from cosmos_curator.core.sensors.data_integrity.report import OverallStatus, StreamResult, render_text, to_json
from cosmos_curator.core.sensors.data_integrity.session_runner import run_session
from cosmos_curator.core.sensors.scripts._cli_cloud import (
    CloudCliError,
    add_cloud_credential_args,
    get_cloud_object_size,
    is_cloud_uri,
    resolve_s3_endpoint_url,
)

PASS_EXIT_CODE = 0
FAIL_EXIT_CODE = 1
ERROR_EXIT_CODE = 2

# Session verdict -> process exit code. Mirrors the single-video CLI's contract:
# PASS = 0, FAIL = 1, and anything that could not be checked (ERROR) = 2.
_EXIT_CODES = {
    OverallStatus.PASS: PASS_EXIT_CODE,
    OverallStatus.FAIL: FAIL_EXIT_CODE,
    OverallStatus.ERROR: ERROR_EXIT_CODE,
}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="di-session",
        description="Run every data-integrity metric against every stream of a single session.",
        epilog=(
            "Exit codes: 0 = every stream PASS; 1 = at least one stream FAIL and none errored; "
            "2 = a stream could not be opened/decoded, no streams were discovered, or an input error occurred. "
            "A stream that could not be measured outranks a FAIL, since the session verdict stays incomplete."
        ),
    )
    parser.add_argument(
        "--session-path",
        required=True,
        metavar="PATH",
        help=(
            "Local directory or file, or an s3:// / az:// prefix for one session (typically a single clips/<uuid>/)."
        ),
    )
    parser.add_argument(
        "--expected-hz",
        type=positive_finite_float,
        default=None,
        metavar="HZ",
        help=(
            "Authoritative expected sample rate for the rate, gap, and jitter checks, applied to every "
            "stream. Defaults to each stream's declared rate, which is best-effort and will not catch a "
            "uniformly wrong capture rate; SKIPs those checks where neither is available."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=non_negative_int,
        default=0,
        metavar="N",
        help="Timestamps per metric update; 0 (default) feeds each stream's whole array at once.",
    )
    parser.add_argument(
        "--limit",
        type=non_negative_int,
        default=0,
        metavar="N",
        help="Cap the number of streams checked; 0 (default) means all. Useful for sampling a large session.",
    )
    default_max_workers = available_cpu_count()
    parser.add_argument(
        "--max-workers",
        type=positive_int,
        default=default_max_workers,
        metavar="N",
        help=(
            f"Streams to check concurrently (default: one per usable CPU, so {default_max_workers} here). "
            "Each stream spends most of its time waiting on the source, so overlapping them helps most "
            "on cloud sessions. Use 1 for serial checking and a live byte counter."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit the JSON report to stdout.")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Print a per-stream progress line to stderr as each stream is opened and finished "
            "(default: on). stdout stays a clean report / JSON payload. Use --no-progress to silence."
        ),
    )
    add_cloud_credential_args(parser)
    return parser.parse_args(argv)


_PROGRESS_MIN_INTERVAL_S = 0.5  # throttle the in-place byte-counter refresh


class _CountingReader(io.BufferedIOBase):
    """Wrap a binary stream and report cumulative bytes read via a callback.

    Every ``read`` is forwarded to the wrapped stream and the running total is
    handed to ``on_bytes``. ``readinto`` / ``read1`` route through ``read`` so the
    count is captured whichever access pattern PyAV uses. ``close`` is a no-op: the
    underlying cloud stream is owned by ``open_cloud_source``'s context manager, so
    the sensor closing this wrapper must not close it out from under that manager.
    """

    def __init__(self, raw: BinaryIO, on_bytes: Callable[[int], None]) -> None:
        super().__init__()
        self._raw = raw
        self._on_bytes = on_bytes
        self._count = 0

    def read(self, size: int | None = -1) -> bytes:
        data = self._raw.read(size if size is not None else -1)
        if data:
            self._count += len(data)
            self._on_bytes(self._count)
        return data

    def read1(self, size: int = -1) -> bytes:
        return self.read(size)

    def readinto(self, b: "memoryview | bytearray") -> int:  # type: ignore[override]
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self._raw.seek(offset, whence)

    def tell(self) -> int:
        return self._raw.tell()

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True


class _Progress:
    """Per-stream progress printer for :func:`run_session` (stderr only).

    Prints the source on start, the verdict on finish, and -- when checking one
    stream at a time -- an in-place refreshing ``N.N MB read`` counter in between.
    Concurrent streams cannot share that one redrawn line, so with several in flight
    the counter is dropped and only the start/finish lines are printed; the byte
    total still lands on the finish line. All output goes to stderr so the report /
    ``--json`` on stdout stays a clean single document.

    Hooks are called from :func:`run_session`'s worker threads, so writes are
    serialised on a lock to keep lines from interleaving mid-write.
    """

    def __init__(
        self,
        size_lookup: Callable[[str], int | None] | None = None,
        *,
        live_byte_counter: bool = True,
    ) -> None:
        self._bytes: dict[int, int] = {}
        self._last_emit: dict[int, float] = {}
        self._size_lookup = size_lookup
        self._live_byte_counter = live_byte_counter
        self._lock = threading.Lock()

    def _write(self, text: str) -> None:
        with self._lock:
            sys.stderr.write(text)
            sys.stderr.flush()

    def start(self, index: int, total: int, source: str) -> None:
        self._write(f"[{index}/{total}] {source}\n")

    def make_wrapper(self, index: int, total: int, source: str) -> Callable[[BinaryIO], BinaryIO]:
        # One cheap HEAD up front so the counter can show read / total (pct%);
        # None (unknown / lookup disabled) falls back to a bare "read" figure. The lookup
        # is injected, so guard it here instead of trusting every implementation to be
        # exception-safe -- a cosmetic byte count must never abandon the streams behind it.
        #
        # Skipped outright without the live counter, because that counter is the only
        # thing that reads the total: the finish line reports bytes actually read. The
        # lookup is not free enough to spend on a discarded value -- on a cloud source it
        # builds a fresh client per stream, and that parsing holds the interpreter lock,
        # so a dozen concurrent streams serialise behind it.
        total_bytes: int | None = None
        if self._live_byte_counter and self._size_lookup is not None:
            try:
                total_bytes = self._size_lookup(source)
            except Exception:  # noqa: BLE001 - progress detail only; unknown size is fine
                total_bytes = None

        def _wrap(stream: BinaryIO) -> BinaryIO:
            def _on_bytes(count: int) -> None:
                self._bytes[index] = count
                if not self._live_byte_counter:
                    return
                now = time.monotonic()
                if now - self._last_emit.get(index, 0.0) >= _PROGRESS_MIN_INTERVAL_S:
                    self._last_emit[index] = now
                    if total_bytes:
                        pct = min(100, int(count * 100 / total_bytes))
                        detail = f"{count / 1e6:.1f}/{total_bytes / 1e6:.1f} MB ({pct}%)"
                    else:
                        detail = f"{count / 1e6:.1f} MB read"
                    self._write(f"\r[{index}/{total}]   {detail}")

            return cast("BinaryIO", _CountingReader(stream, _on_bytes))

        return _wrap

    def finish(self, index: int, total: int, result: StreamResult) -> None:
        mb = self._bytes.get(index, 0) / 1e6
        tail = f" ({mb:.1f} MB read)" if mb else ""
        # Leading \r + trailing pad overwrite the in-place byte counter on this line;
        # with no counter drawn there is nothing to overwrite.
        prefix, pad = ("\r", "          ") if self._live_byte_counter else ("", "")
        self._write(f"{prefix}[{index}/{total}] -> {result.status.value}{tail}{pad}\n")


def main(argv: list[str] | None = None) -> int:
    """Discover and check every stream in the session, then print the report."""
    args = _parse_args(argv)
    endpoint_url = resolve_s3_endpoint_url(args.endpoint_url)

    def _size_lookup(source: str) -> int | None:
        if not is_cloud_uri(source):
            return None
        return get_cloud_object_size(
            source,
            s3_profile_name=args.s3_profile_name,
            azure_profile_name=args.azure_profile_name,
            endpoint_url=endpoint_url,
        )

    progress = _Progress(size_lookup=_size_lookup, live_byte_counter=args.max_workers == 1) if args.progress else None
    try:
        report = run_session(
            args.session_path,
            expected_hz=args.expected_hz,
            batch_size=args.batch_size,
            limit=args.limit,
            s3_profile_name=args.s3_profile_name,
            azure_profile_name=args.azure_profile_name,
            endpoint_url=endpoint_url,
            max_workers=args.max_workers,
            on_stream_start=progress.start if progress else None,
            on_stream_finish=progress.finish if progress else None,
            make_stream_wrapper=progress.make_wrapper if progress else None,
        )
        # Rendering and writing stay inside the handler: serializing a report can fail
        # (non-finite measurements) and so can the write itself (a closed pipe), and both
        # owe the caller exit code 2 rather than a traceback.
        sys.stdout.write((to_json(report) if args.json else render_text(report)) + "\n")
        return _EXIT_CODES[report.status]
    except (CloudCliError, FileNotFoundError) as e:
        sys.stderr.write(f"error: {e}\n")
        return ERROR_EXIT_CODE
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"error: could not check session {args.session_path!r}: {e}\n")
        return ERROR_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
