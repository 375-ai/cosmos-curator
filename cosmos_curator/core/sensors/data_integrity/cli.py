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
# plain uv venv is the simplest cross-platform path:
#
#   VENV=~/.venvs/cosmos-curator-di
#   uv venv --python 3.13 $VENV
#   uv pip install --python $VENV/bin/python attrs numpy av 'smart_open[s3,azure]' azure-identity loguru
#   uv pip install --python $VENV/bin/python --no-deps -e .
#   alias di-check="$VENV/bin/python -m cosmos_curator.core.sensors.data_integrity.cli"
#
# Run against an S3 clip (add --json or --perf-profile; --help lists all flags):
#
#   di-check --source /path/to/local/clip.mp4
#   di-check --source s3://bucket/path/clip.mp4 --s3-profile-name my-profile
#

"""Run every data-integrity metric against a single video and report the verdict.

Thin CLI wrapper: the per-stream engine (running the metrics, judging them, and
serialising measurements) lives in
:mod:`cosmos_curator.core.sensors.data_integrity.cli_common`, shared with the
single-session tool. This module owns only the single-video report shapes, the
argument surface, and the exit-code contract.
"""

import argparse
import json
import sys
import time

from cosmos_curator.core.sensors.data_integrity.cli_common import (
    CheckResult,
    CheckStatus,
    ResolvedConfig,
    VideoInfo,
    non_negative_int,
    overall_status,
    positive_finite_float,
    run_checks,
)
from cosmos_curator.core.sensors.scripts._cli_cloud import (
    CloudCliError,
    add_cloud_credential_args,
    resolve_s3_endpoint_url,
    validate_source,
)

PASS_EXIT_CODE = 0
FAIL_EXIT_CODE = 1
ERROR_EXIT_CODE = 2


def _format_expected_hz_line(resolved_cfg: ResolvedConfig) -> str:
    """Format the ``expected_hz:`` line for the human-readable report header.

    Uses the enum ``.value`` directly so the ``source:`` annotation matches the
    JSON payload's ``expected_hz_source`` field verbatim (``user`` / ``header`` /
    ``unavailable``). New variants automatically render without additional wiring.
    """
    value = f"{resolved_cfg.expected_hz:.3f}" if resolved_cfg.expected_hz is not None else "N/A"
    return f"  expected_hz: {value} (source: {resolved_cfg.expected_hz_source.value})\n"


def _render_human(
    source: str,
    video_info: VideoInfo,
    resolved_cfg: ResolvedConfig,
    results: list[CheckResult],
) -> str:
    header = (
        f"Data-integrity report for: {source}\n"
        f"  codec: {video_info.codec_name}"
        f"   has_bframes: {str(video_info.has_bframes).lower()}"
        f"   num_samples: {video_info.num_samples}"
        f"   start_ns: {video_info.start_ns}"
        f"   end_ns: {video_info.end_ns}\n"
        f"{_format_expected_hz_line(resolved_cfg)}\n"
    )
    name_width = max(len(r.name) for r in results)
    lines = [f"  {r.name.ljust(name_width)}  {r.status.value:<7}  {r.reason}\n" for r in results]
    overall = overall_status(results)
    return header + "".join(lines) + f"\nOverall: {overall.value}\n"


def _render_json(
    source: str,
    effective_batch_size: int,
    video_info: VideoInfo,
    resolved_cfg: ResolvedConfig,
    results: list[CheckResult],
) -> str:
    payload: dict[str, object] = {
        "source": source,
        "expected_hz": resolved_cfg.expected_hz,
        "expected_hz_source": resolved_cfg.expected_hz_source.value,
        "batch_size": effective_batch_size,
        "video": video_info.to_dict(),
        "checks": [
            {
                "name": r.name,
                "status": r.status.value,
                "reason": r.reason,
                "measurement": r.measurement,
                "evaluation": r.evaluation,
            }
            for r in results
        ],
        "overall_status": overall_status(results).value,
    }
    return json.dumps(payload, indent=2, allow_nan=False) + "\n"


def _render_perf(stats: dict[str, float], *, video_info: VideoInfo, batch_size: int) -> str:
    """Format per-phase wall-clock timings for stderr.

    One ``key = value`` line per field, greppable and stable for scripting. Timing
    keys are milliseconds. Emitted on stderr so ``--json`` stdout stays a clean
    single-document payload.
    """
    total_ms = sum(stats.values())
    lines = [
        f"  sensor_init_ms  = {stats.get('sensor_init_ms', 0.0):.3f}",
        f"  stream_ms       = {stats.get('stream_ms', 0.0):.3f}",
        f"  evaluate_ms     = {stats.get('evaluate_ms', 0.0):.3f}",
        f"  render_ms       = {stats.get('render_ms', 0.0):.3f}",
        f"  total_ms        = {total_ms:.3f}",
        f"  num_samples     = {video_info.num_samples}",
        f"  has_bframes     = {video_info.has_bframes}",
        f"  codec           = {video_info.codec_name}",
        f"  batch_size      = {batch_size}",
    ]
    # Leading newline so the block visually separates from the report above (which
    # ends with "Overall: <status>\n" in human mode or the closing "}\n" in JSON mode).
    return "\nPer-phase performance stats:\n" + "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="di-check",
        description="Run every data-integrity metric against a single video and print the verdict.",
        epilog=(
            "Exit codes: 0 = every check PASS or SKIPPED; 1 = at least one FAIL; "
            "2 = input, credential, or runtime error."
        ),
    )
    parser.add_argument(
        "--source",
        required=True,
        metavar="URI",
        help="Local path, s3://, or az:// URI to the video file.",
    )
    parser.add_argument(
        "--expected-hz",
        type=positive_finite_float,
        default=None,
        metavar="HZ",
        help=(
            "Authoritative expected sample rate for the rate, gap, and jitter checks. "
            "Defaults to the container's declared rate, which is best-effort and will not catch a "
            "uniformly wrong capture rate; SKIPs those checks if neither is available."
        ),
    )
    parser.add_argument(
        "--stream-idx",
        type=non_negative_int,
        default=0,
        metavar="N",
        help="Video stream index (default: 0).",
    )
    parser.add_argument(
        "--batch-size",
        type=non_negative_int,
        default=0,
        metavar="N",
        help="Timestamps per metric update; 0 (default) feeds the whole array at once.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the JSON report to stdout.")
    parser.add_argument(
        "--perf-profile",
        action="store_true",
        help="Emit per-phase wall-clock timings to stderr after the report.",
    )
    add_cloud_credential_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run every data-integrity metric on the supplied video and print the report."""
    args = _parse_args(argv)
    stats: dict[str, float] | None = {} if args.perf_profile else None
    try:
        validate_source(args.source)
        results, video_info, resolved_cfg = run_checks(
            args.source,
            expected_hz=args.expected_hz,
            stream_idx=args.stream_idx,
            batch_size=args.batch_size,
            s3_profile_name=args.s3_profile_name,
            azure_profile_name=args.azure_profile_name,
            endpoint_url=resolve_s3_endpoint_url(args.endpoint_url),
            stats=stats,
        )

        # Rendering and writing stay inside the handler: serializing a report can fail
        # (non-finite measurements) and so can the write itself (a closed pipe), and both
        # owe the caller exit code 2 rather than a traceback.
        t_render_start = time.perf_counter()
        output: str = (
            _render_json(args.source, args.batch_size, video_info, resolved_cfg, results)
            if args.json
            else _render_human(args.source, video_info, resolved_cfg, results)
        )
        if stats is not None:
            stats["render_ms"] = (time.perf_counter() - t_render_start) * 1000

        sys.stdout.write(output)
        if stats is not None:
            # Flush the report first so interactive users see it above the perf block.
            sys.stdout.flush()
            sys.stderr.write(_render_perf(stats, video_info=video_info, batch_size=args.batch_size))
        return FAIL_EXIT_CODE if overall_status(results) is CheckStatus.FAIL else PASS_EXIT_CODE
    except CloudCliError as e:
        sys.stderr.write(f"error: {e}\n")
        return ERROR_EXIT_CODE
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"error: could not evaluate data integrity for {args.source!r}: {e}\n")
        return ERROR_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
