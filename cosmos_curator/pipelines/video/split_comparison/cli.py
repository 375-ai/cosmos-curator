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
r"""Combined measure + eval entry point.

Default: measure both outputs (loads + diffs + one GPU embedding pass), persist
the measurement tables, and evaluate them in the same process -- no read-back,
so eval rides the measurements while they are still hot in memory.

``--skip-measure``: re-evaluate an existing measurements root with (possibly
new) thresholds. The measurement tables are read once and run through the same
eval; no GPU, no source IO. Outputs and caption scope (``--no-captions``) are
recovered from the manifest when not re-supplied.

``--skip-eval``: measure only -- persist the measurement tables + manifest and
write no issues. Re-evaluate later with ``--skip-measure``.

  # measure + eval
  python -m cosmos_curator.pipelines.video.split_comparison.cli \\
      --output-a s3://.../run_a --output-b s3://.../run_b \\
      --measurements-path s3://.../measurements/a_vs_b

  # re-eval only, stricter caption threshold, kept under its own name
  python -m cosmos_curator.pipelines.video.split_comparison.cli \\
      --measurements-path s3://.../measurements/a_vs_b \\
      --skip-measure --min-caption-similarity 0.9 --eval-name strict
"""

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import ValidationError

from cosmos_curator.pipelines.video.split_comparison import report, store, summary
from cosmos_curator.pipelines.video.split_comparison.config import DEFAULT_PROFILE_NAME, SplitComparisonConfig
from cosmos_curator.pipelines.video.split_comparison.eval import evaluate
from cosmos_curator.pipelines.video.split_comparison.load import DEFAULT_LANCE_VERSION
from cosmos_curator.pipelines.video.split_comparison.measure.core import Measurements
from cosmos_curator.pipelines.video.split_comparison.result_model import Issue


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch to a subcommand (``score-histogram``) or run the default measure/eval flow."""
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "score-histogram":
        return _score_histogram_main(raw[1:])
    return _measure_eval_main(raw)


def _score_histogram_main(argv: Sequence[str]) -> int:
    """Print the all-clips score distributions for a measurements root (text / JSON / PNG)."""
    parser = argparse.ArgumentParser(
        prog="cosmos-curator split-compare score-histogram",
        description="Bucket the all-clips score distributions of a measurements root, with the "
        "policy thresholds marked. Reads clip.lance / window.lance / eval.json.",
    )
    parser.add_argument("--measurements-path", required=True, help="Measurements root to read.")
    parser.add_argument("--profile", default=None, help="Storage profile for reads (default: 'default').")
    parser.add_argument(
        "--eval-name",
        default=None,
        help="Read thresholds from eval/<name>/eval.json (match the run's --eval-name); default: root eval.json.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        choices=report.METRIC_NAMES,
        dest="metrics",
        help="Histogram(s) to show (repeatable); default: all. Choices: " + ", ".join(report.METRIC_NAMES),
    )
    parser.add_argument(
        "--bins", type=int, default=report.DEFAULT_BINS, help="Histogram bin count (default: %(default)s)."
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON (buckets/counts/stats) instead of text bars.")
    parser.add_argument("--png", metavar="PATH", help="Also write a PNG figure to PATH (requires matplotlib).")
    parser.add_argument("--png-title", metavar="TEXT", help="Title (suptitle) for the PNG figure.")
    args = parser.parse_args(argv)
    if args.bins < 1:
        sys.stderr.write("--bins must be >= 1\n")
        return 2

    profile = args.profile or DEFAULT_PROFILE_NAME
    try:
        histograms = report.build_histograms(
            args.measurements_path,
            eval_name=args.eval_name,
            profile=profile,
            bins=args.bins,
            metrics=report.metrics_for(args.metrics),
        )
    except (OSError, ValueError) as err:
        sys.stderr.write(f"Failed to build histograms: {err}\n")
        return 2

    if histograms and all(h.threshold is None for h in histograms):
        # A missing/unreadable eval.json (or an --eval-name mismatch) leaves every bucket untagged;
        # say so on stderr rather than emit an unmarked report that reads like "nothing failing".
        sys.stderr.write(
            "warning: no policy thresholds found (missing/unreadable eval.json or --eval-name mismatch); "
            "buckets are not tagged passing/failing.\n"
        )

    if args.json:
        sys.stdout.write(json.dumps(report.to_json(histograms), indent=2) + "\n")
    else:
        sys.stdout.write(report.render_text(histograms))
    if args.png:
        try:
            report.render_png(histograms, args.png, profile=profile, title=args.png_title)
        except (RuntimeError, OSError, ValueError) as err:
            sys.stderr.write(f"{err}\n")
            return 2
        logger.info("Wrote PNG to {}", args.png)
    return 0


def _measure_eval_main(argv: Sequence[str]) -> int:
    """Run measure and/or eval per the skip flags; write outputs; return an exit code."""
    args = _build_parser().parse_args(argv)
    if args.skip_measure and args.skip_eval:
        sys.stderr.write("--skip-measure and --skip-eval together leave nothing to do\n")
        return 2
    try:
        config = _resolve_config(args)
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as err:
        sys.stderr.write(f"Failed to build config: {err}\n")
        return 2

    started = time.perf_counter()
    if args.skip_measure:
        logger.info("Skipping measure; reading measurements from {}", args.measurements_path)
        measurements = store.read_measurements(args.measurements_path, profile_name=config.profile_name)
    else:
        try:
            measurements = _measure_and_persist(config, args)
        except ValueError as err:
            # A bad --num-gpus override or a GPU-less host surfaces as a ValueError from
            # resolve_num_gpus(); report it cleanly as exit 2 (bad setup) rather than letting
            # it escape as a traceback that exits 1 -- which would collide with "issues found".
            sys.stderr.write(f"Measure failed: {err}\n")
            return 2

    if args.skip_eval:
        # Measure-only: measurements + manifest are persisted; no eval outputs.
        elapsed = time.perf_counter() - started
        sys.stdout.write(_format_measure_summary(measurements, args, elapsed_sec=elapsed))
        return 0

    compared_summaries = not args.no_summary and _summaries_available(args, config)
    summary_issues: list[Issue] = []
    if compared_summaries:
        summary_issues = summary.summary_issues(
            args.measurements_path, profile_name=config.profile_name, policy=config.summary
        )
    result = evaluate(measurements, config=config, summary_issues=summary_issues)
    provenance = {
        "output_a": config.output_a,
        "output_b": config.output_b,
        "skip_measure": args.skip_measure,
        "summary_compared": compared_summaries,
        "device": None if args.skip_measure else "cuda",
        "fp16": None if args.skip_measure else args.fp16,
        "lance_version": args.lance_version,
    }
    payload = store.write_eval(
        result,
        args.measurements_path,
        eval_name=args.eval_name,
        profile_name=config.profile_name,
        provenance=provenance,
    )
    elapsed = time.perf_counter() - started
    sys.stdout.write(_format_summary(payload, args, elapsed_sec=elapsed))
    # Non-zero exit when any issue fired, so this composes in scripts like the v1 CLI.
    return 0 if payload.get("total_issues", 0) == 0 else 1


def _measure_and_persist(config: SplitComparisonConfig, args: argparse.Namespace) -> Measurements:
    # Lazy import so the re-eval path (--skip-measure) never loads Ray.
    from cosmos_curator.pipelines.video.split_comparison.measure.ray import run as run_measure  # noqa: PLC0415

    measurements = run_measure(
        config,
        lance_version=args.lance_version,
        num_gpus=args.num_gpus,
        show_progress=args.progress,
        fp16=args.fp16,
    )
    store.write_measurements(
        measurements,
        args.measurements_path,
        config=config,
        device="cuda",
        fp16=args.fp16,
        lance_version=args.lance_version,
        summaries_snapshotted=not args.no_summary,
    )
    if not args.no_summary:
        # Snapshot both summary.json into the measurements root so summary eval (now
        # or on later re-eval) stays self-contained.
        status = store.snapshot_summaries(config, args.measurements_path)
        logger.info("Snapshotted summaries: {}", status)
    return measurements


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cosmos-curator split-compare",
        description="Measure two split outputs and evaluate the measurements (Ray multi-GPU by default).",
    )
    parser.add_argument("--config", type=Path, metavar="PATH", help="JSON file conforming to SplitComparisonConfig.")
    parser.add_argument("--output-a", help="Output root A (required to measure; recovered from manifest on re-eval).")
    parser.add_argument("--output-b", help="Output root B (required to measure; recovered from manifest on re-eval).")
    parser.add_argument("--measurements-path", required=True, help="Measurements root (read and/or written here).")
    parser.add_argument("--profile", default=None, help="Storage profile for reads/writes (default: 'default').")
    parser.add_argument(
        "--num-gpus",
        type=int,
        help="Number of GPU actors for the measure fan-out (default: Ray's detected GPU count).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Run the caption model in half precision on GPU (~2x faster forward; shifts similarities slightly). "
        "No-op off-GPU.",
    )
    parser.add_argument("--lance-version", default=DEFAULT_LANCE_VERSION, help="Source Lance dataset version subdir.")
    parser.add_argument("--no-captions", action="store_true", help="Skip caption measurement/eval (metadata only).")
    parser.add_argument("--min-caption-similarity", type=float, help="Override caption similarity threshold (0..1).")
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Caption embedding batch (GPU encode chunk); also the block-count target -- smaller means "
        "more blocks and better load balancing (default 128).",
    )
    parser.add_argument(
        "--summary-token-rel-tol",
        type=float,
        help="Override summary token-count relative tolerance (0..1); suppresses benign token-total drift.",
    )
    parser.add_argument("--skip-measure", action="store_true", help="Re-eval an existing measurements root only.")
    parser.add_argument("--skip-eval", action="store_true", help="Measure only; persist measurements, write no issues.")
    parser.add_argument("--no-summary", action="store_true", help="Skip summary.json snapshot + comparison.")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show Ray Data's rich progress bars during the measure fan-out.",
    )
    parser.add_argument("--eval-name", help="Write eval outputs under eval/<name>/ instead of the root.")
    return parser


def _resolve_config(args: argparse.Namespace) -> SplitComparisonConfig:
    """Build and validate the config from ``--config`` (a full spec file) or from flags.

    ``--config`` is the complete comparison spec; it is mutually exclusive with the
    config-shaping flags (targets, profile, and policy overrides) -- combining them is
    rejected rather than silently dropping the flag. In flags mode the inputs are
    assembled into a dict and validated in one construction, so out-of-range values
    (e.g. ``--batch-size 0``) raise ``ValidationError`` here rather than deep in the run.
    On re-eval, measure-time facts the caller didn't re-supply are recovered from the
    manifest: outputs (via :func:`_resolve_outputs`) and caption scope (via
    :func:`_resolve_compare_captions`).
    """
    shaping = {
        "--output-a": args.output_a is not None,
        "--output-b": args.output_b is not None,
        "--profile": args.profile is not None,
        "--no-captions": args.no_captions,
        "--min-caption-similarity": args.min_caption_similarity is not None,
        "--batch-size": args.batch_size is not None,
        "--summary-token-rel-tol": args.summary_token_rel_tol is not None,
    }
    if args.config is not None:
        conflicting = [flag for flag, given in shaping.items() if given]
        if conflicting:
            joined = ", ".join(conflicting)
            msg = f"--config is the full spec; it cannot be combined with {joined} (edit the config file instead)"
            raise ValueError(msg)
        return SplitComparisonConfig.model_validate(json.loads(args.config.read_text()))

    profile_name = args.profile or DEFAULT_PROFILE_NAME
    output_a, output_b = _resolve_outputs(args, profile_name)
    data: dict[str, Any] = {"output_a": output_a, "output_b": output_b, "profile_name": profile_name}
    data["compare_captions"] = _resolve_compare_captions(args, profile_name)
    if args.min_caption_similarity is not None:
        data.setdefault("caption", {})["min_similarity"] = args.min_caption_similarity
    if args.batch_size is not None:
        data.setdefault("caption", {})["encode_batch_size"] = args.batch_size
    if args.summary_token_rel_tol is not None:
        data.setdefault("summary", {})["token_count_rel_tolerance"] = args.summary_token_rel_tol
    return SplitComparisonConfig.model_validate(data)


def _resolve_outputs(args: argparse.Namespace, profile_name: str) -> tuple[str, str]:
    """Resolve (output_a, output_b): from flags, or the manifest when re-evaluating."""
    output_a, output_b = args.output_a, args.output_b
    if args.skip_measure and not (output_a and output_b):
        manifest = store.read_manifest(args.measurements_path, profile_name=profile_name)
        output_a = output_a or manifest.get("output_a")
        output_b = output_b or manifest.get("output_b")
    if not (output_a and output_b):
        msg = "both --output-a and --output-b are required (unless re-evaluating a manifest that records them)"
        raise ValueError(msg)
    return output_a, output_b


def _resolve_compare_captions(args: argparse.Namespace, profile_name: str) -> bool:
    """Resolve compare_captions: ``--no-captions`` forces it off; else recover from the manifest.

    A root measured ``--no-captions`` holds no caption/window data, so a ``--skip-measure``
    re-eval must not turn caption comparison back on just because the flag wasn't repeated --
    the manifest is authoritative, mirroring how outputs (:func:`_resolve_outputs`) and summary
    scope (:func:`_summaries_available`) are recovered. Passing ``--no-captions`` still narrows a
    captioned root to metadata-only. Defaults to ``True`` for a fresh measure and for legacy
    manifests without the field (prior behavior).
    """
    if args.no_captions:
        return False
    if args.skip_measure:
        manifest = store.read_manifest(args.measurements_path, profile_name=profile_name)
        return bool(manifest.get("compare_captions", True))
    return True


def _summaries_available(args: argparse.Namespace, config: SplitComparisonConfig) -> bool:
    """Whether summary snapshots exist in the measurements root to compare.

    A fresh measure controls snapshotting in-process (gated by ``--no-summary``); on
    re-eval (``--skip-measure``) the answer comes from the manifest the measure wrote,
    so a root measured ``--no-summary`` isn't compared against absent snapshots.
    Defaults to ``True`` for legacy manifests without the flag (prior behavior).
    """
    if not args.skip_measure:
        return True
    manifest = store.read_manifest(args.measurements_path, profile_name=config.profile_name)
    return bool(manifest.get("summaries_snapshotted", True))


def _format_measure_summary(measurements: Measurements, args: argparse.Namespace, *, elapsed_sec: float) -> str:
    lines = [
        "split comparison (measure-only) complete",
        f"  stats: {measurements.stats}",
        f"  runtime: {elapsed_sec:.2f}s",
        f"  measurements: {args.measurements_path}",
    ]
    return "\n".join(lines) + "\n"


def _format_summary(payload: dict[str, object], args: argparse.Namespace, *, elapsed_sec: float) -> str:
    mode = "re-eval" if args.skip_measure else "measure+eval"
    out_dir = store.eval_output_dir(args.measurements_path, args.eval_name)
    lines = [
        f"split comparison ({mode}) complete",
        f"  clips: total={payload.get('total_clips')} passed={payload.get('clips_passed')} "
        f"failed={payload.get('clips_failed')}",
        f"  issues: {payload.get('total_issues')} by_code={payload.get('issues_by_code')}",
        f"  runtime: {elapsed_sec:.2f}s",
        f"  eval outputs: {out_dir}",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    logger.disable("ray")
    sys.exit(main())
