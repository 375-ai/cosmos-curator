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
"""Eval phase: turn measurement tables into issues + per-clip verdicts.

Eval consumes the *measurement* tables (clip + window), never the source
datasets -- so the identical code path serves both the combined run (tables
still in memory from the measure phase, no read-back) and standalone re-eval
(tables read from a measurements root). No GPU, no source IO: applying the
policy to recorded diffs is a cheap scan, which is what makes re-evaluation
under different thresholds fast.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, cast

import pyarrow as pa
from loguru import logger

from cosmos_curator.pipelines.video.split_comparison.config import ScoreTolerance, SplitComparisonConfig
from cosmos_curator.pipelines.video.split_comparison.eval_schema import CLIP_VERDICT_SCHEMA
from cosmos_curator.pipelines.video.split_comparison.measure.core import Measurements
from cosmos_curator.pipelines.video.split_comparison.result_model import (
    ISSUE_SCHEMA,
    Issue,
    IssueCode,
    make_issue,
)

_FEATURE_STRUCTURE = "metadata_structure"
_FEATURE_AESTHETIC = "aesthetic_score"
_FEATURE_MOTION = "motion_score"
_FEATURE_CAPTIONS = "captions"


class _ScoreCheck(NamedTuple):
    """One scalar-score check: which measurement columns to read, how to label the issue, which tolerance."""

    prefix: str  # measurement column prefix, e.g. "aesthetic" -> aesthetic_a/_b/_abs_diff
    field: str  # issue field name, e.g. "aesthetic_score"
    feature: str
    code: IssueCode  # mismatch code emitted when out of tolerance
    policy_attr: str  # SplitComparisonConfig attribute holding this check's ScoreTolerance


_SCORE_CHECKS: tuple[_ScoreCheck, ...] = (
    _ScoreCheck("aesthetic", "aesthetic_score", _FEATURE_AESTHETIC, "aesthetic_score_mismatch", "aesthetic"),
    _ScoreCheck("motion_global_mean", "motion_score.global_mean", _FEATURE_MOTION, "motion_score_mismatch", "motion"),
    _ScoreCheck(
        "motion_per_patch_min_256",
        "motion_score.per_patch_min_256",
        _FEATURE_MOTION,
        "motion_score_mismatch",
        "motion",
    ),
)

# Equality measurements: (clip-row equal column, issue field name).
_EQUALITY_CHECKS: tuple[tuple[str, str], ...] = (
    ("valid_equal", "valid"),
    ("has_caption_equal", "has_caption"),
    ("rejection_stage_equal", "rejection_stage"),
)

# Score diff columns rolled into the verdict's max_score_abs_diff.
_ABS_DIFF_COLUMNS: tuple[str, ...] = (
    "aesthetic_abs_diff",
    "motion_global_mean_abs_diff",
    "motion_per_patch_min_256_abs_diff",
)


class EvalResult(NamedTuple):
    """The eval phase's output: issues, per-clip verdicts, and a summary block."""

    issues: pa.Table
    verdicts: pa.Table
    summary: dict[str, Any]


def evaluate(
    measurements: Measurements,
    *,
    config: SplitComparisonConfig,
    summary_issues: Sequence[Issue] = (),
) -> EvalResult:
    """Apply ``config``'s tolerances to the measurement tables; return issues + verdicts.

    Tolerances come from ``config.aesthetic`` / ``config.motion`` and the caption
    threshold from ``config.caption.min_similarity`` -- the same config that
    drives measure, so re-eval is a tolerance change, not a new model.

    ``summary_issues`` are pre-computed summary-level (summary.json) issues folded
    into the issue table and counts; they carry no clip id, so per-clip verdicts
    are unaffected.
    """
    clip_rows = measurements.clip_table.to_pylist()
    # --no-captions (compare_captions False) is metadata-only: skip caption-window eval
    # entirely -- including on re-eval against a root that *was* measured with captions.
    # Gating here (not just at measure) keeps it from materializing/scanning window.lance.
    window_rows = measurements.window_table.to_pylist() if config.compare_captions else []
    logger.info(
        "Evaluating {} clip rows + {} window rows ({} summary issues)",
        len(clip_rows),
        len(window_rows),
        len(summary_issues),
    )

    issues: list[Issue] = []
    issues_per_clip: Counter[str] = Counter()
    for row in clip_rows:
        clip_issues = _eval_clip(row, config)
        issues.extend(clip_issues)
        issues_per_clip[row["clip_uuid"]] += len(clip_issues)
    for row in window_rows:
        window_issues = _eval_window(row, config)
        issues.extend(window_issues)
        issues_per_clip[row["clip_uuid"]] += len(window_issues)
    issues.extend(summary_issues)

    verdicts = [
        _verdict_row(row, issues_per_clip[row["clip_uuid"]], compare_captions=config.compare_captions)
        for row in clip_rows
    ]
    issue_table = pa.Table.from_pylist(issues, schema=ISSUE_SCHEMA)
    verdict_table = pa.Table.from_pylist(verdicts, schema=CLIP_VERDICT_SCHEMA)
    return EvalResult(
        issues=issue_table,
        verdicts=verdict_table,
        summary=_summary(verdicts, issue_table, config=config),
    )


def _eval_clip(row: Mapping[str, Any], config: SplitComparisonConfig) -> list[Issue]:
    """Evaluate one clip measurement row into issues."""
    clip_uuid = row["clip_uuid"]
    video = row.get("video_uuid")
    present_a = row["present_a"]
    present_b = row["present_b"]
    if present_a != present_b:
        # One-sided clip: nothing to compare beyond presence itself.
        return [
            make_issue(
                code="metadata_one_sided",
                message="Clip present in only one output",
                feature=_FEATURE_STRUCTURE,
                video=video,
                clip=clip_uuid,
                output="b" if present_a else "a",
            ),
        ]

    issues: list[Issue] = []
    for check in _SCORE_CHECKS:
        issues.extend(_score_issue(row, clip_uuid, video, check, config))
    for equal_col, field in _EQUALITY_CHECKS:
        if row.get(equal_col) is False:
            issues.append(
                make_issue(
                    code="clip_field_mismatch",
                    message=f"Clip field {field!r} differs between outputs",
                    feature=_FEATURE_STRUCTURE,
                    video=video,
                    clip=clip_uuid,
                    field=field,
                    details={"a": row.get(f"{field}_a"), "b": row.get(f"{field}_b")},
                ),
            )
    return issues


def _score_issue(
    row: Mapping[str, Any],
    clip_uuid: str,
    video: str | None,
    check: _ScoreCheck,
    config: SplitComparisonConfig,
) -> list[Issue]:
    """Evaluate one scalar score measurement (raw a/b + abs diff) into 0/1 issues."""
    value_a = row.get(f"{check.prefix}_a")
    value_b = row.get(f"{check.prefix}_b")
    if value_a is None and value_b is None:
        return []
    if (value_a is None) != (value_b is None):
        return [
            make_issue(
                code="metadata_value_one_sided",
                message=f"{check.field} present on only one output",
                feature=check.feature,
                video=video,
                clip=clip_uuid,
                field=check.field,
                output="b" if value_a is not None else "a",
            ),
        ]
    abs_diff = row.get(f"{check.prefix}_abs_diff")
    policy: ScoreTolerance = getattr(config, check.policy_attr)
    if abs_diff is None or _within_tolerance(abs_diff, cast("float", value_a), cast("float", value_b), policy):
        return []
    return [
        make_issue(
            code=check.code,
            message=f"{check.field} differs between outputs",
            feature=check.feature,
            video=video,
            clip=clip_uuid,
            field=check.field,
            details={"a": value_a, "b": value_b, "abs_diff": abs_diff},
        ),
    ]


def _eval_window(row: Mapping[str, Any], config: SplitComparisonConfig) -> list[Issue]:
    """Evaluate one window measurement row into issues."""
    clip_uuid = row["clip_uuid"]
    video = row.get("video_uuid")
    present_a = row["present_a"]
    present_b = row["present_b"]
    if present_a != present_b:
        return [
            make_issue(
                code="metadata_value_one_sided",
                message="Caption window/model present on only one output",
                feature=_FEATURE_CAPTIONS,
                video=video,
                clip=clip_uuid,
                field="caption",
                output="b" if present_a else "a",
                details={"model": row.get("model"), "kind": row.get("kind")},
            ),
        ]
    similarity = row.get("similarity")
    threshold = config.caption.min_similarity
    if similarity is None or similarity >= threshold:
        return []
    return [
        make_issue(
            code="caption_similarity_below_threshold",
            message=f"Caption similarity {similarity:.3f} below threshold {threshold:.3f}",
            feature=_FEATURE_CAPTIONS,
            video=video,
            clip=clip_uuid,
            field="caption",
            details={
                "start_ns": row.get("start_ns"),
                "end_ns": row.get("end_ns"),
                "model": row.get("model"),
                "kind": row.get("kind"),
                "similarity": similarity,
                "threshold": threshold,
            },
        ),
    ]


def _verdict_row(row: Mapping[str, Any], num_issues: int, *, compare_captions: bool) -> dict[str, Any]:
    diffs = [row.get(col) for col in _ABS_DIFF_COLUMNS]
    present = [diff for diff in diffs if diff is not None]
    return {
        "clip_uuid": row["clip_uuid"],
        "video_uuid": row.get("video_uuid"),
        "present_a": row["present_a"],
        "present_b": row["present_b"],
        "num_issues": num_issues,
        "passed": num_issues == 0,
        # The rollup is a stored clip-table column from measure; suppress it when captions
        # are out of scope so a --no-captions re-eval doesn't echo a captioned root's value
        # (a fresh --no-captions run has no rollup) -- eval output tracks eval config, not history.
        "min_caption_similarity": row.get("min_caption_similarity") if compare_captions else None,
        "max_score_abs_diff": max(present) if present else None,
    }


def _summary(
    verdicts: Sequence[Mapping[str, Any]],
    issues: pa.Table,
    *,
    config: SplitComparisonConfig,
) -> dict[str, Any]:
    passed = sum(1 for v in verdicts if v["passed"])
    codes = issues["code"].to_pylist() if issues.num_rows else []
    features = issues["feature"].to_pylist() if issues.num_rows else []
    return {
        "policy": {
            "aesthetic": config.aesthetic.model_dump(),
            "motion": config.motion.model_dump(),
            "caption_min_similarity": config.caption.min_similarity,
            "compare_captions": config.compare_captions,
        },
        "total_clips": len(verdicts),
        "clips_passed": passed,
        "clips_failed": len(verdicts) - passed,
        "total_issues": issues.num_rows,
        "issues_by_code": dict(Counter(codes)),
        "issues_by_feature": dict(Counter(f for f in features if f is not None)),
    }


def _within_tolerance(abs_diff: float, value_a: float, value_b: float, policy: ScoreTolerance) -> bool:
    """Pass if the abs diff is within the absolute or relative tolerance."""
    if abs_diff <= policy.abs_tolerance:
        return True
    larger = max(abs(value_a), abs(value_b))
    return larger > 0 and abs_diff / larger <= policy.rel_tolerance
