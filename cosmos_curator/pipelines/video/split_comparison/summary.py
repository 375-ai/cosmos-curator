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
"""Summary-level (summary.json) comparison: wire summary_loader + summary_compare.

The split pipeline's ``summary.json`` carries run/per-video aggregates the clip
lance does not (input counts, source bytes, remux/transcode/webp counts, the
filtered-by breakdowns). This module runs the package's ``summary_loader`` +
``summary_compare`` over the summary snapshots written into the measurements root
by the measure phase (see ``store.snapshot_summaries``), so re-eval stays
self-contained.
"""

from typing import Any, cast

from loguru import logger

from cosmos_curator.pipelines.video.split_comparison import store
from cosmos_curator.pipelines.video.split_comparison.config import SummaryPolicy
from cosmos_curator.pipelines.video.split_comparison.result_model import Issue, make_issue
from cosmos_curator.pipelines.video.split_comparison.summary_compare import compare_summaries
from cosmos_curator.pipelines.video.split_comparison.summary_loader import load_summary


def summary_issues(root: str, *, profile_name: str, policy: SummaryPolicy) -> list[Issue]:
    """Compare the two snapshotted summaries under ``root``; return issue rows.

    A snapshot that is missing/unreadable yields a ``summary_load_failed`` issue
    for that side and skips the field comparison (which needs both summaries).
    """
    loaded: dict[str, Any] = {}
    issues: list[Issue] = []
    for side in ("a", "b"):
        snapshot_dir = store.summary_snapshot_dir(root, side)
        try:
            loaded[side] = load_summary(snapshot_dir, profile_name=profile_name)
        except Exception as exc:  # noqa: BLE001 -- any load failure becomes a structured issue
            logger.warning("Failed to load snapshot summary {}: {}", side, exc)
            issues.append(
                make_issue(
                    code="summary_load_failed",
                    message=f"Failed to load snapshot summary for output {side.upper()}: {exc}",
                    feature="summary",
                    output=side,
                    details={"error_type": exc.__class__.__name__, "error": str(exc)},
                ),
            )
    if "a" in loaded and "b" in loaded:
        table = compare_summaries(loaded["a"], loaded["b"], policy)
        # compare_summaries returns an ISSUE_SCHEMA table; its rows are Issue rows as-is.
        issues.extend(cast("list[Issue]", table.to_pylist()))
    return issues
