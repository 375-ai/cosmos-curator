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
"""Persistence for caption judge reports."""

import json
from collections import Counter
from pathlib import Path
from typing import assert_never

import smart_open  # type: ignore[import-untyped]

from cosmos_curator.core.utils.storage import storage_utils
from cosmos_curator.pipelines.ray_data.caption_judge.config import ReportFormat
from cosmos_curator.pipelines.ray_data.caption_judge.result_model import Report


def _is_local_path(path: str) -> bool:
    return "://" not in path


def _ensure_local_parent(path: str) -> None:
    if _is_local_path(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_report(report: Report, path: str, *, report_format: ReportFormat) -> str:
    """Persist a caption judge report and return the written path."""
    if report_format == "json":
        _write_json(report, path)
    else:
        assert_never(report_format)
    return path


def _write_json(report: Report, path: str) -> None:
    payload: dict[str, object] = {
        "passed": report.passed,
        "baseline": report.baseline,
        "candidate": report.candidate,
        "baseline_metadata": report.baseline_metadata,
        "candidate_metadata": report.candidate_metadata,
        "summary": build_summary(report),
        "issues": _issue_rows(report),
    }
    if report.caption_model_baseline is not None:
        payload["caption_model_baseline"] = report.caption_model_baseline
    if report.caption_model_candidate is not None:
        payload["caption_model_candidate"] = report.caption_model_candidate
    if report.config is not None:
        payload["config"] = report.config.model_dump(mode="json")
    body = json.dumps(payload, indent=2, sort_keys=True)
    _ensure_local_parent(path)
    transport_params = storage_utils.get_smart_open_params(path)
    with smart_open.open(path, "w", encoding="utf-8", **transport_params) as handle:
        handle.write(body)


def _issue_rows(report: Report) -> list[dict[str, object]]:
    return [{key: value for key, value in row.items() if value is not None} for row in report.issues.to_pylist()]


def build_summary(report: Report) -> dict[str, object]:
    """Build the report summary block from run stats and issue rows."""
    rows = report.issues.to_pylist() if report.issues.num_rows else []
    codes = [str(row["code"]) for row in rows if row.get("code") is not None]
    outputs = [str(row["output"]) for row in rows if row.get("output") is not None]
    return {
        "clips_in_baseline": report.stats.clips_in_baseline,
        "clips_in_candidate": report.stats.clips_in_candidate,
        "clips_in_both": report.stats.clips_in_both,
        "windows_in_baseline": report.stats.windows_in_baseline,
        "windows_in_candidate": report.stats.windows_in_candidate,
        "windows_in_both": report.stats.windows_in_both,
        "windows_judged": report.stats.windows_judged,
        "total_issues": report.issues.num_rows,
        "issues_by_code": dict(Counter(codes)),
        "issues_by_output": dict(Counter(outputs)),
        "runtime_sec": report.runtime_sec,
    }
