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
"""Tests for caption judge report persistence."""

import json
from pathlib import Path

import pyarrow as pa

from cosmos_curator.pipelines.ray_data.caption_judge.config import CaptionJudgePipelineConfig
from cosmos_curator.pipelines.ray_data.caption_judge.report_io import write_report
from cosmos_curator.pipelines.ray_data.caption_judge.result_model import (
    ISSUE_SCHEMA,
    CaptionJudgeStats,
    Report,
    make_issue,
)


def test_write_report_json_includes_summary_issues_and_config(tmp_path: Path) -> None:
    """JSON reports include summary counts, issue rows, and the effective config."""
    config = CaptionJudgePipelineConfig(
        schema_version=1,
        kind="caption_judge",
        input={
            "baseline": "output-a",
            "candidate": "output-b",
        },
        output={"report_path": "report.json"},
    )
    issues = pa.Table.from_pylist(
        [
            make_issue(
                "caption_judge_prefers_baseline",
                "Judge preferred baseline over candidate",
                video_uuid="video-1",
                clip_uuid="clip-1",
                output="candidate",
                caption_model_baseline="qwen",
                caption_model_candidate="cosmos3_nano",
                winner="baseline",
                confidence=0.8,
                reason="more accurate",
                baseline_errors=[],
                candidate_errors=["miss"],
            )
        ],
        schema=ISSUE_SCHEMA,
    )
    report = Report(
        issues=issues,
        passed=False,
        stats=CaptionJudgeStats(clips_in_baseline=1, clips_in_candidate=1, clips_in_both=1, windows_judged=1),
        baseline=config.baseline,
        candidate=config.candidate,
        baseline_metadata=config.baseline_metadata,
        candidate_metadata=config.candidate_metadata,
        caption_model_baseline="qwen",
        caption_model_candidate="cosmos3_nano",
        config=config,
    )
    target = str(tmp_path / "report.json")

    result = write_report(report, target, report_format="json")

    assert result == target
    payload = json.loads(Path(target).read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["baseline"] == "output-a"
    assert payload["candidate"] == "output-b"
    assert payload["baseline_metadata"] == "output-a/lance/v0"
    assert payload["candidate_metadata"] == "output-b/lance/v0"
    assert payload["caption_model_baseline"] == "qwen"
    assert payload["caption_model_candidate"] == "cosmos3_nano"
    assert payload["summary"]["windows_judged"] == 1
    assert payload["summary"]["issues_by_code"] == {"caption_judge_prefers_baseline": 1}
    assert payload["issues"][0]["candidate_errors"] == ["miss"]
    assert payload["issues"][0]["winner"] == "baseline"
    assert payload["config"]["input"]["baseline"] == "output-a"
    assert "caption_model_baseline" not in payload["config"]["input"]
    assert "caption_model_candidate" not in payload["config"]["input"]
    assert payload["config"]["output"]["report_path"] == "report.json"
    assert payload["config"]["judge"]["max_output_tokens"] == 8192
    assert "profile_name" not in payload["config"]["execution"]
    assert payload["config"]["execution"]["max_workers_per_node"] == 16
