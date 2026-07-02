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

"""Tests for the config-driven ``cosmos-curator pipeline`` CLI."""

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from cosmos_curator.client.cli import cosmos_curator

runner = CliRunner()


def _write_config(path: Path, *, extra: str = "") -> Path:
    path.write_text(
        f"""schema_version: 1
kind: video_split
input:
  video_path: /videos
output:
  clip_path: /clips
{extra}
""",
        encoding="utf-8",
    )
    return path


def _write_caption_judge_config(path: Path, *, extra: str = "") -> Path:
    path.write_text(
        f"""schema_version: 1
kind: caption_judge
input:
  baseline: /baseline
  candidate: /candidate
output:
  report_path: /reports/caption_judge_report.json
{extra}
""",
        encoding="utf-8",
    )
    return path


def test_pipeline_validate_reports_valid_config(tmp_path: Path) -> None:
    """Validate resolves defaults and reports success."""
    config_path = _write_config(tmp_path / "split.yaml")

    result = runner.invoke(cosmos_curator, ["pipeline", "validate", str(config_path), "--json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {"ok": True, "selected_presets": []}


def test_pipeline_validate_reports_invalid_config_as_json(tmp_path: Path) -> None:
    """Validate --json returns structured errors for tools."""
    config_path = _write_config(tmp_path / "split.yaml", extra="caption:\n  bad_field: true")

    result = runner.invoke(cosmos_curator, ["pipeline", "validate", str(config_path), "--json"])

    assert result.exit_code == 2
    payload = json.loads(result.stderr)
    assert payload["ok"] is False
    assert payload["error"] == "invalid"
    assert "bad_field" in payload["message"]


def test_pipeline_validate_reports_valid_caption_judge_config(tmp_path: Path) -> None:
    """Validate dispatches by pipeline kind."""
    config_path = _write_caption_judge_config(tmp_path / "caption_judge.yaml")

    result = runner.invoke(cosmos_curator, ["pipeline", "validate", str(config_path), "--json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {"ok": True}


def test_pipeline_validate_rejects_caption_judge_caption_model_fields(tmp_path: Path) -> None:
    """Caption judge model names are inferred from metadata, not user-authored config."""
    config_path = tmp_path / "caption_judge.yaml"
    config_path.write_text(
        """schema_version: 1
kind: caption_judge
input:
  baseline: /baseline
  candidate: /candidate
  caption_model_baseline: qwen
output:
  report_path: /reports/caption_judge_report.json
""",
        encoding="utf-8",
    )

    result = runner.invoke(cosmos_curator, ["pipeline", "validate", str(config_path), "--json"])

    assert result.exit_code == 2
    payload = json.loads(result.stderr)
    assert payload["error"] == "invalid"
    assert "caption_model_baseline" in payload["message"]


def test_pipeline_render_outputs_resolved_config_with_overrides(tmp_path: Path) -> None:
    """Render prints canonical JSON after presets and --set overrides."""
    config_path = _write_config(tmp_path / "split.yaml", extra="caption:\n  preset: balanced")

    result = runner.invoke(
        cosmos_curator,
        ["pipeline", "render", str(config_path), "--set", "caption.enabled=false"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["caption"] == {
        "enabled": False,
        "model": "qwen",
        "batch_size": 32,
        "preprocess_mode": "model",
    }
    assert payload["input"]["limit"] is None
    assert payload["split"]["limit_clips"] is None
    assert "preset" not in payload["caption"]


def test_pipeline_render_outputs_caption_judge_config_with_overrides(tmp_path: Path) -> None:
    """Render prints the resolved caption judge config after --set overrides."""
    config_path = _write_caption_judge_config(tmp_path / "caption_judge.yaml")

    result = runner.invoke(
        cosmos_curator,
        ["pipeline", "render", str(config_path), "--set", "judge.max_output_tokens=1024"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["kind"] == "caption_judge"
    assert payload["input"]["baseline"] == "/baseline"
    assert payload["input"]["candidate"] == "/candidate"
    assert payload["output"]["report_path"] == "/reports/caption_judge_report.json"
    assert "caption_model_baseline" not in payload["input"]
    assert "caption_model_candidate" not in payload["input"]
    assert payload["judge"]["model_name"] == "gcp/google/gemini-3.1-pro-preview"
    assert payload["judge"]["max_output_tokens"] == 1024
    assert "profile_name" not in payload["execution"]
    assert payload["execution"]["max_workers_per_node"] == 16


def test_pipeline_schema_outputs_user_config_schema() -> None:
    """Schema prints JSON Schema for video_split."""
    result = runner.invoke(cosmos_curator, ["pipeline", "schema", "video_split", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["title"] == "UserVideoSplitConfig"
    assert "RawCaptionConfig" in payload["$defs"]


def test_pipeline_schema_outputs_caption_judge_schema() -> None:
    """Schema prints JSON Schema for caption_judge."""
    result = runner.invoke(cosmos_curator, ["pipeline", "schema", "caption_judge", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["title"] == "UserCaptionJudgeConfig"
    assert "CaptionJudgeProviderConfig" in payload["$defs"]
    assert "CaptionJudgeInputConfig" in payload["$defs"]
    assert "CaptionJudgeOutputConfig" in payload["$defs"]
    assert "CaptionJudgeExecutionConfig" in payload["$defs"]


def test_pipeline_template_help_lists_caption_judge() -> None:
    """Template help advertises every supported pipeline kind."""
    result = runner.invoke(cosmos_curator, ["pipeline", "template", "--help"])

    assert result.exit_code == 0
    assert "caption_judge" in result.stdout
    assert "--profile" not in result.stdout


def test_pipeline_template_outputs_base_yaml_by_default() -> None:
    """Template prints the smallest editable YAML config by default."""
    result = runner.invoke(cosmos_curator, ["pipeline", "template", "video_split"])

    assert result.exit_code == 0
    payload = yaml.safe_load(result.stdout)
    assert payload == {
        "schema_version": 1,
        "kind": "video_split",
        "input": {"video_path": "/config/input"},
        "output": {"clip_path": "/config/output"},
    }


def test_pipeline_template_outputs_caption_judge_base_yaml() -> None:
    """Caption judge templates are available from the pipeline CLI."""
    result = runner.invoke(cosmos_curator, ["pipeline", "template", "caption_judge"])

    assert result.exit_code == 0
    payload = yaml.safe_load(result.stdout)
    assert payload == {
        "schema_version": 1,
        "kind": "caption_judge",
        "input": {
            "baseline": "/config/output/baseline",
            "candidate": "/config/output/candidate",
        },
        "output": {"report_path": "/config/output/caption_judge_report.json"},
    }


def test_pipeline_template_outputs_base_json_with_required_fields() -> None:
    """Template --json gives agents the template plus required author inputs."""
    result = runner.invoke(cosmos_curator, ["pipeline", "template", "video_split", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["kind"] == "video_split"
    assert "profile" not in payload
    assert [field["path"] for field in payload["required_fields"]] == [
        "schema_version",
        "kind",
        "input.video_path",
        "output.clip_path",
    ]
    assert payload["config"] == {
        "schema_version": 1,
        "kind": "video_split",
        "input": {"video_path": "/config/input"},
        "output": {"clip_path": "/config/output"},
    }


def test_pipeline_template_outputs_caption_judge_json_with_required_fields() -> None:
    """Template --json gives agents caption judge author inputs."""
    result = runner.invoke(cosmos_curator, ["pipeline", "template", "caption_judge", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["kind"] == "caption_judge"
    assert "profile" not in payload
    assert [field["path"] for field in payload["required_fields"]] == [
        "schema_version",
        "kind",
        "input.baseline",
        "input.candidate",
        "output.report_path",
    ]
    assert payload["config"] == {
        "schema_version": 1,
        "kind": "caption_judge",
        "input": {
            "baseline": "/config/output/baseline",
            "candidate": "/config/output/candidate",
        },
        "output": {"report_path": "/config/output/caption_judge_report.json"},
    }


def test_pipeline_template_rejects_profile_option() -> None:
    """Template profiles are not part of the public pipeline template CLI."""
    result = runner.invoke(cosmos_curator, ["pipeline", "template", "video_split", "--profile", "smoke"])

    assert result.exit_code != 0
    assert "--profile" in result.stderr


def test_pipeline_presets_list_and_show() -> None:
    """Preset inspection commands are JSON-friendly."""
    list_result = runner.invoke(cosmos_curator, ["pipeline", "presets", "list", "--json"])
    show_result = runner.invoke(cosmos_curator, ["pipeline", "presets", "show", "caption.balanced", "--json"])

    assert list_result.exit_code == 0
    assert show_result.exit_code == 0
    assert "caption.balanced" in {preset["qualified_name"] for preset in json.loads(list_result.stdout)["presets"]}
    assert json.loads(show_result.stdout)["fragment"]["batch_size"] == 32


def test_pipeline_run_is_not_host_cli_command(tmp_path: Path) -> None:
    """Pipeline execution is exposed through the runtime Pixi task, not the host config CLI."""
    config_path = _write_config(tmp_path / "split.yaml")

    result = runner.invoke(cosmos_curator, ["pipeline", "run", str(config_path)])

    assert result.exit_code != 0
