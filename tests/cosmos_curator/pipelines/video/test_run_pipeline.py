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
"""Tests for the video run_pipeline config-mode entry point."""

import pathlib
import sys
from typing import TYPE_CHECKING

import pytest

from cosmos_curator.pipelines.ray_data.caption_judge import pipeline as caption_judge_pipeline
from cosmos_curator.pipelines.video import run_pipeline as video_run_pipeline

if TYPE_CHECKING:
    import argparse


def test_config_mode_dispatches_caption_judge_pipeline(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config mode dispatches the new caption_judge pipeline kind."""
    config_path = tmp_path / "caption_judge.yaml"
    config_path.write_text(
        "\n".join(
            [
                "pipeline: caption_judge",
                "args:",
                "  schema_version: 1",
                "  kind: caption_judge",
                "  input:",
                f"    baseline: {tmp_path / 'baseline'}",
                f"    candidate: {tmp_path / 'candidate'}",
                "    caption_model_baseline: qwen",
                "    caption_model_candidate: cosmos3_nano",
                "  output:",
                f"    report_path: {tmp_path / 'report.json'}",
            ]
        ),
        encoding="utf-8",
    )
    captured: dict[str, argparse.Namespace] = {}

    monkeypatch.setattr(sys, "argv", ["run_pipeline", str(config_path)])
    monkeypatch.setattr(
        caption_judge_pipeline,
        "nvcf_run_caption_judge",
        lambda args: captured.setdefault("args", args),
    )

    video_run_pipeline.cli()

    args = captured["args"]
    assert args.input == {
        "baseline": str(tmp_path / "baseline"),
        "candidate": str(tmp_path / "candidate"),
        "caption_model_baseline": "qwen",
        "caption_model_candidate": "cosmos3_nano",
    }
    assert args.output == {"report_path": str(tmp_path / "report.json")}
