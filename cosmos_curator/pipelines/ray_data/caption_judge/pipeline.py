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
"""NVCF/config entry point for the caption judge pipeline."""

import argparse
import sys

from pydantic import ValidationError

from cosmos_curator.pipelines.ray_data.caption_judge.config import CaptionJudgePipelineConfig
from cosmos_curator.pipelines.ray_data.caption_judge.driver import run_caption_judge_pipeline
from cosmos_curator.pipelines.ray_data.caption_judge.report_io import write_report


def nvcf_run_caption_judge(args: argparse.Namespace) -> None:
    """Run caption judging from config-mode args."""
    try:
        config = CaptionJudgePipelineConfig.model_validate(vars(args))
    except ValidationError as exc:
        msg = f"Invalid caption_judge config: {exc}"
        raise SystemExit(msg) from exc
    report = run_caption_judge_pipeline(config=config)
    path = write_report(report, config.output.report_path, report_format=config.output.report_format)
    sys.stdout.write(
        f"{'PASSED' if report.passed else 'FAILED'} caption judge: "
        f"{report.issues.num_rows} issues, {report.stats.windows_judged} judged windows, report: {path}\n"
    )
