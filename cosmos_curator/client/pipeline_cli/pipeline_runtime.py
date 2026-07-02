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

"""Run-only entrypoint for config-backed pipelines inside runtime environments."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, NoReturn, Protocol

import typer
from pydantic import ValidationError
from typer import Argument, Option

from cosmos_curator.client.pipeline_cli.pipeline_config import PipelineName, load_pipeline_kind


def main(
    config: Annotated[Path, Argument(help="Path to a JSON/YAML pipeline config.")],
    set_overrides: Annotated[
        list[str] | None,
        Option("--set", help="Small resolved-config override in dotted PATH=VALUE form."),
    ] = None,
    *,
    json_output: Annotated[bool, Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """Run a pipeline from a JSON/YAML config."""
    try:
        kind = load_pipeline_kind(config)
        run_pipeline = _RUNTIME_BUILDERS[kind](config, set_overrides=set_overrides or [])
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        _fail("invalid", exc, json_output=json_output)

    try:
        output = run_pipeline()
    except Exception as exc:
        if json_output:
            _fail("runtime", exc, json_output=True)
        raise

    if json_output:
        typer.echo(json.dumps(output.json_payload, indent=2))
    else:
        typer.echo(output.message)


@dataclass(frozen=True)
class _RuntimeOutput:
    json_payload: dict[str, object]
    message: str


class _RuntimeBuilder(Protocol):
    def __call__(self, config: Path, *, set_overrides: list[str]) -> Callable[[], _RuntimeOutput]: ...


def _run_video_split(
    config: Path,
    *,
    set_overrides: list[str],
) -> Callable[[], _RuntimeOutput]:
    from cosmos_curator.pipelines.ray_data.video_split.config import resolve_video_split_config  # noqa: PLC0415

    resolution = resolve_video_split_config(config, overrides=set_overrides)

    def run() -> _RuntimeOutput:
        from cosmos_curator.pipelines.ray_data.video_split.pipeline import run_config  # noqa: PLC0415

        clips_written = run_config(resolution.config)
        return _RuntimeOutput(
            json_payload={"clips_written": clips_written},
            message=f"Wrote {clips_written} clip(s)",
        )

    return run


def _run_caption_judge(
    config: Path,
    *,
    set_overrides: list[str],
) -> Callable[[], _RuntimeOutput]:
    from cosmos_curator.pipelines.ray_data.caption_judge.config import resolve_caption_judge_config  # noqa: PLC0415

    resolved_config = resolve_caption_judge_config(config, overrides=set_overrides)

    def run() -> _RuntimeOutput:
        from cosmos_curator.pipelines.ray_data.caption_judge.driver import run_caption_judge_pipeline  # noqa: PLC0415
        from cosmos_curator.pipelines.ray_data.caption_judge.report_io import write_report  # noqa: PLC0415

        report = run_caption_judge_pipeline(config=resolved_config)
        report_path = write_report(
            report,
            resolved_config.output.report_path,
            report_format=resolved_config.output.report_format,
        )
        status = "PASSED" if report.passed else "FAILED"
        return _RuntimeOutput(
            json_payload={
                "passed": report.passed,
                "issues": report.issues.num_rows,
                "windows_judged": report.stats.windows_judged,
                "report_path": report_path,
            },
            message=(
                f"{status} caption judge: {report.issues.num_rows} issues, "
                f"{report.stats.windows_judged} judged windows, report: {report_path}"
            ),
        )

    return run


_RUNTIME_BUILDERS: dict[PipelineName, _RuntimeBuilder] = {
    "video_split": _run_video_split,
    "caption_judge": _run_caption_judge,
}


def _fail(code: str, exc: Exception, *, json_output: bool) -> NoReturn:
    if json_output:
        typer.echo(json.dumps({"ok": False, "error": code, "message": str(exc)}, indent=2), err=True)
    else:
        typer.echo(str(exc), err=True)
    raise typer.Exit(2)


if __name__ == "__main__":
    typer.run(main)
