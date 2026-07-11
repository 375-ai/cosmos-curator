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

"""Tests for CuratorStage runtime-environment construction."""

import pytest

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, PipelineTask


class _StageWithCondaEnv(CuratorStage):
    """Minimal stage that pins a conda env so env_info builds a Pixi runtime env."""

    @property
    def conda_env_name(self) -> str | None:
        return "unified"

    def process_data(self, task: list[PipelineTask]) -> list[PipelineTask] | None:
        return task


def _worker_env_vars(monkeypatch: pytest.MonkeyPatch, env: dict[str, str | None]) -> dict[str, str]:
    for name, value in env.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    runtime_env = _StageWithCondaEnv().env_info.to_ray_runtime_env()
    return dict(runtime_env.get("env_vars", {}))


def test_env_info_text_mode_does_not_forward_logging_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """In the default text mode, worker actors get no forwarded logging env vars."""
    env_vars = _worker_env_vars(
        monkeypatch,
        {"PYTHON_LOG_FORMAT": None, "PYTHON_LOG": "debug", "CURATOR_RUN_ID": "abc"},
    )
    assert "PYTHON_LOG_FORMAT" not in env_vars
    assert "PYTHON_LOG" not in env_vars
    assert "CURATOR_RUN_ID" not in env_vars


def test_env_info_json_mode_forwards_logging_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """JSON mode forwards the logging toggle/identity vars into the actor env."""
    env_vars = _worker_env_vars(
        monkeypatch,
        {"PYTHON_LOG_FORMAT": "json", "PYTHON_LOG": "debug", "CURATOR_RUN_ID": "abc"},
    )
    assert env_vars["PYTHON_LOG_FORMAT"] == "json"
    assert env_vars["PYTHON_LOG"] == "debug"
    assert env_vars["CURATOR_RUN_ID"] == "abc"


def test_env_info_json_mode_omits_unset_optional_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only env vars that are actually set are forwarded (no empty placeholders)."""
    env_vars = _worker_env_vars(
        monkeypatch,
        {"PYTHON_LOG_FORMAT": "json", "PYTHON_LOG": None, "CURATOR_RUN_ID": None},
    )
    assert env_vars["PYTHON_LOG_FORMAT"] == "json"
    assert "PYTHON_LOG" not in env_vars
    assert "CURATOR_RUN_ID" not in env_vars
