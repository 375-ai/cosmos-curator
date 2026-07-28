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

"""Unit tests for the single-video data-integrity CLI (main + exit codes)."""

import pytest

from cosmos_curator.core.sensors.data_integrity import cli


def _stub_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the check phase succeed so a test can isolate the reporting phase."""
    monkeypatch.setattr(cli, "validate_source", lambda _source: None)
    monkeypatch.setattr(cli, "run_checks", lambda *_a, **_k: ([], None, None))


def _raise_render(*_args: object, **_kwargs: object) -> str:
    msg = "unserializable measurement"
    raise ValueError(msg)


@pytest.mark.parametrize(
    ("renderer", "argv"),
    [
        ("_render_human", ["--source", "clip.mp4"]),
        ("_render_json", ["--source", "clip.mp4", "--json"]),
    ],
)
def test_render_failure_exits_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    renderer: str,
    argv: list[str],
) -> None:
    """A failure while rendering the report yields exit code 2, not a traceback.

    Reporting runs after the checks have already succeeded, so it has to sit inside the
    same handler; otherwise a non-finite measurement or a closed stdout escapes as a
    traceback and breaks the documented exit-code contract.
    """
    _stub_checks(monkeypatch)
    monkeypatch.setattr(cli, renderer, _raise_render)

    assert cli.main(argv) == cli.ERROR_EXIT_CODE
    assert "error:" in capsys.readouterr().err
