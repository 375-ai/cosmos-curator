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
"""Tests for split pipeline CLI argument wiring."""

import argparse
import json
from pathlib import Path

import pytest

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curator.core.utils.config.args_utils import fill_default_args
from cosmos_curator.pipelines.common.model_constraints import PreprocessMode
from cosmos_curator.pipelines.video.captioning.captioning_builders import CaptioningConfig
from cosmos_curator.pipelines.video.read_write.metadata_writer_stage import ClipWriterStage
from cosmos_curator.pipelines.video.splitting_pipeline import _assemble_stages, _setup_parser
from cosmos_curator.pipelines.video.utils.data_model import VllmConfig

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SPLIT_INVOKE_TEMPLATES = (
    _REPO_ROOT / "examples/nvcf/function/invoke_video_split_full.json",
    _REPO_ROOT / "examples/workflow/template_invoke_video_split.json",
)
_CI_SPLIT_CONFIGS = sorted((_REPO_ROOT / "examples/ci").glob("*.json"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _setup_parser(parser)
    return parser


def _stage_object(stage: CuratorStage | CuratorStageSpec) -> CuratorStage:
    if isinstance(stage, CuratorStageSpec):
        return stage.stage
    return stage


def _caption_args(extra_args: list[str]) -> argparse.Namespace:
    input_path = Path.cwd() / "tmp-input"
    output_path = Path.cwd() / "tmp-output"
    return _parser().parse_args(
        [
            "--input-video-path",
            input_path.as_posix(),
            "--output-clip-path",
            output_path.as_posix(),
            "--no-generate-embeddings",
            *extra_args,
        ]
    )


def _capture_captioning_config(monkeypatch: pytest.MonkeyPatch) -> dict[str, CaptioningConfig]:
    captured: dict[str, CaptioningConfig] = {}

    def fake_build_captioning_stages(config: CaptioningConfig) -> list[CuratorStage | CuratorStageSpec]:
        captured["config"] = config
        return []

    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.splitting_pipeline.build_captioning_stages", fake_build_captioning_stages
    )
    return captured


def test_caption_quality_flags_default_enabled() -> None:
    """Caption quality flags should default to enabled."""
    args = _parser().parse_args([])

    assert args.caption_quality_flags_enabled is True


def test_no_caption_quality_flags_disables_flags() -> None:
    """The disable flag should set caption_quality_flags_enabled to False."""
    args = _parser().parse_args(["--no-caption-quality-flags"])

    assert args.caption_quality_flags_enabled is False


def test_caption_quality_stats_default_enabled() -> None:
    """Run-level caption quality stats should default to enabled."""
    args = _parser().parse_args([])

    assert args.caption_quality_stats_enabled is True


def test_no_caption_quality_stats_disables_artifact() -> None:
    """The disable flag should set caption_quality_stats_enabled to False."""
    args = _parser().parse_args(["--no-caption-quality-stats"])

    assert args.caption_quality_stats_enabled is False


def test_no_caption_quality_stats_reaches_clip_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage assembly should pass the disable flag to ClipWriterStage."""
    monkeypatch.setattr("cosmos_curator.pipelines.video.splitting_pipeline.build_captioning_stages", lambda _: [])
    input_path = Path.cwd() / "tmp-input"
    output_path = Path.cwd() / "tmp-output"
    args = _parser().parse_args(
        [
            "--input-video-path",
            input_path.as_posix(),
            "--output-clip-path",
            output_path.as_posix(),
            "--no-generate-embeddings",
            "--no-caption-quality-stats",
        ]
    )

    stages = _assemble_stages(args)
    writers = [stage for stage in map(_stage_object, stages) if isinstance(stage, ClipWriterStage)]

    assert len(writers) == 1
    assert writers[0]._caption_quality_stats_enabled is False


def test_write_all_caption_json_default_disabled() -> None:
    """Aggregate caption JSON should be opt-in."""
    args = _parser().parse_args([])

    assert args.write_all_caption_json is False


def test_write_all_caption_json_opt_in() -> None:
    """The positive flag should enable aggregate caption JSON."""
    args = _parser().parse_args(["--write-all-caption-json"])

    assert args.write_all_caption_json is True


def test_no_write_all_caption_json_flag_removed(capsys: pytest.CaptureFixture[str]) -> None:
    """The old negative flag should no longer be accepted."""
    with pytest.raises(SystemExit):
        _parser().parse_args(["--no-write-all-caption-json"])

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unrecognized arguments: --no-write-all-caption-json" in captured.err


def test_deprecated_vllm_preprocess_args_default_to_none() -> None:
    """Legacy preprocessing flags stay parseable with inert defaults."""
    args = _parser().parse_args([])

    assert args.qwen_preprocess_dtype is None
    assert args.qwen_model_does_preprocess is None


def test_deprecated_vllm_preprocess_args_are_documented() -> None:
    """Help text should point legacy preprocessing users at the new flag."""
    help_text = _parser().format_help()

    assert "--qwen-preprocess-dtype" in help_text
    assert "--qwen-model-does-preprocess" in help_text
    assert "--vllm-preprocess-mode" in help_text
    assert "Deprecated" in help_text


def test_debug_save_vllm_frames_help_text_names_png_preview_and_stats() -> None:
    """Debug frame help should describe both preview PNGs and tensor stats."""
    help_text = _parser().format_help()

    assert "--debug-save-vllm-frames" in help_text
    assert "PNG preview" in help_text
    assert "frame_stats.json" in help_text


@pytest.mark.parametrize(
    "legacy_args",
    [
        ["--qwen-preprocess-dtype", "float16"],
        ["--qwen-model-does-preprocess"],
    ],
)
def test_deprecated_vllm_preprocess_args_raise_migration_error(legacy_args: list[str]) -> None:
    """Using a legacy preprocessing flag should fail with the replacement flag named."""
    args = _caption_args(legacy_args)

    with pytest.raises(ValueError, match="--vllm-preprocess-mode"):
        _assemble_stages(args)


def test_ci_split_config_fixtures_exist() -> None:
    """The pre-canned CI split config library should be present and discoverable."""
    assert _CI_SPLIT_CONFIGS, "no examples/ci/*.json scenario configs found"


@pytest.mark.parametrize("config_path", _CI_SPLIT_CONFIGS, ids=lambda p: p.name)
def test_ci_split_configs_only_use_known_args(config_path: Path) -> None:
    """Pre-canned CI split configs must not carry typoed/unknown arg keys.

    Config mode (``run_pipeline <config.json>``) fills missing args with their
    defaults and silently ignores unrecognized keys, so a typo would otherwise
    sail through CI while quietly exercising the default instead of the intended
    value. Guard against that by checking every key against the split parser.
    """
    config = json.loads(config_path.read_text())
    assert config.get("pipeline") == "split", config_path.as_posix()

    valid_dests = set(vars(_parser().parse_args([])))
    unknown = sorted(set(config["args"]) - valid_dests)
    assert not unknown, f"{config_path.name}: unknown split args {unknown}"


def test_split_invoke_templates_use_supported_vllm_preprocess_mode() -> None:
    """Split invoke templates should not pass legacy qwen preprocessing args."""
    deprecated_args = {"qwen_preprocess_dtype", "qwen_model_does_preprocess"}

    for template_path in _SPLIT_INVOKE_TEMPLATES:
        invoke_args = json.loads(template_path.read_text())["args"]

        assert deprecated_args.isdisjoint(invoke_args), template_path.as_posix()
        assert invoke_args["vllm_preprocess_mode"] == PreprocessMode.CURATOR.value


def test_fill_default_args_does_not_inject_qwen_model_does_preprocess() -> None:
    """JSON config mode must not backfill a falsey default for the deprecated store_true flag."""
    input_path = Path.cwd() / "tmp-input"
    output_path = Path.cwd() / "tmp-output"
    args = argparse.Namespace(
        input_video_path=input_path.as_posix(),
        output_clip_path=output_path.as_posix(),
        vllm_preprocess_mode=PreprocessMode.CURATOR.value,
    )

    fill_default_args(args, _setup_parser, omit_dests=frozenset({"qwen_model_does_preprocess"}))
    assert not hasattr(args, "qwen_model_does_preprocess")


def test_legacy_qwen_model_does_preprocess_false_in_json_config_is_ignored() -> None:
    """Pre-migration invoke JSON used ``false`` for the deprecated store_true flag."""
    args = _caption_args([])
    args.qwen_model_does_preprocess = False
    args.vllm_preprocess_mode = PreprocessMode.CURATOR.value

    _assemble_stages(args)


def test_legacy_qwen_model_does_preprocess_true_raises_migration_error() -> None:
    """Explicit legacy model-preprocess opt-in must still fail."""
    args = _caption_args([])
    args.qwen_model_does_preprocess = True

    with pytest.raises(ValueError, match="--vllm-preprocess-mode"):
        _assemble_stages(args)


@pytest.mark.parametrize(
    "caption_algo",
    ["qwen", "qwen3_6_27b", "qwen3_vl_30b", "cosmos_r1", "cosmos_r2", "nemotron"],
)
def test_vllm_video_max_pixels_reaches_sync_vllm_configs(
    monkeypatch: pytest.MonkeyPatch,
    caption_algo: str,
) -> None:
    """Accepted regular sync vLLM backends receive both resize-budget carriers."""
    captured = _capture_captioning_config(monkeypatch)
    args = _caption_args(
        [
            "--captioning-algorithm",
            caption_algo,
            "--vllm-video-max-pixels-per-frame",
            "100500",
        ]
    )

    _assemble_stages(args)

    config = captured["config"]
    assert config.window_config.video_max_pixels_per_frame == 100500
    assert isinstance(config.backend, VllmConfig)
    assert config.backend.video_max_pixels_per_frame == 100500


@pytest.mark.parametrize("caption_algo", ["vllm_async", "gemini", "openai"])
def test_vllm_video_max_pixels_rejects_non_sync_vllm(
    monkeypatch: pytest.MonkeyPatch,
    caption_algo: str,
) -> None:
    """The sync-only flag is rejected for async and API captioning paths."""
    _capture_captioning_config(monkeypatch)
    args = _caption_args(
        [
            "--captioning-algorithm",
            caption_algo,
            "--vllm-video-max-pixels-per-frame",
            "100500",
        ]
    )

    with pytest.raises(ValueError, match="regular windowed sync vLLM"):
        _assemble_stages(args)


@pytest.mark.parametrize("value", ["100351", "602113"])
def test_vllm_video_max_pixels_rejects_values_outside_bounds(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    """The flag is rejected outside the accepted upper-bound domain."""
    _capture_captioning_config(monkeypatch)
    args = _caption_args(["--captioning-algorithm", "qwen", "--vllm-video-max-pixels-per-frame", value])

    with pytest.raises(ValueError, match=r"integer in \[100352, 602112\]"):
        _assemble_stages(args)


def test_vllm_video_max_pixels_rejects_unsupported_caption_algorithm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage assembly still rejects future unsupported caption algorithms."""
    _capture_captioning_config(monkeypatch)
    args = _caption_args(["--captioning-algorithm", "qwen", "--vllm-video-max-pixels-per-frame", "100500"])
    args.captioning_algorithm = "future_backend"

    with pytest.raises(RuntimeError, match="Unsupported captioning algorithm"):
        _assemble_stages(args)


def test_nemotron_forces_model_preprocess_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Selecting nemotron must force vLLM/HF-owned preprocessing."""
    captured = _capture_captioning_config(monkeypatch)
    args = _caption_args(["--captioning-algorithm", "nemotron"])

    _assemble_stages(args)

    config = captured["config"]
    assert isinstance(config.backend, VllmConfig)
    assert config.backend.preprocess_mode == PreprocessMode.MODEL
    assert config.backend.model_preprocess_enabled is True


def test_vllm_video_max_pixels_help_text_names_scope_bounds_and_grid() -> None:
    """Help text should describe the upper-bound scope, bounds, and grid quantization."""
    help_text = _parser().format_help()

    assert "--vllm-video-max-pixels-per-frame" in help_text
    assert "regular" in help_text
    assert "windowed sync vLLM" in help_text
    assert "[100352, 602112]" in help_text
    assert "28 for CPU prep" in help_text
    assert "32 for Qwen3" in help_text
