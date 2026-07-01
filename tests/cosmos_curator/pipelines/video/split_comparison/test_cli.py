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
"""Tests for the CLI's config resolution: overrides are applied and validated."""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pytest
from pydantic import ValidationError

from cosmos_curator.pipelines.video.split_comparison import store
from cosmos_curator.pipelines.video.split_comparison.cli import _build_parser, _resolve_config, main
from cosmos_curator.pipelines.video.split_comparison.config import SplitComparisonConfig
from cosmos_curator.pipelines.video.split_comparison.measure import ray as raymod
from cosmos_curator.pipelines.video.split_comparison.measure.core import Measurements
from cosmos_curator.pipelines.video.split_comparison.measure.schema import (
    CLIP_MEASUREMENT_SCHEMA,
    WINDOW_MEASUREMENT_SCHEMA,
)

_BASE = ["--output-a", "/a", "--output-b", "/b", "--measurements-path", "measurements/ab"]


def _parse(extra: list[str]) -> argparse.Namespace:
    return _build_parser().parse_args([*_BASE, *extra])


def test_valid_caption_overrides_are_applied() -> None:
    """In-range --min-caption-similarity / --batch-size flow into the resolved config."""
    config = _resolve_config(_parse(["--min-caption-similarity", "0.9", "--batch-size", "256"]))
    assert config.caption.min_similarity == 0.9
    assert config.caption.encode_batch_size == 256


def test_no_captions_flag_disables_caption_comparison() -> None:
    """--no-captions sets compare_captions=False on the resolved config."""
    assert _resolve_config(_parse(["--no-captions"])).compare_captions is False


@pytest.mark.parametrize(
    "flag",
    [
        ["--batch-size", "0"],  # encode_batch_size ge=1
        ["--min-caption-similarity", "5.0"],  # min_similarity le=1
        ["--min-caption-similarity", "-0.1"],  # min_similarity ge=0
        ["--summary-token-rel-tol", "-1.0"],  # token_count_rel_tolerance ge=0
    ],
)
def test_out_of_range_override_raises_at_resolution(flag: list[str]) -> None:
    """Out-of-range CLI overrides fail validation when the config is built, not deep in the run."""
    with pytest.raises(ValidationError):
        _resolve_config(_parse(flag))


@pytest.mark.parametrize(
    "flag",
    [
        ["--batch-size", "0"],
        ["--min-caption-similarity", "5.0"],
    ],
)
def test_main_exits_2_on_invalid_override(flag: list[str]) -> None:
    """main() surfaces a bad override as a clean exit code 2 (caught before any measure/eval)."""
    assert main([*_BASE, *flag]) == 2


def test_main_exits_2_on_measure_valueerror(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A measure-phase ValueError (e.g. no GPUs / bad --num-gpus from resolve_num_gpus) exits 2, not 1.

    An uncaught exception would exit 1 -- colliding with the "issues found" contract -- and dump a
    traceback. main() must catch it, print a concise message, and return 2 (bad setup).
    """

    def no_gpu(*_a: object, **_k: object) -> object:
        msg = "no GPUs detected by Ray; the measure requires at least one GPU"
        raise ValueError(msg)

    monkeypatch.setattr(raymod, "run", no_gpu)
    assert main(_BASE) == 2
    err = capsys.readouterr().err
    assert "no GPUs detected" in err
    assert "Traceback" not in err


def _write_measure_root(tmp_path: Path, *, with_summaries: bool) -> str:
    """Persist a minimal measurements root (empty clip/window tables) and return its path.

    ``with_summaries=False`` mirrors a ``--no-summary`` measure: measurement tables +
    manifest are written, but no summary_a/summary_b snapshots.
    """
    root = str(tmp_path / "m")
    config = SplitComparisonConfig(output_a="/a", output_b="/b")
    measurements = Measurements(
        clip_table=pa.Table.from_pylist([], schema=CLIP_MEASUREMENT_SCHEMA),
        window_table=pa.Table.from_pylist([], schema=WINDOW_MEASUREMENT_SCHEMA),
        stats={},
    )
    store.write_measurements(
        measurements,
        root,
        config=config,
        device="cuda",
        fp16=False,
        lance_version="v0",
        summaries_snapshotted=with_summaries,
    )
    return root


def test_reeval_of_no_summary_root_skips_summary_comparison(tmp_path: Path) -> None:
    """A root measured without summaries must not emit phantom summary_load_failed issues on re-eval.

    With empty measurement tables the only possible issues are summary-load failures, so a clean
    re-eval is exit 0; comparing absent snapshots would fire two summary_load_failed -> exit 1.
    """
    root = _write_measure_root(tmp_path, with_summaries=False)
    rc = main(["--skip-measure", "--measurements-path", root, "--output-a", "/a", "--output-b", "/b"])
    assert rc == 0


def test_config_file_alone_resolves_to_its_spec(tmp_path: Path) -> None:
    """A --config file with no shaping flags resolves to exactly the spec it describes."""
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"output_a": "/a", "output_b": "/b", "profile_name": "p"}))
    args = _build_parser().parse_args(["--measurements-path", "measurements/ab", "--config", str(cfg)])
    config = _resolve_config(args)
    assert (config.output_a, config.output_b, config.profile_name) == ("/a", "/b", "p")


@pytest.mark.parametrize(
    "flag",
    [
        ["--output-a", "/x"],
        ["--output-b", "/x"],
        ["--profile", "p"],
        ["--no-captions"],
        ["--min-caption-similarity", "0.9"],
        ["--batch-size", "256"],
        ["--summary-token-rel-tol", "0.1"],
    ],
)
def test_config_is_mutually_exclusive_with_shaping_flags(tmp_path: Path, flag: list[str]) -> None:
    """--config is the full spec; combining it with a config-shaping flag exits 2, not a silent drop."""
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"output_a": "/a", "output_b": "/b"}))
    rc = main(["--measurements-path", str(tmp_path / "m"), "--config", str(cfg), *flag])
    assert rc == 2


def test_reeval_no_captions_skips_caption_eval(tmp_path: Path) -> None:
    """--no-captions on re-eval skips caption-window eval even on a captioned root.

    The window table holds one sub-threshold caption row, so with captions on it fires a
    caption_similarity_below_threshold issue (exit 1); --no-captions must skip it (exit 0).
    """
    root = str(tmp_path / "m")
    config = SplitComparisonConfig(output_a="/a", output_b="/b")
    window = pa.Table.from_pylist(
        [
            {
                "clip_uuid": "c1",
                "video_uuid": "v",
                "start_ns": 0,
                "end_ns": 1,
                "model": "qwen",
                "kind": "caption",
                "present_a": True,
                "present_b": True,
                "identical": False,
                "similarity": 0.5,
                "len_a": 1,
                "len_b": 1,
            }
        ],
        schema=WINDOW_MEASUREMENT_SCHEMA,
    )
    store.write_measurements(
        Measurements(
            clip_table=pa.Table.from_pylist([], schema=CLIP_MEASUREMENT_SCHEMA),
            window_table=window,
            stats={},
        ),
        root,
        config=config,
        device="cuda",
        fp16=False,
        lance_version="v0",
        summaries_snapshotted=False,
    )
    base = ["--skip-measure", "--measurements-path", root, "--output-a", "/a", "--output-b", "/b"]
    assert main(base) == 1  # captions on: 0.5 < 0.85 fires caption_similarity_below_threshold
    assert main([*base, "--no-captions"]) == 0  # metadata-only: caption eval skipped


def test_reeval_recovers_no_captions_scope_from_manifest(tmp_path: Path) -> None:
    """A root measured --no-captions stays metadata-only on re-eval without repeating the flag.

    The manifest records compare_captions=False, so re-eval recovers it (like outputs) rather
    than defaulting captions back on: the sub-threshold window row must not fire an issue (exit 0).
    """
    root = str(tmp_path / "m")
    config = SplitComparisonConfig(output_a="/a", output_b="/b", compare_captions=False)
    window = pa.Table.from_pylist(
        [
            {
                "clip_uuid": "c1",
                "video_uuid": "v",
                "start_ns": 0,
                "end_ns": 1,
                "model": "qwen",
                "kind": "caption",
                "present_a": True,
                "present_b": True,
                "identical": False,
                "similarity": 0.5,
                "len_a": 1,
                "len_b": 1,
            }
        ],
        schema=WINDOW_MEASUREMENT_SCHEMA,
    )
    store.write_measurements(
        Measurements(
            clip_table=pa.Table.from_pylist([], schema=CLIP_MEASUREMENT_SCHEMA),
            window_table=window,
            stats={},
        ),
        root,
        config=config,
        device="cuda",
        fp16=False,
        lance_version="v0",
        summaries_snapshotted=False,
    )
    # No --no-captions on re-eval: recovered from the manifest, so caption eval stays skipped.
    assert main(["--skip-measure", "--measurements-path", root, "--output-a", "/a", "--output-b", "/b"]) == 0
