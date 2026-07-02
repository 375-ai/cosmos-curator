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
"""Tests for caption judge metadata joins."""

import pyarrow as pa
import pytest
from pydantic import ValidationError

from cosmos_curator.pipelines.ray_data.caption_judge import driver as driver_module
from cosmos_curator.pipelines.ray_data.caption_judge.config import CaptionJudgePipelineConfig
from cosmos_curator.pipelines.ray_data.caption_judge.driver import _stage_sizing, prepare_caption_judge_inputs
from cosmos_curator.pipelines.ray_data.caption_judge.result_model import JUDGE_JOB_SCHEMA

_WINDOW_STRUCT = pa.struct(
    [
        pa.field("start_ns", pa.int64()),
        pa.field("end_ns", pa.int64()),
        pa.field("caption_status", pa.string()),
        pa.field("caption_failure_reason", pa.string()),
        pa.field("captions", pa.map_(pa.string(), pa.large_string())),
    ]
)
_TEST_METADATA_SCHEMA = pa.schema(
    [
        ("clip_uuid", pa.string()),
        ("video_uuid", pa.string()),
        ("clip_location", pa.large_string()),
        ("windows", pa.list_(_WINDOW_STRUCT)),
    ]
)


def _config(**overrides: object) -> CaptionJudgePipelineConfig:
    values = {
        "schema_version": 1,
        "kind": "caption_judge",
        "input": {
            "baseline": "output-a",
            "candidate": "output-b",
        },
        "output": {"report_path": "report.json"},
    }
    input_fields = {
        "baseline",
        "candidate",
        "clip_limit",
    }
    output_fields = {"report_path", "report_format"}
    execution_fields = {"progress", "max_workers_per_node"}
    for key, value in overrides.items():
        if key in input_fields:
            values["input"][key] = value
        elif key in output_fields:
            values["output"][key] = value
        elif key in execution_fields:
            execution = values.setdefault("execution", {})
            assert isinstance(execution, dict)
            execution[key] = value
        else:
            values[key] = value
    return CaptionJudgePipelineConfig.model_validate(values)


def _metadata(rows: list[dict[str, object]]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=_TEST_METADATA_SCHEMA)


def _job_table() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "video_uuid": "video-1",
                "clip_uuid": "clip-1",
                "start_ns": 0,
                "end_ns": 10,
                "clip_location": "/clips/b.mp4",
                "caption_model_baseline": "qwen",
                "caption_model_candidate": "cosmos3_nano",
                "baseline_caption": "baseline",
                "candidate_caption": "candidate",
            }
        ],
        schema=JUDGE_JOB_SCHEMA,
    )


def test_config_derives_metadata_paths_from_output_roots() -> None:
    """Runtime config exposes output roots and derives the Lance metadata URIs."""
    config = _config(baseline="/config/output/qwen/", candidate="s3://bucket/cosmos3_nano")

    assert config.input.baseline == "/config/output/qwen/"
    assert config.input.candidate == "s3://bucket/cosmos3_nano"
    assert config.baseline_metadata == "/config/output/qwen/lance/v0"
    assert config.candidate_metadata == "s3://bucket/cosmos3_nano/lance/v0"


def test_config_requires_report_path() -> None:
    """The report path is an explicit user contract, not an implicit default."""
    with pytest.raises(ValidationError, match="report_path"):
        CaptionJudgePipelineConfig.model_validate(
            {
                "schema_version": 1,
                "kind": "caption_judge",
                "input": {
                    "baseline": "output-a",
                    "candidate": "output-b",
                },
                "output": {},
            }
        )


@pytest.mark.parametrize("field_name", ["clip_uuid", "video_uuid"])
def test_config_rejects_uuid_filters(field_name: str) -> None:
    """Caption judge compares the provided metadata outputs; UUID scoping belongs upstream."""
    config = {
        "schema_version": 1,
        "kind": "caption_judge",
        "input": {
            "baseline": "output-a",
            "candidate": "output-b",
            field_name: "some-uuid",
        },
        "output": {"report_path": "report.json"},
    }

    with pytest.raises(ValidationError, match=field_name):
        CaptionJudgePipelineConfig.model_validate(config)


def test_prepare_inputs_emits_jobs_only_for_changed_comparable_windows() -> None:
    """Changed comparable windows become jobs; unchanged windows are skipped."""
    baseline_metadata = _metadata(
        [
            {
                "clip_uuid": "clip-1",
                "video_uuid": "video-1",
                "clip_location": "/clips/a.mp4",
                "windows": [
                    {
                        "start_ns": 0,
                        "end_ns": 1_000_000_000,
                        "caption_status": "success",
                        "caption_failure_reason": None,
                        "captions": {"qwen": "baseline"},
                    },
                    {
                        "start_ns": 1_000_000_000,
                        "end_ns": 2_000_000_000,
                        "caption_status": "success",
                        "caption_failure_reason": None,
                        "captions": {"qwen": "same"},
                    },
                ],
            }
        ]
    )
    candidate_metadata = _metadata(
        [
            {
                "clip_uuid": "clip-1",
                "video_uuid": "video-1",
                "clip_location": "/clips/b.mp4",
                "windows": [
                    {
                        "start_ns": 0,
                        "end_ns": 1_000_000_000,
                        "caption_status": "success",
                        "caption_failure_reason": None,
                        "captions": {"cosmos3_nano": "candidate"},
                    },
                    {
                        "start_ns": 1_000_000_000,
                        "end_ns": 2_000_000_000,
                        "caption_status": "success",
                        "caption_failure_reason": None,
                        "captions": {"cosmos3_nano": "same"},
                    },
                ],
            }
        ]
    )

    issues, jobs, stats = prepare_caption_judge_inputs(baseline_metadata, candidate_metadata, config=_config())

    assert issues.num_rows == 0
    assert jobs.num_rows == 1
    job = jobs.to_pylist()[0]
    assert job["clip_location"] == "/clips/b.mp4"
    assert job["start_ns"] == 0
    assert job["end_ns"] == 1_000_000_000
    assert job["baseline_caption"] == "baseline"
    assert job["candidate_caption"] == "candidate"
    assert stats.clips_in_both == 1
    assert stats.windows_in_both == 2


def test_prepare_inputs_infers_caption_models_from_metadata() -> None:
    """Caption model names are inferred from each side's captions map."""
    baseline_metadata = _metadata(
        [
            {
                "clip_uuid": "clip-1",
                "video_uuid": "video-1",
                "clip_location": "/clips/a.mp4",
                "windows": [
                    {
                        "start_ns": 0,
                        "end_ns": 10,
                        "caption_status": "success",
                        "caption_failure_reason": None,
                        "captions": {"qwen": "baseline"},
                    }
                ],
            }
        ]
    )
    candidate_metadata = _metadata(
        [
            {
                "clip_uuid": "clip-1",
                "video_uuid": "video-1",
                "clip_location": "/clips/b.mp4",
                "windows": [
                    {
                        "start_ns": 0,
                        "end_ns": 10,
                        "caption_status": "success",
                        "caption_failure_reason": None,
                        "captions": {"cosmos3_nano": "candidate"},
                    }
                ],
            }
        ]
    )

    issues, jobs, _ = prepare_caption_judge_inputs(baseline_metadata, candidate_metadata, config=_config())

    assert issues.num_rows == 0
    row = jobs.to_pylist()[0]
    assert row["caption_model_baseline"] == "qwen"
    assert row["caption_model_candidate"] == "cosmos3_nano"
    assert row["baseline_caption"] == "baseline"
    assert row["candidate_caption"] == "candidate"


def test_prepare_inputs_rejects_ambiguous_caption_model_inference() -> None:
    """A side with multiple caption keys is ambiguous."""
    baseline_metadata = _metadata(
        [
            {
                "clip_uuid": "clip-1",
                "video_uuid": "video-1",
                "clip_location": "/clips/a.mp4",
                "windows": [
                    {
                        "start_ns": 0,
                        "end_ns": 10,
                        "caption_status": "success",
                        "caption_failure_reason": None,
                        "captions": {"qwen": "baseline", "other_model": "other"},
                    }
                ],
            }
        ]
    )
    candidate_metadata = _metadata(
        [
            {
                "clip_uuid": "clip-1",
                "video_uuid": "video-1",
                "clip_location": "/clips/b.mp4",
                "windows": [
                    {
                        "start_ns": 0,
                        "end_ns": 10,
                        "caption_status": "success",
                        "caption_failure_reason": None,
                        "captions": {"cosmos3_nano": "candidate"},
                    }
                ],
            }
        ]
    )

    with pytest.raises(ValueError, match="Cannot infer caption_model_baseline"):
        prepare_caption_judge_inputs(baseline_metadata, candidate_metadata, config=_config())


def test_prepare_inputs_uses_candidate_media() -> None:
    """The candidate output root supplies the clip_location sent to the judge."""
    baseline_metadata = _metadata(
        [
            {
                "clip_uuid": "clip-1",
                "video_uuid": "video-1",
                "clip_location": "/clips/a.mp4",
                "windows": [
                    {
                        "start_ns": 0,
                        "end_ns": 10,
                        "caption_status": "success",
                        "caption_failure_reason": None,
                        "captions": {"qwen": "baseline"},
                    }
                ],
            }
        ]
    )
    candidate_metadata = _metadata(
        [
            {
                "clip_uuid": "clip-1",
                "video_uuid": "video-1",
                "clip_location": "/clips/b.mp4",
                "windows": [
                    {
                        "start_ns": 0,
                        "end_ns": 10,
                        "caption_status": "success",
                        "caption_failure_reason": None,
                        "captions": {"cosmos3_nano": "candidate"},
                    }
                ],
            }
        ]
    )

    _, jobs, _ = prepare_caption_judge_inputs(
        baseline_metadata,
        candidate_metadata,
        config=_config(),
    )

    assert jobs.to_pylist()[0]["clip_location"] == "/clips/b.mp4"


def test_prepare_inputs_rejects_windows_without_nanosecond_bounds() -> None:
    """Legacy frame-only windows are not accepted as comparable window identities."""
    baseline_metadata = _metadata(
        [
            {
                "clip_uuid": "clip-1",
                "video_uuid": "video-1",
                "clip_location": "/clips/a.mp4",
                "windows": [
                    {
                        "start_ns": None,
                        "end_ns": None,
                        "caption_status": "success",
                        "caption_failure_reason": None,
                        "captions": {"qwen": "legacy frame-bounds caption"},
                    }
                ],
            }
        ]
    )
    candidate_metadata = _metadata(
        [
            {
                "clip_uuid": "clip-1",
                "video_uuid": "video-1",
                "clip_location": "/clips/b.mp4",
                "windows": [
                    {
                        "start_ns": 0,
                        "end_ns": 10,
                        "caption_status": "success",
                        "caption_failure_reason": None,
                        "captions": {"cosmos3_nano": "candidate"},
                    }
                ],
            }
        ]
    )

    issues, jobs, _ = prepare_caption_judge_inputs(baseline_metadata, candidate_metadata, config=_config())

    assert jobs.num_rows == 0
    rows = issues.to_pylist()
    assert rows[0]["code"] == "caption_window_not_comparable"
    assert "invalid nanosecond bounds" in rows[0]["message"]


def test_prepare_inputs_reports_missing_caption_as_not_comparable() -> None:
    """A joined window with a missing model caption is recorded as not comparable."""
    baseline_metadata = _metadata(
        [
            {
                "clip_uuid": "clip-1",
                "video_uuid": "video-1",
                "clip_location": "/clips/a.mp4",
                "windows": [
                    {
                        "start_ns": 0,
                        "end_ns": 10,
                        "caption_status": "success",
                        "caption_failure_reason": None,
                        "captions": {"qwen": "baseline"},
                    }
                ],
            }
        ]
    )
    candidate_metadata = _metadata(
        [
            {
                "clip_uuid": "clip-1",
                "video_uuid": "video-1",
                "clip_location": "/clips/b.mp4",
                "windows": [
                    {
                        "start_ns": 0,
                        "end_ns": 10,
                        "caption_status": "error",
                        "caption_failure_reason": "provider",
                        "captions": {},
                    },
                    {
                        "start_ns": 10,
                        "end_ns": 20,
                        "caption_status": "success",
                        "caption_failure_reason": None,
                        "captions": {"cosmos3_nano": "candidate"},
                    },
                ],
            }
        ]
    )

    issues, jobs, _ = prepare_caption_judge_inputs(baseline_metadata, candidate_metadata, config=_config())

    assert jobs.num_rows == 0
    row = issues.to_pylist()[0]
    assert row["output"] == "candidate"
    assert "missing caption" in row["message"]


def test_prepare_inputs_preserves_missing_clip_uuid_issue_with_clip_limit() -> None:
    """Unidentifiable metadata rows remain visible even when clip_limit samples valid clips."""
    baseline_metadata = _metadata(
        [
            {
                "clip_uuid": None,
                "video_uuid": "video-1",
                "clip_location": "/clips/missing-clip.mp4",
                "windows": [],
            },
            {
                "clip_uuid": "clip-1",
                "video_uuid": "video-1",
                "clip_location": "/clips/a.mp4",
                "windows": [
                    {
                        "start_ns": 0,
                        "end_ns": 10,
                        "caption_status": "success",
                        "caption_failure_reason": None,
                        "captions": {"qwen": "same"},
                    }
                ],
            },
        ]
    )
    candidate_metadata = _metadata(
        [
            {
                "clip_uuid": "clip-1",
                "video_uuid": "video-1",
                "clip_location": "/clips/b.mp4",
                "windows": [
                    {
                        "start_ns": 0,
                        "end_ns": 10,
                        "caption_status": "success",
                        "caption_failure_reason": None,
                        "captions": {"cosmos3_nano": "same"},
                    }
                ],
            }
        ]
    )

    issues, jobs, _ = prepare_caption_judge_inputs(baseline_metadata, candidate_metadata, config=_config(clip_limit=1))

    assert jobs.num_rows == 0
    rows = issues.to_pylist()
    assert len(rows) == 1
    assert rows[0]["clip_uuid"] is None
    assert "missing clip_uuid" in rows[0]["message"]


def test_stage_sizing_caps_workers_by_nodes_and_input_windows() -> None:
    """Judge workers are capped by live nodes, per-node limit, and input size."""
    assert _stage_sizing(
        0,
        max_workers_per_node=8,
        target_batch_size=4,
        node_count=2,
    ) == (0, 0, 0)
    assert _stage_sizing(
        3,
        max_workers_per_node=8,
        target_batch_size=4,
        node_count=2,
    ) == (3, 3, 1)
    assert _stage_sizing(
        100,
        max_workers_per_node=8,
        target_batch_size=4,
        node_count=2,
    ) == (16, 25, 4)


def test_run_judge_stage_configures_ray_data_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    """The judge stage opts into Ray Data's rich progress UI before Ray init."""
    captured: dict[str, bool] = {}

    def fake_configure_ray_data_progress(*, progress: bool) -> None:
        captured["progress"] = progress

    def fake_ensure_ray_initialized() -> None:
        msg = "stop before Ray starts"
        raise RuntimeError(msg)

    monkeypatch.setattr(driver_module, "configure_ray_data_progress", fake_configure_ray_data_progress)
    monkeypatch.setattr(driver_module, "ensure_ray_initialized", fake_ensure_ray_initialized)

    with pytest.raises(RuntimeError, match="stop before Ray starts"):
        driver_module.run_judge_stage(_job_table(), config=_config(progress=False))

    assert captured == {"progress": False}


def test_run_judge_stage_disables_actor_restarts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hosted judge actors do not use Ray Data's default infinite actor restarts."""
    captured: dict[str, object] = {}

    class FakeDataset:
        def map_batches(self, *args: object, **kwargs: object) -> "FakeIssuesDataset":
            captured["map_batches_args"] = args
            captured["map_batches_kwargs"] = kwargs
            return FakeIssuesDataset()

    class FakeIssuesDataset:
        def iter_rows(self) -> list[dict[str, object]]:
            return []

    def fake_from_arrow(table: pa.Table, *, override_num_blocks: int) -> FakeDataset:
        captured["table"] = table
        captured["override_num_blocks"] = override_num_blocks
        return FakeDataset()

    def fake_configure_ray_data_progress(*, progress: bool) -> None:
        captured["progress"] = progress

    monkeypatch.setattr(driver_module, "configure_ray_data_progress", fake_configure_ray_data_progress)
    monkeypatch.setattr(driver_module, "ensure_ray_initialized", lambda: None)
    monkeypatch.setattr(driver_module, "live_ray_node_count", lambda: 1)
    monkeypatch.setattr(driver_module.ray.data, "from_arrow", fake_from_arrow)

    result = driver_module.run_judge_stage(_job_table(), config=_config())

    assert result.num_rows == 0
    assert captured["override_num_blocks"] == 1
    map_batches_kwargs = captured["map_batches_kwargs"]
    assert isinstance(map_batches_kwargs, dict)
    assert map_batches_kwargs["max_restarts"] == 0
