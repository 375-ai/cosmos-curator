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
"""Driver and metadata join logic for caption judging."""

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any, cast

import attrs
import lance
import pyarrow as pa
import ray
from loguru import logger
from ray.data import ActorPoolStrategy

from cosmos_curator.core.utils.pixi_runtime_envs import PixiRuntimeEnv
from cosmos_curator.core.utils.storage import storage_utils
from cosmos_curator.pipelines.ray_data._runtime import (
    capped_slots_for_items,
    configure_ray_data_progress,
    ensure_ray_initialized,
    live_ray_node_count,
)
from cosmos_curator.pipelines.ray_data.caption_judge._judge import CaptionJudgeActor
from cosmos_curator.pipelines.ray_data.caption_judge.config import CaptionJudgePipelineConfig, CaptionSide
from cosmos_curator.pipelines.ray_data.caption_judge.result_model import (
    ISSUE_SCHEMA,
    JUDGE_JOB_SCHEMA,
    CaptionJudgeStats,
    Issue,
    JudgeJob,
    Report,
    empty_issues,
    make_issue,
)

WindowKey = tuple[str, int, int]

_METADATA_COLUMNS = [
    "clip_uuid",
    "video_uuid",
    "clip_location",
    "windows",
]
_MAP_ENTRY_LEN = 2
_JUDGE_CPUS_PER_WORKER = 0.25
_JUDGE_TARGET_BATCH_SIZE = 4


@attrs.define(frozen=True)
class _WindowSide:
    output_label: CaptionSide
    video_uuid: str
    clip_uuid: str
    clip_location: str | None
    start_ns: int
    end_ns: int
    caption: str | None
    caption_status: str | None
    caption_failure_reason: str | None


@attrs.define(frozen=True)
class _WindowCollection:
    windows: dict[WindowKey, _WindowSide]
    clip_uuids: frozenset[str]
    issues: tuple[Issue, ...]


@attrs.define(frozen=True)
class _CaptionModelSelection:
    baseline: str
    candidate: str


def run_caption_judge_pipeline(*, config: CaptionJudgePipelineConfig) -> Report:
    """Run the caption judge pipeline and return its report."""
    started = time.perf_counter()
    logger.info("Reading caption judge baseline metadata: {}", config.baseline_metadata)
    baseline_metadata = read_metadata_table(config.baseline_metadata)
    logger.info("Reading caption judge candidate metadata: {}", config.candidate_metadata)
    candidate_metadata = read_metadata_table(config.candidate_metadata)

    caption_models = _infer_caption_models(baseline_metadata, candidate_metadata, config=config)
    pre_judge_issues, jobs, stats = prepare_caption_judge_inputs(
        baseline_metadata,
        candidate_metadata,
        config=config,
        caption_models=caption_models,
    )
    if jobs.num_rows:
        logger.info("Running caption judge stage for {} windows", jobs.num_rows)
        judge_issues = run_judge_stage(jobs, config=config)
    else:
        logger.info("No caption windows require hosted judging")
        judge_issues = empty_issues()
    issues = pa.concat_tables([pre_judge_issues, judge_issues])
    return Report(
        issues=issues,
        passed=issues.num_rows == 0,
        stats=attrs.evolve(stats, windows_judged=jobs.num_rows),
        baseline=config.input.baseline,
        candidate=config.input.candidate,
        baseline_metadata=config.baseline_metadata,
        candidate_metadata=config.candidate_metadata,
        caption_model_baseline=caption_models.baseline,
        caption_model_candidate=caption_models.candidate,
        runtime_sec=time.perf_counter() - started,
        config=config,
    )


def read_metadata_table(metadata_uri: str) -> pa.Table:
    """Read a Lance metadata dataset as an Arrow table with only required columns."""
    dataset = lance.dataset(
        metadata_uri,
        storage_options=storage_utils.get_lance_storage_options(metadata_uri),
    )
    return dataset.to_table(columns=_METADATA_COLUMNS)


def prepare_caption_judge_inputs(
    baseline_metadata: pa.Table,
    candidate_metadata: pa.Table,
    *,
    config: CaptionJudgePipelineConfig,
    caption_models: _CaptionModelSelection | None = None,
) -> tuple[pa.Table, pa.Table, CaptionJudgeStats]:
    """Build pre-judge issues and hosted-judge jobs from two metadata tables."""
    if caption_models is None:
        caption_models = _infer_caption_models(baseline_metadata, candidate_metadata, config=config)
    caption_model_baseline = caption_models.baseline
    caption_model_candidate = caption_models.candidate
    baseline_windows = _collect_caption_windows(
        baseline_metadata,
        output_label="baseline",
        caption_model=caption_model_baseline,
        caption_models=caption_models,
    )
    candidate_windows = _collect_caption_windows(
        candidate_metadata,
        output_label="candidate",
        caption_model=caption_model_candidate,
        caption_models=caption_models,
    )
    if config.input.clip_limit is not None:
        allowed = frozenset(
            sorted(baseline_windows.clip_uuids | candidate_windows.clip_uuids)[: config.input.clip_limit]
        )
        baseline_windows = _filter_collection(baseline_windows, allowed)
        candidate_windows = _filter_collection(candidate_windows, allowed)

    issues: list[Issue] = [*baseline_windows.issues, *candidate_windows.issues]
    jobs: list[JudgeJob] = []
    all_keys = sorted(set(baseline_windows.windows) | set(candidate_windows.windows))
    for key in all_keys:
        baseline_side = baseline_windows.windows.get(key)
        candidate_side = candidate_windows.windows.get(key)
        if baseline_side is None or candidate_side is None:
            if baseline_side is None:
                missing: CaptionSide = "baseline"
                present = candidate_side
            else:
                missing = "candidate"
                present = baseline_side
            assert present is not None
            issues.append(
                _make_config_issue(
                    "caption_window_not_comparable",
                    f"Window exists only in {present.output_label} output",
                    caption_models=caption_models,
                    video_uuid=present.video_uuid,
                    clip_uuid=key[0],
                    start_ns=key[1],
                    end_ns=key[2],
                    output=missing,
                )
            )
            continue
        if baseline_side.video_uuid != candidate_side.video_uuid:
            issues.append(
                _make_config_issue(
                    "caption_window_not_comparable",
                    "Window has mismatched video_uuid values",
                    caption_models=caption_models,
                    video_uuid=baseline_side.video_uuid,
                    clip_uuid=key[0],
                    start_ns=key[1],
                    end_ns=key[2],
                    output=None,
                )
            )
            continue
        if baseline_side.caption is None or candidate_side.caption is None:
            missing = "baseline" if baseline_side.caption is None else "candidate"
            side = baseline_side if baseline_side.caption is None else candidate_side
            issues.append(
                _make_config_issue(
                    "caption_window_not_comparable",
                    _missing_caption_message(
                        side,
                        caption_model=caption_model_baseline if missing == "baseline" else caption_model_candidate,
                    ),
                    caption_models=caption_models,
                    video_uuid=side.video_uuid,
                    clip_uuid=key[0],
                    start_ns=key[1],
                    end_ns=key[2],
                    output=missing,
                    baseline_caption=baseline_side.caption,
                    candidate_caption=candidate_side.caption,
                )
            )
            continue
        if baseline_side.caption == candidate_side.caption:
            continue
        if not candidate_side.clip_location:
            issues.append(
                _make_config_issue(
                    "caption_window_not_comparable",
                    "Candidate output is missing clip_location",
                    caption_models=caption_models,
                    video_uuid=candidate_side.video_uuid,
                    clip_uuid=key[0],
                    start_ns=key[1],
                    end_ns=key[2],
                    output="candidate",
                    baseline_caption=baseline_side.caption,
                    candidate_caption=candidate_side.caption,
                )
            )
            continue
        jobs.append(
            JudgeJob(
                video_uuid=candidate_side.video_uuid,
                clip_uuid=key[0],
                start_ns=key[1],
                end_ns=key[2],
                clip_location=candidate_side.clip_location,
                caption_model_baseline=caption_model_baseline,
                caption_model_candidate=caption_model_candidate,
                baseline_caption=baseline_side.caption,
                candidate_caption=candidate_side.caption,
            )
        )

    stats = CaptionJudgeStats(
        clips_in_baseline=len(baseline_windows.clip_uuids),
        clips_in_candidate=len(candidate_windows.clip_uuids),
        clips_in_both=len(baseline_windows.clip_uuids & candidate_windows.clip_uuids),
        windows_in_baseline=len(baseline_windows.windows),
        windows_in_candidate=len(candidate_windows.windows),
        windows_in_both=len(set(baseline_windows.windows) & set(candidate_windows.windows)),
    )
    return (
        pa.Table.from_pylist(issues, schema=ISSUE_SCHEMA),
        pa.Table.from_pylist(jobs, schema=JUDGE_JOB_SCHEMA),
        stats,
    )


def _infer_caption_models(
    baseline_metadata: pa.Table,
    candidate_metadata: pa.Table,
    *,
    config: CaptionJudgePipelineConfig,
) -> _CaptionModelSelection:
    allowed_clip_uuids = _allowed_clip_uuids_for_inference(baseline_metadata, candidate_metadata, config=config)
    caption_model_baseline = _resolve_caption_model(
        baseline_metadata,
        output_label="baseline",
        allowed_clip_uuids=allowed_clip_uuids,
    )
    caption_model_candidate = _resolve_caption_model(
        candidate_metadata,
        output_label="candidate",
        allowed_clip_uuids=allowed_clip_uuids,
    )
    return _CaptionModelSelection(baseline=caption_model_baseline, candidate=caption_model_candidate)


def _resolve_caption_model(
    table: pa.Table,
    *,
    output_label: CaptionSide,
    allowed_clip_uuids: frozenset[str] | None,
) -> str:
    field_name = f"caption_model_{output_label}"
    caption_models = sorted(_caption_models_in_table(table, allowed_clip_uuids=allowed_clip_uuids))
    if len(caption_models) == 1:
        model_name = caption_models[0]
        logger.info("Inferred {}={} from {} metadata", field_name, model_name, output_label)
        return model_name
    if not caption_models:
        msg = f"Cannot infer {field_name}: {output_label} output has no caption models"
        raise ValueError(msg)
    models = ", ".join(caption_models)
    msg = (
        f"Cannot infer {field_name}: {output_label} output has multiple caption models ({models}); "
        "caption_judge requires exactly one base caption model per side"
    )
    raise ValueError(msg)


def _caption_models_in_table(
    table: pa.Table,
    *,
    allowed_clip_uuids: frozenset[str] | None,
) -> set[str]:
    caption_models: set[str] = set()
    for row in table.to_pylist():
        if not _row_matches_selection(row, allowed_clip_uuids=allowed_clip_uuids):
            continue
        for window in _iter_window_dicts(row.get("windows")):
            caption_models.update(_caption_model_names(window.get("captions")))
    return caption_models


def _caption_model_names(captions: object) -> set[str]:
    if isinstance(captions, Mapping):
        return {key for key in captions if isinstance(key, str)}
    if not isinstance(captions, Sequence) or isinstance(captions, (bytes, str)):
        return set()
    names: set[str] = set()
    for item in captions:
        if isinstance(item, (list, tuple)) and len(item) == _MAP_ENTRY_LEN and isinstance(item[0], str):
            names.add(item[0])
    return names


def _allowed_clip_uuids_for_inference(
    baseline_metadata: pa.Table,
    candidate_metadata: pa.Table,
    *,
    config: CaptionJudgePipelineConfig,
) -> frozenset[str] | None:
    if config.input.clip_limit is None:
        return None
    clip_uuids = _selected_clip_uuids(baseline_metadata) | _selected_clip_uuids(candidate_metadata)
    return frozenset(sorted(clip_uuids)[: config.input.clip_limit])


def _selected_clip_uuids(table: pa.Table) -> set[str]:
    clip_uuids: set[str] = set()
    for row in table.to_pylist():
        if _row_matches_selection(row, allowed_clip_uuids=None):
            clip_uuid = _string_or_none(row.get("clip_uuid"))
            if clip_uuid is not None:
                clip_uuids.add(clip_uuid)
    return clip_uuids


def _row_matches_selection(
    row: Mapping[str, Any],
    *,
    allowed_clip_uuids: frozenset[str] | None,
) -> bool:
    clip_uuid = _string_or_none(row.get("clip_uuid"))
    video_uuid = _string_or_none(row.get("video_uuid"))
    if clip_uuid is None or video_uuid is None:
        return False
    return allowed_clip_uuids is None or clip_uuid in allowed_clip_uuids


def _collect_caption_windows(
    table: pa.Table,
    *,
    output_label: CaptionSide,
    caption_model: str,
    caption_models: _CaptionModelSelection,
) -> _WindowCollection:
    windows: dict[WindowKey, _WindowSide] = {}
    clip_uuids: set[str] = set()
    issues: list[Issue] = []
    for row in table.to_pylist():
        clip_uuid = _string_or_none(row.get("clip_uuid"))
        video_uuid = _string_or_none(row.get("video_uuid"))
        if clip_uuid is None:
            issues.append(
                _make_config_issue(
                    "caption_window_not_comparable",
                    f"{output_label.capitalize()} metadata row is missing clip_uuid",
                    caption_models=caption_models,
                    output=output_label,
                )
            )
            continue
        if video_uuid is None:
            issues.append(
                _make_config_issue(
                    "caption_window_not_comparable",
                    f"{output_label.capitalize()} metadata row is missing video_uuid",
                    caption_models=caption_models,
                    clip_uuid=clip_uuid,
                    output=output_label,
                )
            )
            continue
        clip_uuids.add(clip_uuid)
        clip_location = _string_or_none(row.get("clip_location"))
        for window in _iter_window_dicts(row.get("windows")):
            start_ns = _int_or_none(window.get("start_ns"))
            end_ns = _int_or_none(window.get("end_ns"))
            if start_ns is None or end_ns is None or end_ns <= start_ns:
                issues.append(
                    _make_config_issue(
                        "caption_window_not_comparable",
                        f"{output_label.capitalize()} window has invalid nanosecond bounds",
                        caption_models=caption_models,
                        video_uuid=video_uuid,
                        clip_uuid=clip_uuid,
                        start_ns=start_ns,
                        end_ns=end_ns,
                        output=output_label,
                    )
                )
                continue
            key = (clip_uuid, start_ns, end_ns)
            if key in windows:
                issues.append(
                    _make_config_issue(
                        "caption_window_not_comparable",
                        f"{output_label.capitalize()} output has duplicate windows for clip_uuid/start_ns/end_ns",
                        caption_models=caption_models,
                        video_uuid=video_uuid,
                        clip_uuid=clip_uuid,
                        start_ns=start_ns,
                        end_ns=end_ns,
                        output=output_label,
                    )
                )
                continue
            windows[key] = _WindowSide(
                output_label=output_label,
                video_uuid=video_uuid,
                clip_uuid=clip_uuid,
                clip_location=clip_location,
                start_ns=start_ns,
                end_ns=end_ns,
                caption=_caption_text(window.get("captions"), caption_model),
                caption_status=_string_or_none(window.get("caption_status")),
                caption_failure_reason=_string_or_none(window.get("caption_failure_reason")),
            )
    return _WindowCollection(windows=windows, clip_uuids=frozenset(clip_uuids), issues=tuple(issues))


def _filter_collection(collection: _WindowCollection, allowed_clip_uuids: frozenset[str]) -> _WindowCollection:
    return _WindowCollection(
        windows={key: side for key, side in collection.windows.items() if key[0] in allowed_clip_uuids},
        clip_uuids=frozenset(clip for clip in collection.clip_uuids if clip in allowed_clip_uuids),
        issues=tuple(
            issue
            for issue in collection.issues
            if issue.get("clip_uuid") is None or issue.get("clip_uuid") in allowed_clip_uuids
        ),
    )


def _iter_window_dicts(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (bytes, str)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _caption_text(captions: object, caption_model: str) -> str | None:
    value = _map_get(captions, caption_model)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _map_get(value: object, key: str) -> object | None:
    if isinstance(value, Mapping):
        return value.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (bytes, str)):
        return None
    for item in value:
        if isinstance(item, tuple) and len(item) == _MAP_ENTRY_LEN and item[0] == key:
            return cast("object", item[1])
        if isinstance(item, list) and len(item) == _MAP_ENTRY_LEN and item[0] == key:
            return cast("object", item[1])
    return None


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _missing_caption_message(side: _WindowSide, *, caption_model: str) -> str:
    base = f"{side.output_label.capitalize()} output is missing caption for model {caption_model!r}"
    if side.caption_status is None:
        return base
    if side.caption_failure_reason:
        return f"{base} (status={side.caption_status}, failure_reason={side.caption_failure_reason})"
    return f"{base} (status={side.caption_status})"


def _make_config_issue(  # noqa: PLR0913 -- passes through the persisted issue schema fields.
    code: str,
    message: str,
    *,
    caption_models: _CaptionModelSelection,
    video_uuid: str | None = None,
    clip_uuid: str | None = None,
    start_ns: int | None = None,
    end_ns: int | None = None,
    output: str | None = None,
    baseline_caption: str | None = None,
    candidate_caption: str | None = None,
) -> Issue:
    return make_issue(
        code,  # type: ignore[arg-type]
        message,
        video_uuid=video_uuid,
        clip_uuid=clip_uuid,
        start_ns=start_ns,
        end_ns=end_ns,
        output=output,
        caption_model_baseline=caption_models.baseline,
        caption_model_candidate=caption_models.candidate,
        baseline_caption=baseline_caption,
        candidate_caption=candidate_caption,
    )


def run_judge_stage(jobs: pa.Table, *, config: CaptionJudgePipelineConfig) -> pa.Table:
    """Run hosted caption judging over the prepared job table."""
    if jobs.num_rows == 0:
        return empty_issues()
    configure_ray_data_progress(progress=config.execution.progress)
    ensure_ray_initialized()
    node_count = live_ray_node_count()
    pool_size, num_blocks, batch_size = _stage_sizing(
        jobs.num_rows,
        max_workers_per_node=config.execution.max_workers_per_node,
        target_batch_size=_JUDGE_TARGET_BATCH_SIZE,
        node_count=node_count,
    )
    logger.info(
        "caption judge stage: windows={} nodes={} max_workers_per_node={} workers={} blocks={} batch_size={} "
        "(target={})",
        jobs.num_rows,
        node_count,
        config.execution.max_workers_per_node,
        pool_size,
        num_blocks,
        batch_size,
        _JUDGE_TARGET_BATCH_SIZE,
    )
    ds = ray.data.from_arrow(jobs, override_num_blocks=num_blocks)
    issues_ds = ds.map_batches(
        CaptionJudgeActor,
        batch_format="pyarrow",
        batch_size=batch_size,
        compute=ActorPoolStrategy(size=pool_size),
        num_cpus=_JUDGE_CPUS_PER_WORKER,
        runtime_env=PixiRuntimeEnv(CaptionJudgeActor.conda_env_name),
        fn_constructor_kwargs={"config": config},
        max_restarts=0,
    )
    rows = list(issues_ds.iter_rows())
    return pa.Table.from_pylist(rows, schema=ISSUE_SCHEMA)


def _stage_sizing(
    num_rows: int,
    *,
    max_workers_per_node: int,
    target_batch_size: int,
    node_count: int,
) -> tuple[int, int, int]:
    if num_rows <= 0:
        return 0, 0, 0
    pool_size = capped_slots_for_items(
        num_items=num_rows,
        num_nodes=node_count,
        max_per_node=max_workers_per_node,
    )
    num_blocks = max(pool_size, math.ceil(num_rows / target_batch_size))
    num_blocks = min(num_rows, num_blocks)
    batch_size = math.ceil(num_rows / num_blocks)
    return pool_size, num_blocks, batch_size
