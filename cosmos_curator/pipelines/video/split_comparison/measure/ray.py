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
"""Ray Data fan-out of the window/caption measure across GPU actors.

Only the GPU-bound window/caption work fans out; clip-scalar measurement stays on
the driver. Both Lance tables are ``ray.put`` once -- actors ``ray.get`` them
zero-copy from the object store. The dataset Ray Data schedules over is the tiny
alignment index ``(clip_uuid, idx_a, idx_b)``; each actor ``take``s its batch's
rows from the shared tables, runs :func:`core.measure_window_batch` on its GPU,
and returns the (text-free) window measurement rows. The actor body is just
:func:`core.measure_window_batch`; everything here is scheduling around it.

This is the measure phase's entry point (:func:`run`). The Ray import is isolated
to this module so the pure measure core stays Ray-free and unit-testable.
"""

import math

import pyarrow as pa
import ray
from loguru import logger

from cosmos_curator.pipelines.video.split_comparison.caption_embedding import load_caption_model
from cosmos_curator.pipelines.video.split_comparison.config import SplitComparisonConfig
from cosmos_curator.pipelines.video.split_comparison.load import DEFAULT_LANCE_VERSION, index_by_clip
from cosmos_curator.pipelines.video.split_comparison.measure.core import (
    ClipPair,
    Measurements,
    apply_caption_model_opts,
    clip_measurements_columnar,
    clip_row_index,
    load_both,
    measure_stats,
    measure_window_batch,
    merge_caption_rollups,
)
from cosmos_curator.pipelines.video.split_comparison.measure.schema import WINDOW_MEASUREMENT_SCHEMA

# Schema of the tiny alignment-index dataset Ray Data schedules over: one row per
# clip in the union, with its row position in each source table (null if absent).
_ALIGNMENT_INDEX_SCHEMA = pa.schema(
    [
        pa.field("clip_uuid", pa.string(), nullable=False),
        pa.field("idx_a", pa.int64()),
        pa.field("idx_b", pa.int64()),
    ]
)


def _configure_progress(*, enabled: bool) -> None:
    """Toggle Ray Data's progress bars; use the rich renderer when enabled."""
    ctx = ray.data.DataContext.get_current()
    ctx.enable_progress_bars = enabled
    if enabled:
        ctx.enable_rich_progress_bars = True
        ctx.use_ray_tqdm = False


def resolve_num_gpus(override: int | None) -> int:
    """Return the number of GPU actors: the override if given, else Ray's detected GPU count."""
    if override is not None:
        if override <= 0:
            error_msg = f"resolve_num_gpus: num_gpus override must be a positive integer, got {override}"
            raise ValueError(error_msg)
        return override
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    gpus = int(ray.cluster_resources().get("GPU", 0))  # type: ignore[no-untyped-call]
    if gpus < 1:
        error_msg = "no GPUs detected by Ray; the measure requires at least one GPU"
        raise ValueError(error_msg)
    return gpus


class WindowActor:
    """Ray Data actor: measure caption windows for a batch of clips on one GPU.

    Holds the two shared source tables (read zero-copy from the object store) and a
    caption model loaded once on the actor's GPU. Each call takes its batch's rows
    out of the tables and runs :func:`measure_window_batch`.
    """

    def __init__(
        self,
        ref_a: "ray.ObjectRef[pa.Table]",
        ref_b: "ray.ObjectRef[pa.Table]",
        *,
        model_id: str,
        encode_batch_size: int,
        fp16: bool,
    ) -> None:
        """Resolve the shared tables from the object store and load the model on this actor's GPU."""
        self._table_a: pa.Table = ray.get(ref_a)
        self._table_b: pa.Table = ray.get(ref_b)
        self._model = apply_caption_model_opts(load_caption_model(model_id, device="cuda"), fp16=fp16, device="cuda")
        self._encode_batch_size = encode_batch_size

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Measure the caption windows for this batch's clips; return window rows."""
        uuids = batch["clip_uuid"].to_pylist()
        ids_a = pa.array([v for v in batch["idx_a"].to_pylist() if v is not None], type=pa.int64())
        ids_b = pa.array([v for v in batch["idx_b"].to_pylist() if v is not None], type=pa.int64())
        rows_a = index_by_clip(self._table_a.take(ids_a))
        rows_b = index_by_clip(self._table_b.take(ids_b))
        pairs = [ClipPair(uuid, rows_a.get(uuid), rows_b.get(uuid)) for uuid in uuids]
        window_rows = measure_window_batch(pairs, model=self._model, encode_batch_size=self._encode_batch_size)
        return pa.Table.from_pylist(window_rows, schema=WINDOW_MEASUREMENT_SCHEMA)


def run(
    config: SplitComparisonConfig,
    *,
    lance_version: str = DEFAULT_LANCE_VERSION,
    num_gpus: int | None = None,
    show_progress: bool = True,
    fp16: bool = False,
) -> Measurements:
    """Measure both outputs, fanning the window/caption work out over GPU actors.

    Clip-scalar measurements are computed on the driver in one vectorized pass
    (:func:`core.clip_measurements_columnar`) -- a flat-scalar join that never
    materializes the nested caption text, so the driver does no per-clip
    ``to_pylist``; the heavy caption work is the only thing that fans out to the
    Ray Data actor pool (one GPU each). Block count is derived from the caption
    batch size (see :func:`_stage_sizing`). Returns the ``Measurements`` (clip +
    window tables plus run stats).
    """
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    _configure_progress(enabled=show_progress)

    table_a, table_b = load_both(config, lance_version=lance_version)
    index_a = clip_row_index(table_a)
    index_b = clip_row_index(table_b)
    union = sorted(set(index_a) | set(index_b))
    clip_table = clip_measurements_columnar(table_a, table_b)

    window_table = pa.Table.from_pylist([], schema=WINDOW_MEASUREMENT_SCHEMA)
    if config.compare_captions:
        # GPUs are only required for the caption fan-out; resolving them here (not up front)
        # lets a --no-captions, clip-only measure run on a host with no GPU.
        pool_size = resolve_num_gpus(num_gpus)
        logger.info("Measuring {} clips via Ray ({} GPU actor(s))", len(union), pool_size)
        window_table = _measure_windows_ray(
            table_a, table_b, index_a, index_b, union, config=config, pool_size=pool_size, fp16=fp16
        )
    else:
        logger.info("Measuring {} clips (clip-only; captions disabled)", len(union))

    clip_table = merge_caption_rollups(clip_table, window_table)
    stats = measure_stats(union, index_a, index_b, window_table)
    logger.info("Measured {} clips: {}", len(union), stats)
    return Measurements(clip_table=clip_table, window_table=window_table, stats=stats)


def _measure_windows_ray(  # noqa: PLR0913 -- fan-out needs the tables, indexes, union, and pool config
    table_a: pa.Table,
    table_b: pa.Table,
    index_a: dict[str, int],
    index_b: dict[str, int],
    union: list[str],
    *,
    config: SplitComparisonConfig,
    pool_size: int,
    fp16: bool,
) -> pa.Table:
    """Run the window/caption measure over a Ray Data actor pool; return the window table.

    Returns the concatenated per-actor output as a single ``WINDOW_MEASUREMENT_SCHEMA``
    table -- kept columnar (no driver-side ``to_pylist``); downstream reads only the
    columns it needs.
    """
    empty = pa.Table.from_pylist([], schema=WINDOW_MEASUREMENT_SCHEMA)
    index_rows = [{"clip_uuid": uuid, "idx_a": index_a.get(uuid), "idx_b": index_b.get(uuid)} for uuid in union]
    if not index_rows:
        return empty
    ref_a = ray.put(table_a)
    ref_b = ray.put(table_b)
    pool_size, num_blocks, batch_size = _stage_sizing(len(index_rows), pool_size, config.caption.encode_batch_size)
    logger.info(
        "window stage: clips={} actors={} blocks={} batch_size={} (target={})",
        len(index_rows),
        pool_size,
        num_blocks,
        batch_size,
        config.caption.encode_batch_size,
    )
    index_dataset = ray.data.from_arrow(
        pa.Table.from_pylist(index_rows, schema=_ALIGNMENT_INDEX_SCHEMA), override_num_blocks=num_blocks
    )
    produced = index_dataset.map_batches(
        WindowActor,
        fn_constructor_kwargs={
            "ref_a": ref_a,
            "ref_b": ref_b,
            "model_id": config.caption.model_id,
            "encode_batch_size": config.caption.encode_batch_size,
            "fp16": fp16,
        },
        batch_size=batch_size,
        batch_format="pyarrow",
        concurrency=pool_size,
        num_gpus=1,
    )
    tables = ray.get(produced.to_arrow_refs())
    return pa.concat_tables(tables) if tables else empty


def _stage_sizing(num_rows: int, workers: int, target_batch_size: int) -> tuple[int, int, int]:
    """Resolve ``(pool_size, num_blocks, batch_size)`` from the actor count and target batch.

    Block count = ``ceil(num_rows / target_batch_size)``, floored at ``pool_size`` so
    every actor gets at least one block and capped at ``num_rows`` so we never
    overshoot. Many small blocks let Ray work-steal across actors, which kills the
    long tail from uneven per-clip caption work (most clips embed nothing; a few
    embed many). The realized ``batch_size`` is ``ceil(num_rows / num_blocks)``.
    """
    pool_size = min(workers, num_rows)
    num_blocks = max(pool_size, math.ceil(num_rows / target_batch_size))
    num_blocks = min(num_rows, num_blocks)
    batch_size = math.ceil(num_rows / num_blocks)
    return pool_size, num_blocks, batch_size
