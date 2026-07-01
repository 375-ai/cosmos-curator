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
"""Ray-free measure core: the primitives the Ray driver builds on.

No thresholds, no Ray -- this module computes facts (per-side values, absolute
differences, caption cosine similarities) that the evaluation phase turns into
issues. The Ray driver (:mod:`ray`) wires these primitives into a full measure
run; keeping them here, Ray-free, is what lets them be unit-tested without a GPU
or a Ray runtime. The single expensive step is the caption embedding
(:func:`measure_window_batch`), run as one batched pass per actor block.
"""

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, NamedTuple

import pyarrow as pa
import pyarrow.compute as _compute
from loguru import logger

from cosmos_curator.pipelines.video.split_comparison.caption_embedding import cosine_similarity_batch
from cosmos_curator.pipelines.video.split_comparison.config import SplitComparisonConfig
from cosmos_curator.pipelines.video.split_comparison.load import DEFAULT_LANCE_VERSION, load_clip_metadata
from cosmos_curator.pipelines.video.split_comparison.measure.schema import CLIP_MEASUREMENT_SCHEMA

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

# pyarrow.compute's generated functions are only partially in the type stubs; bind it as
# Any so the columnar ops below type-check without a per-call ignore on every pc.* access.
pc: Any = _compute

_CAPTION_KINDS: tuple[tuple[str, str], ...] = (("caption", "captions"), ("enhanced", "enhanced_captions"))


class Measurements(NamedTuple):
    """The measure phase's output: two Arrow tables plus run statistics."""

    clip_table: pa.Table
    window_table: pa.Table
    stats: dict[str, int]


class ClipPair(NamedTuple):
    """One clip aligned across both outputs; either side is ``None`` when one-sided."""

    clip_uuid: str
    row_a: Mapping[str, Any] | None
    row_b: Mapping[str, Any] | None


class _CaptionSimilarityJob(NamedTuple):
    """A divergent caption pair whose similarity must be embedded; ``row`` is patched in place."""

    row: dict[str, Any]
    text_a: str
    text_b: str


def load_both(
    config: SplitComparisonConfig, *, lance_version: str = DEFAULT_LANCE_VERSION
) -> tuple[pa.Table, pa.Table]:
    """Load both outputs' clip-metadata tables concurrently.

    The two loads are independent S3 scans and the dominant cost; they're I/O-bound,
    so a 2-thread pool overlaps the network/Lance reads despite the GIL.
    """
    logger.info("Loading clip metadata concurrently: a={} b={}", config.output_a, config.output_b)
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(
            load_clip_metadata, config.output_a, profile_name=config.profile_name, version=lance_version
        )
        future_b = pool.submit(
            load_clip_metadata, config.output_b, profile_name=config.profile_name, version=lance_version
        )
        return future_a.result(), future_b.result()


def apply_caption_model_opts(model: "SentenceTransformer", *, fp16: bool, device: str) -> "SentenceTransformer":
    """Apply inference-time optimizations to a freshly loaded caption model.

    ``fp16`` casts the model to half precision -- roughly 2x faster forward on GPU
    at a small numeric cost (caption similarities shift slightly). Skipped off-GPU,
    where half precision isn't useful and some ops are unimplemented for it.
    """
    if fp16 and device.startswith("cuda"):
        return model.half()
    if fp16:
        logger.warning("fp16 requested but device={!r} is not CUDA; keeping full precision", device)
    return model


def measure_window_batch(
    pairs: Sequence[ClipPair],
    *,
    model: "SentenceTransformer",
    encode_batch_size: int,
    show_progress_bar: bool = False,
) -> list[dict[str, Any]]:
    """Window measurements (breakout + caption embedding) for a batch of pairs.

    The GPU-bound half of the measure, isolated as the future Ray Data actor body:
    walk each pair's caption windows, queue the divergent pairs, embed them in one
    batched pass, and return the window rows. Caption dedup is batch-scoped. The
    caller computes clip-scalar measurements separately.
    """
    window_rows: list[dict[str, Any]] = []
    caption_similarity_jobs: list[_CaptionSimilarityJob] = []
    for pair in pairs:
        rows, jobs = _window_measurements(pair.clip_uuid, _video_uuid(pair.row_a, pair.row_b), pair.row_a, pair.row_b)
        window_rows.extend(rows)
        caption_similarity_jobs.extend(jobs)
    if caption_similarity_jobs:
        compute_caption_similarities(
            caption_similarity_jobs,
            model=model,
            batch_size=encode_batch_size,
            show_progress_bar=show_progress_bar,
        )
    return window_rows


def clip_row_index(table: pa.Table) -> dict[str, int]:
    """Map ``clip_uuid -> row index`` from the clip_uuid column only.

    Cheap (one string column, not the whole table) -- this is the per-side
    alignment index the Ray actors use to ``take`` only their batch's rows from
    the shared tables. Null clip_uuids are skipped; duplicates keep the last index
    (matches :func:`index_by_clip`).
    """
    index: dict[str, int] = {}
    for row_index, clip_uuid in enumerate(table.column("clip_uuid").to_pylist()):
        if clip_uuid is not None:
            index[str(clip_uuid)] = row_index
    return index


def measure_stats(
    union: Sequence[str],
    index_a: Mapping[str, int],
    index_b: Mapping[str, int],
    window_table: pa.Table,
) -> dict[str, int]:
    """Derive run statistics from the clip union, per-side presence, and the window table.

    Reads only the ``identical`` column off the (columnar) window table -- avoids
    materializing the full rows on the driver.
    """
    identical = window_table["identical"].to_pylist()
    return {
        "num_clips": len(union),
        "num_clips_one_sided": sum(1 for uuid in union if uuid not in index_a or uuid not in index_b),
        "num_windows": window_table.num_rows,
        "num_windows_identical": sum(1 for flag in identical if flag is True),
        "num_caption_pairs_embedded": sum(1 for flag in identical if flag is False),
    }


def clip_measurements_columnar(table_a: pa.Table, table_b: pa.Table) -> pa.Table:
    """Compute the clip measurement table from the two source tables, vectorized.

    Project each side to flat scalars (motion struct flattened; num_windows via list
    length, so nested caption text is never materialized), full-outer join on
    ``clip_uuid``, then compute diffs/equality with ``pyarrow.compute``. The per-row
    reference implementation (``_clip_measurement``) lives in the parity test
    ``tests/.../measure/test_core.py``, which pins this function's semantics.
    ``min/mean_caption_similarity`` are left null here -- :func:`merge_caption_rollups`
    fills them from the window
    measurements. Equality matches Python's ``==``-with-None semantics (see
    :func:`_equal_col`). Produces ``CLIP_MEASUREMENT_SCHEMA``.

    Assumes ``clip_uuid`` is unique per source table (producer contract). A duplicate
    would fan the full-outer join into a Cartesian product for that clip; we don't
    guard against it -- it's an upstream-invariant violation, not expected input.
    """
    joined = _clip_scalars(table_a).join(
        _clip_scalars(table_b), keys="clip_uuid", join_type="full outer", left_suffix="_a", right_suffix="_b"
    )
    present_a = pc.is_valid(joined["present_a"])
    present_b = pc.is_valid(joined["present_b"])
    both_present = pc.and_(present_a, present_b)
    nulls = pa.nulls(joined.num_rows, pa.float64())
    columns: dict[str, Any] = {
        "clip_uuid": joined["clip_uuid"],
        "video_uuid": pc.coalesce(joined["video_uuid_a"], joined["video_uuid_b"]),
        "present_a": present_a,
        "present_b": present_b,
        **_score_triple_columnar(joined, "aesthetic"),
        **_score_triple_columnar(joined, "motion_global_mean"),
        **_score_triple_columnar(joined, "motion_per_patch_min_256"),
        "valid_a": joined["valid_a"],
        "valid_b": joined["valid_b"],
        "valid_equal": _equal_col(joined["valid_a"], joined["valid_b"], both_present),
        "has_caption_a": joined["has_caption_a"],
        "has_caption_b": joined["has_caption_b"],
        "has_caption_equal": _equal_col(joined["has_caption_a"], joined["has_caption_b"], both_present),
        "rejection_stage_a": joined["rejection_stage_a"],
        "rejection_stage_b": joined["rejection_stage_b"],
        "rejection_stage_equal": _equal_col(joined["rejection_stage_a"], joined["rejection_stage_b"], both_present),
        "num_windows_a": joined["num_windows_a"],
        "num_windows_b": joined["num_windows_b"],
        "min_caption_similarity": nulls,
        "mean_caption_similarity": nulls,
    }
    return pa.table(columns).cast(CLIP_MEASUREMENT_SCHEMA)


def merge_caption_rollups(clip_table: pa.Table, window_table: pa.Table) -> pa.Table:
    """Fill ``min``/``mean_caption_similarity`` on the clip table from the window table.

    Reads only the ``clip_uuid`` + ``similarity`` columns off the (columnar) window
    table. min/mean are computed in Python (``sum/len``) -- Arrow's vectorized mean
    differs from this by ULPs -- then merged columnar via a left join on ``clip_uuid``.
    Clips with no computed window similarity keep null rollups.
    """
    by_clip: dict[str, list[float]] = {}
    clip_uuids = window_table["clip_uuid"].to_pylist()
    similarities = window_table["similarity"].to_pylist()
    for clip_uuid, similarity in zip(clip_uuids, similarities, strict=True):
        if similarity is not None:
            by_clip.setdefault(clip_uuid, []).append(similarity)
    if not by_clip:
        return clip_table
    rollup = pa.table(
        {
            "clip_uuid": pa.array(list(by_clip), type=pa.string()),
            "min_caption_similarity": pa.array([min(sims) for sims in by_clip.values()], type=pa.float64()),
            "mean_caption_similarity": pa.array(
                [sum(sims) / len(sims) for sims in by_clip.values()], type=pa.float64()
            ),
        }
    )
    base = clip_table.drop_columns(["min_caption_similarity", "mean_caption_similarity"])
    merged = base.join(rollup, keys="clip_uuid", join_type="left outer")
    return pa.table({name: merged[name] for name in CLIP_MEASUREMENT_SCHEMA.names}).cast(CLIP_MEASUREMENT_SCHEMA)


def _clip_scalars(table: pa.Table) -> pa.Table:
    """Project a source table to the flat scalar columns the clip join needs (no caption text)."""
    motion = table["motion_score"]
    return pa.table(
        {
            "clip_uuid": table["clip_uuid"],
            "video_uuid": table["video_uuid"],
            # Marker so the full-outer join records per-side presence (null => absent on that side).
            "present": pa.array([True] * table.num_rows, type=pa.bool_()),
            "aesthetic": _finite_col(table["aesthetic_score"]),
            "motion_global_mean": _finite_col(pc.struct_field(motion, "global_mean")),
            "motion_per_patch_min_256": _finite_col(pc.struct_field(motion, "per_patch_min_256")),
            "valid": table["valid"],
            "has_caption": table["has_caption"],
            "rejection_stage": table["rejection_stage"],
            # num_windows via list length -> never materializes the nested window contents.
            "num_windows": pc.fill_null(pc.cast(pc.list_value_length(table["windows"]), pa.int64()), 0),
        }
    )


def _score_triple_columnar(joined: pa.Table, prefix: str) -> dict[str, pa.Array]:
    """Columnar ``{prefix}_a`` / ``_b`` / ``_abs_diff`` (abs diff; null if either side is null)."""
    col_a = joined[f"{prefix}_a"]
    col_b = joined[f"{prefix}_b"]
    return {
        f"{prefix}_a": col_a,
        f"{prefix}_b": col_b,
        f"{prefix}_abs_diff": pc.abs(pc.subtract(col_a, col_b)),
    }


def _finite_col(values: pa.ChunkedArray) -> pa.Array:
    """Float column with absent / non-finite (NaN, +-inf) values as null."""
    arr = pc.cast(values, pa.float64())
    return pc.if_else(pc.is_finite(arr), arr, pa.scalar(None, type=pa.float64()))


def _equal_col(col_a: pa.ChunkedArray, col_b: pa.ChunkedArray, both_present: pa.Array) -> pa.Array:
    """Per-clip equality flag with Python ``==``-with-None semantics.

    Replicates: both null -> True, one null -> False,
    both present non-null -> their equality. Null when the clip is one-sided
    (``both_present`` False), since there is nothing to compare.
    """
    both_null = pc.and_(pc.is_null(col_a), pc.is_null(col_b))
    equal = pc.or_(pc.fill_null(pc.equal(col_a, col_b), False), both_null)  # noqa: FBT003 -- positional fill value
    return pc.if_else(both_present, equal, pa.scalar(None, type=pa.bool_()))


def _video_uuid(row_a: Mapping[str, Any] | None, row_b: Mapping[str, Any] | None) -> str | None:
    """Return the clip's video_uuid from whichever side is present (consistent across both sides)."""
    a = row_a or {}
    b = row_b or {}
    return str(a.get("video_uuid") or b.get("video_uuid") or "") or None


def _window_measurements(
    clip_uuid: str,
    video_uuid: str | None,
    row_a: Mapping[str, Any] | None,
    row_b: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[_CaptionSimilarityJob]]:
    """Return one window row per ``(window, model, kind)`` across both sides, plus embed jobs.

    Identical text resolves to similarity ``1.0`` in the row; divergent pairs present
    on both sides get an :class:`_CaptionSimilarityJob` holding the row to patch once embedding
    completes. One-sided windows/models leave ``similarity`` null.
    """
    windows_a = _windows_by_bounds(row_a)
    windows_b = _windows_by_bounds(row_b)
    rows: list[dict[str, Any]] = []
    jobs: list[_CaptionSimilarityJob] = []
    # bounds are (start_ns, end_ns) with int | None elements (see _windows_by_bounds); a null
    # bound (absent in source) is unorderable against an int, so sort None first via a sentinel.
    for bounds in sorted(
        set(windows_a) | set(windows_b), key=lambda b: tuple(float("-inf") if v is None else v for v in b)
    ):
        start_ns, end_ns = bounds
        win_a = windows_a.get(bounds)
        win_b = windows_b.get(bounds)
        for kind, key in _CAPTION_KINDS:
            captions_a = _caption_map(win_a, key)
            captions_b = _caption_map(win_b, key)
            for model in sorted(set(captions_a) | set(captions_b)):
                present_a = model in captions_a
                present_b = model in captions_b
                text_a = captions_a.get(model, "")
                text_b = captions_b.get(model, "")
                identical = present_a and present_b and text_a == text_b
                row = {
                    "clip_uuid": clip_uuid,
                    "video_uuid": video_uuid,
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "model": model,
                    "kind": kind,
                    "present_a": present_a,
                    "present_b": present_b,
                    "identical": identical if (present_a and present_b) else None,
                    "similarity": 1.0 if identical else None,
                    "len_a": len(text_a),
                    "len_b": len(text_b),
                }
                rows.append(row)
                if present_a and present_b and not identical:
                    jobs.append(_CaptionSimilarityJob(row=row, text_a=text_a, text_b=text_b))
    return rows, jobs


def compute_caption_similarities(
    jobs: Sequence[_CaptionSimilarityJob],
    *,
    model: "SentenceTransformer",
    batch_size: int,
    show_progress_bar: bool,
) -> None:
    """Compute each job's caption cosine similarity and write it to ``job.row["similarity"]``.

    Runs one batched ``encode()`` over all ``jobs`` (caption dedup is batch-scoped),
    then fills each job's window row in place -- the deferred resolution of the
    ``similarity`` left null at window-measurement time. Mutates the rows; returns nothing.
    """
    similarities = cosine_similarity_batch(
        model,
        [job.text_a for job in jobs],
        [job.text_b for job in jobs],
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )
    for job, sim in zip(jobs, similarities, strict=True):
        job.row["similarity"] = float(sim)


def _windows(row: Mapping[str, Any] | None) -> list[Any]:
    """Return the row's caption windows as a list, or empty when absent/malformed."""
    if row is None:
        return []
    windows = row.get("windows")
    if not isinstance(windows, Sequence) or isinstance(windows, (bytes, str)):
        return []
    return list(windows)


def _windows_by_bounds(row: Mapping[str, Any] | None) -> dict[tuple[int | None, int | None], Mapping[str, Any]]:
    return {(w.get("start_ns"), w.get("end_ns")): w for w in _windows(row) if isinstance(w, Mapping)}


def _caption_map(window: Mapping[str, Any] | None, key: str) -> dict[str, str]:
    if window is None:
        return {}
    raw = window.get(key)
    if isinstance(raw, Mapping):
        items: Any = raw.items()
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, str)):
        items = raw
    else:
        return {}
    captions: dict[str, str] = {}
    for pair in items:
        if isinstance(pair, (tuple, list)) and len(pair) == 2:  # noqa: PLR2004 -- (model, text) pair
            model, text = pair
            if model is not None and text is not None:
                captions[str(model)] = str(text)
    return captions
