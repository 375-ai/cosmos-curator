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
"""Persistence for the measure phase: two Lance datasets plus a JSON manifest.

Layout under the measurements root (local path or cloud URL):

    <root>/clip.lance       -- CLIP_MEASUREMENT_SCHEMA
    <root>/window.lance     -- WINDOW_MEASUREMENT_SCHEMA
    <root>/manifest.json    -- provenance: the two compared outputs, the measure
                               config, schema versions, and run statistics

The manifest makes a measurements root self-describing so the (separate)
evaluation phase can re-evaluate it -- and verify it is evaluating the right
thing -- without being handed the original config out of band.
"""

import datetime
import json
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
import smart_open  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curator.core.utils.storage import storage_utils
from cosmos_curator.pipelines.video.split_comparison.config import SplitComparisonConfig
from cosmos_curator.pipelines.video.split_comparison.eval import EvalResult
from cosmos_curator.pipelines.video.split_comparison.measure.core import Measurements
from cosmos_curator.pipelines.video.split_comparison.measure.schema import MEASUREMENT_SCHEMA_VERSION

MANIFEST_NAME = "manifest.json"
CLIP_DATASET_NAME = "clip.lance"
WINDOW_DATASET_NAME = "window.lance"

# Each side's summary.json is snapshotted into its own subdir so the v1
# ``load_summary(output_root)`` (which appends ``summary.json``) reads it verbatim.
SUMMARY_SNAPSHOT_SUBDIRS = {"a": "summary_a", "b": "summary_b"}

ISSUES_DATASET_NAME = "issues.lance"
VERDICT_DATASET_NAME = "clip_verdict.lance"
EVAL_SUMMARY_NAME = "eval.json"

MANIFEST_SCHEMA_VERSION = 1


def eval_output_dir(root: str, eval_name: str | None) -> str:
    """Resolve the eval output directory: the root, or ``<root>/eval/<name>`` when named.

    A named eval keeps multiple policy results side by side under one
    measurements root; the unnamed default writes alongside the measurements.
    """
    return root if eval_name is None else _join(_join(root, "eval"), eval_name)


def _is_local(path: str) -> bool:
    """Whether ``path`` is a local filesystem path (vs a remote storage URI).

    Delegates to the canonical scheme detection so it can't drift from the rest of
    the storage layer (a hand-rolled ``"://" in path`` check mishandles ``file://``).
    """
    return not storage_utils.is_remote_path(path)


def _join(root: str, name: str) -> str:
    """Join ``name`` onto ``root`` as a string URI, via the canonical path joiner."""
    return str(storage_utils.get_full_path(root, name))


def _write_lance(table: pa.Table, uri: str, *, profile_name: str) -> None:
    if _is_local(uri):
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
    storage_options = storage_utils.get_lance_storage_options(uri, profile_name=profile_name)
    lance.write_dataset(table, uri, mode="overwrite", storage_options=storage_options)


def _write_json(payload: dict[str, Any], uri: str, *, profile_name: str) -> None:
    """Write a JSON document (indented, sorted keys) to ``uri``, mkdir-ing a local parent first."""
    if _is_local(uri):
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
    transport_params = storage_utils.get_smart_open_params(uri, profile_name=profile_name)
    with smart_open.open(uri, "w", encoding="utf-8", **transport_params) as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True))


def build_manifest(  # noqa: PLR0913 -- a manifest records many provenance fields
    measurements: Measurements,
    *,
    config: SplitComparisonConfig,
    device: str,
    fp16: bool,
    lance_version: str,
    created_at: str,
    summaries_snapshotted: bool,
) -> dict[str, Any]:
    """Assemble the provenance manifest dict for a measurements root.

    Records the measure-time knobs that shape the stored values -- including
    ``device`` and ``fp16`` (half precision shifts the caption similarities), so a
    root is reproducible. ``summaries_snapshotted`` records whether summary comparison
    is *in scope* for this root (i.e. it was not measured ``--no-summary``), so a later
    ``--skip-measure`` re-eval knows whether to compare summaries. It's intent, not a
    byte count: a per-side snapshot that's missing or unwritable surfaces as a
    ``summary_load_failed`` issue, it doesn't flip this flag.
    """
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "measurement_schema_version": MEASUREMENT_SCHEMA_VERSION,
        "created_at": created_at,
        "output_a": config.output_a,
        "output_b": config.output_b,
        "lance_version": lance_version,
        "profile_name": config.profile_name,
        "compare_captions": config.compare_captions,
        "caption_model_id": config.caption.model_id,
        "encode_batch_size": config.caption.encode_batch_size,
        "device": device,
        "fp16": fp16,
        "clip_dataset": CLIP_DATASET_NAME,
        "window_dataset": WINDOW_DATASET_NAME,
        "summaries_snapshotted": summaries_snapshotted,
        "stats": measurements.stats,
    }


def write_measurements(  # noqa: PLR0913 -- writes datasets + a many-field provenance manifest
    measurements: Measurements,
    root: str,
    *,
    config: SplitComparisonConfig,
    device: str,
    fp16: bool,
    lance_version: str,
    summaries_snapshotted: bool,
) -> dict[str, Any]:
    """Write the clip + window datasets and the manifest under ``root``; return the manifest.

    ``root`` is used verbatim (local path or cloud URL). ``device`` / ``fp16`` /
    ``summaries_snapshotted`` are recorded in the manifest as measure-time provenance
    (see :func:`build_manifest`). Returns the manifest dict so callers can echo it
    without re-reading.
    """
    clip_uri = _join(root, CLIP_DATASET_NAME)
    window_uri = _join(root, WINDOW_DATASET_NAME)
    manifest_uri = _join(root, MANIFEST_NAME)

    _write_lance(measurements.clip_table, clip_uri, profile_name=config.profile_name)
    _write_lance(measurements.window_table, window_uri, profile_name=config.profile_name)

    created_at = datetime.datetime.now(datetime.UTC).isoformat()
    manifest = build_manifest(
        measurements,
        config=config,
        device=device,
        fp16=fp16,
        lance_version=lance_version,
        created_at=created_at,
        summaries_snapshotted=summaries_snapshotted,
    )
    _write_json(manifest, manifest_uri, profile_name=config.profile_name)
    return manifest


def _read_lance_table(uri: str, *, profile_name: str) -> pa.Table:
    storage_options = storage_utils.get_lance_storage_options(uri, profile_name=profile_name)
    return lance.dataset(uri, storage_options=storage_options).to_table()


def read_manifest(root: str, *, profile_name: str) -> dict[str, Any]:
    """Read the measure-phase manifest from a measurements root."""
    manifest_uri = _join(root, MANIFEST_NAME)
    transport_params = storage_utils.get_smart_open_params(manifest_uri, profile_name=profile_name)
    with smart_open.open(manifest_uri, "r", encoding="utf-8", **transport_params) as handle:
        result: dict[str, Any] = json.loads(handle.read())
    return result


def read_measurements(root: str, *, profile_name: str) -> Measurements:
    """Read the clip + window measurement datasets back from a measurements root.

    Used by re-eval (``--skip-measure``): the tables are read once, then handed
    to the same ``evaluate`` the combined run uses. ``stats`` is left empty --
    the manifest is the authoritative source of measure-run statistics.
    """
    clip_table = _read_lance_table(_join(root, CLIP_DATASET_NAME), profile_name=profile_name)
    window_table = _read_lance_table(_join(root, WINDOW_DATASET_NAME), profile_name=profile_name)
    return Measurements(clip_table=clip_table, window_table=window_table, stats={})


def write_eval(
    result: EvalResult,
    root: str,
    *,
    eval_name: str | None,
    profile_name: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Write issues + verdict datasets and the eval summary under the eval output dir.

    Returns the written ``eval.json`` payload (provenance + policy + counts).
    """
    out_dir = eval_output_dir(root, eval_name)
    _write_lance(result.issues, _join(out_dir, ISSUES_DATASET_NAME), profile_name=profile_name)
    _write_lance(result.verdicts, _join(out_dir, VERDICT_DATASET_NAME), profile_name=profile_name)

    payload: dict[str, Any] = {
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "eval_name": eval_name,
        "measurements_root": root,
        **provenance,
        **result.summary,
    }
    _write_json(payload, _join(out_dir, EVAL_SUMMARY_NAME), profile_name=profile_name)
    return payload


def summary_snapshot_dir(root: str, side: str) -> str:
    """Directory under the measurements root holding side ``a``/``b``'s snapshotted summary.json."""
    return _join(root, SUMMARY_SNAPSHOT_SUBDIRS[side])


def _snapshot_one_summary(source_output_root: str, dest_dir: str, *, profile_name: str) -> bool:
    """Copy ``<source_output_root>/summary.json`` into ``<dest_dir>/summary.json``; best-effort.

    Returns False (and logs) if the source summary.json is missing/unreadable, or if
    the snapshot write fails (disk full, permission) -- neither aborts the measure run,
    so the (expensive, already-persisted) measurement tables aren't lost to a failed
    side-artifact. eval surfaces an un-snapshotted side as a ``summary_load_failed``
    issue when it reads the absent snapshot -- the same on the combined run and re-eval.
    """
    src = str(storage_utils.get_full_path(source_output_root, "summary.json"))
    dst = str(storage_utils.get_full_path(dest_dir, "summary.json"))
    try:
        src_params = storage_utils.get_smart_open_params(src, profile_name=profile_name)
        with smart_open.open(src, "rb", **src_params) as handle:
            data = handle.read()
    except Exception as exc:  # noqa: BLE001 -- a missing/unreadable source summary is non-fatal
        # Surface the actual error: a missing object and a permission/auth/transient I/O
        # failure both land here, and across backends (local vs S3) "missing" isn't a single
        # exception type, so reporting the cause beats asserting "no summary.json".
        logger.warning("Could not read source summary.json from {}; treating as absent ({})", src, exc)
        return False
    try:
        if _is_local(dst):
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
        dst_params = storage_utils.get_smart_open_params(dst, profile_name=profile_name)
        with smart_open.open(dst, "wb", **dst_params) as handle:
            handle.write(data)
    except Exception as exc:  # noqa: BLE001 -- a failed snapshot write is non-fatal; surfaced as summary_load_failed on eval
        logger.warning("Failed to write summary snapshot to {} ({})", dst, exc)
        return False
    return True


def snapshot_summaries(config: SplitComparisonConfig, root: str) -> dict[str, bool]:
    """Snapshot both outputs' summary.json into the measurements root; return per-side success."""
    return {
        "a": _snapshot_one_summary(config.output_a, summary_snapshot_dir(root, "a"), profile_name=config.profile_name),
        "b": _snapshot_one_summary(config.output_b, summary_snapshot_dir(root, "b"), profile_name=config.profile_name),
    }
