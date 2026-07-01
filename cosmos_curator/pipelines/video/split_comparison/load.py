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
"""Load a split output's clip-metadata Lance dataset as an Arrow table.

A whole output's clip metadata is one columnar scan. ``load_clip_metadata``
projects to exactly the columns the measure phase consumes
(:data:`REQUIRED_CLIP_COLUMNS`) -- which both shrinks the S3 read + materialization
(everything else, including the large ``embedding`` and ``windows`` text we don't
need, stays on the wire) and doubles as a schema check: a missing required column
fails fast with the dataset URI. It returns the ``pa.Table`` so the driver can hand
it to Ray (``ray.put`` -> zero-copy shared, workers ``take`` their batch).
``index_by_clip`` materializes a table into a ``clip_uuid -> row`` dict for the
row-wise window logic -- done once over the whole table single-process, or
per-batch in a Ray worker.
"""

from typing import Any

import lance
import pyarrow as pa
import pyarrow.compute as _compute
from loguru import logger

from cosmos_curator.core.utils.storage import storage_utils

# pyarrow.compute's generated functions are only partially in the type stubs; bind it as
# Any so columnar ops type-check without a per-call ignore on every pc.* access.
pc: Any = _compute

# Default dataset version subdirectory written by the split pipeline
# (``<output_root>/lance/v0``). Exposed as a knob because the pipeline versions
# the schema under this path.
DEFAULT_LANCE_VERSION = "v0"

# The exact set of source clip-metadata columns the measure phase reads. Loading
# projects to these (smaller read/materialize + a fast schema check). ``windows``
# carries the nested caption maps; the rest are clip-level scalars. Widen this set
# whenever a new clip measurement starts reading another source column.
REQUIRED_CLIP_COLUMNS: frozenset[str] = frozenset(
    {
        "clip_uuid",
        "video_uuid",
        "aesthetic_score",
        "motion_score",
        "valid",
        "has_caption",
        "rejection_stage",
        "windows",
    }
)


def _lance_dataset_uri(output_root: str, *, version: str = DEFAULT_LANCE_VERSION) -> str:
    """Return the URI of an output's clip-metadata Lance dataset.

    ``str(get_full_path(...))`` yields a backend-usable URI (``s3://...`` /
    local path) the same way the v1 loaders build their paths.
    """
    return str(storage_utils.get_full_path(output_root, "lance", version))


def load_clip_metadata(
    output_root: str,
    *,
    profile_name: str,
    version: str = DEFAULT_LANCE_VERSION,
) -> pa.Table:
    """Load ``<output_root>/lance/<version>``, projected to :data:`REQUIRED_CLIP_COLUMNS`.

    The projection both trims the read/materialization and validates the schema:
    a missing required column raises ``ValueError`` naming the dataset and the
    missing columns. Raises whatever ``lance`` raises on a missing/unreadable
    dataset -- the CLI surfaces that as a load failure rather than a partial
    comparison.
    """
    uri = _lance_dataset_uri(output_root, version=version)
    storage_options = storage_utils.get_lance_storage_options(uri, profile_name=profile_name)
    dataset = lance.dataset(uri, storage_options=storage_options)
    missing = REQUIRED_CLIP_COLUMNS - set(dataset.schema.names)
    if missing:
        error_msg = f"clip-metadata dataset at {uri} is missing required columns: {sorted(missing)}"
        raise ValueError(error_msg)
    table = dataset.to_table(columns=sorted(REQUIRED_CLIP_COLUMNS))
    _require_unique_clip_uuid(table, uri)
    logger.info("Loaded {} clip rows from {}", table.num_rows, uri)
    return table


def _require_unique_clip_uuid(table: pa.Table, uri: str) -> None:
    """Fail loud at the load boundary if the unique-``clip_uuid`` producer contract is violated.

    ``clip_uuid`` is the per-clip primary key: one row per clip. A duplicate would
    otherwise be handled silently and *inconsistently* downstream -- the clip-diff join
    fans out on it while window alignment (:func:`index_by_clip` / ``clip_row_index``)
    collapses last-wins -- yielding a garbled comparison with no error. Validating the
    invariant once here keeps it auditable and surfaces a violation as the upstream bug
    it is, rather than silent noise in the report. ``count_distinct`` ignores nulls, so a
    null ``clip_uuid`` (also a contract violation) trips this too.
    """
    distinct = pc.count_distinct(table.column("clip_uuid")).as_py()
    if distinct != table.num_rows:
        error_msg = (
            f"clip-metadata dataset at {uri} violates the unique-clip_uuid contract: "
            f"{table.num_rows} rows but {distinct} distinct clip_uuids"
        )
        raise ValueError(error_msg)


def index_by_clip(table: pa.Table) -> dict[str, dict[str, Any]]:
    """Materialize a clip-metadata table into a ``clip_uuid -> row`` dict.

    The source table is one row per clip -- a clip's windows are a nested
    ``list<struct>`` column on that single row, so multi-window clips are *not*
    duplicates. Rows with a null ``clip_uuid`` are skipped (the schema marks it
    non-null, so this is defensive). ``clip_uuid`` is unique by producer contract,
    enforced at load by :func:`_require_unique_clip_uuid`; this collapses last-wins and
    logs as a backstop, but the warning does not fire in normal use because the source
    table passed that check and the production alignment path feeds already-unique rows.
    """
    by_clip: dict[str, dict[str, Any]] = {}
    for row in table.to_pylist():
        clip_uuid = row.get("clip_uuid")
        if clip_uuid is None:
            continue
        by_clip[str(clip_uuid)] = row
    if len(by_clip) != table.num_rows:
        logger.warning(
            "clip-metadata table has {} rows but {} distinct clip_uuids -- duplicates collapsed",
            table.num_rows,
            len(by_clip),
        )
    return by_clip
