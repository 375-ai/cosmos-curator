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
"""Shared runtime helpers for Ray Data-native pipelines."""

from typing import TYPE_CHECKING, cast

import ray

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# Default per-node concurrency cap for IO-bound Ray Data work.
DEFAULT_IO_SLOTS_PER_NODE = 16


def configure_ray_data_progress(*, progress: bool) -> None:
    """Configure Ray Data progress output before creating datasets."""
    ctx = ray.data.DataContext.get_current()
    ctx.enable_progress_bars = progress
    ctx.enable_operator_progress_bars = progress
    ctx.enable_rich_progress_bars = progress
    ctx.print_on_execution_start = progress
    ctx.use_ray_tqdm = False


def ensure_ray_initialized() -> None:
    """Initialize Ray locally when the caller has not already connected to a cluster."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)


def live_ray_node_count() -> int:
    """Return the number of live Ray nodes, floored at one for local callers."""
    nodes = cast("Sequence[Mapping[str, object]]", ray.nodes())  # type: ignore[no-untyped-call]
    return max(1, sum(1 for node in nodes if bool(node.get("Alive"))))


def capped_slots_for_items(*, num_items: int, num_nodes: int, max_per_node: int) -> int:
    """Cap concurrent work by both input item count and cluster node count."""
    if num_items <= 0:
        return 0
    cluster_cap = max_per_node * max(1, num_nodes)
    return min(num_items, cluster_cap)
