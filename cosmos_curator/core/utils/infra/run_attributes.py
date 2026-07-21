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

"""Run-level attributes for OTLP traces and metrics (local / Slurm).

Values are read from the process environment and standard OS identity
helpers.  Keys mirror the Slurm Prometheus service-discovery labels
where applicable.

For driver-pushed metrics, these are copied onto every exported data
point as Prom labels (Mimir/Grafana queryable).  Traces still attach
them to the OTel resource.
"""

import os
import pwd
import socket
from collections.abc import Sequence
from json import JSONDecodeError, loads

from cosmos_curator.core.utils.infra.tracing import short_hostname

# When set to ``0`` / ``false`` / ``no``, :func:`collect_run_attributes`
# returns an empty dict.  Propagated to Ray workers for tracing.
ENV_OTLP_RUN_ATTRIBUTES = "COSMOS_CURATOR_OTLP_RUN_ATTRIBUTES"
# Optional JSON mapping of output attribute -> literal value.
# Values are attached directly as labels/attributes.
ENV_OTLP_RUN_ATTRIBUTES_VALUES = "COSMOS_CURATOR_OTLP_RUN_ATTRIBUTES_VALUES"
# Optional JSON mapping of output attribute -> env var name(s).
# Values point to env var names (with optional fallback lists).
ENV_OTLP_RUN_ATTRIBUTES_MAP = "COSMOS_CURATOR_OTLP_RUN_ATTRIBUTES_MAP"


def otlp_run_attributes_enabled() -> bool:
    """Return whether run attributes should be attached to OTLP exports."""
    raw = os.environ.get(ENV_OTLP_RUN_ATTRIBUTES, "1").strip().lower()
    return raw not in ("0", "false", "no")


def set_otlp_run_attributes_enabled(*, enabled: bool) -> None:
    """Publish the run-attribute toggle for this process and Ray workers."""
    if enabled:
        os.environ.pop(ENV_OTLP_RUN_ATTRIBUTES, None)
    else:
        os.environ[ENV_OTLP_RUN_ATTRIBUTES] = "0"


def short_host_label(host: str) -> str:
    """Return the host label (segment before the first dot)."""
    return host.split(".", maxsplit=1)[0] if host else host


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _login_user() -> str | None:
    for name in ("USER", "LOGNAME"):
        value = os.environ.get(name)
        if value:
            return value
    try:
        return pwd.getpwuid(os.getuid()).pw_name
    except (KeyError, OSError):
        return None


def _apply_extra_attributes_from_map(attrs: dict[str, str]) -> None:
    """Merge caller-defined attributes from ``ENV_OTLP_RUN_ATTRIBUTES_MAP``.

    Map value must be JSON object where each key is output attribute name and each
    value is either an env var name (str) or fallback env var list (list[str]).
    """
    raw = os.environ.get(ENV_OTLP_RUN_ATTRIBUTES_MAP, "").strip()
    if not raw:
        return
    try:
        mapping = loads(raw)
    except JSONDecodeError:
        return
    if not isinstance(mapping, dict):
        return

    for attr_key, env_spec in mapping.items():
        if not isinstance(attr_key, str) or not attr_key:
            continue
        env_names: list[str]
        if isinstance(env_spec, str):
            env_names = [env_spec]
        elif isinstance(env_spec, list) and all(isinstance(v, str) for v in env_spec):
            env_names = [v for v in env_spec if v]
        else:
            continue
        if not env_names:
            continue
        value = _first_env(*env_names)
        if value:
            attrs[attr_key] = value


def _apply_extra_attributes_from_values(attrs: dict[str, str]) -> None:
    """Merge caller-defined literal attributes from ``ENV_OTLP_RUN_ATTRIBUTES_VALUES``."""
    raw = os.environ.get(ENV_OTLP_RUN_ATTRIBUTES_VALUES, "").strip()
    if not raw:
        return
    try:
        values = loads(raw)
    except JSONDecodeError:
        return
    if not isinstance(values, dict):
        return
    attrs.update(
        {
            attr_key: value
            for attr_key, value in values.items()
            if isinstance(attr_key, str) and attr_key and isinstance(value, str) and value
        }
    )


def collect_run_attributes() -> dict[str, str]:
    """Collect run-level resource attributes from the environment.

    Omits empty values.  Callers gate export via
    :attr:`MetricsPushConfig.include_run_attributes`,
    :attr:`TracingConfig.include_run_attributes`, or
    :func:`otlp_run_attributes_enabled` on workers.
    """
    if not otlp_run_attributes_enabled():
        return {}

    attrs: dict[str, str] = {}

    def _set(key: str, *env_names: str) -> None:
        value = _first_env(*env_names)
        if value:
            attrs[key] = value

    _set("slurm_job_id", "SLURM_JOB_ID", "SLURM_JOBID")
    _set("slurm_job_user", "SLURM_JOB_USER")
    _set("slurm_job_name", "SLURM_JOB_NAME")
    _set("slurm_restart_count", "SLURM_RESTART_COUNT")
    _set("slurm_array_task_id", "SLURM_ARRAY_TASK_ID")
    _set("slurm_partition", "SLURM_JOB_PARTITION")
    _set("slurm_account", "SLURM_JOB_ACCOUNT")
    _set("slurm_cluster", "SLURM_CLUSTER_NAME")
    _set("slurm_num_nodes", "SLURM_JOB_NUM_NODES", "SLURM_NNODES")
    _set("slurm_head_node", "PRIMARY_NODE_HOSTNAME", "HEAD_NODE_ADDR")
    _set("slurm_node_name", "SLURMD_NODENAME")
    _set("ray_job_id", "RAY_JOB_ID")

    if "slurm_job_user" not in attrs:
        user = _login_user()
        if user:
            attrs["user.name"] = user

    full_host = socket.gethostname()
    short = short_hostname()
    if full_host and full_host != short:
        attrs["host.full.name"] = full_host

    _apply_extra_attributes_from_map(attrs)
    _apply_extra_attributes_from_values(attrs)
    return attrs


__all__: Sequence[str] = (
    "ENV_OTLP_RUN_ATTRIBUTES",
    "ENV_OTLP_RUN_ATTRIBUTES_MAP",
    "ENV_OTLP_RUN_ATTRIBUTES_VALUES",
    "collect_run_attributes",
    "otlp_run_attributes_enabled",
    "set_otlp_run_attributes_enabled",
    "short_host_label",
)
