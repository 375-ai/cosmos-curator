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

"""Shared helpers for config-backed pipeline CLI entrypoints."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

PipelineName = Literal["video_split", "caption_judge", "robot_action_split"]
PIPELINE_NAMES = frozenset({"video_split", "caption_judge", "robot_action_split"})


@dataclass(frozen=True)
class PipelineCliSpec:
    """CLI operations for one config-backed pipeline kind."""

    template_yaml: Callable[[], str]
    template_payload: Callable[[], dict[str, Any]]
    validate: Callable[[Path, Sequence[str]], dict[str, object]]
    render: Callable[[Path, Sequence[str]], str]
    schema_json: Callable[[], str]


def load_pipeline_kind(config: Path) -> PipelineName:
    """Load and validate the ``kind`` field from a pipeline config file."""
    with config.open(encoding="utf-8") as f:
        if config.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(f)
        else:
            import json  # noqa: PLC0415

            data = json.load(f)
    if not isinstance(data, dict):
        msg = f"Config must be a YAML/JSON mapping, got {type(data).__name__}"
        raise TypeError(msg)
    # robot-action-split uses a hyphenated kind; normalize to underscore for lookup.
    raw_kind = data.get("kind")
    kind = raw_kind.replace("-", "_") if isinstance(raw_kind, str) else raw_kind
    if not isinstance(kind, str) or kind not in PIPELINE_NAMES:
        valid = ", ".join(sorted(PIPELINE_NAMES))
        msg = f"Config must contain a valid 'kind' key (got: {raw_kind!r}). Valid pipeline kinds: {valid}"
        raise ValueError(msg)
    return cast("PipelineName", kind)


def pipeline_cli_spec(kind: PipelineName) -> PipelineCliSpec:
    """Return the CLI spec for a supported pipeline kind."""
    return _PIPELINE_CLI_SPECS[kind]


def _video_split_template_yaml() -> str:
    from cosmos_curator.pipelines.ray_data.video_split.config import (  # noqa: PLC0415
        user_config_to_yaml,
        video_split_config_template,
    )

    return user_config_to_yaml(video_split_config_template())


def _video_split_template_payload() -> dict[str, Any]:
    from cosmos_curator.pipelines.ray_data.video_split.config import video_split_template_payload  # noqa: PLC0415

    return video_split_template_payload()


def _video_split_validate(config: Path, overrides: Sequence[str]) -> dict[str, object]:
    from cosmos_curator.pipelines.ray_data.video_split.config import resolve_video_split_config  # noqa: PLC0415

    resolution = resolve_video_split_config(config, overrides=overrides)
    return {"ok": True, "selected_presets": resolution.selected_presets}


def _video_split_render(config: Path, overrides: Sequence[str]) -> str:
    from cosmos_curator.pipelines.ray_data.video_split.config import (  # noqa: PLC0415
        resolve_video_split_config,
        resolved_config_to_json,
    )

    resolution = resolve_video_split_config(config, overrides=overrides)
    return resolved_config_to_json(resolution.config)


def _video_split_schema_json() -> str:
    from cosmos_curator.pipelines.ray_data.video_split.config import user_video_split_schema_json  # noqa: PLC0415

    return user_video_split_schema_json()


def _caption_judge_template_yaml() -> str:
    from cosmos_curator.pipelines.ray_data.caption_judge.config import (  # noqa: PLC0415
        caption_judge_config_template,
        caption_judge_config_to_yaml,
    )

    return caption_judge_config_to_yaml(caption_judge_config_template())


def _caption_judge_template_payload() -> dict[str, Any]:
    from cosmos_curator.pipelines.ray_data.caption_judge.config import caption_judge_template_payload  # noqa: PLC0415

    return caption_judge_template_payload()


def _caption_judge_validate(config: Path, overrides: Sequence[str]) -> dict[str, object]:
    from cosmos_curator.pipelines.ray_data.caption_judge.config import resolve_caption_judge_config  # noqa: PLC0415

    resolve_caption_judge_config(config, overrides=overrides)
    return {"ok": True}


def _caption_judge_render(config: Path, overrides: Sequence[str]) -> str:
    from cosmos_curator.pipelines.ray_data.caption_judge.config import (  # noqa: PLC0415
        caption_judge_config_to_json,
        resolve_caption_judge_config,
    )

    resolved_config = resolve_caption_judge_config(config, overrides=overrides)
    return caption_judge_config_to_json(resolved_config)


def _caption_judge_schema_json() -> str:
    from cosmos_curator.pipelines.ray_data.caption_judge.config import user_caption_judge_schema_json  # noqa: PLC0415

    return user_caption_judge_schema_json()


def _robot_action_split_template_yaml() -> str:
    from cosmos_curator.next.recipes.robot_action_split.pipeline_kind import ROBOT_ACTION_SPLIT_KIND  # noqa: PLC0415

    return ROBOT_ACTION_SPLIT_KIND.template_yaml()


def _robot_action_split_template_payload() -> dict[str, Any]:
    from cosmos_curator.next.recipes.robot_action_split.pipeline_kind import ROBOT_ACTION_SPLIT_KIND  # noqa: PLC0415

    return ROBOT_ACTION_SPLIT_KIND.template_payload()


def _robot_action_split_validate(config: Path, overrides: Sequence[str]) -> dict[str, object]:
    from cosmos_curator.next.recipes.robot_action_split.pipeline_kind import ROBOT_ACTION_SPLIT_KIND  # noqa: PLC0415

    return ROBOT_ACTION_SPLIT_KIND.validate(config, overrides)


def _robot_action_split_render(config: Path, overrides: Sequence[str]) -> str:
    from cosmos_curator.next.recipes.robot_action_split.pipeline_kind import ROBOT_ACTION_SPLIT_KIND  # noqa: PLC0415

    return ROBOT_ACTION_SPLIT_KIND.render(config, overrides)


def _robot_action_split_schema_json() -> str:
    from cosmos_curator.next.recipes.robot_action_split.pipeline_kind import ROBOT_ACTION_SPLIT_KIND  # noqa: PLC0415

    return ROBOT_ACTION_SPLIT_KIND.schema_json()


_PIPELINE_CLI_SPECS: dict[PipelineName, PipelineCliSpec] = {
    "video_split": PipelineCliSpec(
        template_yaml=_video_split_template_yaml,
        template_payload=_video_split_template_payload,
        validate=_video_split_validate,
        render=_video_split_render,
        schema_json=_video_split_schema_json,
    ),
    "caption_judge": PipelineCliSpec(
        template_yaml=_caption_judge_template_yaml,
        template_payload=_caption_judge_template_payload,
        validate=_caption_judge_validate,
        render=_caption_judge_render,
        schema_json=_caption_judge_schema_json,
    ),
    "robot_action_split": PipelineCliSpec(
        template_yaml=_robot_action_split_template_yaml,
        template_payload=_robot_action_split_template_payload,
        validate=_robot_action_split_validate,
        render=_robot_action_split_render,
        schema_json=_robot_action_split_schema_json,
    ),
}
