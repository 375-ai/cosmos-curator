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
"""Tests for the Depth Anything V2 metric-depth model interface.

The CPU tests cover variant selection and the weight-download contract; the
GPU-marked test needs the real checkpoint staged in the local workspace.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from cosmos_curator.core.utils.model import model_utils
from cosmos_curator.models.depth_anything_v2 import (
    DEPTH_VARIANTS,
    DepthAnythingV2Model,
    model_id_for_variant,
    weights_name_for_variant,
)


def test_only_the_selected_variant_is_declared_for_download() -> None:
    """Enabling one variant must not drag the other onto every node."""
    outdoor = DepthAnythingV2Model("outdoor")
    indoor = DepthAnythingV2Model("indoor")

    assert outdoor.model_id_names == ["depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"]
    assert indoor.model_id_names == ["depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"]
    assert outdoor.model_id_names != indoor.model_id_names


def test_default_variant_is_outdoor() -> None:
    """Street, driving and surveillance footage is the common case."""
    assert DepthAnythingV2Model().variant == "outdoor"


def test_runs_in_the_default_pixi_env() -> None:
    """No dedicated environment: the model rides on the existing torch/transformers stack."""
    assert DepthAnythingV2Model().conda_env_name == "default"


def test_unknown_variant_fails_at_construction() -> None:
    """A typo should fail on the driver, not inside a remote actor."""
    with pytest.raises(ValueError, match="Unknown Depth Anything V2 variant"):
        DepthAnythingV2Model("outdoors")  # type: ignore[arg-type]


@pytest.mark.parametrize("variant", DEPTH_VARIANTS)
def test_every_variant_is_registered_for_download(variant: str) -> None:
    """The short name in all_models.json must resolve to the same repo id.

    Read straight from the repo config rather than ``get_all_models()``, which
    resolves the container path and so only works inside the image.
    """
    config_path = Path(__file__).resolve().parents[3] / "cosmos_curator" / "configs" / "all_models.json"
    all_models = json.loads(config_path.read_text())

    weights_name = weights_name_for_variant(variant)  # type: ignore[arg-type]
    assert weights_name in all_models
    assert all_models[weights_name]["model_id"] == model_id_for_variant(variant)  # type: ignore[arg-type]
    # A pinned revision keeps reconstructions reproducible across runs.
    assert all_models[weights_name]["version"]


@pytest.mark.env("default")
def test_metric_depth_inference_returns_metres() -> None:
    """Ensure the model loads real weights and predicts plausible metric depth."""
    model = DepthAnythingV2Model("outdoor")
    weights_dir = model_utils.get_local_dir_for_weights_name(model.model_id_names[0])
    assert weights_dir.exists(), f"Expected pre-downloaded weights at {weights_dir}"

    model.setup()

    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
    depth = model.infer(image)

    # The prediction comes back at the input resolution, in metres, within the
    # outdoor checkpoint's 80 m training ceiling.
    assert depth.shape == (360, 640)
    assert depth.dtype == np.float32
    assert np.isfinite(depth).all()
    assert depth.min() > 0.0
    assert depth.max() < 200.0

    batched = model.infer_batch([image, image])
    assert len(batched) == 2
    np.testing.assert_allclose(batched[0], batched[1])
