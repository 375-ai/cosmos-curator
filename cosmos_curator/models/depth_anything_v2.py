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

"""Depth Anything V2 (metric) monocular depth model.

The *metric* checkpoints predict absolute depth in metres (optical-Z) rather than
the affine-invariant relative inverse-depth the base V2 checkpoints emit. That
matters for :mod:`cosmos_curator.pipelines.video.scene3d`: with metric depth the
scene needs no externally supplied ground plane to recover scale, which is what
makes the 3D reconstruction work on an arbitrary input video.

Two variants ship, differing only in training domain and depth ceiling:

- ``outdoor`` (``max_depth`` 80 m) — driving / surveillance / street scenes
- ``indoor`` (``max_depth`` 20 m) — rooms, corridors, close-range scenes

Weights are float32: the checkpoint is only ~1.3 GB and the scene3d stage runs a
single forward pass per clip, so there is nothing to gain from a reduced-precision
load and metric metres keep their full mantissa.
"""

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import numpy.typing as npt
import torch
from loguru import logger

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.utils.model import model_utils, pixi_utils

if TYPE_CHECKING:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

if pixi_utils.is_running_in_env("default"):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

DepthVariant = Literal["outdoor", "indoor"]
DEPTH_VARIANTS: tuple[DepthVariant, ...] = ("outdoor", "indoor")

# Short name in ``configs/all_models.json`` -> HF repo id.
_MODEL_IDS: dict[DepthVariant, str] = {
    "outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    "indoor": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
}
_WEIGHTS_NAMES: dict[DepthVariant, str] = {
    "outdoor": "depth_anything_v2_metric_outdoor",
    "indoor": "depth_anything_v2_metric_indoor",
}


def model_id_for_variant(variant: DepthVariant) -> str:
    """Return the HF repo id backing ``variant``."""
    return _MODEL_IDS[variant]


def weights_name_for_variant(variant: DepthVariant) -> str:
    """Return the ``cosmos-curator model download --models`` short name for ``variant``."""
    return _WEIGHTS_NAMES[variant]


class DepthAnythingV2Model(ModelInterface):
    """Depth Anything V2 metric-depth wrapper.

    Only the selected variant is declared in :attr:`model_id_names`, so enabling
    the outdoor model never drags the indoor weights onto every node.
    """

    def __init__(self, variant: DepthVariant = "outdoor") -> None:
        """Select the metric checkpoint to load.

        Args:
            variant: ``"outdoor"`` (80 m ceiling) or ``"indoor"`` (20 m ceiling).

        """
        if variant not in _MODEL_IDS:
            msg = f"Unknown Depth Anything V2 variant {variant!r}; expected one of {DEPTH_VARIANTS}"
            raise ValueError(msg)
        self._variant: DepthVariant = variant
        self._model_id = model_id_for_variant(variant)

    @property
    def variant(self) -> DepthVariant:
        """Return the configured variant name."""
        return self._variant

    @property
    def model_id(self) -> str:
        """Return the HF repo id of the configured variant."""
        return self._model_id

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name.

        Returns:
            The conda environment name.

        """
        return "default"

    @property
    def model_id_names(self) -> list[str]:
        """Get the model ID names.

        Returns:
            A list holding only the selected variant's HF repo id.

        """
        return [self._model_id]

    def setup(self) -> None:
        """Load the image processor and metric-depth model onto the GPU."""
        if not torch.cuda.is_available():
            msg = "Depth Anything V2 requires a CUDA-capable GPU but none was found"
            raise RuntimeError(msg)

        logger.info(f"Setting up Depth Anything V2 ({self._variant})")
        model_dir = model_utils.get_local_dir_for_weights_name(self._model_id)
        if not model_dir.exists():
            weights_name = weights_name_for_variant(self._variant)
            msg = (
                f"Depth Anything V2 weights not found at {model_dir}. "
                f"Download via: cosmos-curator model download --models {weights_name}"
            )
            raise FileNotFoundError(msg)

        mem_before = torch.cuda.memory_allocated()

        self.processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
        model_cls = cast("Any", AutoModelForDepthEstimation)
        self.model = model_cls.from_pretrained(
            model_dir,
            torch_dtype=torch.float32,
            local_files_only=True,
        ).cuda()
        self.model.eval()

        mem_after = torch.cuda.memory_allocated()
        footprint_gb = (mem_after - mem_before) / 1024**3
        logger.info(f"Depth Anything V2 load footprint: {footprint_gb:.3f} GB ({mem_after - mem_before:,} bytes)")

    def infer(self, image_rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        """Predict metric depth (metres, optical-Z) for one RGB image.

        Args:
            image_rgb: ``(H, W, 3)`` uint8 RGB image.

        Returns:
            ``(H, W)`` float32 depth in metres, resampled back to the input size.

        """
        return self.infer_batch([image_rgb])[0]

    def infer_batch(self, images_rgb: list[npt.NDArray[np.uint8]]) -> list[npt.NDArray[np.float32]]:
        """Predict metric depth for a batch of equally sized RGB images.

        The processor resizes to a multiple of 14 with the aspect ratio kept, so
        ``post_process_depth_estimation`` is used to resample each prediction back
        to its original resolution.

        Args:
            images_rgb: list of ``(H, W, 3)`` uint8 RGB images.

        Returns:
            One ``(H, W)`` float32 metric-depth map per input image.

        """
        if not images_rgb:
            return []
        inputs = self.processor(images=images_rgb, return_tensors="pt")
        inputs = {key: value.cuda() for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        target_sizes = [(int(image.shape[0]), int(image.shape[1])) for image in images_rgb]
        processed = self.processor.post_process_depth_estimation(outputs, target_sizes=target_sizes)
        return [entry["predicted_depth"].detach().float().cpu().numpy().astype(np.float32) for entry in processed]
