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

"""Import regression guard for the vendored SeedVR2 windowed inference module.

torchvision 0.26 removed the video decoding/encoding utilities (``VideoReader``,
``write_video`` and friends -- moved to TorchCodec) while keeping the image
decoders such as ``read_image``. The ``seedvr`` pixi environment pins
``torchvision==0.26.0``, so any lingering ``from torchvision.io import
VideoReader``/``write_video`` in ``inference_seedvr2_window`` breaks the whole
SeedVR2 super-resolution stage at *import time* -- before a single frame is
processed. This test imports the module in the ``seedvr`` environment and fails
loudly if that regression ever comes back.
"""

import importlib
from types import ModuleType

import pytest

_MODULE_PATH = "cosmos_curator.pipelines.video.super_resolution.inference_seedvr2_window"


def _import_or_skip() -> ModuleType:
    """Import the vendored module, skipping when the SeedVR source repo is unavailable."""
    try:
        return importlib.import_module(_MODULE_PATH)
    except ImportError as exc:
        top_level = (getattr(exc, "name", None) or "").split(".")[0]
        # The module self-bootstraps the SeedVR source repo (``common`` / ``data``)
        # from ``SEEDVR_ROOT``. When that source tree is not provisioned (e.g. a slim
        # image without SEEDVR_ROOT), the import fails on ``common``/``data`` *before*
        # ever reaching the torchvision import -- that is an environment gap, not the
        # torchvision regression this test guards, so skip rather than fail.
        if top_level in {"common", "data"}:
            pytest.skip(f"SeedVR source repo unavailable ({exc}); torchvision import not reached")
        raise


@pytest.mark.env("seedvr")
def test_inference_seedvr2_window_imports() -> None:
    """The vendored module must import cleanly under torchvision 0.26 (PyAV I/O)."""
    module = _import_or_skip()

    # Public entry points the pipeline adapter and the standalone CLI rely on.
    # Intentionally only the public surface -- asserting private helpers would
    # couple this guard to the module's internal decomposition.
    for symbol in ("configure_runner", "generation_step", "cut_videos"):
        assert hasattr(module, symbol), f"missing expected symbol: {symbol}"

    # The removed torchvision video I/O symbols must not have crept back in.
    import torchvision.io as tvio  # noqa: PLC0415

    assert hasattr(tvio, "read_image"), "torchvision.io.read_image should still be available"
    assert not hasattr(tvio, "VideoReader"), "torchvision 0.26 removed VideoReader; do not import it"
    assert not hasattr(tvio, "write_video"), "torchvision 0.26 removed write_video; do not import it"


@pytest.mark.env("seedvr")
def test_torchvision_io_video_shim_enables_read_video() -> None:
    """The shim must make ``from torchvision.io.video import read_video`` resolve.

    The *upstream* SeedVR variant scripts (``inference_seedvr2_*.py``) still do
    ``from torchvision.io.video import read_video`` at import time. torchvision 0.26
    dropped that module, so ``_resolve_variant_module`` installs a PyAV-backed shim
    before importing them. Without this the SR stage dies at worker setup with
    ``ModuleNotFoundError: No module named 'torchvision.io.video'`` -- exercised
    end-to-end only in the (weight-dependent) CI SR scenario, so guard it cheaply here.
    """
    module = _import_or_skip()

    module._install_torchvision_video_shim()

    video_mod = importlib.import_module("torchvision.io.video")
    assert callable(getattr(video_mod, "read_video", None)), "shim must expose a callable read_video"
