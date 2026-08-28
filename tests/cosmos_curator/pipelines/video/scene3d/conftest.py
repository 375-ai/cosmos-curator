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
"""Fixtures for the 3D reconstruction tests."""

import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.pipelines.video.scene3d.calibration import Calib
from tests.cosmos_curator.pipelines.video.scene3d.scene_fixtures import make_calib, render_ground_depth


@pytest.fixture
def calib() -> Calib:
    """Return a camera 8 m up, tilted 20 degrees down."""
    return make_calib()


@pytest.fixture
def ground_depth(calib: Calib) -> npt.NDArray[np.float32]:
    """Return the exact ground-plane depth map for the ``calib`` fixture."""
    return render_ground_depth(calib)
