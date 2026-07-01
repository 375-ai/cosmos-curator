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
"""Measure phase: produce the clip + window measurement tables for two outputs.

* :mod:`core` -- engine-agnostic measurement primitives (no Ray).
* :mod:`ray` -- the Ray Data driver (:func:`ray.run`) that fans the caption work
  over GPU actors; the package's measure entry point.
* :mod:`schema` -- Arrow schemas for the clip / window measurement tables.

``core`` and ``schema`` stay Ray-free so the measurement logic can be unit-tested
without a GPU or a Ray runtime.
"""
