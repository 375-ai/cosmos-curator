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
"""Split-output comparison over the columnar Lance clip-metadata dataset.

Two phases, combined via :mod:`cli`:

* **measure** (:mod:`measure` -- the ``measure.ray`` driver over ``measure.core``
  primitives, persisted by :mod:`store`): diff every clip and caption window of two
  outputs into durable ``clip.lance`` / ``window.lance`` + a provenance manifest.
  The one expensive step -- caption-similarity embedding -- fans out over a Ray
  Data GPU actor pool; no thresholds are applied.
* **eval** (:mod:`eval`): apply tolerance/threshold policy to the measurement
  tables -> issues + per-clip verdicts. Consumes the tables, never the source, so
  it runs right after measure (locality) or standalone for fast re-evaluation.

Summary-level (summary.json) comparison is layered in via :mod:`summary`.

See ``docs/curator/design/split-comparison.md`` for the module architecture, the
Ray fan-out, and the Arrow facts that shape the design.
"""
