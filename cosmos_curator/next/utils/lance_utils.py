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

"""Shared Lance utilities for Curator Next recipes."""

import lance


def open_dataset(uri: str, *, storage_options: dict[str, str] | None = None) -> lance.LanceDataset | None:
    """Open a Lance dataset, returning ``None`` only when the target does not exist."""
    try:
        return lance.dataset(uri, storage_options=storage_options)
    except (OSError, ValueError) as exc:
        message = str(exc).lower()
        if "not found" in message or "does not exist" in message:
            return None
        raise
