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
"""Tests for SplitComparisonConfig (pydantic v2) and its nested policy types."""

import pytest
from pydantic import ValidationError

from cosmos_curator.pipelines.video.split_comparison.config import (
    CaptionPolicy,
    ScoreTolerance,
    SplitComparisonConfig,
    SummaryPolicy,
)


def _config(**overrides: object) -> SplitComparisonConfig:
    """Build a config with placeholder targets so tests only spell out the override under test."""
    return SplitComparisonConfig(output_a="/a", output_b="/b", **overrides)  # type: ignore[arg-type]


# --- construction + defaults --------------------------------------------------------


def test_default_config_constructs_with_expected_policy_defaults() -> None:
    """Construction fills every nested policy with documented defaults."""
    config = _config()

    assert config.profile_name == "default"
    assert config.compare_captions is True
    assert config.caption.model_id == "BAAI/bge-small-en-v1.5"
    assert config.caption.min_similarity == 0.85
    assert config.caption.encode_batch_size == 128
    assert config.aesthetic.abs_tolerance == ScoreTolerance().abs_tolerance


def test_config_overrides_propagate_to_nested_policies() -> None:
    """Custom nested policies replace defaults without affecting other fields."""
    config = _config(
        caption=CaptionPolicy(model_id="intfloat/e5-small-v2", min_similarity=0.9),
        compare_captions=False,
    )

    assert config.caption.model_id == "intfloat/e5-small-v2"
    assert config.caption.min_similarity == 0.9
    assert config.compare_captions is False
    assert config.aesthetic.abs_tolerance == ScoreTolerance().abs_tolerance


# --- frozen=True (top-level and nested) ---------------------------------------------


def test_config_is_frozen() -> None:
    """Top-level config is immutable; rebinding raises ValidationError."""
    config = _config()
    with pytest.raises(ValidationError):
        config.profile_name = "other"  # type: ignore[misc]


def test_caption_policy_is_frozen() -> None:
    """Nested policies are frozen too; nested mutation is the classic foot-gun this prevents."""
    policy = CaptionPolicy()
    with pytest.raises(ValidationError):
        policy.model_id = "other"  # type: ignore[misc]


# --- strict=True (no type coercion) --------------------------------------------------


def test_strict_mode_rejects_string_for_int_field() -> None:
    """strict=True: "5" must NOT silently coerce to 5 (would mask config-file typos)."""
    with pytest.raises(ValidationError):
        CaptionPolicy(encode_batch_size="128")  # type: ignore[arg-type]


def test_strict_mode_rejects_string_for_bool_field() -> None:
    """strict=True: "true" must NOT silently coerce to True."""
    with pytest.raises(ValidationError):
        _config(compare_captions="true")  # type: ignore[arg-type]


# --- extra="forbid" (unknown fields rejected) ---------------------------------------


def test_unknown_field_rejected() -> None:
    """A typo in a field name fails at construction instead of being silently ignored."""
    with pytest.raises(ValidationError):
        _config(profil_name="x")  # type: ignore[call-arg] -- intentional typo


def test_unknown_field_in_nested_model_rejected() -> None:
    """Same protection on nested policy models."""
    with pytest.raises(ValidationError):
        ScoreTolerance(abs_tolarence=0.5)  # type: ignore[call-arg] -- intentional typo


# --- validators (ge=, le=, min_length=) ---------------------------------------------


def test_negative_tolerance_rejected() -> None:
    """Negative tolerance is nonsense; pydantic enforces ge=0.0."""
    with pytest.raises(ValidationError):
        ScoreTolerance(abs_tolerance=-1.0)


def test_min_similarity_outside_zero_to_one_rejected() -> None:
    """min_similarity is a cosine similarity; out-of-[0,1] is rejected."""
    with pytest.raises(ValidationError):
        CaptionPolicy(min_similarity=1.5)
    with pytest.raises(ValidationError):
        CaptionPolicy(min_similarity=-0.1)


def test_zero_encode_batch_size_rejected() -> None:
    """encode_batch_size must be ge=1; zero is not a usable batch."""
    with pytest.raises(ValidationError):
        CaptionPolicy(encode_batch_size=0)


def test_empty_string_for_required_min_length_field_rejected() -> None:
    """profile_name has min_length=1; empty string fails validation."""
    with pytest.raises(ValidationError):
        _config(profile_name="")


def test_empty_output_path_rejected() -> None:
    """output_a / output_b are required and min_length=1; empty string is not a path."""
    with pytest.raises(ValidationError):
        SplitComparisonConfig(output_a="", output_b="/b")


def test_missing_output_a_rejected() -> None:
    """output_a is required (no default); construction without it fails."""
    with pytest.raises(ValidationError):
        SplitComparisonConfig(output_b="/b")  # type: ignore[call-arg]


# --- JSON round-trip ----------------------------------------------------------------


def test_default_config_round_trips_through_json() -> None:
    """Dump to JSON, load back, equal -- the model.validate_json contract."""
    original = _config(
        compare_captions=False,
        aesthetic=ScoreTolerance(abs_tolerance=0.01, rel_tolerance=0.02),
        summary=SummaryPolicy(token_count_abs_tolerance=5.0),
    )

    payload = original.model_dump_json()
    reloaded = SplitComparisonConfig.model_validate_json(payload)

    assert reloaded == original
