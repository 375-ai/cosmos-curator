# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for nanosecond timing helpers used by the clip metadata Lance schema."""

import pytest

from cosmos_curator.pipelines.video.utils.ns_timing import (
    NS_PER_SECOND,
    frame_pts_bounds_ns,
    seconds_to_ns,
)


class TestSecondsToNs:
    """Tests for ``seconds_to_ns``."""

    def test_whole_and_fractional_seconds(self) -> None:
        """Whole and fractional seconds convert to integer nanoseconds."""
        assert seconds_to_ns(1.0) == NS_PER_SECOND
        assert seconds_to_ns(1.5) == 1_500_000_000
        assert seconds_to_ns(0.0) == 0

    def test_rounds_to_nearest_nanosecond(self) -> None:
        """Sub-nanosecond values round to the nearest whole nanosecond."""
        assert seconds_to_ns(3.6e-9) == 4
        assert seconds_to_ns(1.0000000004) == 1_000_000_000
        assert seconds_to_ns(1.0000000006) == 1_000_000_001

    def test_returns_int(self) -> None:
        """The result is a Python int, not a float."""
        assert isinstance(seconds_to_ns(2.0), int)


class TestFramePtsBoundsNs:
    """Tests for ``frame_pts_bounds_ns`` (per-frame PTS in integer nanoseconds)."""

    def test_absolute_source_bounds(self) -> None:
        """Non-relative bounds keep the absolute source timeline."""
        pts_ns = [10_000_000_000, 10_500_000_000, 11_000_000_000]
        assert frame_pts_bounds_ns(pts_ns, 0, 2) == (10_000_000_000, 11_000_000_000)

    def test_clip_relative_bounds_subtract_first_pts(self) -> None:
        """relative_to_first yields clip-relative ns even when PTS does not start at 0."""
        pts_ns = [10_000_000_000, 10_500_000_000, 11_000_000_000, 11_500_000_000]
        assert frame_pts_bounds_ns(pts_ns, 1, 3, relative_to_first=True) == (500_000_000, 1_500_000_000)

    def test_clip_relative_when_pts_starts_at_zero(self) -> None:
        """Clip-relative bounds are unchanged when PTS already starts at 0."""
        pts_ns = [0, 500_000_000, 1_000_000_000, 1_500_000_000]
        assert frame_pts_bounds_ns(pts_ns, 0, 2, relative_to_first=True) == (0, 1_000_000_000)

    def test_end_frame_clamped_to_last_index(self) -> None:
        """An end_frame past the last index clamps to the final PTS rather than raising."""
        pts_ns = [0, 500_000_000, 1_000_000_000, 1_500_000_000]
        assert frame_pts_bounds_ns(pts_ns, 0, 99, relative_to_first=True) == (0, 1_500_000_000)

    def test_negative_start_frame_clamped_to_zero(self) -> None:
        """A negative start_frame clamps to the first index."""
        pts_ns = [0, 500_000_000, 1_000_000_000]
        assert frame_pts_bounds_ns(pts_ns, -5, 1, relative_to_first=True) == (0, 500_000_000)

    def test_empty_pts_raises(self) -> None:
        """Empty PTS is a programming error and raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            frame_pts_bounds_ns([], 0, 0)
