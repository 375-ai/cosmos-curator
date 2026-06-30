# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the saturation-aware scheduler opt-in on ``XennaRunner``."""

import argparse

from cosmos_curator.core.interfaces.runner_interface import XennaRunner
from cosmos_xenna.pipelines.private.specs import (
    SaturationAwareConfig,
    SchedulerKind,
)


class TestRunnerInterfaceScheduler:
    """Pin the scheduler-flag round-trip from ``XennaRunner`` to ``StreamingSpecificSpec``."""

    def test_default_runner_uses_fragmentation_based_scheduler(self) -> None:
        """The default ``XennaRunner()`` keeps the legacy scheduler kind."""
        spec = XennaRunner()._streaming_spec
        assert spec.scheduler is SchedulerKind.FRAGMENTATION_BASED

    def test_with_saturation_aware_flips_scheduler_kind(self) -> None:
        """``with_saturation_aware()`` sets ``scheduler=SchedulerKind.SATURATION_AWARE``."""
        spec = XennaRunner.with_saturation_aware()._streaming_spec
        assert spec.scheduler is SchedulerKind.SATURATION_AWARE
        assert isinstance(spec.saturation_aware, SaturationAwareConfig)

    def test_with_streaming_overrides_general_path_still_supports_scheduler(self) -> None:
        """The general ``with_streaming_overrides`` path also accepts ``scheduler=...``."""
        spec = XennaRunner.with_streaming_overrides(
            scheduler=SchedulerKind.SATURATION_AWARE,
        )._streaming_spec
        assert spec.scheduler is SchedulerKind.SATURATION_AWARE


class TestRunnerInterfaceFromArgs:
    """Pin the ``XennaRunner.from_args`` CLI dispatch contract."""

    def test_from_args_none_returns_fragmentation_based_default(self) -> None:
        """``args=None`` keeps the legacy scheduler so callers without argparse stay safe."""
        spec = XennaRunner.from_args(None)._streaming_spec
        assert spec.scheduler is SchedulerKind.FRAGMENTATION_BASED

    def test_from_args_missing_attr_returns_fragmentation_based_default(self) -> None:
        """A bare ``Namespace()`` with no scheduler attribute falls back to FRAGMENTATION_BASED."""
        spec = XennaRunner.from_args(argparse.Namespace())._streaming_spec
        assert spec.scheduler is SchedulerKind.FRAGMENTATION_BASED

    def test_from_args_fragmentation_based_returns_default_runner(self) -> None:
        """The explicit ``FRAGMENTATION_BASED`` value matches the default-runner spec."""
        args = argparse.Namespace(xenna_streaming_scheduler="FRAGMENTATION_BASED")
        spec = XennaRunner.from_args(args)._streaming_spec
        assert spec.scheduler is SchedulerKind.FRAGMENTATION_BASED

    def test_from_args_saturation_aware_returns_saturation_aware_runner(self) -> None:
        """``SATURATION_AWARE`` dispatches to ``with_saturation_aware()`` and attaches a default config."""
        args = argparse.Namespace(xenna_streaming_scheduler="SATURATION_AWARE")
        spec = XennaRunner.from_args(args)._streaming_spec
        assert spec.scheduler is SchedulerKind.SATURATION_AWARE
        assert isinstance(spec.saturation_aware, SaturationAwareConfig)

    def test_from_args_unknown_value_falls_back_to_fragmentation_based(self) -> None:
        """Unknown scheduler strings stay safe (validator rejects bad values upstream at parse time)."""
        args = argparse.Namespace(xenna_streaming_scheduler="not-a-scheduler")
        spec = XennaRunner.from_args(args)._streaming_spec
        assert spec.scheduler is SchedulerKind.FRAGMENTATION_BASED
