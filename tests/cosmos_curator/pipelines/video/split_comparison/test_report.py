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
"""Tests for the streamlit-free report module: histogram bucketing + threshold marking."""

import json
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest

from cosmos_curator.pipelines.video.split_comparison import report


def test_threshold_is_an_exact_edge_and_below_buckets_are_flagged() -> None:
    """A below-threshold metric: the threshold is a bin edge and buckets under it are flagged failing."""
    values = np.clip(np.concatenate([np.full(100, 0.95), np.linspace(0.70, 0.84, 20)]), 0.0, 1.0)
    hist = report.compute_histogram("sim", values, threshold=0.85, fail_above=False, bins=10, clamp_unit=True)

    assert any(np.isclose(0.85, e) for e in hist.upper_edges)  # threshold split into an edge
    # every bucket entirely below the threshold is failing; nothing at/above is.
    for upper, failing in zip(hist.upper_edges, hist.failing, strict=True):
        assert failing == (upper <= 0.85 + 1e-9)
    assert hist.breached == int((values < 0.85).sum()) == 20


def test_exact_endpoint_bars_split_out() -> None:
    """A clamped metric breaks out exact 0 and exact 1 into their own singleton bars."""
    values = np.concatenate([np.full(50, 1.0), np.full(3, 0.0), np.linspace(0.6, 0.99, 20)])
    hist = report.compute_histogram("sim", values, threshold=0.85, fail_above=False, bins=10, clamp_unit=True)

    # first bucket is the exact-0 singleton, last is exact-1
    assert hist.exact[0]
    assert hist.lower_edges[0] == hist.upper_edges[0] == 0.0
    assert hist.counts[0] == 3
    assert hist.exact[-1]
    assert hist.lower_edges[-1] == hist.upper_edges[-1] == 1.0
    assert hist.counts[-1] == 50
    # the 50 exact-1 values are NOT lumped into an interior bucket
    assert sum(c for c, ex in zip(hist.counts, hist.exact, strict=True) if not ex) == 20
    assert sum(hist.counts) == values.size


def test_fail_above_flags_high_buckets() -> None:
    """A tolerance metric fails on the high side: buckets *strictly* above the tolerance are flagged."""
    values = np.concatenate([np.zeros(50), np.linspace(0.06, 0.5, 12)])
    hist = report.compute_histogram("aes", values, threshold=0.05, fail_above=True, bins=10)

    assert any(np.isclose(0.05, e) for e in hist.lower_edges)
    # A bucket is failing only when its whole range is strictly above the threshold; the boundary
    # bucket starting at the threshold spans the passing value == threshold, so it is not flagged.
    for lower, failing in zip(hist.lower_edges, hist.failing, strict=True):
        assert failing == (lower > 0.05 + 1e-9)
    assert hist.breached == int((values > 0.05).sum()) == 12
    # failing buckets hold the breached values minus the boundary [0.05, .) bucket's count -- no
    # passing value is ever colored failing.
    boundary_idx = next(i for i, lo in enumerate(hist.lower_edges) if np.isclose(lo, 0.05))
    failing_count = sum(c for c, f in zip(hist.counts, hist.failing, strict=True) if f)
    assert failing_count == 12 - hist.counts[boundary_idx]


def test_endpoint_threshold_singleton_is_not_failing() -> None:
    """A similarity threshold at the 1.0 endpoint: the exact-1 singleton passes (1.0 >= 1.0)."""
    values = np.concatenate([np.full(30, 1.0), np.linspace(0.6, 0.99, 20)])
    hist = report.compute_histogram("sim", values, threshold=1.0, fail_above=False, bins=10, clamp_unit=True)

    # the exact-1 singleton holds only the passing threshold value, so it must not be red...
    assert hist.lower_edges[-1] == hist.upper_edges[-1] == 1.0
    assert hist.exact[-1]
    assert not hist.failing[-1]
    # ...and coloring agrees with breached (nothing is strictly below 1.0 among the exact-1 values).
    assert hist.breached == int((values < 1.0).sum()) == 20
    failing_count = sum(c for c, f in zip(hist.counts, hist.failing, strict=True) if f)
    assert failing_count == hist.breached


def test_value_exactly_on_threshold_is_not_failing() -> None:
    """A score-|Δ| value sitting exactly on the tolerance passes eval, so its bucket is not flagged."""
    values = np.concatenate([np.zeros(20), np.full(5, 0.05), np.full(7, 0.2)])
    hist = report.compute_histogram("aes", values, threshold=0.05, fail_above=True, bins=10)

    # the five values at exactly 0.05 land in the [0.05, .) boundary bucket, which must not be red
    boundary_idx = next(i for i, lo in enumerate(hist.lower_edges) if np.isclose(lo, 0.05))
    assert hist.counts[boundary_idx] == 5
    assert not hist.failing[boundary_idx]
    assert hist.breached == 7  # only the strictly-above (0.2) values breach


def test_all_zero_diffs_become_the_exact_zero_bar() -> None:
    """All-identical (zero) differences land entirely in the exact-0 bar, not a range bucket."""
    hist = report.compute_histogram("aes", np.zeros(1000), threshold=0.05, fail_above=True, bins=10)
    assert hist.exact == [True]
    assert hist.lower_edges == [0.0]
    assert hist.upper_edges == [0.0]
    assert hist.counts == [1000]
    assert hist.failing == [False]  # zero diff passes an above-tolerance metric
    assert hist.breached == 0


def test_empty_values() -> None:
    """No data yields an n=0 histogram, not a crash."""
    hist = report.compute_histogram("aes", np.array([], dtype=np.float64), threshold=0.05, fail_above=True)
    assert hist.n == 0
    assert hist.stats == {}


def test_counts_sum_to_n() -> None:
    """Every value lands in exactly one bucket (bucketing from 0 covers the whole range)."""
    values = np.concatenate([np.zeros(30), np.linspace(0.01, 0.9, 70)])
    hist = report.compute_histogram("aes", values, threshold=0.05, fail_above=True, bins=15)
    assert sum(hist.counts) == values.size


def test_to_json_shape() -> None:
    """JSON carries per-bucket lower/upper/count/failing plus stats and threshold."""
    hist = report.compute_histogram("sim", np.linspace(0.5, 1.0, 100), threshold=0.85, fail_above=False, bins=8)
    payload = report.to_json([hist])
    entry = payload["histograms"][0]
    assert entry["threshold"] == 0.85
    assert set(entry["buckets"][0]) == {"lower", "upper", "count", "failing", "exact"}


def _write_root(tmp_path: Path) -> str:
    """Write a minimal measurements root (clip/window lance + eval.json) and return its path."""
    root = tmp_path / "m"
    root.mkdir()
    lance.write_dataset(
        pa.table({"similarity": np.clip(np.concatenate([np.full(200, 0.97), np.full(5, 0.8)]), 0, 1)}),
        str(root / "window.lance"),
    )
    lance.write_dataset(
        pa.table(
            {
                "aesthetic_abs_diff": np.concatenate([np.zeros(180), np.full(5, 0.3)]),
                "motion_global_mean_abs_diff": np.zeros(185),
                "motion_per_patch_min_256_abs_diff": np.zeros(185),
            }
        ),
        str(root / "clip.lance"),
    )
    (root / "eval.json").write_text(
        json.dumps(
            {
                "policy": {
                    "caption_min_similarity": 0.85,
                    "aesthetic": {"abs_tolerance": 0.05},
                    "motion": {"abs_tolerance": 0.05},
                }
            }
        )
    )
    return str(root)


def test_metrics_for_selection() -> None:
    """metrics_for resolves names (order-preserving, de-duplicated) and defaults to all."""
    assert report.metrics_for(None) == list(report.METRICS)
    picked = report.metrics_for(["aesthetic", "caption", "aesthetic"])
    assert [m.name for m in picked] == ["aesthetic", "caption"]


def test_build_histograms_respects_metric_filter(tmp_path: Path) -> None:
    """A metric filter limits which histograms are built (and their order)."""
    only = report.metrics_for(["caption"])
    histograms = report.build_histograms(_write_root(tmp_path), bins=10, metrics=only)
    assert [h.title for h in histograms] == ["Caption similarity (per window)"]


def test_build_histograms_end_to_end(tmp_path: Path) -> None:
    """build_histograms loads the tables + policy and buckets all four metrics with thresholds applied."""
    histograms = report.build_histograms(_write_root(tmp_path), bins=10)
    assert [h.title for h in histograms] == [m.title for m in report.METRICS]
    similarity = histograms[0]
    assert similarity.n == 205
    assert similarity.threshold == 0.85
    assert similarity.breached == 5  # the five 0.8 windows are below threshold
    # motion per-patch is all zeros -> a single exact-0 bar
    assert histograms[3].exact == [True]
    assert histograms[3].counts == [185]


def test_render_png_bytes_is_pure_and_encodes_png() -> None:
    """render_png_bytes returns a valid PNG in memory with no filesystem/cloud IO."""
    histograms = [
        report.compute_histogram("sim", np.linspace(0.5, 1.0, 50), threshold=0.85, fail_above=False, clamp_unit=True),
        report.compute_histogram("empty", np.array([], dtype=np.float64), threshold=None, fail_above=True),
    ]
    data = report.render_png_bytes(histograms, title="t")
    assert data[:8] == b"\x89PNG\r\n\x1a\n"  # PNG signature


def test_render_png_writes_bytes_to_local_path(tmp_path: Path) -> None:
    """render_png is a thin writer: it persists render_png_bytes' output to the path."""
    hist = report.compute_histogram("aes", np.linspace(0.0, 0.5, 40), threshold=0.05, fail_above=True)
    out = tmp_path / "nested" / "hist.png"
    report.render_png([hist], str(out))
    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_missing_measurements_root_raises(tmp_path: Path) -> None:
    """A missing/unreadable clip.lance fails loudly instead of returning an empty 'no data' report."""
    missing = str(tmp_path / "does-not-exist")
    with pytest.raises(ValueError, match="Missing or unreadable required dataset"):
        report.load_score_distributions(missing)
    with pytest.raises(ValueError, match="Missing or unreadable required dataset"):
        report.build_histograms(missing, bins=10)


def test_named_eval_thresholds_are_resolved(tmp_path: Path) -> None:
    """Thresholds for a --eval-name run live under eval/<name>/eval.json and need the matching eval_name."""
    root = _write_root(tmp_path)
    # relocate the policy to a named-eval subdir, mirroring store.write_eval(..., eval_name="strict")
    named_dir = Path(root) / "eval" / "strict"
    named_dir.mkdir(parents=True)
    (named_dir / "eval.json").write_text(
        json.dumps({"policy": {"caption_min_similarity": 0.85, "aesthetic": {"abs_tolerance": 0.05}}})
    )
    (Path(root) / "eval.json").unlink()  # no default eval.json -> only the named one exists

    # without eval_name the thresholds are absent (histograms still render, just unmarked)
    assert report.load_thresholds(root) == {}
    assert report.build_histograms(root, bins=10)[0].threshold is None

    # with the matching eval_name the policy thresholds are found
    assert report.load_thresholds(root, eval_name="strict")["caption_min_similarity"] == 0.85
    assert report.build_histograms(root, eval_name="strict", bins=10)[0].threshold == 0.85
