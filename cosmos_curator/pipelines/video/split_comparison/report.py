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
"""Streamlit-free reporting for a split_comparison measurements root.

Loads the all-clips score distributions from the measurement tables
(``clip.lance`` / ``window.lance``) and buckets them into histograms with the
policy threshold always inserted as an exact bin edge, so each bucket is
unambiguously on the passing or failing side. Shared by the ``score-histogram``
CLI (text / JSON / PNG) and the Streamlit viewer's Distributions page.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lance
import numpy as np
import numpy.typing as npt
import smart_open  # type: ignore[import-untyped]

from cosmos_curator.core.utils.storage import storage_utils
from cosmos_curator.pipelines.video.split_comparison.store import eval_output_dir

_CLIP_DATASET = "clip.lance"
_WINDOW_DATASET = "window.lance"
_EVAL_SUMMARY = "eval.json"
DEFAULT_BINS = 30
_EMPTY_FLOATS: npt.NDArray[np.float64] = np.array([], dtype=np.float64)


@dataclass(frozen=True)
class ScoreDistributions:
    """Per-metric value arrays across *all* clips/windows (nulls dropped).

    Sourced from the measurement tables, not the issue table -- so they cover
    every clip, not just the ones that breached a threshold.
    """

    caption_similarity: npt.NDArray[np.float64]  # window.lance ``similarity`` (per window)
    aesthetic_abs_diff: npt.NDArray[np.float64]  # clip.lance ``aesthetic_abs_diff`` (per clip)
    motion_global_mean_abs_diff: npt.NDArray[np.float64]
    motion_per_patch_min_256_abs_diff: npt.NDArray[np.float64]


@dataclass(frozen=True)
class Metric:
    """Describes one distribution: its CLI name, where its values live, threshold, and failing side."""

    name: str  # short selector for --metric
    key: str  # attribute on ScoreDistributions
    title: str
    # Nested keys into eval.json ``policy``. For the score-|Δ| metrics this is the *absolute*
    # tolerance only: eval also passes a clip within ``rel_tolerance`` (``abs_diff / max(|a|,|b|)``),
    # which needs the per-clip a/b magnitudes we don't carry here. So a bucket marked failing against
    # this edge may still have passed eval via the relative tolerance -- see ``compute_histogram``.
    threshold_path: tuple[str, ...]
    fail_above: bool  # score |Δ| fails above tolerance; similarity fails below threshold
    clamp_unit: bool = False  # clamp to [0, 1] (cosine similarity can land a hair above 1.0)


METRICS: tuple[Metric, ...] = (
    Metric(
        "caption",
        "caption_similarity",
        "Caption similarity (per window)",
        ("caption_min_similarity",),
        fail_above=False,
        clamp_unit=True,
    ),
    Metric(
        "aesthetic",
        "aesthetic_abs_diff",
        "Aesthetic score |Δ| (per clip)",
        ("aesthetic", "abs_tolerance"),
        fail_above=True,
    ),
    Metric(
        "motion-global",
        "motion_global_mean_abs_diff",
        "Motion |Δ| — global mean (per clip)",
        ("motion", "abs_tolerance"),
        fail_above=True,
    ),
    Metric(
        "motion-per-patch",
        "motion_per_patch_min_256_abs_diff",
        "Motion |Δ| — per-patch min 256 (per clip)",
        ("motion", "abs_tolerance"),
        fail_above=True,
    ),
)

METRIC_NAMES: tuple[str, ...] = tuple(m.name for m in METRICS)
_METRICS_BY_NAME: dict[str, Metric] = {m.name: m for m in METRICS}


def metrics_for(names: list[str] | None) -> list[Metric]:
    """Resolve selector names to Metrics (order-preserving, de-duplicated); all metrics when ``None``."""
    if not names:
        return list(METRICS)
    seen: dict[str, Metric] = {}
    for name in names:
        seen.setdefault(name, _METRICS_BY_NAME[name])
    return list(seen.values())


@dataclass(frozen=True)
class Histogram:
    """A bucketed distribution for one metric, with the threshold as an exact edge."""

    title: str
    n: int
    stats: dict[str, float]  # mean, median, p95, min, max (empty when n == 0)
    threshold: float | None
    fail_above: bool
    breached: int | None  # count strictly beyond the threshold (below, or above when fail_above)
    lower_edges: list[float] = field(default_factory=list)
    upper_edges: list[float] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    failing: list[bool] = field(default_factory=list)  # per bucket: entirely on the failing side
    exact: list[bool] = field(default_factory=list)  # per bucket: an exact-endpoint singleton (lower == upper)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _join(root: str, *names: str) -> str:
    return str(storage_utils.get_full_path(root, *names))


def _read_float_columns(uri: str, *, profile: str, columns: list[str]) -> dict[str, npt.NDArray[np.float64]]:
    """Read ``columns`` from a Lance table as float arrays with nulls/NaNs dropped.

    Raises (``FileNotFoundError`` / ``OSError`` / ``ValueError``) when the dataset can't be opened
    or a column is absent -- callers that treat a table as optional catch these explicitly rather
    than relying on an empty return to signal failure. A readable-but-empty table yields one empty
    array per column.
    """
    storage_options = storage_utils.get_lance_storage_options(uri, profile_name=profile)
    table = lance.dataset(uri, storage_options=storage_options).to_table(columns=columns)
    result: dict[str, npt.NDArray[np.float64]] = {}
    for column in columns:
        values = table.column(column).to_numpy(zero_copy_only=False).astype(np.float64)
        result[column] = values[~np.isnan(values)]
    return result


def load_score_distributions(root: str, *, profile: str = "default") -> ScoreDistributions:
    """Read the all-clips score arrays from a measurements root's measurement tables."""
    clip_uri = _join(root, _CLIP_DATASET)
    try:
        clip = _read_float_columns(
            clip_uri,
            profile=profile,
            columns=["aesthetic_abs_diff", "motion_global_mean_abs_diff", "motion_per_patch_min_256_abs_diff"],
        )
    except (FileNotFoundError, ValueError, OSError) as err:
        # clip.lance is the required core table: fail loudly rather than emitting a "no data" report
        # that looks like a valid empty result to CI and terminal workflows.
        msg = f"Missing or unreadable required dataset: {clip_uri}"
        raise ValueError(msg) from err
    # window.lance may legitimately lack ``similarity`` for --no-captions roots; tolerate its absence.
    try:
        window = _read_float_columns(_join(root, _WINDOW_DATASET), profile=profile, columns=["similarity"])
    except (FileNotFoundError, ValueError, OSError):
        window = {}
    return ScoreDistributions(
        caption_similarity=window.get("similarity", _EMPTY_FLOATS),
        aesthetic_abs_diff=clip.get("aesthetic_abs_diff", _EMPTY_FLOATS),
        motion_global_mean_abs_diff=clip.get("motion_global_mean_abs_diff", _EMPTY_FLOATS),
        motion_per_patch_min_256_abs_diff=clip.get("motion_per_patch_min_256_abs_diff", _EMPTY_FLOATS),
    )


def load_thresholds(root: str, *, eval_name: str | None = None, profile: str = "default") -> dict[str, Any]:
    """Read the eval ``policy`` (thresholds/tolerances) from ``eval.json``; ``{}`` if absent.

    A named eval writes its ``eval.json`` under ``<root>/eval/<name>/`` (see ``store.eval_output_dir``),
    so ``eval_name`` must match the run's ``--eval-name`` for the thresholds to be found.
    """
    uri = _join(eval_output_dir(root, eval_name), _EVAL_SUMMARY)
    try:
        params = storage_utils.get_smart_open_params(uri, profile_name=profile)
        with smart_open.open(uri, "r", encoding="utf-8", **params) as handle:
            payload: dict[str, Any] = json.load(handle)
    except (FileNotFoundError, ValueError, OSError):
        return {}
    policy = payload.get("policy", {})
    return policy if isinstance(policy, dict) else {}


def _threshold_for(metric: Metric, policy: dict[str, Any]) -> float | None:
    node: Any = policy
    for key in metric.threshold_path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return float(node) if isinstance(node, (int, float)) else None


# --------------------------------------------------------------------------- #
# Bucketing
# --------------------------------------------------------------------------- #


def compute_histogram(  # noqa: PLR0913 -- a histogram is defined by its values + these bucketing knobs
    title: str,
    values: npt.NDArray[np.float64],
    *,
    threshold: float | None,
    fail_above: bool,
    bins: int = DEFAULT_BINS,
    clamp_unit: bool = False,
) -> Histogram:
    """Bucket ``values`` into an exact-0 bar, interior range buckets, and (bounded metrics) an exact-max bar.

    The exact endpoints get their own singleton bars so, e.g., "how many captions are *exactly* 1.0
    (identical)" or "how many clips are *exactly* 0 (unchanged)" isn't hidden inside the edge bucket.
    Interior buckets span ``(0, upper]`` from 0, with ``threshold`` inserted as an exact edge; a bucket
    is ``failing`` when its whole range is on the bad side (lower > threshold for score |Δ|, upper <=
    threshold for similarity -- eval passes a value sitting exactly on the threshold). ``clamp_unit``
    bounds the domain to ``[0, 1]`` and adds the exact-1 bar.

    For the score-|Δ| metrics ``threshold`` is the *absolute* tolerance only; eval additionally passes a
    clip within its ``rel_tolerance`` (a per-clip a/b magnitude not available here). ``failing`` and
    ``breached`` therefore over-count relative to eval -- they are an upper bound on the true failures.
    """
    if values.size == 0:
        return Histogram(title=title, n=0, stats={}, threshold=threshold, fail_above=fail_above, breached=None)
    if clamp_unit:
        values = np.clip(values, 0.0, 1.0)
    low, high = float(values.min()), float(values.max())
    stats = {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "min": low,
        "max": high,
    }
    breached: int | None = None
    if threshold is not None:
        breached = int((values > threshold).sum()) if fail_above else int((values < threshold).sum())

    split_low = 0.0
    split_high = 1.0 if clamp_unit else None  # bounded metrics also break out the exact top
    upper_bound = 1.0 if clamp_unit else high

    lowers: list[float] = []
    uppers: list[float] = []
    counts: list[int] = []
    exact: list[bool] = []

    def _add(lo: float, up: float, count: int, *, is_exact: bool) -> None:
        lowers.append(lo)
        uppers.append(up)
        counts.append(int(count))
        exact.append(is_exact)

    # exact-0 singleton (always -- these metrics are non-negative)
    _add(split_low, split_low, int((values == split_low).sum()), is_exact=True)

    # interior values, with the broken-out exact endpoints removed
    interior = values[values != split_low]
    if split_high is not None:
        interior = interior[interior != split_high]
    if interior.size and upper_bound > 0.0:
        edges = np.linspace(0.0, upper_bound, bins + 1)
        if threshold is not None and 0.0 < threshold < upper_bound:
            edges = np.unique(np.append(edges, threshold))
        interior_counts, edges = np.histogram(interior, bins=edges)
        for lo, up, count in zip(edges[:-1], edges[1:], interior_counts, strict=True):
            _add(float(lo), float(up), int(count), is_exact=False)

    # exact-max singleton for bounded metrics (e.g. similarity == 1.0)
    if split_high is not None:
        _add(split_high, split_high, int((values == split_high).sum()), is_exact=True)

    failing = [_bucket_failing(lo, up, threshold, fail_above=fail_above) for lo, up in zip(lowers, uppers, strict=True)]
    return Histogram(
        title=title,
        n=int(values.size),
        stats=stats,
        threshold=threshold,
        fail_above=fail_above,
        breached=breached,
        lower_edges=lowers,
        upper_edges=uppers,
        counts=counts,
        failing=failing,
        exact=exact,
    )


def _bucket_failing(lower: float, upper: float, threshold: float | None, *, fail_above: bool) -> bool:
    """Whether a bucket ``[lower, upper]`` sits entirely on the failing side of the threshold.

    Eval treats a value *exactly* on the threshold as passing (``abs_diff <= tolerance``;
    ``similarity >= threshold``). Since ``threshold`` is inserted as an exact bin edge, the bucket
    starting at it (``lower == threshold``) still contains that passing value, so for the score-|Δ|
    metrics a bucket is only entirely failing when ``lower > threshold`` (strict). Similarity fails
    strictly below the threshold, so a half-open ``upper <= threshold`` bucket holds only failing
    values. A singleton bucket exactly at the threshold (``lower == upper == threshold``, e.g. the
    exact-0/exact-1 buckets) contains only the passing threshold value, so it uses strict ``<`` to
    stay consistent with ``breached``.
    """
    if threshold is None:
        return False
    if fail_above:
        return lower > threshold
    return upper < threshold if lower == upper else upper <= threshold


def build_histograms(
    root: str,
    *,
    eval_name: str | None = None,
    profile: str = "default",
    bins: int = DEFAULT_BINS,
    metrics: list[Metric] | None = None,
) -> list[Histogram]:
    """Load a measurements root and bucket the selected metric distributions (all when ``metrics`` is None)."""
    dist = load_score_distributions(root, profile=profile)
    policy = load_thresholds(root, eval_name=eval_name, profile=profile)
    histograms: list[Histogram] = []
    for metric in metrics if metrics is not None else METRICS:
        values: npt.NDArray[np.float64] = getattr(dist, metric.key)
        histograms.append(
            compute_histogram(
                metric.title,
                values,
                threshold=_threshold_for(metric, policy),
                fail_above=metric.fail_above,
                bins=bins,
                clamp_unit=metric.clamp_unit,
            )
        )
    return histograms


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

_BAR_WIDTH = 40


def _stats_line(hist: Histogram) -> str:
    if hist.n == 0:
        return "  no data"
    s = hist.stats
    line = (
        f"  n={hist.n}  mean={s['mean']:.4g}  median={s['median']:.4g}  "
        f"p95={s['p95']:.4g}  min={s['min']:.4g}  max={s['max']:.4g}"
    )
    if hist.threshold is not None and hist.breached is not None:
        side = "above" if hist.fail_above else "below"
        line += f"  |  {side} {hist.threshold:.3g}: {hist.breached}"
    return line


def render_text(histograms: list[Histogram]) -> str:
    """Render histograms as horizontal unicode bar charts (log-scaled) with the threshold marked."""
    blocks: list[str] = []
    for hist in histograms:
        lines = [hist.title, _stats_line(hist)]
        if hist.n == 0:
            blocks.append("\n".join(lines))
            continue
        max_count = max(hist.counts) or 1
        scale = np.log1p(max_count)
        prev_failing = None
        for lower, count, failing, is_exact in zip(
            hist.lower_edges, hist.counts, hist.failing, hist.exact, strict=True
        ):
            if hist.threshold is not None and prev_failing is not None and failing != prev_failing:
                lines.append(f"  {'':>9} ── threshold {hist.threshold:.3g} ──")
            prev_failing = failing
            bar = "█" * round(_BAR_WIDTH * np.log1p(count) / scale) if count else ""
            tag = ("  above" if hist.fail_above else "  below") if failing else ""
            # Range buckets are labelled by their lower edge ([label, next)); exact bars by "=X".
            label = f"={lower:.4g}" if is_exact else f"{lower:.4g}"
            lines.append(f"  {label:>9} {bar:<{_BAR_WIDTH}} {count:>8,}{tag}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def to_json(histograms: list[Histogram]) -> dict[str, Any]:
    """Serialize histograms to a JSON-ready dict (buckets, counts, failing flags, stats, threshold)."""
    return {
        "histograms": [
            {
                "title": h.title,
                "n": h.n,
                "stats": h.stats,
                "threshold": h.threshold,
                "fail_above": h.fail_above,
                "breached": h.breached,
                "buckets": [
                    {"lower": lo, "upper": up, "count": ct, "failing": fl, "exact": ex}
                    for lo, up, ct, fl, ex in zip(
                        h.lower_edges, h.upper_edges, h.counts, h.failing, h.exact, strict=True
                    )
                ],
            }
            for h in histograms
        ]
    }


def render_png_bytes(histograms: list[Histogram], *, title: str | None = None) -> bytes:
    """Render all histograms into one PNG figure (one subplot per metric) and return the bytes.

    Pure: builds the figure in memory and returns the encoded PNG with no filesystem or cloud
    IO -- :func:`render_png` is the thin writer that persists these bytes. Requires matplotlib
    (raises ``RuntimeError`` if absent). ``title`` sets a figure suptitle.
    """
    import io  # noqa: PLC0415 -- local to the optional PNG path

    try:
        import matplotlib as mpl  # type: ignore[import-not-found]  # noqa: PLC0415 -- optional dep, only for --png

        mpl.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        msg = "PNG output requires matplotlib; run in an environment that provides it (e.g. the dev env)"
        raise RuntimeError(msg) from exc

    fig, axes = plt.subplots(len(histograms), 1, figsize=(11, 3.2 * len(histograms)), squeeze=False)
    for ax, hist in zip((a[0] for a in axes), histograms, strict=True):
        ax.set_title(hist.title, loc="left", fontsize=10)
        if hist.n == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_yticks([])
            ax.set_xticks([])
            continue
        # Categorical bars (one slot per bucket) so the exact-endpoint singletons get their own bars
        # alongside the range buckets. Labels: "=X" for exact bars, the lower edge for ranges.
        positions = list(range(len(hist.counts)))
        # Range buckets labelled by lower edge ([label, next)); exact-endpoint bars by "=X".
        labels = [f"={lo:.3g}" if ex else f"{lo:.3g}" for lo, ex in zip(hist.lower_edges, hist.exact, strict=True)]
        colors = ["#d62728" if f else "#4c78a8" for f in hist.failing]
        bars = ax.bar(positions, hist.counts, width=0.9, color=colors, edgecolor="white", linewidth=0.3)
        ax.set_yscale("symlog")
        ax.set_ylabel("count")
        ax.margins(y=0.18)  # headroom for the count labels above the bars
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.bar_label(bars, labels=[f"{c:,}" if c else "" for c in hist.counts], fontsize=7, padding=2)
        # Threshold divider: a vertical line at the failing<->passing boundary between two slots.
        if hist.threshold is not None:
            for i in range(1, len(hist.failing)):
                if hist.failing[i] != hist.failing[i - 1]:
                    ax.axvline(i - 0.5, color="#d62728", linestyle="--", linewidth=1)
                    ax.annotate(
                        f"threshold {hist.threshold:.3g}",
                        xy=(i - 0.5, 1.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(3, -3),
                        textcoords="offset points",
                        color="#d62728",
                        fontsize=8,
                        va="top",
                    )
                    break
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97) if title else None)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    return buffer.getvalue()


def render_png(histograms: list[Histogram], path: str, *, profile: str = "default", title: str | None = None) -> None:
    """Write the histogram figure to ``path`` (local or a cloud URL via ``smart_open``).

    A thin writer over :func:`render_png_bytes` so the CLI can drop the figure straight into S3
    when run inside a container. ``profile`` selects storage credentials; ``title`` is a suptitle.
    """
    data = render_png_bytes(histograms, title=title)
    if not storage_utils.is_remote_path(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    params = storage_utils.get_smart_open_params(path, profile_name=profile)
    with smart_open.open(path, "wb", **params) as handle:
        handle.write(data)
