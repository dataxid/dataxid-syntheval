"""Profile comparison — stat deltas, alert diffs, distribution overlays."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl  # noqa: TC002 — used at runtime
from dataxid_profiling import Alert  # noqa: TC002 — used at runtime
from dataxid_profiling._correlations import CorrelationResult  # noqa: TC002 — used at runtime

if TYPE_CHECKING:
    from dataxid_profiling import ProfileReport

# Numeric fields where delta = original - synthetic is meaningful
_NUMERIC_DELTA_FIELDS = (
    "mean", "std", "variance", "median", "min", "max", "range", "sum",
    "q25", "q75", "p5", "p95", "iqr", "cv", "mad",
    "skewness", "kurtosis",
    "missing_pct", "distinct_count", "distinct_pct",
    "zeros_pct", "negative_pct",
)

_CATEGORICAL_DELTA_FIELDS = (
    "missing_pct", "distinct_count", "distinct_pct", "imbalance",
    "length_min", "length_max", "length_mean",
)

_BOOLEAN_DELTA_FIELDS = (
    "missing_pct", "true_pct", "false_pct", "true_count", "false_count",
)

_DATETIME_DELTA_FIELDS = ("missing_pct", "distinct_count", "distinct_pct")


def compute_column_diffs(
    original: ProfileReport,
    synthetic: ProfileReport,
) -> dict[str, dict[str, Any]]:
    """Compute per-column stat deltas between original and synthetic profiles.

    Returns a dict keyed by column name. Each value contains the column type
    and numeric deltas (original - synthetic) for relevant statistics.
    Columns present in only one profile are skipped.
    """
    orig_stats = original.stats
    syn_stats = synthetic.stats
    shared_columns = [c for c in orig_stats if c in syn_stats]

    result: dict[str, dict[str, Any]] = {}
    for col in shared_columns:
        orig = orig_stats[col]
        syn = syn_stats[col]
        col_type = orig.get("column_type")
        if col_type is None:
            continue

        type_name = col_type.name if hasattr(col_type, "name") else str(col_type)
        delta_fields = _fields_for_type(type_name)
        deltas = _compute_deltas(orig, syn, delta_fields)
        deltas["type"] = type_name
        result[col] = deltas

    return result


def _fields_for_type(type_name: str) -> tuple[str, ...]:
    mapping: dict[str, tuple[str, ...]] = {
        "NUMERIC": _NUMERIC_DELTA_FIELDS,
        "CATEGORICAL": _CATEGORICAL_DELTA_FIELDS,
        "BOOLEAN": _BOOLEAN_DELTA_FIELDS,
        "DATETIME": _DATETIME_DELTA_FIELDS,
    }
    return mapping.get(type_name, ("missing_pct",))


def _compute_deltas(
    orig: dict[str, Any],
    syn: dict[str, Any],
    fields: tuple[str, ...],
) -> dict[str, Any]:
    """Compute original - synthetic for each field. Skip if either side is None."""
    deltas: dict[str, Any] = {}
    for field in fields:
        orig_val = orig.get(field)
        syn_val = syn.get(field)
        if orig_val is None or syn_val is None:
            deltas[f"{field}_delta"] = None
            continue
        try:
            deltas[f"{field}_delta"] = float(orig_val) - float(syn_val)
        except (TypeError, ValueError):
            deltas[f"{field}_delta"] = None
    return deltas


# -- Alert diff ---------------------------------------------------------------


_OVERVIEW_FIELDS = (
    "n_rows", "n_columns", "missing_cells", "missing_cells_pct",
    "duplicate_rows", "duplicate_rows_pct",
)


def compute_overview_diff(
    original: ProfileReport,
    synthetic: ProfileReport,
) -> dict[str, dict[str, Any]]:
    """Compare dataset-level overview stats.

    Returns {field: {original, synthetic, delta}} for each overview metric.
    """
    orig_ov = original.overview
    syn_ov = synthetic.overview

    result: dict[str, dict[str, Any]] = {}
    for field in _OVERVIEW_FIELDS:
        o = orig_ov.get(field)
        s = syn_ov.get(field)
        delta = None
        if isinstance(o, (int, float)) and isinstance(s, (int, float)):
            delta = round(s - o, 6)
        result[field] = {"original": o, "synthetic": s, "delta": delta}

    orig_types = orig_ov.get("type_distribution", {})
    syn_types = syn_ov.get("type_distribution", {})
    result["type_distribution"] = {
        "original": orig_types,
        "synthetic": syn_types,
        "delta": None,
    }
    return result


def _alert_key(alert: Alert) -> tuple[str | None, str]:
    """Identity key for an alert: (column, alert_type name)."""
    return (alert.column, alert.alert_type.name)


def compute_alert_diff(
    original: ProfileReport,
    synthetic: ProfileReport,
) -> dict[str, list[dict[str, Any]]]:
    """Compare alerts between original and synthetic profiles.

    Returns {"new": [...], "resolved": [...]}.
    - new: alerts present in synthetic but not in original
    - resolved: alerts present in original but not in synthetic
    """
    orig_keys = {_alert_key(a) for a in original.alerts}
    syn_keys = {_alert_key(a) for a in synthetic.alerts}

    new_keys = syn_keys - orig_keys
    resolved_keys = orig_keys - syn_keys

    new = [
        _serialize_alert(a) for a in synthetic.alerts if _alert_key(a) in new_keys
    ]
    resolved = [
        _serialize_alert(a) for a in original.alerts if _alert_key(a) in resolved_keys
    ]

    return {"new": new, "resolved": resolved}


def _serialize_alert(alert: Alert) -> dict[str, Any]:
    return {
        "column": alert.column,
        "alert_type": alert.alert_type.name,
        "value": alert.value,
        "details": alert.details,
    }


# -- Distribution overlay -----------------------------------------------------


def compute_distribution_overlays(
    original: ProfileReport,
    synthetic: ProfileReport,
) -> dict[str, dict[str, Any]]:
    """Build per-column overlay data for chart rendering.

    For NUMERIC columns: returns histogram bins from both profiles aligned by
    breakpoint so they can be rendered as an overlay bar chart.

    For CATEGORICAL columns: returns top-value frequencies from both profiles
    merged by value so they can be rendered as a grouped bar chart.

    Columns present in only one profile are skipped.
    """
    orig_stats = original.stats
    syn_stats = synthetic.stats
    shared = [c for c in orig_stats if c in syn_stats]

    result: dict[str, dict[str, Any]] = {}
    for col in shared:
        orig = orig_stats[col]
        syn = syn_stats[col]
        col_type = orig.get("column_type")
        if col_type is None:
            continue

        type_name = col_type.name if hasattr(col_type, "name") else str(col_type)

        if type_name == "NUMERIC":
            overlay = _numeric_histogram_overlay(orig, syn)
        elif type_name == "CATEGORICAL":
            overlay = _categorical_frequency_overlay(orig, syn)
        else:
            continue

        if overlay is not None:
            result[col] = {"type": type_name, "overlay": overlay}

    return result


_OVERLAY_BINS = 20


def _numeric_histogram_overlay(
    orig: dict[str, Any],
    syn: dict[str, Any],
) -> list[dict[str, Any]] | None:
    """Build aligned histogram bins for both profiles.

    Profiles may have different breakpoints, so we create a shared set of
    equally-spaced bins spanning the full range of both histograms and
    redistribute counts into them.
    """
    orig_hist = orig.get("histogram", [])
    syn_hist = syn.get("histogram", [])
    if not orig_hist and not syn_hist:
        return None

    all_bps = [h["breakpoint"] for h in orig_hist] + [h["breakpoint"] for h in syn_hist]
    lo, hi = min(all_bps), max(all_bps)
    if lo == hi:
        return None

    n_bins = _OVERLAY_BINS
    step = (hi - lo) / n_bins
    edges = [lo + i * step for i in range(n_bins + 1)]

    orig_counts = _rebin(orig_hist, edges)
    syn_counts = _rebin(syn_hist, edges)
    orig_total = sum(orig_counts) or 1
    syn_total = sum(syn_counts) or 1

    return [
        {
            "breakpoint": round((edges[i] + edges[i + 1]) / 2, 1),
            "original": round(orig_counts[i] / orig_total * 100, 2),
            "synthetic": round(syn_counts[i] / syn_total * 100, 2),
            "original_count": orig_counts[i],
            "synthetic_count": syn_counts[i],
        }
        for i in range(n_bins)
    ]


def _rebin(
    hist: list[dict[str, Any]], edges: list[float]
) -> list[int]:
    """Place existing histogram counts into new bins defined by edges."""
    n_bins = len(edges) - 1
    counts = [0] * n_bins
    for h in hist:
        bp = h["breakpoint"]
        c = h["count"]
        for i in range(n_bins):
            if edges[i] <= bp < edges[i + 1]:
                counts[i] += c
                break
        else:
            counts[-1] += c
    return counts


def _categorical_frequency_overlay(
    orig: dict[str, Any],
    syn: dict[str, Any],
) -> list[dict[str, Any]] | None:
    """Merge top-value frequencies into [{value, original, synthetic}]."""
    orig_top = orig.get("top_values", [])
    syn_top = syn.get("top_values", [])
    if not orig_top and not syn_top:
        return None

    orig_map = {v["value"]: v["count"] for v in orig_top}
    syn_map = {v["value"]: v["count"] for v in syn_top}
    all_values = list(dict.fromkeys(
        [v["value"] for v in orig_top] + [v["value"] for v in syn_top]
    ))
    orig_total = sum(orig_map.values()) or 1
    syn_total = sum(syn_map.values()) or 1

    return [
        {
            "value": val,
            "original": round(orig_map.get(val, 0) / orig_total * 100, 2),
            "synthetic": round(syn_map.get(val, 0) / syn_total * 100, 2),
            "original_count": orig_map.get(val, 0),
            "synthetic_count": syn_map.get(val, 0),
        }
        for val in all_values
    ]


# -- Correlation matrix diff --------------------------------------------------


def compute_correlation_diffs(
    original: ProfileReport,
    synthetic: ProfileReport,
) -> dict[str, pl.DataFrame]:
    """Compute per-method correlation matrix differences (original - synthetic).

    Returns a dict keyed by method name (e.g. "pearson"). Each value is a
    Polars DataFrame with a "column" label column and numeric diff columns,
    only for columns shared by both matrices.
    """
    orig_corrs = original.correlations
    syn_corrs = synthetic.correlations
    shared_methods = [m for m in orig_corrs if m in syn_corrs]

    result: dict[str, pl.DataFrame] = {}
    for method in shared_methods:
        diff = _subtract_matrices(orig_corrs[method], syn_corrs[method])
        if diff is not None:
            result[method] = diff

    return result


def compute_correlation_matrices(
    original: ProfileReport,
    synthetic: ProfileReport,
) -> dict[str, dict[str, pl.DataFrame]]:
    """Extract raw correlation matrices for shared methods.

    Returns {method: {"original": df, "synthetic": df}} where each DataFrame
    contains only columns shared by both profiles.
    """
    orig_corrs = original.correlations
    syn_corrs = synthetic.correlations
    shared_methods = [m for m in orig_corrs if m in syn_corrs]

    result: dict[str, dict[str, pl.DataFrame]] = {}
    for method in shared_methods:
        pair = _extract_shared_matrices(orig_corrs[method], syn_corrs[method])
        if pair is not None:
            result[method] = pair

    return result


def _extract_shared_matrices(
    orig: CorrelationResult,
    syn: CorrelationResult,
) -> dict[str, pl.DataFrame] | None:
    """Align two correlation matrices on shared columns and return both."""
    orig_cols = [c for c in orig.matrix.columns if c != "column"]
    syn_cols = [c for c in syn.matrix.columns if c != "column"]
    shared = [c for c in orig_cols if c in syn_cols]
    if not shared:
        return None

    orig_rows = {row["column"]: row for row in orig.matrix.iter_rows(named=True)}
    syn_rows = {row["column"]: row for row in syn.matrix.iter_rows(named=True)}
    shared_rows = [c for c in shared if c in orig_rows and c in syn_rows]

    def _build_df(rows_map: dict[str, Any]) -> pl.DataFrame:
        rows: list[dict[str, Any]] = []
        for row_name in shared_rows:
            entry: dict[str, Any] = {"column": row_name}
            for col_name in shared:
                val = rows_map[row_name].get(col_name)
                entry[col_name] = float(val) if val is not None else None
            rows.append(entry)
        return pl.DataFrame(rows)

    return {"original": _build_df(orig_rows), "synthetic": _build_df(syn_rows)}


def _subtract_matrices(
    orig: CorrelationResult,
    syn: CorrelationResult,
) -> pl.DataFrame | None:
    """Subtract two correlation matrices, aligning on shared columns."""
    orig_cols = [c for c in orig.matrix.columns if c != "column"]
    syn_cols = [c for c in syn.matrix.columns if c != "column"]
    shared = [c for c in orig_cols if c in syn_cols]
    if not shared:
        return None

    orig_rows = {row["column"]: row for row in orig.matrix.iter_rows(named=True)}
    syn_rows = {row["column"]: row for row in syn.matrix.iter_rows(named=True)}
    shared_rows = [c for c in shared if c in orig_rows and c in syn_rows]

    rows: list[dict[str, Any]] = []
    for row_name in shared_rows:
        entry: dict[str, Any] = {"column": row_name}
        for col_name in shared:
            o_val = orig_rows[row_name].get(col_name)
            s_val = syn_rows[row_name].get(col_name)
            if o_val is not None and s_val is not None:
                entry[col_name] = float(o_val) - float(s_val)
            else:
                entry[col_name] = None
        rows.append(entry)

    return pl.DataFrame(rows)


# -- Interaction overlays -----------------------------------------------------

_SCATTER_SAMPLE = 1500


def compute_interaction_overlays(
    original: ProfileReport,
    synthetic: ProfileReport,
) -> dict[str, Any]:
    """Build raw interaction data for dynamic JS rendering.

    Returns a dict with:
      numeric_columns, categorical_columns,
      original_numeric_data, synthetic_numeric_data  (col → sampled float list),
      original_boxplot_stats, synthetic_boxplot_stats (cat → num → group list).
    """
    orig_inter = original.interactions
    syn_inter = synthetic.interactions
    empty: dict[str, Any] = {
        "numeric_columns": [],
        "categorical_columns": [],
        "original_numeric_data": {},
        "synthetic_numeric_data": {},
        "original_boxplot_stats": {},
        "synthetic_boxplot_stats": {},
    }
    if orig_inter is None or syn_inter is None:
        return empty

    shared_num = [c for c in orig_inter.numeric_columns if c in syn_inter.numeric_columns]
    shared_cat = [
        c for c in orig_inter.categorical_columns
        if c in syn_inter.categorical_columns
    ]

    orig_num = _downsample_numeric_data(orig_inter.numeric_data, shared_num)
    syn_num = _downsample_numeric_data(syn_inter.numeric_data, shared_num)
    orig_box = _serialize_boxplot_stats(orig_inter.boxplot_stats, shared_cat, shared_num)
    syn_box = _serialize_boxplot_stats(syn_inter.boxplot_stats, shared_cat, shared_num)

    return {
        "numeric_columns": shared_num,
        "categorical_columns": shared_cat,
        "original_numeric_data": orig_num,
        "synthetic_numeric_data": syn_num,
        "original_boxplot_stats": orig_box,
        "synthetic_boxplot_stats": syn_box,
    }


def _downsample_numeric_data(
    numeric_data: dict[str, list[float]],
    columns: list[str],
) -> dict[str, list[float]]:
    """Downsample each column's values to _SCATTER_SAMPLE."""
    result: dict[str, list[float]] = {}
    for col in columns:
        vals = numeric_data.get(col, [])
        n = len(vals)
        if n <= _SCATTER_SAMPLE:
            result[col] = vals
        else:
            step = n / _SCATTER_SAMPLE
            result[col] = [vals[int(i * step)] for i in range(_SCATTER_SAMPLE)]
    return result


def _serialize_boxplot_stats(
    boxplot_stats: dict[str, Any],
    cat_columns: list[str],
    num_columns: list[str],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Convert BoxPlotGroup objects to plain dicts, keyed by cat → num."""
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for cat_col in cat_columns:
        cat_data = boxplot_stats.get(cat_col, {})
        if not cat_data:
            continue
        num_dict: dict[str, list[dict[str, Any]]] = {}
        for num_col in num_columns:
            groups = cat_data.get(num_col, [])
            if not groups:
                continue
            num_dict[num_col] = [
                {
                    "category": g.category,
                    "min": g.min,
                    "q1": g.q1,
                    "median": g.median,
                    "q3": g.q3,
                    "max": g.max,
                    "outliers": getattr(g, "outliers", []),
                }
                for g in groups
            ]
        if num_dict:
            result[cat_col] = num_dict
    return result
