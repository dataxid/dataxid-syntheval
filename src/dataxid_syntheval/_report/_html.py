"""HTML report rendering via Jinja2."""

from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl  # noqa: TC002 — used at runtime
from jinja2 import Environment, FileSystemLoader

from dataxid_syntheval._report._charts import diff_heatmap, grouped_bar, overlay_histogram

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def render_html(
    *,
    title: str,
    version: str,
    overview_diff: dict[str, dict[str, Any]],
    column_diffs: dict[str, dict[str, Any]],
    alert_diff: dict[str, list[dict[str, Any]]],
    distribution_overlays: dict[str, dict[str, Any]],
    correlation_diffs: dict[str, pl.DataFrame],
    correlation_matrices: dict[str, dict[str, pl.DataFrame]],
    interaction_overlays: dict[str, Any],
    original_stats: dict[str, dict[str, Any]],
    synthetic_stats: dict[str, dict[str, Any]],
    original_rows: int,
    synthetic_rows: int,
) -> str:
    """Render the SynthEval comparison report as a self-contained HTML string."""
    env = _build_env()
    template = env.get_template("syntheval.html.j2")

    columns = _prepare_column_comparison(column_diffs, original_stats, synthetic_stats)
    dist_charts = _prepare_distribution_charts(distribution_overlays)
    corr_charts = _prepare_correlation_charts(correlation_diffs, correlation_matrices)
    interactions = _prepare_interaction_data(interaction_overlays)

    return template.render(
        title=title,
        version=version,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        n_columns=len(column_diffs),
        original_rows=original_rows,
        synthetic_rows=synthetic_rows,
        overview=_prepare_overview(overview_diff),
        columns=columns,
        alert_diff=alert_diff,
        distribution_charts=dist_charts,
        correlation_charts=corr_charts,
        interactions=interactions,
        logo_b64=_load_asset_b64("dataxid_logo.png"),
        icon_b64=_load_asset_b64("icon.png"),
    )


_OVERVIEW_DISPLAY: dict[str, tuple[str, str]] = {
    "n_rows": ("Rows", "int"),
    "n_columns": ("Columns", "int"),
    "missing_cells_pct": ("Missing %", "pct"),
    "duplicate_rows_pct": ("Duplicate %", "pct"),
}


def _prepare_overview(
    overview_diff: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build display-ready overview comparison rows."""
    rows: list[dict[str, Any]] = []
    for field, (label, fmt) in _OVERVIEW_DISPLAY.items():
        entry = overview_diff.get(field, {})
        o, s, d = entry.get("original"), entry.get("synthetic"), entry.get("delta")
        rows.append({
            "label": label,
            "original": _fmt_overview(o, fmt),
            "synthetic": _fmt_overview(s, fmt),
            "delta": _fmt_overview(d, fmt, is_delta=True),
        })

    orig_types = overview_diff.get("type_distribution", {}).get("original", {})
    syn_types = overview_diff.get("type_distribution", {}).get("synthetic", {})
    rows.append({
        "label": "Types",
        "original": _fmt_type_dist(orig_types),
        "synthetic": _fmt_type_dist(syn_types),
        "delta": "—",
    })
    return rows


def _fmt_overview(val: Any, fmt: str, *, is_delta: bool = False) -> str:
    if val is None:
        return "—"
    if fmt == "int":
        prefix = "+" if is_delta and val > 0 else ""
        return f"{prefix}{int(val):,}"
    if fmt == "pct":
        prefix = "+" if is_delta and val > 0 else ""
        return f"{prefix}{val * 100 if abs(val) < 1 else val:.1f}%"
    return str(val)


def _fmt_type_dist(types: dict[str, int]) -> str:
    if not types:
        return "—"
    parts = [f"{v} {k.lower()}" for k, v in types.items()]
    return " / ".join(parts)


def _load_asset_b64(filename: str) -> str:
    path = _TEMPLATE_DIR / filename
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _build_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=True,
    )
    env.filters["format_delta"] = _format_delta
    env.filters["format_stat"] = _format_stat
    env.filters["format_alert_value"] = _format_alert_value
    return env


def _format_delta(value: Any) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
        sign = "+" if v > 0 else ""
        return f"{sign}{v:,.4f}"
    except (TypeError, ValueError):
        return str(value)


def _format_alert_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        v = float(value)
        if 0 < v <= 1:
            return f"{v * 100:.1f}%"
        return f"{v:,.2f}"
    except (TypeError, ValueError):
        return str(value)


def _format_stat(value: Any) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
        if abs(v) >= 1_000_000:
            return f"{v:,.0f}"
        if abs(v) >= 100:
            return f"{v:,.2f}"
        return f"{v:,.4f}"
    except (TypeError, ValueError):
        return str(value)


_STAT_DISPLAY: dict[str, str] = {
    "mean": "Mean",
    "std": "Std Dev",
    "variance": "Variance",
    "median": "Median",
    "min": "Min",
    "max": "Max",
    "range": "Range",
    "sum": "Sum",
    "q25": "Q25",
    "q75": "Q75",
    "p5": "P5",
    "p95": "P95",
    "iqr": "IQR",
    "cv": "CV",
    "mad": "MAD",
    "skewness": "Skewness",
    "kurtosis": "Kurtosis",
    "missing_pct": "Missing %",
    "distinct_count": "Distinct",
    "distinct_pct": "Distinct %",
    "zeros_pct": "Zeros %",
    "negative_pct": "Negative %",
    "imbalance": "Imbalance",
    "length_min": "Length Min",
    "length_max": "Length Max",
    "length_mean": "Length Mean",
    "true_pct": "True %",
    "false_pct": "False %",
    "true_count": "True Count",
    "false_count": "False Count",
}

_PRIMARY_STATS = (
    "mean", "std", "median", "min", "max", "missing_pct",
    "distinct_count", "imbalance", "true_pct", "false_pct",
)


def _prepare_column_comparison(
    column_diffs: dict[str, dict[str, Any]],
    original_stats: dict[str, dict[str, Any]],
    synthetic_stats: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build per-column comparison data: type, chart-ready stats with orig/syn/delta.

    Returns {col_name: {type, stats: [{name, label, original, synthetic, delta, primary}]}}.
    """
    result: dict[str, dict[str, Any]] = {}
    for col_name, diffs in column_diffs.items():
        col_type = diffs.get("type", "")
        orig = original_stats.get(col_name, {})
        syn = synthetic_stats.get(col_name, {})

        stats: list[dict[str, Any]] = []
        for key, val in diffs.items():
            if key == "type":
                continue
            stat_name = key.removesuffix("_delta")
            stats.append({
                "name": stat_name,
                "label": _STAT_DISPLAY.get(stat_name, stat_name),
                "original": orig.get(stat_name),
                "synthetic": syn.get(stat_name),
                "delta": val,
                "primary": stat_name in _PRIMARY_STATS,
            })

        result[col_name] = {"type": col_type, "stats": stats}
    return result


_MAX_HISTOGRAM_BINS = 20


def _thin_bins(bins: list[dict[str, Any]], max_bins: int) -> list[dict[str, Any]]:
    """Keep only the top N bins by total count for readable charts.

    Sorts the result back by breakpoint to preserve x-axis order.
    Bins with zero counts on both sides are dropped first.
    """
    if len(bins) <= max_bins:
        return bins
    non_empty = [b for b in bins if b["original"] + b["synthetic"] > 0]
    if len(non_empty) <= max_bins:
        return non_empty
    ranked = sorted(non_empty, key=lambda b: b["original"] + b["synthetic"], reverse=True)
    selected = ranked[:max_bins]
    return sorted(selected, key=lambda b: b["breakpoint"])


def _prepare_distribution_charts(
    overlays: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Render distribution overlay charts per column. Returns {col_name: html}."""
    charts: dict[str, str] = {}
    for idx, (col_name, data) in enumerate(overlays.items()):
        col_type = data["type"]
        overlay = data["overlay"]
        div_id = f"dist_{idx}"

        if col_type == "NUMERIC":
            charts[col_name] = overlay_histogram(
                div_id, _thin_bins(overlay, _MAX_HISTOGRAM_BINS),
                title=f"{col_name} — Distribution",
            )
        elif col_type == "CATEGORICAL":
            charts[col_name] = grouped_bar(
                div_id, overlay, title=f"{col_name} — Top Values"
            )
    return charts


_DIFF_RANGE: dict[str, tuple[float, float]] = {
    "pearson": (-2.0, 2.0),
    "spearman": (-2.0, 2.0),
    "kendall": (-2.0, 2.0),
    "cramers_v": (-1.0, 1.0),
    "phik": (-1.0, 1.0),
}

_RAW_RANGE: dict[str, tuple[float, float]] = {
    "pearson": (-1.0, 1.0),
    "spearman": (-1.0, 1.0),
    "kendall": (-1.0, 1.0),
    "cramers_v": (0.0, 1.0),
    "phik": (0.0, 1.0),
}


def _df_to_heatmap_data(
    df: pl.DataFrame,
) -> tuple[list[str], list[str], list[list[float]]]:
    """Convert a correlation DataFrame to labels + 2D data for heatmap."""
    labels = df["column"].to_list()
    numeric_cols = [c for c in df.columns if c != "column"]
    data: list[list[float]] = []
    for row in df.iter_rows(named=True):
        data.append([float(row[c]) if row[c] is not None else 0.0 for c in numeric_cols])
    return labels, numeric_cols, data


def _prepare_correlation_charts(
    correlation_diffs: dict[str, pl.DataFrame],
    correlation_matrices: dict[str, dict[str, pl.DataFrame]],
) -> list[dict[str, Any]]:
    """Build Original / Synthetic / Diff heatmaps per correlation method."""
    charts: list[dict[str, Any]] = []
    for method, diff_df in correlation_diffs.items():
        display_name = method.replace("_", " ").title()
        diff_range = _DIFF_RANGE.get(method, (-1.0, 1.0))
        raw_range = _RAW_RANGE.get(method, (-1.0, 1.0))
        matrices = correlation_matrices.get(method, {})

        labels, cols, diff_data = _df_to_heatmap_data(diff_df)
        diff_html = diff_heatmap(
            f"corr_{method}_diff", cols, labels, diff_data,
            title=f"{display_name} — Diff", value_range=diff_range,
        )

        orig_html = ""
        if "original" in matrices:
            _, o_cols, o_data = _df_to_heatmap_data(matrices["original"])
            orig_html = diff_heatmap(
                f"corr_{method}_orig", o_cols, labels, o_data,
                title=f"{display_name} — Original", value_range=raw_range,
            )

        syn_html = ""
        if "synthetic" in matrices:
            _, s_cols, s_data = _df_to_heatmap_data(matrices["synthetic"])
            syn_html = diff_heatmap(
                f"corr_{method}_syn", s_cols, labels, s_data,
                title=f"{display_name} — Synthetic", value_range=raw_range,
            )

        charts.append({
            "name": display_name,
            "method": method,
            "original_html": orig_html,
            "synthetic_html": syn_html,
            "diff_html": diff_html,
        })
    return charts


def _prepare_interaction_data(
    overlays: dict[str, Any],
) -> dict[str, Any] | None:
    """Convert interaction overlays to template-ready data with JSON strings.

    Returns None when there are no interaction columns to display.
    """
    num_cols = overlays.get("numeric_columns", [])
    cat_cols = overlays.get("categorical_columns", [])
    if not num_cols and not cat_cols:
        return None

    return {
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "orig_numeric_json": json.dumps(
            overlays.get("original_numeric_data", {}), ensure_ascii=False,
        ),
        "syn_numeric_json": json.dumps(
            overlays.get("synthetic_numeric_data", {}), ensure_ascii=False,
        ),
        "orig_boxplot_json": json.dumps(
            overlays.get("original_boxplot_stats", {}), ensure_ascii=False,
        ),
        "syn_boxplot_json": json.dumps(
            overlays.get("synthetic_boxplot_stats", {}), ensure_ascii=False,
        ),
    }
