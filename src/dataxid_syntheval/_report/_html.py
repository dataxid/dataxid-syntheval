"""HTML report rendering via Jinja2."""

from __future__ import annotations

import base64
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
    column_diffs: dict[str, dict[str, Any]],
    alert_diff: dict[str, list[dict[str, Any]]],
    distribution_overlays: dict[str, dict[str, Any]],
    correlation_diffs: dict[str, pl.DataFrame],
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
    corr_charts = _prepare_correlation_diff_charts(correlation_diffs)

    return template.render(
        title=title,
        version=version,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        n_columns=len(column_diffs),
        original_rows=original_rows,
        synthetic_rows=synthetic_rows,
        columns=columns,
        alert_diff=alert_diff,
        distribution_charts=dist_charts,
        correlation_diff_charts=corr_charts,
        logo_b64=_load_asset_b64("dataxid_logo.png"),
        icon_b64=_load_asset_b64("icon.png"),
    )


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
    "median": "Median",
    "min": "Min",
    "max": "Max",
    "range": "Range",
    "q25": "Q25",
    "q75": "Q75",
    "p5": "P5",
    "p95": "P95",
    "iqr": "IQR",
    "skewness": "Skewness",
    "kurtosis": "Kurtosis",
    "missing_pct": "Missing %",
    "distinct_count": "Distinct",
    "distinct_pct": "Distinct %",
    "zeros_pct": "Zeros %",
    "true_pct": "True %",
    "false_pct": "False %",
}

_PRIMARY_STATS = ("mean", "std", "median", "min", "max", "missing_pct")


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


def _prepare_correlation_diff_charts(
    correlation_diffs: dict[str, pl.DataFrame],
) -> list[dict[str, str]]:
    """Build one diff heatmap per correlation method."""
    charts: list[dict[str, str]] = []
    for method, df in correlation_diffs.items():
        labels = df["column"].to_list()
        numeric_cols = [c for c in df.columns if c != "column"]
        data: list[list[float]] = []
        for row in df.iter_rows(named=True):
            data.append([float(row[c]) if row[c] is not None else 0.0 for c in numeric_cols])

        div_id = f"corrdiff_{method}"
        display_name = method.replace("_", " ").title()
        vrange = _DIFF_RANGE.get(method, (-1.0, 1.0))
        chart_html = diff_heatmap(
            div_id, numeric_cols, labels, data,
            title=f"{display_name} Diff", value_range=vrange,
        )
        charts.append({"name": display_name, "div_id": div_id, "chart_html": chart_html})
    return charts
