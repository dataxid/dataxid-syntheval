"""ECharts-based chart renderers for SynthEval HTML reports.

Reuses brand palette from dataxid-profiling for visual consistency.
"""

from __future__ import annotations

import json
from typing import Any

BRAND_TEAL = "#0d3b3b"
BRAND_CORAL = "#e8845c"
BRAND_PURPLE = "#b06aed"
BRAND_PEACH = "#f4a683"

CHART_HEIGHT = "400px"
HEATMAP_MIN_HEIGHT = 450
HEATMAP_PX_PER_LABEL = 45


def _short_number(val: float | int | str) -> str:
    """Format a numeric label: 55234 → '55.2K', 1200000 → '1.2M'."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return str(val)
    abs_v = abs(v)
    if abs_v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if abs_v >= 1_000:
        return f"{v / 1_000:.1f}K"
    if abs_v == 0 or abs_v >= 1:
        return f"{v:.0f}"
    return f"{v:.2f}"


def _side_by_side_option(
    labels: list[str],
    orig_pcts: list[float],
    syn_pcts: list[float],
    title: str,
    *,
    orig_counts: list[int] | None = None,
    syn_counts: list[int] | None = None,
) -> dict[str, Any]:
    """Shared ECharts option for side-by-side bar charts (proportion %)."""
    n = len(labels)
    rotate = 45 if n > 10 else (30 if n > 5 else 0)
    max_width = 24 if n > 15 else (36 if n > 8 else 50)

    def _series_data(pcts: list[float], counts: list[int] | None) -> list[Any]:
        if counts is None:
            return pcts
        return [{"value": p, "count": c} for p, c in zip(pcts, counts, strict=True)]

    return {
        "title": {"text": title, "left": "center", "top": "2%", "textStyle": {"fontSize": 14}},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"data": ["Original", "Synthetic"], "top": "2%", "right": 10},
        "grid": {"left": "8%", "right": "5%", "bottom": "18%", "top": "16%", "containLabel": True},
        "xAxis": {
            "type": "category",
            "data": labels,
            "axisLabel": {"rotate": rotate, "fontSize": 10, "interval": 0},
        },
        "yAxis": {
            "type": "value",
            "axisLabel": {"formatter": "{value}%"},
        },
        "barGap": "5%",
        "barCategoryGap": "25%",
        "barMaxWidth": max_width,
        "series": [
            {
                "name": "Original",
                "type": "bar",
                "data": _series_data(orig_pcts, orig_counts),
                "itemStyle": {"color": BRAND_TEAL},
            },
            {
                "name": "Synthetic",
                "type": "bar",
                "data": _series_data(syn_pcts, syn_counts),
                "itemStyle": {"color": BRAND_CORAL},
            },
        ],
    }


def overlay_histogram(
    div_id: str,
    bins: list[dict[str, Any]],
    *,
    title: str = "",
    label_key: str = "breakpoint",
) -> str:
    """Render a side-by-side bar chart comparing original vs synthetic distributions.

    *bins* is a list of dicts with keys:
      <label_key>, "original" (pct), "synthetic" (pct),
      optionally "original_count", "synthetic_count".
    """
    labels = [_short_number(b[label_key]) for b in bins]
    orig_pcts = [b["original"] for b in bins]
    syn_pcts = [b["synthetic"] for b in bins]
    first = bins[0] if bins else {}
    o_counts = [b["original_count"] for b in bins] if "original_count" in first else None
    s_counts = [b["synthetic_count"] for b in bins] if "synthetic_count" in first else None

    option = _side_by_side_option(
        labels, orig_pcts, syn_pcts, title,
        orig_counts=o_counts, syn_counts=s_counts,
    )
    return _wrap(div_id, option, CHART_HEIGHT)


def grouped_bar(
    div_id: str,
    items: list[dict[str, Any]],
    *,
    title: str = "",
) -> str:
    """Render a side-by-side bar chart for categorical value comparison.

    *items* is a list of dicts with keys:
      "value", "original" (pct), "synthetic" (pct),
      optionally "original_count", "synthetic_count".
    """
    labels = [str(it["value"]) for it in items]
    orig_pcts = [it["original"] for it in items]
    syn_pcts = [it["synthetic"] for it in items]
    first = items[0] if items else {}
    o_counts = [it["original_count"] for it in items] if "original_count" in first else None
    s_counts = [it["synthetic_count"] for it in items] if "synthetic_count" in first else None

    option = _side_by_side_option(
        labels, orig_pcts, syn_pcts, title,
        orig_counts=o_counts, syn_counts=s_counts,
    )
    return _wrap(div_id, option, CHART_HEIGHT)


def diff_heatmap(
    div_id: str,
    x_labels: list[str],
    y_labels: list[str],
    data: list[list[float]],
    *,
    title: str = "",
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> str:
    """Render a heatmap showing correlation matrix differences.

    *data* is a 2D list (rows × cols) of float diff values.
    """
    flat: list[list[int | float]] = []
    for row_idx, row in enumerate(data):
        for col_idx, val in enumerate(row):
            flat.append([col_idx, row_idx, round(val, 3)])

    option = {
        "title": {"text": title, "left": "center", "textStyle": {"fontSize": 13}},
        "tooltip": {"position": "top"},
        "grid": {"left": "15%", "right": "12%", "bottom": "15%", "top": "15%"},
        "xAxis": {
            "type": "category",
            "data": x_labels,
            "splitArea": {"show": True},
            "axisLabel": {"rotate": 30, "fontSize": 10},
        },
        "yAxis": {
            "type": "category",
            "data": y_labels,
            "splitArea": {"show": True},
            "axisLabel": {"fontSize": 10},
        },
        "visualMap": {
            "min": value_range[0],
            "max": value_range[1],
            "calculable": True,
            "orient": "vertical",
            "right": "2%",
            "top": "center",
            "inRange": {
                "color": [
                    BRAND_TEAL, "#1a6b6b", "#4a9e9e", "#a8d5d5",
                    "#f4e8d0", BRAND_PEACH, BRAND_CORAL, BRAND_PURPLE,
                ],
            },
        },
        "series": [
            {
                "type": "heatmap",
                "data": flat,
                "label": {"show": True, "fontSize": 10},
                "emphasis": {
                    "itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0,0,0,0.5)"}
                },
            }
        ],
    }

    dyn_height = max(HEATMAP_MIN_HEIGHT, len(y_labels) * HEATMAP_PX_PER_LABEL + 100)
    option_json = json.dumps(option, ensure_ascii=False)
    escaped = option_json.replace("&", "&amp;").replace('"', "&quot;")

    return (
        f'<div id="{div_id}" class="echart-lazy" '
        f'style="width:100%;height:{dyn_height}px" '
        f'data-option="{escaped}" data-has-fn="1"></div>'
    )


def _wrap(div_id: str, option: dict[str, Any], height: str) -> str:
    option_json = json.dumps(option, ensure_ascii=False)
    escaped = option_json.replace("&", "&amp;").replace('"', "&quot;")
    return (
        f'<div id="{div_id}" class="echart-lazy" '
        f'style="width:100%;height:{height}" '
        f'data-option="{escaped}" data-has-bar-tooltip="1"></div>'
    )
