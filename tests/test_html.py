from __future__ import annotations

from typing import TYPE_CHECKING

from dataxid_syntheval._diff import (
    compute_alert_diff,
    compute_column_diffs,
    compute_correlation_diffs,
    compute_distribution_overlays,
)
from dataxid_syntheval._report._html import render_html

if TYPE_CHECKING:
    from dataxid_profiling import ProfileReport


class TestRenderHtml:
    def _render(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ) -> str:
        return render_html(
            title="Test SynthEval Report",
            version="0.1.0",
            column_diffs=compute_column_diffs(original_report, synthetic_report),
            alert_diff=compute_alert_diff(original_report, synthetic_report),
            distribution_overlays=compute_distribution_overlays(
                original_report, synthetic_report
            ),
            correlation_diffs=compute_correlation_diffs(original_report, synthetic_report),
            original_stats=original_report.stats,
            synthetic_stats=synthetic_report.stats,
            original_rows=original_report.df.height,
            synthetic_rows=synthetic_report.df.height,
        )

    def test_returns_html_string(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        html = self._render(original_report, synthetic_report)
        assert isinstance(html, str)
        assert html.startswith("<!DOCTYPE html>")

    def test_contains_title(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        html = self._render(original_report, synthetic_report)
        assert "Test SynthEval Report" in html

    def test_contains_echarts_cdn(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        html = self._render(original_report, synthetic_report)
        assert "echarts" in html

    def test_contains_column_names(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        html = self._render(original_report, synthetic_report)
        assert "age" in html
        assert "income" in html
        assert "city" in html

    def test_contains_brand_colors(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        html = self._render(original_report, synthetic_report)
        assert "#0d3b3b" in html

    def test_contains_logo(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        html = self._render(original_report, synthetic_report)
        assert "data:image/png;base64," in html

    def test_contains_footer(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        html = self._render(original_report, synthetic_report)
        assert "dataxid-syntheval" in html

    def test_contains_distribution_charts(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        html = self._render(original_report, synthetic_report)
        assert "echarts.init" in html
        assert "Original" in html
        assert "Synthetic" in html
