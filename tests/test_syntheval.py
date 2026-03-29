from __future__ import annotations

from typing import TYPE_CHECKING

from dataxid_profiling import ProfileReport  # noqa: TC002 — used in isinstance()

from dataxid_syntheval import SynthEval, SynthEvalConfig

if TYPE_CHECKING:
    import polars as pl


class TestSynthEvalFromDataFrames:
    def test_creates_from_dataframes(
        self, original_df: pl.DataFrame, synthetic_df: pl.DataFrame
    ):
        report = SynthEval(original=original_df, synthetic=synthetic_df)
        assert isinstance(report.original_report, ProfileReport)
        assert isinstance(report.synthetic_report, ProfileReport)

    def test_preserves_shape(
        self, original_df: pl.DataFrame, synthetic_df: pl.DataFrame
    ):
        report = SynthEval(original=original_df, synthetic=synthetic_df)
        assert report.original_report.df.shape == original_df.shape
        assert report.synthetic_report.df.shape == synthetic_df.shape

    def test_default_config(
        self, original_df: pl.DataFrame, synthetic_df: pl.DataFrame
    ):
        report = SynthEval(original=original_df, synthetic=synthetic_df)
        assert report.config.title == "SynthEval Report"
        assert report.config.mode == "complete"


class TestSynthEvalFromProfileReports:
    def test_creates_from_reports(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        report = SynthEval(original=original_report, synthetic=synthetic_report)
        assert report.original_report is original_report
        assert report.synthetic_report is synthetic_report


class TestSynthEvalConfig:
    def test_custom_config(
        self, original_df: pl.DataFrame, synthetic_df: pl.DataFrame
    ):
        cfg = SynthEvalConfig(title="Custom", mode="overview")
        report = SynthEval(original=original_df, synthetic=synthetic_df, config=cfg)
        assert report.config.title == "Custom"
        assert report.config.mode == "overview"

    def test_kwargs_forwarded(
        self, original_df: pl.DataFrame, synthetic_df: pl.DataFrame
    ):
        report = SynthEval(
            original=original_df, synthetic=synthetic_df, title="Via Kwargs"
        )
        assert report.config.title == "Via Kwargs"


class TestSynthEvalRepr:
    def test_repr_contains_title(
        self, original_df: pl.DataFrame, synthetic_df: pl.DataFrame
    ):
        report = SynthEval(
            original=original_df, synthetic=synthetic_df, title="Test"
        )
        text = repr(report)
        assert "Test" in text
        assert "original_rows=100" in text
        assert "synthetic_rows=100" in text


class TestSynthEvalDiff:
    def test_diff_returns_dict(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        se = SynthEval(original=original_report, synthetic=synthetic_report)
        d = se.diff
        assert isinstance(d, dict)
        assert "column_diffs" in d
        assert "alert_diff" in d
        assert "distribution_overlays" in d
        assert "correlation_diffs" in d

    def test_diff_is_cached(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        se = SynthEval(original=original_report, synthetic=synthetic_report)
        d1 = se.diff
        d2 = se.diff
        assert d1 is d2

    def test_diff_column_diffs_has_columns(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        se = SynthEval(original=original_report, synthetic=synthetic_report)
        cd = se.diff["column_diffs"]
        assert "age" in cd
        assert "income" in cd
        assert "city" in cd

    def test_diff_alert_diff_structure(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        se = SynthEval(original=original_report, synthetic=synthetic_report)
        ad = se.diff["alert_diff"]
        assert "new" in ad
        assert "resolved" in ad


class TestSynthEvalToHtml:
    def test_to_html_returns_string(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        se = SynthEval(original=original_report, synthetic=synthetic_report)
        html = se.to_html()
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html

    def test_to_html_contains_title(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        se = SynthEval(
            original=original_report, synthetic=synthetic_report, title="My Report"
        )
        html = se.to_html()
        assert "My Report" in html

    def test_to_html_writes_file(
        self, original_report: ProfileReport, synthetic_report: ProfileReport, tmp_path
    ):
        se = SynthEval(original=original_report, synthetic=synthetic_report)
        out = tmp_path / "report.html"
        html = se.to_html(out)
        assert out.exists()
        assert out.read_text(encoding="utf-8") == html

    def test_to_html_contains_charts(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        se = SynthEval(original=original_report, synthetic=synthetic_report)
        html = se.to_html()
        assert "echarts.init" in html
        assert "Original" in html
        assert "Synthetic" in html
