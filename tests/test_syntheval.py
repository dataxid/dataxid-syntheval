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
