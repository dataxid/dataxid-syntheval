from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from dataxid_profiling import ProfileReport  # noqa: TC002 — used in isinstance()

from dataxid_syntheval._config import SynthEvalConfig
from dataxid_syntheval._ingest import ingest

if TYPE_CHECKING:
    import polars as pl


class TestDataFrameInput:
    def test_dataframe_returns_profile_report(self, original_df: pl.DataFrame):
        config = SynthEvalConfig()
        result = ingest(original_df, config)
        assert isinstance(result, ProfileReport)

    def test_dataframe_preserves_shape(self, original_df: pl.DataFrame):
        config = SynthEvalConfig()
        result = ingest(original_df, config)
        assert result.df.shape == original_df.shape

    def test_dataframe_preserves_columns(self, original_df: pl.DataFrame):
        config = SynthEvalConfig()
        result = ingest(original_df, config)
        assert result.df.columns == original_df.columns


class TestProfileReportInput:
    def test_profile_report_passthrough(self, original_report: ProfileReport):
        config = SynthEvalConfig()
        result = ingest(original_report, config)
        assert result is original_report

    def test_no_reprofiling(self, original_report: ProfileReport):
        config = SynthEvalConfig()
        result = ingest(original_report, config)
        assert result.config is original_report.config


class TestInvalidInput:
    def test_dict_raises(self):
        config = SynthEvalConfig()
        with pytest.raises(TypeError, match="Unsupported input type"):
            ingest({"a": [1, 2]}, config)

    def test_list_raises(self):
        config = SynthEvalConfig()
        with pytest.raises(TypeError, match="Unsupported input type"):
            ingest([1, 2, 3], config)

    def test_none_raises(self):
        config = SynthEvalConfig()
        with pytest.raises(TypeError, match="Unsupported input type"):
            ingest(None, config)

    def test_string_raises(self):
        config = SynthEvalConfig()
        with pytest.raises(TypeError, match="Unsupported input type"):
            ingest("data.csv", config)
