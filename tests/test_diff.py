from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl  # noqa: TC002
import pytest

from dataxid_syntheval._diff import (
    compute_alert_diff,
    compute_column_diffs,
    compute_correlation_diffs,
    compute_distribution_overlays,
)

if TYPE_CHECKING:
    from dataxid_profiling import ProfileReport


class TestColumnDiffsNumeric:
    def test_numeric_columns_present(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        diffs = compute_column_diffs(original_report, synthetic_report)
        assert "age" in diffs
        assert "income" in diffs

    def test_numeric_type_label(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        diffs = compute_column_diffs(original_report, synthetic_report)
        assert diffs["age"]["type"] == "NUMERIC"

    def test_numeric_has_mean_delta(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        diffs = compute_column_diffs(original_report, synthetic_report)
        assert "mean_delta" in diffs["age"]
        assert isinstance(diffs["age"]["mean_delta"], float)

    def test_numeric_has_std_delta(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        diffs = compute_column_diffs(original_report, synthetic_report)
        assert "std_delta" in diffs["income"]
        assert isinstance(diffs["income"]["std_delta"], float)

    def test_numeric_delta_sign(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        """Delta = original - synthetic. If original mean > synthetic mean, delta > 0."""
        diffs = compute_column_diffs(original_report, synthetic_report)
        orig_mean = original_report.stats["age"]["mean"]
        syn_mean = synthetic_report.stats["age"]["mean"]
        expected = orig_mean - syn_mean
        assert diffs["age"]["mean_delta"] == pytest.approx(expected)

    def test_numeric_all_expected_fields(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        diffs = compute_column_diffs(original_report, synthetic_report)
        age = diffs["age"]
        for field in ("mean", "std", "median", "min", "max", "missing_pct", "distinct_count"):
            assert f"{field}_delta" in age


class TestColumnDiffsCategorical:
    def test_categorical_columns_present(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        diffs = compute_column_diffs(original_report, synthetic_report)
        assert "city" in diffs
        assert "gender" in diffs

    def test_categorical_type_label(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        diffs = compute_column_diffs(original_report, synthetic_report)
        assert diffs["city"]["type"] == "CATEGORICAL"

    def test_categorical_has_missing_delta(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        diffs = compute_column_diffs(original_report, synthetic_report)
        assert "missing_pct_delta" in diffs["city"]

    def test_categorical_has_distinct_delta(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        diffs = compute_column_diffs(original_report, synthetic_report)
        assert "distinct_count_delta" in diffs["city"]


class TestColumnDiffsBoolean:
    def test_boolean_columns_present(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        diffs = compute_column_diffs(original_report, synthetic_report)
        assert "is_active" in diffs

    def test_boolean_type_label(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        diffs = compute_column_diffs(original_report, synthetic_report)
        assert diffs["is_active"]["type"] == "BOOLEAN"

    def test_boolean_has_true_pct_delta(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        diffs = compute_column_diffs(original_report, synthetic_report)
        assert "true_pct_delta" in diffs["is_active"]
        assert isinstance(diffs["is_active"]["true_pct_delta"], float)


class TestColumnDiffsEdgeCases:
    def test_shared_columns_only(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        """Only columns present in both profiles should appear in diffs."""
        diffs = compute_column_diffs(original_report, synthetic_report)
        orig_cols = set(original_report.stats.keys())
        syn_cols = set(synthetic_report.stats.keys())
        for col in diffs:
            assert col in orig_cols
            assert col in syn_cols


class TestAlertDiff:
    def test_returns_new_and_resolved(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        diff = compute_alert_diff(original_report, synthetic_report)
        assert "new" in diff
        assert "resolved" in diff
        assert isinstance(diff["new"], list)
        assert isinstance(diff["resolved"], list)

    def test_new_alerts_not_in_original(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        diff = compute_alert_diff(original_report, synthetic_report)
        orig_keys = {
            (a.column, a.alert_type.name) for a in original_report.alerts
        }
        for alert in diff["new"]:
            key = (alert["column"], alert["alert_type"])
            assert key not in orig_keys

    def test_resolved_alerts_not_in_synthetic(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        diff = compute_alert_diff(original_report, synthetic_report)
        syn_keys = {
            (a.column, a.alert_type.name) for a in synthetic_report.alerts
        }
        for alert in diff["resolved"]:
            key = (alert["column"], alert["alert_type"])
            assert key not in syn_keys

    def test_identical_profiles_no_diff(self, original_report: ProfileReport):
        diff = compute_alert_diff(original_report, original_report)
        assert diff["new"] == []
        assert diff["resolved"] == []

    def test_alert_serialization(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        diff = compute_alert_diff(original_report, synthetic_report)
        for alert in diff["new"] + diff["resolved"]:
            assert "column" in alert
            assert "alert_type" in alert
            assert isinstance(alert["alert_type"], str)
            assert "value" in alert
            assert "details" in alert


class TestDistributionOverlaysNumeric:
    def test_numeric_columns_have_overlay(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        overlays = compute_distribution_overlays(original_report, synthetic_report)
        assert "age" in overlays
        assert overlays["age"]["type"] == "NUMERIC"

    def test_numeric_overlay_has_breakpoints(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        overlays = compute_distribution_overlays(original_report, synthetic_report)
        bins = overlays["age"]["overlay"]
        assert isinstance(bins, list)
        assert len(bins) > 0
        for b in bins:
            assert "breakpoint" in b
            assert "original" in b
            assert "synthetic" in b

    def test_numeric_overlay_values_are_proportions(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        overlays = compute_distribution_overlays(original_report, synthetic_report)
        for b in overlays["age"]["overlay"]:
            assert isinstance(b["original"], float)
            assert isinstance(b["synthetic"], float)
            assert 0 <= b["original"] <= 100
            assert 0 <= b["synthetic"] <= 100
            assert isinstance(b["original_count"], int)
            assert isinstance(b["synthetic_count"], int)

    def test_numeric_overlay_sorted_by_breakpoint(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        overlays = compute_distribution_overlays(original_report, synthetic_report)
        bps = [b["breakpoint"] for b in overlays["age"]["overlay"]]
        assert bps == sorted(bps)


class TestDistributionOverlaysCategorical:
    def test_categorical_columns_have_overlay(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        overlays = compute_distribution_overlays(original_report, synthetic_report)
        assert "city" in overlays
        assert overlays["city"]["type"] == "CATEGORICAL"

    def test_categorical_overlay_has_values(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        overlays = compute_distribution_overlays(original_report, synthetic_report)
        items = overlays["city"]["overlay"]
        assert isinstance(items, list)
        assert len(items) > 0
        for item in items:
            assert "value" in item
            assert "original" in item
            assert "synthetic" in item

    def test_categorical_overlay_counts_non_negative(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        overlays = compute_distribution_overlays(original_report, synthetic_report)
        for item in overlays["city"]["overlay"]:
            assert item["original"] >= 0
            assert item["synthetic"] >= 0


class TestDistributionOverlaysEdgeCases:
    def test_boolean_excluded(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        """Boolean columns don't have meaningful distribution overlays."""
        overlays = compute_distribution_overlays(original_report, synthetic_report)
        assert "is_active" not in overlays

    def test_identical_profiles_same_counts(self, original_report: ProfileReport):
        overlays = compute_distribution_overlays(original_report, original_report)
        if "age" in overlays:
            for b in overlays["age"]["overlay"]:
                assert b["original"] == b["synthetic"]


class TestCorrelationDiffs:
    def test_returns_dict_of_dataframes(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        diffs = compute_correlation_diffs(original_report, synthetic_report)
        assert isinstance(diffs, dict)
        for method, df in diffs.items():
            assert isinstance(method, str)
            assert isinstance(df, pl.DataFrame)

    def test_shared_methods_only(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        diffs = compute_correlation_diffs(original_report, synthetic_report)
        orig_methods = set(original_report.correlations)
        syn_methods = set(synthetic_report.correlations)
        for method in diffs:
            assert method in orig_methods
            assert method in syn_methods

    def test_diff_has_column_label(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        diffs = compute_correlation_diffs(original_report, synthetic_report)
        for df in diffs.values():
            assert "column" in df.columns

    def test_diff_matrix_is_square(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        diffs = compute_correlation_diffs(original_report, synthetic_report)
        for df in diffs.values():
            numeric_cols = [c for c in df.columns if c != "column"]
            assert df.height == len(numeric_cols)

    def test_identical_profiles_zero_diff(self, original_report: ProfileReport):
        diffs = compute_correlation_diffs(original_report, original_report)
        for df in diffs.values():
            numeric_cols = [c for c in df.columns if c != "column"]
            for col in numeric_cols:
                for val in df[col].to_list():
                    if val is not None:
                        assert val == pytest.approx(0.0, abs=1e-12)

    def test_diagonal_always_zero(
        self, original_report: ProfileReport, synthetic_report: ProfileReport
    ):
        """Diagonal = self-correlation (1.0) - self-correlation (1.0) = 0.0."""
        diffs = compute_correlation_diffs(original_report, synthetic_report)
        for df in diffs.values():
            labels = df["column"].to_list()
            numeric_cols = [c for c in df.columns if c != "column"]
            for i, label in enumerate(labels):
                if label in numeric_cols:
                    val = df[label][i]
                    if val is not None:
                        assert val == pytest.approx(0.0, abs=1e-12)
