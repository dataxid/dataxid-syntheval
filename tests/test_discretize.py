from __future__ import annotations

import polars as pl

from dataxid_syntheval._discretize import discretize_dataframes

OTHER = "(other)"


class TestDiscretize:
    def test_numeric_columns_binned(self):
        df = pl.DataFrame({"age": [18, 25, 35, 45, 55, 65]})
        result = discretize_dataframes(
            training=df, synthetic=df, holdout=None,
            n_bins_univariate=5, n_bins_bivariate=5,
        )
        assert result.univariate.training.shape == (6, 1)
        assert result.univariate.training["age"].dtype == pl.Utf8

    def test_categorical_passthrough(self):
        df = pl.DataFrame({"city": ["A", "B", "C", "A", "B", "C"]})
        result = discretize_dataframes(
            training=df, synthetic=df, holdout=None,
            n_bins_univariate=10, n_bins_bivariate=10,
        )
        assert result.univariate.training["city"].to_list() == [
            "A", "B", "C", "A", "B", "C"
        ]

    def test_boolean_passthrough(self):
        df = pl.DataFrame({"flag": [True, False, True, False]})
        result = discretize_dataframes(
            training=df, synthetic=df, holdout=None,
            n_bins_univariate=10, n_bins_bivariate=10,
        )
        assert result.univariate.training["flag"].dtype == pl.Utf8

    def test_bin_edges_from_training(self):
        training = pl.DataFrame({"x": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]})
        synthetic = pl.DataFrame({"x": [55.0, 60.0]})
        result = discretize_dataframes(
            training=training, synthetic=synthetic, holdout=None,
            n_bins_univariate=5, n_bins_bivariate=5,
        )
        assert result.univariate.synthetic.shape == (2, 1)

    def test_holdout_discretized_when_provided(self):
        df = pl.DataFrame({"val": [1, 2, 3, 4, 5]})
        result = discretize_dataframes(
            training=df, synthetic=df, holdout=df,
            n_bins_univariate=5, n_bins_bivariate=5,
        )
        assert result.univariate.holdout is not None
        assert result.univariate.holdout.shape == (5, 1)

    def test_holdout_none_when_not_provided(self):
        df = pl.DataFrame({"val": [1, 2, 3, 4, 5]})
        result = discretize_dataframes(
            training=df, synthetic=df, holdout=None,
            n_bins_univariate=5, n_bins_bivariate=5,
        )
        assert result.univariate.holdout is None

    def test_null_handling(self):
        df = pl.DataFrame({"x": [1.0, None, 3.0, None, 5.0]})
        result = discretize_dataframes(
            training=df, synthetic=df, holdout=None,
            n_bins_univariate=3, n_bins_bivariate=3,
        )
        assert result.univariate.training.shape == (5, 1)

    def test_shared_columns_only(self):
        training = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        synthetic = pl.DataFrame({"a": [1, 2], "b": [3, 4], "d": [7, 8]})
        result = discretize_dataframes(
            training=training, synthetic=synthetic, holdout=None,
            n_bins_univariate=3, n_bins_bivariate=3,
        )
        assert list(result.univariate.training.columns) == ["a", "b"]
        assert list(result.univariate.synthetic.columns) == ["a", "b"]


class TestQuantileBinning:
    def test_quantile_edges_spread_skewed_data(self):
        # 9 values clustered low + 1 extreme outlier.
        df = pl.DataFrame({"x": [10, 12, 14, 15, 18, 20, 22, 25, 30, 1000]})
        result = discretize_dataframes(
            training=df, synthetic=df, holdout=None,
            n_bins_univariate=10, n_bins_bivariate=10,
        )
        # Equal-width would put 9/10 values in a single bin (~2 distinct bins).
        # Quantile binning spreads the low cluster across many distinct bins.
        distinct_bins = result.univariate.training["x"].n_unique()
        assert distinct_bins >= 4

    def test_collapsed_quantiles_do_not_crash(self):
        # Almost all identical values: quantiles collapse to < 2 distinct edges,
        # so the column is not registered for binning and passes through safely.
        df = pl.DataFrame({"x": [1, 1, 1, 1, 1, 1, 1, 1, 2, 100]})
        result = discretize_dataframes(
            training=df, synthetic=df, holdout=None,
            n_bins_univariate=10, n_bins_bivariate=10,
        )
        assert result.univariate.training.shape == (10, 1)
        assert result.univariate.training["x"].dtype == pl.Utf8


class TestOtherBucket:
    def test_out_of_range_synthetic_goes_to_other(self):
        training = pl.DataFrame({"x": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]})
        synthetic = pl.DataFrame({"x": [25.0, 9999.0, -5.0]})
        result = discretize_dataframes(
            training=training, synthetic=synthetic, holdout=None,
            n_bins_univariate=5, n_bins_bivariate=5,
        )
        vals = result.univariate.synthetic["x"].to_list()
        assert vals.count(OTHER) == 2  # 9999 (above max) and -5 (below min)
        assert "__NULL__" not in vals

    def test_categorical_top_c_merged_to_other(self):
        training = pl.DataFrame(
            {"c": ["a"] * 10 + ["b"] * 8 + ["c"] * 6 + ["d"] * 2 + ["e"] * 1 + ["f"] * 1}
        )
        result = discretize_dataframes(
            training=training, synthetic=training, holdout=None,
            n_bins_univariate=3, n_bins_bivariate=3,
        )
        kept = set(result.univariate.training["c"].to_list())
        assert {"a", "b", "c"}.issubset(kept)
        assert "d" not in kept and "e" not in kept and "f" not in kept
        assert OTHER in kept

    def test_missing_stays_null_not_other(self):
        df = pl.DataFrame({"x": [1.0, None, 3.0, None, 5.0]})
        result = discretize_dataframes(
            training=df, synthetic=df, holdout=None,
            n_bins_univariate=3, n_bins_bivariate=3,
        )
        vals = result.univariate.training["x"].to_list()
        assert "__NULL__" in vals
        assert OTHER not in vals

    def test_boolean_values_roundtrip_not_collapsed(self):
        df = pl.DataFrame({"flag": [True, False, True, False, True]})
        result = discretize_dataframes(
            training=df, synthetic=df, holdout=None,
            n_bins_univariate=5, n_bins_bivariate=5,
        )
        vals = set(result.univariate.training["flag"].to_list())
        # Both boolean values are kept (not collapsed to (other)).
        assert OTHER not in vals
        assert vals == {"true", "false"}

    def test_holdout_out_of_range_goes_to_other(self):
        training = pl.DataFrame({"x": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]})
        holdout = pl.DataFrame({"x": [25.0, 9999.0]})
        result = discretize_dataframes(
            training=training, synthetic=training, holdout=holdout,
            n_bins_univariate=5, n_bins_bivariate=5,
        )
        assert result.univariate.holdout is not None
        assert result.univariate.holdout["x"].to_list().count(OTHER) == 1


class TestPerLevelDiscretization:
    def test_returns_univariate_and_bivariate_frames(self):
        df = pl.DataFrame({"x": list(range(200))})
        result = discretize_dataframes(
            training=df, synthetic=df, holdout=None,
            n_bins_univariate=100, n_bins_bivariate=10,
        )
        uni_bins = result.univariate.training["x"].n_unique()
        biv_bins = result.bivariate.training["x"].n_unique()
        assert uni_bins > biv_bins
