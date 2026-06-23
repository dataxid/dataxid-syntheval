from __future__ import annotations

import polars as pl

from dataxid_syntheval._discretize import discretize_dataframes


class TestDiscretize:
    def test_numeric_columns_binned(self):
        df = pl.DataFrame({"age": [18, 25, 35, 45, 55, 65]})
        result = discretize_dataframes(training=df, synthetic=df, holdout=None, n_bins=5)
        assert result.training.shape == (6, 1)
        assert result.training["age"].dtype == pl.Utf8

    def test_categorical_passthrough(self):
        df = pl.DataFrame({"city": ["A", "B", "C", "A", "B", "C"]})
        result = discretize_dataframes(training=df, synthetic=df, holdout=None, n_bins=10)
        assert result.training["city"].to_list() == ["A", "B", "C", "A", "B", "C"]

    def test_boolean_passthrough(self):
        df = pl.DataFrame({"flag": [True, False, True, False]})
        result = discretize_dataframes(training=df, synthetic=df, holdout=None, n_bins=10)
        assert result.training["flag"].dtype == pl.Utf8

    def test_bin_edges_from_training(self):
        training = pl.DataFrame({"x": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]})
        synthetic = pl.DataFrame({"x": [55.0, 60.0]})
        result = discretize_dataframes(
            training=training, synthetic=synthetic, holdout=None, n_bins=5,
        )
        assert result.synthetic.shape == (2, 1)

    def test_holdout_discretized_when_provided(self):
        df = pl.DataFrame({"val": [1, 2, 3, 4, 5]})
        result = discretize_dataframes(training=df, synthetic=df, holdout=df, n_bins=5)
        assert result.holdout is not None
        assert result.holdout.shape == (5, 1)

    def test_holdout_none_when_not_provided(self):
        df = pl.DataFrame({"val": [1, 2, 3, 4, 5]})
        result = discretize_dataframes(training=df, synthetic=df, holdout=None, n_bins=5)
        assert result.holdout is None

    def test_null_handling(self):
        df = pl.DataFrame({"x": [1.0, None, 3.0, None, 5.0]})
        result = discretize_dataframes(training=df, synthetic=df, holdout=None, n_bins=3)
        assert result.training.shape == (5, 1)

    def test_shared_columns_only(self):
        training = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        synthetic = pl.DataFrame({"a": [1, 2], "b": [3, 4], "d": [7, 8]})
        result = discretize_dataframes(
            training=training, synthetic=synthetic, holdout=None, n_bins=3,
        )
        assert list(result.training.columns) == ["a", "b"]
        assert list(result.synthetic.columns) == ["a", "b"]
