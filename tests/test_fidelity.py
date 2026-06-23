from __future__ import annotations

import polars as pl

from dataxid_syntheval._discretize import discretize_dataframes
from dataxid_syntheval._fidelity import compute_fidelity


class TestUnivariateAccuracy:
    def test_identical_data_high_accuracy(self):
        df = pl.DataFrame({"x": list(range(100)), "cat": ["A", "B"] * 50})
        disc = discretize_dataframes(training=df, synthetic=df, holdout=None, n_bins=10)
        result = compute_fidelity(disc, max_bivariate_pairs=None)
        assert result.univariate >= 95.0

    def test_completely_different_data_low_accuracy(self):
        training = pl.DataFrame({"x": list(range(0, 50))})
        synthetic = pl.DataFrame({"x": list(range(950, 1000))})
        disc = discretize_dataframes(
            training=training, synthetic=synthetic, holdout=None, n_bins=10
        )
        result = compute_fidelity(disc, max_bivariate_pairs=None)
        assert result.univariate < 50.0

    def test_per_column_scores_present(self):
        df = pl.DataFrame({"a": list(range(50)), "b": ["X", "Y"] * 25})
        disc = discretize_dataframes(training=df, synthetic=df, holdout=None, n_bins=5)
        result = compute_fidelity(disc, max_bivariate_pairs=None)
        assert "a" in result.per_column
        assert "b" in result.per_column

    def test_score_range_0_100(self):
        import random

        rng = random.Random(42)
        training = pl.DataFrame({"x": [rng.gauss(0, 1) for _ in range(200)]})
        synthetic = pl.DataFrame({"x": [rng.gauss(0.5, 1.5) for _ in range(200)]})
        disc = discretize_dataframes(
            training=training, synthetic=synthetic, holdout=None, n_bins=10
        )
        result = compute_fidelity(disc, max_bivariate_pairs=None)
        assert 0.0 <= result.univariate <= 100.0


class TestBivariateAccuracy:
    def test_identical_data_high_bivariate(self):
        import random

        rng = random.Random(42)
        n = 200
        df = pl.DataFrame({
            "x": [rng.gauss(0, 1) for _ in range(n)],
            "y": [rng.gauss(0, 1) for _ in range(n)],
        })
        disc = discretize_dataframes(training=df, synthetic=df, holdout=None, n_bins=5)
        result = compute_fidelity(disc, max_bivariate_pairs=None)
        assert result.bivariate >= 90.0

    def test_max_pairs_cap_applied(self):
        import random

        rng = random.Random(42)
        n = 100
        df = pl.DataFrame({f"c{i}": [rng.gauss(0, 1) for _ in range(n)] for i in range(20)})
        disc = discretize_dataframes(training=df, synthetic=df, holdout=None, n_bins=5)
        result = compute_fidelity(disc, max_bivariate_pairs=3)
        assert result.bivariate >= 0.0


class TestFidelityWithHoldout:
    def test_holdout_reference_computed(self):
        import random

        rng = random.Random(42)
        n = 200
        training = pl.DataFrame({"x": [rng.gauss(0, 1) for _ in range(n)]})
        synthetic = pl.DataFrame({"x": [rng.gauss(0.1, 1) for _ in range(n)]})
        holdout = pl.DataFrame({"x": [rng.gauss(0, 1) for _ in range(n)]})
        disc = discretize_dataframes(
            training=training, synthetic=synthetic, holdout=holdout, n_bins=10
        )
        result = compute_fidelity(disc, max_bivariate_pairs=None)
        assert result.univariate_max is not None
        assert result.univariate_max > 0.0

    def test_no_holdout_reference_is_none(self):
        df = pl.DataFrame({"x": list(range(50))})
        disc = discretize_dataframes(training=df, synthetic=df, holdout=None, n_bins=5)
        result = compute_fidelity(disc, max_bivariate_pairs=None)
        assert result.univariate_max is None
        assert result.bivariate_max is None


class TestOverallFidelity:
    def test_overall_is_mean_of_uni_and_bi(self):
        df = pl.DataFrame({"x": list(range(100)), "y": list(range(100))})
        disc = discretize_dataframes(training=df, synthetic=df, holdout=None, n_bins=5)
        result = compute_fidelity(disc, max_bivariate_pairs=None)
        assert abs(result.univariate - result.bivariate) < 20
