from __future__ import annotations

import polars as pl

from dataxid_syntheval._score import ScoreResult, compute_scores


class TestScoreOrchestration:
    def _make_dfs(self, seed_t=42, seed_s=99, seed_h=7, n=200):
        import random

        rng_t = random.Random(seed_t)
        rng_s = random.Random(seed_s)
        rng_h = random.Random(seed_h)

        def make(rng):
            return pl.DataFrame({
                "x": [rng.gauss(0, 1) for _ in range(n)],
                "y": [rng.gauss(5, 2) for _ in range(n)],
                "cat": [rng.choice(["A", "B", "C", "D"]) for _ in range(n)],
            })

        return make(rng_t), make(rng_s), make(rng_h)

    def test_returns_score_result(self):
        training, synthetic, holdout = self._make_dfs()
        result = compute_scores(
            training=training, synthetic=synthetic, holdout=holdout,
            n_bins=10, max_sample_size=200, max_bivariate_pairs=None,
        )
        assert isinstance(result, ScoreResult)

    def test_fidelity_in_range(self):
        training, synthetic, holdout = self._make_dfs()
        result = compute_scores(
            training=training, synthetic=synthetic, holdout=holdout,
            n_bins=10, max_sample_size=200, max_bivariate_pairs=None,
        )
        assert 0.0 <= result.fidelity <= 100.0

    def test_overall_equals_fidelity(self):
        training, synthetic, holdout = self._make_dfs()
        result = compute_scores(
            training=training, synthetic=synthetic, holdout=holdout,
            n_bins=10, max_sample_size=200, max_bivariate_pairs=None,
        )
        assert result.overall == result.fidelity

    def test_privacy_present_with_holdout(self):
        training, synthetic, holdout = self._make_dfs()
        result = compute_scores(
            training=training, synthetic=synthetic, holdout=holdout,
            n_bins=10, max_sample_size=200, max_bivariate_pairs=None,
        )
        assert result.privacy is not None
        assert len(result.privacy.assessments) == 3

    def test_privacy_none_without_holdout(self):
        training, synthetic, _ = self._make_dfs()
        result = compute_scores(
            training=training, synthetic=synthetic, holdout=None,
            n_bins=10, max_sample_size=200, max_bivariate_pairs=None,
        )
        assert result.privacy is None

    def test_fidelity_detail_accessible(self):
        training, synthetic, holdout = self._make_dfs()
        result = compute_scores(
            training=training, synthetic=synthetic, holdout=holdout,
            n_bins=10, max_sample_size=200, max_bivariate_pairs=None,
        )
        assert result.fidelity_detail.univariate > 0
        assert result.fidelity_detail.bivariate > 0
        assert len(result.fidelity_detail.per_column) == 3
