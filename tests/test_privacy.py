from __future__ import annotations

import polars as pl

from dataxid_syntheval._encode_privacy import encode_for_privacy
from dataxid_syntheval._privacy import PrivacyResult, compute_privacy


class TestPrivacyMetrics:
    def _make_data(self, seed_train=42, seed_syn=99, seed_hol=7, n=200):
        import random

        rng_t = random.Random(seed_train)
        rng_s = random.Random(seed_syn)
        rng_h = random.Random(seed_hol)
        cats = ["A", "B", "C", "D", "E"]
        training = pl.DataFrame(
            {
                "x1": [rng_t.gauss(0, 1) for _ in range(n)],
                "x2": [rng_t.gauss(5, 2) for _ in range(n)],
                "x3": [rng_t.gauss(-3, 0.5) for _ in range(n)],
                "x4": [rng_t.gauss(10, 3) for _ in range(n)],
                "cat1": [rng_t.choice(cats) for _ in range(n)],
                "cat2": [rng_t.choice(cats) for _ in range(n)],
            }
        )
        synthetic = pl.DataFrame(
            {
                "x1": [rng_s.gauss(0, 1) for _ in range(n)],
                "x2": [rng_s.gauss(5, 2) for _ in range(n)],
                "x3": [rng_s.gauss(-3, 0.5) for _ in range(n)],
                "x4": [rng_s.gauss(10, 3) for _ in range(n)],
                "cat1": [rng_s.choice(cats) for _ in range(n)],
                "cat2": [rng_s.choice(cats) for _ in range(n)],
            }
        )
        holdout = pl.DataFrame(
            {
                "x1": [rng_h.gauss(0, 1) for _ in range(n)],
                "x2": [rng_h.gauss(5, 2) for _ in range(n)],
                "x3": [rng_h.gauss(-3, 0.5) for _ in range(n)],
                "x4": [rng_h.gauss(10, 3) for _ in range(n)],
                "cat1": [rng_h.choice(cats) for _ in range(n)],
                "cat2": [rng_h.choice(cats) for _ in range(n)],
            }
        )
        return training, synthetic, holdout

    def _encode(self, training, synthetic, holdout):
        return encode_for_privacy(
            training=training, synthetic=synthetic, holdout=holdout
        )

    def test_returns_privacy_result(self):
        training, synthetic, holdout = self._make_data()
        emb = self._encode(training, synthetic, holdout)
        result = compute_privacy(emb, max_sample_size=200)
        assert isinstance(result, PrivacyResult)

    def test_dcr_share_near_50_for_good_synthetic(self):
        training, synthetic, holdout = self._make_data()
        emb = self._encode(training, synthetic, holdout)
        result = compute_privacy(emb, max_sample_size=200)
        assert 0.3 <= result.dcr_share <= 0.7

    def test_dcr_share_high_for_memorized_data(self):
        import random

        rng = random.Random(42)
        n = 100
        cats = ["A", "B", "C", "D", "E"]
        training = pl.DataFrame(
            {
                "x1": [rng.gauss(0, 1) for _ in range(n)],
                "x2": [rng.gauss(5, 2) for _ in range(n)],
                "x3": [rng.gauss(-3, 0.5) for _ in range(n)],
                "x4": [rng.gauss(10, 3) for _ in range(n)],
                "cat1": [rng.choice(cats) for _ in range(n)],
                "cat2": [rng.choice(cats) for _ in range(n)],
            }
        )
        synthetic = training.clone()
        rng2 = random.Random(7)
        holdout = pl.DataFrame(
            {
                "x1": [rng2.gauss(0, 1) for _ in range(n)],
                "x2": [rng2.gauss(5, 2) for _ in range(n)],
                "x3": [rng2.gauss(-3, 0.5) for _ in range(n)],
                "x4": [rng2.gauss(10, 3) for _ in range(n)],
                "cat1": [rng2.choice(cats) for _ in range(n)],
                "cat2": [rng2.choice(cats) for _ in range(n)],
            }
        )
        emb = self._encode(training, synthetic, holdout)
        result = compute_privacy(emb, max_sample_size=100)
        assert result.dcr_share > 0.7
        assert result.ims_training > 0.5

    def test_ims_low_for_novel_data(self):
        training, synthetic, holdout = self._make_data(n=500)
        emb = self._encode(training, synthetic, holdout)
        result = compute_privacy(emb, max_sample_size=500)
        assert result.ims_training < 0.05

    def test_nndr_ratio_near_one_for_good_synthetic(self):
        training, synthetic, holdout = self._make_data()
        emb = self._encode(training, synthetic, holdout)
        result = compute_privacy(emb, max_sample_size=200)
        assert 0.5 <= result.nndr_ratio <= 2.0

    def test_assessments_generated(self):
        training, synthetic, holdout = self._make_data()
        emb = self._encode(training, synthetic, holdout)
        result = compute_privacy(emb, max_sample_size=200)
        assert len(result.assessments) == 3
        metrics = [a.metric for a in result.assessments]
        assert "dcr_share" in metrics
        assert "ims" in metrics
        assert "nndr" in metrics

    def test_assessment_text_not_empty(self):
        training, synthetic, holdout = self._make_data()
        emb = self._encode(training, synthetic, holdout)
        result = compute_privacy(emb, max_sample_size=200)
        for a in result.assessments:
            assert len(a.text) > 10

    def test_subsample_applied(self):
        training, synthetic, holdout = self._make_data(n=1000)
        emb = self._encode(training, synthetic, holdout)
        result = compute_privacy(emb, max_sample_size=50)
        assert isinstance(result, PrivacyResult)
