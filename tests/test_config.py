from __future__ import annotations

import pytest

from dataxid_syntheval._config import SynthEvalConfig


class TestSynthEvalConfigDefaults:
    def test_default_title(self):
        cfg = SynthEvalConfig()
        assert cfg.title == "SynthEval Report"

    def test_default_mode(self):
        cfg = SynthEvalConfig()
        assert cfg.mode == "complete"


class TestSynthEvalConfigCustom:
    def test_custom_title(self):
        cfg = SynthEvalConfig(title="My Evaluation")
        assert cfg.title == "My Evaluation"

    def test_overview_mode(self):
        cfg = SynthEvalConfig(mode="overview")
        assert cfg.mode == "overview"


class TestSynthEvalConfigImmutable:
    def test_frozen(self):
        cfg = SynthEvalConfig()
        with pytest.raises(AttributeError):
            cfg.title = "Changed"  # type: ignore[misc]


class TestSynthEvalConfigValidation:
    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            SynthEvalConfig(mode="invalid")


class TestScoringConfig:
    def test_default_scoring_enabled(self):
        cfg = SynthEvalConfig()
        assert cfg.scoring is True
        assert cfg.n_bins == 10
        assert cfg.max_sample_size == 5000
        assert cfg.max_bivariate_pairs is None

    def test_custom_scoring_config(self):
        cfg = SynthEvalConfig(
            scoring=False, n_bins=20, max_sample_size=10000, max_bivariate_pairs=100,
        )
        assert cfg.scoring is False
        assert cfg.n_bins == 20
        assert cfg.max_sample_size == 10000
        assert cfg.max_bivariate_pairs == 100

    def test_invalid_n_bins(self):
        with pytest.raises(ValueError, match="n_bins must be"):
            SynthEvalConfig(n_bins=1)

    def test_invalid_max_sample_size(self):
        with pytest.raises(ValueError, match="max_sample_size must be"):
            SynthEvalConfig(max_sample_size=0)

    def test_max_bivariate_pairs_none_allowed(self):
        cfg = SynthEvalConfig(max_bivariate_pairs=None)
        assert cfg.max_bivariate_pairs is None
