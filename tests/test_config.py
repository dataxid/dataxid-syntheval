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
        assert cfg.n_bins is None
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


class TestNTopValues:
    def test_default(self):
        config = SynthEvalConfig()
        assert config.n_top_values == 50

    def test_custom(self):
        config = SynthEvalConfig(n_top_values=10)
        assert config.n_top_values == 10

    def test_rejects_below_one(self):
        with pytest.raises(ValueError, match="n_top_values must be >= 1"):
            SynthEvalConfig(n_top_values=0)


class TestPerLevelBins:
    def test_defaults_match_paper(self):
        cfg = SynthEvalConfig()
        assert cfg.n_bins_univariate == 100
        assert cfg.n_bins_bivariate == 10

    def test_legacy_n_bins_applies_to_both_when_set(self):
        with pytest.warns(DeprecationWarning):
            cfg = SynthEvalConfig(n_bins=25)
        assert cfg.n_bins_univariate == 25
        assert cfg.n_bins_bivariate == 25

    def test_legacy_n_bins_takes_precedence_over_explicit_per_level(self):
        with pytest.warns(DeprecationWarning):
            cfg = SynthEvalConfig(n_bins=25, n_bins_univariate=80)
        assert cfg.n_bins_univariate == 25

    def test_explicit_per_level_overrides_defaults(self):
        cfg = SynthEvalConfig(n_bins_univariate=80, n_bins_bivariate=8)
        assert cfg.n_bins_univariate == 80
        assert cfg.n_bins_bivariate == 8

    def test_rejects_per_level_below_two(self):
        with pytest.raises(ValueError, match="n_bins_univariate must be >= 2"):
            SynthEvalConfig(n_bins_univariate=1)
        with pytest.raises(ValueError, match="n_bins_bivariate must be >= 2"):
            SynthEvalConfig(n_bins_bivariate=1)
