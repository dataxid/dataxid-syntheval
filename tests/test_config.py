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
