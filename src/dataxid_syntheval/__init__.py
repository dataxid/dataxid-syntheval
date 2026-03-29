"""Synthetic data quality evaluation with Polars-native performance."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dataxid_syntheval._config import SynthEvalConfig
from dataxid_syntheval._ingest import ingest

if TYPE_CHECKING:
    from dataxid_profiling import ProfileReport

__version__ = "0.1.0"

__all__ = ["SynthEval", "SynthEvalConfig"]


class SynthEval:
    """Main entry point for synthetic data evaluation.

    Usage:
        report = SynthEval(original=real_df, synthetic=syn_df)
        report = SynthEval(original=real_profile, synthetic=syn_profile)
    """

    def __init__(
        self,
        original: Any,
        synthetic: Any,
        *,
        config: SynthEvalConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            self._config = SynthEvalConfig(**kwargs)

        self._original_report: ProfileReport = ingest(original, self._config)
        self._synthetic_report: ProfileReport = ingest(synthetic, self._config)

    @property
    def config(self) -> SynthEvalConfig:
        return self._config

    @property
    def original_report(self) -> ProfileReport:
        return self._original_report

    @property
    def synthetic_report(self) -> ProfileReport:
        return self._synthetic_report

    def __repr__(self) -> str:
        return (
            f"SynthEval("
            f"title='{self._config.title}', "
            f"original_rows={self._original_report.df.height}, "
            f"synthetic_rows={self._synthetic_report.df.height})"
        )
