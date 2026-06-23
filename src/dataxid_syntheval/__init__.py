"""Synthetic data quality evaluation with Polars-native performance."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from dataxid_syntheval._config import SynthEvalConfig
from dataxid_syntheval._diff import (
    compute_alert_diff,
    compute_column_diffs,
    compute_correlation_diffs,
    compute_correlation_matrices,
    compute_distribution_overlays,
    compute_interaction_overlays,
    compute_overview_diff,
)
from dataxid_syntheval._ingest import ingest
from dataxid_syntheval._privacy import PrivacyAssessment, PrivacyResult
from dataxid_syntheval._report._html import render_html
from dataxid_syntheval._score import ScoreResult, compute_scores

if TYPE_CHECKING:
    from dataxid_profiling import ProfileReport

__version__ = "0.2.0"

__all__ = [
    "SynthEval",
    "SynthEvalConfig",
    "ScoreResult",
    "PrivacyResult",
    "PrivacyAssessment",
    "__version__",
]


class SynthEval:
    """Main entry point for synthetic data evaluation."""

    def __init__(
        self,
        original: Any,
        synthetic: Any,
        *,
        holdout: Any | None = None,
        config: SynthEvalConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            self._config = SynthEvalConfig(**kwargs)

        self._original_report: ProfileReport = ingest(original, self._config)
        self._synthetic_report: ProfileReport = ingest(synthetic, self._config)
        self._holdout_df: pl.DataFrame | None = None
        if holdout is not None:
            if isinstance(holdout, pl.DataFrame):
                self._holdout_df = holdout
            else:
                type_name = type(holdout).__qualname__
                msg = f"holdout must be pl.DataFrame, got {type_name}"
                raise TypeError(msg)

        self._diff_cache: dict[str, Any] | None = None
        self._scores_cache: ScoreResult | None = None

    @property
    def config(self) -> SynthEvalConfig:
        return self._config

    @property
    def original_report(self) -> ProfileReport:
        return self._original_report

    @property
    def synthetic_report(self) -> ProfileReport:
        return self._synthetic_report

    @property
    def scores(self) -> ScoreResult | None:
        """Lazily computed quality scores (fidelity + privacy)."""
        if not self._config.scoring:
            return None
        if self._scores_cache is None:
            self._scores_cache = compute_scores(
                training=self._original_report.df,
                synthetic=self._synthetic_report.df,
                holdout=self._holdout_df,
                n_bins=self._config.n_bins,
                max_sample_size=self._config.max_sample_size,
                max_bivariate_pairs=self._config.max_bivariate_pairs,
            )
        return self._scores_cache

    @property
    def diff(self) -> dict[str, Any]:
        """Lazily computed comparison results."""
        if self._diff_cache is None:
            orig = self._original_report
            syn = self._synthetic_report
            self._diff_cache = {
                "overview_diff": compute_overview_diff(orig, syn),
                "column_diffs": compute_column_diffs(orig, syn),
                "alert_diff": compute_alert_diff(orig, syn),
                "distribution_overlays": compute_distribution_overlays(orig, syn),
                "correlation_diffs": compute_correlation_diffs(orig, syn),
                "correlation_matrices": compute_correlation_matrices(orig, syn),
                "interaction_overlays": compute_interaction_overlays(orig, syn),
            }
        return self._diff_cache

    def to_html(self, path: str | Path | None = None) -> str:
        """Render the comparison as a self-contained HTML report."""
        d = self.diff
        html = render_html(
            title=self._config.title,
            version=__version__,
            overview_diff=d["overview_diff"],
            column_diffs=d["column_diffs"],
            alert_diff=d["alert_diff"],
            distribution_overlays=d["distribution_overlays"],
            correlation_diffs=d["correlation_diffs"],
            correlation_matrices=d["correlation_matrices"],
            interaction_overlays=d["interaction_overlays"],
            original_stats=self._original_report.stats,
            synthetic_stats=self._synthetic_report.stats,
            original_rows=self._original_report.df.height,
            synthetic_rows=self._synthetic_report.df.height,
        )
        if path is not None:
            Path(path).write_text(html, encoding="utf-8")
        return html

    def __repr__(self) -> str:
        return (
            f"SynthEval("
            f"title='{self._config.title}', "
            f"original_rows={self._original_report.df.height}, "
            f"synthetic_rows={self._synthetic_report.df.height})"
        )
