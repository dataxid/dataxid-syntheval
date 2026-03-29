"""Synthetic data quality evaluation with Polars-native performance."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from dataxid_syntheval._report._html import render_html

if TYPE_CHECKING:
    from dataxid_profiling import ProfileReport

__version__ = "0.1.0"

__all__ = ["SynthEval", "SynthEvalConfig"]


class SynthEval:
    """Main entry point for synthetic data evaluation.

    Usage:
        se = SynthEval(original=real_df, synthetic=syn_df)
        se.diff                 # programmatic access
        se.to_html("report.html")  # interactive HTML report
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
        self._diff_cache: dict[str, Any] | None = None

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
    def diff(self) -> dict[str, Any]:
        """Lazily computed comparison results.

        Returns a dict with keys: overview_diff, column_diffs, alert_diff,
        distribution_overlays, correlation_diffs, interaction_overlays.
        """
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
        """Render the comparison as a self-contained HTML report.

        If *path* is given the HTML is also written to that file.
        Always returns the HTML string.
        """
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
