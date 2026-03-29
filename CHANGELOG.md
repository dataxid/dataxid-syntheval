# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-03-29

### Added

- `SynthEval` class — dual-input (DataFrame or ProfileReport), lazy diff computation
- `SynthEval.diff` — column-level stat deltas, alert change detection, distribution overlays, correlation matrix diffs
- `SynthEval.to_html()` — self-contained interactive HTML report with ECharts
- Proportion-based distribution charts for fair comparison across different dataset sizes
- Tabbed column comparison with side-by-side bar charts and "Show more" stat toggle
- Correlation diff heatmaps for Pearson, Spearman, Kendall, Cramér's V and Phik
- Alert change section with trigger values
- Lazy chart initialization for correct rendering in tabbed layouts
- `SynthEvalConfig` — frozen dataclass for evaluation configuration
- `ingest()` — dual-input: accepts `pl.DataFrame` or `ProfileReport`
- 83 tests covering config, ingest, diff engine, charts, HTML report and SynthEval integration
- GitHub templates — CODEOWNERS, PR template, issue templates
