# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-03-29

### Added

- `SynthEval` class — dual-input (DataFrame or ProfileReport), lazy diff computation
- `SynthEval.diff` — column-level stat deltas, alert change detection, distribution overlays, correlation matrix diffs
- `SynthEval.to_html()` — self-contained interactive HTML report with ECharts
- Dataset overview section with comparative stats (rows, missing %, duplicate %, type distribution)
- Proportion-based distribution charts with trend line overlay and smart Y-axis scaling
- Tabbed column comparison with expanded statistics and "Show more" toggle
- Bivariate interactions — dropdown-based scatter plots and box plots with outlier display
- Correlation comparison — Original, Synthetic, and Diff heatmaps per method (Pearson, Spearman, Kendall, Cramér's V, Phik)
- Alert change section with trigger values
- Lazy chart initialization with resize support for tabbed layouts
- Dynamic chart sizing based on content
- `SynthEvalConfig` — frozen dataclass for evaluation configuration
- `ingest()` — dual-input: accepts `pl.DataFrame` or `ProfileReport`
- 100 tests covering config, ingest, diff engine, charts, HTML report and SynthEval integration
- GitHub templates — CODEOWNERS, PR template, issue templates
