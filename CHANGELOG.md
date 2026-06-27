# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-06-27

### Changed

- Fidelity scoring now uses quantile (equal-frequency) binning for numeric columns. Equal-width binning previously inflated fidelity on skewed or outlier-heavy columns; quantile binning reflects true distribution differences.
- Binning cardinality is now per interaction level, using finer bins for univariate than for bivariate comparison.
- Out-of-range synthetic values and rare categories are grouped into an explicit `(other)` bucket; missing values remain a separate bucket.

### Added

- `SynthEvalConfig` fields: `n_bins_univariate`, `n_bins_bivariate`

### Deprecated

- `SynthEvalConfig.n_bins` — use `n_bins_univariate` / `n_bins_bivariate` instead. When set, it applies to both levels and emits a `DeprecationWarning`.

### Fixed

- Categorical distribution overlays now represent the long tail with an aligned "Other" bucket and use non-missing row counts for truthful percentages.

## [0.2.0] - 2026-06-23

### Added

- `SynthEval.scores` property: holdout-based fidelity & privacy quality assessment
- Fidelity scoring: univariate/bivariate TVD accuracy with per-column breakdown
- Privacy scoring: DCR share, IMS, NNDR ratio (L2 distance in continuous embedding space)
- `_encode_privacy` pipeline: QuantileTransformer + Model2Vec (potion-base-8M) + PCA
- Bundled potion-base-8M static embedder for categorical feature encoding
- `SynthEvalConfig` fields: `scoring`, `n_bins`, `max_sample_size`, `max_bivariate_pairs`
- `holdout` parameter on `SynthEval.__init__`
- `THIRD_PARTY_LICENSES` file for bundled assets attribution

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
