# dataxid-syntheval

[![PyPI version](https://img.shields.io/pypi/v/dataxid-syntheval.svg)](https://pypi.org/project/dataxid-syntheval/)
[![Python versions](https://img.shields.io/pypi/pyversions/dataxid-syntheval.svg)](https://pypi.org/project/dataxid-syntheval/)
[![License](https://img.shields.io/pypi/l/dataxid-syntheval.svg)](https://github.com/dataxid/dataxid-syntheval/blob/main/LICENSE)

Synthetic data quality evaluation — compare original and synthetic datasets with interactive HTML reports.

## Quickstart

```python
import polars as pl
from dataxid_syntheval import SynthEval

original = pl.read_csv("original.csv")
synthetic = pl.read_csv("synthetic.csv")

se = SynthEval(original=original, synthetic=synthetic)
se.to_html("report.html")
```

Programmatic access:

```python
diffs = se.diff
diffs["column_diffs"]          # per-column stat deltas
diffs["alert_diff"]            # new / resolved alerts
diffs["distribution_overlays"] # histogram & frequency overlays
diffs["correlation_diffs"]     # correlation matrix differences
```

## Quality Scores

Compute fidelity and privacy metrics with a holdout dataset:

```python
import polars as pl
from dataxid_syntheval import SynthEval

original = pl.read_csv("train.csv")
synthetic = pl.read_csv("synthetic.csv")
holdout = pl.read_csv("holdout.csv")

se = SynthEval(original=original, synthetic=synthetic, holdout=holdout)

scores = se.scores
scores.fidelity                    # overall fidelity (%)
scores.fidelity_detail.univariate  # univariate accuracy (%)
scores.fidelity_detail.bivariate   # bivariate accuracy (%)
scores.fidelity_detail.per_column  # per-column breakdown

scores.privacy.dcr_share           # DCR share
scores.privacy.ims_training        # identical match share
scores.privacy.nndr_ratio          # NNDR ratio
scores.privacy.assessments         # descriptive text assessments
```

Privacy metrics require a holdout dataset. Without holdout, `se.scores.privacy` is `None`.

## Features

- **Column-level stat comparison** — mean, std, median, min/max, missing %, distinct count and more
- **Alert change detection** — new and resolved data quality alerts between profiles
- **Distribution overlays** — proportion-based histograms and categorical frequency charts for fair comparison across different dataset sizes
- **Correlation matrix diffs** — Pearson, Spearman, Kendall, Cramér's V, Phik
- **Interactive HTML report** — tabbed column comparison, ECharts visualizations, lazy chart rendering
- Built on [dataxid-profiling](https://github.com/dataxid/dataxid-profiling) and Polars

## Installation

```bash
pip install dataxid-syntheval
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Links

- [Changelog](CHANGELOG.md)
- [GitHub Issues](https://github.com/dataxid/dataxid-syntheval/issues)
- [dataxid-profiling](https://github.com/dataxid/dataxid-profiling)

## License

[Apache-2.0](LICENSE)
