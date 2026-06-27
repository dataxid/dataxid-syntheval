from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

OTHER_BIN = "(other)"


@dataclass
class LevelFrames:
    training: pl.DataFrame
    synthetic: pl.DataFrame
    holdout: pl.DataFrame | None


@dataclass
class DiscretizedData:
    univariate: LevelFrames
    bivariate: LevelFrames


def discretize_dataframes(
    *,
    training: pl.DataFrame,
    synthetic: pl.DataFrame,
    holdout: pl.DataFrame | None,
    n_bins_univariate: int,
    n_bins_bivariate: int,
) -> DiscretizedData:
    uni = _discretize_one(
        training=training, synthetic=synthetic, holdout=holdout, n_bins=n_bins_univariate
    )
    biv = _discretize_one(
        training=training, synthetic=synthetic, holdout=holdout, n_bins=n_bins_bivariate
    )
    return DiscretizedData(univariate=uni, bivariate=biv)


def _discretize_one(
    *,
    training: pl.DataFrame,
    synthetic: pl.DataFrame,
    holdout: pl.DataFrame | None,
    n_bins: int,
) -> LevelFrames:
    shared_cols = _shared_columns(training, synthetic, holdout)

    training = training.select(shared_cols)
    synthetic = synthetic.select(shared_cols)
    if holdout is not None:
        holdout = holdout.select(shared_cols)

    numeric_types = (
        pl.Float32, pl.Float64,
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    )

    bin_edges: dict[str, list[float]] = {}
    for col in shared_cols:
        if training[col].dtype in numeric_types:
            s = training[col].drop_nulls()
            col_min = s.min()
            col_max = s.max()
            if col_min is not None and col_max is not None and col_min < col_max:
                probs = np.linspace(0.0, 1.0, n_bins + 1)
                quantiles = [s.quantile(float(p)) for p in probs]
                edges = sorted({q for q in quantiles if q is not None})
                if len(edges) >= 2:
                    bin_edges[col] = edges

    top_categories: dict[str, list[Any]] = {}
    for col in shared_cols:
        if training[col].dtype not in numeric_types:
            counts = (
                training.select(col)
                .drop_nulls()
                .group_by(col)
                .len()
                .sort(["len", col], descending=[True, False])
                .head(n_bins)
            )
            top_categories[col] = counts[col].to_list()

    training_disc = _discretize_df(training, bin_edges, top_categories)
    synthetic_disc = _discretize_df(synthetic, bin_edges, top_categories)
    holdout_disc = (
        _discretize_df(holdout, bin_edges, top_categories)
        if holdout is not None
        else None
    )

    return LevelFrames(
        training=training_disc,
        synthetic=synthetic_disc,
        holdout=holdout_disc,
    )


def _shared_columns(
    training: pl.DataFrame,
    synthetic: pl.DataFrame,
    holdout: pl.DataFrame | None,
) -> list[str]:
    cols = set(training.columns) & set(synthetic.columns)
    if holdout is not None:
        cols &= set(holdout.columns)
    return sorted(cols)


def _discretize_df(
    df: pl.DataFrame,
    bin_edges: dict[str, list[float]],
    top_categories: dict[str, list[Any]],
) -> pl.DataFrame:
    result = df.clone()
    for col in result.columns:
        if col in bin_edges:
            edges = bin_edges[col]
            lo, hi = edges[0], edges[-1]
            result = result.with_columns(
                pl.when(pl.col(col).is_null())
                .then(pl.lit("__NULL__"))
                .when((pl.col(col) < lo) | (pl.col(col) > hi))
                .then(pl.lit(OTHER_BIN))
                .otherwise(pl.col(col).cut(edges[1:-1]).cast(pl.Utf8))
                .alias(col)
            )
        elif col in top_categories:
            keep = pl.Series(top_categories[col]).cast(pl.Utf8).to_list()
            result = result.with_columns(
                pl.when(pl.col(col).is_null())
                .then(pl.lit("__NULL__"))
                .when(pl.col(col).cast(pl.Utf8).is_in(keep))
                .then(pl.col(col).cast(pl.Utf8))
                .otherwise(pl.lit(OTHER_BIN))
                .alias(col)
            )
        else:
            result = result.with_columns(
                pl.col(col).cast(pl.Utf8).fill_null("__NULL__").alias(col)
            )
    return result
