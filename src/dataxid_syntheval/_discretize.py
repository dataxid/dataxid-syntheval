from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class DiscretizedData:
    training: pl.DataFrame
    synthetic: pl.DataFrame
    holdout: pl.DataFrame | None


def discretize_dataframes(
    *,
    training: pl.DataFrame,
    synthetic: pl.DataFrame,
    holdout: pl.DataFrame | None,
    n_bins: int,
) -> DiscretizedData:
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
            col_min = training[col].drop_nulls().min()
            col_max = training[col].drop_nulls().max()
            if col_min is not None and col_max is not None and col_min < col_max:
                edges = np.linspace(col_min, col_max, n_bins + 1).tolist()
                bin_edges[col] = edges

    training_disc = _discretize_df(training, bin_edges)
    synthetic_disc = _discretize_df(synthetic, bin_edges)
    holdout_disc = _discretize_df(holdout, bin_edges) if holdout is not None else None

    return DiscretizedData(
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
    df: pl.DataFrame, bin_edges: dict[str, list[float]]
) -> pl.DataFrame:
    result = df.clone()
    for col in result.columns:
        if col in bin_edges:
            edges = bin_edges[col]
            result = result.with_columns(
                pl.col(col)
                .cut(edges[1:-1])
                .cast(pl.Utf8)
                .fill_null("__NULL__")
                .alias(col)
            )
        else:
            result = result.with_columns(
                pl.col(col).cast(pl.Utf8).fill_null("__NULL__").alias(col)
            )
    return result
