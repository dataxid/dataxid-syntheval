"""Fidelity scoring via Total Variation Distance on discretized distributions.

Reference: Platzer & Reutterer (2021), Frontiers in Big Data, 4:679939.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

    from dataxid_syntheval._discretize import DiscretizedData


@dataclass(frozen=True)
class FidelityDetail:
    univariate: float
    bivariate: float
    univariate_max: float | None
    bivariate_max: float | None
    per_column: dict[str, float]


def compute_fidelity(
    disc: DiscretizedData, *, max_bivariate_pairs: int | None
) -> FidelityDetail:
    columns = disc.training.columns

    uni_scores = _univariate_accuracy(disc.training, disc.synthetic, columns)
    univariate = _mean(list(uni_scores.values()))

    pairs = _select_pairs(columns, max_bivariate_pairs)
    bivariate = _bivariate_accuracy(disc.training, disc.synthetic, pairs)

    univariate_max: float | None = None
    bivariate_max: float | None = None
    if disc.holdout is not None:
        hol_uni = _univariate_accuracy(disc.training, disc.holdout, columns)
        univariate_max = _mean(list(hol_uni.values()))
        bivariate_max = _bivariate_accuracy(disc.training, disc.holdout, pairs)

    return FidelityDetail(
        univariate=round(univariate, 2),
        bivariate=round(bivariate, 2),
        univariate_max=round(univariate_max, 2) if univariate_max is not None else None,
        bivariate_max=round(bivariate_max, 2) if bivariate_max is not None else None,
        per_column={k: round(v, 2) for k, v in uni_scores.items()},
    )


def _univariate_accuracy(
    reference: pl.DataFrame, target: pl.DataFrame, columns: list[str]
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for col in columns:
        ref_freq = _frequency(reference[col])
        tgt_freq = _frequency(target[col])
        tvd = _total_variation_distance(ref_freq, tgt_freq)
        scores[col] = (1.0 - tvd) * 100.0
    return scores


def _bivariate_accuracy(
    reference: pl.DataFrame, target: pl.DataFrame, pairs: list[tuple[str, str]]
) -> float:
    if not pairs:
        return 100.0
    scores: list[float] = []
    for c1, c2 in pairs:
        ref_freq = _joint_frequency(reference, c1, c2)
        tgt_freq = _joint_frequency(target, c1, c2)
        tvd = _total_variation_distance(ref_freq, tgt_freq)
        scores.append((1.0 - tvd) * 100.0)
    return _mean(scores)


def _select_pairs(
    columns: list[str], max_pairs: int | None
) -> list[tuple[str, str]]:
    all_pairs = list(combinations(columns, 2))
    cap = max_pairs if max_pairs is not None else 1000
    return all_pairs[:cap]


def _frequency(series: pl.Series) -> dict[str, float]:
    counts = series.value_counts()
    total = counts["count"].sum()
    if total == 0:
        return {}
    return {
        row[series.name]: row["count"] / total
        for row in counts.iter_rows(named=True)
    }


def _joint_frequency(df: pl.DataFrame, c1: str, c2: str) -> dict[str, float]:
    joint = df.select(c1, c2).group_by([c1, c2]).len()
    total = joint["len"].sum()
    if total == 0:
        return {}
    return {
        f"{row[c1]}|{row[c2]}": row["len"] / total
        for row in joint.iter_rows(named=True)
    }


def _total_variation_distance(p: dict[str, float], q: dict[str, float]) -> float:
    all_keys = set(p.keys()) | set(q.keys())
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in all_keys)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
