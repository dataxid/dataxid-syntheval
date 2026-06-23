from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

from dataxid_syntheval._discretize import discretize_dataframes
from dataxid_syntheval._encode_privacy import encode_for_privacy
from dataxid_syntheval._fidelity import FidelityDetail, compute_fidelity
from dataxid_syntheval._privacy import PrivacyResult, compute_privacy


@dataclass(frozen=True)
class ScoreResult:
    fidelity: float
    fidelity_detail: FidelityDetail
    privacy: PrivacyResult | None
    overall: float


def compute_scores(
    *,
    training: pl.DataFrame,
    synthetic: pl.DataFrame,
    holdout: pl.DataFrame | None,
    n_bins: int,
    max_sample_size: int,
    max_bivariate_pairs: int | None,
) -> ScoreResult:
    disc = discretize_dataframes(
        training=training, synthetic=synthetic, holdout=holdout, n_bins=n_bins
    )

    fidelity_detail = compute_fidelity(disc, max_bivariate_pairs=max_bivariate_pairs)
    fidelity_score = round((fidelity_detail.univariate + fidelity_detail.bivariate) / 2, 2)

    privacy: PrivacyResult | None = None
    if holdout is not None:
        embeddings = encode_for_privacy(
            training=training, synthetic=synthetic, holdout=holdout
        )
        privacy = compute_privacy(embeddings, max_sample_size=max_sample_size)

    return ScoreResult(
        fidelity=fidelity_score,
        fidelity_detail=fidelity_detail,
        privacy=privacy,
        overall=fidelity_score,
    )
