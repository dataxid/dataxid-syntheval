"""Privacy metrics (DCR, NNDR, IMS) with holdout-based comparison.

Reference: Platzer & Reutterer (2021), Frontiers in Big Data, 4:679939.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    from dataxid_syntheval._encode_privacy import PrivacyEmbeddings


@dataclass(frozen=True)
class PrivacyAssessment:
    metric: str
    value: float
    reference: float
    text: str


@dataclass(frozen=True)
class PrivacyResult:
    dcr_share: float
    ims_training: float
    ims_holdout: float
    dcr_training: float
    dcr_holdout: float
    nndr_training: float
    nndr_holdout: float
    nndr_ratio: float
    assessments: list[PrivacyAssessment]


def compute_privacy(embeddings: PrivacyEmbeddings, *, max_sample_size: int) -> PrivacyResult:
    assert embeddings.holdout is not None

    training, synthetic, holdout = _subsample(
        embeddings.training, embeddings.synthetic, embeddings.holdout, max_sample_size
    )

    dcr_syn_trn, nndr_syn_trn = _calculate_dcrs_nndrs(data=training, query=synthetic)
    dcr_syn_hol, nndr_syn_hol = _calculate_dcrs_nndrs(data=holdout, query=synthetic)
    dcr_hol_trn, _ = _calculate_dcrs_nndrs(data=training, query=holdout)

    n = len(synthetic)
    closer = np.sum(dcr_syn_trn < dcr_syn_hol)
    tied = np.sum(dcr_syn_trn == dcr_syn_hol)
    dcr_share = float((closer + 0.5 * tied) / n)

    ims_training = float(np.mean(dcr_syn_trn <= 1e-6))
    ims_holdout = float(np.mean(dcr_hol_trn <= 1e-6))

    dcr_training_avg = float(np.mean(dcr_syn_trn))
    dcr_holdout_avg = float(np.mean(dcr_syn_hol))

    nndr_training = _nndr_aggregate(nndr_syn_trn)
    nndr_holdout = _nndr_aggregate(nndr_syn_hol)
    nndr_ratio = nndr_training / max(nndr_holdout, 1e-10)

    assessments = [
        _assess_dcr_share(dcr_share),
        _assess_ims(ims_training, ims_holdout),
        _assess_nndr(nndr_ratio),
    ]

    return PrivacyResult(
        dcr_share=round(dcr_share, 4),
        ims_training=round(ims_training, 4),
        ims_holdout=round(ims_holdout, 4),
        dcr_training=round(dcr_training_avg, 6),
        dcr_holdout=round(dcr_holdout_avg, 6),
        nndr_training=round(nndr_training, 4),
        nndr_holdout=round(nndr_holdout, 4),
        nndr_ratio=round(nndr_ratio, 4),
        assessments=assessments,
    )


def _calculate_dcrs_nndrs(
    data: np.ndarray, query: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """L2 k=2 nearest neighbors → DCR and per-record NNDR."""
    data = data[data[:, 0].argsort()]

    index = NearestNeighbors(n_neighbors=2, algorithm="auto", metric="l2")
    index.fit(data)
    dcrs, _ = index.kneighbors(query)

    dcr = dcrs[:, 0]
    nndr = (dcrs[:, 0] + 1e-8) / (dcrs[:, 1] + 1e-8)
    return dcr, nndr


def _nndr_aggregate(nndrs: np.ndarray) -> float:
    """10th smallest NNDR value (same as MostlyAI QA)."""
    if len(nndrs) < 10:
        return float(np.sort(nndrs)[0]) if len(nndrs) > 0 else 0.0
    return float(np.sort(nndrs)[9])


def _subsample(
    training: np.ndarray, synthetic: np.ndarray, holdout: np.ndarray, max_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_size = min(len(training), len(synthetic), len(holdout), max_size)
    rng = np.random.default_rng(seed=42)
    if len(training) > min_size:
        training = training[rng.choice(len(training), min_size, replace=False)]
    if len(synthetic) > min_size:
        synthetic = synthetic[rng.choice(len(synthetic), min_size, replace=False)]
    if len(holdout) > min_size:
        holdout = holdout[rng.choice(len(holdout), min_size, replace=False)]
    return training, synthetic, holdout


def _assess_dcr_share(dcr_share: float) -> PrivacyAssessment:
    ref = 0.50
    if dcr_share <= 0.52:
        text = "Synthetic records are equally distant to training and holdout data."
    elif dcr_share <= 0.55:
        text = (
            f"{dcr_share:.0%} of synthetic records are closer to training"
            " — within expected sampling variance."
        )
    elif dcr_share <= 0.65:
        text = (
            f"{dcr_share:.0%} of synthetic records are closer to training"
            " than holdout — suggests some overfitting to training data."
        )
    else:
        text = (
            f"{dcr_share:.0%} of synthetic records are closer to training"
            " than holdout — indicates potential memorization of"
            " training data."
        )
    return PrivacyAssessment("dcr_share", dcr_share, ref, text)


def _assess_ims(ims_train: float, ims_hold: float) -> PrivacyAssessment:
    if ims_train <= ims_hold:
        text = "Identical match rate is within holdout baseline."
    elif ims_train < 0.01:
        text = (
            f"Identical match rate ({ims_train:.2%}) slightly exceeds"
            f" holdout baseline ({ims_hold:.2%})."
        )
    else:
        text = (
            f"Identical match rate ({ims_train:.1%}) exceeds holdout"
            f" baseline ({ims_hold:.1%}) — some synthetic records"
            " replicate training records."
        )
    return PrivacyAssessment("ims", ims_train, ims_hold, text)


def _assess_nndr(nndr_ratio: float) -> PrivacyAssessment:
    ref = 1.0
    delta = (nndr_ratio - 1.0) * 100
    if abs(delta) <= 10:
        text = "NNDR ratio is consistent with holdout baseline."
    elif delta < 0:
        text = (
            f"NNDR ratio ({nndr_ratio:.3f}): synthetic records are"
            f" {abs(delta):.0f}% closer to training data than to holdout."
        )
    else:
        text = (
            f"NNDR ratio ({nndr_ratio:.3f}): synthetic records are"
            f" {abs(delta):.0f}% farther from training data than from holdout."
        )
    return PrivacyAssessment("nndr", nndr_ratio, ref, text)
