"""Privacy-specific encoding: QuantileTransformer + Model2Vec + PCA + L2 normalize."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, normalize

from dataxid_syntheval.assets import load_embedder

_NULL_SENTINEL = "__NULL__"
_EMPTY_SENTINEL = "__EMPTY__"
_RARE_SENTINEL = "__RARE__"


@dataclass
class PrivacyEmbeddings:
    training: np.ndarray
    synthetic: np.ndarray
    holdout: np.ndarray | None


def encode_for_privacy(
    *,
    training: pl.DataFrame,
    synthetic: pl.DataFrame,
    holdout: pl.DataFrame | None,
) -> PrivacyEmbeddings:
    """Encode dataframes into L2-normalized vectors for privacy distance calculations."""
    shared_cols = sorted(set(training.columns) & set(synthetic.columns))
    if holdout is not None:
        shared_cols = sorted(set(shared_cols) & set(holdout.columns))

    training = training.select(shared_cols)
    synthetic = synthetic.select(shared_cols)
    holdout_sel = holdout.select(shared_cols) if holdout is not None else None

    numeric_cols = [
        c for c in shared_cols
        if training[c].dtype in (
            pl.Float32, pl.Float64,
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        )
    ]
    string_cols = [c for c in shared_cols if c not in numeric_cols]

    trn_num, syn_num, hol_num = _encode_numerics(
        training, synthetic, holdout_sel, numeric_cols
    )
    trn_str, syn_str, hol_str = _encode_strings(
        training, synthetic, holdout_sel, string_cols
    )

    trn_encoded = np.hstack([trn_num, trn_str]) if trn_str.shape[1] > 0 else trn_num
    syn_encoded = np.hstack([syn_num, syn_str]) if syn_str.shape[1] > 0 else syn_num
    hol_encoded = None
    if hol_num is not None and hol_str is not None:
        hol_encoded = np.hstack([hol_num, hol_str]) if hol_str.shape[1] > 0 else hol_num

    trn_encoded = normalize(trn_encoded, norm="l2")
    syn_encoded = normalize(syn_encoded, norm="l2")
    if hol_encoded is not None:
        hol_encoded = normalize(hol_encoded, norm="l2")

    return PrivacyEmbeddings(
        training=trn_encoded,
        synthetic=syn_encoded,
        holdout=hol_encoded,
    )


def _encode_numerics(
    training: pl.DataFrame,
    synthetic: pl.DataFrame,
    holdout: pl.DataFrame | None,
    cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """QuantileTransformer → uniform [-0.5, 0.5], null → 0.0 + N/A flag."""
    if not cols:
        empty = np.zeros((training.height, 0))
        return empty, np.zeros((synthetic.height, 0)), (
            np.zeros((holdout.height, 0)) if holdout is not None else None
        )

    result_trn: list[np.ndarray] = []
    result_syn: list[np.ndarray] = []
    result_hol: list[np.ndarray] = []

    for col in cols:
        trn_vals = training[col].cast(pl.Float64).to_numpy()
        syn_vals = synthetic[col].cast(pl.Float64).to_numpy()
        hol_vals = holdout[col].cast(pl.Float64).to_numpy() if holdout is not None else None

        trn_mask = np.isnan(trn_vals)
        syn_mask = np.isnan(syn_vals)
        hol_mask = np.isnan(hol_vals) if hol_vals is not None else None

        fit_data = np.concatenate([trn_vals, hol_vals]) if hol_vals is not None else trn_vals
        fit_data = fit_data[~np.isnan(fit_data)]

        n_quantiles = min(100, len(fit_data))
        if n_quantiles < 2 or len(fit_data) == 0:
            result_trn.append(np.zeros((len(trn_vals), 1)))
            result_syn.append(np.zeros((len(syn_vals), 1)))
            if hol_vals is not None:
                result_hol.append(np.zeros((len(hol_vals), 1)))
            continue

        qt = QuantileTransformer(output_distribution="uniform", n_quantiles=n_quantiles)
        qt.fit(fit_data.reshape(-1, 1))

        trn_t = qt.transform(np.nan_to_num(trn_vals, nan=0.0).reshape(-1, 1)) - 0.5
        syn_t = qt.transform(np.nan_to_num(syn_vals, nan=0.0).reshape(-1, 1)) - 0.5

        trn_t[trn_mask] = 0.0
        syn_t[syn_mask] = 0.0

        result_trn.append(trn_t)
        result_syn.append(syn_t)

        if hol_vals is not None:
            hol_t = qt.transform(np.nan_to_num(hol_vals, nan=0.0).reshape(-1, 1)) - 0.5
            hol_t[hol_mask] = 0.0
            result_hol.append(hol_t)

        has_nulls = trn_mask.any() or (hol_mask is not None and hol_mask.any())
        if has_nulls:
            trn_flag = (trn_mask.astype(float) - 0.5).reshape(-1, 1)
            syn_flag = (syn_mask.astype(float) - 0.5).reshape(-1, 1)
            result_trn.append(trn_flag)
            result_syn.append(syn_flag)
            if hol_vals is not None:
                hol_flag = (hol_mask.astype(float) - 0.5).reshape(-1, 1)
                result_hol.append(hol_flag)

    trn_out = np.hstack(result_trn)
    syn_out = np.hstack(result_syn)
    hol_out = np.hstack(result_hol) if result_hol else None

    return trn_out, syn_out, hol_out


def _encode_strings(
    training: pl.DataFrame,
    synthetic: pl.DataFrame,
    holdout: pl.DataFrame | None,
    cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Model2Vec embedding → PCA (2-4 dims per column)."""
    if not cols:
        empty = np.zeros((training.height, 0))
        return empty, np.zeros((synthetic.height, 0)), (
            np.zeros((holdout.height, 0)) if holdout is not None else None
        )

    embedder = load_embedder()

    result_trn: list[np.ndarray] = []
    result_syn: list[np.ndarray] = []
    result_hol: list[np.ndarray] = []

    for col in cols:
        trn_col = training[col].cast(pl.Utf8).fill_null(_NULL_SENTINEL).to_list()
        syn_col = synthetic[col].cast(pl.Utf8).fill_null(_NULL_SENTINEL).to_list()
        hol_col = (
            holdout[col].cast(pl.Utf8).fill_null(_NULL_SENTINEL).to_list()
            if holdout is not None else []
        )

        trn_col = [v if v != "" else _EMPTY_SENTINEL for v in trn_col]
        syn_col = [v if v != "" else _EMPTY_SENTINEL for v in syn_col]
        hol_col = [v if v != "" else _EMPTY_SENTINEL for v in hol_col]

        ori_values = trn_col + hol_col
        unique_vals = list(dict.fromkeys(ori_values))

        syn_col_mapped = [v if v in set(unique_vals) else _RARE_SENTINEL for v in syn_col]

        all_tokens = unique_vals + [_RARE_SENTINEL]
        embeds = embedder.encode(all_tokens)

        n_unique = len(unique_vals)
        if n_unique <= 20:
            dims = 2
        elif n_unique <= 100:
            dims = 3
        else:
            dims = 4
        dims = min(dims, len(all_tokens))

        pca = PCA(n_components=dims)
        pca_embeds = pca.fit_transform(embeds)

        token_to_idx = {t: i for i, t in enumerate(all_tokens)}

        trn_pca = pca_embeds[[token_to_idx[v] for v in trn_col]]
        syn_pca = pca_embeds[[token_to_idx[v] for v in syn_col_mapped]]

        result_trn.append(trn_pca)
        result_syn.append(syn_pca)

        if holdout is not None:
            hol_pca = pca_embeds[[token_to_idx[v] for v in hol_col]]
            result_hol.append(hol_pca)

    trn_out = np.hstack(result_trn)
    syn_out = np.hstack(result_syn)
    hol_out = np.hstack(result_hol) if result_hol else None

    return trn_out, syn_out, hol_out
