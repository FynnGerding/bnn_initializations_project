"""Aggregate metrics and perform analysis."""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ..dataio.storage import artifacts_path, atomic_write_parquet

LOGGER = logging.getLogger(__name__)


_META_IDENTIFIER_COLUMNS = {"dataset_id"}
_RESULT_IDENTIFIER_COLUMNS = {"dataset_id", "split", "prior"}


def _infer_meta_features(
    meta: pd.DataFrame,
    *,
    min_finite_ratio: float = 0.8, # require >=80% usable values
    min_unique_finite: int = 3, # at least 3 distinct finite values
    min_std: float = 1e-12, # exclude near-constant after coercion
    max_abs: float = 1e12, # drop columns with absurd magnitudes
    allow_numeric_strings: bool = False,
) -> list[str]:
    """Select numeric meta-features with robust finite variation (no infs, few NaNs)."""
    feats: list[str] = []

    for col in meta.columns:
        if col in _META_IDENTIFIER_COLUMNS:
            continue

        s = meta[col]

        # Exclude datetime/timedelta explicitly
        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s):
            continue

        # Must be numeric(ish)
        if pd.api.types.is_numeric_dtype(s):
            coerced = pd.to_numeric(s, errors="coerce")  # pd.NA -> NaN
        else:
            if not allow_numeric_strings:
                continue
            coerced = pd.to_numeric(s, errors="coerce")

        arr = coerced.to_numpy(dtype="float64")

        # Reject columns that contain any ±inf at all (strict)
        if np.isinf(arr).any():
            continue

        finite = np.isfinite(arr)
        n = arr.size
        n_finite = int(finite.sum())

        # Require enough finite values
        if n_finite == 0 or (n_finite / max(n, 1)) < min_finite_ratio:
            continue

        vals = arr[finite]

        # Reject absurd magnitudes that tend to destabilize regressors/scalers
        if np.nanmax(np.abs(vals)) > max_abs:
            continue

        # Require real variation among finite values
        if len(np.unique(vals)) < min_unique_finite:
            continue
        if np.nanstd(vals) <= min_std:
            continue

        feats.append(col)

    return feats


def _infer_metric_columns(results: pd.DataFrame) -> list[str]:
    """Pick numeric metric columns from results table."""
    metrics: list[str] = []
    for col in results.columns:
        if col in _RESULT_IDENTIFIER_COLUMNS:
            continue
        series = results[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        if series.notna().sum() == 0:
            continue
        if len(series.unique()) == 1:
            continue
        metrics.append(col)
    return metrics


def _load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_path = artifacts_path("meta_features.parquet")
    results_path = artifacts_path("results.parquet")
    if not meta_path.exists():
        raise FileNotFoundError("Meta-features file not found. Run extract_meta first.")
    if not results_path.exists():
        raise FileNotFoundError("Results file not found. Run train_eval first.")
    return pd.read_parquet(meta_path), pd.read_parquet(results_path)

def _meta_regression(
    meta: pd.DataFrame,
    results: pd.DataFrame,
    features: Sequence[str],
    metrics: Sequence[str],
) -> pd.DataFrame:
    if "nll" not in results.columns or "nll" not in metrics:
        raise ValueError("NLL metric is required for meta-regression.")

    rows: list[dict] = []

    for prior, group in results.groupby("prior", dropna=False):
        group = group[["nll", "dataset_id"]]
        merged = group.merge(meta, on="dataset_id", how="left")

        if merged.empty:
            print("Meta-Regression: No data for prior:", prior)
            continue

        y_all = pd.to_numeric(merged["nll"], errors="coerce").to_numpy(dtype="float64")
        X_all = merged[list(features)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")

        # mask rows where y and ALL features are finite
        finite_rows = np.isfinite(y_all) & np.all(np.isfinite(X_all), axis=1)
        if finite_rows.sum() < 2:
            print(f"Meta-Regression: not enough finite rows for prior: {prior}")
            continue

        y = y_all[finite_rows]
        X = X_all[finite_rows, :]

        # keep columns with at least 2 unique finite values
        kept_cols_idx = []
        kept_cols_names = []
        for j, name in enumerate(features):
            col = X[:, j]
            # guard unique-finite & std
            u = np.unique(col[np.isfinite(col)])
            if u.size < 2:
                continue
            if np.nanstd(col) <= 1e-12:
                continue
            kept_cols_idx.append(j)
            kept_cols_names.append(name)

        if not kept_cols_idx:
            print(f"Meta-Regression: all features constant or invalid for prior: {prior}")
            continue

        X = X[:, kept_cols_idx]

        # standardize columns
        col_std = X.std(axis=0, ddof=0)
        # avoid divide-by-zero
        scale = np.where(col_std > 0, col_std, 1.0)
        Xs = (X - X.mean(axis=0)) / scale

        intercept = np.ones((y.shape[0], 1), dtype=float)
        X_design = np.concatenate([intercept, Xs], axis=1)

        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
        except np.linalg.LinAlgError:
            print("Meta-Regression: LinAlgError for prior:", prior)
            continue

        row = {"prior": prior, "intercept": float(coeffs[0]), "n_samples": int(y.size)}
        # map back to original feature names and scales
        for name, beta_std, s in zip(kept_cols_names, coeffs[1:], scale):
            row[name] = float(beta_std / s)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    ordered = ["prior", "n_samples", "intercept", *features]
    df = df.reindex(columns=[c for c in ordered if c in df.columns])
    return df


def _all_nan_datasets_from_empty_rows(df: pd.DataFrame, id_cols: list[str]) -> set:
        """Return dataset_ids that have any row which is all-NaN across non-id columns."""
        present_id = [c for c in id_cols if c in df.columns]
        value_cols = [c for c in df.columns if c not in present_id]
        if not value_cols or "dataset_id" not in present_id:
            return set()
        mask_id_ok = df["dataset_id"].notna()
        mask_empty = df[value_cols].isna().all(axis=1)
        return set(df.loc[mask_id_ok & mask_empty, "dataset_id"].astype(object))


def _post_clean_datasets(
    meta: pd.DataFrame,
    results: pd.DataFrame,
    *,
    linthresh: float = 1.0,
    k: float = 1.5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop datasets where no prior achieves decent performance, where "decent" is defined
    via a robust cutoff on the global NLL distribution using a symlog transform and
    a Hampel/MAD high-tail rule.

    Also removes any rows with non-finite (NaN/inf) NLL values before analysis.

    A dataset is kept if, for ANY (dataset_id, split), at least one prior has NLL <= cutoff.
    """

    # --- helpers ---
    def symlog(v: np.ndarray, linthresh: float) -> np.ndarray:
        return np.sign(v) * np.log1p(np.abs(v) / linthresh)

    def inv_symlog(z: np.ndarray, linthresh: float) -> np.ndarray:
        return np.sign(z) * (np.expm1(np.abs(z)) * linthresh)
    
    meta_id_cols = ["dataset_id"]
    results_id_cols = ["dataset_id", "split", "prior"]
    
    # check for all-NaN rows and drop implicated datasets
    empty_meta = _all_nan_datasets_from_empty_rows(meta, meta_id_cols)
    empty_results = _all_nan_datasets_from_empty_rows(results, results_id_cols)
    empty_any = empty_meta | empty_results

    if meta.shape[0]:
        meta = meta.loc[~meta.isna().all(axis=1)].copy()
    if results.shape[0]:
        results = results.loc[~results.isna().all(axis=1)].copy()

    if empty_any:
        # remove implicated datasets from both tables
        meta = meta.loc[~meta["dataset_id"].isin(empty_any)].copy()
        results = results.loc[~results["dataset_id"].isin(empty_any)].copy()
        print(f"Dropped {len(empty_any)} dataset(s) due to empty all-NaN row(s) in meta/results.")

    # check NLL presence
    if "nll" not in results.columns:
        raise ValueError("`results` must include an 'nll' column.")

    # drop non-finite NLL rows
    finite_mask = np.isfinite(results["nll"])
    n_dropped_rows = (~finite_mask).sum()
    if n_dropped_rows > 0:
        print(f"Dropped {n_dropped_rows} rows with non-finite NLL values.")
    results = results.loc[finite_mask].copy()

    if results.empty:
        raise ValueError("No results with finite NLL values remain after cleaning.")

    # --- compute robust cutoff from global NLLs ---
    x = results["nll"].to_numpy()
    z = symlog(x, linthresh=linthresh)
    med = np.median(z)
    mad = np.median(np.abs(z - med))
    scale = 1.4826 * (mad + 1e-12)
    thr_z = med + k * scale
    thr_x = float(inv_symlog(np.array([thr_z]), linthresh=linthresh)[0])

    # --- mark “decent” rows (<= cutoff) ---
    decent_mask = results["nll"] <= thr_x
    res_with_flag = results.assign(_decent=decent_mask)

    # keep dataset if ANY prior in ANY split is decent
    decent_by_group = (
        res_with_flag.groupby(["dataset_id", "split"], as_index=False)["_decent"].any()
    )
    decent_datasets = set(decent_by_group.loc[decent_by_group["_decent"], "dataset_id"])

    # --- filter meta/results ---
    cleaned_meta = meta[meta["dataset_id"].isin(decent_datasets)].reset_index(drop=True)
    cleaned_results = results[results["dataset_id"].isin(decent_datasets)].reset_index(drop=True)

    n_dropped_datasets = meta["dataset_id"].nunique() - cleaned_meta["dataset_id"].nunique()
    print(
        f"Dropped {n_dropped_datasets} dataset(s) with no decent performance. "
        f"[Hampel k={k}, linthresh={linthresh}, cutoff≈{thr_x:.6g}]"
    )

    return cleaned_meta, cleaned_results

def _feature_metric_correlations(
    meta: pd.DataFrame,
    results: pd.DataFrame,
    features: Sequence[str],
    metrics: Sequence[str],
    methods: Sequence[str] = ("pearson", "spearman"),
    min_pairs: int = 3,
) -> pd.DataFrame:
    """
    Compute correlations between each (meta-feature, metric) pair for each prior.

    Returns a tidy DataFrame with:
      prior, metric, feature, method, n, r, p_value
    """
    try:
        from scipy import stats as _scistats
    except Exception:
        _scistats = None

    allowed = {"pearson", "spearman"}
    meths = [m.lower() for m in methods if m.lower() in allowed]
    if not meths:
        raise ValueError("No valid correlation methods. Use any of: 'pearson', 'spearman'.")

    rows: list[Dict[str, object]] = []

    res_keep = list(_RESULT_IDENTIFIER_COLUMNS | set(metrics))
    meta_keep = list(_META_IDENTIFIER_COLUMNS | set(features))
    res_use = results[res_keep].copy()
    meta_use = meta[meta_keep].copy()

    for prior, grp in res_use.groupby("prior", dropna=False):
        merged = grp.merge(meta_use, on="dataset_id", how="left", suffixes=("", "_meta"))
        if merged.empty:
            continue

        M = merged[list(metrics)].apply(pd.to_numeric, errors="coerce")
        F = merged[list(features)].apply(pd.to_numeric, errors="coerce")

        # for each metric-feature pair, compute chosen correlations
        for metric in metrics:
            y = M[metric].to_numpy(dtype="float64")
            for feature in features:
                x = F[feature].to_numpy(dtype="float64")

                # finite & variability guards
                finite = np.isfinite(x) & np.isfinite(y)
                if finite.sum() < min_pairs:
                    # not enough data
                    for method in meths:
                        rows.append({
                            "prior": prior,
                            "metric": metric,
                            "feature": feature,
                            "method": method,
                            "n": int(finite.sum()),
                            "r": np.nan,
                            "p_value": np.nan,
                        })
                    continue

                xf = x[finite]
                yf = y[finite]

                if (np.nanstd(xf) <= 0) or (np.nanstd(yf) <= 0):
                    for method in meths:
                        rows.append({
                            "prior": prior,
                            "metric": metric,
                            "feature": feature,
                            "method": method,
                            "n": int(xf.size),
                            "r": np.nan,
                            "p_value": np.nan,
                        })
                    continue

                for method in meths:
                    if method == "pearson":
                        r = float(np.corrcoef(xf, yf)[0, 1])
                        if _scistats is not None and np.isfinite(r):
                            n = xf.size
                            r_clip = np.clip(r, -0.999999999, 0.999999999)
                            t = r_clip * np.sqrt(max(n - 2, 1) / max(1.0 - r_clip**2, 1e-15))
                            p = float(2.0 * (1.0 - _scistats.t.cdf(np.abs(t), df=max(n - 2, 1))))
                        else:
                            p = np.nan
                    elif method == "spearman":
                        xr = pd.Series(xf).rank(method="average").to_numpy(dtype="float64")
                        yr = pd.Series(yf).rank(method="average").to_numpy(dtype="float64")
                        r = float(np.corrcoef(xr, yr)[0, 1])
                        if _scistats is not None and np.isfinite(r):
                            n = xr.size
                            r_clip = np.clip(r, -0.999999999, 0.999999999)
                            t = r_clip * np.sqrt(max(n - 2, 1) / max(1.0 - r_clip**2, 1e-15))
                            p = float(2.0 * (1.0 - _scistats.t.cdf(np.abs(t), df=max(n - 2, 1))))
                        else:
                            p = np.nan
                    else:
                        r, p = np.nan, np.nan

                    rows.append({
                        "prior": prior,
                        "metric": metric,
                        "feature": feature,
                        "method": method,
                        "n": int(xf.size),
                        "r": r,
                        "p_value": p,
                    })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[["prior", "method", "metric", "feature", "n", "r", "p_value"]]
        df = df.sort_values(["prior", "method", "metric", "feature"]).reset_index(drop=True)

    return df

def run_analysis() -> None:
    """Perform post-hoc analysis combining meta-features and results tables."""
    meta, results = _load_tables()

    # drop dataset if no priors achieves decent performance
    meta, results = _post_clean_datasets(meta, results)

    features_for_analysis = _infer_meta_features(meta)

    # drop manually selected uninformative features
    drop = ['g_mean.mean', 'g_mean.sd', 'h_mean.mean', 'h_mean.sd', 'nr_cat', 'cat_to_num', 'num_to_cat', 'sd_ratio']
    features_for_analysis = [f for f in features_for_analysis if f not in drop]

    metric_columns = _infer_metric_columns(results)

    # keep only valid columns
    meta = meta[list(_META_IDENTIFIER_COLUMNS) + features_for_analysis]
    results = results[list(_RESULT_IDENTIFIER_COLUMNS) + metric_columns]

    analysis_dir = artifacts_path("analysis")

    # compute meta-regression
    coefficients_df = _meta_regression(meta, results, features_for_analysis, metric_columns)

    save_path = analysis_dir / "meta_regression_coefficients.csv"

    # if analysis dir doesn't exist, create it
    analysis_dir.mkdir(parents=True, exist_ok=True)

    coefficients_df.to_csv(save_path, index=False)
    LOGGER.info("Saved meta-regression coefficients to %s", save_path)

    corr_df = _feature_metric_correlations(
        meta=meta,
        results=results,
        features=features_for_analysis,
        metrics=metric_columns,
        methods=("pearson", "spearman"),
        min_pairs=3,
    )
    corr_df.to_csv(analysis_dir / "feature_metric_correlations.csv", index=False)
    LOGGER.info("Saved feature-metric correlations to %s", analysis_dir / "feature_metric_correlations.csv")