"""Dataset discovery and persistence pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import openml
import re

from ..dataio import manifest as manifest_io
from ..dataio.storage import atomic_write_npz, dataset_npz_path
from .common import load_yaml_config

LOGGER = logging.getLogger(__name__)

@dataclass
class DatasetFilters:
    min_rows: int
    max_rows: int
    max_features: int
    only_numeric_target: bool
    drop_list: List[str]
    max_datasets: int
    source: str = "openml"


def _load_filters() -> DatasetFilters:
    cfg = load_yaml_config("datasets.yaml")
    return DatasetFilters(
        min_rows=int(cfg.get("min_rows", 0)),
        max_rows=int(cfg.get("max_rows", 100_000)),
        max_features=int(cfg.get("max_features", 100)),
        only_numeric_target=bool(cfg.get("only_numeric_target", True)),
        drop_list=list(cfg.get("drop_list", [])),
        max_datasets=int(cfg.get("max_datasets", 30)),
        source=str(cfg.get("source", "openml")),
    )


def _should_skip(row: pd.Series, filters: DatasetFilters) -> bool:
    nrows = row.get("number_of_instances", np.nan)
    nfeat = row.get("number_of_features", np.nan)

    if not np.isfinite(nrows) or not np.isfinite(nfeat):
        return True

    if int(nrows) < filters.min_rows:
        return True
    if int(nrows) > filters.max_rows:
        return True
    if int(nfeat) > filters.max_features:
        return True
    return False

_NAME_POSITIVE = re.compile(r"(target|^y$|label|response|outcome|score|price|value)", re.I)
_NAME_NEGATIVE = re.compile(r"(future|next|pred|leak|ground[_-]?truth)", re.I)


def _is_datetime_like(s: pd.Series) -> bool:
    if not isinstance(s, pd.Series):
        return False
    if s.dtype.kind in {"M"}:
        return True
    try:
        pd.to_datetime(s.sample(min(len(s), 200), random_state=0), errors="raise", infer_datetime_format=True)
        return True
    except Exception:
        return False


def _is_id_like(s: pd.Series) -> bool:
    # if a column is a counter or mostly unique values, drop it
    n = len(s)
    nunique = s.nunique(dropna=True)
    if n == 0:
        return False
    if nunique >= 0.95 * n:
        return True
    if pd.api.types.is_integer_dtype(s) and s.is_monotonic_increasing:
        return True
    return False


def _score_candidate(colname: str, s: pd.Series, numeric_df: pd.DataFrame) -> float:
    # hard rejections
    if not pd.api.types.is_numeric_dtype(s):
        return -np.inf
    nunq = s.nunique(dropna=True)
    if nunq < 10 or nunq / max(len(s), 1) < 0.02:
        return -np.inf
    if s.isna().mean() > 0.4:
        return -np.inf
    if _is_datetime_like(s) or _is_id_like(s):
        return -np.inf

    score = 0.0
    # name-based hints
    if _NAME_POSITIVE.search(colname):
        score += 2.0
    if _NAME_NEGATIVE.search(colname):
        score -= 1.5

    # inspect correlations
    try:
        corr = numeric_df.corr(numeric_only=True)[colname].abs().sort_values(ascending=False)
        if len(corr) > 1:
            max_single = corr.iloc[1]
            mean_top = corr.iloc[1 : min(6, len(corr))].mean()
            score += float(np.clip(mean_top * 0.5, 0, 1.5))
            if max_single > 0.9:  # near-duplicate of a feature, avoid leakage
                score -= 3.0
    except Exception:
        pass

    return score


def _infer_target_column(df_all: pd.DataFrame) -> Optional[str]:
    """Infer a plausible numeric target from a full dataframe that includes all attributes."""
    numeric_cols = [c for c in df_all.columns if pd.api.types.is_numeric_dtype(df_all[c])]
    if not numeric_cols:
        return None

    numeric_df = df_all[numeric_cols].copy()
    scores = {}
    for c in numeric_cols:
        scores[c] = _score_candidate(c, df_all[c], numeric_df)

    if not scores:
        return None
    # Choose best-scoring column
    best = max(scores, key=scores.get)
    return best if np.isfinite(scores[best]) else None


def _first_numeric_from_list(cand_list: List[str], df_all: pd.DataFrame) -> Optional[str]:
    for name in cand_list:
        if name in df_all.columns and pd.api.types.is_numeric_dtype(df_all[name]):
            return name
    return None


def _target_from_tasks(did: int) -> Optional[str]:
    """Ask OpenML for a Supervised Regression task and use its target."""
    try:
        tasks = openml.tasks.list_tasks(data_id=did, output_format="dataframe")
        if tasks is None or len(tasks) == 0:
            return None
        # Filter to supervised regression
        mask = tasks["task_type"].astype(str).str.contains("Supervised Regression", case=False, na=False)
        reg_tasks = tasks[mask]
        if reg_tasks.empty:
            return None
        # Prefer the task with most runs, then newest
        sort_cols = [c for c in ["NumberOfRuns", "runs"] if c in reg_tasks.columns]
        if sort_cols:
            reg_tasks = reg_tasks.sort_values(by=sort_cols + ["task_id"], ascending=[False] * len(sort_cols) + [True])
        else:
            reg_tasks = reg_tasks.sort_values(by=["task_id"], ascending=True)
        tid = int(reg_tasks.iloc[0]["task_id"])
        task = openml.tasks.get_task(tid)
        return getattr(task, "target_name", None)
    except Exception as e:
        LOGGER.debug("Task lookup failed for did=%s: %s", did, e)
        return None

def _prepare_arrays(
    X_df: pd.DataFrame,
    y_series: pd.Series,
    *,
    impute: bool = True,
) -> tuple[np.ndarray, np.ndarray, bool, np.ndarray, np.ndarray, float, float]:
    """
    Mean-impute and standardize features and target.

    Args:
        X_df: Feature dataframe.
        y_series: Target series.
        impute: Whether to mean-impute missing values.
    Returns:
        Standardized feature array, target array, imputed flag, feature means, feature stds.
    """
    if impute:
        imputer = SimpleImputer(strategy="mean")
        X_proc = imputer.fit_transform(X_df)
        imputed = True
    else:
        data = pd.concat([X_df, y_series], axis=1).dropna()
        X_proc = data.iloc[:, :-1].to_numpy()
        y_series = data.iloc[:, -1]
        imputed = False

    X_scaler = StandardScaler()
    X_std = X_scaler.fit_transform(X_proc)

    y = y_series.to_numpy(dtype=np.float64, copy=True).reshape(-1, 1)

    y_scaler = StandardScaler()
    y_std = y_scaler.fit_transform(y).ravel()

    feature_means = X_scaler.mean_.astype(np.float32)
    feature_stds = X_scaler.scale_.astype(np.float32)
    target_mean = float(y_scaler.mean_)
    target_std = float(y_scaler.scale_)

    return (
        X_std.astype(np.float32),
        y_std.astype(np.float32),
        imputed,
        feature_means,
        feature_stds,
        target_mean,
        target_std,
    )


def run_discovery(
    *,
    max_datasets: Optional[int] = None,
    source: Optional[str] = None,
    force: bool = False,
) -> dict[str, int]:
    """Run dataset discovery and persist artefacts."""
    filters = _load_filters()
    if max_datasets is not None:
        filters.max_datasets = int(max_datasets)
    if source is not None:
        filters.source = source

    if filters.source != "openml":
        raise NotImplementedError("Only OpenML discovery is implemented at present.")

    manifest_existing = manifest_io.read_manifest(status=None)
    seen_ids = set(manifest_existing["dataset_id"]) if not manifest_existing.empty else set()

    # pull metadata
    dataset_list = openml.datasets.list_datasets(output_format="dataframe")
    dataset_list = dataset_list.rename(
        columns={
            "did": "dataset_id",
            "NumberOfInstances": "number_of_instances",
            "NumberOfFeatures": "number_of_features",
            "NumberOfNumericFeatures": "number_of_numeric_features",
            "NumberOfSymbolicFeatures": "number_of_symbolic_features",
            "NumberOfMissingValues": "number_of_missing_values",
            "name": "name",
        }
    )

    # metadata filtering before downloads
    df_meta = dataset_list.copy()

    for c in [
        "dataset_id",
        "number_of_instances",
        "number_of_features",
        "number_of_numeric_features",
        "number_of_symbolic_features",
    ]:
        if c in df_meta.columns:
            df_meta[c] = pd.to_numeric(df_meta[c], errors="coerce")

    df_meta.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_meta = df_meta.dropna(subset=["dataset_id", "number_of_instances", "number_of_features"])

    df_meta["dataset_id"] = df_meta["dataset_id"].astype(int)

    if "number_of_numeric_features" in df_meta.columns:
        df_meta["number_of_numeric_features"] = df_meta["number_of_numeric_features"].fillna(
            df_meta["number_of_features"]
        )
    else:
        df_meta["number_of_numeric_features"] = df_meta["number_of_features"]

    if "number_of_symbolic_features" in df_meta.columns:
        df_meta["number_of_symbolic_features"] = df_meta["number_of_symbolic_features"].fillna(
            df_meta["number_of_features"] - df_meta["number_of_numeric_features"]
        )
    else:
        df_meta["number_of_symbolic_features"] = (
            df_meta["number_of_features"] - df_meta["number_of_numeric_features"]
        )

    df_meta["number_of_symbolic_features"] = df_meta["number_of_symbolic_features"].clip(lower=0)

    drop_set = set(filters.drop_list)

    def _prepass_ok(row: pd.Series) -> bool:
        if f"openml-{int(row['dataset_id'])}" in drop_set:
            return False
        if not force and f"openml-{int(row['dataset_id'])}" in seen_ids:
            return False
        if _should_skip(row, filters):
            return False
        num_num = row.get("number_of_numeric_features", np.nan)
        num_tot = row.get("number_of_features", np.nan)
        if not np.isfinite(num_num) or not np.isfinite(num_tot):
            return False
        if num_num < num_tot:
            return False
        return True

    df_meta = df_meta[df_meta.apply(_prepass_ok, axis=1)]

    records: List[dict] = []
    successes = 0

    iterable = tqdm(df_meta.itertuples(index=False), total=len(df_meta), desc="Discover")

    for row in iterable:
        dataset_id_int = int(row.dataset_id)
        dataset_id = f"openml-{dataset_id_int}"

        manifest_row = {
            "dataset_id": dataset_id,
            "source": "openml",
            "name": row.name,
            "n_rows": int(row.number_of_instances),
            "n_features": int(row.number_of_features),
            "task_type": "regression",
            "target_name": None,
            "status": "skip",
            "error_msg": None,
        }

        try:
            dataset = openml.datasets.get_dataset(dataset_id_int)
            target_name = (dataset.default_target_attribute or "").strip() or None

            task_target = _target_from_tasks(dataset_id_int) if not target_name else None
            if task_target:
                target_name = task_target

            X_all = None
            attr_names = None
            if not target_name:
                X_all, y_unused, categorical_all, attr_names = dataset.get_data(dataset_format="dataframe", target=None)
                inferred = _infer_target_column(X_all)
                if inferred:
                    target_name = inferred

            if not target_name:
                manifest_row["status"] = "skip"
                manifest_row["error_msg"] = "no target found (no default, no task, inference failed)"
                records.append(manifest_row)
                if successes >= filters.max_datasets:
                    break
                continue

            if "," in target_name:
                if X_all is None:
                    X_all, y_unused, categorical_all, attr_names = dataset.get_data(dataset_format="dataframe", target=None)
                names = [t.strip() for t in target_name.split(",")]
                first_numeric = _first_numeric_from_list(names, X_all)
                if first_numeric is None:
                    manifest_row["status"] = "skip"
                    manifest_row["error_msg"] = f"multi-target has no numeric column among {names}"
                    records.append(manifest_row)
                    if successes >= filters.max_datasets:
                        break
                    continue
                target_name = first_numeric

            # load X/y
            X, y, categorical, feature_names = dataset.get_data(dataset_format="dataframe", target=target_name)

            if filters.only_numeric_target and not pd.api.types.is_numeric_dtype(y):
                manifest_row["status"] = "skip"
                manifest_row["error_msg"] = "non-numeric target"
                records.append(manifest_row)
                if successes >= filters.max_datasets:
                    break
                continue

            has_categorical = bool(any(bool(flag) for flag in (categorical or [])))
            if has_categorical or any(not pd.api.types.is_numeric_dtype(X[c]) for c in X.columns):
                manifest_row["status"] = "skip"
                manifest_row["error_msg"] = "categorical features present"
                records.append(manifest_row)
                if successes >= filters.max_datasets:
                    break
                continue

            X_std, y_arr, imputed, feature_means, feature_stds, target_mean, target_std = _prepare_arrays(X, y, impute=True)
            
            # drop zero columns
            nonzero_mask = np.any(X_std != 0.0, axis=0)
            X_std = X_std[:, nonzero_mask]

            # check if sufficient data is left
            if X_std.shape[1] < 10:
                manifest_row["status"] = "skip"
                manifest_row["error_msg"] = f"insufficient columns after preprocessing ({X_std.shape[1]} < 10)"
                records.append(manifest_row)
                if successes >= filters.max_datasets:
                    break
                continue
            data_path = dataset_npz_path(dataset_id)
            atomic_write_npz(
                data_path,
                {
                    "X": X_std,
                    "y": y_arr,
                    "feature_means": feature_means,
                    "feature_stds": feature_stds,
                    "target_mean": np.array([target_mean], dtype=np.float32),
                    "target_std": np.array([target_std], dtype=np.float32),
                    "feature_names": np.asarray(feature_names, dtype="U128"),
                    "target_name": np.asarray([target_name], dtype="U128"),
                },
            )
            manifest_row.update(
                {
                    "status": "ok",
                    "target_name": target_name,
                    "error_msg": None,
                    "imputed_mean": imputed,
                }
            )
            successes += 1
            records.append(manifest_row)

        except Exception as exc:
            LOGGER.exception("Failed to process dataset %s", dataset_id)
            manifest_row["status"] = "skip"
            manifest_row["error_msg"] = f"exception: {type(exc).__name__}: {exc}"
            records.append(manifest_row)

        if successes >= filters.max_datasets:
            break

    if records:
        manifest_io.upsert_manifest_rows(records)

    return {"discovered": successes, "total_records": len(records)}
