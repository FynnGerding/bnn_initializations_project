"""Meta-feature extraction pipeline."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import signal
from pymfe.mfe import MFE

from ..dataio import manifest as manifest_io
from ..dataio.storage import artifacts_path, atomic_write_parquet, load_npz_arrays
from ..features.mfe_extract import compute_mfe
from ..features.meta_utils import mean_abs_corr, nonlinearity_index, outlier_rate_y

LOGGER = logging.getLogger(__name__)

def timeout_handler(signum, frame):
    raise TimeoutError("Execution took longer than 10 seconds")

def _kurtosis(y: np.ndarray) -> float:
    centered = y - np.mean(y)
    m2 = np.mean(centered ** 2)
    if m2 < 1e-12:
        return 0.0
    m4 = np.mean(centered ** 4)
    return float(m4 / (m2 ** 2) - 3.0)


def _skew(y: np.ndarray) -> float:
    centered = y - np.mean(y)
    m2 = np.mean(centered ** 2)
    if m2 < 1e-12:
        return 0.0
    m3 = np.mean(centered ** 3)
    return float(m3 / (m2 ** 1.5))


def _process_dataset(dataset_id: str, force: bool, existing_ids: set[str]) -> dict[str, float] | None:
    if not force and dataset_id in existing_ids:
        LOGGER.info("Skipping %s (already processed). Use --force to recompute.", dataset_id)
        return None

    features = {"dataset_id": dataset_id}
    dataset_path = artifacts_path("datasets", f"{dataset_id}.npz")
    try:
        data = load_npz_arrays(dataset_path)
    except FileNotFoundError:
        LOGGER.error("Dataset arrays not found for %s at %s", dataset_id, dataset_path)
        return None


    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(100)
    try:
        X = np.asarray(data['X'], dtype=np.float64)
        y = np.asarray(data['y'])
        
        mfe = MFE(groups=["general", "statistical", "info-theory"])
        mfe.fit(X, y)
        names, values = mfe.extract()
        features.update({name: float(value) for name, value in zip(names, values)})
    except TimeoutError as e:
        LOGGER.warning("PyMFE timeout for %s: %s", dataset_id, e)
    except Exception as e:
        LOGGER.warning("PyMFE failed for %s: %s", dataset_id, e)
    finally:
        signal.alarm(0) 
    return features


def run_meta_feature_extraction(*, jobs: int = 1, force: bool = False) -> dict[str, int]:
    """Compute meta-features for all datasets marked 'ok'."""
    if jobs > 1:
        LOGGER.warning("Parallel processing disabled. Running sequentially despite jobs=%d.", jobs)
    manifest = manifest_io.read_manifest(status="ok")
    dataset_ids = list(manifest["dataset_id"])

    meta_path = artifacts_path("meta_features.parquet")
    existing = pd.read_parquet(meta_path) if meta_path.exists() else pd.DataFrame()
    existing_ids = set(existing["dataset_id"]) if not existing.empty else set()

    rows: list[dict[str, float]] = []
    for dataset_id in dataset_ids:
        LOGGER.info("Processing %s", dataset_id)
        row = _process_dataset(dataset_id, force, existing_ids)
        if row is not None:
            rows.append(row)

    if not rows:
        return {"updated": 0, "total": len(dataset_ids)}

    df_new = pd.DataFrame(rows)
    if not existing.empty:
        df_combined = pd.concat([existing, df_new], ignore_index=True, sort=False)
    else:
        df_combined = df_new

    df_combined = df_combined.drop_duplicates(subset=["dataset_id"], keep="last")
    atomic_write_parquet(df_combined, meta_path)
    return {"updated": len(rows), "total": len(dataset_ids)}

if __name__ == "__main__":
    run_meta_feature_extraction()
