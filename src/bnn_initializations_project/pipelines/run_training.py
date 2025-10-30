"""Training and evaluation pipeline for Bayesian neural networks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Sequence

import jax
import jax.numpy as jnp
import jax.random as jrnd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..dataio.storage import atomic_write_parquet
from ..dataio import manifest as manifest_io
from ..dataio.splits import make_splits
from ..dataio.storage import artifacts_path, load_npz_arrays
from ..metrics.predictive import coverage, nll_gaussian, posterior_predictive_draws, rmse
from ..metrics.summary import result_row
from ..models.inference import fit_bnn
from ..models.bnn import Bayesian_MLP
from .common import load_yaml_config

LOGGER = logging.getLogger(__name__)


def _activation_from_config(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    if name in ("tanh", "jax.nn.tanh"):
        return jnp.tanh
    if name in ("relu", "jax.nn.relu"):
        import jax.nn as jnn

        return jnn.relu
    raise ValueError(f"Unsupported activation: {name}")


def _resolve_layer_widths(raw: Sequence[int | str], input_dim: int) -> List[int]:
    widths: List[int] = []
    for item in raw:
        if isinstance(item, str) and item.upper() == "INPUT_DIM":
            widths.append(int(input_dim))
        else:
            widths.append(int(item))
    return widths


@dataclass
class TrainConfig:
    layer_widths: Sequence[int | str]
    activation_name: str
    lik_sigma: float
    prior_scale: float
    init_scheme: str
    num_samples: int
    num_burn_in: int
    step_size: float
    seed: int
    id_split_ratio: float
    ood_tail_fraction: float
    max_ppc_samples: int


def _load_train_config() -> TrainConfig:
    model_cfg = load_yaml_config("model.yaml")
    run_cfg = load_yaml_config("run.yaml")
    priors_cfg = model_cfg.get("priors", [])
    return TrainConfig(
        layer_widths=model_cfg["layer_widths"],
        activation_name=model_cfg.get("activation", "tanh"),
        lik_sigma=float(model_cfg.get("lik_sigma", 1)),
        prior_scale=float(model_cfg.get("prior_scale", 1.0)),
        init_scheme=model_cfg.get("init_scheme", "isotropic_gaussian"),
        num_samples=int(run_cfg.get("num_samples", 50000)),
        num_burn_in=int(run_cfg.get("num_burn_in", 1000)),
        step_size=float(run_cfg.get("step_size", 0.02)),
        seed=int(run_cfg.get("seed", 0)),
        id_split_ratio=float(run_cfg.get("id_split_ratio", 0.7)),
        ood_tail_fraction=float(run_cfg.get("ood_tail_fraction", 0.1)),
        max_ppc_samples=int(run_cfg.get("max_ppc_samples", 256)),
    )


def _posterior_means(samples: jnp.ndarray, unravel_fn, X: np.ndarray, activation) -> jnp.ndarray:
    X_jax = jnp.asarray(X, dtype=jnp.float32)

    def _forward(theta: jnp.ndarray) -> jnp.ndarray:
        params = unravel_fn(theta)
        return Bayesian_MLP.forward(params, X_jax, activation=activation)

    preds = jnp.asarray(jax.vmap(_forward)(samples))
    return jnp.mean(preds, axis=0)


def _evaluate_split(
    split_name: str,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    *,
    samples: jnp.ndarray,
    unravel_fn,
    activation,
    lik_sigma: float,
    key: jrnd.PRNGKey,
    max_ppc_samples: int,
) -> dict[str, float]:
    if X_eval.shape[0] == 0:
        return {
            "split": split_name,
            "rmse": np.nan,
            "nll": np.nan,
            "cov90": np.nan,
            "iw90": np.nan,
            "cov95": np.nan,
            "iw95": np.nan,
        }

    mu = _posterior_means(samples, unravel_fn, X_eval, activation)
    draws = posterior_predictive_draws(
        samples,
        unravel_fn,
        X_eval,
        lik_sigma,
        activation,
        key,
        S=min(max_ppc_samples, samples.shape[0]),
    )
    y_eval = y_eval.reshape(mu.shape)
    cov90, iw90 = coverage(y_eval, draws, alpha=0.1)
    cov95, iw95 = coverage(y_eval, draws, alpha=0.05)
    metrics = {
        "split": split_name,
        "rmse": rmse(y_eval, mu),
        "nll": nll_gaussian(y_eval, mu, lik_sigma),
        "cov90": cov90,
        "iw90": iw90,
        "cov95": cov95,
        "iw95": iw95,
    }
    return metrics


def _process_dataset(
    dataset_id: str,
    *,
    prior_name: str,
    nu: float,
    config: TrainConfig,
    existing_keys: set[tuple[str, str, str]],
    force: bool,
    jobs_seed_offset: int,
    return_model: bool = False,
) -> List[dict]:
    key_base = jrnd.PRNGKey(config.seed + jobs_seed_offset)

    if (
        not force
        and (dataset_id, prior_name, "id") in existing_keys
        and (dataset_id, prior_name, "ood") in existing_keys
    ):
        LOGGER.info("Skipping %s (%s) - already present", dataset_id, prior_name)
        return []

    dataset_path = artifacts_path("datasets", f"{dataset_id}.npz")
    arrays = load_npz_arrays(dataset_path)
    X = arrays["X"]
    y = arrays["y"]
    input_dim = X.shape[1]

    layer_widths = _resolve_layer_widths(config.layer_widths, input_dim)
    activation = _activation_from_config(config.activation_name)

    splits = make_splits(
        X,
        y,
        seed=config.seed + jobs_seed_offset,
        id_ratio=config.id_split_ratio,
        ood_tail_fraction=config.ood_tail_fraction,
    )

    fit = fit_bnn(
        splits["X_train"],
        splits["y_train"],
        layer_widths=layer_widths,
        activation=activation,
        prior_name=prior_name,
        nu=nu,
        lik_sigma=config.lik_sigma,
        init_scheme=config.init_scheme,
        num_samples=config.num_samples,
        num_burn_in=config.num_burn_in,
        step_size=config.step_size,
        seed=config.seed + jobs_seed_offset,
        prior_scale=config.prior_scale,
    )

    samples = fit.samples
    unravel = fit.unravel_fn

    key_id, key_ood = jrnd.split(key_base)
    metrics_id = _evaluate_split(
        "id",
        splits["X_id"],
        splits["y_id"],
        samples=samples,
        unravel_fn=unravel,
        activation=activation,
        lik_sigma=config.lik_sigma,
        key=key_id,
        max_ppc_samples=config.max_ppc_samples,
    )
    metrics_ood = _evaluate_split(
        "ood",
        splits["X_ood"],
        splits["y_ood"],
        samples=samples,
        unravel_fn=unravel,
        activation=activation,
        lik_sigma=config.lik_sigma,
        key=key_ood,
        max_ppc_samples=config.max_ppc_samples,
    )

    rows = []
    cfg_row = {
        "layer_widths": layer_widths,
        "activation": config.activation_name,
        "lik_sigma": config.lik_sigma,
        "nu": nu,
        "prior_scale": config.prior_scale,
        "init_scheme": config.init_scheme,
    }
    for metrics in (metrics_id, metrics_ood):
        diagnostics = dict(fit.diagnostics)
        diagnostics["seed"] = config.seed + jobs_seed_offset
        diagnostics["num_draws"] = int(samples.shape[0])
        row = result_row(
            dataset_id=dataset_id,
            prior=prior_name,
            split=metrics["split"],
            metrics=metrics,
            diagnostics=diagnostics,
            config=cfg_row,
        )
        rows.append(row)
    if return_model:
        return rows, fit
    return rows


def run_training(
    *,
    prior: str,
    nu: float,
    jobs: int = 1,
    force: bool = False,
) -> dict[str, int]:
    """Train and evaluate the BNN for each dataset.

    Args:
        prior: Name of the prior to evaluate.
        nu: Degrees of freedom for Student-t prior.
        jobs: Parallel jobs.
        force: Recompute even if results exist.

    Returns:
        Dictionary with number of updated results.
    """

    config = _load_train_config()
    manifest = manifest_io.read_manifest(status="ok")
    dataset_ids = list(manifest["dataset_id"])

    results_path = artifacts_path("results.parquet")
    if results_path.exists():
        df_existing = pd.read_parquet(results_path)
    else:
        df_existing = pd.DataFrame()
    existing_keys = (
        set(zip(df_existing.get("dataset_id", []), df_existing.get("prior", []), df_existing.get("split", [])))
        if not df_existing.empty
        else set()
    )

    tasks = Parallel(n_jobs=jobs)(
        delayed(_process_dataset)(
            dataset_id,
            prior_name=prior,
            nu=nu,
            config=config,
            existing_keys=existing_keys,
            force=force,
            jobs_seed_offset=index,
        )
        for index, dataset_id in enumerate(dataset_ids)
    )

    rows = [row for sublist in tasks for row in sublist]
    if not rows:
        return {"updated": 0}

    df_new = pd.DataFrame(rows)
    if not df_existing.empty:
        df_combined = pd.concat([df_existing, df_new], ignore_index=True, sort=False)
    else:
        df_combined = df_new
    df_combined = df_combined.drop_duplicates(subset=["dataset_id", "prior", "split"], keep="last")

    atomic_write_parquet(df_combined, results_path)
    return {"updated": len(rows)}
