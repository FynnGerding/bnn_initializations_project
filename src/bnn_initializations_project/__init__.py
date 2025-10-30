from __future__ import annotations

from .models import (
    BayesianMLPConfig,
    Bayesian_MLP,
    FitResult,
    PriorName,
    build_log_posterior_fn,
    fit_bnn,
    flatten_params,
    initialize_prior,
    logpdf_laplace,
    logpdf_normal,
    logpdf_student_t,
    _xavier_std,
)
from .pipelines import (
    run_analysis,
    run_discovery,
    run_meta_feature_extraction,
    run_training,
)

from .visualization import (
    plot_correlation_summary,
    plot_metric_correlation_heatmap,
)


__all__ = [
    # Core model utilities
    "_xavier_std",
    "BayesianMLPConfig",
    "Bayesian_MLP",
    "FitResult",
    "PriorName",
    "build_log_posterior_fn",
    "fit_bnn",
    "flatten_params",
    "initialize_prior",
    "logpdf_laplace",
    "logpdf_normal",
    "logpdf_student_t",
    # High-level pipelines
    "run_analysis",
    "run_discovery",
    "run_meta_feature_extraction",
    "run_training",
    # Visualization utilities
    "plot_correlation_summary",
    "plot_metric_correlation_heatmap",
]
