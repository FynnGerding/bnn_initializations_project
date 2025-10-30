"""Model definitions and inference utilities."""

from __future__ import annotations

from .bnn import Bayesian_MLP, BayesianMLPConfig, build_log_posterior_fn, flatten_params, initialize_prior, _xavier_std
from .inference import FitResult, fit_bnn
from .priors_impl import PriorName, logpdf_laplace, logpdf_normal, logpdf_student_t

__all__ = [
    "_xavier_std",
    "Bayesian_MLP",
    "BayesianMLPConfig",
    "build_log_posterior_fn",
    "fit_bnn",
    "FitResult",
    "flatten_params",
    "initialize_prior",
    "logpdf_laplace",
    "logpdf_normal",
    "logpdf_student_t",
    "PriorName",
]
