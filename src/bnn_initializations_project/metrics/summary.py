"""Helpers for building results table rows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

__all__ = ["result_row"]


def _format_arch(layer_widths: Sequence[int]) -> str:
    return "[" + ", ".join(str(w) for w in layer_widths) + "]"


def result_row(
    dataset_id: str,
    prior: str,
    split: str,
    metrics: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a single results row for persistence."""
    row = {
        "dataset_id": dataset_id,
        "prior": prior,
        "split": split,
        "rmse": metrics.get("rmse"),
        "nll": metrics.get("nll"),
        "cov90": metrics.get("cov90"),
        "iw90": metrics.get("iw90"),
        "cov95": metrics.get("cov95"),
        "iw95": metrics.get("iw95"),
        "accept_rate": diagnostics.get("accept_rate"),
        "num_draws": diagnostics.get("num_draws"),
        "seed": diagnostics.get("seed"),
        "arch": _format_arch(config.get("layer_widths", [])),
        "activation": config.get("activation"),
        "sigma_lik": config.get("lik_sigma"),
        "nu": config.get("nu"),
        "prior_scale": config.get("prior_scale"),
        "init_scheme": config.get("init_scheme"),
    }
    return row
