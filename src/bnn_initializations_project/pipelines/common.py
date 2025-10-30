"""Common helpers shared across CLI pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def load_yaml_config(filename: str) -> dict[str, Any]:
    """Load a YAML configuration from the configs directory."""
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data
