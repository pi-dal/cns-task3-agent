"""Configuration loading and validation."""

import json
from pathlib import Path
from typing import Optional

from .contracts import DataSource


def load_data_sources(path: str) -> list[DataSource]:
    """Load data source manifest from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [DataSource(**entry) for entry in raw]


def resolve_config_path(config_arg: Optional[str]) -> str:
    """Resolve config path from argument or default."""
    if config_arg:
        return config_arg
    default = Path("configs/data_sources.example.json")
    if default.exists():
        return str(default)
    raise FileNotFoundError(
        "No config path specified and default configs/data_sources.example.json not found."
    )
