"""Configuration loading and validation."""

import json
from pathlib import Path
from typing import Optional

from .contracts import Task3Config, DataSource


def load_task3_config(path: str) -> Task3Config:
    """Load a complete Task3Config from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    sources_raw = raw.get("sources", [])
    sources = [DataSource(**entry) for entry in sources_raw]
    return Task3Config(
        run_name=raw.get("run_name", "default"),
        output_dir=raw.get("output_dir", "."),
        sources=sources,
    )


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
