"""Data and pipeline contract types for CNS Task 3."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataSource:
    """A public data source entry."""
    source_id: str
    kind: str
    uri: str
    license: str = ""
    notes: str = ""


@dataclass
class PipelineStage:
    """A named pipeline stage with optional result."""
    name: str
    status: str = "pending"  # pending | running | done | failed
    summary: str = ""


@dataclass
class AuditEvent:
    """A single audit event entry."""
    stage: str
    action: str
    status: str = "ok"
    detail: str = ""
    timestamp: Optional[str] = None


@dataclass
class RunSummary:
    """Summary of a full pipeline run."""
    run_name: str
    stages: list[PipelineStage] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)
    conclusion: str = ""
