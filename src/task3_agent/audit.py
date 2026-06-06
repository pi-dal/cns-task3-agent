"""Audit log and run summary generation."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .contracts import AuditEvent, RunSummary, PipelineStage


def log_event(log_path: str, event: AuditEvent) -> None:
    """Append an audit event to the JSONL log file."""
    event.timestamp = event.timestamp or datetime.now(timezone.utc).isoformat()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event.__dict__, ensure_ascii=False) + "\n")


def write_run_summary(summary_path: str, summary: RunSummary) -> None:
    """Write a structured run summary to disk."""
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    data = {
        "run_name": summary.run_name,
        "stages": [s.__dict__ for s in summary.stages],
        "data_sources": summary.data_sources,
        "conclusion": summary.conclusion,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_run_readme(result_dir: str, summary: RunSummary) -> None:
    """Write a human-readable README for a run result."""
    os.makedirs(result_dir, exist_ok=True)
    lines = [f"# Run: {summary.run_name}", ""]
    for s in summary.stages:
        lines.append(f"## {s.name}")
        lines.append(f"- Status: {s.status}")
        lines.append(f"- Summary: {s.summary}")
        lines.append("")
    lines.append("## Data Sources")
    for ds in summary.data_sources:
        lines.append(f"- {ds}")
    lines.append("")
    lines.append(f"## Conclusion")
    lines.append(summary.conclusion or "No conclusion recorded.")
    lines.append("")
    path = os.path.join(result_dir, "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
