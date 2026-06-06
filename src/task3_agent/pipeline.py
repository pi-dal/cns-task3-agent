"""Stub pipeline stages for CNS Task 3 scaffold."""

from datetime import datetime, timezone

from .contracts import PipelineStage, AuditEvent, RunSummary
from .audit import log_event, write_run_summary, write_run_readme


def run_pipeline(
    data_sources: list,
    run_name: str,
    log_path: str,
    summary_path: str,
    result_dir: str,
) -> RunSummary:
    """Run the three-stage stub pipeline."""
    stages = [
        PipelineStage(name="literature", status="running"),
        PipelineStage(name="data", status="running"),
        PipelineStage(name="baseline", status="running"),
    ]

    for stage in stages:
        log_event(log_path, AuditEvent(
            stage=stage.name,
            action="start",
            detail=f"Starting {stage.name} stage.",
        ))
        stage.status = "done"
        stage.summary = f"{stage.name} stage completed (stub)."
        log_event(log_path, AuditEvent(
            stage=stage.name,
            action="complete",
            detail=stage.summary,
        ))

    summary = RunSummary(
        run_name=run_name,
        stages=stages,
        data_sources=[s.source_id for s in data_sources],
        conclusion="Scaffold run completed. All stages are stubs.",
    )

    write_run_summary(summary_path, summary)
    write_run_readme(result_dir, summary)
    return summary
