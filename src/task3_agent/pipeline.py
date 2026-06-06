"""Stub pipeline stages for CNS Task 3 scaffold."""

from datetime import datetime, timezone

from .contracts import PipelineStage, AuditEvent, RunSummary, Task3Config
from .audit import log_event, write_run_summary, write_run_readme


def run_pipeline(config: Task3Config) -> RunSummary:
    """Run the three-stage stub pipeline."""
    run_name = config.run_name
    output_dir = config.output_dir
    log_path = f"{output_dir}/agent/log.jsonl"
    summary_path = f"{output_dir}/agent/run_summary.json"
    result_dir = f"{output_dir}/runs/{run_name}/result"

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
        data_sources=[s.source_id for s in config.sources],
        conclusion="Scaffold run completed. All stages are stubs.",
    )

    write_run_summary(summary_path, summary)
    write_run_readme(result_dir, summary)
    return summary
