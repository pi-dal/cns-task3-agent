"""Stub pipeline stages for CNS Task 3 scaffold."""

from datetime import datetime, timezone

from .contracts import PipelineStage, AuditEvent, RunSummary, Task3Config, DataSource
from .audit import log_event, write_run_summary, write_run_readme
from .data import acquire_pdb, normalize_pdb


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
        PipelineStage(name="baseline", status="pending"),
    ]

    for stage in stages:
        log_event(log_path, AuditEvent(
            stage=stage.name,
            action="start",
            detail=f"Starting {stage.name} stage.",
        ))
        if stage.name == "data":
            _run_data_stage(log_path, output_dir, config.sources)
        stage.status = "done"
        stage.summary = f"{stage.name} stage completed."
        log_event(log_path, AuditEvent(
            stage=stage.name,
            action="complete",
            detail=stage.summary,
        ))

    summary = RunSummary(
        run_name=run_name,
        stages=stages,
        data_sources=[s.source_id for s in config.sources],
        conclusion="Pipeline run completed.",
    )

    write_run_summary(summary_path, summary)
    write_run_readme(result_dir, summary)
    return summary


def _run_data_stage(log_path: str, output_dir: str, sources: list[DataSource]) -> None:
    """Run the data acquisition + normalization stage."""
    import os
    import os
    pdb_test = "configs/test_data/1ubq.pdb"
    if not os.path.exists(pdb_test):
        log_event(log_path, AuditEvent(
            stage="data", action="skip",
            detail=f"Test PDB not found at {pdb_test}, skipping acquisition",
        ))
        return
    
    # Use file:// URI pointing to the directory containing the PDB
    pdb_dir = os.path.abspath(os.path.dirname(pdb_test))
    local_source = DataSource(
        source_id="pdb_test",
        kind="protein_structure",
        uri=f"file://{pdb_dir}",
        format="pdb",
    )
    
    pdb_path = acquire_pdb("1ubq", output_dir, local_source, log_path)
    result = normalize_pdb(pdb_path, "1ubq", "pdb", log_path)
    
    log_event(log_path, AuditEvent(
        stage="data", action="complete",
        detail=f"Acquired + normalized 1ubq: {result['sequence_length']} residues",
    ))
