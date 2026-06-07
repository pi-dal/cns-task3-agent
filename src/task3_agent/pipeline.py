"""Stub pipeline stages for CNS Task 3 scaffold."""

import json
import os
from datetime import datetime, timezone

from .contracts import PipelineStage, AuditEvent, RunSummary, Task3Config, DataSource
from .audit import log_event, write_run_summary, write_run_readme
from .data import acquire_pdb, normalize_pdb
from .train import load_normalized_inputs, train_baseline


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
        elif stage.name == "baseline":
            _run_baseline_stage(log_path, output_dir, config, run_name)
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
    """Run the data acquisition + normalization stage.
    
    Reads an entry manifest (configs/entry_manifest.json) defining which
    PDB entries to process, then acquires and normalizes each one.
    """
    manifest_path = "configs/entry_manifest.json"
    if not os.path.exists(manifest_path):
        log_event(log_path, AuditEvent(
            stage="data", action="skip",
            detail=f"No entry manifest at {manifest_path}. Define entries to process.",
        ))
        return
    
    with open(manifest_path, "r") as f:
        entries = json.load(f)
    
    if not isinstance(entries, list):
        log_event(log_path, AuditEvent(stage="data", action="error", detail="entry_manifest must be a list"))
        return
    
    counts = {"acquired": 0, "normalized": 0, "skipped": 0, "failed": 0}
    for entry in entries:
        entry_id = entry.get("entry_id", "")
        source_id = entry.get("source_id", "")
        if not entry_id or not source_id:
            continue
        
        source = next((s for s in sources if s.source_id == source_id), None)
        if source is None:
            log_event(log_path, AuditEvent(stage="data", action="skip", detail=f"Unknown source {source_id} for {entry_id}"))
            continue
        
        try:
            pdb_path = acquire_pdb(entry_id, output_dir, source, log_path)
            result = normalize_pdb(pdb_path, entry_id, source.format, log_path)
            counts["normalized"] += 1
        except Exception as e:
            counts["failed"] += 1
            log_event(log_path, AuditEvent(stage="data", action="error", detail=f"Failed {entry_id}: {e}"))
    
    log_event(log_path, AuditEvent(
        stage="data", action="summary",
        detail=json.dumps(counts),
    ))


def _run_baseline_stage(log_path: str, output_dir: str, config: Task3Config, run_name: str) -> None:
    """Run the baseline training stage.
    
    Loads normalized.pt from the data stage output and trains a minimal MLP.
    """
    from pathlib import Path
    
    # Collect entry directories from data stage output
    entry_dirs = []
    for source in config.sources:
        src_output = os.path.join(output_dir, source.source_id)
        if not os.path.exists(src_output):
            continue
        for entry_id in os.listdir(src_output):
            npt_path = os.path.join(src_output, entry_id, "normalized.pt")
            if os.path.exists(npt_path):
                entry_dirs.append(os.path.dirname(npt_path))
    
    if not entry_dirs:
        log_event(log_path, AuditEvent(stage="baseline", action="skip", detail="No normalized artifacts found"))
        return
    
    input_data = load_normalized_inputs(entry_dirs)
    artifact_dir = os.path.join(output_dir, "runs", run_name, "artifacts", "checkpoints")
    
    metadata = train_baseline(input_data, artifact_dir)
    
    log_event(log_path, AuditEvent(
        stage="baseline", action="complete",
        detail=f"Baseline trained: {metadata['num_samples']} samples, loss={metadata['final_loss']:.4f}",
    ))
