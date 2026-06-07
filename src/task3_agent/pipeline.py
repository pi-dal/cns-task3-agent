"""Full CNS Task 3 pipeline with PerResidueVAE ensemble generation."""

import glob
import json
import os
import shutil
from datetime import datetime, timezone

from .contracts import PipelineStage, AuditEvent, RunSummary, Task3Config, DataSource
from .audit import log_event, write_run_summary, write_run_readme
from .data import acquire_pdb, normalize_pdb
from .train import train_vae


# Best config from autoresearch
BEST_CONFIG = {
    "latent_dim": 24,
    "hidden_dim": 192,
    "num_epochs": 6000,
    "learning_rate": 5e-4,
    "beta": 0.001,
    "noise_std": 0.05,
    "ensemble_temp": 2.0,
    "ensemble_samples": 20,
}


def run_pipeline(config: Task3Config) -> RunSummary:
    """Run the three-stage CNS Task 3 pipeline."""
    run_name = config.run_name
    output_dir = config.output_dir
    input_dir = config.input_dir
    log_path = f"{output_dir}/agent/log.jsonl"
    summary_path = f"{output_dir}/agent/run_summary.json"
    result_dir = f"{output_dir}/runs/{run_name}/result"
    artifact_dir = f"{output_dir}/runs/{run_name}/artifacts/checkpoints"

    stages = [
        PipelineStage(name="literature", status="running"),
        PipelineStage(name="data", status="running"),
        PipelineStage(name="baseline", status="pending"),
        PipelineStage(name="ensemble", status="pending"),
    ]

    entry_dirs = []

    for stage in stages:
        log_event(log_path, AuditEvent(
            stage=stage.name,
            action="start",
            detail=f"Starting {stage.name} stage.",
        ))

        if stage.name == "data":
            entry_dirs = _run_data_stage(log_path, output_dir, config.sources, config.run_name, input_dir)
        elif stage.name == "baseline":
            _run_baseline_stage(log_path, entry_dirs, config, run_name, artifact_dir)
        elif stage.name == "ensemble":
            _run_ensemble_stage(log_path, entry_dirs, result_dir, artifact_dir)

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
        conclusion="Pipeline run completed — PerResidueVAE ensemble generated.",
    )

    write_run_summary(summary_path, summary)
    return summary


def _run_data_stage(log_path: str, output_dir: str, sources: list[DataSource],
                    run_name: str, input_dir: str = "") -> list[str]:
    """Acquire and normalize PDB structures.

    Priority:
    1. Scan input_dir for .pdb files (competition mode: /saisdata/)
    2. Fall back to configs/entry_manifest.json (local dev mode: download from RCSB)
    """
    entry_dirs = []

    # Priority 1: Scan input_dir for .pdb files
    if input_dir and os.path.isdir(input_dir):
        pdb_files = glob.glob(os.path.join(input_dir, "*.pdb"))
        if pdb_files:
            log_event(log_path, AuditEvent(
                stage="data", action="scan",
                detail=f"Found {len(pdb_files)} PDB files in {input_dir}",
            ))
            for pdb_path in sorted(pdb_files):
                entry_id = os.path.splitext(os.path.basename(pdb_path))[0].lower()
                try:
                    # Copy to output dir and normalize
                    entry_dir = os.path.join(output_dir, "saisdata", entry_id)
                    os.makedirs(entry_dir, exist_ok=True)
                    dst_path = os.path.join(entry_dir, f"structure.pdb")
                    shutil.copy2(pdb_path, dst_path)
                    normalize_pdb(dst_path, entry_id, "pdb", log_path)
                    entry_dirs.append(entry_dir)
                    log_event(log_path, AuditEvent(
                        stage="data", action="ok",
                        detail=f"Processed {entry_id} from {pdb_path}",
                    ))
                except Exception as e:
                    log_event(log_path, AuditEvent(
                        stage="data", action="error",
                        detail=f"Failed {entry_id} from {pdb_path}: {e}",
                    ))
            log_event(log_path, AuditEvent(
                stage="data", action="summary",
                detail=f"Processed {len(entry_dirs)} entries from input_dir",
            ))
            return entry_dirs

    # Priority 2: Fall back to manifest
    manifest_path = "configs/entry_manifest.json"
    if not os.path.exists(manifest_path):
        log_event(log_path, AuditEvent(
            stage="data", action="skip",
            detail=f"No input_dir files and no manifest at {manifest_path}",
        ))
        return []

    with open(manifest_path) as f:
        entries = json.load(f)

    for entry in entries:
        entry_id = entry.get("entry_id", "")
        source_id = entry.get("source_id", "")
        if not entry_id or not source_id:
            continue

        source = next((s for s in sources if s.source_id == source_id), None)
        if source is None:
            continue

        try:
            pdb_path = acquire_pdb(entry_id, output_dir, source, log_path)
            normalize_pdb(pdb_path, entry_id, source.format, log_path)
            entry_dir = os.path.join(output_dir, source_id, entry_id)
            entry_dirs.append(entry_dir)
            log_event(log_path, AuditEvent(
                stage="data", action="ok", detail=f"Processed {entry_id}",
            ))
        except Exception as e:
            log_event(log_path, AuditEvent(
                stage="data", action="error", detail=f"Failed {entry_id}: {e}",
            ))

    log_event(log_path, AuditEvent(
        stage="data", action="summary",
        detail=f"Processed {len(entry_dirs)} entries from manifest",
    ))
    return entry_dirs


def _run_baseline_stage(log_path: str, entry_dirs: list[str], config: Task3Config,
                        run_name: str, artifact_dir: str) -> None:
    """Train PerResidueVAE with best config."""  
    if not entry_dirs:
        log_event(log_path, AuditEvent(stage="baseline", action="skip", detail="No data available"))
        return

    metadata = train_vae(
        entry_dirs=entry_dirs,
        checkpoint_dir=artifact_dir,
        run_name=run_name,
        **BEST_CONFIG,
    )

    log_event(log_path, AuditEvent(
        stage="baseline", action="complete",
        detail=f"VAE trained: {metadata['num_samples']} samples, loss={metadata['final_loss']:.6f}",
    ))


def _run_ensemble_stage(log_path: str, entry_dirs: list[str], result_dir: str, artifact_dir: str) -> None:
    """Generate conformational ensemble README with results summary."""
    import torch
    
    # Load metadata
    meta_path = os.path.join(artifact_dir, "metadata.json")
    if not os.path.exists(meta_path):
        log_event(log_path, AuditEvent(stage="ensemble", action="skip", detail="No metadata found"))
        return

    with open(meta_path) as f:
        metadata = json.load(f)

    # Write enhanced result README
    readme_path = os.path.join(result_dir, "README.md")
    lines = [
        f"# CNS Task 3 — Protein Conformational Ensemble",
        f"",
        f"**Run**: {metadata['run_name']}",
        f"**Model**: PerResidueVAE (latent={metadata['model_config']['latent_dim']}, hidden={metadata['model_config']['hidden_dim']})",
        f"**Training**: {metadata['num_epochs']} epochs, final loss={metadata['final_loss']:.6f}",
        f"**Ensemble**: {metadata['ensemble_config']['n_samples']} samples per protein, temp={metadata['ensemble_config']['temp']}",
        f"**Created**: {metadata['created_at']}",
        f"",
        f"## Input Proteins",
        f"",
    ]
    for eid in metadata['input_entries']:
        lines.append(f"- {eid}")
    lines.append("")
    lines.append("## Generated Ensembles")
    lines.append("")
    lines.append("| Entry | Residues | Pairwise RMSD (Å) |")
    lines.append("|-------|----------|-------------------|")
    for info in metadata['ensembles']:
        lines.append(f"| {info['entry_id']} | {info['n_residues']} | {info['ensemble_pairwise_rmsd']:.3f} |")
    lines.append("")

    os.makedirs(result_dir, exist_ok=True)
    with open(readme_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    log_event(log_path, AuditEvent(
        stage="ensemble", action="complete",
        detail=f"Ensemble results written: {readme_path}",
    ))
