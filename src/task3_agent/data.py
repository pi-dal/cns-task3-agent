"""PDB data acquisition and normalization for CNS Task 3."""

import os
import json
import torch
import urllib.request
from pathlib import Path
from typing import Optional

from .contracts import DataSource
from .audit import AuditEvent, log_event


def acquire_pdb(entry_id: str, output_dir: str, source: DataSource, log_path: str) -> str:
    """Download a PDB file from RCSB or copy from local path.
    
    Args:
        entry_id: PDB entry ID (e.g. '1ubq').
        output_dir: Base output directory.
        source: DataSource config with URI.
        log_path: Path to audit log.
    
    Returns:
        Path to downloaded/copied PDB file.
    """
    entry_dir = os.path.join(output_dir, source.source_id, entry_id)
    os.makedirs(entry_dir, exist_ok=True)
    pdb_path = os.path.join(entry_dir, f"structure.{source.format}")
    
    if os.path.exists(pdb_path):
        log_event(log_path, AuditEvent(stage="data", action="skip", detail=f"{entry_id} already exists"))
        return pdb_path
    
    # Download from RCSB
    if source.source_id == "pdb":
        url = f"https://files.rcsb.org/download/{entry_id}.{source.format}"
        log_event(log_path, AuditEvent(stage="data", action="download", detail=f"Downloading {entry_id} from {url}"))
        try:
            urllib.request.urlretrieve(url, pdb_path)
            return pdb_path
        except Exception as e:
            log_event(log_path, AuditEvent(stage="data", action="fail", detail=f"Failed to download {entry_id}: {e}"))
            raise
    else:
        # Local or custom source: URI points to directory, append entry_id.format
        base_path = source.uri.replace("file://", "", 1) if source.uri.startswith("file://") else source.uri
        src_path = os.path.join(base_path, f"{entry_id}.{source.format}")
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Local file not found: {src_path}")
        import shutil
        shutil.copy2(src_path, pdb_path)
        log_event(log_path, AuditEvent(stage="data", action="copy", detail=f"Copied {entry_id} from {src_path}"))
        return pdb_path


def _copy_local(src_uri: str, dst: str, entry_id: str, log_path: str) -> str:
    import shutil
    # Strip file:// prefix
    local_path = src_uri.replace("file://", "", 1)
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local PDB source not found: {local_path}")
    shutil.copy2(local_path, dst)
    log_event(log_path, AuditEvent(stage="data", action="copy", detail=f"Copied {entry_id} from {local_path}"))
    return dst


def normalize_pdb(pdb_path: str, entry_id: str, source_format: str, log_path: str) -> dict:
    """Normalize a PDB file to a fixed-schema normalized.pt dict.
    
    Returns dict with keys:
        coords: torch.Tensor of shape (N, 3) — backbone coordinates.
        residue_types: list[str] — one-letter residue codes.
        sequence_length: int.
        source_format: str — original file format.
        entry_id: str.
    """
    coords, residues = _parse_pdb(pdb_path)
    normalized = {
        "coords": torch.tensor(coords, dtype=torch.float32),
        "residue_types": residues,
        "sequence_length": len(residues),
        "source_format": source_format,
        "entry_id": entry_id,
    }
    
    entry_dir = os.path.dirname(pdb_path)
    out_path = os.path.join(entry_dir, "normalized.pt")
    torch.save(normalized, out_path)
    
    log_event(log_path, AuditEvent(
        stage="data", action="normalize",
        detail=f"Normalized {entry_id}: {len(residues)} residues, {len(coords)} atoms",
    ))
    return normalized


def _parse_pdb(pdb_path: str) -> tuple[list, list]:
    """Parse a PDB file and extract CA coordinates + residue types.
    
    Returns (coords_list, residues_list).
    """
    coords = []
    residues = []
    
    ONE_LETTER = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E",
        "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I",
        "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
        "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S",
        "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    }
    
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    res_name = line[17:20].strip()
                    coords.append([x, y, z])
                    residues.append(ONE_LETTER.get(res_name, "X"))
                except (ValueError, IndexError):
                    continue
    
    return coords, residues
