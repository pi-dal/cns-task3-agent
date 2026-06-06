"""Test PDB data acquisition + normalization."""

import os, tempfile, torch
from task3_agent.data import acquire_pdb, normalize_pdb
from task3_agent.contracts import DataSource


def test_normalize_pdb():
    """Verify normalized.pt has all required fields with correct shapes."""
    test_pdb = "configs/test_data/1ubq.pdb"
    assert os.path.exists(test_pdb), f"Test PDB missing: {test_pdb}"
    
    result = normalize_pdb(test_pdb, "1ubq", "pdb", "/dev/null")
    
    required_fields = {"coords", "residue_types", "sequence_length", "source_format", "entry_id"}
    assert required_fields.issubset(result.keys()), f"Missing fields: {required_fields - result.keys()}"
    
    coords = result["coords"]
    assert isinstance(coords, torch.Tensor), "coords must be torch.Tensor"
    assert coords.ndim == 2, f"coords shape: {coords.shape}, expected (N, 3)"
    assert coords.shape[1] == 3, f"coords shape: {coords.shape}, expected (N, 3)"
    
    assert result["entry_id"] == "1ubq"
    assert result["source_format"] == "pdb"
    assert result["sequence_length"] > 0
    assert len(result["residue_types"]) == result["sequence_length"]
    
    print(f"normalize_pdb OK: {result['sequence_length']} residues, {coords.shape}")


def test_acquire_and_normalize():
    """Verify acquisition + normalization produces all expected artifacts."""
    with tempfile.TemporaryDirectory() as tmp:
        source = DataSource(
            source_id="pdb_test",
            kind="protein_structure",
            uri=f"file://{os.path.dirname(os.path.abspath('configs/test_data/1ubq.pdb'))}",
            format="pdb",
        )
        pdb_path = acquire_pdb("1ubq", tmp, source, "/dev/null")
        assert os.path.exists(pdb_path), "PDB file not acquired"
        
        result = normalize_pdb(pdb_path, "1ubq", "pdb", "/dev/null")
        assert os.path.exists(os.path.join(tmp, "pdb_test", "1ubq", "normalized.pt")), "normalized.pt missing"
        assert result["sequence_length"] == 2  # ALA + GLY
        print(f"acquire + normalize OK: saved normalized.pt")
