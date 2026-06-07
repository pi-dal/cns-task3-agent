"""Test VAE training + checkpoint/metadata contract."""

import os, tempfile, torch, json
from task3_agent.train import load_normalized_inputs, train_vae
from task3_agent.contracts import Task3Config, DataSource


def make_mock_npt(tmp_dir: str, entry_id: str, num_atoms: int):
    """Create a mock normalized.pt for testing."""
    entry_dir = os.path.join(tmp_dir, "test_source", entry_id)
    os.makedirs(entry_dir, exist_ok=True)
    data = {
        "coords": torch.randn(num_atoms, 3),
        "residue_types": ["A"] * num_atoms,
        "sequence_length": num_atoms,
        "source_format": "pdb",
        "entry_id": entry_id,
    }
    torch.save(data, os.path.join(entry_dir, "normalized.pt"))


def test_load_normalized_inputs():
    with tempfile.TemporaryDirectory() as tmp:
        make_mock_npt(tmp, "1ubq", 5)
        make_mock_npt(tmp, "4hhb", 7)

        entry_dirs = [
            os.path.join(tmp, "test_source", "1ubq"),
            os.path.join(tmp, "test_source", "4hhb"),
        ]
        result = load_normalized_inputs(entry_dirs)

        assert result["num_samples"] == 12  # 5 + 7
        assert result["entry_ids"] == ["1ubq", "4hhb"]
        assert result["coords"].shape == (12, 3)
        assert len(result["entry_boundaries"]) == 2
        print(f"load_normalized_inputs OK: {result['num_samples']} samples, {result['entry_ids']}")


def test_training_checkpoint_contract():
    """Verify VAE training produces checkpoint with correct metadata."""
    with tempfile.TemporaryDirectory() as tmp:
        make_mock_npt(tmp, "1ubq", 5)
        make_mock_npt(tmp, "4hhb", 7)

        entry_dirs = [
            os.path.join(tmp, "test_source", "1ubq"),
            os.path.join(tmp, "test_source", "4hhb"),
        ]

        ckpt_dir = os.path.join(tmp, "artifacts", "checkpoints")
        metadata = train_vae(
            entry_dirs=entry_dirs,
            checkpoint_dir=ckpt_dir,
            run_name="test-run",
            latent_dim=8,
            hidden_dim=32,
            num_epochs=5,
        )

        # Check checkpoint file
        assert os.path.exists(os.path.join(ckpt_dir, "vae_best.pt")), "vae_best.pt missing"

        # Check metadata.json
        meta_path = os.path.join(ckpt_dir, "metadata.json")
        assert os.path.exists(meta_path), "metadata.json missing"

        with open(meta_path) as f:
            meta = json.load(f)

        # Verify all required fields
        required = [
            "run_name", "checkpoint_path", "input_entries", "num_samples",
            "num_epochs", "final_loss", "model_config", "ensembles", "created_at"
        ]
        for field in required:
            assert field in meta, f"Missing metadata field: {field}"

        assert meta["num_samples"] == 12  # 5 + 7
        assert meta["input_entries"] == ["1ubq", "4hhb"]
        assert meta["model_config"]["model_type"] == "PerResidueVAE"
        assert isinstance(meta["final_loss"], float)
        assert len(meta["ensembles"]) == 2

        print(f"training checkpoint contract OK: {meta['num_samples']} samples, loss={meta['final_loss']:.4f}")
