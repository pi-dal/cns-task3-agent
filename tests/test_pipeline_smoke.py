"""Smoke test: pipeline runs and produces artifacts."""

from task3_agent.pipeline import run_pipeline
from task3_agent.contracts import DataSource, Task3Config
import tempfile, os


def test_pipeline_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        config = Task3Config(
            run_name="test-run",
            output_dir=tmp,
            sources=[DataSource(source_id="test", kind="test", uri="https://example.com/")],
        )
        summary = run_pipeline(config)
        assert len(summary.stages) == 4
        assert summary.run_name == "test-run"
        assert os.path.exists(os.path.join(tmp, "agent", "log.jsonl"))
        assert os.path.exists(os.path.join(tmp, "agent", "run_summary.json"))
        # result README only exists if data/baseline stages produce output;
        # with no input_dir and no manifest in temp dir, data stage returns empty → skip baseline → skip ensemble
        print("pipeline smoke OK (data+baseline skipped due to no input)")


def test_pipeline_with_input_dir():
    """Pipeline with PDB files in input_dir should process them."""
    import task3_agent.pipeline
    # Patch BEST_CONFIG for fast test (tiny model, few epochs)
    original = task3_agent.pipeline.BEST_CONFIG
    task3_agent.pipeline.BEST_CONFIG = dict(
        original,
        num_epochs=5, latent_dim=4, hidden_dim=16, ensemble_samples=2,
    )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = os.path.join(tmp, "saisdata")
            os.makedirs(input_dir, exist_ok=True)

            for entry_id, num_ca in [("1ubq", 2), ("4ins", 3)]:
                pdb_path = os.path.join(input_dir, f"{entry_id}.pdb")
                with open(pdb_path, "w") as f:
                    for i in range(num_ca):
                        # Fixed-width PDB ATOM format (column positions are strict)
                        serial = i + 1
                        x = float(i * 2)
                        y = float(i * 3)
                        z = float(i)
                        line = f"ATOM  {serial:>5d}  CA  ALA A{serial:>4d}    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C\n"
                        f.write(line)
                    f.write("END\n")

            config = Task3Config(
                run_name="test-run",
                output_dir=tmp,
                input_dir=input_dir,
                sources=[DataSource(source_id="pdb", kind="protein_structure", uri="file://dummy")],
            )
            summary = run_pipeline(config)
            assert len(summary.stages) == 4
            assert summary.run_name == "test-run"
            assert os.path.exists(os.path.join(tmp, "agent", "log.jsonl"))
            assert os.path.exists(os.path.join(tmp, "agent", "run_summary.json"))
            assert os.path.exists(os.path.join(tmp, "runs", "test-run", "result", "README.md"))
            print("pipeline with input_dir OK")
    finally:
        task3_agent.pipeline.BEST_CONFIG = original
