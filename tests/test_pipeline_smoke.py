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
        assert len(summary.stages) == 3
        assert summary.run_name == "test-run"
        assert os.path.exists(os.path.join(tmp, "agent", "log.jsonl"))
        assert os.path.exists(os.path.join(tmp, "agent", "run_summary.json"))
        assert os.path.exists(os.path.join(tmp, "runs", "test-run", "result", "README.md"))
        print("pipeline smoke OK")
