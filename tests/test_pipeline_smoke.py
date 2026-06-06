"""Smoke test: pipeline runs and produces artifacts."""

from task3_agent.pipeline import run_pipeline
from task3_agent.contracts import DataSource, Task3Config
import tempfile, os


def test_pipeline_smoke():
    config = Task3Config(
        run_name="test-run",
        output_dir="/tmp/test-out",
        sources=[DataSource(source_id="test", kind="test", uri="https://example.com/")],
    )
    summary = run_pipeline(config)
    assert len(summary.stages) == 3
    assert summary.run_name == "test-run"
    print("pipeline smoke OK")
