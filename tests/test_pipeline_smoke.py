"""Smoke test: pipeline runs and produces artifacts."""

from task3_agent.pipeline import run_pipeline
from task3_agent.contracts import DataSource
import tempfile, os


def test_pipeline_smoke():
    sources = [DataSource(source_id="test", kind="test", uri="https://example.com/")]
    with tempfile.TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, "log.jsonl")
        summary_path = os.path.join(tmp, "run_summary.json")
        result_dir = os.path.join(tmp, "result")
        summary = run_pipeline(sources, "test-run", log_path, summary_path, result_dir)
        assert len(summary.stages) == 3
        assert os.path.exists(log_path)
        assert os.path.exists(summary_path)
        assert os.path.exists(os.path.join(result_dir, "README.md"))
        print("pipeline smoke OK")
