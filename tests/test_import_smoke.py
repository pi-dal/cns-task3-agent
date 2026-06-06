"""Smoke test: verify package imports work."""

from task3_agent.contracts import DataSource, Task3Config, PipelineStage, AuditEvent, RunSummary
from task3_agent.config import load_task3_config
from task3_agent.audit import log_event, write_run_summary, write_run_readme
from task3_agent.pipeline import run_pipeline


def test_imports():
    assert DataSource
    assert Task3Config
    assert PipelineStage
    assert AuditEvent
    assert RunSummary
    assert load_task3_config
    assert log_event
    assert write_run_summary
    assert write_run_readme
    assert run_pipeline
    print("smoke import OK")
