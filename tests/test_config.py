"""Test config loading."""

from task3_agent.config import load_task3_config
import tempfile, json


def test_load_task3_config():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({
            "run_name": "test-run",
            "output_dir": "/tmp/test",
            "sources": [
                {"source_id": "pdb", "kind": "structure", "uri": "https://rcsb.org/"}
            ]
        }, f)
        path = f.name
    cfg = load_task3_config(path)
    assert cfg.run_name == "test-run"
    assert cfg.output_dir == "/tmp/test"
    assert len(cfg.sources) == 1
    assert cfg.sources[0].source_id == "pdb"
    print("Task3Config load OK")
