"""Test config loading."""

from task3_agent.config import load_data_sources
import tempfile, json


def test_load_data_sources():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([
            {"source_id": "pdb", "kind": "structure", "uri": "https://rcsb.org/"}
        ], f)
        path = f.name
    sources = load_data_sources(path)
    assert len(sources) == 1
    assert sources[0].source_id == "pdb"
    print("config load OK")
