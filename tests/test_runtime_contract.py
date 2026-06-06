"""Test container runtime contract: input/output mapping and packing."""

import tempfile, os, zipfile
from task3_agent.runtime import resolve_input_dir, resolve_output_dir, pack_result


def test_resolve_input_fail():
    """Missing input should fail fast."""
    try:
        resolve_input_dir("/nonexistent")
        assert False, "Should have raised"
    except FileNotFoundError:
        pass


def test_resolve_output_creates():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "output")
        assert not os.path.exists(out)
        result = resolve_output_dir(out)
        assert os.path.isdir(out)
        assert result == out


def test_pack_result():
    with tempfile.TemporaryDirectory() as tmp:
        # Create some output files
        os.makedirs(os.path.join(tmp, "agent"))
        open(os.path.join(tmp, "agent", "log.jsonl"), "w").close()
        os.makedirs(os.path.join(tmp, "runs", "test", "result"))
        open(os.path.join(tmp, "runs", "test", "result", "README.md"), "w").close()
        
        zip_path = pack_result(tmp, os.path.join(tmp, "result.zip"))
        assert os.path.exists(zip_path)
        
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            assert any("log.jsonl" in n for n in names)
            assert any("README.md" in n for n in names)
