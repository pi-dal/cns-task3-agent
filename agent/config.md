# Agent Configuration Guide

## Research Pipeline

The agent follows a three-stage pipeline:

1. **Literature**: Survey relevant methods and identify baseline approaches.
2. **Data**: Acquire and normalize public data sources.
3. **Baseline**: Run a baseline conformational generation / scoring method.

## Data Sources

See `configs/data_sources.example.json` for the source manifest.

Each source entry provides:
- `source_id`: unique identifier
- `kind`: type of resource
- `uri`: primary access URL
- `license`: usage terms
- `notes`: context and usage guidance

## Audit Artifacts

Each run produces:
- `agent/log.jsonl`: sequential event log
- `agent/run_summary.json`: key decisions and results
- `runs/<run_name>/result/README.md`: human-readable run report

## Constraints

- No training loop or generative model in the first slice.
- Pipeline stages are stubs that log audit events only.
