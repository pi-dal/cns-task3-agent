# CNS Task 3 — Protein Conformational Ensemble Generation

AI agent scaffold for the AI4S CNS Challenge, Task 3.

## Overview

This agent handles race 6 (Task 3) of the competition: **AI Agent for Protein Conformational Ensemble Generation**.

Unlike PDE (Task 4), this task focuses on:
- Public data source acquisition and normalization
- Configurable research pipeline (literature → data → baseline)
- Agent audit trail (config, prompt, log, run summary)

## Repository Structure

```
cns-task3-agent/
├── configs/
│   └── data_sources.example.json    # Public data source manifest
├── agent/
│   ├── config.md                     # Agent configuration guide
│   └── prompt.md                     # Agent prompt specification
├── src/
│   └── task3_agent/
│       ├── __init__.py
│       ├── contracts.py              # Data/pipeline contract types
│       ├── config.py                 # Configuration loading
│       ├── audit.py                  # Audit log + run summary
│       ├── pipeline.py               # Stub pipeline stages
│       └── cli.py                    # CLI entrypoint
├── tests/
│   ├── test_import_smoke.py
│   ├── test_config.py
│   └── test_pipeline_smoke.py
├── docs/plans/
└── README.md
```

## Usage

```bash
task3-agent run --config configs/data_sources.example.json
```

## Current Status

First slice: scaffold only (no model, no training). See `docs/plans/`.
