"""CLI entrypoint for CNS Task 3 agent."""

import argparse
import sys
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(
        description="CNS Task 3 — Protein Conformational Ensemble Generation Agent"
    )
    parser.add_argument(
        "--config", "-c",
        default="",
        help="Path to data source config JSON.",
    )
    parser.add_argument(
        "--run-name", "-n",
        default=f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        help="Run name (default: auto timestamp).",
    )
    args = parser.parse_args()

    from .config import resolve_config_path, load_data_sources
    from .pipeline import run_pipeline

    config_path = resolve_config_path(args.config)
    print(f"[task3-agent] Using config: {config_path}")

    data_sources = load_data_sources(config_path)
    print(f"[task3-agent] Loaded {len(data_sources)} data sources")

    run_name = args.run_name
    log_path = f"agent/log.jsonl"
    summary_path = f"agent/run_summary.json"
    result_dir = f"runs/{run_name}/result"

    summary = run_pipeline(data_sources, run_name, log_path, summary_path, result_dir)
    print(f"[task3-agent] Pipeline complete. Summary: {summary_path}")
    print(f"[task3-agent] Result: {result_dir}/README.md")


if __name__ == "__main__":
    main()
