"""CLI entrypoint for CNS Task 3 agent."""

import argparse
import sys
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(
        description="CNS Task 3 — Protein Conformational Ensemble Generation Agent"
    )
    sub = parser.add_subparsers(dest="command")
    run_parser = sub.add_parser("run", help="Run the agent pipeline.")
    run_parser.add_argument(
        "--config", "-c",
        default="",
        help="Path to Task3Config JSON.",
    )
    run_parser.add_argument(
        "--run-name", "-n",
        default=f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        help="Run name (default: auto timestamp).",
    )
    args = parser.parse_args()
    if args.command != "run":
        parser.print_help()
        return

    from .config import resolve_config_path, load_task3_config
    from .pipeline import run_pipeline

    config_path = resolve_config_path(args.config)
    print(f"[task3-agent] Using config: {config_path}")

    cfg = load_task3_config(config_path)
    if not cfg.run_name or cfg.run_name == "default":
        cfg.run_name = args.run_name
    print(f"[task3-agent] Loaded config: run_name={cfg.run_name}, {len(cfg.sources)} sources")

    summary = run_pipeline(cfg)
    output_dir = cfg.output_dir
    print(f"[task3-agent] Pipeline complete. Summary: {output_dir}/agent/run_summary.json")
    print(f"[task3-agent] Result: {output_dir}/runs/{cfg.run_name}/result/README.md")


if __name__ == "__main__":
    main()
