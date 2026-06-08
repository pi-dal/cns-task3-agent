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
    run_parser.add_argument(
        "--input-dir",
        default=None,
        help="Input data directory (default: /saisdata or INPUT_DIR env).",
    )
    run_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: /saisresult or OUTPUT_DIR env).",
    )
    args = parser.parse_args()
    if args.command != "run":
        parser.print_help()
        return

    from .config import resolve_config_path, load_task3_config
    from .pipeline import run_pipeline
    from .runtime import resolve_input_dir, resolve_output_dir, pack_result

    config_path = resolve_config_path(args.config)
    print(f"[task3-agent] Using config: {config_path}")

    # Map input/output to container contract
    input_dir = resolve_input_dir(args.input_dir)
    output_dir = resolve_output_dir(args.output_dir)
    print(f"[task3-agent] Input: {input_dir}, Output: {output_dir}")

    cfg = load_task3_config(config_path)
    cfg.output_dir = output_dir
    cfg.input_dir = input_dir
    if not cfg.run_name or cfg.run_name == "default":
        cfg.run_name = args.run_name
    print(f"[task3-agent] Loaded config: run_name={cfg.run_name}, input_dir={input_dir}, {len(cfg.sources)} sources")

    summary = run_pipeline(cfg)
    print(f"[task3-agent] Pipeline complete. Summary: {output_dir}/agent/run_summary.json")
    print(f"[task3-agent] Result: {output_dir}/runs/{cfg.run_name}/result/README.md")

    # Pack final result
    result_zip = pack_result(output_dir, f"{output_dir}/output.zip")
    print(f"[task3-agent] Result zip: {result_zip}")


if __name__ == "__main__":
    main()
