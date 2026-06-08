#!/bin/bash
set -euo pipefail

# CNS Task 3 — container entrypoint
# Expected layout:
#   /saisdata/  — input dataset
#   /app/       — agent code (container)
#   /saisresult/ — output directory

INPUT_DIR="${INPUT_DIR:-/saisdata}"
OUTPUT_DIR="${OUTPUT_DIR:-/saisresult}"
RESULT_ZIP="${OUTPUT_DIR}/output.zip"

# Determine repo root: prefer /app (container), fallback to script dir (local dev)
if [ -d "/app" ]; then
    REPO_ROOT="/app"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$SCRIPT_DIR"
fi

echo "[run.sh] Starting CNS Task 3 agent..."
echo "[run.sh] Repo root: $REPO_ROOT"

# Fail-fast: check input
if [ ! -d "$INPUT_DIR" ]; then
    echo "[run.sh] ERROR: Input directory $INPUT_DIR not found."
    exit 1
fi
echo "[run.sh] Input directory: $INPUT_DIR"

# Ensure output directory
mkdir -p "$OUTPUT_DIR"

# Run agent pipeline
cd "$REPO_ROOT"
pdm run task3-agent run --config "$REPO_ROOT/configs/data_sources.example.json" --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR"

echo "[run.sh] Packing result..."
cd "$OUTPUT_DIR"
zip -r "$RESULT_ZIP" . -x "*.zip" > /dev/null 2>&1

if [ -f "$RESULT_ZIP" ]; then
    echo "[run.sh] Result: $RESULT_ZIP ($(stat -f%z "$RESULT_ZIP" 2>/dev/null || stat -c%s "$RESULT_ZIP" 2>/dev/null || echo "unknown") bytes)"
else
    echo "[run.sh] ERROR: Failed to create $RESULT_ZIP"
    exit 1
fi

echo "[run.sh] Done."
