#!/bin/bash
# End-to-end sweep: run all configs, extract, fit.
#
# Usage:
#   bash scripts/run_sweep.sh <ssh_host> [--cann-path /path] [--soc Ascend310P3]
#
# Example:
#   bash scripts/run_sweep.sh user@310p-device

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
HOST="${1:?Usage: run_sweep.sh <ssh_host>}"
shift

CANN_PATH="${CANN_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
SOC="${SOC:-Ascend310P3}"
REMOTE_DIR="${REMOTE_DIR:-/tmp/add_bench}"
LAUNCH_COUNT="${LAUNCH_COUNT:-100}"
RESULTS="$ROOT/results"

# Parse optional args
while [[ $# -gt 0 ]]; do
    case $1 in
        --cann-path) CANN_PATH="$2"; shift 2;;
        --soc) SOC="$2"; shift 2;;
        --remote-dir) REMOTE_DIR="$2"; shift 2;;
        --launch-count) LAUNCH_COUNT="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

echo "=== Add Intrinsic Benchmark Sweep ==="
echo "Host: $HOST | SOC: $SOC | CANN: $CANN_PATH"
echo "Launch count: $LAUNCH_COUNT"
echo ""

# Step 1: Validation cells
echo "--- Step 1: Validation cells ---"
python3 "$SCRIPT_DIR/run_bench.py" \
    --config "$ROOT/config/validation_cells.yaml" \
    --host "$HOST" --remote-dir "$REMOTE_DIR" \
    --cann-path "$CANN_PATH" --soc "$SOC" \
    --launch-count "$LAUNCH_COUNT" \
    --output "$RESULTS/raw/"

python3 "$SCRIPT_DIR/extract_results.py" \
    --raw-dir "$RESULTS/raw/" \
    --config "$ROOT/config/validation_cells.yaml" \
    --output "$RESULTS/validation_results.yaml"

echo ""
echo ">>> REVIEW validation_results.yaml before proceeding <<<"
echo ">>> Check: V1 BW ~256, V1 vs V2 (WAW), V3 instr count <<<"
read -p "Continue with main sweep? [y/N] " -n 1 -r
echo ""
[[ $REPLY =~ ^[Yy]$ ]] || exit 0

# Step 2: Main sweep per dtype
echo "--- Step 2: Main sweep ---"
for dtype in fp16 fp32 int32; do
    echo "  Sweeping $dtype ..."
    python3 "$SCRIPT_DIR/run_bench.py" \
        --config "$ROOT/config/sweep_${dtype}.yaml" \
        --host "$HOST" --remote-dir "$REMOTE_DIR" \
        --cann-path "$CANN_PATH" --soc "$SOC" \
        --launch-count "$LAUNCH_COUNT" --skip-build \
        --output "$RESULTS/raw/"

    python3 "$SCRIPT_DIR/extract_results.py" \
        --raw-dir "$RESULTS/raw/" \
        --config "$ROOT/config/sweep_${dtype}.yaml" \
        --output "$RESULTS/add_${dtype}_results.yaml"
done

# Step 3: Lambda sweep
echo "--- Step 3: Lambda_setup sweep ---"
python3 "$SCRIPT_DIR/run_bench.py" \
    --config "$ROOT/config/lambda_sweep.yaml" \
    --host "$HOST" --remote-dir "$REMOTE_DIR" \
    --cann-path "$CANN_PATH" --soc "$SOC" \
    --launch-count "$LAUNCH_COUNT" --skip-build \
    --output "$RESULTS/raw/"

python3 "$SCRIPT_DIR/extract_results.py" \
    --raw-dir "$RESULTS/raw/" \
    --config "$ROOT/config/lambda_sweep.yaml" \
    --output "$RESULTS/add_lambda_results.yaml"

# Step 4: Fit affine model
echo "--- Step 4: Fit affine model ---"
python3 "$SCRIPT_DIR/fit_affine.py" \
    --sweep-results \
        "$RESULTS/add_fp16_results.yaml" \
        "$RESULTS/add_fp32_results.yaml" \
        "$RESULTS/add_int32_results.yaml" \
    --lambda-results "$RESULTS/add_lambda_results.yaml" \
    --output "$RESULTS/fitted/add_affine_fit.yaml"

echo ""
echo "=== DONE ==="
echo "Results:  $RESULTS/"
echo "Fit:      $RESULTS/fitted/add_affine_fit.yaml"
