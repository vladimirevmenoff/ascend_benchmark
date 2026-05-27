#!/bin/bash
# End-to-end sweep — run directly on the 310P device.
#
# Usage:
#   cd vector_intrinsic_bench
#   bash scripts/run_sweep.sh
#
# Environment variables (optional):
#   CANN_PATH       default: /usr/local/Ascend/ascend-toolkit/latest
#   SOC             default: Ascend310P3
#   LAUNCH_COUNT    default: 100

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="$ROOT/scripts"
RESULTS="$ROOT/results"

CANN_PATH="${CANN_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
SOC="${SOC:-Ascend310P3}"
LAUNCH_COUNT="${LAUNCH_COUNT:-100}"

echo "=== Add Intrinsic Benchmark Sweep ==="
echo "SOC: $SOC | CANN: $CANN_PATH | Launches: $LAUNCH_COUNT"
echo ""

# Step 1: Validation cells (builds kernel)
echo "--- Step 1: Validation cells ---"
python3 "$SCRIPTS/run_bench.py" \
    --config "$ROOT/config/validation_cells.yaml" \
    --cann-path "$CANN_PATH" --soc "$SOC" \
    --launch-count "$LAUNCH_COUNT"

python3 "$SCRIPTS/extract_results.py" \
    --raw-dir "$RESULTS/raw/" \
    --config "$ROOT/config/validation_cells.yaml" \
    --output "$RESULTS/validation_results.yaml"

echo ""
echo ">>> Check results/validation_results.yaml <<<"
echo ">>> V1 BW ~256? V1 vs V2 (WAW)? V3 instr count? <<<"
read -p "Continue? [y/N] " -n 1 -r
echo ""
[[ $REPLY =~ ^[Yy]$ ]] || exit 0

# Step 2: Main sweep
echo "--- Step 2: Main sweep ---"
for dtype in fp16 fp32 int32; do
    echo "  $dtype ..."
    python3 "$SCRIPTS/run_bench.py" \
        --config "$ROOT/config/sweep_${dtype}.yaml" \
        --cann-path "$CANN_PATH" --soc "$SOC" \
        --launch-count "$LAUNCH_COUNT" --skip-build

    python3 "$SCRIPTS/extract_results.py" \
        --raw-dir "$RESULTS/raw/" \
        --config "$ROOT/config/sweep_${dtype}.yaml" \
        --output "$RESULTS/add_${dtype}_results.yaml"
done

# Step 3: Lambda sweep
echo "--- Step 3: Lambda sweep ---"
python3 "$SCRIPTS/run_bench.py" \
    --config "$ROOT/config/lambda_sweep.yaml" \
    --cann-path "$CANN_PATH" --soc "$SOC" \
    --launch-count "$LAUNCH_COUNT" --skip-build

python3 "$SCRIPTS/extract_results.py" \
    --raw-dir "$RESULTS/raw/" \
    --config "$ROOT/config/lambda_sweep.yaml" \
    --output "$RESULTS/add_lambda_results.yaml"

# Step 4: Fit
echo "--- Step 4: Fit affine model ---"
python3 "$SCRIPTS/fit_affine.py" \
    --sweep-results \
        "$RESULTS/add_fp16_results.yaml" \
        "$RESULTS/add_fp32_results.yaml" \
        "$RESULTS/add_int32_results.yaml" \
    --lambda-results "$RESULTS/add_lambda_results.yaml" \
    --output "$RESULTS/fitted/add_affine_fit.yaml"

echo ""
echo "=== DONE ==="
echo "Fit: $RESULTS/fitted/add_affine_fit.yaml"
