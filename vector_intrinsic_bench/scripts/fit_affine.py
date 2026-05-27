"""
Fit the V-pipe affine model from extracted benchmark results.

Model: T_V_per_call = λ_setup + cycles_per_repeat × repeatTimes

For each (dtype, stride_pattern, blockLen_class), fits λ_setup and
cycles_per_repeat via OLS on the main sweep data. Uses lambda_sweep
data for cross-validation.

Usage:
    python fit_affine.py \
        --sweep-results results/add_fp16_results.yaml \
        --lambda-results results/add_lambda_results.yaml \
        --output results/fitted/add_fp16_fit.yaml
"""

import argparse
import os
from collections import defaultdict
import yaml

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def ols_fit(x, y):
    """Simple OLS: y = a + b*x. Returns (intercept, slope, r_squared)."""
    if not HAS_NUMPY:
        return _ols_fit_pure(x, y)
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)
    if n < 2:
        return (y[0] if n == 1 else 0.0), 0.0, 0.0
    A = np.vstack([np.ones(n), x]).T
    result = np.linalg.lstsq(A, y, rcond=None)
    intercept, slope = result[0]
    y_pred = intercept + slope * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(intercept), float(slope), float(r2)


def _ols_fit_pure(x, y):
    """Fallback OLS without numpy."""
    n = len(x)
    if n < 2:
        return (y[0] if n == 1 else 0.0), 0.0, 0.0
    sx = sum(x)
    sy = sum(y)
    sxx = sum(xi * xi for xi in x)
    sxy = sum(xi * yi for xi, yi in zip(x, y))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return sy / n, 0.0, 0.0
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    y_mean = sy / n
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    ss_res = sum((yi - (intercept + slope * xi)) ** 2 for xi, yi in zip(x, y))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return intercept, slope, r2


def classify_cell(result):
    """Return grouping key for a result cell."""
    cfg = result["config"]
    stride = "broadcast" if cfg["src1RepStride"] == 0 else "contiguous"
    dtype_sizes = {"fp16": 2, "fp32": 4, "int32": 4}
    full_blocklen = 256 // dtype_sizes[cfg["dtype"]]
    blen_class = "full" if cfg["blockLen"] == full_blocklen else "half"
    return (cfg["dtype"], stride, blen_class)


def fit_group(cells):
    """Fit affine model for one (dtype, stride, blockLen) group.

    Uses repeatTimes as x-axis, per_call_v_cycles as y-axis.
    Model: per_call_v_cycles = λ_setup + cycles_per_repeat × repeatTimes
    """
    x = [c["config"]["repeatTimes"] for c in cells]
    y = [c["derived"]["per_call_v_cycles"] for c in cells]

    lambda_setup, cpr, r2 = ols_fit(x, y)

    bytes_per_repeat = cells[0]["derived"]["bytes_per_repeat"]
    bw_peak = bytes_per_repeat / cpr if cpr > 0 else float("inf")

    return {
        "lambda_setup_cyc": round(lambda_setup, 2),
        "cycles_per_repeat": round(cpr, 4),
        "bw_peak_bytes_per_cyc": round(bw_peak, 1),
        "r_squared": round(r2, 4),
        "n_datapoints": len(cells),
        "repeat_values": sorted(set(x)),
        "per_call_v_cycles": [round(yi, 2) for yi in y],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-results", required=True, nargs="+")
    parser.add_argument("--lambda-results", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Load all sweep results
    all_results = []
    for path in args.sweep_results:
        with open(path) as f:
            data = yaml.safe_load(f)
        all_results.extend(r for r in data["results"] if r["status"] == "ok")

    # Group by (dtype, stride, blockLen)
    groups = defaultdict(list)
    for r in all_results:
        key = classify_cell(r)
        groups[key].append(r)

    # Fit each group
    fits = {}
    for (dtype, stride, blen), cells in sorted(groups.items()):
        group_key = f"{dtype}_{stride}_{blen}"
        fit = fit_group(cells)
        fits[group_key] = fit

        flag = "" if fit["r_squared"] >= 0.95 else " *** LOW R² ***"
        print(f"{group_key}: λ={fit['lambda_setup_cyc']:.1f} cyc, "
              f"cpr={fit['cycles_per_repeat']:.3f}, "
              f"BW={fit['bw_peak_bytes_per_cyc']:.1f} B/cyc, "
              f"r²={fit['r_squared']:.4f}{flag}")

    # Lambda sweep cross-validation
    lambda_cv = None
    if args.lambda_results:
        with open(args.lambda_results) as f:
            ldata = yaml.safe_load(f)
        lambda_cells = [r for r in ldata["results"] if r["status"] == "ok"]

        # Group by dtype, fit total_v_cycles vs nCalls
        by_dtype = defaultdict(list)
        for r in lambda_cells:
            by_dtype[r["config"]["dtype"]].append(r)

        lambda_cv = {}
        for dtype, cells in sorted(by_dtype.items()):
            x = [c["config"]["nCalls"] for c in cells]
            y = [c["raw"]["v_cycles"] for c in cells]
            intercept, slope, r2 = ols_fit(x, y)
            lambda_cv[dtype] = {
                "overhead_fixed_cyc": round(intercept, 1),
                "per_call_cost_cyc": round(slope, 2),
                "r_squared": round(r2, 4),
            }
            print(f"Lambda CV {dtype}: fixed={intercept:.1f}, "
                  f"per_call={slope:.2f}, r²={r2:.4f}")

    output_data = {
        "intrinsic": "Add",
        "fits": fits,
    }
    if lambda_cv:
        output_data["lambda_crossval"] = lambda_cv

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

    print(f"\nFits written to {args.output}")

    # Quality gate
    bad_fits = [k for k, v in fits.items() if v["r_squared"] < 0.95]
    if bad_fits:
        print(f"\nWARNING: {len(bad_fits)} groups with r² < 0.95: {bad_fits}")
        print("Consider piecewise model or investigate outliers.")


if __name__ == "__main__":
    main()
