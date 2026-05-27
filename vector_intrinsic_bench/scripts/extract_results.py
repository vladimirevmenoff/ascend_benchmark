"""
Parse msprof output CSVs into structured per-cell results.

Reads per-pipe cycle counts from msprof instruction execution trace.
Outputs results.yaml with per-cell measurements + derived metrics.

Usage:
    python extract_results.py --raw-dir results/raw/ \
                              --config config/sweep_fp16.yaml \
                              --output results/add_fp16_results.yaml
"""

import argparse
import csv
import glob
import os
import statistics
import yaml


KNOWN_PIPES = {"S", "V", "M", "Cube", "MTE1", "MTE2", "MTE3", "Vector"}


def find_instr_csv(msprof_dir):
    """Find the instruction execution CSV from msprof output."""
    patterns = [
        os.path.join(msprof_dir, "**", "core0_instr_exe.csv"),
        os.path.join(msprof_dir, "**", "*instr*.csv"),
    ]
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        if matches:
            return matches[0]
    return None


def find_task_csv(msprof_dir):
    """Find task-level CSV (device mode)."""
    patterns = [
        os.path.join(msprof_dir, "**", "task_time*.csv"),
        os.path.join(msprof_dir, "**", "*task*.csv"),
    ]
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        if matches:
            return matches[0]
    return None


def parse_instr_csv(csv_path):
    """Parse simulator instruction trace CSV. Returns per-pipe cycle totals."""
    pipe_cycles = {}
    pipe_instr_count = {}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pipe = row.get("pipe", "").strip()
            cycles = row.get("cycles", "0").strip()
            if pipe in KNOWN_PIPES and cycles:
                cyc = float(cycles)
                pipe_cycles[pipe] = pipe_cycles.get(pipe, 0) + cyc
                pipe_instr_count[pipe] = pipe_instr_count.get(pipe, 0) + 1

    return pipe_cycles, pipe_instr_count


def parse_task_csv(csv_path):
    """Parse device task CSV. Returns per-pipe cycle totals."""
    pipe_cycles = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_type = row.get("Task Type", "").strip()
            duration = row.get("Task Duration", "0").strip()
            if task_type and duration:
                pipe_cycles[task_type] = pipe_cycles.get(task_type, 0) + float(duration)
    return pipe_cycles, {}


def extract_cell(cell, raw_dir):
    """Extract measurements for one cell from its msprof output."""
    cell_id = cell["id"]
    cell_dir = os.path.join(raw_dir, cell_id)

    if not os.path.isdir(cell_dir):
        return {"id": cell_id, "status": "missing"}

    csv_path = find_instr_csv(cell_dir)
    source = "simulator"
    if csv_path:
        pipe_cycles, pipe_instr_count = parse_instr_csv(csv_path)
    else:
        csv_path = find_task_csv(cell_dir)
        source = "device"
        if csv_path:
            pipe_cycles, pipe_instr_count = parse_task_csv(csv_path)
        else:
            return {"id": cell_id, "status": "no_csv_found"}

    v_cycles = pipe_cycles.get("V", 0) + pipe_cycles.get("Vector", 0)
    s_cycles = pipe_cycles.get("S", 0)
    mte_cycles = sum(pipe_cycles.get(p, 0) for p in ["MTE1", "MTE2", "MTE3", "M"])
    wall_cycles = max(v_cycles, s_cycles, mte_cycles) if pipe_cycles else 0

    n_calls = cell["nCalls"]
    repeat = cell["repeatTimes"]
    blocklen = cell["blockLen"]
    dtype_size = {"fp16": 2, "fp32": 4, "int32": 4}[cell["dtype"]]
    bytes_per_repeat = blocklen * dtype_size  # NOT always 256 if half blockLen

    per_call_v = v_cycles / n_calls if n_calls > 0 else 0
    per_repeat_v = per_call_v / repeat if repeat > 0 else 0
    bw = bytes_per_repeat / per_repeat_v if per_repeat_v > 0 else float("inf")

    v_instr = pipe_instr_count.get("V", 0) + pipe_instr_count.get("Vector", 0)

    return {
        "id": cell_id,
        "status": "ok",
        "source": source,
        "config": {
            "dtype": cell["dtype"],
            "repeatTimes": repeat,
            "nCalls": n_calls,
            "blockLen": blocklen,
            "nDstBufs": cell["nDstBufs"],
            "src1RepStride": cell["src1RepStride"],
        },
        "raw": {
            "v_cycles": round(v_cycles, 1),
            "s_cycles": round(s_cycles, 1),
            "mte_cycles": round(mte_cycles, 1),
            "wall_cycles": round(wall_cycles, 1),
            "v_instruction_count": v_instr,
        },
        "derived": {
            "per_call_v_cycles": round(per_call_v, 2),
            "per_repeat_v_cycles": round(per_repeat_v, 3),
            "bw_bytes_per_cyc": round(bw, 1),
            "bytes_per_repeat": bytes_per_repeat,
        },
        "guards": {
            "v_instr_expected": n_calls,
            "v_instr_actual": v_instr,
            "instr_count_ok": v_instr == n_calls,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results = []
    ok = 0
    for cell in config["cells"]:
        r = extract_cell(cell, args.raw_dir)
        results.append(r)
        if r["status"] == "ok":
            ok += 1

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump({"results": results}, f, default_flow_style=False, sort_keys=False)

    print(f"Extracted {ok}/{len(results)} cells → {args.output}")

    # Print guard failures
    for r in results:
        if r["status"] == "ok" and not r["guards"]["instr_count_ok"]:
            print(f"  WARNING {r['id']}: V-instr count {r['guards']['v_instr_actual']}"
                  f" != expected {r['guards']['v_instr_expected']} — possible compiler folding")


if __name__ == "__main__":
    main()
