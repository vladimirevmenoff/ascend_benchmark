"""
Run Add benchmark cells on a 310P device via SSH.

Usage:
    python run_bench.py --config config/validation_cells.yaml \
                        --host <ssh_host> \
                        --remote-dir /home/user/bench \
                        --cann-path /usr/local/Ascend/ascend-toolkit/latest \
                        --soc Ascend310P3 \
                        --output results/raw/
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import yaml


def pack_tiling(cell):
    """Pack AddBenchTiling struct to bytes matching the C struct layout."""
    # struct AddBenchTiling {
    #   uint32 repeatTimes, nCalls, blockLen;
    #   uint8 dstBlk, src0Blk, src1Blk, dstRep, src0Rep, src1Rep, pad[2];
    #   uint32 nDstBufs, rotElemStep, srcElems, dstTotalElems, gmElems;
    # }
    return struct.pack(
        "<III BBBBBB BB IIIII",
        cell["repeatTimes"],
        cell["nCalls"],
        cell["blockLen"],
        cell["dstBlkStride"],
        cell["src0BlkStride"],
        cell["src1BlkStride"],
        cell["dstRepStride"],
        cell["src0RepStride"],
        cell["src1RepStride"],
        0, 0,  # pad
        cell["nDstBufs"],
        cell["rotElemStep"],
        cell["srcElems"],
        cell["dstTotalElems"],
        cell["gmElems"],
    )


def ssh_cmd(host, cmd):
    """Run command on remote host, return (returncode, stdout, stderr)."""
    r = subprocess.run(
        ["ssh", host, cmd],
        capture_output=True, text=True, timeout=300)
    return r.returncode, r.stdout, r.stderr


def rsync_to(host, local, remote_dir):
    subprocess.run(
        ["rsync", "-az", local, f"{host}:{remote_dir}/"],
        check=True, timeout=120)


def rsync_from(host, remote_path, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    subprocess.run(
        ["rsync", "-az", f"{host}:{remote_path}", local_dir],
        check=True, timeout=120)


def build_on_device(host, remote_dir, cann_path, soc):
    """Compile kernel on device."""
    build_cmd = f"""
cd {remote_dir} && \
source {cann_path}/set_env.sh && \
mkdir -p build && cd build && \
cmake .. -DCMAKE_ASCEND_TOOLKIT_PATH={cann_path} \
         -DASCEND_SOC_VERSION={soc} && \
make -j4
"""
    rc, out, err = ssh_cmd(host, build_cmd)
    if rc != 0:
        print(f"BUILD FAILED:\n{err}", file=sys.stderr)
        sys.exit(1)
    print("Build OK")


def run_cell(host, remote_dir, cann_path, soc, cell, launch_count=100):
    """Run one benchmark cell via msprof on device. Returns msprof output path."""
    cell_id = cell["id"]
    kernel = cell["kernel"]
    tiling_bytes = pack_tiling(cell)
    tiling_hex = tiling_bytes.hex()

    # Write tiling to remote temp file
    tiling_cmd = f"echo '{tiling_hex}' | xxd -r -p > {remote_dir}/tiling.bin"
    ssh_cmd(host, tiling_cmd)

    # Run with msprof
    msprof_out = f"{remote_dir}/msprof_out/{cell_id}"
    run_cmd = f"""
source {cann_path}/set_env.sh && \
mkdir -p {msprof_out} && \
msprof op \
    --soc-version={soc} \
    --aic-metrics=PipeUtilization,ResourceConflictRatio \
    --launch-count={launch_count} \
    --output={msprof_out} \
    -- python3 {remote_dir}/device_runner.py \
        --kernel={kernel} \
        --tiling={remote_dir}/tiling.bin \
        --gm-elems={cell['gmElems']} \
        --dtype={cell['dtype']}
"""
    rc, out, err = ssh_cmd(host, run_cmd)
    if rc != 0:
        print(f"CELL {cell_id} FAILED:\n{err}", file=sys.stderr)
        return None
    return msprof_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", required=True, help="SSH host for 310P device")
    parser.add_argument("--remote-dir", default="/tmp/add_bench")
    parser.add_argument("--cann-path",
                        default="/usr/local/Ascend/ascend-toolkit/latest")
    parser.add_argument("--soc", default="Ascend310P3")
    parser.add_argument("--output", default="results/raw/")
    parser.add_argument("--launch-count", type=int, default=100)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    cells = config["cells"]
    print(f"Running {len(cells)} cells on {args.host} ({args.soc})")

    # Sync kernel sources to device
    bench_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rsync_to(args.host, bench_root + "/", args.remote_dir)

    if not args.skip_build:
        build_on_device(args.host, args.remote_dir, args.cann_path, args.soc)

    results = []
    for i, cell in enumerate(cells):
        print(f"[{i+1}/{len(cells)}] {cell['id']} ... ", end="", flush=True)
        msprof_path = run_cell(
            args.host, args.remote_dir, args.cann_path, args.soc,
            cell, args.launch_count)
        if msprof_path:
            results.append({"cell": cell, "msprof_path": msprof_path})
            print("OK")
        else:
            print("FAIL")

    # Pull msprof results back
    for r in results:
        local_out = os.path.join(args.output, r["cell"]["id"])
        rsync_from(args.host, r["msprof_path"] + "/", local_out)

    summary_path = os.path.join(args.output, "run_summary.yaml")
    with open(summary_path, "w") as f:
        yaml.dump({
            "host": args.host,
            "soc": args.soc,
            "launch_count": args.launch_count,
            "cells_total": len(cells),
            "cells_ok": len(results),
            "cells_failed": len(cells) - len(results),
        }, f)
    print(f"\nDone. {len(results)}/{len(cells)} cells OK. Results in {args.output}")


if __name__ == "__main__":
    main()
