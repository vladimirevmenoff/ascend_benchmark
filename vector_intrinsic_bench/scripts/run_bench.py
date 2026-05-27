"""
Run Add benchmark cells locally on a 310P device.

Usage:
    python3 scripts/run_bench.py --config config/validation_cells.yaml
    python3 scripts/run_bench.py --config config/sweep_fp16.yaml --skip-build
"""

import argparse
import os
import struct
import subprocess
import sys
import yaml


def pack_tiling(cell):
    """Pack AddBenchTiling struct to bytes matching the C struct layout."""
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


def source_env(cann_path):
    """Return shell prefix that sources CANN environment."""
    return f"source {cann_path}/set_env.sh && "


def build_kernel(root, cann_path, soc):
    build_dir = os.path.join(root, "build")
    os.makedirs(build_dir, exist_ok=True)
    cmd = (
        source_env(cann_path) +
        f"cd {build_dir} && "
        f"cmake {root} "
        f"-DCMAKE_ASCEND_TOOLKIT_PATH={cann_path} "
        f"-DASCEND_SOC_VERSION={soc} && "
        f"make -j4"
    )
    rc = subprocess.run(["bash", "-c", cmd])
    if rc.returncode != 0:
        print("BUILD FAILED", file=sys.stderr)
        sys.exit(1)
    print("Build OK")


def run_cell(root, cann_path, soc, cell, launch_count):
    """Run one benchmark cell via msprof locally."""
    cell_id = cell["id"]
    tiling_path = os.path.join(root, "build", "tiling.bin")
    msprof_out = os.path.join(root, "results", "raw", cell_id)
    os.makedirs(msprof_out, exist_ok=True)

    with open(tiling_path, "wb") as f:
        f.write(pack_tiling(cell))

    runner = os.path.join(root, "scripts", "device_runner.py")
    cmd = (
        source_env(cann_path) +
        f"msprof op "
        f"--soc-version={soc} "
        f"--aic-metrics=PipeUtilization,ResourceConflictRatio "
        f"--launch-count={launch_count} "
        f"--output={msprof_out} "
        f"-- python3 {runner} "
        f"--kernel={cell['kernel']} "
        f"--tiling={tiling_path} "
        f"--gm-elems={cell['gmElems']} "
        f"--dtype={cell['dtype']}"
    )
    rc = subprocess.run(["bash", "-c", cmd])
    return msprof_out if rc.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cann-path",
                        default="/usr/local/Ascend/ascend-toolkit/latest")
    parser.add_argument("--soc", default="Ascend310P3")
    parser.add_argument("--launch-count", type=int, default=100)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    cells = config["cells"]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print(f"Running {len(cells)} cells locally ({args.soc})")

    if not args.skip_build:
        build_kernel(root, args.cann_path, args.soc)

    ok, fail = 0, 0
    for i, cell in enumerate(cells):
        print(f"[{i+1}/{len(cells)}] {cell['id']} ... ", end="", flush=True)
        result = run_cell(root, args.cann_path, args.soc, cell, args.launch_count)
        if result:
            ok += 1
            print("OK")
        else:
            fail += 1
            print("FAIL")

    print(f"\nDone. {ok}/{ok+fail} cells OK.")


if __name__ == "__main__":
    main()
