"""
Minimal device-side runner. Loads compiled kernel, allocates GM,
sets tiling, launches kernel. Invoked by msprof on the 310P device.

Usage (called by run_bench.py via SSH):
    python3 device_runner.py --kernel=add_bench_fp16 \
                             --tiling=tiling.bin \
                             --gm-elems=16384 \
                             --dtype=fp16
"""

import argparse
import numpy as np

DTYPE_MAP = {
    "fp16": np.float16,
    "fp32": np.float32,
    "int32": np.int32,
}


def run_acl(kernel_name, tiling_path, gm_elems, np_dtype):
    """Launch kernel via torch_npu / acl."""
    try:
        import torch
        import torch_npu
    except ImportError:
        raise RuntimeError("torch_npu not available on this device")

    device = torch.device("npu:0")

    src0 = torch.randn(gm_elems, dtype=torch.float32).to(np_dtype_to_torch(np_dtype)).to(device)
    src1 = torch.randn(gm_elems, dtype=torch.float32).to(np_dtype_to_torch(np_dtype)).to(device)
    dst = torch.zeros(gm_elems, dtype=np_dtype_to_torch(np_dtype)).to(device)

    with open(tiling_path, "rb") as f:
        tiling_bytes = f.read()

    torch_npu.npu_launch_kernel(
        kernel_name,
        [src0.data_ptr(), src1.data_ptr(), dst.data_ptr()],
        tiling_bytes,
        block_dim=1,
    )
    torch.npu.synchronize()


def np_dtype_to_torch(np_dtype):
    import torch
    return {
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.int32: torch.int32,
    }[np_dtype]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--tiling", required=True)
    parser.add_argument("--gm-elems", type=int, required=True)
    parser.add_argument("--dtype", required=True, choices=DTYPE_MAP.keys())
    args = parser.parse_args()

    np_dtype = DTYPE_MAP[args.dtype]
    run_acl(args.kernel, args.tiling, args.gm_elems, np_dtype)


if __name__ == "__main__":
    main()
