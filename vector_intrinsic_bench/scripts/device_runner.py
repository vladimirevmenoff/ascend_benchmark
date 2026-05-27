"""
Local device runner. Allocates GM, loads tiling, launches kernel.
Called by run_bench.py (or wrapped by msprof).

Usage:
    python3 device_runner.py --kernel=add_bench_fp16 \
                             --tiling=build/tiling.bin \
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


def run_with_torch_npu(kernel_name, tiling_path, gm_elems, np_dtype):
    import torch
    import torch_npu

    torch_dtype = {
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.int32: torch.int32,
    }[np_dtype]

    device = torch.device("npu:0")
    src0 = torch.randn(gm_elems, dtype=torch.float32).to(torch_dtype).to(device)
    src1 = torch.randn(gm_elems, dtype=torch.float32).to(torch_dtype).to(device)
    dst = torch.zeros(gm_elems, dtype=torch_dtype, device=device)

    with open(tiling_path, "rb") as f:
        tiling_bytes = f.read()

    torch_npu.npu_launch_kernel(
        kernel_name,
        [src0.data_ptr(), src1.data_ptr(), dst.data_ptr()],
        tiling_bytes,
        block_dim=1,
    )
    torch.npu.synchronize()


def run_with_acl(kernel_name, tiling_path, gm_elems, np_dtype):
    import acl

    ret = acl.init()
    context, ret = acl.rt.create_context(0)

    elem_size = np.dtype(np_dtype).itemsize
    buf_bytes = gm_elems * elem_size

    src0_dev, ret = acl.rt.malloc(buf_bytes, 0)
    src1_dev, ret = acl.rt.malloc(buf_bytes, 0)
    dst_dev, ret = acl.rt.malloc(buf_bytes, 0)

    src0_host = np.random.randn(gm_elems).astype(np_dtype)
    src1_host = np.random.randn(gm_elems).astype(np_dtype)
    acl.rt.memcpy(src0_dev, buf_bytes, src0_host.ctypes.data, buf_bytes, 1)
    acl.rt.memcpy(src1_dev, buf_bytes, src1_host.ctypes.data, buf_bytes, 1)

    with open(tiling_path, "rb") as f:
        tiling_bytes = f.read()
    tiling_dev, ret = acl.rt.malloc(len(tiling_bytes), 0)
    acl.rt.memcpy(tiling_dev, len(tiling_bytes),
                  np.frombuffer(tiling_bytes, dtype=np.uint8).ctypes.data,
                  len(tiling_bytes), 1)

    args_list = [src0_dev, src1_dev, dst_dev, tiling_dev]
    acl.rt.launch_kernel(kernel_name, 1, 1, args_list, None)
    acl.rt.synchronize_device(0)

    acl.rt.free(src0_dev)
    acl.rt.free(src1_dev)
    acl.rt.free(dst_dev)
    acl.rt.free(tiling_dev)
    acl.rt.destroy_context(context)
    acl.finalize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--tiling", required=True)
    parser.add_argument("--gm-elems", type=int, required=True)
    parser.add_argument("--dtype", required=True, choices=DTYPE_MAP.keys())
    parser.add_argument("--backend", default="torch_npu",
                        choices=["torch_npu", "acl"])
    args = parser.parse_args()

    np_dtype = DTYPE_MAP[args.dtype]

    if args.backend == "torch_npu":
        run_with_torch_npu(args.kernel, args.tiling, args.gm_elems, np_dtype)
    else:
        run_with_acl(args.kernel, args.tiling, args.gm_elems, np_dtype)


if __name__ == "__main__":
    main()
