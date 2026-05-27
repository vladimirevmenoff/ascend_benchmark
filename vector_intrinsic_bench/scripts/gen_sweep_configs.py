"""Generate sweep config YAML files for Add intrinsic benchmarking."""

import yaml
import os

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "config")

DTYPES = {
    "fp16": {"sizeof": 2, "kernel": "add_bench_fp16", "full_blocklen": 128},
    "fp32": {"sizeof": 4, "kernel": "add_bench_fp32", "full_blocklen": 64},
    "int32": {"sizeof": 4, "kernel": "add_bench_int32", "full_blocklen": 64},
}

REPEAT_VALUES = [1, 4, 16, 64, 255]
NCALLS_SWEEP = [1, 8, 32, 128]
NCALLS_DEFAULT = 128
NDSTBUFS_ROTATING = 8
NDSTBUFS_SAME = 1


def make_strides(full_blocklen, half, broadcast_src1):
    """Compute BinaryRepeatParams for a config."""
    if half:
        blocklen = full_blocklen // 2
        rep_stride = 4  # 128B / 32B = 4 blocks per repeat
    else:
        blocklen = full_blocklen
        rep_stride = 8  # 256B / 32B = 8 blocks per repeat

    return {
        "blockLen": blocklen,
        "dstBlkStride": 1,
        "src0BlkStride": 1,
        "src1BlkStride": 1,
        "dstRepStride": rep_stride,
        "src0RepStride": rep_stride,
        "src1RepStride": 0 if broadcast_src1 else rep_stride,
    }


def compute_buffer_sizes(dtype_info, repeat, strides, n_dst_bufs):
    """Compute UB buffer element counts."""
    elem_size = dtype_info["sizeof"]
    rot_elem_step = 256 // elem_size  # elements per 256B rotation offset
    src_elems = repeat * strides["blockLen"]
    src1_elems = strides["blockLen"] if strides["src1RepStride"] == 0 else src_elems
    dst_total_elems = src_elems + (n_dst_bufs - 1) * rot_elem_step
    gm_elems = max(src_elems, dst_total_elems)
    # Round up to 32B alignment
    gm_elems = ((gm_elems * elem_size + 31) // 32) * 32 // elem_size
    return {
        "rotElemStep": rot_elem_step,
        "srcElems": src_elems,
        "dstTotalElems": dst_total_elems,
        "gmElems": gm_elems,
    }


def make_cell(cell_id, dtype_name, repeat, n_calls, n_dst_bufs,
              broadcast_src1, half_blocklen):
    dtype_info = DTYPES[dtype_name]
    strides = make_strides(dtype_info["full_blocklen"], half_blocklen,
                           broadcast_src1)
    bufs = compute_buffer_sizes(dtype_info, repeat, strides, n_dst_bufs)

    return {
        "id": cell_id,
        "kernel": dtype_info["kernel"],
        "dtype": dtype_name,
        "repeatTimes": repeat,
        "nCalls": n_calls,
        "nDstBufs": n_dst_bufs,
        **strides,
        **bufs,
    }


def gen_validation_cells():
    cells = []
    # V1: theory anchor — fp16, rotating dst, contiguous, full
    cells.append(make_cell(
        "V1-theory-anchor", "fp16", 64, 100, 8, False, False))
    # V2: WAW check — same as V1 but single dst
    cells.append(make_cell(
        "V2-waw-check", "fp16", 64, 100, 1, False, False))
    # V3: instruction count — same as V1 (verify V-instr count = 100)
    cells.append(make_cell(
        "V3-instr-count", "fp16", 64, 100, 8, False, False))
    return {"cells": cells, "description": "Validation cells — run before main sweep"}


def gen_main_sweep(dtype_name):
    cells = []
    for repeat in REPEAT_VALUES:
        for broadcast in [False, True]:
            for half in [False, True]:
                stride_tag = "bcast" if broadcast else "contig"
                blen_tag = "half" if half else "full"
                cell_id = f"add-{dtype_name}-rep{repeat}-{stride_tag}-{blen_tag}"
                cells.append(make_cell(
                    cell_id, dtype_name, repeat, NCALLS_DEFAULT,
                    NDSTBUFS_ROTATING, broadcast, half))
    return {
        "cells": cells,
        "description": f"Main sweep for Add {dtype_name} — "
                       f"{len(cells)} cells"
    }


def gen_lambda_sweep():
    cells = []
    for dtype_name in DTYPES:
        for n_calls in NCALLS_SWEEP:
            cell_id = f"lambda-add-{dtype_name}-ncalls{n_calls}"
            cells.append(make_cell(
                cell_id, dtype_name, 64, n_calls,
                NDSTBUFS_ROTATING, False, False))
    return {
        "cells": cells,
        "description": f"Lambda_setup extraction — nCalls sweep at repeat=64, "
                       f"{len(cells)} cells"
    }


def write_yaml(data, filename):
    path = os.path.join(OUTDIR, filename)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"  {path} ({len(data['cells'])} cells)")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print("Generating sweep configs:")

    write_yaml(gen_validation_cells(), "validation_cells.yaml")
    for dtype_name in DTYPES:
        write_yaml(gen_main_sweep(dtype_name), f"sweep_{dtype_name}.yaml")
    write_yaml(gen_lambda_sweep(), "lambda_sweep.yaml")

    total = 3 + sum(20 for _ in DTYPES) + len(DTYPES) * len(NCALLS_SWEEP)
    print(f"\nTotal: {total} cells across all configs")


if __name__ == "__main__":
    main()
