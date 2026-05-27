#include "add_bench_kernel.h"

extern "C" __global__ __aicore__ void add_bench_fp16(
    GM_ADDR src0, GM_ADDR src1, GM_ADDR dst, GM_ADDR tiling)
{
    GET_TILING_DATA(t, tiling);
    AddBenchKernel<half> k;
    k.Init(src0, src1, dst, t);
    k.Process();
}

extern "C" __global__ __aicore__ void add_bench_fp32(
    GM_ADDR src0, GM_ADDR src1, GM_ADDR dst, GM_ADDR tiling)
{
    GET_TILING_DATA(t, tiling);
    AddBenchKernel<float> k;
    k.Init(src0, src1, dst, t);
    k.Process();
}

extern "C" __global__ __aicore__ void add_bench_int32(
    GM_ADDR src0, GM_ADDR src1, GM_ADDR dst, GM_ADDR tiling)
{
    GET_TILING_DATA(t, tiling);
    AddBenchKernel<int32_t> k;
    k.Init(src0, src1, dst, t);
    k.Process();
}
