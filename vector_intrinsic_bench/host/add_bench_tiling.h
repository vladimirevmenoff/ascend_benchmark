#ifndef ADD_BENCH_TILING_H
#define ADD_BENCH_TILING_H

#include <stdint.h>

struct AddBenchTiling {
    uint32_t repeatTimes;
    uint32_t nCalls;
    uint32_t blockLen;         // elements per repeat

    uint8_t dstBlkStride;      // 32B-unit stride between blocks within repeat
    uint8_t src0BlkStride;
    uint8_t src1BlkStride;
    uint8_t dstRepStride;      // 32B-unit stride between repeats (0 = broadcast)
    uint8_t src0RepStride;
    uint8_t src1RepStride;
    uint8_t pad[2];

    uint32_t nDstBufs;         // dst rotation count: 1 (WAW) or 8 (rotating)
    uint32_t rotElemStep;      // elements between dst rotations = 256 / sizeof(T)
    uint32_t srcElems;         // elements per src buffer = repeatTimes * blockLen
    uint32_t dstTotalElems;    // srcElems + (nDstBufs - 1) * rotElemStep
    uint32_t gmElems;          // elements per GM buffer (>= srcElems)
};

#endif
