#ifndef ADD_BENCH_KERNEL_H
#define ADD_BENCH_KERNEL_H

#include "kernel_operator.h"

using namespace AscendC;

struct AddBenchTiling {
    uint32_t repeatTimes;
    uint32_t nCalls;
    uint32_t blockLen;

    uint8_t dstBlkStride;
    uint8_t src0BlkStride;
    uint8_t src1BlkStride;
    uint8_t dstRepStride;
    uint8_t src0RepStride;
    uint8_t src1RepStride;
    uint8_t pad[2];

    uint32_t nDstBufs;
    uint32_t rotElemStep;
    uint32_t srcElems;
    uint32_t dstTotalElems;
    uint32_t gmElems;
};

template <typename T>
class AddBenchKernel {
public:
    __aicore__ inline AddBenchKernel() {}

    __aicore__ inline void Init(
        GM_ADDR gm_src0,
        GM_ADDR gm_src1,
        GM_ADDR gm_dst,
        const AddBenchTiling& t)
    {
        repeatTimes = t.repeatTimes;
        nCalls      = t.nCalls;
        blockLen    = t.blockLen;
        nDstBufs    = t.nDstBufs;
        rotElemStep = t.rotElemStep;
        srcElems    = t.srcElems;

        rp.dstBlkStride  = t.dstBlkStride;
        rp.src0BlkStride = t.src0BlkStride;
        rp.src1BlkStride = t.src1BlkStride;
        rp.dstRepStride  = t.dstRepStride;
        rp.src0RepStride = t.src0RepStride;
        rp.src1RepStride = t.src1RepStride;

        gmSrc0.SetGlobalBuffer((__gm__ T*)gm_src0, t.gmElems);
        gmSrc1.SetGlobalBuffer((__gm__ T*)gm_src1, t.gmElems);
        gmDst.SetGlobalBuffer((__gm__ T*)gm_dst, t.gmElems);

        uint32_t src0Bytes = t.srcElems * sizeof(T);
        uint32_t src1Bytes = (t.src1RepStride == 0)
            ? blockLen * sizeof(T)
            : src0Bytes;
        uint32_t dstBytes = t.dstTotalElems * sizeof(T);
        uint32_t totalBytes = src0Bytes + src1Bytes + dstBytes;

        pipe.InitBuffer(ubBuf, totalBytes);
    }

    __aicore__ inline void Process() {
        LocalTensor<T> ub = ubBuf.template Get<T>();

        uint32_t src1Elems = (rp.src1RepStride == 0) ? blockLen : srcElems;

        LocalTensor<T> src0 = ub;
        LocalTensor<T> src1 = ub[srcElems];
        LocalTensor<T> dstBase = ub[srcElems + src1Elems];

        // --- SETUP: fill src buffers from GM ---
        uint32_t src0Blocks = srcElems * sizeof(T) / 32;
        uint32_t src1Blocks = src1Elems * sizeof(T) / 32;
        DataCopy(src0, gmSrc0, src0Blocks);
        DataCopy(src1, gmSrc1, src1Blocks);
        PipeBarrier<PIPE_ALL>();

        // --- MEASUREMENT: tight Add loop with dst rotation ---
        for (uint32_t c = 0; c < nCalls; c++) {
            uint32_t rotOffset = (c & (nDstBufs - 1)) * rotElemStep;
            Add(dstBase[rotOffset], src0, src1,
                blockLen, repeatTimes, rp);
        }
        PipeBarrier<PIPE_V>();

        // --- TEARDOWN: write last dst to GM (prevents optimize-away) ---
        uint32_t lastRot = ((nCalls - 1) & (nDstBufs - 1)) * rotElemStep;
        uint32_t dstBlocks = srcElems * sizeof(T) / 32;
        DataCopy(gmDst, dstBase[lastRot], dstBlocks);
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> ubBuf;
    GlobalTensor<T> gmSrc0, gmSrc1, gmDst;
    BinaryRepeatParams rp;
    uint32_t repeatTimes, nCalls, blockLen;
    uint32_t nDstBufs, rotElemStep, srcElems;
};

#endif
