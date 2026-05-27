# Add Intrinsic Benchmark — Test Design

## Critical Design Decisions (from advisor review)

### 1. WAW Serialization Problem

Naive loop `Add(dst, src0, src1, ...)` × N_CALLS writes same dst every call.
V-pipe is in-order — back-to-back WAW on same address may serialize at writeback.
This measures **latency**, not **throughput**. Real kernels stream to different dst each iteration.

**Fix**: Rotate through K dst buffers: `Add(dst[i % K], src0, src1, ...)`.
K=4 minimum (matches pipeline depth). K=8 for safety.

**Validation cell**: Compare rotating-dst vs same-dst at (fp16, repeat=64, N_CALLS=100).
If BW differs → WAW is real, entire sweep uses rotating dst.
If BW same → V-pipe handles WAW, can simplify.

### 2. UB Bandwidth Baseline

Single Add: reads 2×256B + writes 1×256B = 768B UB traffic per repeat.
If UB r/w BW < 768 B/cyc, we're measuring UB, not compute.

**Fix**: Run a UB-copy-only reference kernel first (DataCopy UB→UB).
This anchors whether Add is UB-bound or compute-bound.

### 3. Compiler Folding Guard

Verify from msprof instruction trace that issued V-instructions = N_CALLS × repeatTimes.
If fewer → compiler folded the loop → measurement invalid.

### 4. Always Form B

Use explicit `Add(dst, src0, src1, blockLen, repeatTimes, repeatParams)`.
Form A loses control of sweep axes. Form A goes in one sanity row only.

---

## Kernel Architecture

### Tiling Data (host → kernel)

```cpp
struct AddBenchTiling {
    // Sweep parameters
    uint32_t repeatTimes;      // 1..255
    uint32_t nCalls;           // how many Add calls in measurement loop
    uint32_t blockLen;         // elements per repeat (full or partial)

    // Stride config (Form B params)
    uint8_t dstBlkStride;      // block stride (32B units)
    uint8_t src0BlkStride;
    uint8_t src1BlkStride;
    uint8_t dstRepStride;      // repeat stride (32B units), 0=broadcast
    uint8_t src0RepStride;
    uint8_t src1RepStride;

    // Buffer layout
    uint32_t nDstBufs;         // rotating dst count: 1 (same-dst) or 8 (rotating)
    uint32_t elementsToFill;   // for setup: elements per buffer to prefill
    uint32_t ubBytesPerBuf;    // bytes per UB buffer region

    // For GM I/O (setup + teardown)
    uint32_t totalGmElements;
};
```

### UB Buffer Layout

For 8 rotating dst buffers + 2 src buffers = 10 regions.
Each region 8KB (= 255 repeats × 256B/repeat max, rounded up to 8KB).
Total UB needed: 10 × 8KB = 80KB. 310P UB is 192KB — fits easily.

Spacing: 8KB per region (>> 1536B bank conflict period → no conflicts).

```
UB Layout:
  [0x0000] src0     (8KB)
  [0x2000] src1     (8KB)
  [0x4000] dst[0]   (8KB)
  [0x6000] dst[1]   (8KB)
  [0x8000] dst[2]   (8KB)
  [0xA000] dst[3]   (8KB)
  [0xC000] dst[4]   (8KB)
  [0xE000] dst[5]   (8KB)
  [0x10000] dst[6]  (8KB)
  [0x12000] dst[7]  (8KB)

Offsets between any two regions: 8KB = 8192B
8192 % 1536 = 3584 ≠ 0 → no bank conflicts ✓
```

### Kernel Class

```cpp
#include "kernel_operator.h"
using namespace AscendC;

// T = half / float / int32_t — compile-time dtype selection
template <typename T>
class AddBenchKernel {
public:
    __aicore__ inline AddBenchKernel(TPipe* p) : pipe(p) {}

    __aicore__ inline void Init(
        GM_ADDR gm_src0,     // pre-filled source data in GM
        GM_ADDR gm_src1,
        GM_ADDR gm_dst,      // output sink (prevents optimize-away)
        GM_ADDR gm_tiling)
    {
        GET_TILING_DATA(tiling, gm_tiling);
        repeatTimes  = tiling.repeatTimes;
        nCalls       = tiling.nCalls;
        blockLen     = tiling.blockLen;
        nDstBufs     = tiling.nDstBufs;
        elementsToFill = tiling.elementsToFill;
        ubBytesPerBuf  = tiling.ubBytesPerBuf;

        // Copy stride params
        rp.dstBlkStride  = tiling.dstBlkStride;
        rp.src0BlkStride = tiling.src0BlkStride;
        rp.src1BlkStride = tiling.src1BlkStride;
        rp.dstRepStride  = tiling.dstRepStride;
        rp.src0RepStride = tiling.src0RepStride;
        rp.src1RepStride = tiling.src1RepStride;

        // GM setup
        gmSrc0.SetGlobalBuffer((__gm__ T*)gm_src0, tiling.totalGmElements);
        gmSrc1.SetGlobalBuffer((__gm__ T*)gm_src1, tiling.totalGmElements);
        gmDst.SetGlobalBuffer((__gm__ T*)gm_dst, tiling.totalGmElements);

        // Allocate UB: 2 src + nDstBufs dst regions
        uint32_t totalUbBytes = (2 + nDstBufs) * ubBytesPerBuf;
        pipe->InitBuffer(ubBuf, totalUbBytes);
    }

    __aicore__ inline void Process() {
        // --- PHASE 1: SETUP (outside measurement) ---
        LocalTensor<T> ub = ubBuf.Get<T>();
        LocalTensor<T> src0 = ub;                                    // offset 0
        LocalTensor<T> src1 = ub[ubBytesPerBuf / sizeof(T)];        // offset 8KB
        // dst buffers start at offset 2 * ubBytesPerBuf
        // dst[i] = ub[(2 + i) * ubBytesPerBuf / sizeof(T)]

        // Fill src0 and src1 from GM
        DataCopy(src0, gmSrc0, elementsToFill);
        DataCopy(src1, gmSrc1, elementsToFill);
        PipeBarrier<PIPE_ALL>();  // wait for MTE2 to complete

        // --- PHASE 2: MEASUREMENT (Add in tight loop) ---
        for (uint32_t c = 0; c < nCalls; c++) {
            uint32_t dstIdx = c % nDstBufs;
            LocalTensor<T> dst = ub[(2 + dstIdx) * ubBytesPerBuf / sizeof(T)];
            Add(dst, src0, src1, blockLen, repeatTimes, rp);
        }
        PipeBarrier<PIPE_V>();  // wait for all V-pipe to complete

        // --- PHASE 3: TEARDOWN (write back, prevents optimize-away) ---
        // Write last dst buffer to GM
        LocalTensor<T> lastDst = ub[(2 + ((nCalls - 1) % nDstBufs)) * ubBytesPerBuf / sizeof(T)];
        DataCopy(gmDst, lastDst, elementsToFill);
    }

private:
    TPipe* pipe;
    GlobalTensor<T> gmSrc0, gmSrc1, gmDst;
    TBuf<TPosition::VECCALC> ubBuf;
    BinaryRepeatParams rp;
    uint32_t repeatTimes, nCalls, blockLen;
    uint32_t nDstBufs, elementsToFill, ubBytesPerBuf;
};
```

### Entry Points (one per dtype)

```cpp
extern "C" __global__ __aicore__ void add_bench_fp16(
    GM_ADDR src0, GM_ADDR src1, GM_ADDR dst, GM_ADDR tiling) {
    TPipe p;
    AddBenchKernel<half> k(&p);
    k.Init(src0, src1, dst, tiling);
    k.Process();
}

extern "C" __global__ __aicore__ void add_bench_fp32(
    GM_ADDR src0, GM_ADDR src1, GM_ADDR dst, GM_ADDR tiling) {
    TPipe p;
    AddBenchKernel<float> k(&p);
    k.Init(src0, src1, dst, tiling);
    k.Process();
}

extern "C" __global__ __aicore__ void add_bench_int32(
    GM_ADDR src0, GM_ADDR src1, GM_ADDR dst, GM_ADDR tiling) {
    TPipe p;
    AddBenchKernel<int32_t> k(&p);
    k.Init(src0, src1, dst, tiling);
    k.Process();
}
```

---

## Sweep Specification

### Validation Phase (run first, 4 cells)

These gate the entire sweep. If they fail, fix the methodology before proceeding.

| ID | Purpose | dtype | repeat | nCalls | nDstBufs | strides | Expected |
|----|---------|-------|--------|--------|----------|---------|----------|
| V1 | Theory anchor | fp16 | 64 | 100 | 8 (rotating) | contiguous, full blockLen | BW ≈ 256 B/cyc |
| V2 | WAW check | fp16 | 64 | 100 | 1 (same-dst) | contiguous, full blockLen | Compare to V1 |
| V3 | Instruction count | fp16 | 64 | 100 | 8 | contiguous, full blockLen | msprof V-instr count = 100 |
| V4 | UB BW baseline | fp16 | 64 | 100 | 8 | — (UB copy kernel, no Add) | Establishes UB ceiling |

**Decision gate:**
- V1 BW ≈ 256 → measuring V-compute. Proceed.
- V1 BW << 256 and V4 BW ≈ V1 BW → measuring UB BW, not compute. Reduce buffer count or element count.
- V1 ≠ V2 → WAW serialization real. All sweep cells must use rotating dst.
- V1 = V2 → WAW not an issue. Can use nDstBufs=1 for simplicity.
- V3 count ≠ 100 → compiler folded. Add volatile/barrier between calls.

### Main Sweep (60 cells per dtype, 180 total)

**Axes:**

| Axis | Values | Count |
|------|--------|-------|
| dtype | fp16, fp32, int32 | 3 |
| repeatTimes | 1, 4, 16, 64, 255 | 5 |
| stride pattern | contiguous, broadcast-src1 | 2 |
| blockLen | full (256/sizeof(T)), half (128/sizeof(T)) | 2 |
| nCalls | fixed at 128 (large enough for stable measurement) | 1 |

60 cells per dtype = 5 × 2 × 2 × (+ nCalls not swept, fixed at 128).

Wait — nCalls was supposed to be a sweep axis for fitting λ_setup.
Revised: add nCalls sweep for ONE config per dtype to extract λ_setup.

### λ_setup Extraction (12 additional cells)

For each dtype, fix (repeatTimes=64, contiguous, full blockLen) and sweep nCalls:

| nCalls | 1 | 8 | 32 | 128 |

4 × 3 dtypes = 12 cells.

Fit: `total_v_cycles = λ_setup × nCalls + cycles_per_repeat × 64 × nCalls`
→ `total_v_cycles / nCalls = λ_setup + cycles_per_repeat × 64`
→ Plot total_v_cycles vs nCalls → slope = (λ_setup + cpr×64), intercept ≈ 0 if model is clean.

Actually: with varying nCalls, if the Add throughput is constant:
`total_v_cycles = overhead_fixed + nCalls × (λ_setup + cycles_per_repeat × repeatTimes)`

So `total_v_cycles` vs `nCalls` is linear. The slope gives per-call cost.
Then vary repeatTimes (from the main sweep) at fixed nCalls=128:
`per_call_cost = λ_setup + cycles_per_repeat × repeatTimes`
→ slope over repeatTimes gives `cycles_per_repeat`, intercept gives `λ_setup`.

### Stride Configs (concrete values)

**Contiguous, full blockLen (fp16 example):**
```
blockLen = 128 elements (= 256B / 2B per fp16)
dstBlkStride = 1, src0BlkStride = 1, src1BlkStride = 1
dstRepStride = 8, src0RepStride = 8, src1RepStride = 8
```

**Contiguous, full blockLen (fp32 example):**
```
blockLen = 64 elements (= 256B / 4B per fp32)
dstBlkStride = 1, src0BlkStride = 1, src1BlkStride = 1
dstRepStride = 8, src0RepStride = 8, src1RepStride = 8
```

**Broadcast src1, full blockLen (fp16):**
```
blockLen = 128 elements
dstBlkStride = 1, src0BlkStride = 1, src1BlkStride = 1
dstRepStride = 8, src0RepStride = 8, src1RepStride = 0  ← broadcast
```

**Contiguous, half blockLen (fp16):**
```
blockLen = 64 elements (= 128B / 2B)
dstBlkStride = 1, src0BlkStride = 1, src1BlkStride = 1
dstRepStride = 4, src0RepStride = 4, src1RepStride = 4  ← half stride
```

---

## Measurement Extraction

### From msprof simulator

```
msprof op simulator \
    --soc-version=Ascend310P3 \
    --aic-metrics=PipeUtilization \
    --output=bench_out/ \
    -- python3 run_add_bench.py --config <config.yaml>
```

Parse `OPPROF_*/simulator/core0/core0_instr_exe.csv`:
- Column `pipe`: filter for `V` (vector), `S` (scalar), `M` (memory)
- Column `cycles`: sum per pipe
- Column `call_count`: **verify** V call_count = nCalls (compiler folding guard)

### Derived metrics per cell

```yaml
cell_id: add-fp16-rep64-ncalls128-contig-full-rot8
raw:
  v_cycles: 8320
  s_cycles: 12
  mte_cycles: 450   # setup + teardown only
  wall_cycles: 8320  # max(v, s, mte) in measurement region
  v_instruction_count: 128  # must equal nCalls

derived:
  per_call_v_cycles: 65.0          # = v_cycles / nCalls
  per_repeat_v_cycles: 1.015       # = per_call_v_cycles / repeatTimes
  bw_bytes_per_cyc: 252.2          # = 256 / per_repeat_v_cycles
  total_ub_read_bytes: 4194304     # = nCalls × repeatTimes × 2 × 256
  total_ub_write_bytes: 2097152    # = nCalls × repeatTimes × 1 × 256
```

---

## File Structure

```
profiling/
└── vector_intrinsic_bench/
    ├── README.md                    # this design doc
    ├── kernel/
    │   ├── add_bench_kernel.h       # kernel class (template<T>)
    │   ├── add_bench_kernel.cpp     # entry points (fp16/fp32/int32)
    │   └── ub_copy_baseline.h       # UB copy reference kernel (V4)
    ├── host/
    │   ├── add_bench_tiling.h       # AddBenchTiling struct
    │   └── add_bench_tiling.cpp     # tiling logic (trivial: fill struct from config)
    ├── config/
    │   ├── validation_cells.yaml    # V1-V4 configs
    │   ├── sweep_fp16.yaml          # 60 cells for fp16
    │   ├── sweep_fp32.yaml          # 60 cells for fp32
    │   ├── sweep_int32.yaml         # 60 cells for int32
    │   └── lambda_setup_sweep.yaml  # 12 cells for λ_setup extraction
    ├── scripts/
    │   ├── run_add_bench.py         # invokes kernel via acl, passes tiling
    │   ├── run_sweep.sh             # loops over config, calls msprof per cell
    │   ├── extract_results.py       # parses msprof CSV → results.yaml
    │   └── fit_affine.py            # OLS fit → (λ_setup, cycles_per_repeat) per (intrinsic, dtype)
    └── results/                     # gitignored, populated by runs
        ├── raw/                     # per-cell msprof output
        └── fitted/                  # affine model fits
```

---

## Execution Order

1. Build kernel (`bash build.sh --opkernel --soc=ascend310p3`)
2. Run V4 (UB baseline) → establishes UB BW ceiling
3. Run V1 (rotating dst) → theory anchor, expect ~256 B/cyc for fp16
4. Run V2 (same dst) → compare to V1, decide WAW policy
5. Run V3 → verify instruction count
6. **Decision gate**: analyze V1-V4, adjust sweep if needed
7. Run main sweep (180 cells)
8. Run λ_setup sweep (12 cells)
9. Run fit_affine.py → produce per-(intrinsic, dtype) constants
10. Validate: r² ≥ 0.95 for all fits

## Resolved

- **TBuf** (synchronous). No async pipelining needed in synthetic bench.
- **Single core**, `blockDim=1`. Cores are equal — no need to multi-core.
- **No warm-up**. V-pipe has no cache.
