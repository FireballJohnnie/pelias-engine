// File: <project>\moe_track\pelias_serve\zc_mmvq_q4k_q6k.cuh
// Pass 2 / piece 1: the GPU half of a cold ("miss") expert. Kernels that compute the GPU's share of a routed expert's SwiGLU FFN straight from host memory
// (zero-copy, through the device pointers cold_weights_view.h hands out: ColdMat::dev_base / dev_expert()), for the two quantisations the Q4_K_M file uses:
// Q4_K (ggml type 12, 144 bytes per 256 weights) and Q6_K (type 14, 210 bytes per 256 weights).
//
// WHAT ONE LAUNCH SEQUENCE DOES FOR ONE LAYER (the split comes from miss_splitter.h: the GPU owns intermediate rows [0, rows) of each miss expert)
//   zc_gateup_swiglu  h[i] = silu(gate_i . x) * (up_i . x)             for i in [0, rows)             (reads gate/up rows i only)
//   zc_down_partial   ypart[r] = sum over k in [0, rows) of down[r][k] * h[k]   for all r in [0, HIDDEN)   (reads only the K-slice [0, rows) of every down row)
//   zc_add_partials   acc[r] += weight_e * ypart_e[r]
// The CPU does rows [rows, n_ff) the same way and handoff.cuh adds its partial; `rows` is a multiple of 256 (one K-quant super-block), so the K-slice of a down
// row is whole super-blocks and no byte of an expert is read by both sides.
//
// MATHS (a deliberate difference from ggml, read this). ggml's CPU path quantises the activation vector to 8 bits (Q8_K) and does integer dot products. These
// kernels keep the activation in float, dequantise each weight to float exactly as ggml's dequantize_row_q4_K / q6_K do (so every weight value is the same float
// ggml would produce) and accumulate in float. That is at least as accurate as the 8-bit path, but it is NOT bit-identical to what the CPU computes for its
// rows. The effect on the model's output is UNMEASURED; the way to measure it is the KL comparison against stock logits.
//
// THREAD LAYOUT. One warp computes one output row. Lane l handles elements {32*k + l : k = 0..7} of every 256-weight super-block, which is exactly the order
// the weights are packed in, so the 32 lanes read 32 consecutive bytes per load instruction (coalesced) and no weight is touched twice. The warp sums its lanes
// with a butterfly (all lanes end with the full sum). A Q6_K row is 420 bytes, so its super-blocks are only 2-byte aligned: every weight byte is therefore loaded
// as a byte; nothing here assumes more alignment than that.
// The per-lane arithmetic (zc_q4k_lane / zc_q6k_lane) is plain host+device code, so tests/test_cold_path.cu checks it on the CPU against a straight
// dequantise-and-dot, and emulates a warp on the CPU where there is no GPU.
//
// SPEED. Not measured by the author: this file has only been compiled with the CPU stand-in. tests/test_cold_path.cu reads whole experts through these
// kernels from pinned host memory and prints the achieved GB/s next to the 26.2 GB/s the checksum kernel reached on this machine; if the kernel is the slower of the
// two, the loads (all byte loads) are the first thing to change, e.g. stage each row slice through shared memory with 16-byte loads.
//
// PRECISION OF THE EXPECTED RESULT (what the test checks against, derived on paper, no tuned tolerance): a float dot product of n terms computed in any order
// differs from the exact one by at most gamma_n * sum|a_i x_i|, gamma_n = n*u / (1 - n*u), u = 2^-24 (Higham, Accuracy and Stability of Numerical Algorithms,
// section 3.1). expf is documented with up to 2 ulp of error (CUDA C++ Programming Guide, mathematical functions appendix); do not build with --use_fast_math.
#pragma once

#include "miss_splitter.h"
#include "pelias_defs.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(PELIAS_CUDA_SHIM)
#include "../lib/cuda_shim.h"
#elif defined(__CUDACC__)
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

#if defined(__CUDACC__)
#define PELIAS_ZC_HD __host__ __device__
#define PELIAS_ZC_UNROLL _Pragma("unroll")
#else
#define PELIAS_ZC_HD
#define PELIAS_ZC_UNROLL
#endif

namespace pelias {
namespace zc {

constexpr int kTypeQ4K = 12;              // ggml_type values
constexpr int kTypeQ6K = 14;
constexpr int kSuper = QK_K;              // 256 weights per super-block
constexpr int kQ4KBytes = 144;            // d, dmin (2 + 2), scales (12), qs (128)
constexpr int kQ6KBytes = 210;            // ql (128), qh (64), scales (16), d (2)
constexpr int kHiddenSb = HIDDEN / kSuper;     // super-blocks in a gate/up row (K = HIDDEN)
constexpr int kWarps = 8;                 // warps per block: one output row each
constexpr int kBlockThreads = kWarps * 32;

static_assert(HIDDEN % kSuper == 0 && EXPERT_FF % kSuper == 0, "K-quant rows need whole super-blocks");

PELIAS_ZC_HD inline int block_bytes(int type) { return type == kTypeQ4K ? kQ4KBytes : kQ6KBytes; }

PELIAS_ZC_HD inline uint16_t ld_u16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }

// IEEE half -> float, exact. Hardware conversion on the device, bit arithmetic on the host.
PELIAS_ZC_HD inline float half_to_float(uint16_t h) {
#if defined(__CUDA_ARCH__)
    return __half2float(__ushort_as_half(h));
#else
    const uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, mant = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {                                   // subnormal
            int e = -1;
            uint32_t m = mant;
            do { e++; m <<= 1; } while (!(m & 0x400u));
            f = (sign << 31) | (static_cast<uint32_t>(127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
#endif
}

// ggml's get_scale_min_k4: the 6-bit scale and min of sub-block j (0..7) of a Q4_K super-block, packed in its 12 scale bytes.
PELIAS_ZC_HD inline void scale_min_k4(int j, const uint8_t* q, int* d, int* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// One lane's share of a Q4_K super-block at `blk` (144 bytes) against x[0..256): the elements 32*k + lane, k = 0..7.
// Element 32*k + lane lives in qs[32*(k/2) + lane], low nibble for even k, high nibble for odd k, with sub-block scale/min number k.
PELIAS_ZC_HD inline float q4k_lane(const uint8_t* blk, const float* x, int lane) {
    const float d = half_to_float(ld_u16(blk));
    const float dmin = half_to_float(ld_u16(blk + 2));
    const uint8_t* sc = blk + 4;
    const uint8_t* qs = blk + 16;
    float acc = 0.0f;
    PELIAS_ZC_UNROLL
    for (int g = 0; g < 4; ++g) {
        const int q = qs[32 * g + lane];
        int s0, m0, s1, m1;
        scale_min_k4(2 * g, sc, &s0, &m0);
        scale_min_k4(2 * g + 1, sc, &s1, &m1);
        const float lo = (d * static_cast<float>(s0)) * static_cast<float>(q & 0xF) - dmin * static_cast<float>(m0);
        const float hi = (d * static_cast<float>(s1)) * static_cast<float>(q >> 4) - dmin * static_cast<float>(m1);
        acc += lo * x[64 * g + lane];
        acc += hi * x[64 * g + 32 + lane];
    }
    return acc;
}

// One lane's share of a Q6_K super-block at `blk` (210 bytes) against x[0..256): the elements 128*n + 32*m + lane, n = 0..1, m = 0..3.
PELIAS_ZC_HD inline float q6k_lane(const uint8_t* blk, const float* x, int lane) {
    const uint8_t* ql = blk;
    const uint8_t* qh = blk + 128;
    const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192);
    const float d = half_to_float(ld_u16(blk + 208));
    const int is = lane >> 4;
    float acc = 0.0f;
    PELIAS_ZC_UNROLL
    for (int n = 0; n < 2; ++n) {
        const int a = ql[64 * n + lane], b = ql[64 * n + 32 + lane], h = qh[32 * n + lane];
        const int q0 = ((a & 0xF) | (((h >> 0) & 3) << 4)) - 32;
        const int q1 = ((b & 0xF) | (((h >> 2) & 3) << 4)) - 32;
        const int q2 = ((a >> 4) | (((h >> 4) & 3) << 4)) - 32;
        const int q3 = ((b >> 4) | (((h >> 6) & 3) << 4)) - 32;
        const int8_t* s = sc + 8 * n + is;
        acc += (d * static_cast<float>(s[0]) * static_cast<float>(q0)) * x[128 * n + lane];
        acc += (d * static_cast<float>(s[2]) * static_cast<float>(q1)) * x[128 * n + 32 + lane];
        acc += (d * static_cast<float>(s[4]) * static_cast<float>(q2)) * x[128 * n + 64 + lane];
        acc += (d * static_cast<float>(s[6]) * static_cast<float>(q3)) * x[128 * n + 96 + lane];
    }
    return acc;
}

// Super-blocks [sb0, sb1) of one row, one lane's partial. x[0] is the first activation of super-block 0 (so a K-slice starting at sb0 > 0 still passes the whole vector).
PELIAS_ZC_HD inline float row_lane_partial(const uint8_t* row, int type, int sb0, int sb1, const float* x, int lane) {
    float acc = 0.0f;
    if (type == kTypeQ4K) {
        for (int sb = sb0; sb < sb1; ++sb) acc += q4k_lane(row + static_cast<size_t>(sb) * kQ4KBytes, x + sb * kSuper, lane);
    } else {
        for (int sb = sb0; sb < sb1; ++sb) acc += q6k_lane(row + static_cast<size_t>(sb) * kQ6KBytes, x + sb * kSuper, lane);
    }
    return acc;
}

// The 32 lanes of a warp, run one after the other on the CPU (the stand-in for a GPU in the sandbox tests; the sum order differs from the butterfly, which the
// derived error bound allows).
inline float row_dot_emulated(const uint8_t* row, int type, int sb0, int sb1, const float* x) {
    float acc = 0.0f;
    for (int lane = 0; lane < 32; ++lane) acc += row_lane_partial(row, type, sb0, sb1, x, lane);
    return acc;
}

// ---- job descriptions (passed to the kernels by value; MAX_COLD experts at most) ------------------------------------------------------------------------
struct ZcExpertJob {
    const uint8_t* gate;           // GPU-side addresses of THIS expert's matrices (ColdMat::dev_expert(e)); never the CPU addresses
    const uint8_t* up;
    const uint8_t* down;
    int32_t        gate_type, up_type, down_type;               // ggml type ids; 12 or 14
    int32_t        gate_row_bytes, up_row_bytes, down_row_bytes;
    int32_t        rows;           // the GPU's intermediate rows [0, rows): a multiple of 256 in 0..EXPERT_FF; 0 = nothing for the GPU
    float          weight;         // router weight, applied when the partial is added to the accumulator
};

struct ZcLayerJob {
    int32_t     n;                 // experts in e[]
    ZcExpertJob e[MAX_COLD];
};

#if defined(__CUDACC__) || defined(PELIAS_CUDA_SHIM)

// Warp-wide dot of super-blocks [sb0, sb1) of `row` with x. ALL 32 lanes must call it together; every lane returns the full sum.
__device__ __forceinline__ float row_dot_warp(const uint8_t* row, int type, int sb0, int sb1, const float* x, int lane) {
    float acc = row_lane_partial(row, type, sb0, sb1, x, lane);
PELIAS_ZC_UNROLL
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    return acc;
}

// Launch: grid (ceil(EXPERT_FF / kWarps), n_experts), block kBlockThreads. x: HIDDEN floats; h: n_experts * EXPERT_FF floats (row-major by expert slot).
// g_out / u_out (optional, may be null): the raw gate and up dot products, same layout as h, for tests.
__global__ void zc_gateup_swiglu(ZcLayerJob job, const float* __restrict__ x, float* __restrict__ h, float* __restrict__ g_out, float* __restrict__ u_out) {
    const int e = static_cast<int>(blockIdx.y);
    if (e >= job.n) return;
    const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
    const int i = static_cast<int>(blockIdx.x) * kWarps + warp;
    const ZcExpertJob& J = job.e[e];
    if (i >= J.rows) return;                                   // the whole warp leaves together (i is the same for all its lanes)
    const float g = row_dot_warp(J.gate + static_cast<size_t>(i) * static_cast<size_t>(J.gate_row_bytes), J.gate_type, 0, kHiddenSb, x, lane);
    const float u = row_dot_warp(J.up + static_cast<size_t>(i) * static_cast<size_t>(J.up_row_bytes), J.up_type, 0, kHiddenSb, x, lane);
    if (lane == 0) {
        const float s = g / (1.0f + expf(-g));                 // silu
        h[e * EXPERT_FF + i] = s * u;
        if (g_out) g_out[e * EXPERT_FF + i] = g;
        if (u_out) u_out[e * EXPERT_FF + i] = u;
    }
}

// Launch: grid (HIDDEN / kWarps, n_experts), block kBlockThreads. ypart: n_experts * HIDDEN floats (unweighted partial of the K-slice [0, rows)).
__global__ void zc_down_partial(ZcLayerJob job, const float* __restrict__ h, float* __restrict__ ypart) {
    const int e = static_cast<int>(blockIdx.y);
    if (e >= job.n) return;
    const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
    const int r = static_cast<int>(blockIdx.x) * kWarps + warp;
    const ZcExpertJob& J = job.e[e];
    if (J.rows <= 0 || r >= HIDDEN) return;
    const float y = row_dot_warp(J.down + static_cast<size_t>(r) * static_cast<size_t>(J.down_row_bytes), J.down_type, 0, J.rows / kSuper, h + e * EXPERT_FF, lane);
    if (lane == 0) ypart[e * HIDDEN + r] = y;
}

// Launch: grid HIDDEN / kBlockThreads, block kBlockThreads. acc[r] += sum_e weight_e * ypart_e[r] over the experts that have GPU rows, in slot order.
__global__ void zc_add_partials(ZcLayerJob job, const float* __restrict__ ypart, float* __restrict__ acc) {
    const int r = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (r >= HIDDEN) return;
    float s = acc[r];
    for (int e = 0; e < job.n; ++e)
        if (job.e[e].rows > 0) s += job.e[e].weight * ypart[e * HIDDEN + r];
    acc[r] = s;
}

#endif  // device half

}  // namespace zc
}  // namespace pelias
