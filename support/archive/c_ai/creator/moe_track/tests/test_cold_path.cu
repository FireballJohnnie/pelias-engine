// File: <project>\moe_track\tests\test_cold_path.cu
// End-to-end test of the cold path of ONE layer, the way the engine will run it, on pinned host memory:
//   miss_splitter.h  decides which intermediate rows of each miss expert the GPU takes and which the CPU takes
//   handoff.cuh      the GPU rings the doorbell with x and the CPU's tasks; a CPU worker thread answers with its partial; the GPU adds it (the 03 rev 2 protocol)
//   zc_mmvq_q4k_q6k.cuh   the GPU computes its rows straight from host memory through the device pointers
//   the CPU worker here is a plain float dequantise-and-dot (a stand-in for cold_pool.h, which needs ggml's headers): it is slow, and its speed means nothing.
// What it checks (all against an independent double-precision, split-free computation of the same experts):
//   1. the per-lane arithmetic of the kernels on the CPU: for every one of the 256 positions of random Q4_K and Q6_K blocks, a one-hot activation returns exactly
//      the dequantised weight (so every weight is read by exactly one lane, with the right scale and bits), and random activations agree within the derived bound
//   2. per scenario (0, 25, 37.5, 50, 100 % of the rows to the GPU; 1-4 miss experts): the GPU's gate/up dots, SwiGLU rows, down partials, the CPU's own values,
//      the final accumulator, and the final accumulator against the split-free reference
//   3. (GPU build) the speed at which the kernels read whole experts from pinned memory, next to the 26.2 GB/s the checksum-read kernel reached
// TOLERANCES ARE DERIVED, NOT TUNED. A float dot product of n terms, in any summation order, is within gamma_n * sum|a_i x_i| of the exact value, gamma_n = n u/(1-n u),
// u = 2^-24 (Higham, Accuracy and Stability of Numerical Algorithms, 3.1). The SwiGLU row h = silu(g) u adds expf (<= 2 ulp, CUDA C++ Programming Guide), one
// addition, one division and one multiplication: relative error <= 7 * 2^-24, rounded up to 2^-21. silu'(g) <= 1.1 (its maximum is 1.0998...), so for
// h = silu(g) u:  |dh| <= 1.1 (|u| + |du|) |dg| + |silu(g)| |du|  (plus the rounding above). These feed the final bound row by row. No speed limit is applied anywhere.
//
// Modes: default = synthetic random-but-valid Q4_K/Q6_K experts in mapped pinned memory (no model file needed). With --gguf FILE (and a build with
// -DPELIAS_COLDPATH_WITH_VIEW) the scenarios also run on a layer of the real model through cold_weights_view.
//
// Build on WSL (GPU run; the toolchain from the earlier runs):
//   R=$HOME/stuart_creator_engine/cuda_shim
//   $R/bin/nvcc -O2 -std=c++17 -arch=sm_89 -I$R/include --cudart shared -L$R/lib64 -Xlinker -rpath,$R/lib64 -Xcompiler -pthread test_cold_path.cu -I../pelias_serve -o ~/p1/t_cold_path
//   with the real model as well:  add  -DPELIAS_COLDPATH_WITH_VIEW -DPELIAS_COLD_VIEW_CUDA -x cu ../pelias_serve/cold_weights_view.cpp  (put -x cu before both sources)
// Run:    ~/p1/t_cold_path [--gguf FILE [--layer N]] [--no-throughput]       Exit 0 = every correctness check passed. Server off.
// Without a GPU (CPU stand-in, GPU stages emulated lane by lane; proves the harness, the maths and the bounds, NOT any device code):
//   g++ -O2 -std=c++17 -pthread -Wall -Wextra -Werror -x c++ -DPELIAS_CUDA_SHIM test_cold_path.cu -I../pelias_serve -o test_cold_path
// Syntax check of the GPU branch with the stand-in:  add -DPELIAS_COLDPATH_REAL -fsyntax-only
// standard headers first: the CPU shim defines __noinline__ as a macro, which breaks libstdc++ headers included after it
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "../lib/pl_cuda.h"

#include "handoff.cuh"
#include "miss_splitter.h"
#include "zc_mmvq_q4k_q6k.cuh"

#if defined(PELIAS_COLDPATH_WITH_VIEW)
#include "cold_weights_view.h"
#endif
#if defined(PELIAS_GGML_CHECK)
#include "ggml.h"
#endif

#if defined(__CUDACC__) || defined(PELIAS_COLDPATH_REAL)
#define PELIAS_COLDPATH_GPU 1
#else
#define PELIAS_COLDPATH_GPU 0
#endif

using namespace pelias;
using namespace pelias::handoff;

// =====================================================================================================================================================
// small utilities
// =====================================================================================================================================================
static int g_fail = 0;
static void check(const std::string& name, bool ok, const std::string& extra = std::string()) {
    std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", name.c_str(), (!ok && !extra.empty()) ? "   -> " : "", (!ok && !extra.empty()) ? extra.c_str() : "");
    std::fflush(stdout);
    if (!ok) ++g_fail;
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ULL + 0x1234567ULL) {}
    uint64_t next() { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return s * 0x2545F4914F6CDD1DULL; }
    uint32_t u32() { return static_cast<uint32_t>(next() >> 32); }
    float uni() { return static_cast<float>(u32() >> 8) / 16777216.0f; }                 // [0, 1)
    float sym() { return uni() * 2.0f - 1.0f; }                                         // [-1, 1)
};

static const double kU = 1.0 / 16777216.0;                                              // 2^-24
static double gam(double n) { return n * kU / (1.0 - n * kU); }
static const double kCh = 1.0 / 2097152.0;                                              // 2^-21, see the header comment
static const double kSiluSlope = 1.1;

static double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// =====================================================================================================================================================
// independent reference: dequantisation written out the way ggml's dequantize_row_q4_K / q6_K do it (sequential, no lane mapping), double-precision dots
// =====================================================================================================================================================
static float ref_h2f(uint16_t h) {
    const int sign = (h >> 15) & 1, ex = (h >> 10) & 0x1F, mant = h & 0x3FF;
    double v;
    if (ex == 0) v = std::ldexp(static_cast<double>(mant), -24);
    else if (ex == 31) v = mant ? NAN : INFINITY;
    else v = std::ldexp(static_cast<double>(1024 + mant), ex - 25);
    return static_cast<float>(sign ? -v : v);
}

static void ref_scale_min(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = static_cast<uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
    }
}

static void ref_dequant_q4k(const uint8_t* b, float* y) {
    const uint8_t* q = b + 16;
    const float d = ref_h2f(static_cast<uint16_t>(b[0] | (b[1] << 8)));
    const float mn = ref_h2f(static_cast<uint16_t>(b[2] | (b[3] << 8)));
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        ref_scale_min(is + 0, b + 4, &sc, &m);
        const float d1 = d * sc, m1 = mn * m;
        ref_scale_min(is + 1, b + 4, &sc, &m);
        const float d2 = d * sc, m2 = mn * m;
        for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
        q += 32;
        is += 2;
    }
}

static void ref_dequant_q6k(const uint8_t* b, float* y) {
    const uint8_t* ql = b;
    const uint8_t* qh = b + 128;
    const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192);
    const float d = ref_h2f(static_cast<uint16_t>(b[208] | (b[209] << 8)));
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = static_cast<int>((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = static_cast<int>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = static_cast<int>((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = static_cast<int>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l + 0] = d * sc[is + 0] * q1;
            y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3;
            y[l + 96] = d * sc[is + 6] * q4;
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

static void ref_dequant_block(int type, const uint8_t* b, float* y) {
    if (type == zc::kTypeQ4K) ref_dequant_q4k(b, y);
    else ref_dequant_q6k(b, y);
}

// float dequantise-and-dot over super-blocks [sb0, sb1) of a row, sequential float accumulation: the CPU worker's arithmetic
static float cpu_dot_float(const uint8_t* row, int type, int sb0, int sb1, const float* x) {
    float acc = 0.0f;
    float tmp[zc::kSuper];
    for (int sb = sb0; sb < sb1; ++sb) {
        ref_dequant_block(type, row + static_cast<size_t>(sb) * zc::block_bytes(type), tmp);
        for (int i = 0; i < zc::kSuper; ++i) acc += tmp[i] * x[sb * zc::kSuper + i];
    }
    return acc;
}

// =====================================================================================================================================================
// matrices: [n_experts][rows][row_bytes], one host address and one device address each
// =====================================================================================================================================================
struct Mat {
    const uint8_t* host = nullptr;
    const uint8_t* dev = nullptr;
    int            type = 0;
    int            rows = 0, cols = 0, n_experts = 0;
    size_t         row_bytes = 0, expert_stride = 0;
    const uint8_t* h_row(int e, int r) const { return host + static_cast<size_t>(e) * expert_stride + static_cast<size_t>(r) * row_bytes; }
    const uint8_t* h_expert(int e) const { return host + static_cast<size_t>(e) * expert_stride; }
    const uint8_t* d_expert(int e) const { return dev + static_cast<size_t>(e) * expert_stride; }
};

struct LayerMats {
    std::string name;
    Mat gate, up, down;
    std::vector<void*> to_free;          // mapped pinned allocations made by make_synth_layer
};

static uint16_t half_bits_from_positive(float f) {      // truncating encoder, normal range only (all that the synthetic data needs)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const int ex = static_cast<int>((u >> 23) & 0xFF) - 127 + 15;
    return static_cast<uint16_t>((ex << 10) | ((u >> 13) & 0x3FF));
}

static void fill_block(int type, uint8_t* b, Rng& rng) {
    const uint16_t d = half_bits_from_positive(1e-4f + 4e-4f * rng.uni());
    if (type == zc::kTypeQ4K) {
        const uint16_t dm = half_bits_from_positive(1e-4f + 4e-4f * rng.uni());
        b[0] = static_cast<uint8_t>(d & 0xFF); b[1] = static_cast<uint8_t>(d >> 8);
        b[2] = static_cast<uint8_t>(dm & 0xFF); b[3] = static_cast<uint8_t>(dm >> 8);
        for (int i = 4; i < zc::kQ4KBytes; ++i) b[i] = static_cast<uint8_t>(rng.u32());
    } else {
        for (int i = 0; i < 192; ++i) b[i] = static_cast<uint8_t>(rng.u32());                      // ql, qh
        for (int i = 192; i < 208; ++i) b[i] = static_cast<uint8_t>(static_cast<int>(rng.u32() % 64u) - 32);   // int8 scales in [-32, 31]
        b[208] = static_cast<uint8_t>(d & 0xFF); b[209] = static_cast<uint8_t>(d >> 8);
    }
}

static bool alloc_mapped(size_t bytes, void** host, void** dev) {
    void* p = nullptr;
    if (cudaHostAlloc(&p, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) return false;
    void* d = nullptr;
    if (cudaHostGetDevicePointer(&d, p, 0) != cudaSuccess) { cudaFreeHost(p); return false; }
    *host = p;
    *dev = d;
    return true;
}

static bool make_mat(LayerMats* L, Mat* m, int type, int rows, int cols, int n_experts, Rng& rng) {
    m->type = type;
    m->rows = rows;
    m->cols = cols;
    m->n_experts = n_experts;
    m->row_bytes = static_cast<size_t>(cols / zc::kSuper) * static_cast<size_t>(zc::block_bytes(type));
    m->expert_stride = static_cast<size_t>(rows) * m->row_bytes;
    const size_t total = m->expert_stride * static_cast<size_t>(n_experts);
    void *h = nullptr, *d = nullptr;
    if (!alloc_mapped(total, &h, &d)) return false;
    L->to_free.push_back(h);
    uint8_t* p = static_cast<uint8_t*>(h);
    const int bb = zc::block_bytes(type);
    for (size_t off = 0; off < total; off += static_cast<size_t>(bb)) fill_block(type, p + off, rng);
    m->host = p;
    m->dev = static_cast<const uint8_t*>(d);
    return true;
}

static bool make_synth_layer(LayerMats* L, const char* name, int n_experts, int tg, int tu, int td, uint64_t seed) {
    Rng rng(seed);
    L->name = name;
    return make_mat(L, &L->gate, tg, EXPERT_FF, HIDDEN, n_experts, rng) && make_mat(L, &L->up, tu, EXPERT_FF, HIDDEN, n_experts, rng) &&
           make_mat(L, &L->down, td, HIDDEN, EXPERT_FF, n_experts, rng);
}

static void free_layer(LayerMats* L) {
    for (void* p : L->to_free) cudaFreeHost(p);
    L->to_free.clear();
}

// =====================================================================================================================================================
// 1. the lane arithmetic on the CPU
// =====================================================================================================================================================
static void test_lane_math() {
    for (int type : {zc::kTypeQ4K, zc::kTypeQ6K}) {
        const char* nm = type == zc::kTypeQ4K ? "Q4_K" : "Q6_K";
        Rng rng(1000 + static_cast<uint64_t>(type));
        const int nblk = 40;
        long long onehot_bad = 0, onehot_n = 0;
        int dot_bad = 0;
        double worst_ratio = 0.0;
        std::vector<uint8_t> blk(static_cast<size_t>(zc::block_bytes(type)));
        std::vector<float> y(zc::kSuper), x(zc::kSuper);
        for (int b = 0; b < nblk; ++b) {
            fill_block(type, blk.data(), rng);
            ref_dequant_block(type, blk.data(), y.data());
            for (int p = 0; p < zc::kSuper; ++p) {
                std::fill(x.begin(), x.end(), 0.0f);
                x[static_cast<size_t>(p)] = 1.0f;
                const float got = zc::row_dot_emulated(blk.data(), type, 0, 1, x.data());
                ++onehot_n;
                if (got != y[static_cast<size_t>(p)]) ++onehot_bad;
            }
            for (int i = 0; i < zc::kSuper; ++i) x[static_cast<size_t>(i)] = rng.sym();
            double exact = 0.0, sabs = 0.0;
            for (int i = 0; i < zc::kSuper; ++i) {
                const double t = static_cast<double>(y[static_cast<size_t>(i)]) * static_cast<double>(x[static_cast<size_t>(i)]);
                exact += t;
                sabs += std::fabs(t);
            }
            const double got = static_cast<double>(zc::row_dot_emulated(blk.data(), type, 0, 1, x.data()));
            const double bound = gam(zc::kSuper) * sabs;
            if (std::fabs(got - exact) > bound) ++dot_bad;
            if (bound > 0) worst_ratio = std::max(worst_ratio, std::fabs(got - exact) / bound);
        }
        char buf[200];
        std::snprintf(buf, sizeof buf, "lane maths %s: %lld one-hot positions exact (%lld wrong); %d random dots within gamma_256 * sum|a x| (%d outside, worst error/bound %.3f)", nm, onehot_n, onehot_bad, nblk,
                      dot_bad, worst_ratio);
        check(buf, onehot_bad == 0 && dot_bad == 0);
    }
}

#if defined(PELIAS_GGML_CHECK) && defined(PELIAS_COLDPATH_WITH_VIEW)
// optional: the reference dequantisation against ggml's own to_float on rows of the real model
static void test_against_ggml(const Mat& m, const char* what) {
    const enum ggml_type gt = m.type == zc::kTypeQ4K ? GGML_TYPE_Q4_K : GGML_TYPE_Q6_K;
    const struct ggml_type_traits* tr = ggml_get_type_traits(gt);
    if (tr == nullptr || tr->to_float == nullptr) { check(std::string("ggml cross-check ") + what + ": ggml has no to_float for this type", false); return; }
    long long bad = 0, n = 0;
    std::vector<float> a(static_cast<size_t>(m.cols)), b(static_cast<size_t>(m.cols));
    for (int e : {0, m.n_experts / 2, m.n_experts - 1})
        for (int r = 0; r < m.rows; r += (m.rows / 64 > 0 ? m.rows / 64 : 1)) {
            tr->to_float(m.h_row(e, r), a.data(), m.cols);
            for (int sb = 0; sb < m.cols / zc::kSuper; ++sb) ref_dequant_block(m.type, m.h_row(e, r) + static_cast<size_t>(sb) * zc::block_bytes(m.type), b.data() + sb * zc::kSuper);
            for (int i = 0; i < m.cols; ++i) {
                ++n;
                if (std::memcmp(&a[static_cast<size_t>(i)], &b[static_cast<size_t>(i)], 4) != 0) ++bad;
            }
        }
    char buf[200];
    std::snprintf(buf, sizeof buf, "ggml cross-check %s: %lld dequantised weights bit-identical to ggml's to_float (%lld differ)", what, n, bad);
    check(buf, bad == 0);
}
#endif

// =====================================================================================================================================================
// the CPU worker (stand-in for cold_pool.h)
// =====================================================================================================================================================
struct WorkerRec {
    bool                    served = false;
    std::string             error;
    std::vector<CpuTask>    tasks;
    std::vector<std::vector<float>> g, u, h;          // per task, EXPERT_FF floats, [row_begin, row_end) filled
    std::vector<float>      x_seen;
    double                  compute_ms = 0.0;
};

static void cpu_worker(HostDoorbell db, uint32_t seq, const LayerMats* L, WorkerRec* rec) {
    if (!db.wait_gpu(seq, 5e6)) { rec->error = "the GPU never rang the bell"; return; }
    Request rq;
    if (!db.read_request(seq, &rq)) { rec->error = "request torn or overtaken"; return; }
    const double t0 = now_us();
    rec->x_seen.assign(rq.x, rq.x + kHidden);
    std::vector<float> partial(kHidden, 0.0f);
    rec->tasks.assign(rq.tasks, rq.tasks + rq.n_tasks);
    rec->g.assign(rq.n_tasks, std::vector<float>(EXPERT_FF, 0.0f));
    rec->u = rec->g;
    rec->h = rec->g;
    for (uint32_t t = 0; t < rq.n_tasks; ++t) {
        const CpuTask& ct = rq.tasks[t];
        const int e = ct.expert, rb = ct.row_begin, re = ct.row_end;
        for (int i = rb; i < re; ++i) {
            const float gv = cpu_dot_float(L->gate.h_row(e, i), L->gate.type, 0, zc::kHiddenSb, rq.x);
            const float uv = cpu_dot_float(L->up.h_row(e, i), L->up.type, 0, zc::kHiddenSb, rq.x);
            rec->g[t][static_cast<size_t>(i)] = gv;
            rec->u[t][static_cast<size_t>(i)] = uv;
            rec->h[t][static_cast<size_t>(i)] = gv / (1.0f + std::exp(-gv)) * uv;
        }
        for (int r = 0; r < HIDDEN; ++r) {
            const float y = cpu_dot_float(L->down.h_row(e, r), L->down.type, rb / zc::kSuper, re / zc::kSuper, rec->h[t].data());
            partial[static_cast<size_t>(r)] += ct.weight * y;
        }
    }
    rec->compute_ms = (now_us() - t0) / 1000.0;
    db.post(seq, partial.data(), kStatusOk);
    rec->served = true;
}

// =====================================================================================================================================================
// the GPU side: real kernels (GPU build) or lane-by-lane emulation (stand-in build)
// =====================================================================================================================================================
__global__ void k_publish(Mailbox* mb, uint32_t seq, int n_tasks, const CpuTask* tasks, const float* x) { publish(mb, seq, 0u, 0u, n_tasks, tasks, x); }
__global__ void k_wait(Mailbox* mb, uint32_t seq, float* acc, long long spin_cycles, uint32_t* d_status) { (void)wait_and_combine(mb, seq, acc, 1.0f, spin_cycles, d_status); }
__global__ void k_clock_cal(long long* out, long long cycles) {
    if (threadIdx.x == 0) {
        const long long t0 = clock64();
        while (clock64() - t0 < cycles) {
        }
        *out = clock64() - t0;
    }
}

struct Ctx {
    cudaStream_t  st = nullptr;
    float *d_x = nullptr, *d_h = nullptr, *d_g = nullptr, *d_u = nullptr, *d_y = nullptr, *d_acc = nullptr;
    CpuTask*      d_tasks = nullptr;
    uint32_t*     d_status = nullptr;
    MailboxHandle mb;
    uint32_t      seq = 0;
    double        ticks_per_us = 0.0;
    long long     spin_cycles = 0;
    bool          gpu = false;
};

struct GpuOut {
    std::vector<float> h, g, u, ypart, acc, cpu_partial;
    uint32_t status = 0;
    float    gpu_ms = 0.0f;
};

[[maybe_unused]] static void emu_stages(const zc::ZcLayerJob& job, const float* x, GpuOut* o) {
    o->h.assign(MAX_COLD * EXPERT_FF, 0.0f);
    o->g = o->h;
    o->u = o->h;
    o->ypart.assign(MAX_COLD * HIDDEN, 0.0f);
    for (int e = 0; e < job.n; ++e) {
        const zc::ZcExpertJob& J = job.e[e];
        for (int i = 0; i < J.rows; ++i) {
            const float g = zc::row_dot_emulated(J.gate + static_cast<size_t>(i) * static_cast<size_t>(J.gate_row_bytes), J.gate_type, 0, zc::kHiddenSb, x);
            const float u = zc::row_dot_emulated(J.up + static_cast<size_t>(i) * static_cast<size_t>(J.up_row_bytes), J.up_type, 0, zc::kHiddenSb, x);
            o->g[static_cast<size_t>(e * EXPERT_FF + i)] = g;
            o->u[static_cast<size_t>(e * EXPERT_FF + i)] = u;
            o->h[static_cast<size_t>(e * EXPERT_FF + i)] = g / (1.0f + std::exp(-g)) * u;
        }
        if (J.rows > 0)
            for (int r = 0; r < HIDDEN; ++r)
                o->ypart[static_cast<size_t>(e * HIDDEN + r)] =
                    zc::row_dot_emulated(J.down + static_cast<size_t>(r) * static_cast<size_t>(J.down_row_bytes), J.down_type, 0, J.rows / zc::kSuper, o->h.data() + e * EXPERT_FF);
    }
    o->acc.assign(HIDDEN, 0.0f);
    for (int r = 0; r < HIDDEN; ++r) {
        float s = 0.0f;
        for (int e = 0; e < job.n; ++e)
            if (job.e[e].rows > 0) s += job.e[e].weight * o->ypart[static_cast<size_t>(e * HIDDEN + r)];
        o->acc[static_cast<size_t>(r)] = s;
    }
}

[[maybe_unused]] static void emu_publish(Mailbox* mb, uint32_t seq, int n_tasks, const CpuTask* tasks, const float* x) {
    std::memcpy(mb->x, x, sizeof(float) * kHidden);
    if (n_tasks > 0) std::memcpy(mb->tasks, tasks, sizeof(CpuTask) * static_cast<size_t>(n_tasks));
    mb->layer = 0;
    mb->n_tasks = static_cast<uint32_t>(n_tasks);
    mb->token_pos = 0;
    std::atomic_thread_fence(std::memory_order_release);
    mb->gpu_seq = seq;
}

[[maybe_unused]] static uint32_t emu_wait_and_combine(Mailbox* mb, uint32_t seq, float* acc) {
    const double t0 = now_us();
    while (static_cast<int32_t>(mb->cpu_seq - seq) < 0)
        if (now_us() - t0 > 5e6) return kStatusGpuTimeout;
    std::atomic_thread_fence(std::memory_order_acquire);
    if (mb->cpu_status != kStatusOk) return kStatusCpuError;
    for (int r = 0; r < HIDDEN; ++r) acc[r] += mb->partial[r];
    return kStatusOk;
}

static bool run_gpu_side(Ctx& C, const zc::ZcLayerJob& job, int n_tasks, const CpuTask* tasks, uint32_t seq, const float* x, GpuOut* o, std::string* err) {
#if PELIAS_COLDPATH_GPU
    (void)err;
    PL_CUDA_CHECK(cudaMemcpy(C.d_x, x, sizeof(float) * HIDDEN, cudaMemcpyHostToDevice));
    PL_CUDA_CHECK(cudaMemset(C.d_acc, 0, sizeof(float) * HIDDEN));
    PL_CUDA_CHECK(cudaMemset(C.d_status, 0, sizeof(uint32_t)));
    PL_CUDA_CHECK(cudaMemset(C.d_y, 0, sizeof(float) * MAX_COLD * HIDDEN));
    PL_CUDA_CHECK(cudaMemset(C.d_h, 0, sizeof(float) * MAX_COLD * EXPERT_FF));
    PL_CUDA_CHECK(cudaMemset(C.d_g, 0, sizeof(float) * MAX_COLD * EXPERT_FF));
    PL_CUDA_CHECK(cudaMemset(C.d_u, 0, sizeof(float) * MAX_COLD * EXPERT_FF));
    if (n_tasks > 0) PL_CUDA_CHECK(cudaMemcpy(C.d_tasks, tasks, sizeof(CpuTask) * static_cast<size_t>(n_tasks), cudaMemcpyHostToDevice));
    // The copies and memsets above run on the legacy default stream and may return before they finish; C.st is a non-blocking stream and does NOT wait for that
    // stream, so without this a late memset could zero results the kernels already wrote (CUDA C++ Programming Guide, API synchronization behavior, and non-blocking streams).
    PL_CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    PL_CUDA_CHECK(cudaEventCreate(&e0));
    PL_CUDA_CHECK(cudaEventCreate(&e1));
    PL_CUDA_CHECK(cudaEventRecord(e0, C.st));
    if (n_tasks > 0) PL_LAUNCH(k_publish, 1, 256, 0, C.st, C.mb.dev, seq, n_tasks, C.d_tasks, C.d_x);
    PL_LAUNCH(zc::zc_gateup_swiglu, dim3((EXPERT_FF + zc::kWarps - 1) / zc::kWarps, job.n), zc::kBlockThreads, 0, C.st, job, C.d_x, C.d_h, C.d_g, C.d_u);
    PL_LAUNCH(zc::zc_down_partial, dim3(HIDDEN / zc::kWarps, job.n), zc::kBlockThreads, 0, C.st, job, C.d_h, C.d_y);
    PL_LAUNCH(zc::zc_add_partials, HIDDEN / zc::kBlockThreads, zc::kBlockThreads, 0, C.st, job, C.d_y, C.d_acc);
    if (n_tasks > 0) PL_LAUNCH(k_wait, 1, 256, 0, C.st, C.mb.dev, seq, C.d_acc, C.spin_cycles, C.d_status);
    PL_CUDA_CHECK(cudaEventRecord(e1, C.st));
    flush_stream(C.st);                      // the 03 rev 2 fix: a freshly launched kernel can stay queued until the driver is called again
    PL_CUDA_CHECK(cudaStreamSynchronize(C.st));
    PL_CUDA_CHECK(cudaGetLastError());
    PL_CUDA_CHECK(cudaEventElapsedTime(&o->gpu_ms, e0, e1));
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    o->h.resize(MAX_COLD * EXPERT_FF);
    o->g.resize(MAX_COLD * EXPERT_FF);
    o->u.resize(MAX_COLD * EXPERT_FF);
    o->ypart.resize(MAX_COLD * HIDDEN);
    o->acc.resize(HIDDEN);
    PL_CUDA_CHECK(cudaMemcpy(o->h.data(), C.d_h, sizeof(float) * o->h.size(), cudaMemcpyDeviceToHost));
    PL_CUDA_CHECK(cudaMemcpy(o->g.data(), C.d_g, sizeof(float) * o->g.size(), cudaMemcpyDeviceToHost));
    PL_CUDA_CHECK(cudaMemcpy(o->u.data(), C.d_u, sizeof(float) * o->u.size(), cudaMemcpyDeviceToHost));
    PL_CUDA_CHECK(cudaMemcpy(o->ypart.data(), C.d_y, sizeof(float) * o->ypart.size(), cudaMemcpyDeviceToHost));
    PL_CUDA_CHECK(cudaMemcpy(o->acc.data(), C.d_acc, sizeof(float) * o->acc.size(), cudaMemcpyDeviceToHost));
    PL_CUDA_CHECK(cudaMemcpy(&o->status, C.d_status, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    return true;
#else
    (void)C;
    if (n_tasks > 0) emu_publish(C.mb.host, seq, n_tasks, tasks, x);
    emu_stages(job, x, o);
    o->status = n_tasks > 0 ? emu_wait_and_combine(C.mb.host, seq, o->acc.data()) : kStatusOk;
    if (err != nullptr && o->status != kStatusOk) *err = "emulated wait_and_combine failed";
    return true;
#endif
}

// =====================================================================================================================================================
// double-precision split-free reference of one expert for one x, with the derived error bounds
// =====================================================================================================================================================
struct ExpertRef {
    std::vector<double> g, u, s, h, dg, du, dh;      // EXPERT_FF each
    std::vector<double> y, sdh, sah;                  // HIDDEN each: sum_k a h ;  sum_k |a| dh ;  sum_k |a| (|h| + dh)
};

static void make_ref(const LayerMats& L, int e, const float* x, ExpertRef* R) {
    const size_t nff = EXPERT_FF;
    R->g.assign(nff, 0); R->u = R->g; R->s = R->g; R->h = R->g; R->dg = R->g; R->du = R->g; R->dh = R->g;
    R->y.assign(HIDDEN, 0); R->sdh = R->y; R->sah = R->y;
    std::vector<float> row(HIDDEN);
    for (int i = 0; i < EXPERT_FF; ++i) {
        for (int which = 0; which < 2; ++which) {
            const Mat& M = which == 0 ? L.gate : L.up;
            for (int sb = 0; sb < zc::kHiddenSb; ++sb) ref_dequant_block(M.type, M.h_row(e, i) + static_cast<size_t>(sb) * zc::block_bytes(M.type), row.data() + sb * zc::kSuper);
            double sum = 0, sabs = 0;
            for (int k = 0; k < HIDDEN; ++k) {
                const double t = static_cast<double>(row[static_cast<size_t>(k)]) * static_cast<double>(x[k]);
                sum += t;
                sabs += std::fabs(t);
            }
            (which == 0 ? R->g : R->u)[static_cast<size_t>(i)] = sum;
            (which == 0 ? R->dg : R->du)[static_cast<size_t>(i)] = gam(HIDDEN) * sabs;
        }
        const size_t k = static_cast<size_t>(i);
        R->s[k] = R->g[k] / (1.0 + std::exp(-R->g[k]));
        R->h[k] = R->s[k] * R->u[k];
        const double D = kSiluSlope * (std::fabs(R->u[k]) + R->du[k]) * R->dg[k] + std::fabs(R->s[k]) * R->du[k];
        R->dh[k] = (1.0 + kCh) * D + kCh * std::fabs(R->h[k]);
    }
    std::vector<float> a(EXPERT_FF);
    for (int r = 0; r < HIDDEN; ++r) {
        for (int sb = 0; sb < EXPERT_FF / zc::kSuper; ++sb) ref_dequant_block(L.down.type, L.down.h_row(e, r) + static_cast<size_t>(sb) * zc::block_bytes(L.down.type), a.data() + sb * zc::kSuper);
        double y = 0, sd = 0, sa = 0;
        for (int k = 0; k < EXPERT_FF; ++k) {
            const double av = a[static_cast<size_t>(k)];
            y += av * R->h[static_cast<size_t>(k)];
            sd += std::fabs(av) * R->dh[static_cast<size_t>(k)];
            sa += std::fabs(av) * (std::fabs(R->h[static_cast<size_t>(k)]) + R->dh[static_cast<size_t>(k)]);
        }
        R->y[static_cast<size_t>(r)] = y;
        R->sdh[static_cast<size_t>(r)] = sd;
        R->sah[static_cast<size_t>(r)] = sa;
    }
}

// =====================================================================================================================================================
// one scenario
// =====================================================================================================================================================
struct Scenario {
    const char* name;
    int         n_miss;
    float       gpu_frac;
};

static bool run_scenario(Ctx& C, const LayerMats& L, const Scenario& sc, uint64_t seed) {
    Rng rng(seed);
    std::vector<float> x(HIDDEN);
    for (float& v : x) v = rng.sym();
    MissExpert miss[MAX_COLD];
    for (int i = 0; i < sc.n_miss; ++i) {
        miss[i].expert = (7 * i + 3 + static_cast<int>(seed % 5)) % L.gate.n_experts;
        miss[i].weight = 0.05f + 0.4f * rng.uni();
    }
    SplitterState st;
    SplitPlan plan;
    split_misses(make_splitter_config(sc.gpu_frac), &st, miss, sc.n_miss, &plan);
    CpuTask tasks[kMaxTasks];
    const int n_tasks = fill_cpu_tasks(plan, tasks);

    zc::ZcLayerJob job;
    std::memset(&job, 0, sizeof job);
    job.n = plan.n_miss;
    for (int i = 0; i < plan.n_miss; ++i) {
        zc::ZcExpertJob& J = job.e[i];
        const int e = plan.task[i].expert;
        J.gate = L.gate.d_expert(e); J.up = L.up.d_expert(e); J.down = L.down.d_expert(e);
        J.gate_type = L.gate.type; J.up_type = L.up.type; J.down_type = L.down.type;
        J.gate_row_bytes = static_cast<int32_t>(L.gate.row_bytes); J.up_row_bytes = static_cast<int32_t>(L.up.row_bytes); J.down_row_bytes = static_cast<int32_t>(L.down.row_bytes);
        J.rows = plan.task[i].split_row;
        J.weight = plan.task[i].weight;
    }

    // the CPU worker starts first and spins on the doorbell, as the engine's worker would
    WorkerRec rec;
    uint32_t seq = 0;
    std::thread worker;
    if (n_tasks > 0) {
        seq = ++C.seq;
        worker = std::thread(cpu_worker, HostDoorbell(C.mb.host), seq, &L, &rec);
    }
    GpuOut out;
    std::string gerr;
    run_gpu_side(C, job, n_tasks, tasks, seq, x.data(), &out, &gerr);
    if (worker.joinable()) worker.join();
    out.cpu_partial.assign(C.mb.host->partial, C.mb.host->partial + kHidden);

    // ---- references and checks
    std::map<int, ExpertRef> refs;
    for (int i = 0; i < plan.n_miss; ++i)
        if (!refs.count(plan.task[i].expert)) make_ref(L, plan.task[i].expert, x.data(), &refs[plan.task[i].expert]);

    long long bad_a = 0, bad_b = 0, bad_c = 0, bad_a_cpu = 0, bad_b_cpu = 0, bad_d = 0, bad_e = 0, bad_f = 0;
    double worst_f = 0.0, max_y = 0.0, max_err = 0.0;
    std::vector<double> cpu_ref(HIDDEN, 0.0), cpu_mag(HIDDEN, 0.0), gsum(HIDDEN, 0.0), gmag(HIDDEN, 0.0), yref(HIDDEN, 0.0), bsd(HIDDEN, 0.0), bsa(HIDDEN, 0.0);
    std::vector<float> a(EXPERT_FF);
    std::vector<std::string> notes;                    // the first few offenders, printed only when a check fails
    const auto note = [&notes](const char* what, int slot, int row, double got, double want, double bound) {
        if (notes.size() >= 8) return;
        char nb[300];
        std::snprintf(nb, sizeof nb, "       offender: %s, expert slot %d, row %d: GPU %.9g, reference %.9g, difference %.3g, allowed %.3g", what, slot, row, got, want, std::fabs(got - want), bound);
        notes.push_back(nb);
    };
    for (int t = 0; t < plan.n_miss; ++t) {
        const int e = plan.task[t].expert, rows = plan.task[t].split_row;
        const double w = static_cast<double>(plan.task[t].weight);
        const ExpertRef& R = refs[e];
        // which worker task is this expert (CPU rows [rows, EXPERT_FF))?
        int ct = -1;
        for (size_t q = 0; q < rec.tasks.size(); ++q)
            if (rec.tasks[q].expert == e && rec.tasks[q].row_begin == rows) ct = static_cast<int>(q);
        for (int k = 0; k < rows; ++k) {                                       // A and B on the GPU's rows
            const size_t ix = static_cast<size_t>(t * EXPERT_FF + k);
            if (std::fabs(static_cast<double>(out.g[ix]) - R.g[static_cast<size_t>(k)]) > R.dg[static_cast<size_t>(k)]) { ++bad_a; note("gate", t, k, out.g[ix], R.g[static_cast<size_t>(k)], R.dg[static_cast<size_t>(k)]); }
            if (std::fabs(static_cast<double>(out.u[ix]) - R.u[static_cast<size_t>(k)]) > R.du[static_cast<size_t>(k)]) { ++bad_a; note("up", t, k, out.u[ix], R.u[static_cast<size_t>(k)], R.du[static_cast<size_t>(k)]); }
            const double gg = out.g[ix], uu = out.u[ix];
            const double hh = gg / (1.0 + std::exp(-gg)) * uu;
            if (std::fabs(static_cast<double>(out.h[ix]) - hh) > kCh * std::fabs(hh)) { ++bad_b; note("SwiGLU h (reference built from the GPU's own g and u)", t, k, out.h[ix], hh, kCh * std::fabs(hh)); }
        }
        if (rows < EXPERT_FF && ct >= 0) {                                      // A and B on the CPU's rows
            for (int k = rows; k < EXPERT_FF; ++k) {
                const size_t kk = static_cast<size_t>(k);
                if (std::fabs(static_cast<double>(rec.g[static_cast<size_t>(ct)][kk]) - R.g[kk]) > R.dg[kk]) ++bad_a_cpu;
                if (std::fabs(static_cast<double>(rec.u[static_cast<size_t>(ct)][kk]) - R.u[kk]) > R.du[kk]) ++bad_a_cpu;
                const double gg = rec.g[static_cast<size_t>(ct)][kk], uu = rec.u[static_cast<size_t>(ct)][kk];
                const double hh = gg / (1.0 + std::exp(-gg)) * uu;
                if (std::fabs(static_cast<double>(rec.h[static_cast<size_t>(ct)][kk]) - hh) > kCh * std::fabs(hh)) ++bad_b_cpu;
            }
        }
        for (int r = 0; r < HIDDEN; ++r) {
            for (int sb = 0; sb < EXPERT_FF / zc::kSuper; ++sb) ref_dequant_block(L.down.type, L.down.h_row(e, r) + static_cast<size_t>(sb) * zc::block_bytes(L.down.type), a.data() + sb * zc::kSuper);
            double yg = 0, mg = 0;
            for (int k = 0; k < rows; ++k) {
                const double tm = static_cast<double>(a[static_cast<size_t>(k)]) * static_cast<double>(out.h[static_cast<size_t>(t * EXPERT_FF + k)]);
                yg += tm;
                mg += std::fabs(tm);
            }
            if (rows > 0) {
                const double yp = out.ypart[static_cast<size_t>(t * HIDDEN + r)];
                if (std::fabs(yp - yg) > gam(rows) * mg) { ++bad_c; note("down partial", t, r, yp, yg, gam(rows) * mg); }
                gsum[static_cast<size_t>(r)] += w * yp;
                gmag[static_cast<size_t>(r)] += std::fabs(w * yp);
            }
            if (rows < EXPERT_FF && ct >= 0) {
                double yc = 0, mc = 0;
                for (int k = rows; k < EXPERT_FF; ++k) {
                    const double tm = static_cast<double>(a[static_cast<size_t>(k)]) * static_cast<double>(rec.h[static_cast<size_t>(ct)][static_cast<size_t>(k)]);
                    yc += tm;
                    mc += std::fabs(tm);
                }
                cpu_ref[static_cast<size_t>(r)] += w * yc;
                cpu_mag[static_cast<size_t>(r)] += std::fabs(w) * mc;
            }
            yref[static_cast<size_t>(r)] += w * R.y[static_cast<size_t>(r)];
            bsd[static_cast<size_t>(r)] += std::fabs(w) * R.sdh[static_cast<size_t>(r)];
            bsa[static_cast<size_t>(r)] += std::fabs(w) * R.sah[static_cast<size_t>(r)];
        }
    }
    const double g_cpu = gam(EXPERT_FF + MAX_COLD + 2), g_comb = gam(2.0 * plan.n_miss + 2.0);
    for (int r = 0; r < HIDDEN; ++r) {
        const size_t rr = static_cast<size_t>(r);
        const double pm = n_tasks > 0 ? static_cast<double>(out.cpu_partial[rr]) : 0.0;
        if (n_tasks > 0 && std::fabs(pm - cpu_ref[rr]) > g_cpu * cpu_mag[rr]) ++bad_d;
        const double comp = gsum[rr] + pm;
        if (std::fabs(static_cast<double>(out.acc[rr]) - comp) > g_comb * (gmag[rr] + std::fabs(pm))) ++bad_e;
        const double bound = bsd[rr] + (g_cpu + g_comb) * bsa[rr];
        const double err = std::fabs(static_cast<double>(out.acc[rr]) - yref[rr]);
        if (err > bound) ++bad_f;
        if (bound > 0) worst_f = std::max(worst_f, err / bound);
        max_err = std::max(max_err, err);
        max_y = std::max(max_y, std::fabs(yref[rr]));
    }

    const bool x_ok = n_tasks == 0 || (rec.served && rec.x_seen.size() == static_cast<size_t>(HIDDEN) && std::memcmp(rec.x_seen.data(), x.data(), sizeof(float) * HIDDEN) == 0);
    const bool tasks_ok = rec.tasks.size() == static_cast<size_t>(n_tasks);
    char buf[768];
    std::snprintf(buf, sizeof buf,
                  "%s [%s]: %d miss experts, GPU rows %d / CPU rows %d of %d; x reached the CPU intact, CPU tasks as planned; GPU stage values within bounds (gate/up %lld, SwiGLU %lld, down %lld outside); "
                  "CPU values (%lld, %lld, partial %lld); combine %lld; end-to-end vs the split-free double reference: %lld of %d outputs outside the derived bound (worst error/bound %.2e, max error %.3g on |y| up to %.3g)",
                  sc.name, L.name.c_str(), plan.n_miss, plan.gpu_rows, plan.cpu_rows, plan.n_miss * EXPERT_FF, bad_a, bad_b, bad_c, bad_a_cpu, bad_b_cpu, bad_d, bad_e, bad_f, HIDDEN, worst_f, max_err, max_y);
    const bool ok = x_ok && tasks_ok && out.status == kStatusOk && bad_a == 0 && bad_b == 0 && bad_c == 0 && bad_a_cpu == 0 && bad_b_cpu == 0 && bad_d == 0 && bad_e == 0 && bad_f == 0 && gerr.empty();
    check(buf, ok, "status " + std::to_string(out.status) + (rec.error.empty() ? "" : ", worker: " + rec.error) + (gerr.empty() ? "" : ", " + gerr) + (x_ok ? "" : ", x mismatch") + (tasks_ok ? "" : ", task list mismatch"));
    if (!ok)
        for (const std::string& s : notes) std::printf("%s\n", s.c_str());
    std::printf("     (GPU side %.3f ms%s; CPU stand-in worker %.1f ms, its speed means nothing)\n", static_cast<double>(out.gpu_ms), C.gpu ? "" : " - not measured in the stand-in build", rec.compute_ms);
    return ok;
}

// =====================================================================================================================================================
// 3. speed of the kernels reading whole experts from pinned memory
// =====================================================================================================================================================
#if PELIAS_COLDPATH_GPU
static void throughput(Ctx& C, const LayerMats& L) {
    const int per = 8;
    const int passes = std::min(16, L.gate.n_experts / per);
    if (passes < 2) { std::printf("     throughput: needs at least %d experts, skipped\n", 2 * per); return; }
    std::vector<float> x(HIDDEN);
    Rng rng(77);
    for (float& v : x) v = rng.sym();
    PL_CUDA_CHECK(cudaMemcpy(C.d_x, x.data(), sizeof(float) * HIDDEN, cudaMemcpyHostToDevice));
    PL_CUDA_CHECK(cudaDeviceSynchronize());       // same reason as in run_gpu_side: C.st does not wait for the default stream
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    PL_CUDA_CHECK(cudaEventCreate(&e0));
    PL_CUDA_CHECK(cudaEventCreate(&e1));
    std::vector<double> gbs;
    for (int p = 0; p < passes; ++p) {
        zc::ZcLayerJob job;
        std::memset(&job, 0, sizeof job);
        job.n = per;
        double bytes = 0;
        for (int i = 0; i < per; ++i) {
            const int e = p * per + i;
            zc::ZcExpertJob& J = job.e[i];
            J.gate = L.gate.d_expert(e); J.up = L.up.d_expert(e); J.down = L.down.d_expert(e);
            J.gate_type = L.gate.type; J.up_type = L.up.type; J.down_type = L.down.type;
            J.gate_row_bytes = static_cast<int32_t>(L.gate.row_bytes); J.up_row_bytes = static_cast<int32_t>(L.up.row_bytes); J.down_row_bytes = static_cast<int32_t>(L.down.row_bytes);
            J.rows = EXPERT_FF;
            J.weight = 0.1f;
            bytes += static_cast<double>(L.gate.expert_stride + L.up.expert_stride + L.down.expert_stride);
        }
        PL_CUDA_CHECK(cudaEventRecord(e0, C.st));
        PL_LAUNCH(zc::zc_gateup_swiglu, dim3((EXPERT_FF + zc::kWarps - 1) / zc::kWarps, job.n), zc::kBlockThreads, 0, C.st, job, C.d_x, C.d_h, C.d_g, C.d_u);
        PL_LAUNCH(zc::zc_down_partial, dim3(HIDDEN / zc::kWarps, job.n), zc::kBlockThreads, 0, C.st, job, C.d_h, C.d_y);
        PL_CUDA_CHECK(cudaEventRecord(e1, C.st));
        PL_CUDA_CHECK(cudaStreamSynchronize(C.st));
        PL_CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        PL_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        gbs.push_back(bytes / 1e9 / (static_cast<double>(ms) / 1e3));
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    std::vector<double> sorted = gbs;
    std::sort(sorted.begin(), sorted.end());
    std::printf("     kernel read speed, %d passes of %d whole experts (%.2f MB each, different experts every pass), GB/s (measured, includes the arithmetic, no limit applied):", passes, per,
                static_cast<double>(L.gate.expert_stride + L.up.expert_stride + L.down.expert_stride) * per / 1e6);
    for (double g : gbs) std::printf(" %.2f", g);
    std::printf("\n     median %.2f GB/s   (for comparison, measured earlier with the checksum kernel on pinned memory: 26.2 GB/s)\n", sorted[sorted.size() / 2]);
}
#endif

// =====================================================================================================================================================
// main
// =====================================================================================================================================================
static bool setup_ctx(Ctx* C) {
    (void)cudaSetDeviceFlags(cudaDeviceMapHost);
    (void)cudaGetLastError();
    C->gpu = PELIAS_COLDPATH_GPU != 0;
    PL_CUDA_CHECK(cudaStreamCreateWithFlags(&C->st, cudaStreamNonBlocking));
    PL_CUDA_CHECK(cudaMalloc(&C->d_x, sizeof(float) * HIDDEN));
    PL_CUDA_CHECK(cudaMalloc(&C->d_h, sizeof(float) * MAX_COLD * EXPERT_FF));
    PL_CUDA_CHECK(cudaMalloc(&C->d_g, sizeof(float) * MAX_COLD * EXPERT_FF));
    PL_CUDA_CHECK(cudaMalloc(&C->d_u, sizeof(float) * MAX_COLD * EXPERT_FF));
    PL_CUDA_CHECK(cudaMalloc(&C->d_y, sizeof(float) * MAX_COLD * HIDDEN));
    PL_CUDA_CHECK(cudaMalloc(&C->d_acc, sizeof(float) * HIDDEN));
    PL_CUDA_CHECK(cudaMalloc(&C->d_tasks, sizeof(CpuTask) * kMaxTasks));
    PL_CUDA_CHECK(cudaMalloc(&C->d_status, sizeof(uint32_t)));
    PL_CUDA_CHECK(alloc_mailbox(&C->mb));
    HostDoorbell(C->mb.host).reset(0);
#if PELIAS_COLDPATH_GPU
    // clock64 ticks per second, measured against the wall clock; the CPU stand-in worker is slow, so the GPU's wait is allowed 0.5 s (far below the display driver's 2 s)
    long long* d = nullptr;
    PL_CUDA_CHECK(cudaMalloc(&d, sizeof(long long)));
    PL_LAUNCH(k_clock_cal, 1, 32, 0, C->st, d, 1000000LL);
    PL_CUDA_CHECK(cudaStreamSynchronize(C->st));
    const long long cycles = 150000000LL;
    const double t0 = now_us();
    PL_LAUNCH(k_clock_cal, 1, 32, 0, C->st, d, cycles);
    PL_CUDA_CHECK(cudaStreamSynchronize(C->st));
    const double us = now_us() - t0;
    long long ticks = 0;
    PL_CUDA_CHECK(cudaMemcpy(&ticks, d, sizeof ticks, cudaMemcpyDeviceToHost));
    cudaFree(d);
    C->ticks_per_us = static_cast<double>(ticks) / us;
    C->spin_cycles = static_cast<long long>(C->ticks_per_us * 500000.0);
    std::printf("clock64: %.1f ticks per microsecond; the GPU's wait for the CPU is bounded at %lld ticks (0.5 s)\n", C->ticks_per_us, C->spin_cycles);
#endif
    return true;
}

int main(int argc, char** argv) {
    std::string gguf;
    [[maybe_unused]] int layer = 0;
    bool do_throughput = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--gguf" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--layer" && i + 1 < argc) layer = std::atoi(argv[++i]);
        else if (a == "--no-throughput") do_throughput = false;
    }
#if PELIAS_COLDPATH_GPU
    cudaDeviceProp prop;
    PL_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s   (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
#else
    std::printf("CPU stand-in build: the GPU stages are emulated lane by lane; no device code runs\n");
    (void)do_throughput;
#endif
    Ctx C;
    if (!setup_ctx(&C)) return 1;

    test_lane_math();

    const Scenario scenarios[] = {
        {"all rows on the CPU", 3, 0.0f},
        {"half the rows to the GPU (one expert cut in two)", 3, 0.5f},
        {"3/8 of the rows to the GPU, 4 experts", 4, 0.375f},
        {"1/4 to the GPU, 2 experts (the cut is inside expert 0)", 2, 0.25f},
        {"all rows on the GPU (no handoff)", 3, 1.0f},
        {"one expert cut 256 / 256", 1, 0.5f},
    };

    LayerMats synth_a, synth_b;
    check("synthetic layer A (gate/up Q4_K, down Q6_K, 64 experts) in mapped pinned memory", make_synth_layer(&synth_a, "synthetic A: Q4_K Q4_K Q6_K", 64, zc::kTypeQ4K, zc::kTypeQ4K, zc::kTypeQ6K, 11));
    check("synthetic layer B (gate Q6_K, up Q4_K, down Q4_K, 16 experts) in mapped pinned memory", make_synth_layer(&synth_b, "synthetic B: Q6_K Q4_K Q4_K", 16, zc::kTypeQ6K, zc::kTypeQ4K, zc::kTypeQ4K, 12));
    uint64_t seed = 100;
    if (synth_a.gate.host != nullptr && synth_b.gate.host != nullptr) {
        for (const Scenario& s : scenarios) run_scenario(C, synth_a, s, ++seed);
        for (const Scenario& s : {scenarios[1], scenarios[3]}) run_scenario(C, synth_b, s, ++seed);
#if PELIAS_COLDPATH_GPU
        if (do_throughput) throughput(C, synth_a);
#endif
    }

#if defined(PELIAS_COLDPATH_WITH_VIEW)
    if (!gguf.empty()) {
        std::printf("\nreal model: %s, layer %d\n", gguf.c_str(), layer);
        ColdWeightsView v;
        ColdViewOptions o;
        o.gguf_path = gguf;
        o.first_layer = layer;
        o.last_layer = layer;
        std::string err;
        const bool opened = v.open(o, &err);
        check("real model: open", opened, err);
        ColdPinReport pr;
        const bool pinned = opened && v.pin(0, &pr);
        check("real model: layer pinned, every matrix has a device pointer", pinned && pr.regions_failed == 0 && v.layer(layer).gate.gpu_ok && v.layer(layer).up.gpu_ok && v.layer(layer).down.gpu_ok, pr.first_error);
        if (pinned && v.layer(layer).gate.gpu_ok) {
            LayerMats real;
            real.name = "real layer " + std::to_string(layer);
            const ColdLayerView& lv = v.layer(layer);
            const ColdMat* ms[3] = {&lv.gate, &lv.up, &lv.down};
            Mat* dst[3] = {&real.gate, &real.up, &real.down};
            for (int i = 0; i < 3; ++i) {
                dst[i]->host = ms[i]->base;
                dst[i]->dev = ms[i]->dev_base;
                dst[i]->type = ms[i]->type_id;
                dst[i]->rows = static_cast<int>(ms[i]->rows);
                dst[i]->cols = static_cast<int>(ms[i]->cols);
                dst[i]->n_experts = static_cast<int>(ms[i]->n_experts);
                dst[i]->row_bytes = ms[i]->row_bytes;
                dst[i]->expert_stride = ms[i]->expert_stride;
            }
#if defined(PELIAS_GGML_CHECK)
            test_against_ggml(real.gate, "gate");
            test_against_ggml(real.up, "up");
            test_against_ggml(real.down, "down");
#endif
            const bool types_ok = (real.gate.type == zc::kTypeQ4K || real.gate.type == zc::kTypeQ6K) && (real.up.type == zc::kTypeQ4K || real.up.type == zc::kTypeQ6K) &&
                                  (real.down.type == zc::kTypeQ4K || real.down.type == zc::kTypeQ6K);
            check("real model: all three matrices are Q4_K or Q6_K", types_ok);
            if (types_ok) {
                for (const Scenario& s : scenarios) run_scenario(C, real, s, ++seed);
#if PELIAS_COLDPATH_GPU
                if (do_throughput) throughput(C, real);
#endif
            }
        }
        v.close();
    }
#else
    if (!gguf.empty()) check("--gguf needs a build with -DPELIAS_COLDPATH_WITH_VIEW (see the header comment)", false);
    (void)layer;
#endif

    free_layer(&synth_a);
    free_layer(&synth_b);
    std::printf("%s\n", g_fail == 0 ? "ALL PASS" : "FAILED");
    return g_fail == 0 ? 0 : 1;
}
