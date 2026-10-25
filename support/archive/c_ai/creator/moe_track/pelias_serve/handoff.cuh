// File: <project>\moe_track\pelias_serve\handoff.cuh
// Pass 1 / piece 2: the GPU <-> CPU doorbell for the cold-expert handoff of one layer.
//
// WHAT IT CARRIES. Per layer, the GPU publishes the layer input x (HIDDEN floats) and the CPU's share of the miss experts (miss_splitter.h: expert id, router
// weight, the intermediate rows [row_begin, row_end) the CPU must compute). The CPU worker pool (cold_pool.h) computes its partial output y_cpu (HIDDEN floats)
// and posts it. The GPU adds it into the layer accumulator: that is the "final combine reduction".
//
// THE PROTOCOL (the one microbench 03 rev 2 measured; see the numbers below):
//   GPU, one thread block:  write payload -> every thread __threadfence_system() -> __syncthreads() -> thread 0 writes gpu_seq (the doorbell)
//   CPU:                    spin on gpu_seq (acquire) -> copy the request -> compute -> write partial and cpu_status -> fence (release) -> write cpu_seq
//   GPU, thread 0:          spin on cpu_seq, bounded by a cycle limit -> __threadfence_system() -> __syncthreads() -> all threads read the partial with
//                           16-byte volatile loads and add it into the accumulator. On timeout or a CPU error status nothing is added and the status is
//                           written to a device word so the engine can recompute that layer another way.
//   Sequence numbers only grow (compared as signed differences, so wrap-around is harmless). A layer with no CPU work simply does not ring the bell.
//
// LAYOUT RULES. The mailbox is one mapped, pinned allocation. The doorbell word of each direction sits alone on a 128-byte line (so the side that spins on
// it is not disturbed by payload traffic), and the two directions are on different lines (the GPU writes lines 0..N, the CPU writes the lines after
// them; nothing is written by both). Every payload array is 16-byte aligned so the GPU moves it with 16-byte vector accesses.
//
// THE FLUSH FIX. Microbench 03 showed that on this WSL2 setup a launched kernel can stay in the driver's queue until the driver is called again: a CPU that
// only spins on a mapped flag waits forever (the first test version timed out). One cudaStreamQuery() right after the launch fixes it and costs 0.11 us;
// polling cudaStreamQuery() in the wait loop was no faster (kick 94.4 us vs loop 93.5 us). flush_stream() is that call; use it after every launch that
// contains a publish().
//
// MEASURED (microbench 03 rev 2, this machine, GPU kernel + CPU spinning, nothing else running):
//   ping-pong, flag only                      3.36 us median   8.67 us p99
//   ping-pong + the GPU reading a 4 KB payload   13.34 us median  18.71 us p99     <- that test read it with 2-byte loads, one dependent round trip each
//   launch + stream sync per handoff          ~94 us           (so a launch per handoff is not an option)
// This file moves 8 KB each way with 16-byte loads/stores issued as one batch, which should cost much less than the 10 us the 2-byte loads added; that is
// a prediction, not a measurement. tests/test_handoff.cu measures it and checks that every byte arrives intact.
// No timing limit is set here: the only measured reference is 03's 13.3 us median / 18.7 us p99 (a different transfer pattern), and the limit that matters comes from the
// speed model (each us per layer costs 0.04 ms per token over 40 layers), whose spine and hit-rate terms are still unmeasured.
//
// USE FROM A RESIDENT KERNEL OR FROM ORDINARY KERNELS. publish() and wait_and_combine() are block-level device functions: call them from one persistent
// block (the form that was measured) or from two ordinary kernels of the layer's stream (publish at the end of the router kernel, wait_and_combine at the
// start of the combine kernel). Either way every spin is bounded (default 40 M cycles; wall time = 40 M / the GPU clock, which tests/test_handoff.cu measures and prints) so a lost CPU cannot hang the GPU, and a
// kernel that spins must stay far below the display driver's 2 s limit: never keep one alive across idle time.
//
// The host half (Mailbox, HostDoorbell, alloc_mailbox) needs only the CUDA runtime API; the device half compiles only under nvcc or lib/cuda_shim.h
// (-DPELIAS_CUDA_SHIM, for the host tests). Not run on a GPU by the author: tests/test_handoff.cu is the first run.
#pragma once

#include "miss_splitter.h"
#include "pelias_defs.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#if defined(PELIAS_CUDA_SHIM)
#include "../lib/cuda_shim.h"
#define PELIAS_HANDOFF_HAVE_CUDA 1
#elif defined(__CUDACC__) || defined(PELIAS_HANDOFF_FORCE_CUDA)
#include <cuda_runtime.h>
#define PELIAS_HANDOFF_HAVE_CUDA 1
#else
#define PELIAS_HANDOFF_HAVE_CUDA 0
#endif

#if defined(__CUDACC__)
#define PELIAS_UNROLL _Pragma("unroll")
#else
#define PELIAS_UNROLL
#endif

namespace pelias {
namespace handoff {

constexpr int      kLine      = 128;                 // separation unit: the GPU's L2 line, two CPU cache lines
constexpr int      kHidden    = HIDDEN;              // 2048 floats = 8 KB each way
constexpr int      kMaxTasks  = MAX_COLD;            // 16
constexpr long long kDefaultSpinCycles = 40000000LL; // wall time = this / the GPU clock (measured by tests/test_handoff.cu); the caller can pass its own

constexpr uint32_t kStatusOk         = 0;
constexpr uint32_t kStatusGpuTimeout = 1;            // written by the GPU into its own status word: the CPU result did not arrive in time
constexpr uint32_t kStatusCpuError   = 2;            // the CPU said it could not compute this layer (cpu_status != 0)

// One miss expert's CPU share, 16 bytes. The CPU computes intermediate rows [row_begin, row_end) of this expert (gate/up rows and the matching K-slice of down).
struct alignas(16) CpuTask {
    int32_t expert;
    float   weight;
    int32_t row_begin;
    int32_t row_end;
};
static_assert(sizeof(CpuTask) == 16, "CpuTask must be one 16-byte vector");

struct alignas(kLine) Mailbox {
    // ---- line 0: the GPU's doorbell, alone on its line
    volatile uint32_t gpu_seq;
    uint32_t          pad0_[kLine / 4 - 1];
    // ---- GPU -> CPU payload (all written before the doorbell)
    uint32_t          layer;
    uint32_t          n_tasks;
    uint32_t          token_pos;
    uint32_t          reserved;
    CpuTask           tasks[kMaxTasks];
    alignas(16) float x[kHidden];
    // ---- the CPU's doorbell, alone on its line (cpu_status shares it: same writer, written first)
    alignas(kLine) volatile uint32_t cpu_seq;
    volatile uint32_t cpu_status;
    uint32_t          pad1_[kLine / 4 - 2];
    // ---- CPU -> GPU payload (written before cpu_seq)
    alignas(16) float partial[kHidden];
};

static_assert(offsetof(Mailbox, gpu_seq) == 0, "doorbell first");
static_assert(offsetof(Mailbox, layer) % 16 == 0 && offsetof(Mailbox, layer) >= kLine, "GPU payload is off the doorbell line and 16-byte aligned");
static_assert(offsetof(Mailbox, tasks) % 16 == 0 && offsetof(Mailbox, x) % 16 == 0 && offsetof(Mailbox, partial) % 16 == 0, "payload arrays are 16-byte aligned");
static_assert(offsetof(Mailbox, cpu_seq) % kLine == 0, "CPU doorbell starts a line");
static_assert(offsetof(Mailbox, cpu_seq) >= offsetof(Mailbox, x) + sizeof(float) * kHidden, "no CPU-written word shares a line with GPU-written payload");
static_assert(offsetof(Mailbox, partial) >= offsetof(Mailbox, cpu_seq) + kLine, "CPU payload is off the CPU doorbell line");
static_assert(sizeof(Mailbox) % kLine == 0, "whole lines");

// The CPU's private copy of one request (taken so that the GPU publishing the next layer cannot change it while the CPU computes).
struct Request {
    uint32_t seq;
    uint32_t layer;
    uint32_t n_tasks;
    uint32_t token_pos;
    CpuTask  tasks[kMaxTasks];
    alignas(16) float x[kHidden];
};

// CPU tasks of a plan: the experts that have rows left for the CPU, in plan order. Returns how many were written (<= kMaxTasks).
PELIAS_SPLIT_HD inline int fill_cpu_tasks(const SplitPlan& plan, CpuTask* out) {
    int k = 0;
    for (int i = 0; i < plan.n_miss && k < kMaxTasks; ++i) {
        if (plan.n_ff - plan.task[i].split_row <= 0) continue;
        out[k].expert = plan.task[i].expert;
        out[k].weight = plan.task[i].weight;
        out[k].row_begin = plan.task[i].split_row;
        out[k].row_end = plan.n_ff;
        ++k;
    }
    return k;
}

// =====================================================================================================================================================
// host half
// =====================================================================================================================================================

inline void cpu_pause() {
#if defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#endif
}

// Ask the driver to start whatever is queued on `stream` (the 03 rev 2 fix). Cheap: 0.11 us when idle. No-op without CUDA.
inline void flush_stream(void* stream) {
#if PELIAS_HANDOFF_HAVE_CUDA
    (void)cudaStreamQuery(static_cast<cudaStream_t>(stream));
#else
    (void)stream;
#endif
}

class HostDoorbell {
public:
    explicit HostDoorbell(Mailbox* mb = nullptr) : mb_(mb) {}
    void attach(Mailbox* mb) { mb_ = mb; }
    Mailbox* mailbox() const { return mb_; }

    // Only while no kernel is using the mailbox. `last_seq` is the sequence number that counts as already done on both sides: the first request must then
    // use last_seq + 1 (a zeroed mailbox means last_seq = 0, first request 1).
    void reset(uint32_t last_seq = 0) const {
        std::memset(static_cast<void*>(mb_), 0, sizeof(Mailbox));
        mb_->gpu_seq = last_seq;
        mb_->cpu_seq = last_seq;
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    uint32_t gpu_seq() const { return mb_->gpu_seq; }

    // Spin until the GPU has rung the bell for `seq` (or a later one). Bounded by timeout_us. With kick_stream and kick_every > 0 the loop also calls
    // flush_stream() every kick_every spins; the default (0) relies on the one flush the launcher did, which measured as fast.
    bool wait_gpu(uint32_t seq, double timeout_us, void* kick_stream = nullptr, uint32_t kick_every = 0) const {
        const auto t0 = std::chrono::steady_clock::now();
        for (uint32_t spin = 0;; ++spin) {
            if (static_cast<int32_t>(mb_->gpu_seq - seq) >= 0) {
                std::atomic_thread_fence(std::memory_order_acquire);
                return true;
            }
            if (kick_every != 0 && kick_stream != nullptr && (spin % kick_every) == 0) flush_stream(kick_stream);
            cpu_pause();
            if ((spin & 1023u) == 0) {
                const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                if (us > timeout_us) return false;
            }
        }
    }

    // Copy the request for `seq` out of the mailbox. Returns false if the GPU has already moved on to a later doorbell (the GPU gave up on this layer after
    // its timeout; drop the work and resync with gpu_seq()). Call after wait_gpu(seq) returned true.
    // A true result means the doorbell was still `seq` after the copy. It does NOT rule out a torn copy: a GPU that gave up on `seq` writes the next payload
    // BEFORE it rings the next bell. That does no harm, because the GPU only abandons `seq` without an answer, so whatever is posted for `seq` is ignored
    // (it waits for a later sequence number). Do not treat an inconsistent payload as an error unless gpu_seq() still equals `seq` afterwards.
    bool read_request(uint32_t seq, Request* out) const {
        if (static_cast<int32_t>(mb_->gpu_seq - seq) > 0) return false;
        out->seq = seq;
        out->layer = mb_->layer;
        uint32_t n = mb_->n_tasks;
        if (n > static_cast<uint32_t>(kMaxTasks)) n = static_cast<uint32_t>(kMaxTasks);
        out->n_tasks = n;
        out->token_pos = mb_->token_pos;
        std::memcpy(out->tasks, mb_->tasks, sizeof(CpuTask) * n);
        std::memcpy(out->x, mb_->x, sizeof(float) * kHidden);
        std::atomic_thread_fence(std::memory_order_acquire);
        return static_cast<int32_t>(mb_->gpu_seq - seq) == 0;      // still the same doorbell: the copy is whole
    }

    // Publish the result for `seq`: partial (kHidden floats; may be null when status != ok), then status, then the doorbell.
    // ONE in-order CPU worker must call wait_gpu / read_request / post, one request at a time. That is what makes a late post safe: if the GPU timed out
    // and rang the bell for a later seq, the worker only gets to that request after posting the old one, and the GPU (waiting for the later seq) ignores
    // the stale cpu_seq, so a stale partial can never be read as the answer to a newer request.
    void post(uint32_t seq, const float* partial, uint32_t status = kStatusOk) const {
        if (partial != nullptr) std::memcpy(mb_->partial, partial, sizeof(float) * kHidden);
        mb_->cpu_status = status;
        std::atomic_thread_fence(std::memory_order_release);
#if defined(__x86_64__) || defined(__i386__)
        _mm_sfence();            // also orders any non-temporal stores a large memcpy might have used
#endif
        mb_->cpu_seq = seq;
    }

private:
    Mailbox* mb_;
};

#if PELIAS_HANDOFF_HAVE_CUDA
struct MailboxHandle {
    Mailbox* host = nullptr;     // CPU pointer
    Mailbox* dev = nullptr;      // pointer to pass to the kernels (equal to host on UVA)
};

// One mapped, pinned, zeroed mailbox. Call cudaSetDeviceFlags(cudaDeviceMapHost) before the first CUDA call of the process if the platform needs it
// (harmless otherwise); microbench 03 did, and it worked.
inline cudaError_t alloc_mailbox(MailboxHandle* h) {
    void* p = nullptr;
    cudaError_t e = cudaHostAlloc(&p, sizeof(Mailbox), cudaHostAllocMapped | cudaHostAllocPortable);
    if (e != cudaSuccess) return e;
    std::memset(p, 0, sizeof(Mailbox));
    void* d = nullptr;
    e = cudaHostGetDevicePointer(&d, p, 0);
    if (e != cudaSuccess) { cudaFreeHost(p); return e; }
    h->host = static_cast<Mailbox*>(p);
    h->dev = static_cast<Mailbox*>(d);
    return cudaSuccess;
}

inline void free_mailbox(MailboxHandle* h) {
    if (h->host != nullptr) cudaFreeHost(h->host);
    h->host = nullptr;
    h->dev = nullptr;
}
#endif

// =====================================================================================================================================================
// device half
// =====================================================================================================================================================
#if defined(__CUDACC__) || defined(PELIAS_CUDA_SHIM)

// 16-byte volatile access to host-mapped memory: not cached on the GPU, issued as one transaction per vector. (Volatile uint4 assignment does not compile in CUDA.)
__device__ __forceinline__ uint4 ld_sys_v4(const void* p) {
#if defined(__CUDA_ARCH__)
    uint4 v;
    asm volatile("ld.volatile.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
#else
    uint4 v;
    std::memcpy(&v, p, sizeof(v));
    return v;
#endif
}

__device__ __forceinline__ void st_sys_v4(void* p, uint4 v) {
#if defined(__CUDA_ARCH__)
    asm volatile("st.volatile.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
#else
    std::memcpy(p, &v, sizeof(v));
#endif
}

// Ring the bell for `seq`. ALL threads of ONE thread block call this together. `x` is the layer input in device memory (kHidden floats, 16-byte aligned);
// `tasks` (n_tasks <= kMaxTasks) may be in shared or global memory. Returns after the doorbell is written (thread 0) / after the barrier (the others).
__device__ __forceinline__ void publish(Mailbox* mb, uint32_t seq, uint32_t layer, uint32_t token_pos, int n_tasks, const CpuTask* tasks, const float* x) {
    const int tid = static_cast<int>(threadIdx.x);
    const int bd = static_cast<int>(blockDim.x);
    if (n_tasks > kMaxTasks) n_tasks = kMaxTasks;
    if (n_tasks < 0) n_tasks = 0;

    constexpr int kVec = kHidden / 4;
    const uint4* xs = reinterpret_cast<const uint4*>(x);
    uint4* xd = reinterpret_cast<uint4*>(mb->x);
    for (int i = tid; i < kVec; i += bd) st_sys_v4(xd + i, xs[i]);

    for (int k = tid; k < n_tasks; k += bd) {                  // a loop: the block may have fewer threads than tasks
        uint4 t;
        t.x = static_cast<unsigned>(tasks[k].expert);
        t.y = static_cast<unsigned>(__float_as_int(tasks[k].weight));
        t.z = static_cast<unsigned>(tasks[k].row_begin);
        t.w = static_cast<unsigned>(tasks[k].row_end);
        st_sys_v4(&mb->tasks[k], t);
    }
    if (tid == 0) {
        uint4 h;
        h.x = layer;
        h.y = static_cast<unsigned>(n_tasks);
        h.z = token_pos;
        h.w = 0u;
        st_sys_v4(&mb->layer, h);
    }
    __threadfence_system();      // every thread: its own stores are performed system-wide before the barrier
    __syncthreads();
    if (tid == 0) {
        mb->gpu_seq = seq;
        __threadfence_system();  // push the doorbell itself out now; it must not sit behind later GPU work
    }
}

// Wait (bounded) for the CPU's result for `seq` and add scale * partial into `acc` (device memory, kHidden floats, 16-byte aligned).
// ALL threads of ONE block call this together; every thread gets the same return value: true if the partial was added.
// On false nothing was added; if d_status != nullptr thread 0 stores kStatusGpuTimeout or kStatusCpuError into *d_status (device memory).
// The loads of the partial are issued as a batch (four 16-byte loads in flight per thread) before the first add uses any of them.
__device__ __forceinline__ bool wait_and_combine(Mailbox* mb, uint32_t seq, float* acc, float scale, long long spin_limit_cycles, uint32_t* d_status) {
    __shared__ int s_state;      // 1 = result arrived and is good, 0 = timed out, 2 = the CPU reported an error
    const int tid = static_cast<int>(threadIdx.x);
    const int bd = static_cast<int>(blockDim.x);
    if (tid == 0) {
        int state = 1;
        const long long t0 = clock64();
        while (static_cast<int32_t>(mb->cpu_seq - seq) < 0) {
            if (clock64() - t0 > spin_limit_cycles) { state = 0; break; }
        }
        __threadfence_system();
        if (state == 1 && mb->cpu_status != kStatusOk) state = 2;
        s_state = state;
    }
    __syncthreads();
    const int state = s_state;
    __syncthreads();             // everyone has read s_state before a later call lets thread 0 overwrite it
    if (state != 1) {
        if (tid == 0 && d_status != nullptr) *d_status = (state == 0) ? kStatusGpuTimeout : kStatusCpuError;
        return false;
    }

    constexpr int kVec = kHidden / 4;
    constexpr int kBatch = 4;
    const uint4* src = reinterpret_cast<const uint4*>(mb->partial);
    float4* dst = reinterpret_cast<float4*>(acc);
    for (int base = tid; base < kVec; base += kBatch * bd) {
        uint4 v[kBatch];
PELIAS_UNROLL
        for (int b = 0; b < kBatch; ++b) {
            v[b].x = v[b].y = v[b].z = v[b].w = 0u;
            const int idx = base + b * bd;
            if (idx < kVec) v[b] = ld_sys_v4(src + idx);
        }
PELIAS_UNROLL
        for (int b = 0; b < kBatch; ++b) {
            const int idx = base + b * bd;
            if (idx < kVec) {
                float4 a = dst[idx];
                a.x += scale * __int_as_float(static_cast<int>(v[b].x));
                a.y += scale * __int_as_float(static_cast<int>(v[b].y));
                a.z += scale * __int_as_float(static_cast<int>(v[b].z));
                a.w += scale * __int_as_float(static_cast<int>(v[b].w));
                dst[idx] = a;
            }
        }
    }
    return true;
}

#endif  // device half

}  // namespace handoff
}  // namespace pelias
