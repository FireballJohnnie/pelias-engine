// File: <project>\moe_track\pelias_serve\cold_pool.h
// Stage E3: a persistent pool of worker threads that computes the CPU ("cold") part of one MoE layer for one decode token:
//
//     out[d] = sum over the n cold experts k of   w_k * ( down_k @ ( silu(gate_k @ x) * (up_k @ x) ) )[d]
//
// with the weights left in their GGUF quantisation (Q4_K / Q6_K / ...) and every dot product done by ggml's own kernels (see pelias_quant.h), exactly as
// llama.cpp's mul_mat_id does it: x is converted to the kernel's activation type (Q8_K) once, the intermediate vector of every expert is converted before the
// down projection, and the routing weight is applied AFTER the down projection (llama.cpp's order).
//
// Why a persistent pool: the workers are created once, optionally pinned to chosen CPUs, and wait for work on an atomic epoch counter. A call costs one atomic
// increment and a few spin-barrier crossings instead of an OpenMP/ggml thread-pool wake-up. Between layers the workers spin; after `park_after_us` without work
// they sleep in short steps so an idle server does not burn the cores.
//
// Synchronisation (the roadmap draft had none between the gate/up phase and the down phase, so a thread could read intermediate values another thread had
// not written yet; here every phase boundary is a barrier):
//     main:   quantise x -> publish the job -> epoch++ (release) ........................................................ wait until done == n_workers (acquire)
//     worker: wait for epoch (acquire) -> A: gate/up rows -> barrier -> A2: quantise the intermediates -> barrier -> B: down rows -> done++ (release)
// Every output row is computed by exactly one thread, always as the same sum in the same order, so the result does not depend on the number of workers.
//
// One caller at a time (run() is not re-entrant). Header-only; needs ggml's headers and -lggml -lggml-cpu -lggml-base.
#pragma once

#include "pelias_defs.h"
#include "pelias_quant.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace pelias {

// One stacked expert matrix: expert e, row r starts at base + e * expert_stride + r * row_bytes.
// A fused gate_up tensor [2*ff rows per expert] is described as two Mat with the same expert_stride: gate = (base, rows = ff), up = (base + ff * row_bytes, rows = ff).
struct Mat {
    const uint8_t * base          = nullptr;
    ggml_type       type          = GGML_TYPE_Q4_K;
    int64_t         rows          = 0;      // rows used per expert (outputs)
    int64_t         cols          = 0;      // inputs
    size_t          row_bytes     = 0;
    size_t          expert_stride = 0;
    const uint8_t * row(int e, int64_t r) const { return base + static_cast<size_t>(e) * expert_stride + static_cast<size_t>(r) * row_bytes; }
};

struct LayerCold {
    Mat gate, up, down;                     // gate/up: rows = ff, cols = hidden;  down: rows = hidden, cols = ff
};

inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
}

class SpinBarrier {
public:
    explicit SpinBarrier(int n = 1) : n_(n) {}
    // spin_yields: busy-wait this many times, then give up the time slice on every further iteration (keeps an oversubscribed machine, such as a test box, moving)
    void wait(int spin_yields) {
        const uint32_t g = gen_.load(std::memory_order_acquire);
        if (count_.fetch_add(1, std::memory_order_acq_rel) == n_ - 1) {
            count_.store(0, std::memory_order_relaxed);
            gen_.fetch_add(1, std::memory_order_release);
        } else {
            int s = 0;
            while (gen_.load(std::memory_order_acquire) == g) {
                if (s < spin_yields) { ++s; cpu_relax(); } else { std::this_thread::yield(); }
            }
        }
    }
private:
    int n_;
    std::atomic<int>      count_{0};
    std::atomic<uint32_t> gen_{0};
};

class ColdPool {
public:
    struct Config {
        int              n_workers     = 7;
        std::vector<int> cpus;                  // CPU ids to pin worker i to (cpus[i]); empty = do not pin. Production: {1..7}, leaving CPU 0 to the CUDA stream thread
        int              spin_yields   = 4096;  // busy-wait iterations before a waiting thread starts yielding its time slice (matters when cores are oversubscribed)
        int              park_after_us = 2000;  // idle this long -> workers sleep in 50 us steps
    };

    explicit ColdPool(const Config & cfg) : cfg_(cfg), nw_(cfg.n_workers), barrier_(cfg.n_workers) {
        if (nw_ < 1 || nw_ > 64) throw std::invalid_argument("ColdPool: n_workers must be 1..64");
        pinned_.assign(nw_, 0);
        for (int t = 0; t < nw_; ++t) threads_.emplace_back([this, t] { worker_main(t); });
        while (ready_.load(std::memory_order_acquire) < nw_) std::this_thread::yield();      // every worker has tried to pin itself; pinned() is now safe to read
    }
    ColdPool(const ColdPool &) = delete;
    ColdPool & operator=(const ColdPool &) = delete;

    ~ColdPool() {
        shutdown_.store(true, std::memory_order_release);
        epoch_.fetch_add(1, std::memory_order_release);
        for (auto & t : threads_) t.join();
    }

    // Which workers could be pinned to the CPU asked for (diagnostics; empty cpus -> all zero)
    const std::vector<int> & pinned() const { return pinned_; }
    int n_workers() const { return nw_; }

    // x: hidden-sized float input of the layer. ids/weights: the n cold experts (expert id and its routing weight). out: hidden floats, overwritten.
    void run(const LayerCold & L, const float * x, const int32_t * ids, const float * weights, int n, float * out) {
        validate(L, n);
        const int hidden = static_cast<int>(L.down.rows);
        if (n == 0) {
            std::memset(out, 0, sizeof(float) * hidden);
            return;
        }
        QuantOps & ogu = ops_gu_;
        QuantOps & odn = ops_dn_;
        std::string err;
        if (!quant_ops_for(L.gate.type, ogu, &err) || !quant_ops_for(L.down.type, odn, &err)) throw std::runtime_error("ColdPool: " + err);
        const int64_t ff = L.gate.rows;
        xq_.resize(ogu.act_row_bytes(L.gate.cols));
        iq_.resize(static_cast<size_t>(n) * odn.act_row_bytes(ff));
        inter_.resize(static_cast<size_t>(n) * ff);
        ogu.quantize_act(x, xq_.data(), L.gate.cols);

        job_.L = &L; job_.ids = ids; job_.w = weights; job_.n = n; job_.out = out;
        job_.ff = ff; job_.hidden = hidden; job_.gu = &ops_gu_; job_.dn = &ops_dn_;
        job_.xq = xq_.data(); job_.iq = iq_.data(); job_.inter = inter_.data();
        job_.iq_row = odn.act_row_bytes(ff);

        done_.store(0, std::memory_order_relaxed);
        epoch_.fetch_add(1, std::memory_order_release);                  // publishes everything written above
        int spins = 0;
        while (done_.load(std::memory_order_acquire) < nw_) wait_step(spins);
    }

private:
    struct Job {
        const LayerCold * L = nullptr;
        const int32_t *   ids = nullptr;
        const float *     w = nullptr;
        int               n = 0;
        float *           out = nullptr;
        int64_t           ff = 0;
        int               hidden = 0;
        const QuantOps *  gu = nullptr;
        const QuantOps *  dn = nullptr;
        const uint8_t *   xq = nullptr;
        uint8_t *         iq = nullptr;
        float *           inter = nullptr;
        size_t            iq_row = 0;
    };

    void validate(const LayerCold & L, int n) const {
        if (n < 0 || n > MAX_COLD) throw std::invalid_argument("ColdPool: n must be 0.." + std::to_string(MAX_COLD));
        if (L.gate.type != L.up.type) throw std::invalid_argument("ColdPool: gate and up must have the same type");
        if (L.gate.rows != L.up.rows || L.gate.cols != L.up.cols) throw std::invalid_argument("ColdPool: gate and up shapes differ");
        if (L.down.cols != L.gate.rows || L.down.rows != L.gate.cols) throw std::invalid_argument("ColdPool: down shape does not match gate/up");
        if (!L.gate.base || !L.up.base || !L.down.base) throw std::invalid_argument("ColdPool: null weight pointer");
        if (L.gate.row_bytes != ggml_row_size(L.gate.type, L.gate.cols) || L.up.row_bytes != ggml_row_size(L.up.type, L.up.cols) ||
            L.down.row_bytes != ggml_row_size(L.down.type, L.down.cols))
            throw std::invalid_argument("ColdPool: row_bytes does not match the type and width");
        if (L.gate.cols % ggml_blck_size(L.gate.type) != 0 || L.down.cols % ggml_blck_size(L.down.type) != 0)
            throw std::invalid_argument("ColdPool: width is not a multiple of the block size");
    }

    void wait_step(int & spins) const {
        if (spins < cfg_.spin_yields) { ++spins; cpu_relax(); } else { std::this_thread::yield(); }
    }

    static void silu_mul(const float g, const float u, float & o) { o = (g / (1.0f + std::exp(-g))) * u; }

    void worker_main(int tid) {
#if defined(__linux__)
        if (!cfg_.cpus.empty() && tid < static_cast<int>(cfg_.cpus.size())) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cfg_.cpus[tid], &set);
            pinned_[tid] = (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0) ? 1 : 0;
        }
#endif
        ready_.fetch_add(1, std::memory_order_release);
        uint64_t seen = 0;
        auto last_work = std::chrono::steady_clock::now();
        int spins = 0;
        for (;;) {
            const uint64_t cur = epoch_.load(std::memory_order_acquire);
            if (cur == seen) {
                if (spins < cfg_.spin_yields) { ++spins; cpu_relax(); continue; }
                std::this_thread::yield();
                if (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - last_work).count() > cfg_.park_after_us)
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                continue;
            }
            seen = cur;
            spins = 0;
            if (shutdown_.load(std::memory_order_acquire)) return;
            work(tid);
            last_work = std::chrono::steady_clock::now();
            done_.fetch_add(1, std::memory_order_release);
        }
    }

    void work(int tid) {
        const Job & j = job_;
        const LayerCold & L = *j.L;
        const int64_t cols = L.gate.cols;

        // phase A: rows of the gate/up projections of all cold experts, split into contiguous ranges
        const int64_t total = static_cast<int64_t>(j.n) * j.ff;
        const int64_t beg = total * tid / nw_, end = total * (tid + 1) / nw_;
        for (int64_t i = beg; i < end; ++i) {
            const int k = static_cast<int>(i / j.ff);
            const int64_t r = i % j.ff;
            const int e = j.ids[k];
            const float g = j.gu->dot(cols, L.gate.row(e, r), j.xq);
            const float u = j.gu->dot(cols, L.up.row(e, r), j.xq);
            silu_mul(g, u, j.inter[static_cast<size_t>(k) * j.ff + r]);
        }
        barrier_.wait(cfg_.spin_yields);                                              // every intermediate value is written before anyone converts or reads it

        // phase A2: convert each expert's intermediate vector to the down projection's activation type
        for (int k = tid; k < j.n; k += nw_) j.dn->quantize_act(j.inter + static_cast<size_t>(k) * j.ff, j.iq + static_cast<size_t>(k) * j.iq_row, j.ff);
        barrier_.wait(cfg_.spin_yields);

        // phase B: rows of the down projection, summed over the cold experts in a fixed order
        const int64_t hb = static_cast<int64_t>(j.hidden) * tid / nw_, he = static_cast<int64_t>(j.hidden) * (tid + 1) / nw_;
        for (int64_t r = hb; r < he; ++r) {
            float acc = 0.0f;
            for (int k = 0; k < j.n; ++k) acc += j.w[k] * j.dn->dot(L.down.cols, L.down.row(j.ids[k], r), j.iq + static_cast<size_t>(k) * j.iq_row);
            j.out[r] = acc;
        }
    }

    Config                   cfg_;
    int                      nw_;
    SpinBarrier              barrier_;
    std::vector<std::thread> threads_;
    std::vector<int>         pinned_;
    std::atomic<uint64_t>    epoch_{0};
    std::atomic<int>         done_{0};
    std::atomic<bool>        shutdown_{false};
    std::atomic<int>         ready_{0};
    Job                      job_;
    QuantOps                 ops_gu_, ops_dn_;
    std::vector<uint8_t>     xq_, iq_;
    std::vector<float>       inter_;
};

}  // namespace pelias
