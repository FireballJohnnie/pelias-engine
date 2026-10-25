// File: <project>\moe_track\pelias_serve\e0_layer_profiler.cuh
// Stage E0: where does a decode token's time go, layer by layer?
//
// For each of the 40 layers the executor drops five timestamps into the stream while it enqueues the layer's work:
//     START -> (attention / DeltaNet) -> AFTER_ATTN -> (router) -> AFTER_ROUTER -> (hot experts on the GPU) -> AFTER_HOT -> (rest of the layer: waiting for the CPU
//     experts, adding them in, residual) -> END
// plus the host-side time the CPU thread spent waiting for the cold-expert pool (add_cpu_wait_ms). end_token() waits for the stream once, reads all the timestamps,
// and adds them into per-layer totals; dump_json() writes the per-token averages.
//
// Output fields per layer (ms per token): spine_ms = attention + router, hot_gpu_ms, tail_ms = END - AFTER_HOT (GPU-side time after the hot experts: waiting for the
// CPU result and merging it), cold_wait_ms = what the host measured around the cold pool, launch_gap_ms = total GPU span - sum of the four stage spans (should be ~0;
// a large value means a mark was placed in the wrong spot). A layer whose marks are incomplete in a token is skipped for that token and counted in `skipped`.
//
// Templated over a timing back end so the arithmetic and the JSON can be tested on the host (HostClockBackend); CudaBackend (below) uses cudaEvent_t and is only
// compiled when CUDA (or lib/cuda_shim.h) is available. Not run on a GPU here.
#pragma once

#include "pelias_defs.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#if defined(PELIAS_CUDA_SHIM)
#include "../lib/cuda_shim.h"
#define PELIAS_E0_HAVE_CUDA 1
#elif defined(__CUDACC__) || defined(PELIAS_E0_FORCE_CUDA)
#include <cuda_runtime.h>
#define PELIAS_E0_HAVE_CUDA 1
#endif

namespace pelias {

enum E0Mark { E0_START = 0, E0_AFTER_ATTN, E0_AFTER_ROUTER, E0_AFTER_HOT, E0_END, E0_N_MARKS };

struct E0LayerTotals {
    double attn_ms = 0, router_ms = 0, hot_ms = 0, tail_ms = 0, total_gpu_ms = 0, cold_wait_ms = 0;
    int64_t tokens = 0, skipped = 0;
};

// Host clock back end: an Event is a timestamp in ms. now_override lets a test drive time by hand.
struct HostClockBackend {
    using Stream = void *;
    using Event = double;
    static inline double now_override = -1.0;
    static double now_ms() {
        if (now_override >= 0.0) return now_override;
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    Event make() { return -1.0; }
    void destroy(Event) {}
    void record(Event & e, Stream) { e = now_ms(); }
    void sync(Stream) {}
    bool valid(Event e) const { return e >= 0.0; }
    double elapsed_ms(Event a, Event b) { return b - a; }
};

#ifdef PELIAS_E0_HAVE_CUDA
struct CudaBackend {
    using Stream = cudaStream_t;
    using Event = cudaEvent_t;
    Event make() { Event e = nullptr; cudaEventCreate(&e); return e; }
    void destroy(Event e) { if (e) cudaEventDestroy(e); }
    void record(Event & e, Stream s) { cudaEventRecord(e, s); }
    void sync(Stream s) { cudaStreamSynchronize(s); }
    bool valid(Event) const { return true; }
    double elapsed_ms(Event a, Event b) { float ms = 0.0f; cudaEventElapsedTime(&ms, a, b); return ms; }
};
#endif

template <class B>
class LayerProfiler {
public:
    using Stream = typename B::Stream;
    using Event = typename B::Event;

    explicit LayerProfiler(B backend = B(), int n_layers = N_LAYERS) : be_(backend), n_(n_layers) {
        ev_.resize(static_cast<size_t>(n_) * E0_N_MARKS);
        marked_.assign(ev_.size(), 0);
        for (auto & e : ev_) e = be_.make();
        tot_.resize(n_);
        pending_wait_.assign(n_, 0.0);
    }
    ~LayerProfiler() { for (auto & e : ev_) be_.destroy(e); }
    LayerProfiler(const LayerProfiler &) = delete;
    LayerProfiler & operator=(const LayerProfiler &) = delete;

    void mark(int layer, E0Mark m, Stream s) {
        if (layer < 0 || layer >= n_) return;
        const size_t i = static_cast<size_t>(layer) * E0_N_MARKS + m;
        be_.record(ev_[i], s);
        marked_[i] = 1;
    }
    void add_cpu_wait_ms(int layer, double ms) { if (layer >= 0 && layer < n_) pending_wait_[layer] += ms; }

    // call once per decoded token, after the last layer was enqueued; blocks until the stream has finished
    void end_token(Stream s) {
        be_.sync(s);
        for (int l = 0; l < n_; ++l) {
            E0LayerTotals & t = tot_[l];
            const size_t b = static_cast<size_t>(l) * E0_N_MARKS;
            bool all = true;
            for (int m = 0; m < E0_N_MARKS; ++m) all = all && marked_[b + m];
            if (all) {
                const double attn = be_.elapsed_ms(ev_[b + E0_START], ev_[b + E0_AFTER_ATTN]);
                const double rtr  = be_.elapsed_ms(ev_[b + E0_AFTER_ATTN], ev_[b + E0_AFTER_ROUTER]);
                const double hot  = be_.elapsed_ms(ev_[b + E0_AFTER_ROUTER], ev_[b + E0_AFTER_HOT]);
                const double tail = be_.elapsed_ms(ev_[b + E0_AFTER_HOT], ev_[b + E0_END]);
                const double tot  = be_.elapsed_ms(ev_[b + E0_START], ev_[b + E0_END]);
                t.attn_ms += attn; t.router_ms += rtr; t.hot_ms += hot; t.tail_ms += tail; t.total_gpu_ms += tot;
                t.cold_wait_ms += pending_wait_[l];
                ++t.tokens;
            } else {
                ++t.skipped;
            }
            pending_wait_[l] = 0.0;
            for (int m = 0; m < E0_N_MARKS; ++m) marked_[b + m] = 0;
        }
        ++tokens_;
    }

    int64_t tokens() const { return tokens_; }
    const E0LayerTotals & layer(int l) const { return tot_[l]; }

    std::string json() const {
        std::ostringstream o;
        double s_spine = 0, s_hot = 0, s_tail = 0, s_cold = 0, s_gap = 0, s_total = 0;
        o << "{\n  \"tokens_profiled\": " << tokens_ << ",\n  \"layers\": [\n";
        for (int l = 0; l < n_; ++l) {
            const E0LayerTotals & t = tot_[l];
            const double d = t.tokens > 0 ? static_cast<double>(t.tokens) : 1.0;
            const double spine = (t.attn_ms + t.router_ms) / d, hot = t.hot_ms / d, tail = t.tail_ms / d, cold = t.cold_wait_ms / d, total = t.total_gpu_ms / d;
            const double gap = total - (spine + hot + tail);
            s_spine += spine; s_hot += hot; s_tail += tail; s_cold += cold; s_gap += gap; s_total += total;
            o << "    {\"layer\": " << l << ", \"is_full_attn\": " << (layer_is_full_attention(l) ? "true" : "false")
              << ", \"tokens\": " << t.tokens << ", \"skipped\": " << t.skipped
              << ", \"spine_ms\": " << spine << ", \"hot_gpu_ms\": " << hot << ", \"tail_ms\": " << tail
              << ", \"cold_wait_ms\": " << cold << ", \"launch_gap_ms\": " << gap << ", \"gpu_span_ms\": " << total << "}" << (l + 1 < n_ ? ",\n" : "\n");
        }
        o << "  ],\n  \"totals_ms_per_token\": {\n"
          << "    \"spine_total_ms\": " << s_spine << ",\n    \"hot_gpu_total_ms\": " << s_hot << ",\n    \"tail_total_ms\": " << s_tail << ",\n"
          << "    \"cold_wait_total_ms\": " << s_cold << ",\n    \"launch_gap_total_ms\": " << s_gap << ",\n    \"gpu_span_total_ms\": " << s_total << "\n  }\n}\n";
        return o.str();
    }

    // never overwrites: returns false if the file exists or cannot be written
    bool dump_json(const std::string & path) const {
        if (std::ifstream(path).good()) return false;
        std::ofstream f(path);
        if (!f) return false;
        f << json();
        return static_cast<bool>(f);
    }

private:
    B                         be_;
    int                       n_;
    std::vector<Event>        ev_;
    std::vector<char>         marked_;
    std::vector<E0LayerTotals> tot_;
    std::vector<double>       pending_wait_;
    int64_t                   tokens_ = 0;
};

}  // namespace pelias
