// File: <project>\moe_track\microbench\03_handoff_latency.cu
// Step 2c: how long does one GPU -> CPU -> GPU handoff take on this WSL2 setup?
//
// This is the per-layer cost a hybrid design pays whenever a layer's experts are computed on the CPU (or a miss is
// fetched): the GPU publishes the router's choice, the CPU notices, produces the expert output, the GPU consumes it.
//
// Build:  nvcc -O3 -std=c++17 -arch=sm_89 03_handoff_latency.cu -o bin/03_handoff_latency   (with the cuda_shim nvcc: see the run notes)
// Run:    bin/03_handoff_latency [--iters 2000] [--rounds 4000] [--timeout-ms 500] [--cpu-layers 28] [--out FILE.json]
//
// Revision 2 (after the first run on this machine): the first version timed out in its two "CPU waits for a GPU flag" modes on
// WSL2 (a kernel launched into a stream may sit in the driver's queue until something flushes it, and the CPU was spinning on
// the flag without ever calling the driver), and it had no mode that measures a handoff WITHOUT a launch + stream sync per
// round. Both are fixed below. Timings in microseconds, median / p90 / p99 / max over the iterations.
//
//   empty_launch        empty kernel + stream sync                                   (the driver's own floor; every sync-based mode includes it)
//   launch_after_plain  GPU writes a flag -> CPU spins on it with NO driver call -> CPU writes 4 KB -> CPU launches the consumer -> sync
//                       This is a PROBE. If it times out and "visible_after_sync" is true, the launch was queued and not started.
//   launch_after_kick   same, but one cudaStreamQuery() right after the launch (it flushes the queue), then a pure spin
//   launch_after_loop   same, but the CPU keeps calling cudaStreamQuery() while it waits
//   spin_wait           the consumer kernel is already queued and polls a mapped flag the CPU sets (bounded spin); uses the first
//                       working nudge mode
//   copy_roundtrip      GPU result copied to pinned memory with cudaMemcpyAsync -> CPU -> 4 KB copied back -> consumer kernel
//   pingpong_flag       ONE resident kernel, `rounds` round trips GPU -> CPU -> GPU through mapped flags; no launch and no sync per
//                       round. The time is one full round trip (CPU sets flag -> GPU sees it -> GPU sets flag -> CPU sees it).
//   pingpong_4k         the same, and the GPU also reads the 4 KB the CPU just wrote (what a layer handoff really does)
//   query_idle          cost of one cudaStreamQuery() call on an idle stream (what a nudge costs the CPU)
// The sync-based modes include a cudaStreamSynchronize, so they cannot be faster than empty_launch. A persistent/polling design
// (the one the <20 us target is about) pays pingpong_4k, not launch_after.
// A spinning kernel holds the GPU while it waits and can misbehave under WDDM/WSL2 scheduling; every wait is bounded and the
// timeouts are counted and reported.
#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "../lib/pl_cuda.h"

namespace {

struct Mailbox {
    alignas(64) volatile int32_t gpu_step;      // written by the GPU, polled by the CPU
    alignas(64) volatile int32_t cpu_step;      // written by the CPU, polled by the (spinning) GPU kernel
    alignas(64) int32_t router_ids[16];         // 64 bytes: top-8 ids + weights as int32
    alignas(64) uint16_t expert_out[2048];      // 4 KB: one token's expert output (2048 x bf16)
};

constexpr long long SPIN_LIMIT_CYCLES = 200000000LL;  // ~80 ms at 2.5 GHz, far below the 2 s display-driver timeout
constexpr int kWarm = 50;                              // iterations discarded at the start of each sync-based mode
constexpr int kPingWarm = 200;                         // round trips discarded at the start of a ping-pong run

__global__ void k_empty() {}

__global__ void k_gpu_publish(Mailbox* mb, int32_t step) {
    if (threadIdx.x < 16) mb->router_ids[threadIdx.x] = step + int32_t(threadIdx.x);
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        mb->gpu_step = step;
    }
}

__global__ void k_gpu_consume(const Mailbox* mb, uint32_t* sink) {
    uint32_t sum = 0;
    for (int i = int(threadIdx.x); i < 2048; i += int(blockDim.x)) sum += mb->expert_out[i];
    atomicAdd(sink, sum);
}

__global__ void k_gpu_wait_and_consume(Mailbox* mb, int32_t step, uint32_t* sink, int* timed_out) {
    if (threadIdx.x == 0) {
        const long long t0 = clock64();
        while (mb->cpu_step < step) {
            if (clock64() - t0 > SPIN_LIMIT_CYCLES) { *timed_out = 1; break; }
        }
        __threadfence_system();
    }
    __syncthreads();
    uint32_t sum = 0;
    for (int i = int(threadIdx.x); i < 2048; i += int(blockDim.x)) sum += mb->expert_out[i];
    atomicAdd(sink, sum);
}

__global__ void k_gpu_write_device_ids(int32_t* d_ids, int32_t step) {
    if (threadIdx.x < 16) d_ids[threadIdx.x] = step + int32_t(threadIdx.x);
}

// One resident kernel: `rounds` times { publish gpu_step = r; wait for cpu_step >= r; optionally read the 4 KB payload }.
// One block; thread 0 does the handshake, the whole block reads the payload. Bounded: gives up after SPIN_LIMIT_CYCLES per wait.
__global__ void k_pingpong(Mailbox* mb, int32_t rounds, int read_payload, uint32_t* sink, int* timed_out) {
    __shared__ int s_abort;
    uint32_t sum = 0;
    for (int32_t r = 1; r <= rounds; ++r) {
        if (threadIdx.x == 0) {
            __threadfence_system();
            mb->gpu_step = r;
            __threadfence_system();
            const long long t0 = clock64();
            int abort_now = 0;
            while (mb->cpu_step < r) {
                if (clock64() - t0 > SPIN_LIMIT_CYCLES) { *timed_out = 1; abort_now = 1; break; }
            }
            __threadfence_system();
            s_abort = abort_now;
        }
        __syncthreads();
        if (s_abort) break;
        if (read_payload) {
            const volatile uint16_t* p = mb->expert_out;
            for (int i = int(threadIdx.x); i < 2048; i += int(blockDim.x)) sum += p[i];
        }
        __syncthreads();
    }
    atomicAdd(sink, sum);
}

struct Stats {
    double median = 0, p90 = 0, p99 = 0, max = 0, mean = 0;
};

Stats summarize(std::vector<double> us) {
    Stats s;
    if (us.empty()) return s;
    std::sort(us.begin(), us.end());
    auto q = [&](double f) { return us[std::min(us.size() - 1, size_t(f * double(us.size())))]; };
    s.median = q(0.5); s.p90 = q(0.9); s.p99 = q(0.99); s.max = us.back();
    double t = 0; for (double v : us) t += v;
    s.mean = t / double(us.size());
    return s;
}

double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

enum class Nudge { None, Once, Loop };   // how the CPU asks the driver to start queued work while it waits

// Spin until *flag >= want. With `loop_nudge` the CPU also calls cudaStreamQuery() on `st` every 64 spins.
bool wait_flag(const volatile int32_t* flag, int32_t want, double timeout_us, cudaStream_t st = nullptr, bool loop_nudge = false) {
    const double t0 = now_us();
    for (uint32_t spin = 0;; ++spin) {
        if (*flag >= want) { std::atomic_thread_fence(std::memory_order_acquire); return true; }
        if (loop_nudge && (spin & 63u) == 0) (void)cudaStreamQuery(st);
        _mm_pause();
        if ((spin & 1023u) == 0 && now_us() - t0 > timeout_us) return false;
    }
}

struct ModeStats {
    std::vector<double> us;
    int cpu_timeouts = 0;
    int gpu_timeouts = 0;
    bool visible_after_sync = false;   // after a CPU-side timeout: was the GPU's write there once the stream was synchronised?
    bool ran = false;
    bool ok() const { return ran && cpu_timeouts == 0 && !us.empty(); }
};

struct Ctx {
    Mailbox* h_mb = nullptr;
    Mailbox* d_mb = nullptr;
    uint32_t* d_sink = nullptr;
    int* h_timeout = nullptr;
    int* d_timeout = nullptr;
    int32_t* d_ids = nullptr;
    int32_t* h_ids = nullptr;
    uint16_t* d_out = nullptr;
    cudaStream_t st{};
    double cpu_timeout_us = 5e5;
    int iters = 2000;
};

void reset_mb(Ctx& c) { c.h_mb->gpu_step = 0; c.h_mb->cpu_step = 0; }

ModeStats run_launch_after(Ctx& c, Nudge nd) {
    ModeStats m; m.ran = true;
    reset_mb(c);
    int32_t step = 0;
    for (int i = 0; i < c.iters + kWarm; ++i) {
        ++step;
        const double t0 = now_us();
        PL_LAUNCH(k_gpu_publish, 1, 32, 0, c.st, c.d_mb, step);
        if (nd != Nudge::None) (void)cudaStreamQuery(c.st);
        if (!wait_flag(&c.h_mb->gpu_step, step, c.cpu_timeout_us, c.st, nd == Nudge::Loop)) {
            ++m.cpu_timeouts;
            PL_CUDA_CHECK(cudaStreamSynchronize(c.st));
            m.visible_after_sync = (c.h_mb->gpu_step >= step);
            break;
        }
        std::memset(c.h_mb->expert_out, step & 0xFF, sizeof c.h_mb->expert_out);
        std::atomic_thread_fence(std::memory_order_release);
        PL_LAUNCH(k_gpu_consume, 1, 128, 0, c.st, c.d_mb, c.d_sink);
        PL_CUDA_CHECK(cudaStreamSynchronize(c.st));
        if (i >= kWarm) m.us.push_back(now_us() - t0);
    }
    return m;
}

ModeStats run_spin_wait(Ctx& c, Nudge nd) {
    ModeStats m; m.ran = true;
    reset_mb(c);
    int32_t step = 0;
    for (int i = 0; i < c.iters + kWarm; ++i) {
        ++step;
        *c.h_timeout = 0;
        const double t0 = now_us();
        PL_LAUNCH(k_gpu_publish, 1, 32, 0, c.st, c.d_mb, step);
        PL_LAUNCH(k_gpu_wait_and_consume, 1, 128, 0, c.st, c.d_mb, step, c.d_sink, c.d_timeout);   // queued before the CPU is done
        if (nd != Nudge::None) (void)cudaStreamQuery(c.st);
        if (!wait_flag(&c.h_mb->gpu_step, step, c.cpu_timeout_us, c.st, nd == Nudge::Loop)) {
            ++m.cpu_timeouts;
            c.h_mb->cpu_step = step;                       // release the queued consumer instead of letting it spin to its limit
            PL_CUDA_CHECK(cudaStreamSynchronize(c.st));
            m.visible_after_sync = (c.h_mb->gpu_step >= step);
            break;
        }
        std::memset(c.h_mb->expert_out, step & 0xFF, sizeof c.h_mb->expert_out);
        std::atomic_thread_fence(std::memory_order_release);
        c.h_mb->cpu_step = step;
        PL_CUDA_CHECK(cudaStreamSynchronize(c.st));
        if (*c.h_timeout) ++m.gpu_timeouts;
        if (i >= kWarm) m.us.push_back(now_us() - t0);
    }
    return m;
}

ModeStats run_copy_roundtrip(Ctx& c) {
    ModeStats m; m.ran = true;
    for (int i = 0; i < c.iters + kWarm; ++i) {
        const int32_t s = int32_t(i + 1);
        const double t0 = now_us();
        PL_LAUNCH(k_gpu_write_device_ids, 1, 32, 0, c.st, c.d_ids, s);
        PL_CUDA_CHECK(cudaMemcpyAsync(c.h_ids, c.d_ids, 64, cudaMemcpyDeviceToHost, c.st));
        PL_CUDA_CHECK(cudaStreamSynchronize(c.st));
        std::memset(c.h_mb->expert_out, s & 0xFF, sizeof c.h_mb->expert_out);
        PL_CUDA_CHECK(cudaMemcpyAsync(c.d_out, c.h_mb->expert_out, 4096, cudaMemcpyHostToDevice, c.st));
        PL_LAUNCH(k_gpu_consume, 1, 128, 0, c.st, c.d_mb, c.d_sink);
        PL_CUDA_CHECK(cudaStreamSynchronize(c.st));
        if (i >= kWarm) m.us.push_back(now_us() - t0);
    }
    return m;
}

// Resident-kernel ping-pong. The sample for round r is the time from the CPU releasing round r-1 to the CPU seeing the GPU's
// flag for round r: GPU sees our flag, (optionally reads 4 KB), GPU writes its flag, CPU sees it.
ModeStats run_pingpong(Ctx& c, int rounds, bool payload) {
    ModeStats m; m.ran = true;
    reset_mb(c);
    *c.h_timeout = 0;
#if defined(__CUDACC__)
    PL_LAUNCH(k_pingpong, 1, 128, 0, c.st, c.d_mb, int32_t(rounds), payload ? 1 : 0, c.d_sink, c.d_timeout);
#else
    // CPU-shim build only (no GPU): a thread stands in for the resident kernel so the CPU side of the protocol can be tested.
    std::thread gpu_emul([&]() { PL_LAUNCH(k_pingpong, 1, 128, 0, c.st, c.d_mb, int32_t(rounds), payload ? 1 : 0, c.d_sink, c.d_timeout); });
#endif
    (void)cudaStreamQuery(c.st);                 // flush: a freshly launched kernel may sit in the queue until the driver is called
    double t_prev = 0;
    bool aborted = false;
    for (int32_t r = 1; r <= rounds; ++r) {
        if (!wait_flag(&c.h_mb->gpu_step, r, c.cpu_timeout_us, c.st, r == 1)) { ++m.cpu_timeouts; aborted = true; break; }
        const double t_seen = now_us();
        if (r > kPingWarm) m.us.push_back(t_seen - t_prev);
        if (payload) std::memset(c.h_mb->expert_out, r & 0xFF, sizeof c.h_mb->expert_out);
        std::atomic_thread_fence(std::memory_order_release);
        c.h_mb->cpu_step = r;
        t_prev = now_us();
    }
    if (aborted) c.h_mb->cpu_step = rounds;      // let the resident kernel run out instead of spinning to its limit
    PL_CUDA_CHECK(cudaStreamSynchronize(c.st));
#if !defined(__CUDACC__)
    gpu_emul.join();
#endif
    if (*c.h_timeout) ++m.gpu_timeouts;
    return m;
}

std::string mode_json(const char* name, const ModeStats& m) {
    const Stats s = summarize(m.us);
    std::ostringstream o;
    o << "    \"" << name << "\": {\"median_us\": " << s.median << ", \"p90_us\": " << s.p90 << ", \"p99_us\": " << s.p99
      << ", \"max_us\": " << s.max << ", \"mean_us\": " << s.mean << ", \"samples\": " << m.us.size()
      << ", \"ran\": " << (m.ran ? "true" : "false") << ", \"cpu_timeouts\": " << m.cpu_timeouts
      << ", \"gpu_spin_timeouts\": " << m.gpu_timeouts << ", \"visible_after_sync\": " << (m.visible_after_sync ? "true" : "false") << "}";
    return o.str();
}

void print_mode(const char* label, const ModeStats& m, const char* note = "") {
    if (!m.ran) { std::printf("[2c] %-19s not run\n", label); return; }
    const Stats s = summarize(m.us);
    std::printf("[2c] %-19s median %8.2f us  p90 %8.2f  p99 %8.2f  max %9.2f   (%zu samples, CPU timeouts %d, GPU spin timeouts %d)%s\n",
                label, s.median, s.p90, s.p99, s.max, m.us.size(), m.cpu_timeouts, m.gpu_timeouts, note);
}

}  // namespace

int main(int argc, char** argv) {
    int iters = 2000, rounds = 4000, cpu_layers = 28;
    double timeout_ms = 500.0;
    std::string out_path = "<project>/logs/floors/handoff_latency.json";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--rounds" && i + 1 < argc) rounds = std::atoi(argv[++i]);
        else if (a == "--timeout-ms" && i + 1 < argc) timeout_ms = std::atof(argv[++i]);
        else if (a == "--cpu-layers" && i + 1 < argc) cpu_layers = std::atoi(argv[++i]);
        else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--iters N] [--rounds N] [--timeout-ms MS] [--cpu-layers N] [--out FILE]\n", argv[0]);
            return 2;
        }
    }
    if (rounds <= kPingWarm + 10) { std::fprintf(stderr, "--rounds must be above %d\n", kPingWarm + 10); return 2; }

    Ctx c;
    c.iters = iters;
    c.cpu_timeout_us = timeout_ms * 1000.0;
    cudaSetDeviceFlags(cudaDeviceMapHost);   // harmless if the context already exists
    PL_CUDA_CHECK(cudaHostAlloc(&c.h_mb, sizeof(Mailbox), cudaHostAllocMapped | cudaHostAllocPortable));
    PL_CUDA_CHECK(cudaHostGetDevicePointer(&c.d_mb, c.h_mb, 0));
    std::memset(static_cast<void*>(c.h_mb), 0, sizeof(Mailbox));
    PL_CUDA_CHECK(cudaMalloc(&c.d_sink, sizeof(uint32_t)));
    PL_CUDA_CHECK(cudaMemset(c.d_sink, 0, sizeof(uint32_t)));
    PL_CUDA_CHECK(cudaHostAlloc(&c.h_timeout, sizeof(int), cudaHostAllocMapped | cudaHostAllocPortable));
    PL_CUDA_CHECK(cudaHostGetDevicePointer(&c.d_timeout, c.h_timeout, 0));
    *c.h_timeout = 0;
    PL_CUDA_CHECK(cudaMalloc(&c.d_ids, 64));
    PL_CUDA_CHECK(cudaHostAlloc(&c.h_ids, 64, cudaHostAllocPortable));
    PL_CUDA_CHECK(cudaMalloc(&c.d_out, 4096));
    PL_CUDA_CHECK(cudaStreamCreateWithFlags(&c.st, cudaStreamNonBlocking));

    for (int warm = 0; warm < 50; ++warm) {             // warm-up: load modules, touch pages, settle clocks
        PL_LAUNCH(k_empty, 1, 1, 0, c.st);
        PL_CUDA_CHECK(cudaStreamSynchronize(c.st));
    }

    // ---- empty launch + sync
    ModeStats empty; empty.ran = true;
    for (int i = 0; i < iters; ++i) {
        const double t0 = now_us();
        PL_LAUNCH(k_empty, 1, 1, 0, c.st);
        PL_CUDA_CHECK(cudaStreamSynchronize(c.st));
        empty.us.push_back(now_us() - t0);
    }

    // ---- cost of one cudaStreamQuery on an idle stream
    ModeStats qidle; qidle.ran = true;
    for (int i = 0; i < 1000; ++i) {
        const double t0 = now_us();
        (void)cudaStreamQuery(c.st);
        qidle.us.push_back(now_us() - t0);
    }

    // ---- CPU waits for a GPU flag: plain probe, then one kick, then a polling nudge
    const ModeStats la_plain = run_launch_after(c, Nudge::None);
    const ModeStats la_kick = run_launch_after(c, Nudge::Once);
    const ModeStats la_loop = run_launch_after(c, Nudge::Loop);
    const ModeStats* la_used = la_plain.ok() ? &la_plain : la_kick.ok() ? &la_kick : la_loop.ok() ? &la_loop : nullptr;
    const char* la_used_name = la_plain.ok() ? "plain" : la_kick.ok() ? "kick" : la_loop.ok() ? "loop" : "none";

    // ---- spin_wait (bounded), with the first nudge that worked for launch_after
    const Nudge sw_nudge = la_plain.ok() ? Nudge::None : la_kick.ok() ? Nudge::Once : Nudge::Loop;
    const ModeStats spin = run_spin_wait(c, sw_nudge);

    const ModeStats copy = run_copy_roundtrip(c);

    // ---- resident-kernel ping-pong
    const ModeStats pp_flag = run_pingpong(c, rounds, false);
    const ModeStats pp_4k = run_pingpong(c, rounds, true);
    const ModeStats* pp_used = pp_4k.ok() ? &pp_4k : pp_flag.ok() ? &pp_flag : nullptr;

    print_mode("empty_launch", empty, "   <- floor of every sync-based mode");
    print_mode("query_idle", qidle, "   <- cost of one cudaStreamQuery");
    print_mode("launch_after_plain", la_plain, la_plain.cpu_timeouts ? (la_plain.visible_after_sync ? "   <- TIMED OUT, but the flag was there after the sync: the launch was queued, not started" : "   <- TIMED OUT and the flag was still missing after the sync") : "");
    print_mode("launch_after_kick", la_kick);
    print_mode("launch_after_loop", la_loop);
    std::printf("[2c] sync-based handoff used for 'handoff_us_per_layer': %s\n", la_used_name);
    print_mode("spin_wait", spin, sw_nudge == Nudge::None ? "   (no nudge)" : sw_nudge == Nudge::Once ? "   (one kick)" : "   (polling nudge)");
    print_mode("copy_roundtrip", copy);
    print_mode("pingpong_flag", pp_flag, "   <- one resident kernel, no launch/sync per round");
    print_mode("pingpong_4k", pp_4k, "   <- same + GPU reads the 4 KB the CPU wrote: the number the <20 us target is about");
    if (la_used) {
        const Stats a = summarize(la_used->us);
        std::printf("[2c] per-token cost with %d CPU-path layers (sync-based): %.3f ms median, %.3f ms p99\n", cpu_layers,
                    a.median * cpu_layers / 1000.0, a.p99 * cpu_layers / 1000.0);
    }
    if (pp_used) {
        const Stats p = summarize(pp_used->us);
        std::printf("[2c] per-token cost with 40 layers handing off through a resident kernel: %.3f ms median, %.3f ms p99\n",
                    p.median * 40.0 / 1000.0, p.p99 * 40.0 / 1000.0);
        std::printf("[2c] floor check (<= 20 us median, <= 50 us p99): %s\n", (p.median <= 20.0 && p.p99 <= 50.0) ? "MET" : "NOT MET");
    } else {
        std::printf("[2c] WARNING: no ping-pong run completed; the resident-kernel handoff could not be measured\n");
    }

    const int total_cpu_timeouts = la_plain.cpu_timeouts + la_kick.cpu_timeouts + la_loop.cpu_timeouts + spin.cpu_timeouts +
                                   pp_flag.cpu_timeouts + pp_4k.cpu_timeouts;
    if (total_cpu_timeouts) std::printf("[2c] note: %d CPU-side waits timed out in total (see the per-mode lines); a mode with timeouts has no valid numbers\n", total_cpu_timeouts);

    const Stats la_s = la_used ? summarize(la_used->us) : Stats();
    const Stats pp_s = pp_used ? summarize(pp_used->us) : Stats();
    const bool batching_suspected = la_plain.cpu_timeouts && la_plain.visible_after_sync;
    std::ostringstream js;
    js << "{\n  \"revision\": 2,\n  \"iterations\": " << iters << ",\n  \"rounds\": " << rounds << ",\n  \"cpu_layers_assumed\": " << cpu_layers << ",\n"
       << "  \"handoff_mode_used\": \"" << la_used_name << "\",\n"
       << "  \"handoff_us_per_layer\": " << la_s.median << ",\n  \"handoff_ms_per_layer\": " << la_s.median / 1000.0 << ",\n"
       << "  \"handoff_ms_per_token_40_layers\": " << la_s.median * 40.0 / 1000.0 << ",\n"
       << "  \"handoff_ms_per_token_cpu_layers\": " << la_s.median * cpu_layers / 1000.0 << ",\n"
       << "  \"handoff_resident_rtt_us\": " << pp_s.median << ",\n  \"handoff_resident_rtt_p99_us\": " << pp_s.p99 << ",\n"
       << "  \"resident_floor_met_20us\": " << ((pp_used && pp_s.median <= 20.0 && pp_s.p99 <= 50.0) ? "true" : "false") << ",\n"
       << "  \"launch_batching_suspected\": " << (batching_suspected ? "true" : "false") << ",\n"
       << "  \"cpu_wait_timeouts\": " << total_cpu_timeouts << ",\n  \"modes\": {\n"
       << mode_json("empty_launch", empty) << ",\n" << mode_json("query_idle", qidle) << ",\n"
       << mode_json("launch_after_plain", la_plain) << ",\n" << mode_json("launch_after_kick", la_kick) << ",\n"
       << mode_json("launch_after_loop", la_loop) << ",\n"
       << mode_json("launch_after", la_used ? *la_used : ModeStats()) << ",\n"
       << mode_json("spin_wait", spin) << ",\n" << mode_json("copy_roundtrip", copy) << ",\n"
       << mode_json("pingpong_flag", pp_flag) << ",\n" << mode_json("pingpong_4k", pp_4k) << "\n  }\n}\n";
    std::ofstream out(out_path);
    if (out) { out << js.str(); std::printf("[2c] wrote %s\n", out_path.c_str()); }
    else std::fprintf(stderr, "cannot write %s; JSON follows\n%s", out_path.c_str(), js.str().c_str());

    cudaStreamDestroy(c.st);
    cudaFree(c.d_sink); cudaFree(c.d_ids); cudaFree(c.d_out);
    cudaFreeHost(c.h_ids); cudaFreeHost(c.h_timeout); cudaFreeHost(c.h_mb);
    return (la_used && pp_used) ? 0 : 5;
}
