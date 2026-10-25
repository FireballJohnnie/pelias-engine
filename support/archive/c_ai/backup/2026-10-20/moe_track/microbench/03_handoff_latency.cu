// File: <project>\moe_track\microbench\03_handoff_latency.cu
// Step 2c: how long does one GPU -> CPU -> GPU handoff take on this WSL2 setup?
//
// This is the per-layer cost a hybrid design pays whenever a layer's experts are computed on the CPU (or a miss is
// fetched): the GPU publishes the router's choice, the CPU notices, produces the expert output, the GPU consumes it.
//
// Build (done by microbench/build.sh):  nvcc -O3 -std=c++17 -arch=sm_89 03_handoff_latency.cu -o bin/03_handoff_latency
// Run:   bin/03_handoff_latency [--iters 2000] [--cpu-layers 28] [--out <project>/logs/floors/handoff_latency.json]
//
// Four measurements, each reported as median / p90 / p99 / max in microseconds over --iters iterations:
//   empty_launch   an empty kernel + stream sync                                   (the driver's own floor)
//   launch_after   GPU writes a flag -> CPU sees it -> CPU writes 4 KB -> CPU launches the consumer kernel -> sync
//   spin_wait      the consumer kernel is already queued and polls a mapped flag the CPU sets (bounded spin, no hang)
//   copy_roundtrip GPU result copied to pinned memory with cudaMemcpyAsync -> CPU -> 4 KB copied back -> consumer kernel
// launch_after is the number to use (it works with ordinary streams and with llama.cpp's graph execution).
// spin_wait shows what a persistent/polling design could save, but a spinning kernel holds the GPU while it waits and can
// misbehave under WDDM/WSL2 scheduling; its timeouts are counted and reported.
#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
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

bool wait_flag(const volatile int32_t* flag, int32_t want, double timeout_us) {
    const double t0 = now_us();
    for (uint32_t spin = 0;; ++spin) {
        if (*flag >= want) { std::atomic_thread_fence(std::memory_order_acquire); return true; }
        _mm_pause();
        if ((spin & 1023u) == 0 && now_us() - t0 > timeout_us) return false;
    }
}

std::string stats_json(const char* name, const Stats& s, const char* extra = "") {
    std::ostringstream o;
    o << "    \"" << name << "\": {\"median_us\": " << s.median << ", \"p90_us\": " << s.p90 << ", \"p99_us\": " << s.p99
      << ", \"max_us\": " << s.max << ", \"mean_us\": " << s.mean << extra << "}";
    return o.str();
}

}  // namespace

int main(int argc, char** argv) {
    int iters = 2000, cpu_layers = 28;
    std::string out_path = "<project>/logs/floors/handoff_latency.json";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--cpu-layers" && i + 1 < argc) cpu_layers = std::atoi(argv[++i]);
        else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
        else { std::fprintf(stderr, "usage: %s [--iters N] [--cpu-layers N] [--out FILE]\n", argv[0]); return 2; }
    }

    cudaSetDeviceFlags(cudaDeviceMapHost);   // harmless if the context already exists
    Mailbox* h_mb = nullptr;
    Mailbox* d_mb = nullptr;
    PL_CUDA_CHECK(cudaHostAlloc(&h_mb, sizeof(Mailbox), cudaHostAllocMapped | cudaHostAllocPortable));
    PL_CUDA_CHECK(cudaHostGetDevicePointer(&d_mb, h_mb, 0));
    std::memset(const_cast<Mailbox*>(h_mb), 0, sizeof(Mailbox));
    uint32_t* d_sink = nullptr;
    int* h_timeout = nullptr;
    int* d_timeout = nullptr;
    int32_t* d_ids = nullptr;
    int32_t* h_ids = nullptr;
    PL_CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint32_t)));
    PL_CUDA_CHECK(cudaMemset(d_sink, 0, sizeof(uint32_t)));
    PL_CUDA_CHECK(cudaHostAlloc(&h_timeout, sizeof(int), cudaHostAllocMapped | cudaHostAllocPortable));
    PL_CUDA_CHECK(cudaHostGetDevicePointer(&d_timeout, h_timeout, 0));
    *h_timeout = 0;
    PL_CUDA_CHECK(cudaMalloc(&d_ids, 64));
    PL_CUDA_CHECK(cudaHostAlloc(&h_ids, 64, cudaHostAllocPortable));
    uint16_t* d_out = nullptr;
    PL_CUDA_CHECK(cudaMalloc(&d_out, 4096));

    cudaStream_t st;
    PL_CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    const double cpu_timeout_us = 2e6;
    std::vector<double> t_empty, t_after, t_spin, t_copy;
    int spin_timeouts = 0, cpu_timeouts = 0;
    int32_t step = 0;

    auto reset_mb = [&]() { h_mb->gpu_step = 0; h_mb->cpu_step = 0; step = 0; };

    for (int warm = 0; warm < 50; ++warm) {             // warm-up: load modules, touch pages, settle clocks
        PL_LAUNCH(k_empty, 1, 1, 0, st);
        PL_CUDA_CHECK(cudaStreamSynchronize(st));
    }

    // ---- empty launch + sync
    for (int i = 0; i < iters; ++i) {
        const double t0 = now_us();
        PL_LAUNCH(k_empty, 1, 1, 0, st);
        PL_CUDA_CHECK(cudaStreamSynchronize(st));
        t_empty.push_back(now_us() - t0);
    }

    // ---- launch_after
    reset_mb();
    for (int i = 0; i < iters + 50; ++i) {
        ++step;
        const double t0 = now_us();
        PL_LAUNCH(k_gpu_publish, 1, 32, 0, st, d_mb, step);
        if (!wait_flag(&h_mb->gpu_step, step, cpu_timeout_us)) { ++cpu_timeouts; break; }
        std::memset(h_mb->expert_out, step & 0xFF, sizeof h_mb->expert_out);
        std::atomic_thread_fence(std::memory_order_release);
        PL_LAUNCH(k_gpu_consume, 1, 128, 0, st, d_mb, d_sink);
        PL_CUDA_CHECK(cudaStreamSynchronize(st));
        if (i >= 50) t_after.push_back(now_us() - t0);
    }

    // ---- spin_wait (bounded)
    reset_mb();
    for (int i = 0; i < iters + 50; ++i) {
        ++step;
        *h_timeout = 0;
        const double t0 = now_us();
        PL_LAUNCH(k_gpu_publish, 1, 32, 0, st, d_mb, step);
        PL_LAUNCH(k_gpu_wait_and_consume, 1, 128, 0, st, d_mb, step, d_sink, d_timeout);   // queued before the CPU is done
        if (!wait_flag(&h_mb->gpu_step, step, cpu_timeout_us)) { ++cpu_timeouts; PL_CUDA_CHECK(cudaStreamSynchronize(st)); break; }
        std::memset(h_mb->expert_out, step & 0xFF, sizeof h_mb->expert_out);
        std::atomic_thread_fence(std::memory_order_release);
        h_mb->cpu_step = step;
        PL_CUDA_CHECK(cudaStreamSynchronize(st));
        if (*h_timeout) ++spin_timeouts;
        if (i >= 50) t_spin.push_back(now_us() - t0);
    }

    // ---- copy_roundtrip
    for (int i = 0; i < iters + 50; ++i) {
        const int32_t s = int32_t(i + 1);
        const double t0 = now_us();
        PL_LAUNCH(k_gpu_write_device_ids, 1, 32, 0, st, d_ids, s);
        PL_CUDA_CHECK(cudaMemcpyAsync(h_ids, d_ids, 64, cudaMemcpyDeviceToHost, st));
        PL_CUDA_CHECK(cudaStreamSynchronize(st));
        std::memset(h_mb->expert_out, s & 0xFF, sizeof h_mb->expert_out);
        PL_CUDA_CHECK(cudaMemcpyAsync(d_out, h_mb->expert_out, 4096, cudaMemcpyHostToDevice, st));
        PL_LAUNCH(k_gpu_consume, 1, 128, 0, st, d_mb, d_sink);
        PL_CUDA_CHECK(cudaStreamSynchronize(st));
        if (i >= 50) t_copy.push_back(now_us() - t0);
    }

    const Stats e = summarize(t_empty), a = summarize(t_after), sp = summarize(t_spin), cp = summarize(t_copy);
    std::printf("[2c] empty launch+sync   median %7.2f us  p99 %7.2f us\n", e.median, e.p99);
    std::printf("[2c] launch_after        median %7.2f us  p99 %7.2f us   <- use this (per CPU-path layer)\n", a.median, a.p99);
    std::printf("[2c] spin_wait           median %7.2f us  p99 %7.2f us   (GPU spin timeouts: %d)\n", sp.median, sp.p99, spin_timeouts);
    std::printf("[2c] copy_roundtrip      median %7.2f us  p99 %7.2f us\n", cp.median, cp.p99);
    std::printf("[2c] per-token cost with %d CPU-path layers (launch_after): %.3f ms median, %.3f ms p99\n", cpu_layers,
                a.median * cpu_layers / 1000.0, a.p99 * cpu_layers / 1000.0);
    if (cpu_timeouts) std::printf("[2c] WARNING: %d CPU-side waits timed out; treat the numbers as invalid\n", cpu_timeouts);

    std::ostringstream js;
    js << "{\n  \"iterations\": " << iters << ",\n  \"cpu_layers_assumed\": " << cpu_layers << ",\n"
       << "  \"handoff_us_per_layer\": " << a.median << ",\n  \"handoff_ms_per_layer\": " << a.median / 1000.0 << ",\n"
       << "  \"handoff_ms_per_token_40_layers\": " << a.median * 40.0 / 1000.0 << ",\n"
       << "  \"handoff_ms_per_token_cpu_layers\": " << a.median * cpu_layers / 1000.0 << ",\n"
       << "  \"cpu_wait_timeouts\": " << cpu_timeouts << ",\n  \"modes\": {\n"
       << stats_json("empty_launch", e) << ",\n" << stats_json("launch_after", a) << ",\n"
       << stats_json("spin_wait", sp, (std::string(", \"gpu_spin_timeouts\": ") + std::to_string(spin_timeouts)).c_str()) << ",\n"
       << stats_json("copy_roundtrip", cp) << "\n  }\n}\n";
    std::ofstream out(out_path);
    if (out) { out << js.str(); std::printf("[2c] wrote %s\n", out_path.c_str()); }
    else std::fprintf(stderr, "cannot write %s; JSON follows\n%s", out_path.c_str(), js.str().c_str());

    cudaStreamDestroy(st);
    cudaFree(d_sink); cudaFree(d_ids); cudaFree(d_out);
    cudaFreeHost(h_ids); cudaFreeHost(h_timeout); cudaFreeHost(h_mb);
    return cpu_timeouts ? 5 : 0;
}
