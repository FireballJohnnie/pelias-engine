// File: <project>\moe_track\microbench\04_dual_bus_test.cu
// Step 2d: do the CPU and the GPU slow each other down when both read host RAM at the same time?
//
// Both the CPU expert path and a GPU miss path read the same DDR4 memory controller. This test measures
//   cpu_alone      N CPU threads streaming pinned host memory
//   gpu_zc_alone   a GPU kernel reading the same kind of memory directly over PCIe (zero-copy)
//   h2d_alone      cudaMemcpyAsync host -> device (what a cache-miss fetch does in llama.cpp's expert cache)
//   concurrent     CPU threads + zero-copy kernel + H2D copies all at once, for a fixed time
// Every side runs for a fixed duration and counts the bytes it really read, so the numbers do not depend on who finishes first.
//
// Build (microbench/build.sh):  nvcc -O3 -std=c++17 -arch=sm_89 -Xcompiler "-fopenmp -mavx2" 04_dual_bus_test.cu -o bin/04_dual_bus_test
// Run with the model server STOPPED:
//   bin/04_dual_bus_test [--cpu-threads 7] [--seconds 3] [--pool-mib 2048] [--out <project>/logs/floors/dual_bus_contention.json] [--force]
#include <immintrin.h>
#include <omp.h>

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

__global__ void k_zero_copy_read(const uint4* __restrict__ p, size_t n_uint4, unsigned long long* out) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    unsigned long long acc = 0;
    for (size_t i = idx; i < n_uint4; i += stride) {
        const uint4 v = p[i];
        acc += (unsigned long long)(v.x ^ v.y) + (unsigned long long)(v.z ^ v.w);
    }
    if (acc == 0x7FFFFFFFFFFFFFFFULL) *out = acc;   // keeps the loads alive
}

inline uint64_t fold(__m256i v) {
    alignas(32) uint64_t t[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(t), v);
    return t[0] ^ t[1] ^ t[2] ^ t[3];
}

// CPU readers: loop over [base, base+bytes) until `stop`; returns bytes read and a checksum
double cpu_readers(const uint8_t* base, size_t bytes, int threads, double seconds, std::atomic<bool>* stop, uint64_t* sink) {
    omp_set_num_threads(threads);
    std::atomic<uint64_t> total{0};
    uint64_t chk = 0;
    const auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel reduction(^ : chk)
    {
        const int id = omp_get_thread_num(), nt = omp_get_num_threads();
        const size_t per = (bytes / size_t(nt)) & ~size_t(127);
        const uint8_t* p = base + per * size_t(id);
        uint64_t mine = 0;
        while (true) {
            __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
            for (size_t b = 0; b + 128 <= per; b += 128) {
                a0 = _mm256_add_epi64(a0, _mm256_load_si256(reinterpret_cast<const __m256i*>(p + b)));
                a1 = _mm256_add_epi64(a1, _mm256_load_si256(reinterpret_cast<const __m256i*>(p + b + 32)));
                a2 = _mm256_add_epi64(a2, _mm256_load_si256(reinterpret_cast<const __m256i*>(p + b + 64)));
                a3 = _mm256_add_epi64(a3, _mm256_load_si256(reinterpret_cast<const __m256i*>(p + b + 96)));
            }
            chk ^= fold(_mm256_add_epi64(_mm256_add_epi64(a0, a1), _mm256_add_epi64(a2, a3)));
            mine += per;
            const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if ((stop && stop->load(std::memory_order_relaxed)) || (!stop && el >= seconds)) break;
        }
        total.fetch_add(mine);
    }
    const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    *sink ^= chk;
    return (double(total.load()) / 1e9) / el;
}

struct GpuRates { double zc_gbs = 0, h2d_gbs = 0; };

// GPU side on its own host thread: zero-copy kernel launches and/or H2D copies until `stop`
void gpu_loop(const uint8_t* d_zc, size_t zc_bytes, const uint8_t* h_copy_src, uint8_t* d_copy_dst, size_t copy_bytes,
              bool do_zc, bool do_h2d, std::atomic<bool>* stop, double seconds, GpuRates* rates, unsigned long long* d_out) {
    cudaStream_t s_zc, s_cp;
    PL_CUDA_CHECK(cudaStreamCreateWithFlags(&s_zc, cudaStreamNonBlocking));
    PL_CUDA_CHECK(cudaStreamCreateWithFlags(&s_cp, cudaStreamNonBlocking));
    const size_t n_uint4 = zc_bytes / sizeof(uint4);
    double zc_total = 0, cp_total = 0;
    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    while (true) {
        if (do_zc) { PL_LAUNCH(k_zero_copy_read, 256, 256, 0, s_zc, reinterpret_cast<const uint4*>(d_zc), n_uint4, d_out); }
        if (do_h2d) PL_CUDA_CHECK(cudaMemcpyAsync(d_copy_dst, h_copy_src, copy_bytes, cudaMemcpyHostToDevice, s_cp));
        if (do_zc) { PL_CUDA_CHECK(cudaStreamSynchronize(s_zc)); zc_total += double(zc_bytes); }
        if (do_h2d) { PL_CUDA_CHECK(cudaStreamSynchronize(s_cp)); cp_total += double(copy_bytes); }
        const double el = elapsed();
        if ((stop && stop->load(std::memory_order_relaxed)) || (!stop && el >= seconds)) break;
    }
    const double el = elapsed();
    rates->zc_gbs = zc_total / 1e9 / el;
    rates->h2d_gbs = cp_total / 1e9 / el;
    cudaStreamDestroy(s_zc);
    cudaStreamDestroy(s_cp);
}

bool server_running() {
    FILE* p = popen("pgrep -x llama-server >/dev/null 2>&1; echo $?", "r");
    if (!p) return false;
    char buf[16] = {0};
    const bool ok = fgets(buf, sizeof buf, p) != nullptr;
    pclose(p);
    return ok && std::atoi(buf) == 0;
}

}  // namespace

int main(int argc, char** argv) {
    int cpu_threads = 7;
    double seconds = 3.0;
    size_t pool_mib = 2048;
    bool force = false;
    std::string out_path = "<project>/logs/floors/dual_bus_contention.json";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--cpu-threads" && i + 1 < argc) cpu_threads = std::atoi(argv[++i]);
        else if (a == "--seconds" && i + 1 < argc) seconds = std::atof(argv[++i]);
        else if (a == "--pool-mib" && i + 1 < argc) pool_mib = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (a == "--force") force = true;
        else { std::fprintf(stderr, "usage: %s [--cpu-threads N] [--seconds S] [--pool-mib M] [--out FILE] [--force]\n", argv[0]); return 2; }
    }
    if (!force && server_running()) {
        std::fprintf(stderr, "llama-server is running; stop it first (llama --stop) or pass --force.\n");
        return 3;
    }

    cudaSetDeviceFlags(cudaDeviceMapHost);
    const size_t third = (pool_mib * 1024ull * 1024ull / 3) & ~size_t(4095);   // CPU region, zero-copy region, copy-source region
    uint8_t* h_pool = nullptr;
    uint8_t* d_pool = nullptr;
    PL_CUDA_CHECK(cudaHostAlloc(&h_pool, third * 3, cudaHostAllocMapped | cudaHostAllocPortable));
    PL_CUDA_CHECK(cudaHostGetDevicePointer(&d_pool, h_pool, 0));
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < third * 3; i += 4096) h_pool[i] = uint8_t((i >> 12) * 131u);

    const uint8_t* cpu_region = h_pool;
    const uint8_t* zc_region_dev = d_pool + third;
    const uint8_t* copy_src = h_pool + 2 * third;
    uint8_t* d_copy_dst = nullptr;
    unsigned long long* d_out = nullptr;
    PL_CUDA_CHECK(cudaMalloc(&d_copy_dst, third));
    PL_CUDA_CHECK(cudaMalloc(&d_out, sizeof(unsigned long long)));
    uint64_t sink = 0;

    // warm-up (page faults, module load, clocks)
    { std::atomic<bool> st{false}; GpuRates r; std::thread t(gpu_loop, zc_region_dev, third, copy_src, d_copy_dst, third, true, true, nullptr, 0.3, &r, d_out); t.join(); (void)st; }
    cpu_readers(cpu_region, third, cpu_threads, 0.3, nullptr, &sink);

    // 1. CPU alone
    const double cpu_alone = cpu_readers(cpu_region, third, cpu_threads, seconds, nullptr, &sink);
    // 2. GPU zero-copy alone
    GpuRates zc_alone;
    { std::thread t(gpu_loop, zc_region_dev, third, copy_src, d_copy_dst, third, true, false, nullptr, seconds, &zc_alone, d_out); t.join(); }
    // 3. H2D copy alone
    GpuRates h2d_alone;
    { std::thread t(gpu_loop, zc_region_dev, third, copy_src, d_copy_dst, third, false, true, nullptr, seconds, &h2d_alone, d_out); t.join(); }
    // 4. everything concurrently, fixed duration: GPU thread runs until the CPU side finishes
    std::atomic<bool> stop{false};
    GpuRates conc;
    std::thread gt(gpu_loop, zc_region_dev, third, copy_src, d_copy_dst, third, true, true, &stop, 0.0, &conc, d_out);
    const double cpu_conc = cpu_readers(cpu_region, third, cpu_threads, seconds, nullptr, &sink);
    stop.store(true);
    gt.join();
    const double combined = cpu_conc + conc.zc_gbs + conc.h2d_gbs;
    // 5. CPU + H2D copies only (the realistic miss path: no zero-copy kernel)
    std::atomic<bool> stop2{false};
    GpuRates conc2;
    std::thread gt2(gpu_loop, zc_region_dev, third, copy_src, d_copy_dst, third, false, true, &stop2, 0.0, &conc2, d_out);
    const double cpu_conc2 = cpu_readers(cpu_region, third, cpu_threads, seconds, nullptr, &sink);
    stop2.store(true);
    gt2.join();

    const double ratio_all = combined / cpu_alone;
    const double ratio_h2d = (cpu_conc2 + conc2.h2d_gbs) / cpu_alone;
    std::printf("[2d] CPU alone (%d threads)          %6.2f GB/s\n", cpu_threads, cpu_alone);
    std::printf("[2d] GPU zero-copy alone             %6.2f GB/s\n", zc_alone.zc_gbs);
    std::printf("[2d] H2D copy alone                  %6.2f GB/s\n", h2d_alone.h2d_gbs);
    std::printf("[2d] all three at once               CPU %6.2f + zero-copy %6.2f + H2D %6.2f = %6.2f GB/s (%.2fx CPU alone)\n",
                cpu_conc, conc.zc_gbs, conc.h2d_gbs, combined, ratio_all);
    std::printf("[2d] CPU + H2D copies only           CPU %6.2f + H2D %6.2f = %6.2f GB/s (%.2fx CPU alone; CPU keeps %.0f%%)\n",
                cpu_conc2, conc2.h2d_gbs, cpu_conc2 + conc2.h2d_gbs, ratio_h2d, 100.0 * cpu_conc2 / cpu_alone);

    std::ostringstream js;
    js << "{\n"
       << "  \"cpu_threads\": " << cpu_threads << ",\n  \"seconds_each\": " << seconds << ",\n"
       << "  \"cpu_alone_gb_s\": " << cpu_alone << ",\n"
       << "  \"gpu_zero_copy_alone_gb_s\": " << zc_alone.zc_gbs << ",\n"
       << "  \"h2d_copy_alone_gb_s\": " << h2d_alone.h2d_gbs << ",\n"
       << "  \"concurrent_all\": {\"cpu_gb_s\": " << cpu_conc << ", \"zero_copy_gb_s\": " << conc.zc_gbs << ", \"h2d_gb_s\": " << conc.h2d_gbs
       << ", \"combined_gb_s\": " << combined << ", \"combined_vs_cpu_alone\": " << ratio_all << "},\n"
       << "  \"concurrent_cpu_plus_h2d\": {\"cpu_gb_s\": " << cpu_conc2 << ", \"h2d_gb_s\": " << conc2.h2d_gbs
       << ", \"combined_gb_s\": " << cpu_conc2 + conc2.h2d_gbs << ", \"combined_vs_cpu_alone\": " << ratio_h2d
       << ", \"cpu_keeps_fraction\": " << cpu_conc2 / cpu_alone << "},\n"
       << "  \"combined_concurrent_gb_s\": " << combined << ",\n  \"combined_vs_cpu_ratio\": " << ratio_all << ",\n"
       << "  \"phase_7a_greenlight\": " << ((zc_alone.zc_gbs >= 15.0 && ratio_all >= 1.25) ? "true" : "false") << ",\n"
       << "  \"checksum\": " << sink << "\n}\n";
    std::ofstream out(out_path);
    if (out) { out << js.str(); std::printf("[2d] wrote %s\n", out_path.c_str()); }
    else std::fprintf(stderr, "cannot write %s; JSON follows\n%s", out_path.c_str(), js.str().c_str());

    cudaFree(d_copy_dst);
    cudaFree(d_out);
    cudaFreeHost(h_pool);
    return 0;
}
