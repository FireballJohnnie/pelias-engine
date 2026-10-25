// File: <project>\moe_track\tests\test_handoff.cu
// The first run of pelias_serve/handoff.cuh on a real GPU. The scenarios are the ones test_handoff_host.cpp already passes with CPU threads standing in for the
// GPU (handoff_scenarios.h), now with one resident block on the device and the CPU worker on a host thread talking through the mapped, pinned mailbox:
//   * payload integrity both ways (8 KB down, 8 KB up, 0..16 tasks), accumulator checked bit for bit, many block sizes
//   * sequence-number wrap, a CPU error status, a late CPU (the GPU gives up, the stale answer is ignored), a dead CPU, no CPU at all
//   * timing: the CPU-side round trip (our post -> GPU combine -> GPU publish -> we see it), the GPU-side time per round, and a block-size sweep
// microbench 03 measured 13.3 us median / 18.7 us p99 for a flag + 4 KB payload read with 2-byte loads. This file moves 8 KB each way with 16-byte batched
// loads. It prints this file's own measured median and p99 next to the microbench 03 reference and the per-token cost they imply; no pass/fail limit is set on timing
// (a limit would be a number nobody measured). Timing never changes the exit code; a correctness failure does.
//
// Build (the toolchain from the earlier runs):
//   R=$HOME/stuart_creator_engine/cuda_shim
//   $R/bin/nvcc -O3 -std=c++17 -arch=sm_89 -I$R/include --cudart shared -L$R/lib64 -Xlinker -rpath,$R/lib64 -Xcompiler -pthread test_handoff.cu -I../pelias_serve -o test_handoff
// Run:    ./test_handoff [--rounds 1500] [--timing-rounds 4000]        Exit 0 = all correctness checks passed.
// Syntax check without a GPU:  g++ -std=c++17 -x c++ -fsyntax-only -DPELIAS_CUDA_SHIM -I../pelias_serve test_handoff.cu   (the shim cannot run this kernel)
// standard headers first: the CPU shim (syntax check only) defines __noinline__ as a macro, which breaks libstdc++ headers included after it
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "../lib/pl_cuda.h"

#include "handoff.cuh"
#include "handoff_scenarios.h"

using namespace pelias;
using namespace pelias::handoff;

__global__ void k_resident(hs::KArgs a) { hs::resident_body(a); }

__global__ void k_clock_cal(long long* out, long long cycles) {
    if (threadIdx.x == 0) {
        const long long t0 = clock64();
        while (clock64() - t0 < cycles) {
        }
        *out = clock64() - t0;
    }
}

static cudaStream_t g_stream = nullptr;

namespace hs {
bool hs_launch_resident(const KArgs& a, int block, std::string* err) {
    PL_LAUNCH(k_resident, 1, block, 0, g_stream, a);
    flush_stream(g_stream);                      // the 03 rev 2 fix: a freshly launched kernel can sit in the queue until the driver is called
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { if (err) *err = std::string("launch: ") + cudaGetErrorString(e); return false; }
    return true;
}
bool hs_wait_resident(std::string* err) {
    const cudaError_t e = cudaStreamSynchronize(g_stream);
    if (e != cudaSuccess) { if (err) *err = std::string("kernel: ") + cudaGetErrorString(e); return false; }
    return true;
}
}  // namespace hs

static int g_fail = 0;
static void check(const std::string& name, bool ok) { std::printf("%s %s\n", ok ? "ok  " : "FAIL", name.c_str()); std::fflush(stdout); if (!ok) ++g_fail; }

static std::string describe(const hs::Analysis& a, const hs::RunOut& o) {
    char b[420];
    std::snprintf(b, sizeof b, " [ran %d, ok %d, fail %d (first %d, last %d), acc mismatches %d, status wrong %d, unserved-but-ok %d, worker: served %d skipped %d payload errors %d (torn-abandoned %d) timeouts %d%s%s]",
                  a.rounds_run, a.n_ok, a.n_fail, a.first_fail, a.last_fail, a.acc_mismatch, a.status_wrong, a.served_mismatch, o.w.served, o.w.skipped, o.w.payload_errors, o.w.torn_abandoned,
                  o.w.timeouts, o.error.empty() ? "" : " error: ", o.error.c_str());
    return b;
}

// clock64 ticks per microsecond, measured against the wall clock (no dependence on the deprecated clock-rate attributes)
static double calibrate_ticks_per_us() {
    long long* d = nullptr;
    PL_CUDA_CHECK(cudaMalloc(&d, sizeof(long long)));
    PL_LAUNCH(k_clock_cal, 1, 32, 0, g_stream, d, 1000000LL);       // warm-up: context, module load
    PL_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    const long long cycles = 150000000LL;
    const double t0 = hs::now_us();
    PL_LAUNCH(k_clock_cal, 1, 32, 0, g_stream, d, cycles);
    PL_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    const double us = hs::now_us() - t0;
    long long ticks = 0;
    PL_CUDA_CHECK(cudaMemcpy(&ticks, d, sizeof ticks, cudaMemcpyDeviceToHost));
    cudaFree(d);
    return static_cast<double>(ticks) / us;
}

int main(int argc, char** argv) {
    int rounds = 1500, timing_rounds = 4000;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--rounds") == 0) rounds = std::atoi(argv[i + 1]);
        if (std::strcmp(argv[i], "--timing-rounds") == 0) timing_rounds = std::atoi(argv[i + 1]);
    }
    if (rounds < 100) rounds = 100;
    if (timing_rounds < 400) timing_rounds = 400;

    (void)cudaSetDeviceFlags(cudaDeviceMapHost);
    (void)cudaGetLastError();
    cudaDeviceProp prop;
    PL_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s   (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    PL_CUDA_CHECK(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));

    MailboxHandle mbh;
    PL_CUDA_CHECK(alloc_mailbox(&mbh));
    check("mailbox: mapped pinned memory, device pointer equals host pointer (unified addressing)", mbh.host != nullptr && mbh.host == mbh.dev);

    const double tpu = calibrate_ticks_per_us();
    std::printf("clock64: %.1f ticks per microsecond (%.2f GHz)\n", tpu, tpu / 1000.0);
    check("clock64 rate is plausible (0.5 - 4 GHz)", tpu > 500.0 && tpu < 4000.0);

    auto clean_run = [&](hs::Scenario s, int expect_rounds) {
        const hs::RunOut o = hs::run_once(s, mbh, tpu);
        const hs::Analysis a = hs::analyze(o, s.rounds);
        const bool ok = o.error.empty() && a.n_ok == expect_rounds && a.n_fail == 0 && a.acc_mismatch == 0 && a.status_wrong == 0 && a.served_mismatch == 0 &&
                        o.w.payload_errors == 0 && o.w.served == expect_rounds && o.w.skipped == 0 && o.w.timeouts == 0;
        check(std::string(s.name) + describe(a, o), ok);
    };

    // ---- correctness ---------------------------------------------------------------------------------------------------------------------------------------
    {
        hs::Scenario s;
        s.name = "block 128: every payload field and x element arrives intact, accumulator exact (task counts 0..16)";
        s.rounds = rounds;
        s.block = 128;
        clean_run(s, rounds);
    }
    for (int blk : {1, 8, 33, 256, 512, 1024}) {
        hs::Scenario s;
        char nm[160];
        std::snprintf(nm, sizeof nm, "block %d: clean", blk);
        s.name = nm;
        s.rounds = 300;
        s.block = blk;
        clean_run(s, 300);
    }
    {
        hs::Scenario s;
        s.name = "sequence numbers wrap through 0xFFFFFFFF -> 0";
        s.rounds = 200;
        s.seq0 = 0xFFFFFFF0u;
        clean_run(s, 200);
    }
    {
        hs::Scenario s;
        s.name = "sequence numbers cross 0x7FFFFFFF -> 0x80000000";
        s.rounds = 200;
        s.seq0 = 0x7FFFFFF0u;
        clean_run(s, 200);
    }
    {
        hs::Scenario s;
        s.name = "CPU reports an error on round 20";
        s.rounds = 200;
        s.wo.error_round = 20;
        s.max_consec_fail = 5;
        const hs::RunOut o = hs::run_once(s, mbh, tpu);
        const hs::Analysis a = hs::analyze(o, s.rounds);
        check(std::string("error round: exactly round 20 is not combined, status = CPU error, nothing else disturbed") + describe(a, o),
              o.error.empty() && a.n_fail == 1 && a.first_fail == 20 && a.n_status_cpu_error == 1 && a.n_ok == s.rounds - 1 && a.acc_mismatch == 0 && a.status_wrong == 0 && o.w.payload_errors == 0);
    }
    {
        hs::Scenario s;
        s.name = "late CPU";
        s.rounds = 300;
        s.spin_limit_us = 3000.0;
        s.max_consec_fail = 400;
        s.wo.delay_round = 100;
        s.wo.delay_us = 30000.0;
        const hs::RunOut o = hs::run_once(s, mbh, tpu);
        const hs::Analysis a = hs::analyze(o, s.rounds);
        int lo = 100, hi = 100;
        while (lo > 1 && o.ok[static_cast<size_t>(lo - 2)] == 0) --lo;
        while (hi < s.rounds && o.ok[static_cast<size_t>(hi)] == 0) ++hi;
        const bool r100_failed = o.ok[99] == 0;
        const int outside = a.n_fail - (r100_failed ? hi - lo + 1 : 0);
        check(std::string("late CPU (30 ms sleep, GPU gives up after 3 ms): round 100 times out with a window of timeouts after it; the stale answer is never used") + describe(a, o),
              o.error.empty() && r100_failed && hi > 100 && hi < s.rounds && outside <= 3 && a.n_status_cpu_error == 0 && a.acc_mismatch == 0 && a.status_wrong == 0 &&
                  a.served_mismatch == 0 && o.w.payload_errors == 0);
        check("late CPU: the worker noticed that the GPU had moved on (skipped abandoned requests, resynced to the latest)", o.w.skipped >= 1);
    }
    {
        hs::Scenario s;
        s.name = "CPU dies after 3 rounds";
        s.rounds = 10;
        s.spin_limit_us = 10000.0;
        s.max_consec_fail = 3;
        s.wo.stop_after_round = 3;
        const double t0 = hs::now_us();
        const hs::RunOut o = hs::run_once(s, mbh, tpu);
        const double ms = (hs::now_us() - t0) / 1000.0;
        const hs::Analysis a = hs::analyze(o, s.rounds);
        check(std::string("dead CPU: rounds 1-3 combine, rounds 4-6 time out, the kernel then stops by itself (7-10 never run)") + describe(a, o),
              o.error.empty() && a.n_ok == 3 && a.n_fail == 3 && a.first_fail == 4 && a.last_fail == 6 && a.rounds_run == 6 && a.n_status_timeout == 3 && a.acc_mismatch == 0 && ms < 1500.0);
    }
    {
        hs::Scenario s;
        s.name = "no CPU at all";
        s.rounds = 5;
        s.worker = false;
        s.spin_limit_us = 10000.0;
        s.max_consec_fail = 3;
        const hs::RunOut o = hs::run_once(s, mbh, tpu);
        const hs::Analysis a = hs::analyze(o, s.rounds);
        check(std::string("no CPU: three timeouts (about 10 ms each), then the kernel ends; the accumulator is untouched") + describe(a, o),
              o.error.empty() && a.n_ok == 0 && a.n_fail == 3 && a.n_status_timeout == 3 && a.acc_mismatch == 0);
        double mean_us = 0;
        for (int r = 0; r < 3; ++r) mean_us += static_cast<double>(o.ticks[static_cast<size_t>(r)]) / tpu / 3.0;
        std::printf("     a timed-out round took %.0f us on the GPU (limit 10000 us)\n", mean_us);
        check("the spin limit is honoured to within a factor of 2 (5 - 20 ms)", mean_us > 5000.0 && mean_us < 20000.0);
    }

    // ---- timing --------------------------------------------------------------------------------------------------------------------------------------------
    std::printf("\ntiming (%d rounds each, first 200 discarded; CPU worker does the minimum: copy the request, 2048 multiply-adds, post)\n", timing_rounds);
    std::printf("  CPU-side round trip = our post -> GPU sees it, adds 8 KB into the accumulator, writes the next 8 KB + tasks, rings the bell -> we see it\n");
    std::printf("  GPU-side round     = clock64 from before the x write to after the combine\n");
    double best_median = 1e30, med128 = 0, p99_128 = 0;
    int best_block = 0;
    for (int blk : {32, 64, 128, 256, 512, 1024}) {
        hs::Scenario s;
        s.name = "timing";
        s.rounds = timing_rounds;
        s.block = blk;
        s.wo.verify = false;
        const hs::RunOut o = hs::run_once(s, mbh, tpu);
        const hs::Analysis a = hs::analyze(o, s.rounds);
        std::vector<double> gap, gpu;
        for (size_t i = 200; i < o.w.gap_us.size(); ++i) gap.push_back(o.w.gap_us[i]);
        for (size_t i = 200; i < o.ticks.size(); ++i)
            if (o.ok[i] == 1) gpu.push_back(static_cast<double>(o.ticks[i]) / tpu);
        const hs::Pctl pc = hs::percentiles(gap), pg = hs::percentiles(gpu);
        std::printf("  block %4d   CPU-side: median %6.2f us  p90 %6.2f  p99 %6.2f  max %8.2f   GPU-side: median %6.2f  p99 %6.2f   (%s, acc mismatches %d)\n", blk, pc.median, pc.p90, pc.p99, pc.max,
                    pg.median, pg.p99, o.error.empty() && a.n_ok == s.rounds ? "all rounds combined" : "NOT ALL ROUNDS COMBINED", a.acc_mismatch);
        check("timing run " + std::to_string(blk) + ": every round combined, accumulator exact", o.error.empty() && a.n_ok == s.rounds && a.acc_mismatch == 0 && o.w.skipped == 0);
        if (pc.median < best_median) { best_median = pc.median; best_block = blk; }
        if (blk == 128) { med128 = pc.median; p99_128 = pc.p99; }
    }
    std::printf("\nbest block size for the handoff: %d (median %.2f us)\n", best_block, best_median);
    std::printf("block 128: median %.2f us, p99 %.2f us   (CPU-side round trip, 8 KB each way, measured here)\n", med128, p99_128);
    std::printf("derived, if all %d layers hand off once per token: %d x median = %.3f ms per token; %d x p99 = %.3f ms (an upper bound for the sum, not the token's own p99)\n", N_LAYERS, N_LAYERS,
                N_LAYERS * med128 / 1000.0, N_LAYERS, N_LAYERS * p99_128 / 1000.0);
    std::printf("(speed model: each us per layer costs %.2f ms per token; what that does to tok/s still depends on the unmeasured spine and hit rate)\n", N_LAYERS / 1000.0);
    std::printf("microbench 03 reference: 13.34 us median / 18.71 us p99 (flag + 4 KB, 2-byte loads)\n");

    free_mailbox(&mbh);
    std::printf("%s\n", g_fail == 0 ? "ALL PASS (correctness)" : "FAILED");
    return g_fail == 0 ? 0 : 1;
}
