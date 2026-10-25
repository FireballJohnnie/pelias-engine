// File: <project>\moe_track\tests\test_handoff_host.cpp
// Runs pelias_serve/handoff.cuh's real device functions (publish, wait_and_combine) on the host: one std::thread per CUDA thread of the block, a std::barrier in
// place of __syncthreads(), a thread_local threadIdx, and the CPU worker side (HostDoorbell) on its own thread. Scenarios are in handoff_scenarios.h (shared with
// test_handoff.cu, which runs them on the GPU). What this proves is the LOGIC: payload integrity, any block size (also fewer threads than tasks), sequence
// wrap-around, CPU error status, a late CPU (timeout + stale post), a dead CPU, no CPU at all. What it cannot prove: GPU memory ordering over PCIe or timing.
//
// Build:  g++ -O2 -std=c++20 -pthread -Wall -Wextra -Werror -DPELIAS_CUDA_SHIM test_handoff_host.cpp -I../pelias_serve -o test_handoff_host
// Run:    ./test_handoff_host        Exit 0 = pass.
#include "../lib/cuda_shim.h"

#include <barrier>
#include <memory>

// per-thread threadIdx and a real barrier for __syncthreads (defined AFTER the shim, BEFORE the kernel code)
static thread_local dim3 tl_threadIdx{0, 0, 0};
static std::barrier<>* g_barrier = nullptr;
#define threadIdx tl_threadIdx
#define __syncthreads() g_barrier->arrive_and_wait()
#include "handoff.cuh"
#include "handoff_scenarios.h"
#undef threadIdx
#undef __syncthreads

using namespace pelias;
using namespace pelias::handoff;

namespace hs {
static std::vector<std::thread> g_threads;
static std::unique_ptr<std::barrier<>> g_bar;
static KArgs g_args;

bool hs_launch_resident(const KArgs& a, int block, std::string*) {
    g_args = a;
    g_bar = std::make_unique<std::barrier<>>(block);
    g_barrier = g_bar.get();
    blockDim = dim3(static_cast<unsigned>(block), 1u, 1u);
    for (int t = 0; t < block; ++t)
        g_threads.emplace_back([t] {
            tl_threadIdx = dim3(static_cast<unsigned>(t), 0u, 0u);
            resident_body(g_args);
        });
    return true;
}
bool hs_wait_resident(std::string*) {
    for (auto& t : g_threads) t.join();
    g_threads.clear();
    g_barrier = nullptr;
    g_bar.reset();
    return true;
}
}  // namespace hs

static int g_fail = 0;
static void check(const std::string& name, bool ok) { std::printf("%s %s\n", ok ? "ok  " : "FAIL", name.c_str()); if (!ok) ++g_fail; }

static std::string describe(const hs::Analysis& a, const hs::RunOut& o) {
    char b[400];
    std::snprintf(b, sizeof b, " [ran %d, ok %d, fail %d (first %d, last %d), acc mismatches %d, status wrong %d, unserved-but-ok %d, worker: served %d skipped %d payload errors %d (torn-abandoned %d) timeouts %d%s%s]",
                  a.rounds_run, a.n_ok, a.n_fail, a.first_fail, a.last_fail, a.acc_mismatch, a.status_wrong, a.served_mismatch, o.w.served, o.w.skipped, o.w.payload_errors, o.w.torn_abandoned, o.w.timeouts,
                  o.error.empty() ? "" : " error: ", o.error.c_str());
    return b;
}

int main() {
    // ---- small pieces ------------------------------------------------------------------------------------------------------------------------------------
    {
        MailboxHandle mbh;
        check("alloc_mailbox: host and device pointers are the same address (UVA)", alloc_mailbox(&mbh) == cudaSuccess && mbh.host != nullptr && mbh.host == mbh.dev);
        check("mailbox is 128-byte aligned and the doorbell lines are separate",
              reinterpret_cast<uintptr_t>(mbh.host) % kLine == 0 && offsetof(Mailbox, cpu_seq) - offsetof(Mailbox, gpu_seq) >= static_cast<size_t>(kLine));

        HostDoorbell db(mbh.host);
        db.reset();
        const auto t0 = std::chrono::steady_clock::now();
        const bool got = db.wait_gpu(1, 3000.0);
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        check("wait_gpu with no doorbell gives up after about its timeout", !got && us >= 3000.0 && us < 500000.0);

        db.reset(0xFFFFFFFEu);
        mbh.host->gpu_seq = 0xFFFFFFFFu;
        check("wait_gpu: wrapped sequence compare (0xFFFFFFFF seen, wants 0xFFFFFFFF)", db.wait_gpu(0xFFFFFFFFu, 1000.0));
        mbh.host->gpu_seq = 1u;                                   // wrapped past zero
        check("wait_gpu: gpu_seq 1 counts as later than 0xFFFFFFFF", db.wait_gpu(0xFFFFFFFFu, 1000.0));
        check("wait_gpu: ... but not yet as 2", !db.wait_gpu(2u, 500.0));

        db.reset();
        mbh.host->gpu_seq = 7;
        mbh.host->n_tasks = 3;
        Request rq;
        check("read_request: the GPU has moved past seq 6 -> false (drop the work)", !db.read_request(6, &rq));
        check("read_request: seq 7 is current -> true, task count carried", db.read_request(7, &rq) && rq.n_tasks == 3);
        mbh.host->n_tasks = 999;
        check("read_request: a corrupt task count is clamped to kMaxTasks", db.read_request(7, &rq) && rq.n_tasks == static_cast<uint32_t>(kMaxTasks));

        alignas(16) float p[kHidden];
        for (int i = 0; i < kHidden; ++i) p[i] = static_cast<float>(i);
        db.post(7, p, kStatusCpuError);
        check("post: partial copied, status stored, doorbell last", mbh.host->cpu_seq == 7 && mbh.host->cpu_status == kStatusCpuError && mbh.host->partial[2047] == 2047.0f);
        db.post(8, nullptr, kStatusOk);
        check("post with no partial leaves the old partial alone", mbh.host->cpu_seq == 8 && mbh.host->partial[5] == 5.0f);
        free_mailbox(&mbh);
    }
    {
        SplitPlan plan;
        plan.n_miss = 3;
        plan.n_ff = 512;
        plan.granule = 256;
        plan.gpu_rows = plan.cpu_rows = 0;
        plan.task[0] = SplitTask{10, 0.5f, 512};     // all GPU: no CPU task
        plan.task[1] = SplitTask{20, 0.25f, 256};    // half
        plan.task[2] = SplitTask{30, 0.125f, 0};     // all CPU
        CpuTask t[kMaxTasks];
        const int n = fill_cpu_tasks(plan, t);
        check("fill_cpu_tasks: experts with no CPU rows are left out, row ranges are [split, n_ff)",
              n == 2 && t[0].expert == 20 && t[0].row_begin == 256 && t[0].row_end == 512 && t[1].expert == 30 && t[1].row_begin == 0 && t[1].row_end == 512 && t[1].weight == 0.125f);
    }

    // ---- scenarios -----------------------------------------------------------------------------------------------------------------------------------------
    MailboxHandle mbh;
    if (alloc_mailbox(&mbh) != cudaSuccess) { std::printf("FAIL cannot allocate the mailbox\n"); return 1; }
    const double tpu = 1000.0;           // the shim's clock64 counts nanoseconds

    auto clean_run = [&](hs::Scenario s, int expect_rounds) {
        const hs::RunOut o = hs::run_once(s, mbh, tpu);
        const hs::Analysis a = hs::analyze(o, s.rounds);
        const bool ok = o.error.empty() && a.n_ok == expect_rounds && a.n_fail == 0 && a.acc_mismatch == 0 && a.status_wrong == 0 && a.served_mismatch == 0 &&
                        o.w.payload_errors == 0 && o.w.served == expect_rounds && o.w.skipped == 0 && o.w.timeouts == 0;
        check(std::string(s.name) + describe(a, o), ok);
    };

    {
        hs::Scenario s;
        s.name = "block 128, 300 rounds: every payload field and x element arrives intact, accumulator exact (task counts 0..16)";
        s.rounds = 300;
        s.block = 128;
        s.wo.verify = true;
        clean_run(s, 300);
    }
    for (int blk : {8, 1, 33, 256}) {
        hs::Scenario s;
        char nm[160];
        std::snprintf(nm, sizeof nm, "block %d (fewer threads than tasks / odd size), 60 rounds: clean", blk);
        s.name = nm;
        s.rounds = 60;
        s.block = blk;
        clean_run(s, 60);
    }
    {
        hs::Scenario s;
        s.name = "sequence numbers wrap through 0xFFFFFFFF -> 0 during the run";
        s.rounds = 64;
        s.seq0 = 0xFFFFFFF0u;
        clean_run(s, 64);
    }
    {
        hs::Scenario s;
        s.name = "sequence numbers cross 0x7FFFFFFF -> 0x80000000 (signed-compare boundary)";
        s.rounds = 64;
        s.seq0 = 0x7FFFFFF0u;
        clean_run(s, 64);
    }
    {
        hs::Scenario s;
        s.name = "CPU reports an error on round 20";
        s.rounds = 60;
        s.wo.error_round = 20;
        s.max_consec_fail = 5;
        const hs::RunOut o = hs::run_once(s, mbh, tpu);
        const hs::Analysis a = hs::analyze(o, s.rounds);
        check(std::string("error round: exactly round 20 is not combined, status = CPU error, nothing else disturbed") + describe(a, o),
              o.error.empty() && a.n_fail == 1 && a.first_fail == 20 && a.n_status_cpu_error == 1 && a.n_ok == 59 && a.acc_mismatch == 0 && a.status_wrong == 0 && o.w.payload_errors == 0);
    }
    for (int blk : {4, 64}) {
        hs::Scenario s;
        s.name = "late CPU";
        s.rounds = 60;
        s.block = blk;                        // 4 emulated threads: little scheduler noise on a small machine; 64: a wide block
        s.spin_limit_us = blk == 4 ? 4000.0 : 8000.0;
        s.max_consec_fail = 400;
        s.wo.delay_round = 30;
        s.wo.delay_us = 40000.0;              // 40 ms: the GPU gives up on round 30 and on several more
        const hs::RunOut o = hs::run_once(s, mbh, tpu);
        const hs::Analysis a = hs::analyze(o, s.rounds);
        // the failing run that contains round 30, and any failures outside it (host scheduling noise on a small machine; tolerated, but they must be timeouts)
        int lo = 30, hi = 30;
        while (lo > 1 && o.ok[static_cast<size_t>(lo - 2)] == 0) --lo;
        while (hi < s.rounds && o.ok[static_cast<size_t>(hi)] == 0) ++hi;
        const bool r30_failed = o.ok[29] == 0;
        const int outside = a.n_fail - (r30_failed ? hi - lo + 1 : 0);
        check(std::string("late CPU, block ") + std::to_string(blk) + " (40 ms sleep, GPU gives up after " + std::to_string(static_cast<int>(s.spin_limit_us / 1000)) + " ms): round 30 times out with a window of timeouts after it; the stale answer is never used" + describe(a, o),
              o.error.empty() && r30_failed && hi > 30 && hi < s.rounds && outside <= 3 && a.n_status_cpu_error == 0 && a.acc_mismatch == 0 && a.status_wrong == 0 &&
                  a.served_mismatch == 0 && o.w.payload_errors == 0);
        check("late CPU: the worker noticed that the GPU had moved on (skipped abandoned requests, resynced to the latest)", o.w.skipped >= 1);
    }
    {
        hs::Scenario s;
        s.name = "CPU dies after 3 rounds";
        s.rounds = 10;
        s.block = 4;
        s.spin_limit_us = 20000.0;
        s.max_consec_fail = 3;
        s.wo.stop_after_round = 3;
        const auto t0 = std::chrono::steady_clock::now();
        const hs::RunOut o = hs::run_once(s, mbh, tpu);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const hs::Analysis a = hs::analyze(o, s.rounds);
        check(std::string("dead CPU: rounds 1-3 combine, rounds 4-6 time out, the kernel then stops by itself (7-10 never run)") + describe(a, o),
              o.error.empty() && a.n_ok == 3 && a.n_fail == 3 && a.first_fail == 4 && a.last_fail == 6 && a.rounds_run == 6 && a.n_status_timeout == 3 && a.acc_mismatch == 0 && ms < 5000.0);
    }
    {
        hs::Scenario s;
        s.name = "no CPU at all";
        s.rounds = 5;
        s.worker = false;
        s.spin_limit_us = 1000.0;
        s.max_consec_fail = 3;
        const hs::RunOut o = hs::run_once(s, mbh, tpu);
        const hs::Analysis a = hs::analyze(o, s.rounds);
        check(std::string("no CPU: three timeouts, then the kernel ends; the accumulator is untouched") + describe(a, o),
              o.error.empty() && a.n_ok == 0 && a.n_fail == 3 && a.n_status_timeout == 3 && a.acc_mismatch == 0);
    }
    {   // a spin limit of 0 must not hang and must not add anything even if the answer is already there
        hs::Scenario s;
        s.name = "spin limit 0 with a CPU that answers instantly";
        s.rounds = 6;
        s.spin_limit_us = 0.0;
        s.max_consec_fail = 6;
        const hs::RunOut o = hs::run_once(s, mbh, tpu);
        const hs::Analysis a = hs::analyze(o, s.rounds);
        check(std::string("spin limit 0: terminates, every combined round is exact, every refused round carries a status") + describe(a, o),
              o.error.empty() && a.acc_mismatch == 0 && a.status_wrong == 0 && a.n_ok + a.n_fail == a.rounds_run);
    }

    // ---- the round trip through the emulated block (timing is NOT meaningful here: 128 host threads and a barrier) -----------------------------------------------
    {
        hs::Scenario s;
        s.name = "emulated timing";
        s.rounds = 150;
        s.block = 32;
        s.wo.verify = false;
        const hs::RunOut o = hs::run_once(s, mbh, tpu);
        std::vector<double> g;
        for (size_t i = 10; i < o.w.gap_us.size(); ++i) g.push_back(o.w.gap_us[i]);
        const hs::Pctl p = hs::percentiles(g);
        std::printf("     emulated round trip (32 host threads + barrier, NOT hardware): median %.1f us  p99 %.1f us over %zu rounds\n", p.median, p.p99, p.n);
    }

    free_mailbox(&mbh);
    std::printf("%s\n", g_fail == 0 ? "ALL PASS" : "FAILED");
    return g_fail == 0 ? 0 : 1;
}
