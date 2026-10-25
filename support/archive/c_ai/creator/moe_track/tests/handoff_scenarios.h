// File: <project>\moe_track\tests\handoff_scenarios.h
// The scenarios for pelias_serve/handoff.cuh, written once and used twice:
//   test_handoff_host.cpp  CPU threads stand in for the GPU block (one std::thread per CUDA thread, std::barrier for __syncthreads) -- checks the logic anywhere
//   test_handoff.cu        the same scenarios on the real GPU, plus timing                                                           -- the first real run
// The includer must already have included handoff.cuh (and, for the host build, the shim with its macros), and must define hs_launch_resident / hs_wait_resident.
//
// WHAT ONE RUN DOES. A resident kernel (ONE block) loops over `rounds`: it writes a layer input x, a task list of 0..16 entries, rings the bell with publish(),
// then wait_and_combine()s the CPU's answer into an accumulator. A CPU worker thread plays the engine's single in-order worker: wait_gpu -> read_request ->
// check the payload -> compute partial[i] = 2*x[i] + round -> post. All values are small integers and scale = 0.5, so the accumulator is exact in float and
// the host can recompute it bit for bit from the per-round ok flags; a stale, torn or misdirected partial changes the sum and is caught.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace hs {
using namespace pelias::handoff;

// ---- the data pattern (host + device) --------------------------------------------------------------------------------------------------------------------
__host__ __device__ inline float x_val(int r, int i) { return static_cast<float>((r * 31 + i) % 1021); }
__host__ __device__ inline float partial_val(float x, int r) { return 2.0f * x + static_cast<float>(r); }
__host__ __device__ inline int tasks_for_round(int r) { return (r - 1) % (kMaxTasks + 1); }          // 0, 1, ... 16, 0, ...
__host__ __device__ inline CpuTask task_for(int r, int k) {
    CpuTask t;
    t.expert = (r * 7 + k * 13) % 256;
    t.weight = 0.0625f * static_cast<float>(k + 1);
    t.row_begin = 256 * (k & 1);
    t.row_end = 512;
    return t;
}
__host__ __device__ inline float acc_init(int i) { return static_cast<float>(i % 7); }

// ---- the resident kernel's arguments and body --------------------------------------------------------------------------------------------------------------
struct KArgs {
    Mailbox*   mb;               // device-side pointer to the mailbox
    float*     x;                // device, kHidden floats (16-byte aligned)
    float*     acc;              // device, kHidden floats (16-byte aligned)
    int32_t*   ok;               // device [rounds]: 1 added, 0 not added, -1 round never ran
    uint32_t*  status;           // device [rounds]: written by wait_and_combine on failure only
    long long* ticks;            // device [rounds]: clock64 ticks from before the x write to after the combine
    uint32_t   seq0;             // sequence number of round 1
    int        rounds;
    long long  spin_limit;       // wait_and_combine's bound, in clock64 ticks
    int        max_consec_fail;  // the kernel stops after this many failed rounds in a row (bounds its run time if the CPU is gone)
    float      scale;
};

__device__ __forceinline__ void resident_body(const KArgs& a) {
    __shared__ CpuTask s_tasks[kMaxTasks];
    const int tid = static_cast<int>(threadIdx.x);
    const int bd = static_cast<int>(blockDim.x);
    int consec = 0;
    for (int r = 1; r <= a.rounds; ++r) {
        const long long t0 = clock64();
        for (int i = tid; i < kHidden; i += bd) a.x[i] = x_val(r, i);
        const int nt = tasks_for_round(r);
        for (int k = tid; k < nt; k += bd) s_tasks[k] = task_for(r, k);
        __syncthreads();
        const uint32_t seq = a.seq0 + static_cast<uint32_t>(r - 1);
        publish(a.mb, seq, static_cast<uint32_t>(r % 40), static_cast<uint32_t>(1000 + r), nt, s_tasks, a.x);
        const bool got = wait_and_combine(a.mb, seq, a.acc, a.scale, a.spin_limit, &a.status[r - 1]);
        if (tid == 0) {
            a.ok[r - 1] = got ? 1 : 0;
            a.ticks[r - 1] = clock64() - t0;
        }
        consec = got ? 0 : consec + 1;
        if (consec >= a.max_consec_fail) break;
    }
}

// supplied by the includer: start the kernel (asynchronously) / wait for it. false + *err on a launch or run-time error.
bool hs_launch_resident(const KArgs& a, int block, std::string* err);
bool hs_wait_resident(std::string* err);

// ---- the CPU worker ---------------------------------------------------------------------------------------------------------------------------------------------
inline double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct WorkerOpts {
    bool     verify = true;          // check every field of every request (off in the timing run)
    int      error_round = 0;        // answer this round with kStatusCpuError (and no partial)
    int      delay_round = 0;        // sleep delay_us after reading this round's request, before answering
    double   delay_us = 0.0;
    double   wait_timeout_us = 5.0e6;
    int      stop_after_round = 0;   // >0: stop answering after this many rounds (the CPU "dies")
};

struct WorkerResult {
    int                 served = 0;
    int                 skipped = 0;          // requests the GPU had already abandoned when the worker got to them
    int                 payload_errors = 0;       // a request that was still current but did not carry what the GPU sent
    int                 torn_abandoned = 0;       // a copy that raced with the GPU abandoning that request (harmless, see Worker::run)
    int                 timeouts = 0;
    std::vector<int>    served_rounds;
    std::vector<double> gap_us;               // our post -> next request seen (a whole round trip)
    std::string         first_error;
};

class Worker {
public:
    Worker(Mailbox* host_mb, uint32_t seq0, int rounds, const WorkerOpts& o) : db_(host_mb), seq0_(seq0), rounds_(rounds), o_(o) {}
    void start() { th_ = std::thread([this] { run(); }); }
    void join() { if (th_.joinable()) th_.join(); }
    const WorkerResult& result() const { return r_; }

private:
    void bad(int round, const char* what) {
        ++r_.payload_errors;
        if (r_.first_error.empty()) { char b[160]; std::snprintf(b, sizeof b, "round %d: %s", round, what); r_.first_error = b; }
    }
    void run() {
        uint32_t want = seq0_;
        while (static_cast<int32_t>(want - seq0_) < rounds_) {
            if (o_.stop_after_round > 0 && static_cast<int32_t>(want - seq0_) >= o_.stop_after_round) return;
            if (!db_.wait_gpu(want, o_.wait_timeout_us)) { ++r_.timeouts; return; }
            const double t_seen = now_us();
            Request rq;
            if (!db_.read_request(want, &rq)) { ++r_.skipped; want = db_.gpu_seq(); continue; }
            const int r = static_cast<int32_t>(rq.seq - seq0_) + 1;
            if (o_.verify) {
                const char* what = nullptr;
                if (rq.layer != static_cast<uint32_t>(r % 40)) what = "layer";
                else if (rq.token_pos != static_cast<uint32_t>(1000 + r)) what = "token_pos";
                else if (rq.n_tasks != static_cast<uint32_t>(tasks_for_round(r))) what = "n_tasks";
                else {
                    for (uint32_t k = 0; k < rq.n_tasks && what == nullptr; ++k) {
                        const CpuTask e = task_for(r, static_cast<int>(k));
                        if (rq.tasks[k].expert != e.expert || rq.tasks[k].weight != e.weight || rq.tasks[k].row_begin != e.row_begin || rq.tasks[k].row_end != e.row_end) what = "task";
                    }
                    for (int i = 0; i < kHidden && what == nullptr; ++i)
                        if (rq.x[i] != x_val(r, i)) what = "x";
                }
                if (what != nullptr) {
                    // A GPU that gave up on this request overwrites the payload for the next one BEFORE it rings the next bell, so a copy that raced with that
                    // can be torn without read_request noticing. That is harmless (the GPU ignores the answer to an abandoned request) and is not an error.
                    if (static_cast<int32_t>(db_.gpu_seq() - rq.seq) != 0) ++r_.torn_abandoned;
                    else bad(r, what);
                }
            }
            if (r == o_.delay_round) std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int64_t>(o_.delay_us)));
            alignas(16) float partial[kHidden];
            for (int i = 0; i < kHidden; ++i) partial[i] = partial_val(rq.x[i], r);
            if (r == o_.error_round) db_.post(rq.seq, nullptr, kStatusCpuError);
            else db_.post(rq.seq, partial, kStatusOk);
            const double t_post = now_us();
            ++r_.served;
            r_.served_rounds.push_back(r);
            if (prev_post_ > 0.0) r_.gap_us.push_back(t_seen - prev_post_);
            prev_post_ = t_post;
            want = rq.seq + 1;
        }
    }
    HostDoorbell  db_;
    uint32_t      seq0_;
    int           rounds_;
    WorkerOpts    o_;
    WorkerResult  r_;
    double        prev_post_ = 0.0;
    std::thread   th_;
};

// ---- one run ------------------------------------------------------------------------------------------------------------------------------------------------------
struct Scenario {
    const char* name = "";
    int         rounds = 200;
    int         block = 128;
    uint32_t    seq0 = 1;
    bool        worker = true;
    double      spin_limit_us = 200000.0;
    int         max_consec_fail = 3;
    WorkerOpts  wo;
};

struct RunOut {
    std::vector<int32_t>  ok;
    std::vector<uint32_t> status;
    std::vector<long long> ticks;
    std::vector<float>    acc;
    WorkerResult          w;
    std::string           error;          // launch / run-time error, empty if none
};

// ticks_per_us converts the scenario's microsecond spin limit to clock64 ticks.
inline RunOut run_once(const Scenario& s, MailboxHandle& mbh, double ticks_per_us) {
    RunOut out;
    const int n = s.rounds;
    float *d_x = nullptr, *d_acc = nullptr;
    int32_t* d_ok = nullptr;
    uint32_t* d_status = nullptr;
    long long* d_ticks = nullptr;
    auto chk = [&](cudaError_t e, const char* what) { if (e != cudaSuccess && out.error.empty()) out.error = std::string(what) + ": " + cudaGetErrorString(e); };
    chk(cudaMalloc(&d_x, sizeof(float) * kHidden), "cudaMalloc x");
    chk(cudaMalloc(&d_acc, sizeof(float) * kHidden), "cudaMalloc acc");
    chk(cudaMalloc(&d_ok, sizeof(int32_t) * static_cast<size_t>(n)), "cudaMalloc ok");
    chk(cudaMalloc(&d_status, sizeof(uint32_t) * static_cast<size_t>(n)), "cudaMalloc status");
    chk(cudaMalloc(&d_ticks, sizeof(long long) * static_cast<size_t>(n)), "cudaMalloc ticks");
    if (!out.error.empty()) return out;

    std::vector<float> acc0(kHidden);
    for (int i = 0; i < kHidden; ++i) acc0[static_cast<size_t>(i)] = acc_init(i);
    chk(cudaMemcpy(d_acc, acc0.data(), sizeof(float) * kHidden, cudaMemcpyHostToDevice), "copy acc");
    chk(cudaMemset(d_ok, 0xFF, sizeof(int32_t) * static_cast<size_t>(n)), "memset ok");
    chk(cudaMemset(d_status, 0, sizeof(uint32_t) * static_cast<size_t>(n)), "memset status");
    chk(cudaMemset(d_ticks, 0, sizeof(long long) * static_cast<size_t>(n)), "memset ticks");

    HostDoorbell db(mbh.host);
    db.reset(s.seq0 - 1u);

    KArgs a;
    a.mb = mbh.dev;
    a.x = d_x;
    a.acc = d_acc;
    a.ok = d_ok;
    a.status = d_status;
    a.ticks = d_ticks;
    a.seq0 = s.seq0;
    a.rounds = n;
    a.spin_limit = static_cast<long long>(s.spin_limit_us * ticks_per_us);
    a.max_consec_fail = s.max_consec_fail;
    a.scale = 0.5f;

    Worker w(mbh.host, s.seq0, n, s.wo);
    if (s.worker) w.start();
    std::string err;
    if (!hs_launch_resident(a, s.block, &err)) out.error = err;
    else {
        if (!hs_wait_resident(&err)) out.error = err;
    }
    if (s.worker) w.join();
    out.w = w.result();

    out.ok.resize(static_cast<size_t>(n));
    out.status.resize(static_cast<size_t>(n));
    out.ticks.resize(static_cast<size_t>(n));
    out.acc.resize(kHidden);
    chk(cudaMemcpy(out.ok.data(), d_ok, sizeof(int32_t) * static_cast<size_t>(n), cudaMemcpyDeviceToHost), "copy ok");
    chk(cudaMemcpy(out.status.data(), d_status, sizeof(uint32_t) * static_cast<size_t>(n), cudaMemcpyDeviceToHost), "copy status");
    chk(cudaMemcpy(out.ticks.data(), d_ticks, sizeof(long long) * static_cast<size_t>(n), cudaMemcpyDeviceToHost), "copy ticks");
    chk(cudaMemcpy(out.acc.data(), d_acc, sizeof(float) * kHidden, cudaMemcpyDeviceToHost), "copy acc back");
    cudaFree(d_x); cudaFree(d_acc); cudaFree(d_ok); cudaFree(d_status); cudaFree(d_ticks);
    return out;
}

// ---- what a run must satisfy --------------------------------------------------------------------------------------------------------------------------------------
struct Analysis {
    int  rounds_run = 0;          // rounds with ok != -1
    int  n_ok = 0;
    int  n_fail = 0;
    int  first_fail = 0;          // 1-based, 0 if none
    int  last_fail = 0;
    bool fails_contiguous = true;
    int  acc_mismatch = 0;        // accumulator elements that differ from the exact recomputation
    int  status_wrong = 0;        // ok rounds with a status, or failed rounds with a code that is neither timeout nor cpu error
    int  served_mismatch = 0;     // rounds the GPU says it combined that the worker never answered
    int  n_status_timeout = 0;
    int  n_status_cpu_error = 0;
};

inline Analysis analyze(const RunOut& o, int rounds) {
    Analysis a;
    std::vector<float> want(kHidden);
    for (int i = 0; i < kHidden; ++i) want[static_cast<size_t>(i)] = acc_init(i);
    std::vector<char> served(static_cast<size_t>(rounds) + 2, 0);
    for (int r : o.w.served_rounds)
        if (r >= 1 && r <= rounds) served[static_cast<size_t>(r)] = 1;
    bool in_fail = false, ended_fail = false;
    for (int r = 1; r <= rounds; ++r) {
        const int32_t ok = o.ok[static_cast<size_t>(r - 1)];
        if (ok == -1) continue;
        ++a.rounds_run;
        if (ok == 1) {
            ++a.n_ok;
            if (o.status[static_cast<size_t>(r - 1)] != 0) ++a.status_wrong;
            if (!served[static_cast<size_t>(r)]) ++a.served_mismatch;
            for (int i = 0; i < kHidden; ++i) want[static_cast<size_t>(i)] += 0.5f * partial_val(x_val(r, i), r);
            if (in_fail) ended_fail = true;
        } else {
            ++a.n_fail;
            if (a.first_fail == 0) a.first_fail = r;
            if (ended_fail) a.fails_contiguous = false;
            a.last_fail = r;
            in_fail = true;
            const uint32_t st = o.status[static_cast<size_t>(r - 1)];
            if (st == kStatusGpuTimeout) ++a.n_status_timeout;
            else if (st == kStatusCpuError) ++a.n_status_cpu_error;
            else ++a.status_wrong;
        }
    }
    for (int i = 0; i < kHidden; ++i)
        if (o.acc[static_cast<size_t>(i)] != want[static_cast<size_t>(i)]) ++a.acc_mismatch;
    return a;
}

struct Pctl { double median = 0, p90 = 0, p99 = 0, max = 0; size_t n = 0; };
inline Pctl percentiles(std::vector<double> v) {
    Pctl p;
    p.n = v.size();
    if (v.empty()) return p;
    std::sort(v.begin(), v.end());
    auto at = [&](double q) { return v[std::min(v.size() - 1, static_cast<size_t>(q * static_cast<double>(v.size())))]; };
    p.median = at(0.5);
    p.p90 = at(0.9);
    p.p99 = at(0.99);
    p.max = v.back();
    return p;
}

}  // namespace hs
