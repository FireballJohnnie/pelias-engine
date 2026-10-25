// File: <project>\moe_track\tests\test_e6_gate.cpp
// Runs pelias_serve/e6_hot_aware_mtp.cuh's kernel on the host: one std::thread per CUDA thread of the block, a std::barrier in place of __syncthreads(), a
// thread_local threadIdx. Checks the arg-max (ties, NaN, -inf), the softmax top-1 probability against a double-precision reference, the cold-miss count and the
// accept rule, for block sizes 32 and 256 and for vocabularies smaller than, equal to and much larger than the block.
//
// Build:  g++ -O2 -std=c++20 -pthread -DPELIAS_CUDA_SHIM test_e6_gate.cpp -I../pelias_serve -o test_e6_gate        Run: ./test_e6_gate        Exit 0 = pass.
#include "../lib/cuda_shim.h"

#include <barrier>
#include <memory>
#include <random>
#include <thread>
#include <vector>
#include <cstdio>

// make the shim's global threadIdx per-thread and __syncthreads a real barrier (defined AFTER the shim, BEFORE the kernel header)
static thread_local dim3 tl_threadIdx{0, 0, 0};
static std::barrier<> * g_barrier = nullptr;
#define threadIdx tl_threadIdx
#define __syncthreads() g_barrier->arrive_and_wait()
#include "e6_hot_aware_mtp.cuh"
#undef threadIdx
#undef __syncthreads

using namespace pelias;

static int g_fail = 0;
static void check(const char * name, bool ok) { std::printf("%s %s\n", ok ? "ok  " : "FAIL", name); if (!ok) ++g_fail; }

template <int BLOCK>
static MtpDraftDecision run_block(const std::vector<float> & logits, const std::vector<int32_t> & ids, const std::vector<int16_t> & slot_map, float min_p, int max_cold) {
    MtpDraftDecision out{-7, -7.0f, -7, -7};
    std::barrier<> bar(BLOCK);
    g_barrier = &bar;
    std::vector<std::thread> th;
    for (int t = 0; t < BLOCK; ++t)
        th.emplace_back([&, t] {
            tl_threadIdx = dim3(t, 0, 0);
            pelias_mtp_hot_gate_kernel<BLOCK>(logits.data(), static_cast<int>(logits.size()), ids.data(), slot_map.data(), min_p, max_cold, &out);
        });
    for (auto & x : th) x.join();
    g_barrier = nullptr;
    return out;
}

static void reference(const std::vector<float> & l, int & tok, double & p) {
    tok = 0; double best = -1e300;
    for (size_t i = 0; i < l.size(); ++i) if (l[i] > best) { best = l[i]; tok = static_cast<int>(i); }
    double s = 0; for (float x : l) s += std::exp(double(x) - best);
    p = 1.0 / s;
}

template <int BLOCK>
static void suite(const char * tag) {
    std::mt19937 rng(BLOCK);
    std::normal_distribution<float> nd(0.0f, 2.5f);
    std::vector<int16_t> slot_map(N_EXPERTS, -1);
    for (int e = 0; e < N_EXPERTS; e += 3) slot_map[e] = static_cast<int16_t>(e / 3 + 1);       // experts 0,3,6,... are hot
    std::vector<int32_t> ids = {0, 3, 6, 9, 1, 2, 12, 15};                                           // 6 hot, 2 cold
    for (int vocab : {1, 7, BLOCK, BLOCK + 1, 1000, 20011}) {
        std::vector<float> l(vocab);
        for (auto & v : l) v = nd(rng);
        int rt; double rp;
        reference(l, rt, rp);
        const auto d = run_block<BLOCK>(l, ids, slot_map, 0.0f, 8);
        char buf[160];
        std::snprintf(buf, sizeof buf, "%s vocab %d: token %d (ref %d), p %.6f (ref %.6f)", tag, vocab, d.draft_token_id, rt, d.draft_top1_prob, rp);
        check(buf, d.draft_token_id == rt && std::fabs(d.draft_top1_prob - rp) <= 1e-5 * std::max(1.0, rp) + 1e-7 && d.blk40_cold_misses == 2 && d.accept_for_verify_batch == 1);
    }
    {   // tie goes to the lowest id, across threads
        std::vector<float> l(5000, 0.0f);
        l[4000] = 5.0f; l[17] = 5.0f; l[3999] = 5.0f;
        const auto d = run_block<BLOCK>(l, ids, slot_map, 0.0f, 8);
        check((std::string(tag) + " ties: lowest token id wins").c_str(), d.draft_token_id == 17);
    }
    {   // NaN entries and -inf entries are ignored
        std::vector<float> l(3000, -INFINITY);
        l[1234] = 2.0f; l[2000] = 1.0f; l[5] = NAN; l[2999] = NAN;
        const auto d = run_block<BLOCK>(l, ids, slot_map, 0.0f, 8);
        const double p = 1.0 / (1.0 + std::exp(-1.0));
        check((std::string(tag) + " NaN and -inf entries are ignored").c_str(), d.draft_token_id == 1234 && std::fabs(d.draft_top1_prob - p) < 1e-6);
    }
    {   // degenerate input is rejected, not guessed
        std::vector<float> allneg(100, -INFINITY), empty;
        auto d1 = run_block<BLOCK>(allneg, ids, slot_map, 0.0f, 8);
        auto d2 = run_block<BLOCK>(empty, ids, slot_map, 0.0f, 8);
        check((std::string(tag) + " all -inf: no draft").c_str(), d1.accept_for_verify_batch == 0 && d1.draft_token_id == -1);
        check((std::string(tag) + " empty vocabulary: no draft").c_str(), d2.accept_for_verify_batch == 0 && d2.draft_token_id == -1);
    }
    {   // thresholds
        std::vector<float> l(500, 0.0f);
        l[42] = 4.0f;                                              // p = e^4 / (e^4 + 499) ~ 0.0987
        const auto lo = run_block<BLOCK>(l, ids, slot_map, 0.05f, 8);
        const auto hi = run_block<BLOCK>(l, ids, slot_map, 0.60f, 8);
        check((std::string(tag) + " confidence below min_draft_prob: not accepted").c_str(), lo.accept_for_verify_batch == 1 && hi.accept_for_verify_batch == 0);
        std::vector<float> sharp(500, 0.0f);
        sharp[9] = 12.0f;
        const auto okc = run_block<BLOCK>(sharp, ids, slot_map, 0.60f, 2);
        const auto bad = run_block<BLOCK>(sharp, ids, slot_map, 0.60f, 1);
        check((std::string(tag) + " 2 cold misses allowed at max 2, refused at max 1").c_str(), okc.accept_for_verify_batch == 1 && bad.accept_for_verify_batch == 0 && okc.blk40_cold_misses == 2);
        std::vector<int32_t> weird = {-1, 999, 0, 3, 6, 9, 12, 15};
        const auto w = run_block<BLOCK>(sharp, weird, slot_map, 0.0f, 8);
        check((std::string(tag) + " invalid expert ids count as misses").c_str(), w.blk40_cold_misses == 2);
    }
}

int main() {
    suite<32>("block 32 ");
    suite<256>("block 256");
    check("accept rule: boundary values accept", mtp_accept_rule(0.60f, 2, 0.60f, 2));
    check("accept rule: just below the probability refuses", !mtp_accept_rule(0.5999f, 0, 0.60f, 2));
    check("argmax_combine: NaN never beats a number", argmax_combine({1.0f, 5}, {NAN, 2}).i == 5 && argmax_combine({NAN, 2}, {1.0f, 5}).i == 5);
    std::printf(g_fail ? "\n%d FAILED\n" : "\nALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
