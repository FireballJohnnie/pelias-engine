// File: <project>\moe_track\pelias_serve\e6_hot_aware_mtp.cuh
// Stage E6: the draft-time gate of the hot-aware MTP drafter.
//
// After blk.40 (the MTP block) has produced its logits for the next token, ONE thread block decides in a single launch whether the 2-token verification batch is
// worth running: it finds the draft token (arg-max of the logits), its probability (softmax of the winner) and how many of blk.40's own selected experts are not in the GPU
// hot bank, and accepts the draft only if probability >= min_draft_prob and cold misses <= max_cold_misses.
//
// Differences from the roadmap draft:
//   * the draft took PROBABILITIES and scanned the 248k-entry vocabulary with ONE thread (several milliseconds); this takes the LOGITS, does a block-wide arg-max, a block-wide
//     sum of exp(l - max) and returns top1_prob = 1 / sum, all in a few tens of microseconds;
//   * ties go to the lowest token id (what numpy.argmax and a sequential scan give);
//   * a degenerate vocabulary (empty, or every logit -inf / NaN) rejects the draft instead of returning garbage.
// The rollback that follows a REJECTED draft is e5_deltanet_state.cuh; its cost model (rollback + one re-decode of the accepted token) is in e6_policy_model.py.
//
// The kernel is written so that every thread reaches every __syncthreads(); tests/test_e6_gate.cpp runs the same source with one real thread per CUDA thread and a
// barrier standing in for __syncthreads(). It has not been compiled with nvcc or run on a GPU.
//
// Launch:  pelias_mtp_hot_gate_kernel<256><<<1, 256, 0, stream>>>(d_logits, vocab, d_topk_ids, d_slot_map, min_p, max_cold, d_decision);
#pragma once

#include "pelias_defs.h"

#include <cmath>
#include <cstdint>

#if defined(PELIAS_CUDA_SHIM)
#include "../lib/cuda_shim.h"
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

namespace pelias {

struct MtpDraftDecision {
    int32_t draft_token_id;
    float   draft_top1_prob;
    int32_t blk40_cold_misses;
    int32_t accept_for_verify_batch;       // 1 = run the 2-token verification batch, 0 = ordinary 1-token decode
};

// the rule itself, shared by the kernel and the host code
inline bool mtp_accept_rule(float top1_prob, int cold_misses, float min_draft_prob, int max_cold_misses) {
    return top1_prob >= min_draft_prob && cold_misses <= max_cold_misses;
}

struct ArgMaxItem { float v; int32_t i; };

// larger value wins; equal values -> smaller index; NaN never wins
__host__ __device__ inline ArgMaxItem argmax_combine(ArgMaxItem a, ArgMaxItem b) {
    const bool b_wins = (b.v > a.v) || (b.v == a.v && b.i < a.i) || (a.v != a.v && b.v == b.v);
    return b_wins ? b : a;
}

template <int BLOCK>
__global__ void pelias_mtp_hot_gate_kernel(const float * __restrict__ logits, int vocab,
                                           const int32_t * __restrict__ topk_ids,       // blk.40's selected experts, TOP_K entries
                                           const int16_t * __restrict__ slot_map,       // hot-bank slot per expert id (N_EXPERTS entries), < 0 = not in the bank
                                           float min_draft_prob, int max_cold_misses,
                                           MtpDraftDecision * __restrict__ out) {
    __shared__ ArgMaxItem s_best[BLOCK];
    __shared__ float      s_sum[BLOCK];
    const int t = threadIdx.x;

    ArgMaxItem mine{-INFINITY, 0x7fffffff};
    for (int v = t; v < vocab; v += BLOCK) {
        const float x = logits[v];
        mine = argmax_combine(mine, ArgMaxItem{x, v});
    }
    s_best[t] = mine;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (t < s) s_best[t] = argmax_combine(s_best[t], s_best[t + s]);
        __syncthreads();
    }
    const ArgMaxItem best = s_best[0];                       // every thread reads the same value; s_best is not written again
    const bool usable = vocab > 0 && best.v > -INFINITY && best.v == best.v && best.i != 0x7fffffff;

    float acc = 0.0f;
    if (usable) {
        for (int v = t; v < vocab; v += BLOCK) {
            const float x = logits[v];
            if (x == x) acc += expf(x - best.v);             // skips NaN; -inf contributes 0
        }
    }
    s_sum[t] = acc;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (t < s) s_sum[t] += s_sum[t + s];
        __syncthreads();
    }

    if (t == 0) {
        int misses = 0;
        for (int k = 0; k < TOP_K; ++k) {
            const int e = topk_ids[k];
            if (e < 0 || e >= N_EXPERTS || slot_map[e] < 0) ++misses;      // an invalid id counts as a miss
        }
        MtpDraftDecision d;
        d.blk40_cold_misses = misses;
        if (usable && s_sum[0] > 0.0f) {
            d.draft_token_id  = best.i;
            d.draft_top1_prob = 1.0f / s_sum[0];
            d.accept_for_verify_batch = mtp_accept_rule(d.draft_top1_prob, misses, min_draft_prob, max_cold_misses) ? 1 : 0;
        } else {
            d.draft_token_id = -1;
            d.draft_top1_prob = 0.0f;
            d.accept_for_verify_batch = 0;
        }
        *out = d;
    }
}

}  // namespace pelias
