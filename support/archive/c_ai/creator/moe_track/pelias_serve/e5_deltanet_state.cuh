// File: <project>\moe_track\pelias_serve\e5_deltanet_state.cuh
// Stage E5: snapshot and rollback of the recurrent state of the 30 Gated DeltaNet layers.
//
// Why: in Qwen3.6-35B-A3B only 10 of the 40 layers keep a normal K/V cache. The other 30 keep a fixed-size state that is overwritten in place by every token, so
// "forget the last token" cannot be done by moving a position counter back. If a drafted (MTP) token is rejected, the state must be put back by copying.
//
// What this manager does: it owns two slots of device-side copies of those states (slot 0 = before a speculative verification batch, slot 1 = a checkpoint at a
// turn/prefix boundary), copies live -> slot on snapshot() and slot -> live on rollback(). That is all.
//
// What it does NOT do (the roadmap draft implied more):
//   * It restores the state as it was BEFORE the snapshot. For a verification batch [accepted_token, draft_token] that means that after a rejected draft the
//     accepted token has to be decoded once more (one normal 1-token step) to bring the state to "after the accepted token". The rollback saves re-reading the
//     prompt, it does not make the accepted token free. Model that cost in the E6 policy (e6_policy_model.py does).
//   * The 10 full-attention layers need no copy: rolling them back is "set n_past to the snapshot position"; their K/V rows beyond it are simply overwritten.
//   * It does not know where llama.cpp keeps the state. The pointers given to register_layer() must be the sequence's slice of each layer's recurrent tensors
//     (llama_memory_recurrent r_l / s_l, one cell of them); finding those is the job of the server adapter (E7, not written).
//
// Cost (the draft's "0.095 ms" assumed 48 MiB at the full 504 GB/s with a single transfer): for this model one slot is 30 x (2 MiB state + 96 KiB conv) = 62.9 MiB.
// A copy reads and writes, so the bandwidth term alone is ~0.26 ms, and 60 small copy calls add launch time; estimate_copy_ms() gives ~0.5 ms. Still tiny next to a
// 24 ms decode step, but measure it (E0) before relying on the smaller number.
//
// The manager is a template over the memory operations so it can be tested on the host (HostOps) and used with CUDA (CudaOps, below).
#pragma once

#include "pelias_defs.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(PELIAS_CUDA_SHIM)
#include "../lib/cuda_shim.h"
#define PELIAS_E5_HAVE_CUDA 1
#elif defined(__CUDACC__) || defined(PELIAS_E5_FORCE_CUDA)
#include <cuda_runtime.h>
#define PELIAS_E5_HAVE_CUDA 1
#endif

namespace pelias {

constexpr int E5_SLOTS = 2;

// bandwidth term (read + write of every byte) plus a launch term per copy call
inline double estimate_copy_ms(size_t bytes_total, int n_copy_calls, double bw_gb_s = 504.0, double per_call_us = 4.0) {
    return 2.0 * static_cast<double>(bytes_total) / (bw_gb_s * 1e9) * 1e3 + n_copy_calls * per_call_us * 1e-3;
}

// bytes of one sequence's DeltaNet state for this model: recurrent matrix (head_k x head_v x n_v_heads, f32) and the short-convolution tail ((kernel-1) x channels, f32)
struct DeltaNetGeometry {
    int head_k = 128, head_v = 128, n_k_heads = 16, n_v_heads = 32, conv_kernel = 4;
    size_t recurrent_bytes() const { return static_cast<size_t>(head_k) * head_v * n_v_heads * 4; }
    size_t conv_bytes() const {
        const size_t channels = static_cast<size_t>(n_v_heads) * head_v + 2u * n_k_heads * head_k;
        return static_cast<size_t>(conv_kernel - 1) * channels * 4;
    }
    size_t per_layer() const { return recurrent_bytes() + conv_bytes(); }
    size_t per_slot(int layers = N_DELTANET) const { return per_layer() * layers; }
};

struct HostOps {
    using Stream = void *;
    void * alloc(size_t n) { return std::malloc(n); }
    void   release(void * p) { std::free(p); }
    bool   copy(void * dst, const void * src, size_t n, Stream) { std::memcpy(dst, src, n); return true; }
};

#ifdef PELIAS_E5_HAVE_CUDA
struct CudaOps {
    using Stream = cudaStream_t;
    void * alloc(size_t n) {
        void * p = nullptr;
        return cudaMalloc(&p, n) == cudaSuccess ? p : nullptr;
    }
    void release(void * p) { cudaFree(p); }
    bool copy(void * dst, const void * src, size_t n, Stream s) { return cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, s) == cudaSuccess; }
};
#endif

template <class Ops>
class DeltaNetStateManager {
public:
    using Stream = typename Ops::Stream;

    explicit DeltaNetStateManager(int expected_layers = N_DELTANET, Ops ops = Ops()) : ops_(ops), expected_(expected_layers) {
        layers_.resize(expected_);
        for (int s = 0; s < E5_SLOTS; ++s) { snap_rec_[s].assign(expected_, nullptr); snap_conv_[s].assign(expected_, nullptr); pos_[s] = -1; }
    }
    DeltaNetStateManager(const DeltaNetStateManager &) = delete;
    DeltaNetStateManager & operator=(const DeltaNetStateManager &) = delete;
    ~DeltaNetStateManager() { release_all(); }

    // idx: 0..expected-1 (the order of the DeltaNet layers). Allocates both slots' copies; on an allocation failure everything this call allocated is released.
    bool register_layer(int idx, int model_layer, void * live_rec, size_t rec_bytes, void * live_conv, size_t conv_bytes) {
        if (idx < 0 || idx >= expected_ || layers_[idx].live_rec != nullptr || !live_rec || !live_conv || rec_bytes == 0 || conv_bytes == 0) return false;
        if (layer_is_full_attention(model_layer)) return false;                               // a full-attention layer has no recurrent state
        void * got[E5_SLOTS][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
        for (int s = 0; s < E5_SLOTS; ++s) {
            got[s][0] = ops_.alloc(rec_bytes);
            got[s][1] = ops_.alloc(conv_bytes);
            if (!got[s][0] || !got[s][1]) {
                for (int t = 0; t <= s; ++t) { if (got[t][0]) ops_.release(got[t][0]); if (got[t][1]) ops_.release(got[t][1]); }
                return false;
            }
        }
        layers_[idx] = {model_layer, live_rec, rec_bytes, live_conv, conv_bytes};
        for (int s = 0; s < E5_SLOTS; ++s) { snap_rec_[s][idx] = got[s][0]; snap_conv_[s][idx] = got[s][1]; }
        bytes_per_slot_ += rec_bytes + conv_bytes;
        ++registered_;
        return true;
    }

    bool ready() const { return registered_ == expected_; }
    int  registered() const { return registered_; }
    size_t bytes_per_slot() const { return bytes_per_slot_; }
    int64_t slot_pos(int slot) const { return (slot >= 0 && slot < E5_SLOTS) ? pos_[slot] : -1; }
    void invalidate(int slot) { if (slot >= 0 && slot < E5_SLOTS) pos_[slot] = -1; }

    // live -> slot. token_pos is the number of tokens the state has absorbed (the position to roll the attention layers back to). Returns false and leaves the slot invalid on any failure.
    bool snapshot(int slot, int64_t token_pos, Stream s) {
        if (!ready() || slot < 0 || slot >= E5_SLOTS || token_pos < 0) return false;
        pos_[slot] = -1;
        for (int i = 0; i < expected_; ++i) {
            if (!ops_.copy(snap_rec_[slot][i], layers_[i].live_rec, layers_[i].rec_bytes, s)) return false;
            if (!ops_.copy(snap_conv_[slot][i], layers_[i].live_conv, layers_[i].conv_bytes, s)) return false;
        }
        pos_[slot] = token_pos;
        return true;
    }

    // slot -> live. Returns the snapshot's token position, or -1 when the slot holds no valid snapshot (nothing is touched then). The slot stays valid (it can be restored again).
    int64_t rollback(int slot, Stream s) {
        if (!ready() || slot < 0 || slot >= E5_SLOTS || pos_[slot] < 0) return -1;
        for (int i = 0; i < expected_; ++i) {
            if (!ops_.copy(layers_[i].live_rec, snap_rec_[slot][i], layers_[i].rec_bytes, s)) return -1;
            if (!ops_.copy(layers_[i].live_conv, snap_conv_[slot][i], layers_[i].conv_bytes, s)) return -1;
        }
        return pos_[slot];
    }

private:
    struct Live { int model_layer = -1; void * live_rec = nullptr; size_t rec_bytes = 0; void * live_conv = nullptr; size_t conv_bytes = 0; };

    void release_all() {
        for (int s = 0; s < E5_SLOTS; ++s)
            for (int i = 0; i < expected_; ++i) {
                if (snap_rec_[s][i]) ops_.release(snap_rec_[s][i]);
                if (snap_conv_[s][i]) ops_.release(snap_conv_[s][i]);
            }
    }

    Ops                      ops_;
    int                      expected_;
    int                      registered_ = 0;
    size_t                   bytes_per_slot_ = 0;
    std::vector<Live>        layers_;
    std::vector<void *>      snap_rec_[E5_SLOTS], snap_conv_[E5_SLOTS];
    int64_t                  pos_[E5_SLOTS];
};

}  // namespace pelias
