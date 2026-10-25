// File: <project>\moe_track\pelias_serve\pelias_defs.h
// Shared constants for the Pelias-Serve pieces (Track 2). They describe Qwen3.6-35B-A3B (qwen35moe); nothing here reads a model file.
// The values are checked against the GGUF by the code that uses them (cold_pool.h refuses matrices of another shape), not assumed silently.
#pragma once

#include <cstdint>

namespace pelias {

constexpr int N_LAYERS        = 40;    // decoder layers; blk.40 is the MTP block
constexpr int N_DELTANET      = 30;    // layers with Gated DeltaNet (layer % 4 != 3)
constexpr int N_FULL_ATTN     = 10;    // layers with full attention (layer % 4 == 3)
constexpr int HIDDEN          = 2048;
constexpr int N_EXPERTS       = 256;
constexpr int TOP_K           = 8;
constexpr int EXPERT_FF       = 512;   // routed-expert intermediate size
constexpr int MAX_COLD        = 16;    // the most experts one call of the cold pool handles (TOP_K is 8; 16 leaves room for batched MTP rows)
constexpr int PP_BYPASS_TOKENS = 64;   // batches of this many tokens or more are prefill and are never taken by the Pelias-Serve decode path

inline bool layer_is_full_attention(int layer) { return layer % 4 == 3; }

}  // namespace pelias
