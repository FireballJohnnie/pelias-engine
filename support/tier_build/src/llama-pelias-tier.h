#pragma once

// Pelias-Tier (Phase 6): a profile-chosen "hot bank" of experts in VRAM, the rest on the CPU, inside the normal llama.cpp graph.
//
// Nothing here does anything unless the environment variable PELIAS_HOT_SET names a hot-set file. With it unset, no memory is allocated,
// no graph node changes and every code path is the stock one.
//
// How a decode token goes through a MoE layer when the bank is active
//   router (stock)  ->  selected experts  ->  two small id lookups (one table lookup each)
//     hot branch : ids_hot  = bank slot of the expert, or slot 0 (an all-zero expert) when the expert is not in the bank
//                  mul_mat_id on the bank tensors (weights live in VRAM)  -> a miss reads the zero expert and contributes exactly 0
//     cold branch: ids_cold = -1 when the expert is in the bank, else the expert id
//                  mul_mat_id on the ORIGINAL tensors (weights live in RAM); ggml-cpu skips id -1 and writes zeros for that row
//   the two weighted sums are added. The result equals the stock result up to float summation order.
//
// The dual path is used only when the batch has fewer tokens than pelias_tier_dual_max_tokens(). Bigger batches (prefill) take the stock path,
// which is the one ggml's scheduler can offload to the GPU; that offload must never see a "-1" id.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>

struct llama_model;

struct pelias_tier_layer {
    int il       = -1;
    int n_expert = 0;
    int n_hot    = 0;                   // experts held in the bank; the bank tensors have n_hot + 1 slots, slot 0 is all zeros

    ggml_tensor * gate    = nullptr;    // bank copies of the layer's expert tensors, ne[2] = n_hot + 1
    ggml_tensor * up      = nullptr;
    ggml_tensor * gate_up = nullptr;
    ggml_tensor * down    = nullptr;

    ggml_tensor * orig_gate    = nullptr;   // the model's own (RAM) expert tensors of this layer, used by the cold branch
    ggml_tensor * orig_up      = nullptr;
    ggml_tensor * orig_gate_up = nullptr;
    ggml_tensor * orig_down    = nullptr;

    ggml_tensor * tab_hot  = nullptr;   // I32 [1, n_expert]: bank slot of expert e (1..n_hot), 0 if e is not in the bank
    ggml_tensor * tab_cold = nullptr;   // I32 [1, n_expert]: -1 if e is in the bank, else e
};

// true when PELIAS_HOT_SET is set (cheap; reads the environment once)
bool pelias_tier_enabled();

// the dual path is used only when a batch has fewer tokens than this: min(PELIAS_TIER_BYPASS_N, GGML_OP_OFFLOAD_MIN_BATCH), both default 32
int pelias_tier_dual_max_tokens();

// Called once after the model's tensors are loaded. Builds the banks named by the hot-set file; logs what it did; never throws.
// gguf_path must be the single-file GGUF the model was loaded from (split files are not supported: the layer is left on the stock path).
void pelias_tier_attach(const llama_model & model, const std::string & gguf_path);

// Called from ~llama_model: frees the banks of that model.
void pelias_tier_detach(const llama_model * model);

// The bank for the layer whose original ffn_down_exps tensor is `down_exps`; nullptr when the layer has no bank.
const pelias_tier_layer * pelias_tier_find(const ggml_tensor * down_exps);

// Diagnostics: called each time a dual-branch graph is built for a layer (graphs are rebuilt when the batch shape changes, not every token).
// If PELIAS_TIER_STATS_FILE is set, a small JSON file with the counts is written when the model is freed.
void pelias_tier_note_dual_build(int il, int n_tokens);
