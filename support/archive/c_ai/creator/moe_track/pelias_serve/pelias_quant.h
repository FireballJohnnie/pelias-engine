// File: <project>\moe_track\pelias_serve\pelias_quant.h
// A thin, checked wrapper over ggml's own CPU quantisation kernels: nothing is re-implemented, so a Q4_K / Q6_K / Q5_K / Q8_0 dot product here is the
// same code llama.cpp runs for mul_mat_id. (The Track 2 draft in the roadmap multiplied the raw 4-bit nibbles by the activations and ignored the block
// scales and minimums, which is not Q4_K; this wrapper exists so that cannot happen.)
//
// Usage:
//   pelias::QuantOps ops;  std::string err;
//   if (!pelias::quant_ops_for(GGML_TYPE_Q4_K, ops, &err)) { ... }
//   std::vector<uint8_t> xq(ops.act_row_bytes(2048));
//   ops.quantize_act(x, xq.data(), 2048);                       // float -> Q8_K
//   float d = ops.dot(2048, weight_row_ptr, xq.data());         // one output element
//
// Link with:  -lggml -lggml-cpu -lggml-base      (headers from <llama.cpp>/ggml/include)
#pragma once

#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <string>

namespace pelias {

struct QuantOps {
    ggml_type         wtype      = GGML_TYPE_F32;   // type of the weight rows
    ggml_type         atype      = GGML_TYPE_F32;   // type the activation must be converted to for vec_dot (Q8_K for the K-quants)
    ggml_vec_dot_t    vec_dot    = nullptr;
    ggml_from_float_t from_float = nullptr;         // float -> atype

    size_t w_row_bytes(int64_t n)   const { return ggml_row_size(wtype, n); }
    size_t act_row_bytes(int64_t n) const { return ggml_row_size(atype, n); }

    // n must be a multiple of the type's block size (256 for the K-quants, 32 for Q8_0/Q4_0)
    void quantize_act(const float * x, void * dst, int64_t n) const { from_float(x, dst, n); }

    float dot(int64_t n, const void * w_row, const void * act_q) const {
        float s = 0.0f;
        vec_dot(static_cast<int>(n), &s, 0, w_row, 0, act_q, 0, 1);
        return s;
    }
};

inline bool quant_ops_for(ggml_type wtype, QuantOps & out, std::string * err = nullptr) {
    ggml_cpu_init();                                 // fills the lookup tables the kernels use; cheap and idempotent
    const ggml_type_traits_cpu * tw = ggml_get_type_traits_cpu(wtype);
    if (!tw || !tw->vec_dot) {
        if (err) *err = std::string("no CPU dot-product kernel for weight type ") + ggml_type_name(wtype);
        return false;
    }
    const ggml_type_traits_cpu * ta = ggml_get_type_traits_cpu(tw->vec_dot_type);
    if (!ta || !ta->from_float) {
        if (err) *err = std::string("no float->") + ggml_type_name(tw->vec_dot_type) + " conversion for activations";
        return false;
    }
    out.wtype      = wtype;
    out.atype      = tw->vec_dot_type;
    out.vec_dot    = tw->vec_dot;
    out.from_float = ta->from_float;
    return true;
}

}  // namespace pelias
