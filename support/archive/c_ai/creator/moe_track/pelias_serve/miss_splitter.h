// File: <project>\moe_track\pelias_serve\miss_splitter.h
// Pass 1 / piece 3: the per-layer row-range allocator for cold ("miss") experts.
//
// WHAT IT DECIDES. For one layer of one decode token, the router has picked top-8 experts; some are in the VRAM hot bank (hits, not handled here) and the rest
// are misses whose weights live in host RAM (cold_weights_view.h). Each miss expert is computed by BOTH sides at once: the GPU reads part of it over PCIe
// (zero-copy) and the CPU reads the other part from DDR4, so the two buses are busy together (microbench 04). The split is by rows of the expert's
// intermediate dimension (the SwiGLU rows):
//
//     h[r] = silu(gate_r . x) * (up_r . x)          for r in the side's row range      (reads only gate/up rows r)
//     y   += w_e * ( down[:, r-range] @ h[r-range] ) (a K-slice of the down projection; reads only that K-slice)
//
// so each side needs no data from the other until the final sum y_gpu + y_cpu (the combine in handoff.cuh). For every miss expert the plan gives one cut
// row `split_row`: the GPU takes rows [0, split_row), the CPU takes rows [split_row, n_ff).
//
// GRANULARITY (read this). A K-slice of the down projection must start and end on a 256-element K-quant super-block (QK_K = 256), so cuts are multiples
// of the granule (default 256). For this model n_ff = 512, so an expert has only TWO granules: one miss expert can be split 0/512, 256/256 or 512/0 rows,
// never 60/40. The allocator therefore works in granules over all miss experts of the layer at once, and carries the rounding error from call to call
// (error diffusion, fixed point), so the LONG-RUN GPU share converges to the requested fraction even though a single layer can only hit multiples of
// 1/(2*n_miss). At most one expert per layer is cut in two (the allocator fills the GPU side from the first expert onward).
//
// THE RATIO. The default 0.60 is the value in the work order. Microbench 04 measured, with both sides busy together, GPU zero-copy 21.1 GB/s and CPU
// 26.6 GB/s, which is 0.44 for the GPU; use gpu_frac_from_bandwidths(21.06, 26.56) or any measured pair. The split should equalise the two sides' finish
// times, so the right number is whatever real kernels sustain; it is a parameter, not a constant of the design.
//
// Header-only, no dependencies beyond <cstdint> and pelias_defs.h. Every function is host+device callable (plain integer arithmetic, no allocation), so the
// plan can be made on the CPU (tests) or inside the GPU router kernel.
#pragma once

#include "pelias_defs.h"

#include <cstdint>

#if defined(__CUDACC__)
#define PELIAS_SPLIT_HD __host__ __device__
#else
#define PELIAS_SPLIT_HD
#endif

namespace pelias {

constexpr int32_t QK_K = 256;                       // K-quant super-block: elements per block of Q4_K / Q6_K (and Q8_K activations)
constexpr float   DEFAULT_GPU_FRAC = 0.60f;         // work-order value; the 04 measurement suggests ~0.44 (see above)
constexpr int32_t SPLIT_CREDIT_LIMIT_Q16 = 16 << 16;   // the carried rounding error is kept within +-16 granules

// One miss expert as the router kernel knows it.
struct MissExpert {
    int32_t expert;     // routed expert index 0..N_EXPERTS-1
    float   weight;     // router weight (applied after the down projection, as llama.cpp does)
};

// One entry of the plan. GPU rows: [0, split_row).  CPU rows: [split_row, n_ff).
struct SplitTask {
    int32_t expert;
    float   weight;
    int32_t split_row;
};

struct SplitPlan {
    int32_t   n_miss;       // entries used in task[]
    int32_t   n_ff;         // intermediate rows per expert
    int32_t   granule;      // cut granularity in rows
    int32_t   gpu_rows;     // total rows the GPU computes (sum over experts)
    int32_t   cpu_rows;     // total rows the CPU computes
    SplitTask task[MAX_COLD];

    PELIAS_SPLIT_HD bool needs_cpu() const { return cpu_rows > 0; }   // false: no handoff is needed for this layer
    PELIAS_SPLIT_HD bool needs_gpu() const { return gpu_rows > 0; }
    PELIAS_SPLIT_HD int32_t gpu_rows_of(int i) const { return task[i].split_row; }
    PELIAS_SPLIT_HD int32_t cpu_rows_of(int i) const { return n_ff - task[i].split_row; }
};

struct SplitterConfig {
    uint32_t gpu_frac_q16 = 39322;      // requested GPU share of the rows, 16.16 fixed point (39322/65536 = 0.60)
    int32_t  n_ff         = EXPERT_FF;  // 512 for this model
    int32_t  granule      = QK_K;       // 256; must not be changed for real K-quant weights
};

// Rounding error carried between calls (units: 1/65536 of a granule). One per splitter; copy it to the device if the plan is made there.
struct SplitterState {
    int32_t credit_q16 = 0;
};

PELIAS_SPLIT_HD inline uint32_t gpu_frac_to_q16(float f) {
    if (!(f > 0.0f)) return 0u;                       // also catches NaN
    if (f >= 1.0f) return 65536u;
    return static_cast<uint32_t>(f * 65536.0f + 0.5f);
}

PELIAS_SPLIT_HD inline SplitterConfig make_splitter_config(float gpu_frac, int32_t n_ff = EXPERT_FF, int32_t granule = QK_K) {
    SplitterConfig c;
    c.gpu_frac_q16 = gpu_frac_to_q16(gpu_frac);
    c.n_ff = n_ff;
    c.granule = granule;
    return c;
}

// GPU share that makes both sides finish together: gpu / (gpu + cpu), from the bandwidths each side sustains while the other is busy.
PELIAS_SPLIT_HD inline float gpu_frac_from_bandwidths(double gpu_gb_s, double cpu_gb_s) {
    const double s = gpu_gb_s + cpu_gb_s;
    return (gpu_gb_s > 0.0 && s > 0.0) ? static_cast<float>(gpu_gb_s / s) : 0.0f;
}

PELIAS_SPLIT_HD inline bool splitter_config_valid(const SplitterConfig& c) {
    return c.n_ff > 0 && c.granule > 0 && c.gpu_frac_q16 <= 65536u;
}

// Make the plan for one layer. `miss` holds the experts that are not in the VRAM bank, in the order the caller wants them filled (GPU side first).
// Returns the number of experts planned (n_miss clipped to MAX_COLD). `state` may be null (then no error is carried: every call rounds on its own).
PELIAS_SPLIT_HD inline int split_misses(const SplitterConfig& cfg, SplitterState* state, const MissExpert* miss, int n_miss, SplitPlan* out) {
    int n = n_miss < 0 ? 0 : n_miss;
    if (n > MAX_COLD) n = MAX_COLD;
    const int32_t g = cfg.granule > 0 ? cfg.granule : QK_K;
    const int32_t n_ff = cfg.n_ff > 0 ? cfg.n_ff : EXPERT_FF;
    const int32_t upe = (n_ff + g - 1) / g;                       // granules per expert (2 for n_ff = 512)
    const int32_t units = n * upe;

    out->n_miss = n;
    out->n_ff = n_ff;
    out->granule = g;
    out->gpu_rows = 0;
    out->cpu_rows = 0;

    // error diffusion: how many granules go to the GPU this call
    const int64_t target = static_cast<int64_t>(units) * static_cast<int64_t>(cfg.gpu_frac_q16 > 65536u ? 65536u : cfg.gpu_frac_q16) +
                           (state ? static_cast<int64_t>(state->credit_q16) : 0);
    int64_t gu = (target + 32768) >> 16;                           // nearest whole granule (arithmetic shift also for a negative target)
    if (gu < 0) gu = 0;
    if (gu > units) gu = units;
    if (state) {
        int64_t credit = target - (gu << 16);
        if (credit > SPLIT_CREDIT_LIMIT_Q16) credit = SPLIT_CREDIT_LIMIT_Q16;
        if (credit < -SPLIT_CREDIT_LIMIT_Q16) credit = -SPLIT_CREDIT_LIMIT_Q16;
        state->credit_q16 = static_cast<int32_t>(credit);
    }

    for (int i = 0; i < n; ++i) {
        int64_t mine = gu - static_cast<int64_t>(i) * upe;         // granules of the pool that fall on expert i
        if (mine < 0) mine = 0;
        if (mine > upe) mine = upe;
        int32_t cut = static_cast<int32_t>(mine) * g;
        if (cut > n_ff) cut = n_ff;                                // the last granule may be short when n_ff is not a multiple of the granule
        out->task[i].expert = miss[i].expert;
        out->task[i].weight = miss[i].weight;
        out->task[i].split_row = cut;
        out->gpu_rows += cut;
        out->cpu_rows += n_ff - cut;
    }
    return n;
}

}  // namespace pelias
