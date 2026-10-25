// File: <project>\moe_track\tests\test_mmid_skip.cpp
// Unit test for the one change the Pelias-Tier patch makes inside ggml: ggml_mul_mat_id on the CPU treats a NEGATIVE expert id as "no work for this
// row" and writes zeros to that output row. It must hold in every CPU code path: the generic one and the repacked-weights one (Q4_K / Q6_K / Q4_0 / ... when
// the weights sit in the CPU "repack" buffer, which is what a normal llama.cpp CPU load uses).
//
// For every weight type and every CPU buffer type that is available, and for batches of 1, 5 and 16 tokens:
//   run A: ids with only valid experts          -> reference rows
//   run B: the same ids with some entries set to -1 (and, in one case, ALL entries set to -1)
//   check: every row whose id is unchanged is bit-identical to run A; every row whose id is -1 is exactly 0 (the output buffer is pre-filled with NaN).
//
// Two modes (K-quant weights have a second, "tiled" CPU kernel that ggml picks only when an expert has >= 8 rows; killing ids changes the rows per expert,
// so the kernel can change between run A and run B, which moves the last float bits but is not what this test is about):
//   ./test_mmid_skip            tiled kernel switched off (GGML_CPU_TILED_MM=0): kept rows must be BIT-IDENTICAL, skipped rows exactly 0
//   ./test_mmid_skip --tiled    tiled kernel left on: kept rows must agree within 2e-3 of the largest output value, skipped rows still exactly 0
// Build:  g++ -O2 -std=c++17 test_mmid_skip.cpp -I<llama.cpp>/ggml/include -L<build>/bin -lggml -lggml-cpu -lggml-base -o test_mmid_skip
// Run:    LD_LIBRARY_PATH=<build>/bin ./test_mmid_skip ; LD_LIBRARY_PATH=<build>/bin ./test_mmid_skip --tiled
// Exit 0 = pass. (On an UNPATCHED ggml this test crashes or fails: that is the expected negative control.)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; std::printf("FAIL "); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

typedef ggml_backend_buffer_type_t * (*get_extra_t)(ggml_backend_dev_t);

static std::vector<float> run_mmid(ggml_backend_t be, ggml_backend_buffer_type_t buft, bool is_extra, ggml_type type, const std::vector<float> & w_f32,
                                   int64_t ne0, int64_t ne1, int64_t n_exp, const std::vector<float> & x, const std::vector<int32_t> & ids,
                                   int64_t n_used, int64_t n_tok, bool & ok) {
    ok = false;
    // weights
    ggml_init_params wp = { ggml_tensor_overhead() * 4, nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * w = ggml_new_tensor_3d(wctx, type, ne0, ne1, n_exp);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, buft);
    if (!wbuf) { ggml_free(wctx); return {}; }
    if (is_extra && w->extra == nullptr) {   // the repack buffer has no packed layout for this weight type: llama.cpp would not put it there either
        ggml_backend_buffer_free(wbuf);
        ggml_free(wctx);
        return {};
    }
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    {
        std::vector<uint8_t> q(ggml_nbytes(w));
        if (type == GGML_TYPE_F32) {
            std::memcpy(q.data(), w_f32.data(), q.size());
        } else if (type == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row(w_f32.data(), (ggml_fp16_t *) q.data(), (int64_t) w_f32.size());
        } else {
            ggml_quantize_init(type);
            ggml_quantize_chunk(type, w_f32.data(), q.data(), 0, ne1 * n_exp, ne0, nullptr);
        }
        ggml_backend_tensor_set(w, q.data(), 0, q.size());
    }

    // graph
    ggml_init_params gp = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * gctx = ggml_init(gp);
    ggml_tensor * src1 = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, ne0, 1, n_tok);
    ggml_tensor * tid  = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, n_used, n_tok);
    ggml_set_input(src1);
    ggml_set_input(tid);
    ggml_tensor * dst = ggml_mul_mat_id(gctx, w, src1, tid);
    ggml_set_output(dst);
    ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, dst);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(ga, gf);

    ggml_backend_tensor_set(src1, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_tensor_set(tid, ids.data(), 0, ids.size() * sizeof(int32_t));
    std::vector<float> nan_fill((size_t) ggml_nelements(dst), std::nanf(""));
    ggml_backend_tensor_set(dst, nan_fill.data(), 0, nan_fill.size() * sizeof(float));
    const ggml_status st = ggml_backend_graph_compute(be, gf);

    std::vector<float> out((size_t) ggml_nelements(dst));
    ggml_backend_tensor_get(dst, out.data(), 0, out.size() * sizeof(float));
    ok = (st == GGML_STATUS_SUCCESS);

    ggml_gallocr_free(ga);
    ggml_free(gctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    return out;
}

int main(int argc, char ** argv) {
    bool tiled = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--tiled") tiled = true;
        else { std::fprintf(stderr, "usage: test_mmid_skip [--tiled]\n"); return 2; }
    }
    if (!tiled) setenv("GGML_CPU_TILED_MM", "0", 1);          // must be set before the first graph runs: ggml reads it once
    std::printf("mode: %s\n", tiled ? "tiled kernel on, tolerance 2e-3 on kept rows" : "tiled kernel off, kept rows bit-identical");
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, 3);

    std::vector<std::pair<std::string, ggml_backend_buffer_type_t>> bufts;
    std::vector<bool> is_extra_v;
    bufts.emplace_back("cpu", ggml_backend_cpu_buffer_type());
    is_extra_v.push_back(false);
    if (ggml_backend_dev_t dev = ggml_backend_get_device(be)) {
        if (ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev)) {
            auto fn = (get_extra_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
            if (fn) {
                int i = 0;
                for (ggml_backend_buffer_type_t * p = fn(dev); p && *p; ++p, ++i) {
                    bufts.emplace_back(std::string("extra:") + ggml_backend_buft_name(*p), *p);
                    is_extra_v.push_back(true);
                }
            }
        }
    }
    std::printf("CPU buffer types under test:");
    for (auto & b : bufts) std::printf(" %s", b.first.c_str());
    std::printf("\n");

    const ggml_type types[] = { GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K };
    const int64_t ne0 = 256, ne1 = 64, n_exp = 8, n_used = 4;
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    int n_cases = 0;

    for (ggml_type type : types) {
        std::vector<float> w((size_t) (ne0 * ne1 * n_exp));
        for (auto & v : w) v = nd(rng) * 0.1f;
        for (size_t bi = 0; bi < bufts.size(); ++bi) {
            auto & bt = bufts[bi];
            const bool is_extra = is_extra_v[bi];
            for (int64_t n_tok : { 1, 5, 16 }) {
                std::vector<float> x((size_t) (ne0 * n_tok));
                for (auto & v : x) v = nd(rng);
                std::vector<int32_t> ids((size_t) (n_used * n_tok));
                for (auto & v : ids) v = (int32_t) (rng() % n_exp);

                bool ok = false;
                std::vector<float> ref = run_mmid(be, bt.second, is_extra, type, w, ne0, ne1, n_exp, x, ids, n_used, n_tok, ok);
                if (!ok || ref.empty()) {
                    std::printf("skip  %s in %s (no packed layout / op not usable for this weight type)\n", ggml_type_name(type), bt.first.c_str());
                    continue;
                }
                for (int variant = 0; variant < 3; ++variant) {
                    std::vector<int32_t> ids2 = ids;
                    std::vector<char> skipped(ids.size(), 0);
                    for (size_t i = 0; i < ids2.size(); ++i) {
                        const bool kill = variant == 0 ? (rng() % 5 < 2) : (variant == 1 ? true : (i % 2 == 0));
                        if (kill) { ids2[i] = -1; skipped[i] = 1; }
                    }
                    std::vector<float> got = run_mmid(be, bt.second, is_extra, type, w, ne0, ne1, n_exp, x, ids2, n_used, n_tok, ok);
                    CHECK(ok && got.size() == ref.size(), "%s/%s tok=%d variant=%d: graph did not run", ggml_type_name(type), bt.first.c_str(), (int) n_tok, variant);
                    if (!ok || got.size() != ref.size()) continue;
                    int bad_kept = 0, bad_zero = 0;
                    float ref_max = 0.f;
                    for (float v : ref) ref_max = std::max(ref_max, std::fabs(v));
                    const float tol = tiled ? 2e-3f * std::max(1.0f, ref_max) : 0.0f;
                    for (int64_t t = 0; t < n_tok; ++t) {
                        for (int64_t u = 0; u < n_used; ++u) {
                            const size_t row = (size_t) ((t * n_used + u) * ne1);
                            for (int64_t r = 0; r < ne1; ++r) {
                                const float g = got[row + r];
                                if (skipped[(size_t) (t * n_used + u)]) {
                                    if (!(g == 0.0f)) ++bad_zero;
                                } else if (tiled ? !(std::fabs(g - ref[row + r]) <= tol) : std::memcmp(&g, &ref[row + r], sizeof(float)) != 0) {
                                    ++bad_kept;
                                }
                            }
                        }
                    }
                    ++n_cases;
                    CHECK(bad_zero == 0, "%s/%s tok=%d variant=%d: %d skipped output values are not zero", ggml_type_name(type), bt.first.c_str(), (int) n_tok, variant, bad_zero);
                    CHECK(bad_kept == 0, "%s/%s tok=%d variant=%d: %d kept output values changed", ggml_type_name(type), bt.first.c_str(), (int) n_tok, variant, bad_kept);
                }
            }
        }
    }
    std::printf("%d cases run\n", n_cases);
    ggml_backend_free(be);
    std::printf(g_fail == 0 ? "ALL PASSED\n" : "%d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
