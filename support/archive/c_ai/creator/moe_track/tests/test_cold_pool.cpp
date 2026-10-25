// File: <project>\moe_track\tests\test_cold_pool.cpp
// Tests pelias_serve/cold_pool.h (Stage E3) against ggml itself: random Q4_K / Q6_K expert matrices, the same layer computed
//   (1) by the persistent pool,
//   (2) by a real ggml graph (mul_mat_id x3, silu, mul, weights, sum) on the CPU backend  -> must agree to float rounding,
//   (3) in plain float from the dequantised weights                                      -> agrees up to the activation quantisation,
// plus: the result does not depend on the number of workers, a fused gate_up layout gives the same answer, the pool survives many back-to-back calls with
// changing expert sets (barrier races show up here, especially under -fsanitize=thread), and bad arguments are refused.
//
// Build:  g++ -O2 -std=c++17 -pthread test_cold_pool.cpp -I../pelias_serve -I<llama.cpp>/ggml/include -L<build>/bin -lggml -lggml-cpu -lggml-base -o test_cold_pool
//         (thread sanitizer:  add -fsanitize=thread -g and run with --quick)
// Run:    LD_LIBRARY_PATH=<build>/bin ./test_cold_pool [--quick]        Exit 0 = pass.
#include "cold_pool.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

using namespace pelias;

static int g_fail = 0;
static void check(const char * name, bool ok, const char * detail = "") {
    std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", name, (!ok && detail[0]) ? "   " : "", ok ? "" : detail);
    if (!ok) ++g_fail;
}

static constexpr int HID = HIDDEN, FF = EXPERT_FF;

struct Layer {
    int E;
    ggml_type tg, td;
    std::vector<uint8_t> gate, up, down;           // plain quantised bytes, [E][rows][row_bytes]
    size_t rb_gu, rb_dn;
    LayerCold view() const {
        LayerCold L;
        L.gate = {gate.data(), tg, FF, HID, rb_gu, static_cast<size_t>(FF) * rb_gu};
        L.up   = {up.data(),   tg, FF, HID, rb_gu, static_cast<size_t>(FF) * rb_gu};
        L.down = {down.data(), td, HID, FF, rb_dn, static_cast<size_t>(HID) * rb_dn};
        return L;
    }
};

static std::vector<uint8_t> quant_stack(ggml_type t, int E, int rows, int cols, std::mt19937 & rng, float scale) {
    std::normal_distribution<float> nd(0.0f, scale);
    std::vector<float> f(static_cast<size_t>(E) * rows * cols);
    for (auto & v : f) v = nd(rng);
    const size_t rb = ggml_row_size(t, cols);
    std::vector<uint8_t> q(static_cast<size_t>(E) * rows * rb);
    ggml_quantize_chunk(t, f.data(), q.data(), 0, static_cast<int64_t>(E) * rows, cols, nullptr);
    return q;
}

static Layer make_layer(int E, ggml_type tg, ggml_type td, uint32_t seed) {
    std::mt19937 rng(seed);
    Layer L;
    L.E = E; L.tg = tg; L.td = td;
    L.rb_gu = ggml_row_size(tg, HID);
    L.rb_dn = ggml_row_size(td, FF);
    L.gate = quant_stack(tg, E, FF, HID, rng, 0.05f);
    L.up   = quant_stack(tg, E, FF, HID, rng, 0.05f);
    L.down = quant_stack(td, E, HID, FF, rng, 0.05f);
    return L;
}

// reference 1: the same layer as a ggml graph on the CPU backend (what llama.cpp runs for the cold experts)
static std::vector<float> graph_reference(const Layer & L, const std::vector<float> & x, const std::vector<int32_t> & ids, const std::vector<float> & w) {
    const int n = static_cast<int>(ids.size());
    ggml_init_params ip = {static_cast<size_t>(160) << 20, nullptr, false};
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * gw = ggml_new_tensor_3d(ctx, L.tg, HID, FF, L.E);
    ggml_tensor * uw = ggml_new_tensor_3d(ctx, L.tg, HID, FF, L.E);
    ggml_tensor * dw = ggml_new_tensor_3d(ctx, L.td, FF, HID, L.E);
    std::memcpy(gw->data, L.gate.data(), ggml_nbytes(gw));
    std::memcpy(uw->data, L.up.data(), ggml_nbytes(uw));
    std::memcpy(dw->data, L.down.data(), ggml_nbytes(dw));
    ggml_tensor * xt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HID, 1, 1);
    std::memcpy(xt->data, x.data(), sizeof(float) * HID);
    ggml_tensor * it = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n, 1);
    std::memcpy(it->data, ids.data(), sizeof(int32_t) * n);
    ggml_tensor * wt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n, 1);
    std::memcpy(wt->data, w.data(), sizeof(float) * n);
    ggml_tensor * g = ggml_mul_mat_id(ctx, gw, xt, it);
    ggml_tensor * u = ggml_mul_mat_id(ctx, uw, xt, it);
    ggml_tensor * a = ggml_mul(ctx, ggml_silu(ctx, g), u);
    ggml_tensor * d = ggml_mul_mat_id(ctx, dw, a, it);
    ggml_tensor * dwt = ggml_mul(ctx, d, wt);
    ggml_tensor * sum = ggml_view_2d(ctx, dwt, HID, 1, dwt->nb[1], 0);
    for (int k = 1; k < n; ++k) sum = ggml_add(ctx, sum, ggml_view_2d(ctx, dwt, HID, 1, dwt->nb[1], static_cast<size_t>(k) * dwt->nb[1]));
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, sum);
    ggml_graph_compute_with_ctx(ctx, gf, 2);
    std::vector<float> out(HID);
    std::memcpy(out.data(), sum->data, sizeof(float) * HID);
    ggml_free(ctx);
    return out;
}

// reference 2: plain float math on the dequantised weights
static std::vector<float> float_reference(const Layer & L, const std::vector<float> & x, const std::vector<int32_t> & ids, const std::vector<float> & w) {
    const ggml_type_traits * tg = ggml_get_type_traits(L.tg);
    const ggml_type_traits * td = ggml_get_type_traits(L.td);
    std::vector<double> out(HID, 0.0);
    std::vector<float> row(std::max(HID, FF));
    for (size_t k = 0; k < ids.size(); ++k) {
        const int e = ids[k];
        std::vector<double> inter(FF);
        for (int r = 0; r < FF; ++r) {
            tg->to_float(L.gate.data() + (static_cast<size_t>(e) * FF + r) * L.rb_gu, row.data(), HID);
            double g = 0; for (int c = 0; c < HID; ++c) g += double(row[c]) * x[c];
            tg->to_float(L.up.data() + (static_cast<size_t>(e) * FF + r) * L.rb_gu, row.data(), HID);
            double u = 0; for (int c = 0; c < HID; ++c) u += double(row[c]) * x[c];
            inter[r] = g / (1.0 + std::exp(-g)) * u;
        }
        for (int r = 0; r < HID; ++r) {
            td->to_float(L.down.data() + (static_cast<size_t>(e) * HID + r) * L.rb_dn, row.data(), FF);
            double d = 0; for (int c = 0; c < FF; ++c) d += double(row[c]) * inter[c];
            out[r] += w[k] * d;
        }
    }
    return std::vector<float>(out.begin(), out.end());
}

static float max_abs(const std::vector<float> & a) { float m = 0; for (float v : a) m = std::max(m, std::fabs(v)); return m; }
static float max_diff(const std::vector<float> & a, const std::vector<float> & b) { float m = 0; for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i])); return m; }

static void pick(std::mt19937 & rng, int E, int n, std::vector<int32_t> & ids, std::vector<float> & w, std::vector<float> & x) {
    std::vector<int> all(E);
    std::iota(all.begin(), all.end(), 0);
    std::shuffle(all.begin(), all.end(), rng);
    ids.assign(all.begin(), all.begin() + n);
    std::uniform_real_distribution<float> ud(0.05f, 0.4f);
    w.resize(n);
    for (auto & v : w) v = ud(rng);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    x.resize(HID);
    for (auto & v : x) v = nd(rng);
}

static void agree_with_references(const char * label, ggml_type tg, ggml_type td, int n_workers, bool quick) {
    const int E = 12;
    Layer L = make_layer(E, tg, td, 1234);
    LayerCold lc = L.view();
    ColdPool::Config cfg;
    cfg.n_workers = n_workers;
    ColdPool pool(cfg);
    std::mt19937 rng(77);
    std::vector<float> out(HID);
    float worst_graph = 0, worst_float = 0;
    bool graph_ok = true, float_ok = true;
    const int cases[] = {1, 2, 5, 8, 3, 7};
    const int reps = quick ? 1 : 3;
    for (int rep = 0; rep < reps; ++rep) {
        for (int n : cases) {
            std::vector<int32_t> ids; std::vector<float> w, x;
            pick(rng, E, n, ids, w, x);
            pool.run(lc, x.data(), ids.data(), w.data(), n, out.data());
            const auto rg = graph_reference(L, x, ids, w);
            const auto rf = float_reference(L, x, ids, w);
            const float sc = max_abs(rg) + 1e-12f;
            const float dg = max_diff(out, rg) / sc, df = max_diff(out, rf) / sc;
            worst_graph = std::max(worst_graph, dg); worst_float = std::max(worst_float, df);
            graph_ok = graph_ok && dg < 2e-3f;
            float_ok = float_ok && df < 6e-2f;
        }
    }
    char buf[200];
    std::snprintf(buf, sizeof buf, "%s, %d workers: equals the ggml mul_mat_id graph (worst relative diff %.2e)", label, n_workers, worst_graph);
    check(buf, graph_ok);
    std::snprintf(buf, sizeof buf, "%s, %d workers: close to plain float math on dequantised weights (worst %.2e; the gap is the 8-bit activation rounding)", label, n_workers, worst_float);
    check(buf, float_ok);
}

int main(int argc, char ** argv) {
    const bool quick = argc > 1 && std::string(argv[1]) == "--quick";
    ggml_cpu_init();

    agree_with_references("Q4_K gate/up + Q4_K down", GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, 3, quick);
    agree_with_references("Q4_K gate/up + Q6_K down", GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, 7, quick);

    // determinism across worker counts, and bit-identical repeats
    {
        Layer L = make_layer(12, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, 99);
        LayerCold lc = L.view();
        std::mt19937 rng(5);
        std::vector<int32_t> ids; std::vector<float> w, x;
        pick(rng, 12, 8, ids, w, x);
        std::vector<std::vector<float>> outs;
        for (int nw : {1, 2, 3, 7}) {
            ColdPool::Config cfg; cfg.n_workers = nw;
            ColdPool pool(cfg);
            std::vector<float> o(HID), o2(HID);
            pool.run(lc, x.data(), ids.data(), w.data(), 8, o.data());
            pool.run(lc, x.data(), ids.data(), w.data(), 8, o2.data());
            check(("repeat call is bit-identical, " + std::to_string(nw) + " workers").c_str(), std::memcmp(o.data(), o2.data(), sizeof(float) * HID) == 0);
            outs.push_back(o);
        }
        bool same = true;
        for (size_t i = 1; i < outs.size(); ++i) same = same && std::memcmp(outs[0].data(), outs[i].data(), sizeof(float) * HID) == 0;
        check("the result does not depend on the number of workers (bit-identical for 1, 2, 3, 7)", same);
    }

    // fused gate_up layout [gate rows | up rows] per expert
    {
        const int E = 6;
        Layer L = make_layer(E, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, 31);
        std::vector<uint8_t> fused(static_cast<size_t>(E) * 2 * FF * L.rb_gu);
        for (int e = 0; e < E; ++e) {
            std::memcpy(fused.data() + (static_cast<size_t>(e) * 2 * FF) * L.rb_gu, L.gate.data() + static_cast<size_t>(e) * FF * L.rb_gu, static_cast<size_t>(FF) * L.rb_gu);
            std::memcpy(fused.data() + (static_cast<size_t>(e) * 2 * FF + FF) * L.rb_gu, L.up.data() + static_cast<size_t>(e) * FF * L.rb_gu, static_cast<size_t>(FF) * L.rb_gu);
        }
        LayerCold sep = L.view(), fu = sep;
        fu.gate = {fused.data(), L.tg, FF, HID, L.rb_gu, static_cast<size_t>(2 * FF) * L.rb_gu};
        fu.up   = {fused.data() + static_cast<size_t>(FF) * L.rb_gu, L.tg, FF, HID, L.rb_gu, static_cast<size_t>(2 * FF) * L.rb_gu};
        ColdPool::Config cfg; cfg.n_workers = 4;
        ColdPool pool(cfg);
        std::mt19937 rng(8);
        std::vector<int32_t> ids; std::vector<float> w, x;
        pick(rng, E, 6, ids, w, x);
        std::vector<float> a(HID), b(HID);
        pool.run(sep, x.data(), ids.data(), w.data(), 6, a.data());
        pool.run(fu, x.data(), ids.data(), w.data(), 6, b.data());
        check("fused gate_up layout gives the same bits as separate gate and up", std::memcmp(a.data(), b.data(), sizeof(float) * HID) == 0);
    }

    // many back-to-back calls with changing expert sets and counts (this is where a missing barrier or a lost wake-up would show)
    {
        Layer L = make_layer(16, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, 4242);
        LayerCold lc = L.view();
        ColdPool::Config cfg; cfg.n_workers = 7;
        ColdPool pool(cfg);
        ColdPool single([] { ColdPool::Config c; c.n_workers = 1; return c; }());
        std::mt19937 rng(2024);
        const int iters = quick ? 60 : 400;
        bool all_same = true;
        std::vector<float> a(HID), b(HID);
        for (int it = 0; it < iters; ++it) {
            const int n = static_cast<int>(rng() % 9);                 // 0..8, including the empty call
            std::vector<int32_t> ids; std::vector<float> w, x;
            pick(rng, 16, std::max(n, 1), ids, w, x);
            ids.resize(n); w.resize(n);
            std::fill(a.begin(), a.end(), 123.0f);
            pool.run(lc, x.data(), ids.data(), w.data(), n, a.data());
            single.run(lc, x.data(), ids.data(), w.data(), n, b.data());
            if (std::memcmp(a.data(), b.data(), sizeof(float) * HID) != 0) { all_same = false; std::printf("   mismatch at iteration %d (n=%d)\n", it, n); break; }
            if (n == 0 && max_abs(a) != 0.0f) { all_same = false; break; }
        }
        check("400 (or 60) back-to-back calls: 7 workers always equal 1 worker; n = 0 gives zeros", all_same);
    }

    // refusals
    {
        Layer L = make_layer(4, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, 1);
        LayerCold lc = L.view();
        ColdPool::Config cfg; cfg.n_workers = 2;
        ColdPool pool(cfg);
        std::vector<float> x(HID, 0.1f), w(MAX_COLD + 1, 0.1f), out(HID);
        std::vector<int32_t> ids(MAX_COLD + 1, 0);
        auto throws = [&](const LayerCold & l, int n) { try { pool.run(l, x.data(), ids.data(), w.data(), n, out.data()); return false; } catch (const std::exception &) { return true; } };
        check("more than MAX_COLD experts is refused", throws(lc, MAX_COLD + 1));
        LayerCold bad = lc; bad.down.rows = HID - 1;
        check("a down matrix that does not fit gate/up is refused", throws(bad, 1));
        bad = lc; bad.up.type = GGML_TYPE_Q6_K;
        check("gate and up of different types are refused", throws(bad, 1));
        bad = lc; bad.gate.row_bytes += 4;
        check("a wrong row size is refused", throws(bad, 1));
        bad = lc; bad.gate.base = nullptr;
        check("a null weight pointer is refused", throws(bad, 1));
        check("an invalid worker count is refused", [] { try { ColdPool::Config c; c.n_workers = 0; ColdPool p(c); return false; } catch (const std::exception &) { return true; } }());
    }

    // pinning: asking for a CPU that exists works (or is reported as not pinned); empty list means no pinning
    {
        ColdPool::Config cfg; cfg.n_workers = 2; cfg.cpus = {0, 1};
        ColdPool p(cfg);
        check("pinned() reports one entry per worker", p.pinned().size() == 2);
        ColdPool::Config c2; c2.n_workers = 2;
        ColdPool p2(c2);
        check("no CPU list: nothing is pinned", p2.pinned()[0] == 0 && p2.pinned()[1] == 0);
    }

    std::printf(g_fail ? "\n%d FAILED\n" : "\nALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
