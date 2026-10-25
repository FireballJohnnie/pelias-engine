// File: <project>\moe_track\tests\test_e0_profiler.cpp
// Host test of pelias_serve/e0_layer_profiler.cuh with a hand-driven clock: known stage durations in, the same numbers out (per layer, per token, in the JSON),
// incomplete layers skipped, no-overwrite on dump, and the CUDA back end compiled against lib/cuda_shim.h.
//
// Build:  g++ -O2 -std=c++17 -Wall -DPELIAS_CUDA_SHIM test_e0_profiler.cpp -I../pelias_serve -o test_e0_profiler        Run: ./test_e0_profiler [out.json]        Exit 0 = pass.
#include "e0_layer_profiler.cuh"

#include <cmath>
#include <cstdlib>

using namespace pelias;

static int g_fail = 0;
static void check(const char * name, bool ok) { std::printf("%s %s\n", ok ? "ok  " : "FAIL", name); if (!ok) ++g_fail; }
static bool near(double a, double b, double tol = 1e-9) { return std::fabs(a - b) <= tol; }

int main(int argc, char ** argv) {
    using P = LayerProfiler<HostClockBackend>;
    P prof;
    double t = 100.0;
    auto at = [&](double dt) { t += dt; HostClockBackend::now_override = t; };
    auto layer_marks = [&](int l, double attn, double rtr, double hot, double tail) {
        at(0.0);     prof.mark(l, E0_START, nullptr);
        at(attn);    prof.mark(l, E0_AFTER_ATTN, nullptr);
        at(rtr);     prof.mark(l, E0_AFTER_ROUTER, nullptr);
        at(hot);     prof.mark(l, E0_AFTER_HOT, nullptr);
        at(tail);    prof.mark(l, E0_END, nullptr);
    };
    const int tokens = 5;
    for (int k = 0; k < tokens; ++k) {
        for (int l = 0; l < N_LAYERS; ++l) {
            const bool full = layer_is_full_attention(l);
            layer_marks(l, full ? 0.20 : 0.06, 0.01, 0.12 + 0.001 * l, 0.05 + (l == 7 ? 0.5 : 0.0));
            prof.add_cpu_wait_ms(l, 0.30 + 0.01 * (k % 2));
        }
        prof.end_token(nullptr);
    }
    check("tokens counted", prof.tokens() == tokens);
    check("layer 0 (DeltaNet): attention 0.06 + router 0.01", near(prof.layer(0).attn_ms / tokens, 0.06, 1e-9) && near(prof.layer(0).router_ms / tokens, 0.01, 1e-9));
    check("layer 3 (full attention): attention 0.20", near(prof.layer(3).attn_ms / tokens, 0.20, 1e-9));
    check("layer 10: hot experts 0.12 + 0.001 * 10", near(prof.layer(10).hot_ms / tokens, 0.13, 1e-9));
    check("layer 7: tail includes the extra 0.5 ms wait", near(prof.layer(7).tail_ms / tokens, 0.55, 1e-9));
    check("cpu wait is accumulated per layer and token (0.30 / 0.31 alternating -> mean 0.304)", near(prof.layer(5).cold_wait_ms / tokens, 0.304, 1e-9));
    check("gpu span equals START..END", near(prof.layer(5).total_gpu_ms / tokens, 0.06 + 0.01 + 0.125 + 0.05, 1e-9));

    const std::string js = prof.json();
    auto grab = [&](const std::string & key) { const size_t p = js.rfind("\"" + key + "\": "); return p == std::string::npos ? -1.0 : std::atof(js.c_str() + p + key.size() + 4); };
    double spine = 0, hot = 0;
    for (int l = 0; l < N_LAYERS; ++l) { const bool full = layer_is_full_attention(l); spine += (full ? 0.20 : 0.06) + 0.01; hot += 0.12 + 0.001 * l; }
    check("JSON totals: spine", near(grab("spine_total_ms"), spine, 1e-6));
    check("JSON totals: hot gpu", near(grab("hot_gpu_total_ms"), hot, 1e-6));
    check("JSON totals: launch gap is ~0 when the stage marks are contiguous", std::fabs(grab("launch_gap_total_ms")) < 1e-9);
    check("JSON mentions 40 layers and the full-attention flags", js.find("\"layer\": 39") != std::string::npos && js.find("\"layer\": 3, \"is_full_attn\": true") != std::string::npos &&
          js.find("\"layer\": 2, \"is_full_attn\": false") != std::string::npos);
    check("JSON braces and brackets balance", [&] { int b = 0, k = 0; for (char c : js) { b += (c == '{') - (c == '}'); k += (c == '[') - (c == ']'); } return b == 0 && k == 0; }());

    // a layer that only got START and END in a token (marks missing in between)
    P p3;
    for (int l = 0; l < N_LAYERS; ++l) {
        if (l == 4) { HostClockBackend::now_override = 1.0; p3.mark(l, E0_START, nullptr); HostClockBackend::now_override = 2.0; p3.mark(l, E0_END, nullptr); }
        else { HostClockBackend::now_override = 1.0; p3.mark(l, E0_START, nullptr); HostClockBackend::now_override = 1.1; p3.mark(l, E0_AFTER_ATTN, nullptr);
               p3.mark(l, E0_AFTER_ROUTER, nullptr); p3.mark(l, E0_AFTER_HOT, nullptr); p3.mark(l, E0_END, nullptr); }
    }
    p3.add_cpu_wait_ms(4, 9.0);
    p3.end_token(nullptr);
    check("a layer with incomplete marks is skipped and counted, others are recorded", p3.layer(4).skipped == 1 && p3.layer(4).tokens == 0 && p3.layer(5).tokens == 1);
    check("a skipped layer's pending CPU wait does not leak into the next token", [&] { p3.end_token(nullptr); return near(p3.layer(4).cold_wait_ms, 0.0); }());
    p3.mark(99, E0_START, nullptr); p3.mark(-1, E0_END, nullptr); p3.add_cpu_wait_ms(77, 1.0);
    check("out-of-range layers are ignored", true);

    // dump never overwrites
    const std::string path = argc > 1 ? argv[1] : "/tmp/e0_profile_test.json";
    std::remove(path.c_str());
    check("dump_json writes a new file", prof.dump_json(path));
    check("dump_json refuses to overwrite", !prof.dump_json(path));

#ifdef PELIAS_E0_HAVE_CUDA
    {   // the CUDA back end compiled against the shim (timings are 0 there; this checks the wiring)
        LayerProfiler<CudaBackend> cp;
        for (int l = 0; l < N_LAYERS; ++l) for (int m = 0; m < E0_N_MARKS; ++m) cp.mark(l, static_cast<E0Mark>(m), nullptr);
        cp.end_token(nullptr);
        check("CudaBackend: wiring compiles and runs against the shim", cp.tokens() == 1 && cp.layer(0).tokens == 1);
    }
#endif
    HostClockBackend::now_override = -1.0;
    std::printf(g_fail ? "\n%d FAILED\n" : "\nALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
