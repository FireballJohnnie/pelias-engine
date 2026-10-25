// File: <project>\moe_track\tests\test_miss_splitter.cpp
// Host test of pelias_serve/miss_splitter.h: every row is given to exactly one side, cuts sit on QK_K = 256 granules, at most one expert is cut in two, the
// long-run GPU share converges to the requested fraction (error diffusion), edge cases (no misses, f = 0 and 1, too many misses, n_ff not a multiple of the
// granule), determinism, and the exact sequence for one miss expert at 0.60. Also prints the share a single layer can actually hit, for this model's n_ff = 512.
//
// Build:  g++ -O2 -std=c++17 -Wall -Wextra -Werror test_miss_splitter.cpp -I../pelias_serve -o test_miss_splitter        Run: ./test_miss_splitter        Exit 0 = pass.
#include "miss_splitter.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace pelias;

static int g_fail = 0;
static void check(const char * name, bool ok) { std::printf("%s %s\n", ok ? "ok  " : "FAIL", name); if (!ok) ++g_fail; }

static bool plan_is_sound(const SplitPlan & p, int n_expected) {
    if (p.n_miss != n_expected) return false;
    int partial = 0;
    int64_t gpu = 0, cpu = 0;
    for (int i = 0; i < p.n_miss; ++i) {
        const int32_t s = p.task[i].split_row;
        if (s < 0 || s > p.n_ff) return false;
        if (s != p.n_ff && s % p.granule != 0) return false;                  // a cut is a multiple of the granule (or the end of the expert)
        if (s > 0 && s < p.n_ff) ++partial;
        if (i > 0) {                                                          // GPU side is filled front to back: no GPU rows after a CPU-only expert
            if (p.task[i].split_row > 0 && p.task[i - 1].split_row < p.n_ff) return false;
        }
        gpu += s;
        cpu += p.n_ff - s;
    }
    return partial <= 1 && gpu == p.gpu_rows && cpu == p.cpu_rows && gpu + cpu == static_cast<int64_t>(p.n_miss) * p.n_ff;
}

static std::vector<MissExpert> random_misses(std::mt19937 & rng, int n) {
    std::vector<MissExpert> m(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) { m[static_cast<size_t>(i)].expert = static_cast<int32_t>(rng() % N_EXPERTS); m[static_cast<size_t>(i)].weight = 0.05f + 0.01f * static_cast<float>(i); }
    return m;
}

int main() {
    std::mt19937 rng(12345);

    // ---- soundness over many random layers and several fractions
    {
        bool sound = true;
        SplitterState st;
        const SplitterConfig cfg = make_splitter_config(0.60f);
        for (int it = 0; it < 20000 && sound; ++it) {
            const int n = 1 + static_cast<int>(rng() % 8);
            const auto m = random_misses(rng, n);
            SplitPlan p;
            const int got = split_misses(cfg, &st, m.data(), n, &p);
            sound = got == n && plan_is_sound(p, n);
            for (int i = 0; i < n && sound; ++i) sound = p.task[i].expert == m[static_cast<size_t>(i)].expert && p.task[i].weight == m[static_cast<size_t>(i)].weight;
        }
        check("20000 random layers at 0.60: rows covered exactly once, cuts on granules, <= 1 cut expert, expert ids/weights carried", sound);
    }

    // ---- long-run share converges to the request
    {
        const float fracs[] = {0.0f, 0.10f, 0.4423f, 0.50f, 0.60f, 0.90f, 1.0f};
        bool ok = true;
        for (float f : fracs) {
            SplitterState st;
            const SplitterConfig cfg = make_splitter_config(f);
            int64_t gpu = 0, tot = 0;
            for (int it = 0; it < 200000; ++it) {
                const int n = 1 + static_cast<int>(rng() % 8);
                const auto m = random_misses(rng, n);
                SplitPlan p;
                split_misses(cfg, &st, m.data(), n, &p);
                gpu += p.gpu_rows;
                tot += static_cast<int64_t>(p.gpu_rows) + p.cpu_rows;
            }
            const double share = static_cast<double>(gpu) / static_cast<double>(tot);
            std::printf("     requested %.4f  ->  long-run GPU share %.5f\n", static_cast<double>(f), share);
            if (std::fabs(share - static_cast<double>(f)) > 1.5e-3) ok = false;
        }
        check("long-run GPU share within 0.0015 of the request (200000 layers each)", ok);
    }

    // ---- without carried state every call rounds alone: still sound
    {
        const SplitterConfig cfg = make_splitter_config(0.60f);
        const auto m = random_misses(rng, 3);
        SplitPlan p;
        split_misses(cfg, nullptr, m.data(), 3, &p);
        check("no state: sound plan; 3 experts x 2 granules at 0.60 -> 4 of 6 granules to the GPU (nearest whole granule)", plan_is_sound(p, 3) && p.gpu_rows == 4 * 256);
    }

    // ---- one miss expert at 0.60: the exact sequence (n_ff = 512 has two granules, so a single call can only give 0, 256 or 512 rows)
    {
        SplitterState st;
        const SplitterConfig cfg = make_splitter_config(0.60f);
        const MissExpert one{7, 0.3f};
        int seq[10];
        for (int i = 0; i < 10; ++i) { SplitPlan p; split_misses(cfg, &st, &one, 1, &p); seq[i] = p.gpu_rows; }
        std::printf("     one miss expert, GPU rows over 10 calls at 0.60:");
        for (int i = 0; i < 10; ++i) std::printf(" %d", seq[i]);
        std::printf("\n");
        const int want[5] = {256, 256, 512, 256, 256};
        bool ok = true;
        for (int i = 0; i < 5; ++i) ok = ok && seq[i] == want[i];
        check("one miss expert at 0.60: 256,256,512,256,256 ... (error diffusion), never 60/40 inside one call", ok);
    }

    // ---- what one layer can hit, for this model
    {
        std::printf("     achievable GPU share per call at 0.60 (n_ff = 512, granule 256), first call from a fresh state:\n");
        for (int n = 1; n <= 8; ++n) {
            SplitterState st;
            const auto m = random_misses(rng, n);
            SplitPlan p;
            split_misses(make_splitter_config(0.60f), &st, m.data(), n, &p);
            std::printf("       %d miss expert(s): GPU %d rows of %d = %.3f\n", n, p.gpu_rows, n * 512, static_cast<double>(p.gpu_rows) / (n * 512.0));
        }
    }

    // ---- edge cases
    {
        SplitPlan p;
        SplitterState st;
        const SplitterConfig cfg = make_splitter_config(0.60f);
        split_misses(cfg, &st, nullptr, 0, &p);
        check("no misses: empty plan, no CPU work, no handoff needed", p.n_miss == 0 && !p.needs_cpu() && !p.needs_gpu() && st.credit_q16 == 0);

        const auto m = random_misses(rng, 24);
        const int got = split_misses(cfg, &st, m.data(), 24, &p);
        check("24 misses are clipped to MAX_COLD (16)", got == MAX_COLD && plan_is_sound(p, MAX_COLD));

        SplitterState s0, s1;
        bool all_cpu = true, all_gpu = true;
        for (int i = 0; i < 1000; ++i) {
            const int n = 1 + static_cast<int>(rng() % 8);
            const auto mm = random_misses(rng, n);
            SplitPlan a, b;
            split_misses(make_splitter_config(0.0f), &s0, mm.data(), n, &a);
            split_misses(make_splitter_config(1.0f), &s1, mm.data(), n, &b);
            all_cpu = all_cpu && a.gpu_rows == 0 && !a.needs_gpu() && a.cpu_rows == n * 512;
            all_gpu = all_gpu && b.cpu_rows == 0 && !b.needs_cpu() && b.gpu_rows == n * 512;
        }
        check("fraction 0 -> everything on the CPU; fraction 1 -> everything on the GPU (no handoff needed)", all_cpu && all_gpu);

        check("gpu_frac_to_q16: 0, 1, NaN, 0.5", gpu_frac_to_q16(0.0f) == 0u && gpu_frac_to_q16(1.0f) == 65536u && gpu_frac_to_q16(std::nanf("")) == 0u && gpu_frac_to_q16(0.5f) == 32768u);
        const float fb = gpu_frac_from_bandwidths(21.06, 26.56);
        std::printf("     gpu_frac_from_bandwidths(21.06, 26.56) = %.4f   (the work order's default is %.2f)\n", static_cast<double>(fb), static_cast<double>(DEFAULT_GPU_FRAC));
        check("gpu_frac_from_bandwidths = gpu / (gpu + cpu)", std::fabs(fb - 21.06f / (21.06f + 26.56f)) < 1e-6f && gpu_frac_from_bandwidths(0.0, 10.0) == 0.0f);
        check("config validity", splitter_config_valid(make_splitter_config(0.6f)) && !splitter_config_valid(make_splitter_config(0.6f, 0, 256)) &&
                                    !splitter_config_valid(make_splitter_config(0.6f, 512, 0)));
    }

    // ---- n_ff that is not a multiple of the granule (not this model, but the code must not overrun)
    {
        SplitterState st;
        SplitterConfig cfg = make_splitter_config(0.75f, 600, 256);           // 3 granules per expert, the last one 88 rows
        bool ok = true;
        for (int it = 0; it < 5000 && ok; ++it) {
            const int n = 1 + static_cast<int>(rng() % 8);
            const auto m = random_misses(rng, n);
            SplitPlan p;
            split_misses(cfg, &st, m.data(), n, &p);
            ok = plan_is_sound(p, n) && p.gpu_rows + p.cpu_rows == n * 600;
            for (int i = 0; i < n && ok; ++i) ok = p.task[i].split_row <= 600;
        }
        check("n_ff = 600, granule 256: cuts never exceed n_ff, all rows covered", ok);
    }

    // ---- determinism: same config, same state, same input -> same plan
    {
        SplitterState a, b;
        const SplitterConfig cfg = make_splitter_config(0.60f);
        bool same = true;
        for (int it = 0; it < 5000 && same; ++it) {
            const int n = 1 + static_cast<int>(rng() % 8);
            const auto m = random_misses(rng, n);
            SplitPlan pa, pb;
            split_misses(cfg, &a, m.data(), n, &pa);
            split_misses(cfg, &b, m.data(), n, &pb);
            same = pa.gpu_rows == pb.gpu_rows && a.credit_q16 == b.credit_q16;
            for (int i = 0; i < n && same; ++i) same = pa.task[i].split_row == pb.task[i].split_row;
        }
        check("deterministic: two splitters fed the same layers agree", same);
    }

    // ---- the carried error stays bounded
    {
        SplitterState st;
        const SplitterConfig cfg = make_splitter_config(0.6f);
        int32_t worst = 0;
        for (int it = 0; it < 100000; ++it) {
            const int n = 1 + static_cast<int>(rng() % 8);
            const auto m = random_misses(rng, n);
            SplitPlan p;
            split_misses(cfg, &st, m.data(), n, &p);
            const int32_t a = st.credit_q16 < 0 ? -st.credit_q16 : st.credit_q16;
            if (a > worst) worst = a;
        }
        std::printf("     largest carried error: %.3f granules\n", static_cast<double>(worst) / 65536.0);
        check("carried error stays within one granule", worst <= 65536);
    }

    std::printf("%s\n", g_fail == 0 ? "ALL PASS" : "FAILED");
    return g_fail == 0 ? 0 : 1;
}
