// File: <project>\moe_track\tests\test_e5_state.cpp
// Host test of pelias_serve/e5_deltanet_state.cuh: snapshot / rollback bookkeeping with real byte comparisons, the two slots, failure handling (allocation
// failure, refused registrations, copy failure), memory balance, the geometry/cost numbers for this model, and the CUDA wrapper compiled against lib/cuda_shim.h.
//
// Build:  g++ -O2 -std=c++17 -DPELIAS_CUDA_SHIM test_e5_state.cpp -I../pelias_serve -o test_e5_state        Run: ./test_e5_state        Exit 0 = pass.
#include "e5_deltanet_state.cuh"

#include <cmath>
#include <map>
#include <random>

using namespace pelias;

static int g_fail = 0;
static void check(const char * name, bool ok) { std::printf("%s %s\n", ok ? "ok  " : "FAIL", name); if (!ok) ++g_fail; }

struct CountingOps {                                   // HostOps that counts allocations and can be told to fail
    using Stream = void *;
    static inline int allocs = 0, frees = 0, fail_after = -1, copies = 0, fail_copy_after = -1;
    void * alloc(size_t n) { if (fail_after >= 0 && allocs >= fail_after) return nullptr; ++allocs; return std::malloc(n); }
    void release(void * p) { ++frees; std::free(p); }
    bool copy(void * d, const void * s, size_t n, Stream) { if (fail_copy_after >= 0 && copies >= fail_copy_after) return false; ++copies; std::memcpy(d, s, n); return true; }
    static void reset() { allocs = frees = copies = 0; fail_after = fail_copy_after = -1; }
};

struct Live {
    std::vector<std::vector<uint8_t>> rec, conv;
    std::vector<int> model_layer;
    Live(int n, uint32_t seed) {
        std::mt19937 rng(seed);
        int l = 0;
        for (int i = 0; i < n; ++i) {
            while (layer_is_full_attention(l)) ++l;
            model_layer.push_back(l++);
            rec.emplace_back(1000 + 37 * i); conv.emplace_back(96 + 5 * i);
        }
        fill(rng);
    }
    void fill(std::mt19937 & rng) { for (auto & v : rec) for (auto & b : v) b = static_cast<uint8_t>(rng()); for (auto & v : conv) for (auto & b : v) b = static_cast<uint8_t>(rng()); }
    void fill(uint32_t seed) { std::mt19937 rng(seed); fill(rng); }
    std::vector<uint8_t> flat() const { std::vector<uint8_t> o; for (auto & v : rec) o.insert(o.end(), v.begin(), v.end()); for (auto & v : conv) o.insert(o.end(), v.begin(), v.end()); return o; }
};

template <class M>
static bool register_all(M & m, Live & L) {
    for (size_t i = 0; i < L.rec.size(); ++i)
        if (!m.register_layer(static_cast<int>(i), L.model_layer[i], L.rec[i].data(), L.rec[i].size(), L.conv[i].data(), L.conv[i].size())) return false;
    return true;
}

int main() {
    // ---- bookkeeping with real bytes ----
    {
        CountingOps::reset();
        {
            Live L(30, 1);
            DeltaNetStateManager<CountingOps> m;
            check("30 layers register", register_all(m, L) && m.ready() && m.registered() == 30);
            size_t expect = 0;
            for (size_t i = 0; i < 30; ++i) expect += L.rec[i].size() + L.conv[i].size();
            check("bytes_per_slot adds up", m.bytes_per_slot() == expect);
            const auto A = L.flat();
            check("a slot never snapshotted rolls back to -1 and touches nothing", m.rollback(0, nullptr) == -1 && L.flat() == A);
            check("snapshot slot 0 at position 100", m.snapshot(0, 100, nullptr) && m.slot_pos(0) == 100);
            L.fill(2u);
            const auto B = L.flat();
            check("snapshot slot 1 (a different state) at position 120", m.snapshot(1, 120, nullptr) && m.slot_pos(1) == 120);
            L.fill(3u);
            check("rollback slot 0 restores state A exactly and reports position 100", m.rollback(0, nullptr) == 100 && L.flat() == A);
            check("rollback slot 1 restores state B exactly and reports position 120", m.rollback(1, nullptr) == 120 && L.flat() == B);
            check("a slot can be restored again after a rollback", m.rollback(1, nullptr) == 120 && L.flat() == B);
            check("slot 0 still holds A after slot 1 was used", m.rollback(0, nullptr) == 100 && L.flat() == A);
            m.invalidate(0);
            L.fill(4u);
            const auto C = L.flat();
            check("an invalidated slot is refused and leaves the live state alone", m.rollback(0, nullptr) == -1 && L.flat() == C);
            check("snapshot overwrites the older one in the same slot", m.snapshot(1, 130, nullptr) && m.rollback(1, nullptr) == 130 && L.flat() == C);
            check("bad slot numbers are refused", !m.snapshot(2, 1, nullptr) && !m.snapshot(-1, 1, nullptr) && m.rollback(2, nullptr) == -1 && m.slot_pos(5) == -1);
            check("a negative position is refused", !m.snapshot(0, -3, nullptr));
        }
        check("every allocation was released by the destructor", CountingOps::allocs == CountingOps::frees && CountingOps::allocs == 30 * 2 * 2);
    }
    // ---- registration rules ----
    {
        CountingOps::reset();
        Live L(3, 7);
        DeltaNetStateManager<CountingOps> m(3);
        check("not ready before all layers are registered; snapshot is refused", !m.ready() && !m.snapshot(0, 1, nullptr) && m.rollback(0, nullptr) == -1);
        check("register layer 0", m.register_layer(0, L.model_layer[0], L.rec[0].data(), L.rec[0].size(), L.conv[0].data(), L.conv[0].size()));
        check("the same index twice is refused", !m.register_layer(0, L.model_layer[0], L.rec[0].data(), L.rec[0].size(), L.conv[0].data(), L.conv[0].size()));
        check("an index out of range is refused", !m.register_layer(3, 4, L.rec[0].data(), 10, L.conv[0].data(), 10) && !m.register_layer(-1, 4, L.rec[0].data(), 10, L.conv[0].data(), 10));
        check("a full-attention layer (3, 7, 11, ...) is refused", !m.register_layer(1, 3, L.rec[1].data(), L.rec[1].size(), L.conv[1].data(), L.conv[1].size()));
        check("null pointers and zero sizes are refused", !m.register_layer(1, 1, nullptr, 10, L.conv[1].data(), 10) && !m.register_layer(1, 1, L.rec[1].data(), 0, L.conv[1].data(), 10));
        const int before = CountingOps::allocs - CountingOps::frees;
        CountingOps::fail_after = CountingOps::allocs + 3;                       // the 4th allocation of the next registration fails
        check("an allocation failure refuses the layer", !m.register_layer(1, L.model_layer[1], L.rec[1].data(), L.rec[1].size(), L.conv[1].data(), L.conv[1].size()));
        check("... and releases what that call had allocated", CountingOps::allocs - CountingOps::frees == before);
        CountingOps::fail_after = -1;
        check("a refused index can be registered again afterwards", m.register_layer(1, L.model_layer[1], L.rec[1].data(), L.rec[1].size(), L.conv[1].data(), L.conv[1].size()) &&
              m.register_layer(2, L.model_layer[2], L.rec[2].data(), L.rec[2].size(), L.conv[2].data(), L.conv[2].size()) && m.ready());
        CountingOps::copies = 0; CountingOps::fail_copy_after = 5;
        check("a copy failure during snapshot returns false and leaves the slot invalid", !m.snapshot(0, 10, nullptr) && m.slot_pos(0) == -1);
        CountingOps::fail_copy_after = -1;
        check("snapshot works again afterwards", m.snapshot(0, 10, nullptr) && m.slot_pos(0) == 10);
        CountingOps::copies = 0; CountingOps::fail_copy_after = 1;
        check("a copy failure during rollback returns -1", m.rollback(0, nullptr) == -1);
        CountingOps::reset();
    }
    // ---- numbers for this model ----
    {
        DeltaNetGeometry g;
        check("recurrent state per layer is 2 MiB (128 x 128 x 32 floats)", g.recurrent_bytes() == 2u * 1024 * 1024);
        check("conv tail per layer is 96 KiB (3 x 8192 floats)", g.conv_bytes() == 96u * 1024);
        const double mib = g.per_slot() / 1048576.0;
        check("one slot is about 62.9 MiB, not 48", std::fabs(mib - 62.8125) < 0.01);
        const double ms = estimate_copy_ms(g.per_slot(), 60);
        check("estimated snapshot time is about half a millisecond (0.26 ms bandwidth + 60 launches), not 0.095", ms > 0.4 && ms < 0.6);
        check("the draft's 0.095 ms would need more than the card's bandwidth for a read+write copy", estimate_copy_ms(48u << 20, 0) > 0.19);
    }
#ifdef PELIAS_E5_HAVE_CUDA
    // ---- the CUDA wrapper, compiled against the shim (copies are memcpy there) ----
    {
        Live L(30, 11);
        DeltaNetStateManager<CudaOps> m;
        check("CudaOps wrapper: registers 30 layers", register_all(m, L) && m.ready());
        const auto A = L.flat();
        cudaStream_t st = nullptr;
        check("CudaOps wrapper: snapshot, change, rollback", m.snapshot(0, 55, st) && (L.fill(99u), true) && m.rollback(0, st) == 55 && L.flat() == A);
    }
#endif
    std::printf(g_fail ? "\n%d FAILED\n" : "\nALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
