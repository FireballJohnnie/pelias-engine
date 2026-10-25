// File: <project>\moe_track\tests\test_cold_weights_view.cpp
// Host test of pelias_serve/cold_weights_view.{h,cpp}. It writes its own GGUF files (the writer here is independent of the reader: the expected offsets, sizes and
// bytes come from the writer, not from the code under test) and checks:
//   * the canonical header parse: v2/v3, general.alignment (64 and the default 32), metadata of every type skipped correctly (strings, arrays, nested arrays),
//     tensor offsets = data_start + stored offset, sizes from the quantisation geometry
//   * the view: tensors found by NAME (file order scrambled), every base 2 MiB aligned, whole 2 MiB pages, zero padding, bytes identical to the file, ColdMat
//     addressing (row/expert), layer ranges, the MTP block (blk.40) only on request, strict_shape, any load_threads, verify_against_file catches a flipped byte
//   * 30+ ways a file can be wrong (truncated at every position of the header, flipped header bytes, bad counts, offsets past the end, missing / mismatched /
//     unknown-type expert tensors): a clean error, nothing left allocated, no crash (build with -fsanitize=address,undefined to prove it)
//   * pin(): in a build without CUDA it says so; with -DPELIAS_CUDA_SHIM it registers every region, gives each matrix its GPU pointer (also when that differs
//     from the CPU address), honours max_bytes, survives a driver failure half-way (registration or device-pointer lookup) and unregisters on close()
// Optional real-file mode:  ./test_cold_weights_view --gguf FILE.gguf [--first N] [--last N] [--no-strict] [--threads N] [--verify] [--pin [MiB]] [--gpu-read]
//   --pin registers the loaded layers with the CUDA driver (cudaHostRegister Mapped|Portable) and reports regions ok / failed, GiB registered, the time it took,
//   whether the device address equals the host address and whether the difference is constant (information only: each pinned matrix gets its own GPU
//   pointer, dev_base, so a different address is not an error); MiB limits how much is registered (default: all loaded layers). Needs a CUDA build.
//
//   --gpu-read (needs --pin and the GPU-check build) runs a kernel over every pinned matrix through dev_base and over sampled rows through dev_row, compares
//   checksums with the CPU's, and reports the read speed of repeated passes (measured; no limit applied).
//
// Build:  g++ -O2 -std=c++17 -pthread -Wall -Wextra -Werror test_cold_weights_view.cpp ../pelias_serve/cold_weights_view.cpp -I../pelias_serve -o test_cold_weights_view
//         add -DPELIAS_CUDA_SHIM for the pin() tests            add -fsanitize=address,undefined -g for the memory-safety run
//         CUDA build for --pin (WSL):  R=$HOME/stuart_creator_engine/cuda_shim
//             $R/bin/nvcc -O2 -std=c++17 -arch=sm_89 -DPELIAS_COLD_VIEW_CUDA -I$R/include --cudart shared -L$R/lib64 -Xlinker -rpath,$R/lib64 -Xcompiler -pthread test_cold_weights_view.cpp ../pelias_serve/cold_weights_view.cpp -I../pelias_serve -o test_cold_weights_view_cuda
// Run:    ./test_cold_weights_view        Exit 0 = pass.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

#include "cold_weights_view.h"
#if defined(PELIAS_CUDA_SHIM)
#include "../lib/cuda_shim.h"      // after the standard headers: the shim defines __noinline__ as a macro
#endif
#if defined(PELIAS_COLD_VIEW_GPU_CHECK)
#include <atomic>
#include <thread>
#include "../lib/pl_cuda.h"        // PL_LAUNCH / PL_HD: real CUDA under nvcc, the CPU stand-in otherwise
#endif

using namespace pelias;

static int g_fail = 0;
static void check(const std::string& name, bool ok, const std::string& extra = std::string()) {
    std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", name.c_str(), (!ok && !extra.empty()) ? "   -> " : "", (!ok && !extra.empty()) ? extra.c_str() : "");
    if (!ok) ++g_fail;
}
static bool contains(const std::string& s, const char* sub) { return s.find(sub) != std::string::npos; }

// ---------------------------------------------------------------------------------------------------------------------------------------------------------
// an independent GGUF writer
// ---------------------------------------------------------------------------------------------------------------------------------------------------------
struct Bytes {
    std::vector<uint8_t> b;
    template <class T> void put(T v) { const size_t n = b.size(); b.resize(n + sizeof(T)); std::memcpy(b.data() + n, &v, sizeof(T)); }
    void str(const std::string& s) { put<uint64_t>(s.size()); b.insert(b.end(), s.begin(), s.end()); }
};

struct STensor {
    std::string          name;
    uint32_t             type = 0;
    std::vector<uint64_t> dims;
    std::vector<uint8_t> data;
    uint64_t             file_off = 0;        // set by the writer
};

// bytes per element-block for the types the writer knows (independent of the reader's table)
static void geom(uint32_t type, uint64_t* blck, uint64_t* bytes) {
    switch (type) {
        case 0: *blck = 1; *bytes = 4; break;        // F32
        case 1: *blck = 1; *bytes = 2; break;        // F16
        case 8: *blck = 32; *bytes = 34; break;      // Q8_0
        case 12: *blck = 256; *bytes = 144; break;   // Q4_K
        case 14: *blck = 256; *bytes = 210; break;   // Q6_K
        default: *blck = 1; *bytes = 16; break;      // unknown type: pretend 16 bytes in total (see make_tensor)
    }
}

static uint64_t splitmix(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static STensor make_tensor(const std::string& name, uint32_t type, std::vector<uint64_t> dims) {
    STensor t;
    t.name = name;
    t.type = type;
    t.dims = dims;
    uint64_t blck, bytes;
    geom(type, &blck, &bytes);
    uint64_t elems = 1;
    for (uint64_t d : dims) elems *= d;
    const uint64_t n = (type == 0 || type == 1 || type == 8 || type == 12 || type == 14) ? elems / blck * bytes : 16;
    t.data.resize(static_cast<size_t>(n));
    uint64_t seed = std::hash<std::string>()(name) ^ 0x1234567ull;
    for (size_t i = 0; i < t.data.size(); i += 8) {
        const uint64_t v = splitmix(seed);
        std::memcpy(t.data.data() + i, &v, std::min<size_t>(8, t.data.size() - i));
    }
    return t;
}

struct SynthSpec {
    uint32_t version = 3;
    uint32_t alignment = 64;
    bool     alignment_kv = true;
    std::string arch = "qwen35moe";
    std::vector<STensor> tensors;
};

struct Written {
    std::vector<uint8_t> file;
    uint64_t             data_start = 0;
    std::vector<size_t>  pos_of_offset;      // where each tensor's stored offset lives in the file
    std::vector<size_t>  pos_of_type;
    std::vector<size_t>  pos_of_ndims;
};

static uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

static Written write_gguf(SynthSpec& sp) {
    Written w;
    Bytes h;
    h.b = {'G', 'G', 'U', 'F'};
    h.put<uint32_t>(sp.version);
    h.put<uint64_t>(sp.tensors.size());
    const size_t pos_nkv = h.b.size();
    h.put<uint64_t>(0);
    uint64_t nkv = 0;
    auto kv_head = [&](const std::string& k, uint32_t t) { h.str(k); h.put<uint32_t>(t); ++nkv; };
    kv_head("general.architecture", 8); h.str(sp.arch);
    kv_head("general.name", 8); h.str("synthetic test model");
    kv_head("some.u8", 0); h.put<uint8_t>(7);
    kv_head("some.i8", 1); h.put<int8_t>(-3);
    kv_head("some.u16", 2); h.put<uint16_t>(4000);
    kv_head("some.i16", 3); h.put<int16_t>(-4000);
    if (sp.alignment_kv) { kv_head("general.alignment", 4); h.put<uint32_t>(sp.alignment); }
    kv_head("some.i32", 5); h.put<int32_t>(-70000);
    kv_head("some.f32", 6); h.put<float>(3.5f);
    kv_head("some.bool", 7); h.put<uint8_t>(1);
    kv_head("some.u64", 10); h.put<uint64_t>(1ull << 40);
    kv_head("some.i64", 11); h.put<int64_t>(-(1ll << 40));
    kv_head("some.f64", 12); h.put<double>(2.25);
    kv_head("tokenizer.ggml.tokens", 9); h.put<uint32_t>(8); h.put<uint64_t>(60);
    for (int i = 0; i < 60; ++i) h.str("tok" + std::to_string(i * 977));
    kv_head("tokenizer.ggml.scores", 9); h.put<uint32_t>(6); h.put<uint64_t>(60);
    for (int i = 0; i < 60; ++i) h.put<float>(static_cast<float>(i));
    kv_head("some.array_of_arrays", 9); h.put<uint32_t>(9); h.put<uint64_t>(3);          // array of 3 arrays of u16
    for (int a = 0; a < 3; ++a) { h.put<uint32_t>(2); h.put<uint64_t>(5); for (int i = 0; i < 5; ++i) h.put<uint16_t>(static_cast<uint16_t>(a * 10 + i)); }
    kv_head("some.empty_array", 9); h.put<uint32_t>(4); h.put<uint64_t>(0);
    kv_head("some.string_array_empty_strings", 9); h.put<uint32_t>(8); h.put<uint64_t>(3); h.str(""); h.str(""); h.str("x");
    std::memcpy(h.b.data() + pos_nkv, &nkv, 8);

    // tensor infos: stored offsets are relative to the data start and each tensor starts on the alignment
    uint64_t rel = 0;
    std::vector<uint64_t> rels;
    for (const STensor& t : sp.tensors) {
        rels.push_back(rel);
        rel = align_up(rel + t.data.size(), sp.alignment);
    }
    for (size_t i = 0; i < sp.tensors.size(); ++i) {
        const STensor& t = sp.tensors[i];
        h.str(t.name);
        w.pos_of_ndims.push_back(h.b.size());
        h.put<uint32_t>(static_cast<uint32_t>(t.dims.size()));
        for (uint64_t d : t.dims) h.put<uint64_t>(d);
        w.pos_of_type.push_back(h.b.size());
        h.put<uint32_t>(t.type);
        w.pos_of_offset.push_back(h.b.size());
        h.put<uint64_t>(rels[i]);
    }
    w.data_start = align_up(h.b.size(), sp.alignment);
    h.b.resize(static_cast<size_t>(w.data_start), 0);
    for (size_t i = 0; i < sp.tensors.size(); ++i) {
        sp.tensors[i].file_off = w.data_start + rels[i];
        h.b.resize(static_cast<size_t>(w.data_start + rels[i]), 0);
        h.b.insert(h.b.end(), sp.tensors[i].data.begin(), sp.tensors[i].data.end());
    }
    h.b.resize(static_cast<size_t>(align_up(h.b.size(), sp.alignment)), 0);
    w.file = h.b;
    return w;
}

static std::string g_dir;
static bool save(const std::string& path, const std::vector<uint8_t>& b, size_t n = SIZE_MAX) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const size_t len = std::min(n, b.size());
    const bool ok = len == 0 || std::fwrite(b.data(), 1, len, f) == len;
    std::fclose(f);
    return ok;
}

// a model with expert tensors the way the real file has them (separate gate/up, mixed Q4_K / Q6_K down), in scrambled file order
static SynthSpec make_model(int n_layers, uint64_t hidden, uint64_t ff, uint64_t n_exp, uint32_t alignment = 64) {
    SynthSpec sp;
    sp.alignment = alignment;
    sp.tensors.push_back(make_tensor("token_embd.weight", 1, {hidden, 32}));
    sp.tensors.push_back(make_tensor("output_norm.weight", 0, {hidden}));
    for (int l = 0; l < n_layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        sp.tensors.push_back(make_tensor(p + "attn_norm.weight", 0, {hidden}));
        sp.tensors.push_back(make_tensor(p + "ffn_down_exps.weight", (l % 2 == 0) ? 14u : 12u, {ff, hidden, n_exp}));
        sp.tensors.push_back(make_tensor(p + "ffn_gate_shexp.weight", 1, {hidden, 64}));
        sp.tensors.push_back(make_tensor(p + "ffn_up_exps.weight", 12, {hidden, ff, n_exp}));
        sp.tensors.push_back(make_tensor(p + "ffn_gate_inp.weight", 0, {hidden, n_exp}));
        sp.tensors.push_back(make_tensor(p + "ffn_gate_exps.weight", 12, {hidden, ff, n_exp}));
    }
    return sp;
}

static const STensor* find_t(const SynthSpec& sp, const std::string& name) {
    for (const STensor& t : sp.tensors) if (t.name == name) return &t;
    return nullptr;
}

static std::string tname(int l, const char* which) { return "blk." + std::to_string(l) + ".ffn_" + which + "_exps.weight"; }

// ---------------------------------------------------------------------------------------------------------------------------------------------------------
static ColdViewOptions opts(const std::string& path) {
    ColdViewOptions o;
    o.gguf_path = path;
    o.strict_shape = false;
    o.load_threads = 3;
    return o;
}

static bool view_matches(const ColdWeightsView& v, const SynthSpec& sp, std::string* why) {
    for (int l : v.layers()) {
        const ColdLayerView& lv = v.layer(l);
        const ColdMat* mats[3] = {&lv.gate, &lv.up, &lv.down};
        const char* names[3] = {"gate", "up", "down"};
        for (int w = 0; w < 3; ++w) {
            const STensor* t = find_t(sp, tname(l, names[w]));
            if (!t) { *why = "reference tensor missing"; return false; }
            const ColdMat& m = *mats[w];
            if (reinterpret_cast<uintptr_t>(m.base) % COLD_HUGE_PAGE != 0) { *why = "blk." + std::to_string(l) + " " + names[w] + ": base not 2 MiB aligned"; return false; }
            if (std::memcmp(m.base, t->data.data(), t->data.size()) != 0) { *why = "blk." + std::to_string(l) + " " + names[w] + ": bytes differ from what was written"; return false; }
            const size_t padded = (t->data.size() + COLD_HUGE_PAGE - 1) / COLD_HUGE_PAGE * COLD_HUGE_PAGE;
            for (size_t i = t->data.size(); i < padded; ++i)
                if (m.base[i] != 0) { *why = "blk." + std::to_string(l) + " " + names[w] + ": padding is not zero"; return false; }
            if (m.type_id != static_cast<int32_t>(t->type) || m.cols != static_cast<int64_t>(t->dims[0]) || m.rows != static_cast<int64_t>(t->dims[1]) || m.n_experts != static_cast<int64_t>(t->dims[2])) {
                *why = "blk." + std::to_string(l) + " " + names[w] + ": type/shape fields wrong";
                return false;
            }
            uint64_t blck, bytes;
            geom(t->type, &blck, &bytes);
            if (m.row_bytes != static_cast<size_t>(t->dims[0] / blck * bytes) || m.expert_stride != m.row_bytes * static_cast<size_t>(m.rows)) { *why = "row_bytes / expert_stride wrong"; return false; }
            for (int e = 0; e < static_cast<int>(m.n_experts); ++e) {
                const uint8_t* want = t->data.data() + static_cast<size_t>(e) * m.expert_stride;
                if (m.expert(e) != m.base + static_cast<size_t>(e) * m.expert_stride || std::memcmp(m.expert(e), want, m.expert_stride) != 0) { *why = "expert() addressing wrong"; return false; }
                for (int64_t r : {int64_t(0), m.rows / 2, m.rows - 1})
                    if (m.row(e, r) != want - t->data.data() + m.base + static_cast<size_t>(r) * m.row_bytes || std::memcmp(m.row(e, r), want + static_cast<size_t>(r) * m.row_bytes, m.row_bytes) != 0) {
                        *why = "row() addressing wrong";
                        return false;
                    }
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------
// GPU read check: does a GPU kernel, reading through each matrix's dev_base / dev_row(), see exactly the bytes the CPU sees through base / row()?
// Only compiled with -DPELIAS_COLD_VIEW_GPU_CHECK, and then this file must be compiled as CUDA (nvcc -x cu). With -DPELIAS_CUDA_SHIM the same kernels run serially
// on the CPU, which tests the checking logic but says nothing about a GPU.
//   * whole tensor: every 16-byte item is hashed together with its index and summed (order-independent, so any thread layout gives the same number); the GPU sum
//     must equal the CPU sum over the same bytes. The padding up to the next 16 bytes is zero on both sides.
//   * sampled rows: 3 experts x 3 rows per matrix, addressed with dev_row() and compared with row(), byte by byte (hashed with the byte index).
//   * speed: after the first (correctness) pass the whole set is read again `reps` times, timed with CUDA events. The figure includes the hash arithmetic, so it
//     is "checksum-read GB/s", not a pure copy speed; the 07 probe used the same kind of kernel. No limit is applied to it.
// ---------------------------------------------------------------------------------------------------------------------------------------------------------
#if defined(PELIAS_COLD_VIEW_GPU_CHECK)
#define GC_GOLD 0x9E3779B97F4A7C15ULL
PL_HD static inline unsigned long long gc_mix(unsigned long long x) {
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}
PL_HD static inline unsigned long long gc_item(unsigned long long i, unsigned long long lo, unsigned long long hi) {
    return gc_mix(lo + GC_GOLD * (2ULL * i + 1ULL)) + gc_mix(hi + GC_GOLD * (2ULL * i + 2ULL));
}
PL_HD static inline unsigned long long gc_byte(unsigned long long j, unsigned long long b) { return gc_mix(b + GC_GOLD * (j + 1ULL)); }

__global__ void k_tensor_sum(const uint4* __restrict__ p, size_t n16, unsigned long long* out) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    unsigned long long s = 0;
    for (size_t i = idx; i < n16; i += stride) {
        const uint4 v = p[i];
        s += gc_item(i, (unsigned long long)v.x | ((unsigned long long)v.y << 32), (unsigned long long)v.z | ((unsigned long long)v.w << 32));
    }
    atomicAdd(&out[blockIdx.x & 63u], s);       // 64 counters instead of one: less contention
}

__global__ void k_row_sum(const uint8_t* const* __restrict__ rows, size_t row_bytes, unsigned long long* out) {
    const uint8_t* r = rows[blockIdx.x];
    unsigned long long s = 0;
    for (size_t j = threadIdx.x; j < row_bytes; j += blockDim.x) s += gc_byte(j, r[j]);
    atomicAdd(&out[blockIdx.x], s);
}

struct GpuReadResult {
    size_t      matrices = 0, bytes = 0, mismatched = 0, rows_checked = 0, rows_bad = 0;
    double      cpu_seconds = 0.0;
    std::vector<double> gbs;                   // one per timed pass
    std::string first_mismatch, first_bad_row, first_error;
};

static bool gpu_read_check(const ColdWeightsView& v, int threads, int reps, GpuReadResult* res) {
    struct Item { int layer; const char* name; const ColdMat* m; size_t n16; unsigned long long cpu, gpu; };
    std::vector<Item> items;
    static const char* const kNames[3] = {"gate", "up", "down"};
    for (int l : v.layers()) {
        const ColdLayerView& lv = v.layer(l);
        const ColdMat* ms[3] = {&lv.gate, &lv.up, &lv.down};
        for (int w = 0; w < 3; ++w)
            if (ms[w]->gpu_ok) items.push_back({l, kNames[w], ms[w], (static_cast<size_t>(ms[w]->n_experts) * ms[w]->expert_stride + 15) / 16, 0, 0});
    }
    if (items.empty()) { res->first_error = "no matrix is gpu_ok (pin first)"; return false; }
    for (const Item& it : items)
        if (reinterpret_cast<uintptr_t>(it.m->dev_base) % 16 != 0) { res->first_error = "blk." + std::to_string(it.layer) + " " + it.name + ": dev_base is not 16-byte aligned"; return false; }

    // CPU side
    const auto t0 = std::chrono::steady_clock::now();
    std::atomic<size_t> next{0};
    auto work = [&]() {
        for (;;) {
            const size_t k = next.fetch_add(1);
            if (k >= items.size()) return;
            unsigned long long s = 0;
            const uint8_t* p = items[k].m->base;
            for (size_t i = 0; i < items[k].n16; ++i) {
                unsigned long long lo, hi;
                std::memcpy(&lo, p + 16 * i, 8);
                std::memcpy(&hi, p + 16 * i + 8, 8);
                s += gc_item(i, lo, hi);
            }
            items[k].cpu = s;
        }
    };
    std::vector<std::thread> th;
    for (int i = 1; i < std::max(1, threads); ++i) th.emplace_back(work);
    work();
    for (auto& t : th) t.join();
    res->cpu_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // GPU side
    const cudaStream_t st = nullptr;
    unsigned long long* d_out = nullptr;
    const uint8_t** d_ptrs = nullptr;
    unsigned long long* d_rsum = nullptr;
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    constexpr int kRows = 9;
    bool good = false;
    do {
        cudaError_t e = cudaMalloc(&d_out, 64 * sizeof(unsigned long long));
        if (e == cudaSuccess) e = cudaMalloc(&d_ptrs, kRows * sizeof(const uint8_t*));
        if (e == cudaSuccess) e = cudaMalloc(&d_rsum, kRows * sizeof(unsigned long long));
        if (e == cudaSuccess) e = cudaEventCreate(&ev0);
        if (e == cudaSuccess) e = cudaEventCreate(&ev1);
        if (e != cudaSuccess) { res->first_error = std::string("setup: ") + cudaGetErrorString(e); break; }

        bool stop = false;
        for (Item& it : items) {                                               // pass 0: correctness, whole tensors
            (void)cudaMemset(d_out, 0, 64 * sizeof(unsigned long long));
            PL_LAUNCH(k_tensor_sum, 256, 256, 0, st, reinterpret_cast<const uint4*>(it.m->dev_base), it.n16, d_out);
            unsigned long long h[64];
            e = cudaGetLastError();
            if (e == cudaSuccess) e = cudaMemcpy(h, d_out, sizeof h, cudaMemcpyDeviceToHost);
            if (e != cudaSuccess) { res->first_error = "blk." + std::to_string(it.layer) + " " + it.name + ": GPU read failed: " + cudaGetErrorString(e); stop = true; break; }
            it.gpu = 0;
            for (unsigned long long x : h) it.gpu += x;
            ++res->matrices;
            res->bytes += static_cast<size_t>(it.m->n_experts) * it.m->expert_stride;
            if (it.gpu != it.cpu) { ++res->mismatched; if (res->first_mismatch.empty()) res->first_mismatch = "blk." + std::to_string(it.layer) + " " + it.name; }
        }
        if (stop) break;

        for (Item& it : items) {                                               // sampled rows through dev_row()
            const ColdMat& m = *it.m;
            const int64_t es[3] = {0, m.n_experts / 2, m.n_experts - 1}, rs[3] = {0, m.rows / 2, m.rows - 1};
            const uint8_t* dp[kRows];
            unsigned long long want[kRows];
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b) {
                    const int k = a * 3 + b;
                    dp[k] = m.dev_row(static_cast<int>(es[a]), rs[b]);
                    const uint8_t* cp = m.row(static_cast<int>(es[a]), rs[b]);
                    unsigned long long s = 0;
                    for (size_t j = 0; j < m.row_bytes; ++j) s += gc_byte(j, cp[j]);
                    want[k] = s;
                }
            unsigned long long got[kRows];
            e = cudaMemcpy(d_ptrs, dp, sizeof dp, cudaMemcpyHostToDevice);
            if (e == cudaSuccess) e = cudaMemset(d_rsum, 0, sizeof got);
            if (e == cudaSuccess) {
                PL_LAUNCH(k_row_sum, kRows, 128, 0, st, d_ptrs, m.row_bytes, d_rsum);
                e = cudaGetLastError();
            }
            if (e == cudaSuccess) e = cudaMemcpy(got, d_rsum, sizeof got, cudaMemcpyDeviceToHost);
            if (e != cudaSuccess) { res->first_error = "blk." + std::to_string(it.layer) + " " + it.name + ": GPU row read failed: " + cudaGetErrorString(e); stop = true; break; }
            for (int k = 0; k < kRows; ++k) {
                ++res->rows_checked;
                if (got[k] != want[k]) {
                    ++res->rows_bad;
                    if (res->first_bad_row.empty()) res->first_bad_row = "blk." + std::to_string(it.layer) + " " + it.name + " expert " + std::to_string(es[k / 3]) + " row " + std::to_string(rs[k % 3]);
                }
            }
        }
        if (stop) break;

        for (int rep = 0; rep < reps; ++rep) {                                 // timed passes
            (void)cudaEventRecord(ev0, st);
            for (Item& it : items) PL_LAUNCH(k_tensor_sum, 256, 256, 0, st, reinterpret_cast<const uint4*>(it.m->dev_base), it.n16, d_out);
            (void)cudaEventRecord(ev1, st);
            e = cudaEventSynchronize(ev1);
            if (e == cudaSuccess) e = cudaGetLastError();
            if (e != cudaSuccess) { res->first_error = std::string("timed pass failed: ") + cudaGetErrorString(e); stop = true; break; }
            float ms = 0.f;
            (void)cudaEventElapsedTime(&ms, ev0, ev1);
            if (ms > 0.f) res->gbs.push_back(static_cast<double>(res->bytes) / 1e9 / (static_cast<double>(ms) / 1e3));
        }
        good = !stop;
    } while (false);
    if (d_out) (void)cudaFree(d_out);
    if (d_ptrs) (void)cudaFree(d_ptrs);
    if (d_rsum) (void)cudaFree(d_rsum);
    if (ev0) (void)cudaEventDestroy(ev0);
    if (ev1) (void)cudaEventDestroy(ev1);
    return good;
}
#endif   // PELIAS_COLD_VIEW_GPU_CHECK

int main(int argc, char** argv) {
    // ---- optional real-file mode arguments
    std::string real_path;
    int real_first = 0, real_last = -1, real_threads = 4;
    bool real_strict = true, real_verify = false, real_pin = false, real_gpu = false;
    unsigned long long real_pin_mib = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--gguf" && i + 1 < argc) real_path = argv[++i];
        else if (a == "--first" && i + 1 < argc) real_first = std::atoi(argv[++i]);
        else if (a == "--last" && i + 1 < argc) real_last = std::atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) real_threads = std::atoi(argv[++i]);
        else if (a == "--no-strict") real_strict = false;
        else if (a == "--verify") real_verify = true;
        else if (a == "--gpu-read") real_gpu = true;
        else if (a == "--pin") {
            real_pin = true;
            if (i + 1 < argc && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9') real_pin_mib = std::strtoull(argv[++i], nullptr, 10);
        }
    }

    char tmpl[] = "/tmp/cwv_test_XXXXXX";
    if (!mkdtemp(tmpl)) { std::printf("FAIL cannot create a temp directory\n"); return 1; }
    g_dir = tmpl;
    const std::string P = g_dir + "/m.gguf";

    check(std::string("build: ") + (cold_view_built_with_cuda() ? "CUDA calls compiled in (shim or runtime)" : "no CUDA calls (pin() reports that)"), true);

    // ---- geometry table, spot values ----------------------------------------------------------------------------------------------------------------------
    {
        const QuantGeom* q4 = quant_geom(12);
        const QuantGeom* q6 = quant_geom(14);
        const QuantGeom* f16 = quant_geom(1);
        const QuantGeom* f32 = quant_geom(0);
        const QuantGeom* q80 = quant_geom(8);
        check("quant_geom: Q4_K 256 elements / 144 bytes, Q6_K 256 / 210, F16 1 / 2, F32 1 / 4, Q8_0 32 / 34, unknown id -> nullptr",
              q4 && q4->block_elems == 256 && q4->block_bytes == 144 && q6 && q6->block_elems == 256 && q6->block_bytes == 210 && f16 && f16->block_bytes == 2 && f32 && f32->block_bytes == 4 &&
                  q80 && q80->block_elems == 32 && q80->block_bytes == 34 && quant_geom(9999) == nullptr);
    }

    // ---- the header parse ----------------------------------------------------------------------------------------------------------------------------------
    {
        SynthSpec sp = make_model(3, 512, 256, 3, 64);
        const Written w = write_gguf(sp);
        check("fixture: data_start is not a multiple of 2 MiB (so the view must copy, not map)", w.data_start % COLD_HUGE_PAGE != 0 && save(P, w.file));
        GgufTable t;
        std::string err;
        const bool ok = parse_gguf_table(P, &t, &err);
        check("parse: v3, alignment 64 from general.alignment, architecture, tensor count, data_start", ok && t.version == 3 && t.alignment == 64 && t.architecture == "qwen35moe" &&
                                                                                                         t.tensors.size() == sp.tensors.size() && t.data_start == w.data_start && t.file_size == w.file.size(), err);
        bool all = ok;
        for (const STensor& st : sp.tensors) {
            const GgufTensorInfo* g = t.find(st.name);
            if (!g || g->file_offset != st.file_off || g->n_bytes != st.data.size() || g->type_id != static_cast<int32_t>(st.type) || g->n_dims != static_cast<int32_t>(st.dims.size()) ||
                !std::equal(st.dims.begin(), st.dims.end(), g->dims))
                all = false;
        }
        check("parse: every tensor's name, type, dims, file offset (= data_start + stored offset) and byte size match what the writer wrote", all);
        check("parse: find() of a missing name is nullptr", t.find("nope") == nullptr);
    }
    {
        SynthSpec sp = make_model(2, 512, 256, 3, 32);
        sp.alignment_kv = false;
        const Written w = write_gguf(sp);
        save(P, w.file);
        GgufTable t;
        std::string err;
        check("parse: no general.alignment -> default 32", parse_gguf_table(P, &t, &err) && t.alignment == 32 && t.data_start == w.data_start && t.find("blk.1.ffn_up_exps.weight")->file_offset == find_t(sp, "blk.1.ffn_up_exps.weight")->file_off, err);
        sp.version = 2;
        const Written w2 = write_gguf(sp);
        save(P, w2.file);
        check("parse: version 2 file", parse_gguf_table(P, &t, &err) && t.version == 2, err);
        sp = make_model(2, 512, 256, 3, 256);
        const Written w3 = write_gguf(sp);
        save(P, w3.file);
        check("parse: alignment 256", parse_gguf_table(P, &t, &err) && t.alignment == 256 && t.data_start % 256 == 0 && t.data_start == w3.data_start, err);
    }

    // ---- the view ------------------------------------------------------------------------------------------------------------------------------------------
    SynthSpec model = make_model(3, 512, 256, 3, 64);
    const Written wm = write_gguf(model);
    save(P, wm.file);
    {
        ColdWeightsView v;
        std::string err;
        const bool ok = v.open(opts(P), &err);
        check("open: succeeds, 3 layers in order, 3 experts", ok && v.is_open() && v.layers() == std::vector<int>({0, 1, 2}) && v.n_experts() == 3 && v.has_layer(2) && !v.has_layer(3) && !v.has_layer(-1), err);
        std::string why;
        check("open: bases 2 MiB aligned, whole pages with zero padding, bytes identical to the file, shapes/types/strides right, row() and expert() address the right bytes", view_matches(v, model, &why), why);
        check("open: arena base is 2 MiB aligned, size is a multiple of 2 MiB, nothing is gpu_ok before pin()",
              reinterpret_cast<uintptr_t>(v.arena_base()) % COLD_HUGE_PAGE == 0 && v.arena_size() % COLD_HUGE_PAGE == 0 && !v.layer(0).gate.gpu_ok && !v.layer(1).down.gpu_ok);
        size_t sum = 0;
        for (int l = 0; l < 3; ++l) for (const char* w : {"gate", "up", "down"}) sum += find_t(model, tname(l, w))->data.size();
        check("open: stats (layers, tensor bytes) are right and describe() mentions them", v.stats().n_layers == 3 && v.stats().tensor_bytes == sum && v.stats().arena_bytes == v.arena_size() && contains(v.describe(), "3 layer(s)"), v.describe());
        std::printf("     %s\n", v.describe().c_str());
        check("verify_against_file: clean copy passes", v.verify_against_file(&err, 2), err);
        uint8_t* p = const_cast<uint8_t*>(v.layer(1).up.base);
        p[100] ^= 0x5A;
        const bool bad = v.verify_against_file(&err, 2);
        check("verify_against_file: a flipped byte is found and named (blk.1 up)", !bad && contains(err, "blk.1") && contains(err, "up"), err);
        p[100] ^= 0x5A;
        check("verify_against_file: passes again once restored", v.verify_against_file(&err, 1), err);
        v.close();
        check("close: nothing left allocated", !v.is_open() && v.arena_base() == nullptr && v.layers().empty() && !v.has_layer(0));
        check("close twice is harmless; reopen works and is identical", (v.close(), true) && v.open(opts(P), &err) && view_matches(v, model, &why), err + why);
        check("open again without close: replaces the old view", v.open(opts(P), &err) && v.layers().size() == 3 && view_matches(v, model, &why), err + why);
    }
    for (int th : {1, 2, 8, 64}) {
        ColdWeightsView v;
        ColdViewOptions o = opts(P);
        o.load_threads = th;
        o.verify_after_load = true;
        std::string err, why;
        check("load_threads " + std::to_string(th) + " with verify_after_load: bytes identical", v.open(o, &err) && view_matches(v, model, &why), err + why);
    }
    {
        ColdWeightsView v;
        ColdViewOptions o = opts(P);
        o.first_layer = 1;
        o.last_layer = 1;
        std::string err;
        check("layer range 1..1: only blk.1", v.open(o, &err) && v.layers() == std::vector<int>({1}) && !v.has_layer(0) && !v.has_layer(2), err);
        o.first_layer = 2;
        o.last_layer = -1;
        check("first_layer 2, last = all: just blk.2", v.open(o, &err) && v.layers() == std::vector<int>({2}), err);
        o.first_layer = 0;
        o.last_layer = 0;
        o.advise_hugepages = false;
        std::string why;
        check("advise_hugepages off: same bytes", v.open(o, &err) && view_matches(v, model, &why), err + why);
        o.first_layer = 3;
        o.last_layer = -1;
        check("first_layer past the last layer: refused", !v.open(o, &err) && !v.is_open() && !err.empty(), err);
        o.first_layer = 0;
        o.last_layer = 5;
        check("last_layer 5 (blk.3 does not exist): refused, names the missing tensor", !v.open(o, &err) && contains(err, "blk.3"), err);
    }
    {
        ColdWeightsView v;
        ColdViewOptions o = opts(P);
        o.strict_shape = true;
        std::string err;
        check("strict_shape: this geometry is not qwen35moe -> refused with a hint", !v.open(o, &err) && contains(err, "strict_shape") && !v.is_open(), err);
    }
    {   // the MTP block
        SynthSpec big = make_model(41, 256, 256, 2, 32);
        const Written wb = write_gguf(big);
        const std::string PB = g_dir + "/big.gguf";
        save(PB, wb.file);
        ColdWeightsView v;
        ColdViewOptions o = opts(PB);
        std::string err, why;
        check("41 layers: default = blk.0..39, blk.40 (MTP) left out", v.open(o, &err) && v.layers().size() == 40 && v.layers().back() == 39 && !v.has_layer(40), err);
        o.include_mtp = true;
        check("include_mtp: blk.40 loaded too, all 41 bytes-identical", v.open(o, &err) && v.layers().size() == 41 && v.has_layer(40) && view_matches(v, big, &why), err + why);
        o.include_mtp = false;
        o.first_layer = 40;
        o.last_layer = 40;
        check("asking for layer 40 without include_mtp: refused, says why", !v.open(o, &err) && contains(err, "MTP"), err);
        o.include_mtp = true;
        check("layer 40 alone with include_mtp", v.open(o, &err) && v.layers() == std::vector<int>({40}), err);
        std::remove(PB.c_str());
    }

    // ---- shapes the loader must refuse ---------------------------------------------------------------------------------------------------------------------
    {
        auto expect_fail = [&](const char* label, SynthSpec sp, const char* needle) {
            const Written w = write_gguf(sp);
            save(P, w.file);
            ColdWeightsView v;
            std::string err;
            const bool ok = v.open(opts(P), &err);
            check(std::string("refused: ") + label, !ok && !v.is_open() && v.arena_base() == nullptr && !err.empty() && (needle == nullptr || contains(err, needle)), err);
        };
        {
            SynthSpec sp = make_model(2, 512, 256, 3);
            sp.tensors.erase(std::remove_if(sp.tensors.begin(), sp.tensors.end(), [](const STensor& t) { return t.name == "blk.1.ffn_down_exps.weight"; }), sp.tensors.end());
            expect_fail("blk.1 has no down tensor", sp, "blk.1.ffn_down_exps");
        }
        {
            SynthSpec sp = make_model(2, 512, 256, 3);
            for (STensor& t : sp.tensors) if (t.name == "blk.0.ffn_up_exps.weight") t = make_tensor(t.name, 12, {512, 512, 3});
            expect_fail("gate and up differ in shape", sp, "differ");
        }
        {
            SynthSpec sp = make_model(2, 512, 256, 3);
            for (STensor& t : sp.tensors) if (t.name == "blk.1.ffn_down_exps.weight") t = make_tensor(t.name, 12, {512, 256, 3});
            expect_fail("down is not the transpose of gate/up", sp, "down");
        }
        {
            SynthSpec sp = make_model(2, 512, 256, 3);
            for (STensor& t : sp.tensors) if (t.name == "blk.1.ffn_gate_exps.weight") t = make_tensor(t.name, 12, {512, 256, 4});
            for (STensor& t : sp.tensors) if (t.name == "blk.1.ffn_up_exps.weight") t = make_tensor(t.name, 12, {512, 256, 4});
            for (STensor& t : sp.tensors) if (t.name == "blk.1.ffn_down_exps.weight") t = make_tensor(t.name, 12, {256, 512, 4});
            expect_fail("layer 1 has a different expert count than layer 0", sp, "differs");
        }
        {
            SynthSpec sp = make_model(2, 512, 256, 3);
            for (STensor& t : sp.tensors) if (t.name == "blk.0.ffn_gate_exps.weight") t = make_tensor(t.name, 12, {512, 256});
            expect_fail("an expert tensor with 2 dimensions", sp, "3-dimensional");
        }
        {
            SynthSpec sp = make_model(2, 512, 256, 3);
            for (STensor& t : sp.tensors) if (t.name == "blk.0.ffn_gate_exps.weight") t = make_tensor(t.name, 99, {512, 256, 3});
            expect_fail("an expert tensor of a quantisation type the reader does not know", sp, "does not know");
        }
        {
            SynthSpec sp;
            sp.tensors.push_back(make_tensor("token_embd.weight", 1, {512, 32}));
            sp.tensors.push_back(make_tensor("blk.0.ffn_gate.weight", 12, {512, 256}));
            expect_fail("a dense model (no expert tensors)", sp, "ffn_gate_exps");
        }
        {   // an unknown type on a tensor we never use is fine
            SynthSpec sp = make_model(2, 512, 256, 3);
            sp.tensors.push_back(make_tensor("rope_freqs.weight", 99, {64}));
            const Written w = write_gguf(sp);
            save(P, w.file);
            ColdWeightsView v;
            std::string err, why;
            check("accepted: an unknown-type tensor elsewhere in the file does not matter", v.open(opts(P), &err) && view_matches(v, sp, &why), err + why);
        }
        {   // the nonexistent file / empty file
            ColdWeightsView v;
            std::string err;
            check("refused: file does not exist", !v.open(opts(g_dir + "/nope.gguf"), &err) && contains(err, "cannot open"), err);
            save(P, {});
            check("refused: empty file", !v.open(opts(P), &err) && !err.empty(), err);
            save(P, {'G', 'G', 'U'});
            check("refused: 3 bytes", !v.open(opts(P), &err) && !err.empty(), err);
        }
    }

    // ---- damaged headers -----------------------------------------------------------------------------------------------------------------------------------
    {
        SynthSpec sp = make_model(2, 256, 256, 2, 32);
        const Written w = write_gguf(sp);
        auto try_bytes = [&](std::vector<uint8_t> b, size_t cut, std::string* err) {
            save(P, b, cut);
            GgufTable t;
            return parse_gguf_table(P, &t, err);
        };
        std::string err;
        {
            std::vector<uint8_t> b = w.file;
            b[0] = 'X';
            check("damaged: bad magic", !try_bytes(b, SIZE_MAX, &err) && contains(err, "magic"), err);
            b = w.file;
            const uint32_t v1 = 1, v4 = 4, vbe = 0x03000000u;
            std::memcpy(b.data() + 4, &v1, 4);
            check("damaged: version 1 (not supported)", !try_bytes(b, SIZE_MAX, &err) && contains(err, "version"), err);
            std::memcpy(b.data() + 4, &v4, 4);
            check("damaged: version 4", !try_bytes(b, SIZE_MAX, &err) && contains(err, "version"), err);
            std::memcpy(b.data() + 4, &vbe, 4);
            check("damaged: a big-endian file (version bytes swapped)", !try_bytes(b, SIZE_MAX, &err) && contains(err, "version"), err);
            b = w.file;
            const uint64_t huge = 1ull << 40;
            std::memcpy(b.data() + 8, &huge, 8);
            check("damaged: absurd tensor count", !try_bytes(b, SIZE_MAX, &err) && contains(err, "implausible"), err);
            b = w.file;
            std::memcpy(b.data() + 16, &huge, 8);
            check("damaged: absurd metadata count", !try_bytes(b, SIZE_MAX, &err) && contains(err, "implausible"), err);
            b = w.file;
            const uint64_t big_n = 1ull << 20;
            std::memcpy(b.data() + 16, &big_n, 8);
            check("damaged: metadata count larger than the file really has -> clean error, not a crash", !try_bytes(b, SIZE_MAX, &err) && !err.empty(), err);
            b = w.file;
            const uint64_t past = w.file.size() * 4;
            std::memcpy(b.data() + w.pos_of_offset[3], &past, 8);
            check("damaged: a tensor offset far past the end of the file", !try_bytes(b, SIZE_MAX, &err) && contains(err, "past the end"), err);
            b = w.file;
            const uint64_t near_end = w.file.size() - w.data_start - 8;
            std::memcpy(b.data() + w.pos_of_offset[2], &near_end, 8);
            check("damaged: a tensor whose last bytes would be past the end", !try_bytes(b, SIZE_MAX, &err) && contains(err, "past the end"), err);
            b = w.file;
            const uint64_t maxoff = UINT64_MAX - 3;
            std::memcpy(b.data() + w.pos_of_offset[2], &maxoff, 8);
            err.clear();
            check("damaged: a tensor offset near 2^64 (overflow)", !try_bytes(b, SIZE_MAX, &err) && contains(err, "past the end"), err);
            b = w.file;
            {   // dims whose product overflows 64 bits
                const uint64_t hugedim = 1ull << 40;
                for (size_t i = 0; i < sp.tensors.size(); ++i)
                    if (sp.tensors[i].dims.size() == 3) { for (int d = 0; d < 3; ++d) std::memcpy(b.data() + w.pos_of_ndims[i] + 4 + 8 * static_cast<size_t>(d), &hugedim, 8); break; }
                err.clear();
                check("damaged: three dimensions of 2^40 (element count overflows 64 bits)", !try_bytes(b, SIZE_MAX, &err) && !err.empty(), err);
            }
            b = w.file;
            const uint32_t nd5 = 5, nd0 = 0;
            std::memcpy(b.data() + w.pos_of_ndims[2], &nd5, 4);
            check("damaged: 5 dimensions", !try_bytes(b, SIZE_MAX, &err) && contains(err, "dimensions"), err);
            std::memcpy(b.data() + w.pos_of_ndims[2], &nd0, 4);
            check("damaged: 0 dimensions", !try_bytes(b, SIZE_MAX, &err) && contains(err, "dimensions"), err);
            b = w.file;
            // a dims[0] that is not a multiple of the Q4_K block: find the first Q4_K tensor and set its dims[0] to 100
            for (size_t i = 0; i < sp.tensors.size(); ++i)
                if (sp.tensors[i].type == 12) { const uint64_t d = 100; std::memcpy(b.data() + w.pos_of_ndims[i] + 4, &d, 8); break; }
            check("damaged: Q4_K row length 100 (not a multiple of 256)", !try_bytes(b, SIZE_MAX, &err) && contains(err, "block size"), err);
        }
        {   // alignment 0 in the metadata
            SynthSpec sp0 = make_model(1, 256, 256, 2, 32);
            sp0.alignment = 0;
            // the writer cannot align to 0: write with 32 and patch the KV afterwards
            sp0.alignment = 32;
            Written w0 = write_gguf(sp0);
            // locate the general.alignment value: key string + type u32 + value u32
            const std::string key = "general.alignment";
            auto it = std::search(w0.file.begin(), w0.file.end(), key.begin(), key.end());
            if (it != w0.file.end()) {
                const size_t pos = static_cast<size_t>(it - w0.file.begin()) + key.size() + 4;
                const uint32_t zero = 0;
                std::memcpy(w0.file.data() + pos, &zero, 4);
                check("damaged: general.alignment = 0", !try_bytes(w0.file, SIZE_MAX, &err) && contains(err, "alignment"), err);
            } else check("damaged: general.alignment = 0 (could not locate the key)", false);
        }
        // truncation at every position of the header, then inside the data
        {
            int clean = 0, total = 0;
            std::string first_bad;
            for (size_t cut = 0; cut < w.data_start; ++cut) {
                ++total;
                GgufTable t;
                save(P, w.file, cut);
                std::string e;
                if (!parse_gguf_table(P, &t, &e) && !e.empty()) ++clean;
                else if (first_bad.empty()) first_bad = "cut at " + std::to_string(cut);
            }
            check("truncated at each of the " + std::to_string(total) + " byte positions of the header: every cut is a clean error", clean == total, first_bad);
            int tclean = 0, ttotal = 0;
            for (size_t cut = w.data_start; cut < w.file.size(); cut += 4099) {
                ++ttotal;
                GgufTable t;
                save(P, w.file, cut);
                std::string e;
                if (!parse_gguf_table(P, &t, &e) && contains(e, "past the end")) ++tclean;
            }
            check("truncated inside the tensor data (" + std::to_string(ttotal) + " cuts): refused, the tensor that no longer fits is named", tclean == ttotal);
        }
        // random byte damage in the header: success or a clean error, never a crash; when it succeeds the view must still match the file
        {
            std::mt19937 rng(777);
            int ok_parse = 0, bad_parse = 0, opened = 0;
            const int iters = 1500;
            for (int it = 0; it < iters; ++it) {
                std::vector<uint8_t> b = w.file;
                const int nflip = 1 + static_cast<int>(rng() % 4);
                for (int k = 0; k < nflip; ++k) b[rng() % w.data_start] = static_cast<uint8_t>(rng());
                save(P, b);
                GgufTable t;
                std::string e;
                if (parse_gguf_table(P, &t, &e)) {
                    ++ok_parse;
                    ColdWeightsView v;
                    ColdViewOptions o = opts(P);
                    o.verify_after_load = true;
                    if (v.open(o, &e)) ++opened;
                } else ++bad_parse;
            }
            std::printf("     random header damage: %d files parsed, %d refused, %d of the parsed ones opened and verified\n", ok_parse, bad_parse, opened);
            check("1500 files with 1-4 random header bytes changed: no crash, no hang, every refusal has a message (run under ASan/UBSan for the memory proof)", ok_parse + bad_parse == iters);
        }
    }

    // ---- pin() ---------------------------------------------------------------------------------------------------------------------------------------------
    save(P, wm.file);
    {
        ColdWeightsView v;
        std::string err;
        v.open(opts(P), &err);
        ColdPinReport rep;
#if defined(PELIAS_CUDA_SHIM)
        pl_shim_registered_now = 0;
        pl_shim_register_calls = 0;
        pl_shim_register_fail_after = -1;
        pl_shim_dev_offset = 0;
        pl_shim_getdev_calls = 0;
        pl_shim_getdev_fail_after = -1;
        bool null_before = true;
        for (int l : v.layers()) for (const ColdMat* m : {&v.layer(l).gate, &v.layer(l).up, &v.layer(l).down}) { if (m->dev_base != nullptr) null_before = false; }
        const bool ok = v.pin(0, &rep);
        bool dev_same = true;
        for (int l : v.layers()) for (const ColdMat* m : {&v.layer(l).gate, &v.layer(l).up, &v.layer(l).down}) { if (m->dev_base != m->base) dev_same = false; }
        check("pin (shim), device address == host address: every region registered, report says identical (offset 0, constant), dev_base == base for all 9 matrices, all layers gpu_ok", null_before && ok && rep.regions_ok == 9 && rep.regions_failed == 0 && rep.regions_skipped == 0 && rep.uva_identical && rep.dev_offset_constant && rep.dev_offset == 0 && dev_same &&
                                                                                                        v.layer(0).gate.gpu_ok && v.layer(1).up.gpu_ok && v.layer(2).down.gpu_ok && rep.pinned_bytes == v.arena_size() && pl_shim_registered_now == 9, rep.first_error);
        v.close();
        check("pin (shim): close() unregisters everything", pl_shim_registered_now == 0);

        // device address != host address (what cudaHostRegister memory gives on WSL2): the GPU side must use its own pointer, at the same offsets
        v.open(opts(P), &err);
        pl_shim_dev_offset = 0x40000000LL;
        ColdPinReport off;
        const bool ok_off = v.pin(0, &off);
        pl_shim_dev_offset = 0;
        bool dev_ok = true;
        for (int l : v.layers()) {
            for (const ColdMat* m : {&v.layer(l).gate, &v.layer(l).up, &v.layer(l).down}) {
                if (!m->gpu_ok || m->dev_base != m->base + 0x40000000LL) dev_ok = false;
                for (int e : {0, 2}) {
                    if (m->dev_expert(e) != m->expert(e) + 0x40000000LL) dev_ok = false;
                    if (m->dev_row(e, 1) != m->row(e, 1) + 0x40000000LL) dev_ok = false;
                }
            }
        }
        check("pin (shim), device address = host address + 1 GiB: not an error; all 9 regions ok, report says NOT identical but constant with the right offset, every matrix has dev_base = base + offset, dev_row/dev_expert follow the same offsets, all layers gpu_ok",
              ok_off && off.regions_ok == 9 && off.regions_failed == 0 && !off.uva_identical && off.dev_offset_constant && off.dev_offset == 0x40000000LL && dev_ok && off.first_error.empty() && pl_shim_registered_now == 9, off.first_error);
        std::string why_off;
        check("pin (shim), different device address: the CPU side is untouched (bytes still intact)", view_matches(v, model, &why_off), why_off);
        v.close();
        check("pin (shim): unregistered again after the different-address case", pl_shim_registered_now == 0);

#if defined(PELIAS_COLD_VIEW_GPU_CHECK)
        // the GPU read check itself (stand-in kernels run serially): clean, and with the device side pointing at a copy that has one flipped byte
        v.open(opts(P), &err);
        pl_shim_dev_offset = 0;
        (void)v.pin(0, &rep);
        GpuReadResult gclean;
        const bool gc_ok = gpu_read_check(v, 2, 1, &gclean);
        check("gpu read check (stand-in kernels), device side = host memory: 9 matrices, every whole-tensor checksum equal, all 81 sampled rows equal", gc_ok && gclean.matrices == 9 && gclean.mismatched == 0 &&
                                                                                                                                                  gclean.rows_checked == 81 && gclean.rows_bad == 0, gclean.first_error);
        v.close();
        v.open(opts(P), &err);
        uint8_t* copy = static_cast<uint8_t*>(std::aligned_alloc(COLD_HUGE_PAGE, v.arena_size()));
        std::memcpy(copy, v.arena_base(), v.arena_size());
        copy[static_cast<size_t>(v.layer(1).up.base - v.arena_base()) + 7] ^= 0x5A;      // one byte inside blk.1 up differs on the "device" side
        pl_shim_dev_offset = static_cast<long long>(copy - v.arena_base());
        (void)v.pin(0, &rep);
        pl_shim_dev_offset = 0;
        GpuReadResult gbad;
        const bool gb_ok = gpu_read_check(v, 2, 1, &gbad);
        check("gpu read check (stand-in kernels), one flipped byte in blk.1 up on the device side: found, exactly that matrix named, the other 8 equal", gb_ok && gbad.mismatched == 1 && gbad.first_mismatch == "blk.1 up" &&
                                                                                                                                      gbad.rows_bad >= 1, gbad.first_mismatch + " / " + gbad.first_error);
        v.close();
        std::free(copy);
        check("pin (shim): unregistered again after the GPU read checks", pl_shim_registered_now == 0);
#endif

        v.open(opts(P), &err);
        pl_shim_getdev_calls = 0;
        pl_shim_getdev_fail_after = 4;                        // cudaHostGetDevicePointer refuses from the 5th region on
        ColdPinReport gd;
        const bool ok_gd = v.pin(0, &gd);
        pl_shim_getdev_fail_after = -1;
        check("pin (shim), cudaHostGetDevicePointer refuses from the 5th region on: 4 ok, the 5 refused ones counted as failed (first one named) and unregistered again, layer 0 gpu_ok with a device pointer, layer 1 not gpu_ok and without device pointers",
              ok_gd && gd.regions_ok == 4 && gd.regions_failed == 5 && gd.regions_skipped == 0 && contains(gd.first_error, "cudaHostGetDevicePointer") && pl_shim_registered_now == 4 && v.layer(0).gate.gpu_ok && v.layer(0).gate.dev_base != nullptr &&
                  v.layer(0).down.dev_base != nullptr && !v.layer(1).gate.gpu_ok && v.layer(1).gate.dev_base == nullptr && v.layer(1).up.dev_base == nullptr && v.layer(1).down.dev_base == nullptr,
              gd.first_error);
        v.close();
        check("pin (shim): nothing left registered after the refused-pointer case", pl_shim_registered_now == 0);

        v.open(opts(P), &err);
        size_t layer0_bytes = 0;
        for (const char* w : {"gate", "up", "down"}) layer0_bytes += (find_t(model, tname(0, w))->data.size() + COLD_HUGE_PAGE - 1) / COLD_HUGE_PAGE * COLD_HUGE_PAGE;
        ColdPinReport lim;
        const bool ok2 = v.pin(layer0_bytes + 1, &lim);
        check("pin (shim) with max_bytes = layer 0 + 1 byte: only layer 0 is pinned and gpu_ok, the rest are counted as skipped", ok2 && lim.regions_ok == 3 && lim.regions_skipped == 6 && lim.pinned_bytes == layer0_bytes &&
                                                                                                                                   v.layer(0).gate.gpu_ok && v.layer(0).down.gpu_ok && !v.layer(1).gate.gpu_ok && !v.layer(2).down.gpu_ok && v.layer(0).up.dev_base != nullptr && v.layer(2).down.dev_base == nullptr, lim.first_error);
        v.close();
        check("pin (shim): unregistered again", pl_shim_registered_now == 0);

        v.open(opts(P), &err);
        pl_shim_register_calls = 0;
        pl_shim_register_fail_after = 4;                      // the driver refuses from the 5th region on
        ColdPinReport part;
        const bool ok3 = v.pin(0, &part);
        pl_shim_register_fail_after = -1;
        check("pin (shim), driver refuses after 4 regions: 4 ok, 1 failed (named), the rest skipped, layer 0 gpu_ok, layer 1 only partly pinned so NOT gpu_ok, nothing crashes",
              ok3 && part.regions_ok == 4 && part.regions_failed == 1 && part.regions_skipped == 4 && contains(part.first_error, "cudaHostRegister") && v.layer(0).gate.gpu_ok && v.layer(0).down.gpu_ok &&
                  !v.layer(1).gate.gpu_ok && !v.layer(1).up.gpu_ok && !v.layer(1).down.gpu_ok && !v.layer(2).gate.gpu_ok,
              part.first_error);
        v.close();
        check("pin (shim): the 4 registered regions are unregistered by close()", pl_shim_registered_now == 0);

        v.open(opts(P), &err);
        pl_shim_register_calls = 0;
        pl_shim_register_fail_after = 0;
        ColdPinReport none;
        const bool ok4 = v.pin(0, &none);
        pl_shim_register_fail_after = -1;
        check("pin (shim), driver refuses the very first region: returns false, nothing is gpu_ok, the view still works for the CPU", !ok4 && none.regions_ok == 0 && none.regions_failed == 1 && !v.layer(0).gate.gpu_ok &&
                                                                                                                                          view_matches(v, model, &err), none.first_error + err);
        v.close();
        check("pin (shim): nothing to unregister", pl_shim_registered_now == 0);
#elif !defined(PELIAS_COLD_VIEW_CUDA)       // a real CUDA build is exercised by the real-file mode (--pin) instead
        const bool ok = v.pin(0, &rep);
        check("pin (no CUDA build): returns false and says the build has no CUDA; nothing is gpu_ok; the view still works", !ok && rep.regions_ok == 0 && contains(rep.first_error, "without CUDA") && !v.layer(0).gate.gpu_ok, rep.first_error);
        std::string why;
        check("pin (no CUDA build): bytes still intact", view_matches(v, model, &why), why);
#endif
    }

    // ---- real file (optional) ------------------------------------------------------------------------------------------------------------------------------
    if (!real_path.empty()) {
        std::printf("\nreal file: %s\n", real_path.c_str());
        ColdWeightsView v;
        ColdViewOptions o;
        o.gguf_path = real_path;
        o.first_layer = real_first;
        o.last_layer = real_last;
        o.strict_shape = real_strict;
        o.load_threads = real_threads;
        std::string err;
        const bool ok = v.open(o, &err);
        check("real file: open", ok, err);
        if (ok) {
            std::printf("     %s\n", v.describe().c_str());
            const ColdLayerView& l0 = v.layer(v.layers().front());
            std::printf("     blk.%d  gate %lld x %lld x %lld type %d (%zu B/row, %zu B/expert)   down %lld x %lld type %d (%zu B/row, %zu B/expert)\n", l0.layer, static_cast<long long>(l0.gate.cols),
                        static_cast<long long>(l0.gate.rows), static_cast<long long>(l0.gate.n_experts), l0.gate.type_id, l0.gate.row_bytes, l0.gate.expert_stride, static_cast<long long>(l0.down.cols),
                        static_cast<long long>(l0.down.rows), l0.down.type_id, l0.down.row_bytes, l0.down.expert_stride);
            bool aligned = true;
            for (int l : v.layers()) {
                const ColdLayerView& lv = v.layer(l);
                for (const ColdMat* m : {&lv.gate, &lv.up, &lv.down})
                    if (reinterpret_cast<uintptr_t>(m->base) % COLD_HUGE_PAGE != 0) aligned = false;
            }
            check("real file: every tensor base is 2 MiB aligned", aligned);
            if (real_strict) {
                bool sizes = true;
                for (int l : v.layers()) {
                    const ColdLayerView& lv = v.layer(l);
                    const size_t g = lv.gate.expert_stride * static_cast<size_t>(lv.gate.n_experts);
                    const size_t d = lv.down.expert_stride * static_cast<size_t>(lv.down.n_experts);
                    if (g != 150994944u || lv.up.expert_stride * static_cast<size_t>(lv.up.n_experts) != 150994944u || (d != 150994944u && d != 220200960u)) sizes = false;
                }
                check("real file: gate/up tensors are 150,994,944 bytes (72 x 2 MiB), down 150,994,944 (Q4_K) or 220,200,960 (Q6_K, 105 x 2 MiB)", sizes);
            }
            if (real_verify) {
                const bool vok = v.verify_against_file(&err, real_threads);
                check("real file: verify_against_file (second read of the whole file range, byte compare)", vok, err);
            }
            if (real_pin) {
                ColdPinReport pr;
                const auto t0 = std::chrono::steady_clock::now();
                const bool pok = v.pin(static_cast<size_t>(real_pin_mib) << 20, &pr);
                const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                size_t gpu_layers = 0;
                for (int l : v.layers()) if (v.layer(l).gate.gpu_ok && v.layer(l).up.gpu_ok && v.layer(l).down.gpu_ok) ++gpu_layers;
                std::printf("     pin: %zu region(s) ok, %zu failed, %zu skipped; %.2f GiB registered in %.2f s; %zu of %zu layer(s) gpu_ok; device address == host address for all: %s; (device - host) offset constant: %s, first offset %lld bytes\n", pr.regions_ok,
                            pr.regions_failed, pr.regions_skipped, static_cast<double>(pr.pinned_bytes) / (1ull << 30), secs, gpu_layers, v.layers().size(), pr.uva_identical ? "yes" : "NO",
                            pr.dev_offset_constant ? "yes" : "NO", static_cast<long long>(pr.dev_offset));
                if (!pr.first_error.empty()) std::printf("     pin first error: %s\n", pr.first_error.c_str());
                bool dev_set = true;                           // every gpu_ok matrix has its own GPU pointer; the others have none
                for (int l : v.layers())
                    for (const ColdMat* m : {&v.layer(l).gate, &v.layer(l).up, &v.layer(l).down})
                        if (m->gpu_ok != (m->dev_base != nullptr)) dev_set = false;
                check("real file: pin() registered the regions it was asked to and gave every pinned matrix a GPU pointer (the GPU address may differ from the CPU address)",
                      pok && pr.regions_failed == 0 && dev_set && (real_pin_mib != 0 || pr.regions_skipped == 0), pr.first_error);
                std::printf("     %s\n", v.describe().c_str());
            }
            if (real_gpu) {
#if defined(PELIAS_COLD_VIEW_GPU_CHECK)
                if (!real_pin) {
                    check("real file: --gpu-read needs --pin as well", false);
                } else {
                    GpuReadResult gr;
                    const bool gok = gpu_read_check(v, real_threads, 3, &gr);
                    std::printf("     gpu read through dev_base: %zu matrices, %.2f GiB; whole-tensor checksum mismatches %zu; sampled rows %zu, wrong %zu; CPU checksum took %.1f s\n", gr.matrices,
                                static_cast<double>(gr.bytes) / (1ull << 30), gr.mismatched, gr.rows_checked, gr.rows_bad, gr.cpu_seconds);
                    if (gr.gbs.empty()) {
                        std::printf("     gpu read speed: not measured in this build\n");
                    } else {
                        std::vector<double> sorted = gr.gbs;
                        std::sort(sorted.begin(), sorted.end());
                        std::printf("     gpu checksum-read speed per timed pass (GB/s, measured, includes the hash arithmetic, no limit applied):");
                        for (double g : gr.gbs) std::printf(" %.2f", g);
                        std::printf("   median %.2f\n", sorted[sorted.size() / 2]);
                    }
                    if (!gr.first_mismatch.empty()) std::printf("     first matrix that differs: %s\n", gr.first_mismatch.c_str());
                    if (!gr.first_bad_row.empty()) std::printf("     first row that differs: %s\n", gr.first_bad_row.c_str());
                    check("real file: a GPU kernel reading through dev_base / dev_row sees the same bytes as the CPU (whole-tensor checksum and sampled rows)",
                          gok && gr.matrices > 0 && gr.mismatched == 0 && gr.rows_bad == 0, gr.first_error);
                }
#else
                check("real file: --gpu-read needs a build with -DPELIAS_COLD_VIEW_GPU_CHECK (compile this file as CUDA, nvcc -x cu)", false);
#endif
            }
        }
    }

    // clean up
    {
        std::string cmd = "rm -rf '" + g_dir + "'";
        if (std::system(cmd.c_str()) != 0) std::printf("note: could not remove %s\n", g_dir.c_str());
    }
    std::printf("%s\n", g_fail == 0 ? "ALL PASS" : "FAILED");
    return g_fail == 0 ? 0 : 1;
}
