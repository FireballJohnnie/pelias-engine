// File: <project>\moe_track\microbench\02_ram_bw_test.cpp
// Step 2b: how fast can this CPU read expert-sized chunks of RAM, with 4 KB pages versus transparent huge pages?
//
// Build (done for you by microbench/build.sh):   g++ -O3 -mavx2 -fopenmp -std=c++17 02_ram_bw_test.cpp -o bin/02_ram_bw_test
// Run with the model server STOPPED (it holds ~20 GB and would distort the result):
//   bin/02_ram_bw_test [--gib 8] [--expert-bytes 1900000] [--experts 256] [--topk 8] [--tokens 6]
//                      [--threads 4,6,7,8,12,16] [--out <project>/logs/floors/ram_bw_floor.json] [--force]
//
// Two access patterns per thread count and page mode:
//   stream : every thread streams a contiguous slice of the whole buffer          -> peak bandwidth of the memory system
//   expert : per token, for each layer, topk random experts are read by ALL threads together (rows split into chunks,
//            dynamic schedule), then a barrier before the next layer.            -> what llama.cpp's CPU expert path does
// "expert" is the number to compare with the fitted CPU expert-path bandwidth (gate G0 prints both).
//
// Every vector lane is folded into the checksum, so the compiler cannot drop the loads.
// After the buffer is touched, the kernel's own count of huge-page-backed bytes (AnonHugePages in /proc/self/smaps)
// is reported next to the result: if the "2MB_hugepages" row shows ~0 kB, THP was refused and that row is just 4 KB pages again.
#include <immintrin.h>
#include <omp.h>
#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Cfg {
    double gib = 8.0;
    size_t expert_bytes = 1900000;     // fleet average from the GGUF audit (microbench/01_gguf_expert_sizes.py)
    int experts = 256;
    int topk = 8;
    int tokens = 6;
    std::vector<int> threads = {4, 6, 7, 8, 12, 16};
    std::string out = "<project>/logs/floors/ram_bw_floor.json";
    bool force = false;
};

std::vector<int> parse_list(const std::string& s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) v.push_back(std::atoi(item.c_str()));
    }
    return v;
}

bool server_running() {
    FILE* p = popen("pgrep -x llama-server >/dev/null 2>&1; echo $?", "r");
    if (!p) return false;
    char buf[16] = {0};
    if (!fgets(buf, sizeof buf, p)) { pclose(p); return false; }
    pclose(p);
    return std::atoi(buf) == 0;
}

// kB of this process's anonymous memory backed by huge pages, read from the kernel
long anon_huge_kb() {
    std::ifstream f("/proc/self/smaps_rollup");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("AnonHugePages:", 0) == 0) return std::atol(line.c_str() + 14);
    }
    return -1;
}

inline uint64_t fold(__m256i v) {
    alignas(32) uint64_t t[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(t), v);
    return t[0] ^ t[1] ^ t[2] ^ t[3];
}

// read [p, p+n) with 256-bit loads, 4 independent accumulators
inline __m256i read_block(const uint8_t* p, size_t n, __m256i acc) {
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
    for (size_t b = 0; b + 128 <= n; b += 128) {
        a0 = _mm256_add_epi64(a0, _mm256_load_si256(reinterpret_cast<const __m256i*>(p + b)));
        a1 = _mm256_add_epi64(a1, _mm256_load_si256(reinterpret_cast<const __m256i*>(p + b + 32)));
        a2 = _mm256_add_epi64(a2, _mm256_load_si256(reinterpret_cast<const __m256i*>(p + b + 64)));
        a3 = _mm256_add_epi64(a3, _mm256_load_si256(reinterpret_cast<const __m256i*>(p + b + 96)));
    }
    return _mm256_add_epi64(acc, _mm256_add_epi64(_mm256_add_epi64(a0, a1), _mm256_add_epi64(a2, a3)));
}

double bench_stream(const uint8_t* buf, size_t bytes, int threads, int repeats, uint64_t& sink) {
    omp_set_num_threads(threads);
    uint64_t local = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; ++r) {
#pragma omp parallel reduction(^ : local)
        {
            const int id = omp_get_thread_num(), nt = omp_get_num_threads();
            size_t per = (bytes / nt) & ~size_t(127);
            local ^= fold(read_block(buf + per * id, per, _mm256_setzero_si256()));
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    sink ^= local;
    return (double(bytes) * repeats / 1e9) / std::chrono::duration<double>(t1 - t0).count();
}

// chunk granularity for splitting one expert across threads, like row blocks in a matmul
constexpr size_t CHUNK = 64 * 1024;

double bench_experts(const uint8_t* buf, const Cfg& c, size_t stride, size_t n_layers, int threads, uint64_t& sink) {
    omp_set_num_threads(threads);
    // fixed random expert choice per (token, layer, slot) so every thread count reads the same bytes
    std::vector<size_t> pick(size_t(c.tokens) * n_layers * c.topk);
    uint64_t rng = 0x9E3779B97F4A7C15ULL;
    for (size_t& x : pick) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        x = rng % size_t(c.experts);
    }
    const size_t exp_bytes = c.expert_bytes & ~size_t(127);
    const size_t chunks_per_expert = (exp_bytes + CHUNK - 1) / CHUNK;
    uint64_t local = 0;
    double bytes_read = 0.0;
    auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel reduction(^ : local)
    {
        for (int tok = 0; tok < c.tokens; ++tok) {
            for (size_t layer = 0; layer < n_layers; ++layer) {
                const uint8_t* layer_base = buf + layer * stride * size_t(c.experts);
                const size_t* sel = &pick[(size_t(tok) * n_layers + layer) * c.topk];
                const size_t total_chunks = size_t(c.topk) * chunks_per_expert;
#pragma omp for schedule(dynamic, 1)
                for (size_t k = 0; k < total_chunks; ++k) {
                    const size_t slot = k / chunks_per_expert, ch = k % chunks_per_expert;
                    const uint8_t* p = layer_base + sel[slot] * stride + ch * CHUNK;
                    const size_t n = std::min(CHUNK, exp_bytes - ch * CHUNK) & ~size_t(127);
                    local ^= fold(read_block(p, n, _mm256_setzero_si256()));
                }  // implicit barrier: the next layer needs this layer's output
            }
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    bytes_read = double(c.tokens) * n_layers * c.topk * double(exp_bytes);
    sink ^= local;
    return (bytes_read / 1e9) / std::chrono::duration<double>(t1 - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    Cfg c;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", name); std::exit(2); }
            return argv[++i];
        };
        if (a == "--gib") c.gib = std::atof(next("--gib").c_str());
        else if (a == "--expert-bytes") c.expert_bytes = std::strtoull(next("--expert-bytes").c_str(), nullptr, 10);
        else if (a == "--experts") c.experts = std::atoi(next("--experts").c_str());
        else if (a == "--topk") c.topk = std::atoi(next("--topk").c_str());
        else if (a == "--tokens") c.tokens = std::atoi(next("--tokens").c_str());
        else if (a == "--threads") c.threads = parse_list(next("--threads"));
        else if (a == "--out") c.out = next("--out");
        else if (a == "--force") c.force = true;
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (!c.force && server_running()) {
        std::fprintf(stderr, "llama-server is running; stop it first (llama --stop) or pass --force. Results would be distorted.\n");
        return 3;
    }

    const size_t stride = (c.expert_bytes + 4095) & ~size_t(4095);           // page-aligned expert slots
    const size_t layer_bytes = stride * size_t(c.experts);
    size_t buffer_bytes = size_t(c.gib * 1024.0 * 1024.0 * 1024.0);
    const size_t n_layers = std::max<size_t>(1, buffer_bytes / layer_bytes);
    buffer_bytes = n_layers * layer_bytes;                                      // whole layers only
    const size_t huge = 2 * 1024 * 1024;
    const size_t map_bytes = (buffer_bytes + huge - 1) / huge * huge;

    std::printf("[2b] buffer %.2f GiB = %zu layers x %d experts x %zu B (slot %zu B); %d tokens x %zu layers x top-%d\n",
                buffer_bytes / 1073741824.0, n_layers, c.experts, c.expert_bytes, stride, c.tokens, n_layers, c.topk);

    std::ostringstream js;
    js << "{\n  \"buffer_gib\": " << buffer_bytes / 1073741824.0 << ",\n  \"expert_bytes\": " << c.expert_bytes
       << ",\n  \"layers_simulated\": " << n_layers << ",\n  \"topk\": " << c.topk << ",\n  \"results\": [\n";
    bool first = true;
    uint64_t sink = 0;
    double best_expert = 0.0, best_stream = 0.0;

    for (int mode = 0; mode < 2; ++mode) {
        const char* name = mode == 0 ? "4KB_pages" : "2MB_hugepages";
        // over-allocate so the start can be aligned to a 2 MB boundary (THP needs aligned ranges)
        void* raw = mmap(nullptr, map_bytes + huge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (raw == MAP_FAILED) { std::perror("mmap"); return 1; }
        uint8_t* buf = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(raw) + huge - 1) & ~(uintptr_t(huge) - 1));
        madvise(buf, map_bytes, mode == 0 ? MADV_NOHUGEPAGE : MADV_HUGEPAGE);
        const long huge_before = anon_huge_kb();
#pragma omp parallel for schedule(static)
        for (size_t p = 0; p < buffer_bytes; p += 4096) buf[p] = uint8_t((p >> 12) * 131u);   // touch every 4 KB page
        const long huge_kb = anon_huge_kb() - std::max(0L, huge_before);

        for (int t : c.threads) {
            const double stream = bench_stream(buf, buffer_bytes, t, 2, sink);
            const double expert = bench_experts(buf, c, stride, n_layers, t, sink);
            best_stream = std::max(best_stream, stream);
            best_expert = std::max(best_expert, expert);
            std::printf("[2b] %-14s threads=%2d  stream %6.2f GB/s   expert-pattern %6.2f GB/s   (huge-page-backed: %ld kB)\n",
                        name, t, stream, expert, huge_kb);
            js << (first ? "" : ",\n") << "    {\"page_mode\": \"" << name << "\", \"threads\": " << t
               << ", \"stream_gb_s\": " << stream << ", \"expert_gb_s\": " << expert << ", \"anon_huge_kb\": " << huge_kb << "}";
            first = false;
        }
        munmap(raw, map_bytes + huge);
    }
    js << "\n  ],\n  \"best_stream_gb_s\": " << best_stream << ",\n  \"best_expert_gb_s\": " << best_expert
       << ",\n  \"checksum\": " << sink << "\n}\n";

    std::ofstream out(c.out);
    if (!out) { std::fprintf(stderr, "cannot write %s (printing JSON instead)\n%s", c.out.c_str(), js.str().c_str()); return 4; }
    out << js.str();
    std::printf("[2b] wrote %s\n", c.out.c_str());
    return 0;
}
