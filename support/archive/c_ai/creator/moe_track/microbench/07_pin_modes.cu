// File: <project>\moe_track\microbench\07_pin_modes.cu
// Step 2f: how can the GPU read the cold-expert weights in host RAM on this machine, and how much of it can be pinned?
//
// Why. Pass 1's real-GGUF run (tests/test_cold_weights_view.cpp --pin) measured: cudaHostRegister(Mapped|Portable) returned success for 12 regions of an
// ordinary (mmap'd, huge-page backed) arena, but cudaHostGetDevicePointer returned an address different from the host address for every one. Memory from
// cudaHostAlloc has identical addresses (the handoff mailbox). This probe measures, for each way of getting host memory the GPU can read:
//   * allocation style     alloc_mapped          cudaHostAlloc(Mapped|Portable), one call per chunk                       (what 03 and 04 use)
//                          reg_mapped_portable   one mmap'd 2 MiB-aligned arena + cudaHostRegister(Mapped|Portable)      (what cold_weights_view does)
//                          reg_mapped            the same with cudaHostRegister(Mapped)
//                          reg_default           the same with cudaHostRegister(Default); no device pointer is expected, the host pointer is tried
//   * pointer given to the kernel   dev   the address cudaHostGetDevicePointer returned     host   the host address itself (what unified addressing would allow)
//   * what it reports      per chunk: host address, device address, difference (is it one constant for all chunks?); cudaPointerGetAttributes for chunk 0;
//                          at 1, 2, 4, 8 ... chunks and at the end: a GPU kernel reads every byte so far and checks it against the pattern the CPU wrote
//                          (mismatch count) and times the read (GB/s, every repetition printed, then the median);
//                          the ceiling: how many chunks could be pinned before the driver said no, and the error text; MemAvailable before and at that point.
// Each (style, pointer) pair runs in its own child process, so a GPU fault or a crash in one does not stop the others; the parent only collects results.
// No pass/fail limits are applied to any number: the output is the measurement. A mismatch count above zero or a kernel error is printed as such.
//
// Sizes are not invented: the default chunk is 144 MiB, one real gate/up tensor (150,994,944 B); the default total is 20 GiB, just above the most the
// model's routed experts can need (40 layers x (2 x 144 MiB + 210 MiB) = 19.45 GiB when every down tensor is Q6_K; 16.9 GiB when none is).
//
// Build (WSL, the toolchain from the earlier runs):
//   R=$HOME/stuart_creator_engine/cuda_shim
//   $R/bin/nvcc -O3 -std=c++17 -arch=sm_89 -I$R/include --cudart shared -L$R/lib64 -Xlinker -rpath,$R/lib64 -Xcompiler "-fopenmp" 07_pin_modes.cu -o ~/p1/07_pin_modes
// Run with the model server STOPPED (it pins and fills up to 20 GiB of RAM, one style at a time):
//   ~/p1/07_pin_modes [--chunk-mib 144] [--max-mib 20480] [--reps 3] [--only STYLE:PTR] [--out <project>/logs/floors/pin_modes.json] [--force]
//   e.g. a quick first look:  ~/p1/07_pin_modes --max-mib 2304
// Syntax / logic check without a GPU (the CPU stand-in; it says nothing about the GPU):
//   g++ -std=c++17 -x c++ -O1 -Wall -Wextra -Werror -DPELIAS_CUDA_SHIM -I../lib 07_pin_modes.cu -o /tmp/07_shim   then   /tmp/07_shim --chunk-mib 2 --max-mib 16
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../lib/pl_cuda.h"

namespace {

constexpr unsigned long long kMix = 0x9E3779B97F4A7C15ULL;   // the pattern: word w holds (w + base) * kMix

// Reads n_uint4 16-byte items, compares each of the two 64-bit words with the pattern, adds the number of wrong words to *bad.
__global__ void k_verify_read(const uint4* __restrict__ p, size_t n_uint4, unsigned long long word_base, unsigned long long* bad) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    unsigned long long nbad = 0;
    for (size_t i = idx; i < n_uint4; i += stride) {
        const uint4 v = p[i];
        const unsigned long long e0 = (word_base + 2ULL * i) * kMix;
        const unsigned long long e1 = (word_base + 2ULL * i + 1ULL) * kMix;
        nbad += (v.x != (unsigned)e0) + (v.y != (unsigned)(e0 >> 32)) + (v.z != (unsigned)e1) + (v.w != (unsigned)(e1 >> 32));
    }
    if (nbad) atomicAdd(bad, nbad);
}

double mem_available_mib() {
    std::ifstream f("/proc/meminfo");
    std::string k;
    while (f >> k) {
        if (k == "MemAvailable:") { double kb = 0; f >> kb; return kb / 1024.0; }
        std::string rest;
        std::getline(f, rest);
    }
    return -1.0;
}

std::string hex(const void* p) {
    char b[32];
    std::snprintf(b, sizeof b, "0x%" PRIxPTR, reinterpret_cast<uintptr_t>(p));
    return b;
}

std::string jnum(double v) {                                 // JSON has no inf / nan
    if (!(v - v == 0.0)) return "null";
    char b[48];
    std::snprintf(b, sizeof b, "%.6g", v);
    return b;
}

std::string jstr(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (static_cast<unsigned char>(c) < 32) o += ' ';
        else o += c;
    }
    return o + "\"";
}

bool server_running() {
    FILE* p = popen("pgrep -x llama-server >/dev/null 2>&1; echo $?", "r");
    if (!p) return false;
    char buf[16] = {0};
    const bool ok = fgets(buf, sizeof buf, p) != nullptr;
    pclose(p);
    return ok && std::atoi(buf) == 0;
}

struct Combo { const char* style; const char* ptr; };
const Combo kCombos[] = {{"alloc_mapped", "dev"},        {"alloc_mapped", "host"},   {"reg_mapped_portable", "dev"}, {"reg_mapped_portable", "host"},
                         {"reg_mapped", "dev"},          {"reg_mapped", "host"},     {"reg_default", "host"}};

// ----------------------------------------------------------------------------------------------------------------------------------------------------------
// child: one (style, pointer) pair. Prints human lines and one final "RESULT {json}" line.
// ----------------------------------------------------------------------------------------------------------------------------------------------------------
int run_child(const std::string& style, const std::string& ptrmode, size_t chunk_mib, size_t max_mib, int reps) {
    const bool is_alloc = style == "alloc_mapped";
    unsigned reg_flags = 0;
    if (style == "reg_mapped_portable") reg_flags = cudaHostRegisterMapped | cudaHostRegisterPortable;
    else if (style == "reg_mapped") reg_flags = cudaHostRegisterMapped;
    else if (style == "reg_default") reg_flags = cudaHostRegisterDefault;
    else if (!is_alloc) { std::fprintf(stderr, "unknown style %s\n", style.c_str()); return 2; }
    const bool use_dev = ptrmode == "dev";

    const size_t chunk = chunk_mib << 20;
    const size_t n_chunks_wanted = std::max<size_t>(1, max_mib / chunk_mib);
    const std::string tag = style + "/" + ptrmode;

    cudaSetDeviceFlags(cudaDeviceMapHost);
    (void)cudaGetLastError();
    cudaStream_t st;
    PL_CUDA_CHECK(cudaStreamCreate(&st));
    cudaEvent_t e0, e1;
    PL_CUDA_CHECK(cudaEventCreate(&e0));
    PL_CUDA_CHECK(cudaEventCreate(&e1));
    unsigned long long* d_bad = nullptr;
    PL_CUDA_CHECK(cudaMalloc(&d_bad, sizeof(unsigned long long)));

    uint8_t* arena = nullptr;
    void* map_base = nullptr;
    size_t map_size = 0;
    if (!is_alloc) {
        map_size = n_chunks_wanted * chunk + (2u << 20);
        map_base = mmap(nullptr, map_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (map_base == MAP_FAILED) { std::printf("RESULT {\"style\":%s,\"ptr\":%s,\"error\":\"mmap failed\"}\n", jstr(style).c_str(), jstr(ptrmode).c_str()); return 0; }
        arena = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(map_base) + (2u << 20) - 1) & ~uintptr_t((2u << 20) - 1));
#ifdef MADV_HUGEPAGE
        (void)madvise(arena, n_chunks_wanted * chunk, MADV_HUGEPAGE);
#endif
    }

    std::vector<uint8_t*> host(n_chunks_wanted, nullptr);
    std::vector<void*> dev(n_chunks_wanted, nullptr);
    std::vector<std::string> dev_err(n_chunks_wanted);
    size_t got = 0;
    size_t n_identical = 0;
    bool delta_const = true;
    long long delta0 = 0;
    bool have_delta = false;
    std::string stop_reason;
    const double mem_before = mem_available_mib();
    double mem_at_stop = -1.0;
    std::string attr_text = "n/a (not an nvcc build)";

    std::vector<std::string> mile_json;
    std::vector<size_t> milestones;
    for (size_t m = 1; m < n_chunks_wanted; m *= 2) milestones.push_back(m);
    milestones.push_back(n_chunks_wanted);

    auto read_test = [&](size_t n, std::string* json) {
        PL_CUDA_CHECK(cudaMemset(d_bad, 0, sizeof(unsigned long long)));
        std::vector<double> gbs;
        std::string kerr;
        for (int rep = 0; rep < reps + 1; ++rep) {            // repetition 0 is a warm-up and is not reported
            PL_CUDA_CHECK(cudaEventRecord(e0, st));
            for (size_t c = 0; c < n; ++c) {
                const void* p = use_dev ? dev[c] : static_cast<const void*>(host[c]);
                if (p == nullptr) { kerr = "no device pointer for chunk " + std::to_string(c) + (dev_err[c].empty() ? "" : ": " + dev_err[c]); break; }
                PL_LAUNCH(k_verify_read, 256, 256, 0, st, reinterpret_cast<const uint4*>(p), chunk / 16, (unsigned long long)(c * (chunk / 8)), d_bad);
            }
            if (!kerr.empty()) break;
            PL_CUDA_CHECK(cudaEventRecord(e1, st));
            const cudaError_t es = cudaEventSynchronize(e1);
            const cudaError_t el = cudaGetLastError();
            if (es != cudaSuccess || el != cudaSuccess) { kerr = std::string("kernel/stream error: ") + cudaGetErrorString(es != cudaSuccess ? es : el); break; }
            float ms = 0.f;
            PL_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            if (rep > 0) gbs.push_back(double(n) * double(chunk) / 1e9 / (ms / 1e3));
        }
        unsigned long long bad = 0;
        const cudaError_t ec = cudaMemcpy(&bad, d_bad, sizeof bad, cudaMemcpyDeviceToHost);
        if (ec != cudaSuccess && kerr.empty()) kerr = std::string("copy of the mismatch counter failed: ") + cudaGetErrorString(ec);
        std::vector<double> sorted = gbs;
        std::sort(sorted.begin(), sorted.end());
        const double med = sorted.empty() ? 0.0 : sorted[sorted.size() / 2];
        std::ostringstream j;
        j << "{\"chunks\":" << n << ",\"mib\":" << n * chunk_mib << ",\"identical_addresses\":" << n_identical << ",\"mismatched_32bit_values\":" << bad << ",\"read_gb_s\":[";
        for (size_t i = 0; i < gbs.size(); ++i) j << (i ? "," : "") << jnum(gbs[i]);
        j << "],\"median_gb_s\":" << jnum(med) << ",\"error\":" << jstr(kerr) << "}";
        *json = j.str();
        std::printf("  %-26s chunks %4zu (%6zu MiB)  addresses identical %zu of %zu   GPU read: ", tag.c_str(), n, n * chunk_mib, n_identical, n);
        if (gbs.empty()) std::printf("none");
        for (double g : gbs) std::printf("%.2f ", g);
        if (!gbs.empty()) std::printf("GB/s, median %.2f", med);
        std::printf("   mismatched 32-bit values %llu%s%s\n", bad, kerr.empty() ? "" : "   ERROR: ", kerr.c_str());
        return kerr.empty();
    };

    // the pattern the GPU will check: word w of chunk c holds (c * chunk/8 + w) * kMix
    auto fill = [&](uint8_t* h, size_t c) {
        uint64_t* q = reinterpret_cast<uint64_t*>(h);
        const uint64_t base = static_cast<uint64_t>(c) * (chunk / 8);
        const long long nw = static_cast<long long>(chunk / 8);
#pragma omp parallel for schedule(static)
        for (long long i = 0; i < nw; ++i) q[i] = (base + static_cast<uint64_t>(i)) * kMix;
    };

    bool read_ok = true;
    size_t next_mile = 0;
    size_t last_read_chunks = 0;
    for (size_t c = 0; c < n_chunks_wanted; ++c) {
        uint8_t* h = nullptr;
        cudaError_t err = cudaSuccess;
        if (is_alloc) {
            void* p = nullptr;
            err = cudaHostAlloc(&p, chunk, cudaHostAllocMapped | cudaHostAllocPortable);
            h = static_cast<uint8_t*>(p);
            if (err == cudaSuccess) fill(h, c);
        } else {
            h = arena + c * chunk;
            fill(h, c);                                                    // populated first, then registered: the order cold_weights_view uses
            err = cudaHostRegister(h, chunk, reg_flags);
        }
        if (err != cudaSuccess) {
            stop_reason = std::string(is_alloc ? "cudaHostAlloc" : "cudaHostRegister") + " of chunk " + std::to_string(c) + " failed: " + cudaGetErrorString(err);
            (void)cudaGetLastError();
            mem_at_stop = mem_available_mib();
            break;
        }
        host[c] = h;
        void* d = nullptr;
        const cudaError_t ed = cudaHostGetDevicePointer(&d, h, 0);
        if (ed == cudaSuccess) {
            dev[c] = d;
            if (d == static_cast<void*>(h)) ++n_identical;
            const long long delta = static_cast<long long>(reinterpret_cast<uintptr_t>(d)) - static_cast<long long>(reinterpret_cast<uintptr_t>(h));
            if (!have_delta) { delta0 = delta; have_delta = true; }
            else if (delta != delta0) delta_const = false;
        } else {
            dev_err[c] = cudaGetErrorString(ed);
            (void)cudaGetLastError();
        }
        if (c < 3) std::printf("  %-26s chunk %zu  host %s  device %s%s%s\n", tag.c_str(), c, hex(h).c_str(), ed == cudaSuccess ? hex(d).c_str() : "(none)", ed == cudaSuccess ? "" : "  cudaHostGetDevicePointer: ", ed == cudaSuccess ? "" : dev_err[c].c_str());
        if (c == 0) {
#if defined(__CUDACC__)
            cudaPointerAttributes a;
            const cudaError_t ea = cudaPointerGetAttributes(&a, h);
            if (ea == cudaSuccess) {
                static const char* names[] = {"unregistered", "host", "device", "managed"};
                attr_text = std::string("type ") + (static_cast<int>(a.type) >= 0 && static_cast<int>(a.type) <= 3 ? names[static_cast<int>(a.type)] : "?") + ", devicePointer " + hex(a.devicePointer) + ", hostPointer " + hex(a.hostPointer);
            } else {
                attr_text = std::string("error: ") + cudaGetErrorString(ea);
                (void)cudaGetLastError();
            }
            std::printf("  %-26s cudaPointerGetAttributes(chunk 0 host address): %s\n", tag.c_str(), attr_text.c_str());
#endif
        }
        got = c + 1;
        if (next_mile < milestones.size() && got == milestones[next_mile]) {
            std::string j;
            read_ok = read_test(got, &j);
            last_read_chunks = got;
            mile_json.push_back(j);
            ++next_mile;
            if (!read_ok) { stop_reason = "read test failed at " + std::to_string(got) + " chunks; stopping this combination"; break; }
        }
    }
    if (read_ok && got > 0 && last_read_chunks != got) {
        std::string j;                                                        // the ceiling was hit between milestones: read what is pinned
        read_test(got, &j);
        mile_json.push_back(j);
    }
    if (!stop_reason.empty()) std::printf("  %-26s stopped after %zu of %zu chunks (%zu MiB): %s   MemAvailable %.0f MiB before, %.0f MiB at the stop\n", tag.c_str(), got, n_chunks_wanted, got * chunk_mib,
                                           stop_reason.c_str(), mem_before, mem_at_stop);

    std::ostringstream js;
    js << "{\"style\":" << jstr(style) << ",\"ptr\":" << jstr(ptrmode) << ",\"chunk_mib\":" << chunk_mib << ",\"chunks_wanted\":" << n_chunks_wanted << ",\"chunks_reached\":" << got
       << ",\"mib_reached\":" << got * chunk_mib << ",\"identical_addresses\":" << n_identical << ",\"device_minus_host_constant\":" << (delta_const ? "true" : "false")
       << ",\"device_minus_host_chunk0\":" << delta0 << ",\"pointer_attributes_chunk0\":" << jstr(attr_text) << ",\"stop_reason\":" << jstr(stop_reason)
       << ",\"mem_available_mib_before\":" << jnum(mem_before) << ",\"mem_available_mib_at_stop\":" << jnum(mem_at_stop) << ",\"milestones\":[";
    for (size_t i = 0; i < mile_json.size(); ++i) js << (i ? "," : "") << mile_json[i];
    js << "]}";
    std::printf("RESULT %s\n", js.str().c_str());
    std::fflush(stdout);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    size_t chunk_mib = 144, max_mib = 20480;
    int reps = 3;
    bool force = false;
    std::string only, child, out_path = "<project>/logs/floors/pin_modes.json";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--chunk-mib" && i + 1 < argc) chunk_mib = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--max-mib" && i + 1 < argc) max_mib = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (a == "--only" && i + 1 < argc) only = argv[++i];
        else if (a == "--child" && i + 1 < argc) child = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (a == "--force") force = true;
        else { std::fprintf(stderr, "usage: %s [--chunk-mib N] [--max-mib N] [--reps N] [--only STYLE:PTR] [--out FILE] [--force]\n", argv[0]); return 2; }
    }
    if (chunk_mib == 0 || chunk_mib % 2 != 0 || max_mib < chunk_mib || reps < 1) {
        std::fprintf(stderr, "--chunk-mib must be a positive even number (2 MiB pages), --max-mib >= --chunk-mib, --reps >= 1\n");
        return 2;
    }
    if (!child.empty()) {
        const size_t colon = child.find(':');
        if (colon == std::string::npos) return 2;
        return run_child(child.substr(0, colon), child.substr(colon + 1), chunk_mib, max_mib, reps);
    }
    if (!force && server_running()) {
        std::fprintf(stderr, "llama-server is running; stop it first (llama --stop) or pass --force.\n");
        return 3;
    }

    char self[4096] = {0};
    const ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
    if (n <= 0) { std::fprintf(stderr, "cannot find my own path\n"); return 2; }
    struct rlimit rl;
    getrlimit(RLIMIT_MEMLOCK, &rl);
    const double mem0 = mem_available_mib();
    std::printf("[2f] chunk %zu MiB, up to %zu MiB (%zu chunks) per combination, %d timed repetitions; MemAvailable %.0f MiB; ulimit -l %s\n", chunk_mib, max_mib, max_mib / chunk_mib, reps, mem0,
                rl.rlim_cur == RLIM_INFINITY ? "unlimited" : (std::to_string(rl.rlim_cur >> 10) + " KiB").c_str());

    std::vector<std::string> results;
    for (const Combo& cb : kCombos) {
        const std::string name = std::string(cb.style) + ":" + cb.ptr;
        if (!only.empty() && only != name) continue;
        std::printf("\n[2f] --- %s ---\n", name.c_str());
        std::fflush(stdout);
        std::string cmd = std::string("\"") + self + "\" --child " + name + " --chunk-mib " + std::to_string(chunk_mib) + " --max-mib " + std::to_string(max_mib) + " --reps " + std::to_string(reps) + " 2>&1";
        FILE* p = popen(cmd.c_str(), "r");
        std::string res;
        if (p) {
            char line[16384];
            while (fgets(line, sizeof line, p)) {
                if (std::strncmp(line, "RESULT ", 7) == 0) { res = line + 7; while (!res.empty() && (res.back() == '\n' || res.back() == '\r')) res.pop_back(); }
                else std::fputs(line, stdout);
            }
            const int status = pclose(p);
            if (res.empty()) {
                std::string why = WIFSIGNALED(status) ? "killed by signal " + std::to_string(WTERMSIG(status)) : "exit code " + std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1);
                std::printf("  this combination produced no result (%s); the lines above are everything it printed\n", why.c_str());
                res = std::string("{\"style\":") + jstr(cb.style) + ",\"ptr\":" + jstr(cb.ptr) + ",\"error\":" + jstr("no result: " + why) + "}";
            }
        } else {
            res = std::string("{\"style\":") + jstr(cb.style) + ",\"ptr\":" + jstr(cb.ptr) + ",\"error\":\"popen failed\"}";
        }
        results.push_back(res);
        std::fflush(stdout);
    }

    std::ostringstream js;
    js << "{\"probe\":\"07_pin_modes\",\"chunk_mib\":" << chunk_mib << ",\"max_mib\":" << max_mib << ",\"reps\":" << reps << ",\"memlock_limit_kib\":"
       << (rl.rlim_cur == RLIM_INFINITY ? -1LL : static_cast<long long>(rl.rlim_cur >> 10)) << ",\"mem_available_mib_at_start\":" << jnum(mem0) << ",\"results\":[";
    for (size_t i = 0; i < results.size(); ++i) js << (i ? ",\n" : "\n") << results[i];
    js << "\n]}\n";
    std::ofstream out(out_path);
    if (out) { out << js.str(); std::printf("\n[2f] wrote %s\n", out_path.c_str()); }
    else std::fprintf(stderr, "cannot write %s; JSON follows\n%s", out_path.c_str(), js.str().c_str());
    return 0;
}
