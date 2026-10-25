// File: <project>\moe_track\pelias_serve\cold_weights_view.cpp
// See cold_weights_view.h for what this is and why. Standard C++17 + POSIX; CUDA only behind -DPELIAS_COLD_VIEW_CUDA (real runtime) or -DPELIAS_CUDA_SHIM (stand-in).
#include "cold_weights_view.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(PELIAS_CUDA_SHIM)
#include "../lib/cuda_shim.h"
#define PELIAS_VIEW_CUDA 1
#elif defined(PELIAS_COLD_VIEW_CUDA)
#include <cuda_runtime.h>
#define PELIAS_VIEW_CUDA 1
#else
#define PELIAS_VIEW_CUDA 0
#endif

namespace pelias {

bool cold_view_built_with_cuda() { return PELIAS_VIEW_CUDA != 0; }

// ---------------------------------------------------------------------------------------------------------------------------------------------------------
// quantisation geometry (the same table lib/gguf_min.py uses; tests/test_gguf_min.py checks that one against gguf-py, test_cold_weights_view checks this one against ggml)
// ---------------------------------------------------------------------------------------------------------------------------------------------------------
static const QuantGeom kGeom[] = {
    {0, "F32", 1, 4},      {1, "F16", 1, 2},      {2, "Q4_0", 32, 18},   {3, "Q4_1", 32, 20},   {6, "Q5_0", 32, 22},   {7, "Q5_1", 32, 24},
    {8, "Q8_0", 32, 34},   {10, "Q2_K", 256, 84}, {11, "Q3_K", 256, 110}, {12, "Q4_K", 256, 144}, {13, "Q5_K", 256, 176}, {14, "Q6_K", 256, 210},
    {16, "IQ2_XXS", 256, 66}, {17, "IQ2_XS", 256, 74}, {18, "IQ3_XXS", 256, 98}, {19, "IQ1_S", 256, 50}, {20, "IQ4_NL", 32, 18}, {21, "IQ3_S", 256, 110},
    {22, "IQ2_S", 256, 82}, {23, "IQ4_XS", 256, 136}, {29, "IQ1_M", 256, 56}, {30, "BF16", 1, 2},
};

const QuantGeom* quant_geom(int32_t type_id) {
    for (const QuantGeom& g : kGeom)
        if (g.type_id == type_id) return &g;
    return nullptr;
}

const GgufTensorInfo* GgufTable::find(const std::string& name) const {
    for (const GgufTensorInfo& t : tensors)
        if (t.name == name) return &t;
    return nullptr;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------------------------------------------------------------------------------------
namespace {

uint64_t align_up_u64(uint64_t v, uint64_t a) { return a == 0 ? v : (v + a - 1) / a * a; }
size_t   align_up_sz(size_t v, size_t a) { return (v + a - 1) / a * a; }

std::string errno_text(const char* what, const std::string& path) {
    return std::string(what) + " '" + path + "': " + std::strerror(errno);
}

bool pread_full(int fd, uint8_t* dst, size_t len, uint64_t off, std::string* err) {
    size_t done = 0;
    while (done < len) {
        const ssize_t n = ::pread(fd, dst + done, len - done, static_cast<off_t>(off + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            if (err) *err = std::string("read failed: ") + std::strerror(errno);
            return false;
        }
        if (n == 0) {
            if (err) *err = "unexpected end of file";
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

// bounds-checked little-endian reader (the GGUF format is little-endian; x86 is too)
struct Rd {
    Rd(const uint8_t* p_, size_t n_) : p(p_), n(n_) {}
    const uint8_t* p;
    size_t         n;
    size_t         pos = 0;
    bool           trunc = false;       // ran past the end of the buffer
    bool           bad = false;         // structurally wrong
    std::string    why;

    bool need(uint64_t k) {
        if (trunc || bad) return false;
        if (k > n - pos) { trunc = true; return false; }
        return true;
    }
    template <class T> T get() {
        T v{};
        if (!need(sizeof(T))) return T{};
        std::memcpy(&v, p + pos, sizeof(T));
        pos += sizeof(T);
        return v;
    }
    bool str(std::string* s) {
        const uint64_t len = get<uint64_t>();
        if (trunc || bad || !need(len)) return false;
        if (s) s->assign(reinterpret_cast<const char*>(p + pos), static_cast<size_t>(len));
        pos += static_cast<size_t>(len);
        return true;
    }
    bool skip(uint64_t k) {
        if (!need(k)) return false;
        pos += static_cast<size_t>(k);
        return true;
    }
    void fail(const std::string& m) { if (!bad) { bad = true; why = m; } }
};

enum : uint32_t { T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 };

int scalar_size(uint32_t t) {
    switch (t) {
        case T_U8: case T_I8: case T_BOOL: return 1;
        case T_U16: case T_I16: return 2;
        case T_U32: case T_I32: case T_F32: return 4;
        case T_U64: case T_I64: case T_F64: return 8;
        default: return -1;
    }
}

bool skip_value(Rd& r, uint32_t t, int depth) {
    if (t == T_STR) return r.str(nullptr);
    if (t == T_ARR) {
        const uint32_t et = r.get<uint32_t>();
        const uint64_t cnt = r.get<uint64_t>();
        if (r.trunc || r.bad) return false;
        if (depth > 2) { r.fail("metadata arrays nested too deeply"); return false; }
        const int sz = scalar_size(et);
        if (sz > 0) {
            if (cnt > UINT64_MAX / static_cast<uint64_t>(sz)) { r.fail("metadata array too large"); return false; }
            return r.skip(cnt * static_cast<uint64_t>(sz));
        }
        for (uint64_t i = 0; i < cnt; ++i)
            if (!skip_value(r, et, depth + 1)) return false;
        return true;
    }
    const int sz = scalar_size(t);
    if (sz < 0) { r.fail("unknown metadata value type " + std::to_string(t)); return false; }
    return r.skip(static_cast<uint64_t>(sz));
}

// 0 = parsed, 1 = the buffer ended early (retry with more), 2 = bad file (*err set)
int parse_header(const uint8_t* buf, size_t n, uint64_t file_size, GgufTable* out, std::string* err) {
    Rd r(buf, n);
    if (n < 4 || std::memcmp(buf, "GGUF", 4) != 0) { if (err) *err = "not a GGUF file (bad magic)"; return 2; }
    r.pos = 4;
    GgufTable t;
    t.file_size = file_size;
    t.version = r.get<uint32_t>();
    if (!r.trunc && t.version != 2 && t.version != 3) { if (err) *err = "unsupported GGUF version " + std::to_string(t.version); return 2; }
    const uint64_t n_tensors = r.get<uint64_t>();
    const uint64_t n_kv = r.get<uint64_t>();
    if (r.trunc) return 1;
    if (n_tensors > (1u << 24) || n_kv > (1u << 24)) { if (err) *err = "implausible tensor/metadata count"; return 2; }

    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key;
        if (!r.str(&key)) break;
        const uint32_t vt = r.get<uint32_t>();
        if (r.trunc || r.bad) break;
        if (key == "general.alignment" && (vt == T_U32 || vt == T_I32)) {
            const uint32_t a = r.get<uint32_t>();
            if (r.trunc) break;
            t.alignment = a;
        } else if (key == "general.alignment" && (vt == T_U64 || vt == T_I64)) {
            const uint64_t a = r.get<uint64_t>();
            if (r.trunc) break;
            t.alignment = static_cast<uint32_t>(a);
        } else if (key == "general.architecture" && vt == T_STR) {
            if (!r.str(&t.architecture)) break;
        } else if (!skip_value(r, vt, 0)) {
            break;
        }
    }
    if (r.bad) { if (err) *err = "metadata: " + r.why; return 2; }
    if (r.trunc) return 1;
    if (t.alignment == 0) { if (err) *err = "general.alignment is 0"; return 2; }

    t.tensors.reserve(static_cast<size_t>(n_tensors));
    for (uint64_t i = 0; i < n_tensors; ++i) {
        GgufTensorInfo ti;
        if (!r.str(&ti.name)) break;
        const uint32_t nd = r.get<uint32_t>();
        if (r.trunc) break;
        if (nd == 0 || nd > 4) { if (err) *err = "tensor '" + ti.name + "' has " + std::to_string(nd) + " dimensions"; return 2; }
        ti.n_dims = static_cast<int32_t>(nd);
        for (uint32_t d = 0; d < nd; ++d) ti.dims[d] = r.get<uint64_t>();
        ti.type_id = static_cast<int32_t>(r.get<uint32_t>());
        ti.data_rel_offset = r.get<uint64_t>();
        if (r.trunc) break;
        t.tensors.push_back(std::move(ti));
    }
    if (r.trunc) return 1;
    t.data_start = align_up_u64(r.pos, t.alignment);

    for (GgufTensorInfo& ti : t.tensors) {
        // the stored offset is untrusted: check it before adding (a value near 2^64 would wrap around to a small, valid-looking offset)
        if (t.data_start > file_size || ti.data_rel_offset > file_size - t.data_start) {
            if (err) *err = "tensor '" + ti.name + "' starts past the end of the file (stored offset " + std::to_string(ti.data_rel_offset) + ", data starts at " + std::to_string(t.data_start) + ", file " + std::to_string(file_size) + ")";
            return 2;
        }
        ti.file_offset = t.data_start + ti.data_rel_offset;
        const QuantGeom* g = quant_geom(ti.type_id);
        if (!g) { ti.n_bytes = 0; continue; }                           // unknown type: not usable, not an error unless somebody asks for it
        uint64_t elems = 1;
        for (int d = 0; d < ti.n_dims; ++d)
            if (__builtin_mul_overflow(elems, ti.dims[d], &elems)) { if (err) *err = "tensor '" + ti.name + "' is implausibly large (element count overflows)"; return 2; }
        if (elems % static_cast<unsigned>(g->block_elems) != 0 || ti.dims[0] % static_cast<unsigned>(g->block_elems) != 0) {
            if (err) *err = "tensor '" + ti.name + "': row length " + std::to_string(ti.dims[0]) + " is not a multiple of the " + g->name + " block size";
            return 2;
        }
        const unsigned __int128 bytes = static_cast<unsigned __int128>(elems / static_cast<unsigned>(g->block_elems)) * static_cast<unsigned>(g->block_bytes);
        if (bytes > UINT64_MAX) { if (err) *err = "tensor '" + ti.name + "' is implausibly large"; return 2; }
        ti.n_bytes = static_cast<uint64_t>(bytes);
        if (ti.file_offset > file_size || ti.n_bytes > file_size - ti.file_offset) {
            if (err) *err = "tensor '" + ti.name + "' extends past the end of the file (offset " + std::to_string(ti.file_offset) + ", " + std::to_string(ti.n_bytes) + " bytes, file " + std::to_string(file_size) + ")";
            return 2;
        }
    }
    *out = std::move(t);
    return 0;
}

size_t huge_backed_bytes(const uint8_t* base, size_t size) {
    FILE* f = std::fopen("/proc/self/smaps", "r");
    if (!f) return 0;
    const uintptr_t a = reinterpret_cast<uintptr_t>(base), b = a + size;
    uintptr_t lo = 0, hi = 0;
    size_t total = 0;
    char line[512];
    while (std::fgets(line, sizeof line, f)) {
        unsigned long s = 0, e = 0;
        char perms[8] = {0};
        if (std::sscanf(line, "%lx-%lx %7s", &s, &e, perms) == 3) { lo = s; hi = e; continue; }
        if (std::strncmp(line, "AnonHugePages:", 14) == 0 && lo < b && hi > a) {
            unsigned long kb = 0;
            std::sscanf(line + 14, "%lu", &kb);
            const size_t overlap = static_cast<size_t>(std::min<uintptr_t>(hi, b) - std::max<uintptr_t>(lo, a));
            total += std::min(static_cast<size_t>(kb) * 1024, overlap);
        }
    }
    std::fclose(f);
    return total;
}

const char* kWhichName[3] = {"gate", "up", "down"};

}  // namespace

// ---------------------------------------------------------------------------------------------------------------------------------------------------------
bool parse_gguf_table(const std::string& path, GgufTable* out, std::string* err) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) { if (err) *err = errno_text("cannot open", path); return false; }
    struct stat st;
    if (::fstat(fd, &st) != 0) { if (err) *err = errno_text("cannot stat", path); ::close(fd); return false; }
    const uint64_t file_size = static_cast<uint64_t>(st.st_size);
    uint64_t cap = std::min<uint64_t>(file_size, static_cast<uint64_t>(16) << 20);
    std::vector<uint8_t> buf;
    bool ok = false;
    for (;;) {
        buf.resize(static_cast<size_t>(cap));
        std::string e;
        if (!pread_full(fd, buf.data(), buf.size(), 0, &e)) { if (err) *err = path + ": " + e; break; }
        const int rc = parse_header(buf.data(), buf.size(), file_size, out, err);
        if (rc == 0) { ok = true; break; }
        if (rc == 2) break;
        if (cap >= file_size || cap >= (static_cast<uint64_t>(1) << 30)) { if (err) *err = path + ": header ends past the readable range (truncated file?)"; break; }
        cap = std::min<uint64_t>(file_size, cap * 4);
    }
    ::close(fd);
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------
ColdWeightsView::ColdWeightsView() = default;
ColdWeightsView::~ColdWeightsView() { close(); }

bool ColdWeightsView::has_layer(int layer) const {
    return std::find(layer_ids_.begin(), layer_ids_.end(), layer) != layer_ids_.end();
}

const ColdLayerView& ColdWeightsView::layer(int layer) const {
    const auto it = std::find(layer_ids_.begin(), layer_ids_.end(), layer);
    if (it == layer_ids_.end()) std::abort();                       // caller broke the has_layer() contract
    return layers_[static_cast<size_t>(it - layer_ids_.begin())];
}

bool ColdWeightsView::open(const ColdViewOptions& opt, std::string* err) {
    close();
    auto fail = [&](const std::string& m) { if (err) *err = m; close(); return false; };
    opt_ = opt;
    std::string e;
    if (!parse_gguf_table(opt.gguf_path, &table_, &e)) return fail(e);

    // ---- which layers
    auto tname = [](int layer, int which) { return "blk." + std::to_string(layer) + ".ffn_" + kWhichName[which] + "_exps.weight"; };
    int max_present = -1;
    for (const GgufTensorInfo& ti : table_.tensors) {
        int l = -1;
        char tail[64] = {0};
        if (std::sscanf(ti.name.c_str(), "blk.%d.%63s", &l, tail) == 2 && std::strcmp(tail, "ffn_gate_exps.weight") == 0) max_present = std::max(max_present, l);
    }
    if (max_present < 0) return fail("no blk.N.ffn_gate_exps.weight tensor in " + opt.gguf_path + " (separate gate/up expert tensors are required; a fused ffn_gate_up_exps layout is not supported yet)");
    int last = opt.last_layer >= 0 ? opt.last_layer : max_present;
    if (opt.last_layer < 0 && !opt.include_mtp) last = std::min(last, N_LAYERS - 1);           // default: the 40 decoder layers, not the MTP block
    if (last >= N_LAYERS && !opt.include_mtp) return fail("layer " + std::to_string(last) + " is the MTP block (the model has " + std::to_string(N_LAYERS) + " decoder layers); set include_mtp to load it");
    if (opt.first_layer < 0 || last < opt.first_layer) return fail("empty layer range");
    for (int l = opt.first_layer; l <= last; ++l) layer_ids_.push_back(l);

    // ---- shapes
    const GgufTensorInfo* sel[64][3];
    if (layer_ids_.size() > 64) return fail("more than 64 layers requested");
    int64_t hidden = 0, ff = 0, n_exp = 0;
    for (size_t i = 0; i < layer_ids_.size(); ++i) {
        for (int w = 0; w < 3; ++w) {
            const GgufTensorInfo* ti = table_.find(tname(layer_ids_[i], w));
            if (!ti) return fail("missing tensor " + tname(layer_ids_[i], w));
            const QuantGeom* g = quant_geom(ti->type_id);
            if (!g) return fail("tensor " + ti->name + " has a type (" + std::to_string(ti->type_id) + ") this reader does not know");
            if (ti->n_dims != 3) return fail("tensor " + ti->name + " is not 3-dimensional");
            sel[i][w] = ti;
        }
        const GgufTensorInfo *g_ = sel[i][0], *u_ = sel[i][1], *d_ = sel[i][2];
        if (g_->dims[0] != u_->dims[0] || g_->dims[1] != u_->dims[1] || g_->dims[2] != u_->dims[2])
            return fail("blk." + std::to_string(layer_ids_[i]) + ": gate and up expert tensors differ in shape");
        if (d_->dims[0] != g_->dims[1] || d_->dims[1] != g_->dims[0] || d_->dims[2] != g_->dims[2])
            return fail("blk." + std::to_string(layer_ids_[i]) + ": down tensor shape does not match gate/up (expected " + std::to_string(g_->dims[1]) + " x " + std::to_string(g_->dims[0]) + ")");
        if (i == 0) { hidden = static_cast<int64_t>(g_->dims[0]); ff = static_cast<int64_t>(g_->dims[1]); n_exp = static_cast<int64_t>(g_->dims[2]); }
        else if (static_cast<int64_t>(g_->dims[0]) != hidden || static_cast<int64_t>(g_->dims[1]) != ff || static_cast<int64_t>(g_->dims[2]) != n_exp)
            return fail("blk." + std::to_string(layer_ids_[i]) + ": expert shape differs from the first layer");
    }
    if (opt.strict_shape && (hidden != HIDDEN || ff != EXPERT_FF || n_exp != N_EXPERTS))
        return fail("expert shape " + std::to_string(hidden) + " x " + std::to_string(ff) + " x " + std::to_string(n_exp) + " is not the qwen35moe geometry (" + std::to_string(HIDDEN) + " x " + std::to_string(EXPERT_FF) + " x " + std::to_string(N_EXPERTS) + "); pass strict_shape=false for another model");
    n_experts_ = n_exp;

    // ---- arena plan: one 2 MiB-aligned region per tensor
    size_t total = 0;
    for (size_t i = 0; i < layer_ids_.size(); ++i)
        for (int w = 0; w < 3; ++w) {
            const GgufTensorInfo* ti = sel[i][w];
            Region rg;
            rg.ptr = reinterpret_cast<const uint8_t*>(total);          // arena offset for now, fixed up after the allocation
            rg.bytes = static_cast<size_t>(ti->n_bytes);
            rg.padded = align_up_sz(rg.bytes, COLD_HUGE_PAGE);
            rg.file_offset = ti->file_offset;
            rg.layer = layer_ids_[i];
            rg.which = w;
            rg.pinned = false;
            rg.dev = nullptr;
            regions_.push_back(rg);
            total += rg.padded;
        }
    arena_size_ = total;
    map_size_ = total + COLD_HUGE_PAGE;
    map_base_ = ::mmap(nullptr, map_size_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (map_base_ == MAP_FAILED) { map_base_ = nullptr; return fail("cannot reserve " + std::to_string(total >> 20) + " MiB of memory: " + std::strerror(errno)); }
    arena_ = reinterpret_cast<uint8_t*>(align_up_sz(reinterpret_cast<size_t>(map_base_), COLD_HUGE_PAGE));
#ifdef MADV_HUGEPAGE
    if (opt.advise_hugepages) (void)::madvise(arena_, arena_size_, MADV_HUGEPAGE);        // failure is fine: the memory just stays 4 KB pages
#endif
    for (Region& rg : regions_) rg.ptr = arena_ + reinterpret_cast<size_t>(rg.ptr);

    // ---- copy the tensors in, in parallel chunks
    const auto t0 = std::chrono::steady_clock::now();
    const int fd = ::open(opt.gguf_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return fail(errno_text("cannot open", opt.gguf_path));
    struct Job { size_t region; uint64_t off; size_t len; };
    std::vector<Job> jobs;
    const size_t chunk = static_cast<size_t>(32) << 20;
    for (size_t i = 0; i < regions_.size(); ++i)
        for (size_t o = 0; o < regions_[i].bytes; o += chunk) jobs.push_back({i, o, std::min(chunk, regions_[i].bytes - o)});
    std::atomic<size_t> next{0};
    std::atomic<bool> failed{false};
    std::mutex mu;
    std::string first_err;
    auto worker = [&]() {
        for (;;) {
            const size_t j = next.fetch_add(1);
            if (j >= jobs.size() || failed.load()) return;
            const Job& jb = jobs[j];
            std::string e2;
            if (!pread_full(fd, const_cast<uint8_t*>(regions_[jb.region].ptr) + jb.off, jb.len, regions_[jb.region].file_offset + jb.off, &e2)) {
                std::lock_guard<std::mutex> lk(mu);
                if (!failed.exchange(true)) first_err = e2;
                return;
            }
        }
    };
    const int nth = std::max(1, opt.load_threads);
    std::vector<std::thread> th;
    for (int i = 1; i < nth; ++i) th.emplace_back(worker);
    worker();
    for (auto& t : th) t.join();
    ::close(fd);
    if (failed.load()) return fail("reading " + opt.gguf_path + ": " + first_err);
    stats_.load_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // ---- the views
    for (size_t i = 0; i < layer_ids_.size(); ++i) {
        ColdLayerView lv;
        lv.layer = layer_ids_[i];
        ColdMat* mats[3] = {&lv.gate, &lv.up, &lv.down};
        for (int w = 0; w < 3; ++w) {
            const GgufTensorInfo* ti = sel[i][w];
            const QuantGeom* g = quant_geom(ti->type_id);
            ColdMat& m = *mats[w];
            m.base = regions_[i * 3 + static_cast<size_t>(w)].ptr;
            m.type_id = ti->type_id;
            m.cols = static_cast<int64_t>(ti->dims[0]);
            m.rows = static_cast<int64_t>(ti->dims[1]);
            m.n_experts = static_cast<int64_t>(ti->dims[2]);
            m.row_bytes = static_cast<size_t>(ti->dims[0] / static_cast<uint64_t>(g->block_elems)) * static_cast<size_t>(g->block_bytes);
            m.expert_stride = static_cast<size_t>(m.rows) * m.row_bytes;
            m.gpu_ok = false;
            stats_.tensor_bytes += static_cast<size_t>(ti->n_bytes);
        }
        layers_.push_back(lv);
    }
    stats_.n_layers = layer_ids_.size();
    stats_.arena_bytes = arena_size_;
    stats_.huge_backed_bytes = huge_backed_bytes(arena_, arena_size_);

    if (opt.verify_after_load && !verify_against_file(&e, nth)) return fail("verification failed: " + e);
    return true;
}

bool ColdWeightsView::verify_against_file(std::string* err, int threads) const {
    if (!arena_) { if (err) *err = "not open"; return false; }
    const int fd = ::open(opt_.gguf_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) { if (err) *err = errno_text("cannot open", opt_.gguf_path); return false; }
    struct Job { size_t region; uint64_t off; size_t len; };
    std::vector<Job> jobs;
    const size_t chunk = static_cast<size_t>(8) << 20;
    for (size_t i = 0; i < regions_.size(); ++i)
        for (size_t o = 0; o < regions_[i].bytes; o += chunk) jobs.push_back({i, o, std::min(chunk, regions_[i].bytes - o)});
    std::atomic<size_t> next{0};
    std::atomic<bool> bad{false};
    std::mutex mu;
    std::string first;
    auto worker = [&]() {
        std::vector<uint8_t> tmp(chunk);
        for (;;) {
            const size_t j = next.fetch_add(1);
            if (j >= jobs.size() || bad.load()) return;
            const Job& jb = jobs[j];
            std::string e2;
            const Region& rg = regions_[jb.region];
            if (!pread_full(fd, tmp.data(), jb.len, rg.file_offset + jb.off, &e2) || std::memcmp(tmp.data(), rg.ptr + jb.off, jb.len) != 0) {
                std::lock_guard<std::mutex> lk(mu);
                if (!bad.exchange(true)) first = e2.empty() ? "blk." + std::to_string(rg.layer) + " " + kWhichName[rg.which] + ": bytes differ from the file at tensor offset " + std::to_string(jb.off) : e2;
                return;
            }
        }
    };
    std::vector<std::thread> th;
    for (int i = 1; i < std::max(1, threads); ++i) th.emplace_back(worker);
    worker();
    for (auto& t : th) t.join();
    ::close(fd);
    if (bad.load()) { if (err) *err = first; return false; }
    return true;
}

bool ColdWeightsView::pin(size_t max_bytes, ColdPinReport* report) {
    ColdPinReport rep;
#if PELIAS_VIEW_CUDA
    if (!arena_) { rep.first_error = "not open"; if (report) *report = rep; return false; }
    (void)cudaSetDeviceFlags(cudaDeviceMapHost);              // must precede the first context on some drivers; "already set" is fine
    (void)cudaGetLastError();
    bool stop = false;
    for (Region& rg : regions_) {
        if (stop || (max_bytes != 0 && rep.pinned_bytes + rg.padded > max_bytes)) { ++rep.regions_skipped; continue; }
        cudaError_t e = cudaHostRegister(const_cast<uint8_t*>(rg.ptr), rg.padded, cudaHostRegisterMapped | cudaHostRegisterPortable);
        if (e != cudaSuccess) {
            ++rep.regions_failed;
            rep.first_error = std::string("cudaHostRegister(blk.") + std::to_string(rg.layer) + " " + kWhichName[rg.which] + ", " + std::to_string(rg.padded >> 20) + " MiB): " + cudaGetErrorString(e);
            (void)cudaGetLastError();
            stop = true;                                       // a driver limit will not get better with the next region
            continue;
        }
        void* dptr = nullptr;
        e = cudaHostGetDevicePointer(&dptr, const_cast<uint8_t*>(rg.ptr), 0);
        if (e != cudaSuccess || dptr == nullptr) {
            ++rep.regions_failed;
            if (rep.first_error.empty())
                rep.first_error = e != cudaSuccess ? std::string("cudaHostGetDevicePointer: ") + cudaGetErrorString(e) : "cudaHostGetDevicePointer returned a null device pointer";
            (void)cudaHostUnregister(const_cast<uint8_t*>(rg.ptr));
            (void)cudaGetLastError();
            continue;
        }
        // The device pointer may differ from the host pointer (it does for cudaHostRegister memory on WSL2). It is kept per region and used for every GPU access.
        const int64_t off = static_cast<int64_t>(reinterpret_cast<intptr_t>(dptr)) - static_cast<int64_t>(reinterpret_cast<intptr_t>(rg.ptr));
        if (rep.regions_ok == 0) rep.dev_offset = off;
        else if (off != rep.dev_offset) rep.dev_offset_constant = false;
        if (off != 0) rep.uva_identical = false;
        rg.dev = static_cast<const uint8_t*>(dptr);
        rg.pinned = true;
        pinned_any_ = true;
        ++rep.regions_ok;
        rep.pinned_bytes += rg.padded;
    }
    for (size_t i = 0; i < layers_.size(); ++i) {
        const bool all = regions_[i * 3].pinned && regions_[i * 3 + 1].pinned && regions_[i * 3 + 2].pinned;
        layers_[i].gate.gpu_ok = layers_[i].up.gpu_ok = layers_[i].down.gpu_ok = all;
        layers_[i].gate.dev_base = all ? regions_[i * 3].dev : nullptr;
        layers_[i].up.dev_base = all ? regions_[i * 3 + 1].dev : nullptr;
        layers_[i].down.dev_base = all ? regions_[i * 3 + 2].dev : nullptr;
    }
    if (report) *report = rep;
    return rep.regions_ok > 0;
#else
    (void)max_bytes;
    rep.first_error = "built without CUDA (compile with -DPELIAS_COLD_VIEW_CUDA)";
    if (report) *report = rep;
    return false;
#endif
}

void ColdWeightsView::close() {
#if PELIAS_VIEW_CUDA
    if (pinned_any_)
        for (Region& rg : regions_)
            if (rg.pinned) { (void)cudaHostUnregister(const_cast<uint8_t*>(rg.ptr)); rg.pinned = false; rg.dev = nullptr; }
#endif
    pinned_any_ = false;
    if (map_base_) ::munmap(map_base_, map_size_);
    map_base_ = nullptr;
    map_size_ = 0;
    arena_ = nullptr;
    arena_size_ = 0;
    regions_.clear();
    layer_ids_.clear();
    layers_.clear();
    n_experts_ = 0;
    stats_ = ColdLoadStats();
    table_ = GgufTable();
}

std::string ColdWeightsView::describe() const {
    char buf[512];
    if (!arena_) return "ColdWeightsView: not open";
    size_t pinned = 0;
    for (const Region& rg : regions_) if (rg.pinned) pinned += rg.padded;
    std::snprintf(buf, sizeof buf, "ColdWeightsView: %zu layer(s), %lld experts each, %.2f GiB of tensors in a %.2f GiB 2 MiB-aligned arena, loaded in %.1f s (%.2f GB/s), huge-page backed %.2f GiB, pinned for the GPU %.2f GiB",
                  stats_.n_layers, static_cast<long long>(n_experts_), static_cast<double>(stats_.tensor_bytes) / (1ull << 30), static_cast<double>(stats_.arena_bytes) / (1ull << 30),
                  stats_.load_seconds, stats_.load_seconds > 0 ? static_cast<double>(stats_.tensor_bytes) / 1e9 / stats_.load_seconds : 0.0,
                  static_cast<double>(stats_.huge_backed_bytes) / (1ull << 30), static_cast<double>(pinned) / (1ull << 30));
    return buf;
}

}  // namespace pelias
