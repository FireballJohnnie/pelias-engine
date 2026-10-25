#include "llama-pelias-tier.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-hparams.h"

#include "ggml-alloc.h"
#include "gguf.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------------------------------------------------------------------
// environment
// ---------------------------------------------------------------------------------------------------------------------------------------
static const char * env_str(const char * name) {
    const char * v = std::getenv(name);
    return (v && *v) ? v : nullptr;
}

static long env_long(const char * name, long dflt) {
    const char * v = env_str(name);
    if (!v) {
        return dflt;
    }
    char * end = nullptr;
    const long x = std::strtol(v, &end, 10);
    return (end && end != v) ? x : dflt;
}

bool pelias_tier_enabled() {
    static const bool on = env_str("PELIAS_HOT_SET") != nullptr;
    return on;
}

int pelias_tier_dual_max_tokens() {
    static const int n = [] {
        const long a = env_long("PELIAS_TIER_BYPASS_N", 32);
        const long b = env_long("GGML_OP_OFFLOAD_MIN_BATCH", 32);   // what ggml-cuda uses to decide to run a CPU-weight op on the GPU
        return (int) std::max<long>(1, std::min(a, b));
    }();
    return n;
}

static bool verbose() {
    static const bool v = env_long("PELIAS_TIER_VERBOSE", 0) != 0;
    return v;
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// registry
// ---------------------------------------------------------------------------------------------------------------------------------------
namespace {

struct tier_bank {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    pelias_tier_layer     layer;

    tier_bank() = default;
    tier_bank(const tier_bank &) = delete;
    tier_bank & operator=(const tier_bank &) = delete;
    ~tier_bank() {
        if (buf) {
            ggml_backend_buffer_free(buf);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

std::mutex                                                                       g_mu;
std::map<int, uint64_t>                                                          g_dual_builds_by_ntok;   // n_tokens of the graph -> number of layer builds
std::unordered_map<const llama_model *, std::vector<std::unique_ptr<tier_bank>>> g_models;
std::unordered_map<const ggml_tensor *, const pelias_tier_layer *>               g_by_down;

}  // namespace

const pelias_tier_layer * pelias_tier_find(const ggml_tensor * down_exps) {
    if (!down_exps || !pelias_tier_enabled()) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_by_down.find(down_exps);
    return it == g_by_down.end() ? nullptr : it->second;
}

void pelias_tier_note_dual_build(int il, int n_tokens) {
    (void) il;
    std::lock_guard<std::mutex> lk(g_mu);
    g_dual_builds_by_ntok[n_tokens] += 1;
}

void pelias_tier_detach(const llama_model * model) {
    if (!pelias_tier_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_models.find(model);
    if (it == g_models.end()) {
        return;
    }
    if (const char * sf = env_str("PELIAS_TIER_STATS_FILE")) {
        if (FILE * f = std::fopen(sf, "w")) {
            std::fprintf(f, "{\n  \"layers_with_bank\": %zu,\n  \"dual_layer_builds_by_n_tokens\": {", it->second.size());
            bool first = true;
            for (const auto & kv : g_dual_builds_by_ntok) {
                std::fprintf(f, "%s\"%d\": %llu", first ? "" : ", ", kv.first, (unsigned long long) kv.second);
                first = false;
            }
            std::fprintf(f, "}\n}\n");
            std::fclose(f);
        }
    }
    for (auto & b : it->second) {
        for (auto g = g_by_down.begin(); g != g_by_down.end();) {
            g = (g->second == &b->layer) ? g_by_down.erase(g) : std::next(g);
        }
    }
    g_models.erase(it);
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// hot-set file
//
//   # comment
//   PELIAS_HOT_SET 1
//   n_expert 256
//   layer 0 : 3 17 22 ...          (the ":" is optional)
// ---------------------------------------------------------------------------------------------------------------------------------------
static bool parse_hot_set(const std::string & path, int n_expert, std::map<int, std::vector<int>> & out, std::string & err) {
    std::ifstream in(path);
    if (!in) {
        err = "cannot open " + path;
        return false;
    }
    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        const size_t hash = line.find('#');
        if (hash != std::string::npos) {
            line.erase(hash);
        }
        for (char & c : line) {
            if (c == ':' || c == ',') {
                c = ' ';
            }
        }
        std::istringstream ss(line);
        std::string key;
        if (!(ss >> key)) {
            continue;
        }
        if (key == "PELIAS_HOT_SET") {
            int ver = 0;
            ss >> ver;
            if (ver != 1) {
                err = "unsupported hot-set version on line " + std::to_string(line_no);
                return false;
            }
        } else if (key == "n_expert") {
            int n = 0;
            ss >> n;
            if (n != n_expert) {
                err = "hot set was made for n_expert=" + std::to_string(n) + " but the model has " + std::to_string(n_expert);
                return false;
            }
        } else if (key == "layer") {
            int il = -1;
            if (!(ss >> il) || il < 0) {
                err = "bad layer number on line " + std::to_string(line_no);
                return false;
            }
            std::set<int> ids;
            int e;
            while (ss >> e) {
                if (e < 0 || e >= n_expert) {
                    err = "expert id " + std::to_string(e) + " out of range on line " + std::to_string(line_no);
                    return false;
                }
                ids.insert(e);
            }
            if (!ss.eof()) {
                err = "unreadable text on line " + std::to_string(line_no);
                return false;
            }
            out[il].assign(ids.begin(), ids.end());   // sorted, unique
        } else {
            err = "unknown keyword '" + key + "' on line " + std::to_string(line_no);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// attach
// ---------------------------------------------------------------------------------------------------------------------------------------
namespace {

struct src_tensor {
    ggml_tensor * orig = nullptr;
    int64_t       gguf_id = -1;
    size_t        file_off = 0;      // absolute offset of expert 0 in the file
    size_t        slice = 0;         // bytes of one expert
};

}  // namespace

static ggml_backend_buffer_type_t pick_buft(const llama_model & model, int il, bool & on_gpu) {
    ggml_backend_dev_t dev = nullptr;
    try {
        dev = model.dev_layer(il);
    } catch (...) {
        dev = nullptr;
    }
    const bool force_cpu = env_str("PELIAS_TIER_DEVICE") && std::strcmp(env_str("PELIAS_TIER_DEVICE"), "cpu") == 0;
    if (!force_cpu && (dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU)) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);   // layer is on the CPU (or unknown): use the first GPU if there is one
    }
    if (force_cpu || dev == nullptr) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    on_gpu = dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU;
    return dev ? ggml_backend_dev_buffer_type(dev) : nullptr;
}

void pelias_tier_attach(const llama_model & model, const std::string & gguf_path) {
    if (!pelias_tier_enabled()) {
        return;
    }
    const char * hs_path = env_str("PELIAS_HOT_SET");
    const int n_expert = (int) model.hparams.n_expert;
    if (n_expert <= 0) {
        LLAMA_LOG_WARN("%s: pelias-tier: the model has no experts; nothing to do\n", __func__);
        return;
    }

    std::map<int, std::vector<int>> hot;
    std::string err;
    if (!parse_hot_set(hs_path, n_expert, hot, err)) {
        LLAMA_LOG_ERROR("%s: pelias-tier: hot-set file rejected (%s). Running without a hot bank.\n", __func__, err.c_str());
        return;
    }

    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * gg = gguf_init_from_file(gguf_path.c_str(), gp);
    if (!gg) {
        LLAMA_LOG_ERROR("%s: pelias-tier: cannot read the GGUF index of %s. Running without a hot bank.\n", __func__, gguf_path.c_str());
        return;
    }
    std::unique_ptr<gguf_context, void (*)(gguf_context *)> gg_guard(gg, gguf_free);
    const size_t data_off = gguf_get_data_offset(gg);

    std::unique_ptr<FILE, int (*)(FILE *)> fp(std::fopen(gguf_path.c_str(), "rb"), std::fclose);
    if (!fp) {
        LLAMA_LOG_ERROR("%s: pelias-tier: cannot open %s. Running without a hot bank.\n", __func__, gguf_path.c_str());
        return;
    }

    const long cap_mib = env_long("PELIAS_TIER_MAX_MIB", 0);

    struct plan_item {
        int il;
        const std::vector<int> * ids;
        src_tensor gate, up, gate_up, down;
    };
    std::vector<plan_item> plan;
    double total_bytes = 0.0;

    auto resolve = [&](ggml_tensor * t, src_tensor & s) -> bool {
        if (!t) {
            return true;   // absent tensors are fine; the caller checks which ones are required
        }
        if (t->ne[2] != n_expert || t->ne[3] != 1) {
            return false;
        }
        const int64_t id = gguf_find_tensor(gg, t->name);
        if (id < 0) {
            return false;   // not in this file (split model?)
        }
        if (gguf_get_tensor_size(gg, id) != ggml_nbytes(t) || (int) gguf_get_tensor_type(gg, id) != (int) t->type) {
            return false;
        }
        s.orig     = t;
        s.gguf_id  = id;
        s.file_off = data_off + gguf_get_tensor_offset(gg, id);
        s.slice    = t->nb[2];
        return true;
    };

    for (auto & kv : hot) {
        const int il = kv.first;
        if (kv.second.empty()) {
            continue;
        }
        if (il >= (int) model.layers.size()) {
            LLAMA_LOG_WARN("%s: pelias-tier: hot set names layer %d but the model has %d layers; skipped\n", __func__, il, (int) model.layers.size());
            continue;
        }
        const llama_layer & L = model.layers[il];
        if (!L.ffn_down_exps || (!L.ffn_gate_exps && !L.ffn_gate_up_exps) || (L.ffn_gate_exps && !L.ffn_up_exps)) {
            LLAMA_LOG_WARN("%s: pelias-tier: layer %d has no routed experts of the supported shape; skipped\n", __func__, il);
            continue;
        }
        if (L.ffn_gate_exps_b || L.ffn_down_exps_b || L.ffn_up_exps_b || L.ffn_gate_up_exps_b ||
            L.ffn_gate_exps_s || L.ffn_down_exps_s || L.ffn_up_exps_s) {
            LLAMA_LOG_WARN("%s: pelias-tier: layer %d has expert biases or scales (not supported); skipped\n", __func__, il);
            continue;
        }
        plan_item p{};
        p.il  = il;
        p.ids = &kv.second;
        bool ok = resolve(L.ffn_down_exps, p.down) && resolve(L.ffn_gate_exps, p.gate) && resolve(L.ffn_up_exps, p.up) &&
                  resolve(L.ffn_gate_up_exps, p.gate_up);
        if (!ok) {
            LLAMA_LOG_WARN("%s: pelias-tier: layer %d: tensors not found with identical shape and type in %s; skipped\n", __func__, il, gguf_path.c_str());
            continue;
        }
        const size_t per_slot = p.down.slice + p.gate.slice + p.up.slice + p.gate_up.slice;
        total_bytes += (double) per_slot * (double) (p.ids->size() + 1) + 2.0 * 4.0 * n_expert;
        plan.push_back(p);
    }

    if (plan.empty()) {
        LLAMA_LOG_WARN("%s: pelias-tier: no usable layers in the hot set; running without a hot bank\n", __func__);
        return;
    }
    if (cap_mib > 0 && total_bytes > (double) cap_mib * 1048576.0) {
        LLAMA_LOG_ERROR("%s: pelias-tier: the hot set needs %.0f MiB but PELIAS_TIER_MAX_MIB is %ld. Nothing was allocated; running without a hot bank.\n",
                        __func__, total_bytes / 1048576.0, cap_mib);
        return;
    }

    std::vector<std::unique_ptr<tier_bank>> banks;
    size_t uploaded = 0;
    bool any_gpu = false;
    std::vector<uint8_t> staging;

    for (const plan_item & p : plan) {
        bool on_gpu = false;
        ggml_backend_buffer_type_t buft = pick_buft(model, p.il, on_gpu);
        if (!buft) {
            LLAMA_LOG_WARN("%s: pelias-tier: layer %d: no buffer type available; skipped\n", __func__, p.il);
            continue;
        }
        any_gpu = any_gpu || on_gpu;

        auto bank = std::make_unique<tier_bank>();
        ggml_init_params ip = { /*.mem_size =*/ ggml_tensor_overhead() * 8 + 1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
        bank->ctx = ggml_init(ip);
        if (!bank->ctx) {
            continue;
        }
        const int n_slots = (int) p.ids->size() + 1;
        auto mk = [&](const src_tensor & s, const char * what) -> ggml_tensor * {
            if (!s.orig) {
                return nullptr;
            }
            ggml_tensor * t = ggml_new_tensor_3d(bank->ctx, s.orig->type, s.orig->ne[0], s.orig->ne[1], n_slots);
            char nm[GGML_MAX_NAME];
            std::snprintf(nm, sizeof(nm), "pelias.blk.%d.%s", p.il, what);
            ggml_set_name(t, nm);
            return t;
        };
        pelias_tier_layer & ly = bank->layer;
        ly.il       = p.il;
        ly.n_expert = n_expert;
        ly.n_hot    = (int) p.ids->size();
        ly.orig_gate    = p.gate.orig;
        ly.orig_up      = p.up.orig;
        ly.orig_gate_up = p.gate_up.orig;
        ly.orig_down    = p.down.orig;
        ly.gate     = mk(p.gate,    "gate_hot");
        ly.up       = mk(p.up,      "up_hot");
        ly.gate_up  = mk(p.gate_up, "gate_up_hot");
        ly.down     = mk(p.down,    "down_hot");
        ly.tab_hot  = ggml_new_tensor_2d(bank->ctx, GGML_TYPE_I32, 1, n_expert);
        ly.tab_cold = ggml_new_tensor_2d(bank->ctx, GGML_TYPE_I32, 1, n_expert);
        ggml_set_name(ly.tab_hot,  "pelias.tab_hot");
        ggml_set_name(ly.tab_cold, "pelias.tab_cold");

        bank->buf = ggml_backend_alloc_ctx_tensors_from_buft(bank->ctx, buft);
        if (!bank->buf) {
            LLAMA_LOG_ERROR("%s: pelias-tier: layer %d: out of memory for %d hot experts; this layer stays on the stock path\n", __func__, p.il, ly.n_hot);
            continue;
        }
        ggml_backend_buffer_set_usage(bank->buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_clear(bank->buf, 0);   // slot 0 of every bank tensor = all-zero expert (a zero Q4_K / Q6_K / F16 block dequantizes to 0)

        bool good = true;
        auto upload = [&](const src_tensor & s, ggml_tensor * dst) {
            if (!s.orig || !dst || !good) {
                return;
            }
            staging.resize(s.slice);
            for (size_t j = 0; j < p.ids->size(); ++j) {
                const size_t off = s.file_off + (size_t) (*p.ids)[j] * s.slice;
                if (fseeko(fp.get(), (off_t) off, SEEK_SET) != 0 || std::fread(staging.data(), 1, s.slice, fp.get()) != s.slice) {
                    LLAMA_LOG_ERROR("%s: pelias-tier: layer %d: read error at offset %zu\n", __func__, p.il, off);
                    good = false;
                    return;
                }
                ggml_backend_tensor_set(dst, staging.data(), (j + 1) * dst->nb[2], s.slice);
                uploaded += s.slice;
            }
        };
        upload(p.gate, ly.gate);
        upload(p.up, ly.up);
        upload(p.gate_up, ly.gate_up);
        upload(p.down, ly.down);
        if (!good) {
            continue;   // bank destroyed by unique_ptr; layer stays on the stock path
        }

        std::vector<int32_t> th(n_expert, 0), tc(n_expert);
        for (int e = 0; e < n_expert; ++e) {
            tc[e] = e;
        }
        for (size_t j = 0; j < p.ids->size(); ++j) {
            const int e = (*p.ids)[j];
            th[e] = (int32_t) (j + 1);
            tc[e] = -1;
        }
        ggml_backend_tensor_set(ly.tab_hot,  th.data(), 0, th.size() * sizeof(int32_t));
        ggml_backend_tensor_set(ly.tab_cold, tc.data(), 0, tc.size() * sizeof(int32_t));

        if (verbose()) {
            LLAMA_LOG_INFO("%s: pelias-tier: layer %2d: %d hot experts (%.1f MiB) in %s\n", __func__, p.il, ly.n_hot,
                           (double) ggml_backend_buffer_get_size(bank->buf) / 1048576.0, ggml_backend_buft_name(buft));
        }
        banks.push_back(std::move(bank));
    }

    if (banks.empty()) {
        LLAMA_LOG_WARN("%s: pelias-tier: no bank could be built; running without a hot bank\n", __func__);
        return;
    }

    const size_t n_banks = banks.size();
    {
        std::lock_guard<std::mutex> lk(g_mu);
        for (auto & b : banks) {
            g_by_down[model.layers[b->layer.il].ffn_down_exps] = &b->layer;
        }
        g_models[&model] = std::move(banks);
    }
    LLAMA_LOG_INFO("%s: pelias-tier: hot bank active for %zu layers, %.1f MiB of expert weights copied%s; dual path for batches under %d tokens\n",
                   __func__, n_banks, (double) uploaded / 1048576.0,
                   any_gpu ? " to the GPU" : " (NO GPU buffer: test mode, the bank is in RAM)", pelias_tier_dual_max_tokens());
}
