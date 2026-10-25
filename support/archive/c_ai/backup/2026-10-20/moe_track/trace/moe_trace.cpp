// File: <project>\moe_track\trace\moe_trace.cpp
// Router-choice recorder for MoE models in llama.cpp.
//
// Loads a GGUF with the same placement flags as production, runs prompts through it (prefill, then real sampled decoding),
// and records, for every token and every MoE layer, WHICH experts the router picked (and optionally their router weights).
// That is the only input the cache / placement simulator needs. Nothing in the model or in llama.cpp is modified: the
// recorder uses the public evaluation callback (llama_context_params.cb_eval) on the graph nodes named "ffn_moe_topk-<layer>"
// and "ffn_moe_weights-<layer>".
//
// Build:   trace/build.sh        (compiles against the llama.cpp tree and build given by LLAMA_PROD / LLAMA_BUILD_BIN)
// Run:     trace/bin/moe_trace -m MODEL.gguf --out PREFIX --prompts-jsonl prompts.jsonl --n-predict 512 [flags]
//
// Inputs (one of):
//   --prompts-jsonl F   one JSON object per line: {"id":"...","domain":"code","text":"..."}   or   {"id","domain","tokens":[1,2,3]}
//   --prompt-file F     the whole file is one prompt (text)
//   --tokens "1,2,3"    one prompt given as token ids (for models without a tokenizer)
//   --tokens-file F     whitespace/comma separated token ids, one prompt
//   --random-tokens N   one prompt of N pseudo-random token ids (smoke tests only)
// Placement (mirror production):  --ngl 99  --n-cpu-moe 28  --threads 0(auto: half the logical CPUs)  --ctx 8192  --ubatch 512  --kv-type q8_0  --flash-attn on|off|auto
// Sampling (defaults = production):  --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0 --seed 1   or   --greedy
// Other:  --n-predict 512  --trace-prefill  --weights  --ignore-eos  --max-seqs N  --domain NAME  --quiet  --backend-dir DIR
//
// Output:
//   PREFIX.pltrace      binary, little-endian (format below)
//   PREFIX.meta.jsonl   one line per sequence: {"seq","id","domain","prompt_tokens","generated_tokens","finish"}
//
// .pltrace layout
//   char[8] "PLTRACE1"; u32 header_len; header_len bytes of JSON  {"version":1,"model","arch","n_layers","moe_layers":[...],
//        "n_expert","n_expert_used","weights":bool,"created"}
//   then blocks until EOF:
//     u8 kind (0 prefill, 1 decode), u8[3] zero, u32 seq, u32 n_tok, u32 pos0,
//     i32 tokens[n_tok]                       tokens FED to the model at positions pos0 .. pos0+n_tok-1
//     u16 experts[n_tok][n_moe_layers][K]     0xFFFF = not recorded (prefill: last layer only sees the final token)
//     (if weights) u16 weights[n_tok][n_moe_layers][K]   IEEE half, router probability of each chosen expert
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <list>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

namespace {

// ------------------------------------------------------------------------------------------------ small helpers
uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = int32_t((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFF) == 0xFF) return uint16_t(sign | 0x7C00u | (mant ? 0x200u : 0u));
    if (exp >= 31) return uint16_t(sign | 0x7C00u);
    if (exp <= 0) {
        if (exp < -10) return uint16_t(sign);
        mant |= 0x800000u;
        const int shift = 14 - exp;
        uint32_t m = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (m & 1u))) m++;
        return uint16_t(sign | m);
    }
    uint32_t bits = (uint32_t(exp) << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (bits & 1u))) bits++;
    return uint16_t(sign | bits);
}

std::string json_escape(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += char(c);
        }
    }
    return o;
}

void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) out += char(cp);
    else if (cp < 0x800) { out += char(0xC0 | (cp >> 6)); out += char(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { out += char(0xE0 | (cp >> 12)); out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F)); }
    else { out += char(0xF0 | (cp >> 18)); out += char(0x80 | ((cp >> 12) & 0x3F)); out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F)); }
}

// Minimal parser for ONE flat JSON object per line: string, number and array-of-integer values. Anything else is skipped.
struct PromptRec {
    std::string id, domain, text;
    std::vector<int32_t> tokens;
    bool has_tokens = false, has_text = false;
};

bool parse_json_line(const std::string& s, PromptRec& r, std::string& err) {
    size_t i = 0;
    auto ws = [&]() { while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n')) ++i; };
    auto parse_string = [&](std::string& out) -> bool {
        if (i >= s.size() || s[i] != '"') return false;
        ++i;
        out.clear();
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) {
                ++i;
                switch (s[i]) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case '/': out += '/'; break;
                    case '\\': out += '\\'; break;
                    case '"': out += '"'; break;
                    case 'u': {
                        auto hex4 = [&](size_t at, uint32_t& v) {
                            if (at + 4 > s.size()) return false;
                            v = 0;
                            for (size_t k = 0; k < 4; ++k) {
                                const char c = s[at + k];
                                v <<= 4;
                                if (c >= '0' && c <= '9') v |= uint32_t(c - '0');
                                else if (c >= 'a' && c <= 'f') v |= uint32_t(c - 'a' + 10);
                                else if (c >= 'A' && c <= 'F') v |= uint32_t(c - 'A' + 10);
                                else return false;
                            }
                            return true;
                        };
                        uint32_t cp = 0;
                        if (!hex4(i + 1, cp)) return false;
                        i += 4;
                        if (cp >= 0xD800 && cp < 0xDC00 && i + 6 < s.size() && s[i + 1] == '\\' && s[i + 2] == 'u') {
                            uint32_t lo = 0;
                            if (hex4(i + 3, lo) && lo >= 0xDC00 && lo < 0xE000) { cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); i += 6; }
                        }
                        append_utf8(out, cp);
                        break;
                    }
                    default: out += s[i];
                }
                ++i;
            } else out += s[i++];
        }
        if (i >= s.size()) return false;
        ++i;
        return true;
    };
    ws();
    if (i >= s.size() || s[i] != '{') { err = "line does not start with {"; return false; }
    ++i;
    while (true) {
        ws();
        if (i < s.size() && s[i] == '}') return true;
        std::string key;
        if (!parse_string(key)) { err = "bad key"; return false; }
        ws();
        if (i >= s.size() || s[i] != ':') { err = "missing ':'"; return false; }
        ++i;
        ws();
        if (i >= s.size()) { err = "truncated"; return false; }
        if (s[i] == '"') {
            std::string v;
            if (!parse_string(v)) { err = "bad string value"; return false; }
            if (key == "id") r.id = v;
            else if (key == "domain") r.domain = v;
            else if (key == "text") { r.text = v; r.has_text = true; }
        } else if (s[i] == '[') {
            ++i;
            std::vector<int32_t> arr;
            while (true) {
                ws();
                if (i < s.size() && s[i] == ']') { ++i; break; }
                char* end = nullptr;
                const long v = std::strtol(s.c_str() + i, &end, 10);
                if (end == s.c_str() + i) { err = "bad integer in array"; return false; }
                arr.push_back(int32_t(v));
                i = size_t(end - s.c_str());
                ws();
                if (i < s.size() && s[i] == ',') ++i;
            }
            if (key == "tokens") { r.tokens = arr; r.has_tokens = true; }
        } else {
            while (i < s.size() && s[i] != ',' && s[i] != '}') ++i;   // number / bool / null: skipped
        }
        ws();
        if (i < s.size() && s[i] == ',') ++i;
    }
}

ggml_type parse_kv_type(const std::string& s, bool& ok) {
    ok = true;
    if (s == "f16") return GGML_TYPE_F16;
    if (s == "f32") return GGML_TYPE_F32;
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    if (s == "q4_0") return GGML_TYPE_Q4_0;
    if (s == "q4_1") return GGML_TYPE_Q4_1;
    if (s == "q5_0") return GGML_TYPE_Q5_0;
    if (s == "q5_1") return GGML_TYPE_Q5_1;
    ok = false;
    return GGML_TYPE_F16;
}

// ------------------------------------------------------------------------------------------------ callback state
struct LayerBuf {
    std::vector<uint16_t> experts;   // tokens-so-far-in-this-decode x K
    std::vector<uint16_t> weights;
    size_t tokens = 0;
    size_t calls = 0;
};

struct TraceState {
    bool want_weights = false;
    int n_expert = 0;
    int K = 0;                              // experts per token (from the tensor)
    std::map<int, LayerBuf> layers;         // layer index -> buffers for the current llama_decode
    std::vector<int> ubatch_sizes;          // tokens per ubatch of the current llama_decode (from the first MoE layer)
    int first_layer = -1;                   // smallest layer index seen (sees every token of every ubatch)
    std::vector<int> moe_layers;            // sorted, fixed after the first decode
    uint64_t bad_values = 0;
    std::string error;
    std::vector<uint8_t> scratch;
};

bool parse_layer_name(const char* name, const char* prefix, int& il) {
    const size_t n = std::strlen(prefix);
    if (std::strncmp(name, prefix, n) != 0) return false;
    const char* p = name + n;
    if (*p == '\0') return false;
    for (const char* q = p; *q; ++q) if (*q < '0' || *q > '9') return false;
    il = std::atoi(p);
    return true;
}

// Reads a (possibly strided, possibly device-resident) tensor span with one backend copy.
// Element (i0,i1,i2) lives at byte i0*nb0 + i1*nb1 + i2*nb2.
bool read_span(const ggml_tensor* t, std::vector<uint8_t>& buf, size_t& span) {
    span = 0;
    for (int d = 0; d < GGML_MAX_DIMS; ++d)
        if (t->ne[d] <= 0) { buf.clear(); return true; }   // empty tensor (e.g. the last layer of a ubatch with no output rows)
    for (int d = 0; d < GGML_MAX_DIMS; ++d) span += size_t(t->ne[d] - 1) * t->nb[d];
    span += ggml_type_size(t->type);
    buf.resize(span);
    ggml_backend_tensor_get(t, buf.data(), 0, span);
    return true;
}

bool eval_cb(struct ggml_tensor* t, bool ask, void* user) {
    auto* st = static_cast<TraceState*>(user);
    int il = -1;
    const bool is_topk = parse_layer_name(t->name, "ffn_moe_topk-", il);
    const bool is_w = st->want_weights && !is_topk && parse_layer_name(t->name, "ffn_moe_weights-", il);
    if (!is_topk && !is_w) return ask ? false : true;   // not ours: let the scheduler run on
    if (ask) return true;

    LayerBuf& lb = st->layers[il];
    if (is_topk) {
        if (t->type != GGML_TYPE_I32) { st->error = "ffn_moe_topk is not I32"; return true; }
        const int K = int(t->ne[0]);
        const int nt = int(t->ne[1]);
        if (st->K == 0) st->K = K;
        if (K != st->K) { st->error = "expert count per token changes between layers"; return true; }
        if (st->first_layer < 0 || il < st->first_layer) st->first_layer = il;

        size_t span = 0;
        read_span(t, st->scratch, span);
        // alignment of this layer's call against the ubatch sizes seen at the first layer
        int full = nt;
        if (il == st->first_layer) st->ubatch_sizes.push_back(nt);
        else if (lb.calls < st->ubatch_sizes.size()) full = st->ubatch_sizes[lb.calls];
        const int pad = std::max(0, full - nt);          // tokens this layer did not see (last layer of a prefill ubatch)
        for (int p = 0; p < pad * K; ++p) lb.experts.push_back(0xFFFF);
        for (int tk = 0; tk < nt; ++tk) {
            for (int k = 0; k < K; ++k) {
                int32_t v;
                std::memcpy(&v, st->scratch.data() + size_t(k) * t->nb[0] + size_t(tk) * t->nb[1], 4);
                if (v < 0 || (st->n_expert > 0 && v >= st->n_expert) || v > 0xFFFE) { st->bad_values++; v = 0xFFFF; }
                lb.experts.push_back(uint16_t(v));
            }
        }
        lb.tokens += size_t(full);
        lb.calls++;
    } else {
        // ffn_moe_weights-<il>: F32 [1, K, n_tokens]
        if (t->type != GGML_TYPE_F32) return true;
        const int K = int(t->ne[1]);
        const int nt = int(t->ne[2]);
        size_t span = 0;
        read_span(t, st->scratch, span);
        // the expert node of this layer ran just before this one, so lb.tokens already counts this ubatch; any shortfall is padding
        const size_t have_tokens = lb.weights.size() / size_t(std::max(K, 1));
        const size_t want_tokens = lb.tokens;               // expert tokens already appended for this layer in this decode
        const size_t pad = want_tokens > have_tokens + size_t(nt) ? want_tokens - have_tokens - size_t(nt) : 0;
        for (size_t p = 0; p < pad * size_t(K); ++p) lb.weights.push_back(0xFFFF);
        for (int tk = 0; tk < nt; ++tk)
            for (int k = 0; k < K; ++k) {
                float f;
                std::memcpy(&f, st->scratch.data() + size_t(k) * t->nb[1] + size_t(tk) * t->nb[2], 4);
                lb.weights.push_back(f32_to_f16(f));
            }
    }
    return true;
}

// ------------------------------------------------------------------------------------------------ output
struct Writer {
    std::ofstream bin;
    std::ofstream meta;
    bool header_written = false;
    uint64_t blocks = 0, decode_tokens = 0, prefill_tokens = 0;
    std::string model_path, arch;
    int n_layers = 0, n_expert = 0;
    bool weights = false;

    void write_header(const TraceState& st) {
        std::ostringstream h;
        h << "{\"version\":1,\"model\":\"" << json_escape(model_path) << "\",\"arch\":\"" << json_escape(arch) << "\",\"n_layers\":" << n_layers
          << ",\"moe_layers\":[";
        for (size_t i = 0; i < st.moe_layers.size(); ++i) h << (i ? "," : "") << st.moe_layers[i];
        h << "],\"n_expert\":" << n_expert << ",\"n_expert_used\":" << st.K << ",\"weights\":" << (weights ? "true" : "false")
          << ",\"created\":" << std::time(nullptr) << "}";
        const std::string s = h.str();
        bin.write("PLTRACE1", 8);
        const uint32_t len = uint32_t(s.size());
        bin.write(reinterpret_cast<const char*>(&len), 4);
        bin.write(s.data(), std::streamsize(s.size()));
        header_written = true;
    }

    // writes the block for the llama_decode that just finished; returns false if the buffers are inconsistent
    bool write_block(TraceState& st, int kind, uint32_t seq, uint32_t pos0, const std::vector<int32_t>& toks, std::string& err) {
        if (st.layers.empty()) { err = "the callback saw no ffn_moe_topk tensors (is this a MoE model? did the graph change?)"; return false; }
        if (st.moe_layers.empty()) {
            for (auto& kv : st.layers) st.moe_layers.push_back(kv.first);
            std::sort(st.moe_layers.begin(), st.moe_layers.end());
        }
        const size_t n = toks.size();
        const size_t K = size_t(st.K);
        for (int il : st.moe_layers) {
            auto it = st.layers.find(il);
            if (it == st.layers.end() || it->second.tokens != n) {
                err = "layer " + std::to_string(il) + " recorded " + std::to_string(it == st.layers.end() ? 0 : it->second.tokens) + " tokens, expected " + std::to_string(n);
                return false;
            }
            if (weights && it->second.weights.size() != n * K) {
                // weights missing for some tokens: pad at the front
                std::vector<uint16_t> w(n * K, 0xFFFF);
                const size_t have = it->second.weights.size();
                if (have <= w.size()) std::copy(it->second.weights.begin(), it->second.weights.end(), w.end() - std::ptrdiff_t(have));
                it->second.weights.swap(w);
            }
        }
        if (!header_written) write_header(st);
        const uint8_t hdr[4] = {uint8_t(kind), 0, 0, 0};
        const uint32_t n32 = uint32_t(n);
        bin.write(reinterpret_cast<const char*>(hdr), 4);
        bin.write(reinterpret_cast<const char*>(&seq), 4);
        bin.write(reinterpret_cast<const char*>(&n32), 4);
        bin.write(reinterpret_cast<const char*>(&pos0), 4);
        bin.write(reinterpret_cast<const char*>(toks.data()), std::streamsize(n * 4));
        std::vector<uint16_t> row(st.moe_layers.size() * K);
        for (size_t tk = 0; tk < n; ++tk) {
            for (size_t li = 0; li < st.moe_layers.size(); ++li)
                std::memcpy(&row[li * K], &st.layers[st.moe_layers[li]].experts[tk * K], K * 2);
            bin.write(reinterpret_cast<const char*>(row.data()), std::streamsize(row.size() * 2));
        }
        if (weights) {
            for (size_t tk = 0; tk < n; ++tk) {
                for (size_t li = 0; li < st.moe_layers.size(); ++li)
                    std::memcpy(&row[li * K], &st.layers[st.moe_layers[li]].weights[tk * K], K * 2);
                bin.write(reinterpret_cast<const char*>(row.data()), std::streamsize(row.size() * 2));
            }
        }
        blocks++;
        (kind == 1 ? decode_tokens : prefill_tokens) += n;
        return bin.good();
    }
};

void reset_decode_buffers(TraceState& st) {
    for (auto& kv : st.layers) {
        kv.second.experts.clear();
        kv.second.weights.clear();
        kv.second.tokens = 0;
        kv.second.calls = 0;
    }
    st.ubatch_sizes.clear();
    st.error.clear();
}

void usage() {
    std::fprintf(stderr,
                 "usage: moe_trace -m MODEL.gguf --out PREFIX (--prompts-jsonl F | --prompt-file F | --tokens \"1,2,3\" | --tokens-file F | --random-tokens N)\n"
                 "       [--n-predict 512] [--ngl 99] [--n-cpu-moe N] [--threads auto] [--ctx 8192] [--ubatch 512] [--kv-type q8_0] [--flash-attn on|off|auto]\n"
                 "       [--temp 1.0] [--top-k 20] [--top-p 0.95] [--min-p 0] [--seed 1] [--greedy] [--trace-prefill] [--weights] [--ignore-eos]\n"
                 "       [--max-seqs N] [--domain NAME] [--quiet] [--backend-dir DIR]\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, out_prefix, prompts_jsonl, prompt_file, tokens_arg, tokens_file, domain = "unspecified", kv_type = "q8_0", flash = "on";
    std::string backend_dir;
#ifdef MOE_TRACE_DEFAULT_BACKEND_DIR
    backend_dir = MOE_TRACE_DEFAULT_BACKEND_DIR;   // set by build.sh: the lib folder of the llama.cpp build this tracer was linked against
#endif
    if (const char* e = std::getenv("MOE_TRACE_BACKEND_DIR")) backend_dir = e;
    int n_predict = 512, ngl = 99, n_cpu_moe = 0, threads = 0, n_ctx = 8192, n_ubatch = 512, top_k = 20, max_seqs = 0, random_tokens = 0;
    float temp = 1.0f, top_p = 0.95f, min_p = 0.0f;
    uint32_t seed = 1;
    bool greedy = false, trace_prefill = false, want_weights = false, ignore_eos = false, quiet = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value after %s\n", a.c_str()); usage(); std::exit(2); }
            return argv[++i];
        };
        if (a == "-m" || a == "--model") model_path = next();
        else if (a == "--out") out_prefix = next();
        else if (a == "--prompts-jsonl") prompts_jsonl = next();
        else if (a == "--prompt-file") prompt_file = next();
        else if (a == "--tokens") tokens_arg = next();
        else if (a == "--tokens-file") tokens_file = next();
        else if (a == "--random-tokens") random_tokens = std::atoi(next());
        else if (a == "--n-predict") n_predict = std::atoi(next());
        else if (a == "--ngl") ngl = std::atoi(next());
        else if (a == "--n-cpu-moe") n_cpu_moe = std::atoi(next());
        else if (a == "--threads") threads = std::atoi(next());
        else if (a == "--ctx") n_ctx = std::atoi(next());
        else if (a == "--ubatch") n_ubatch = std::atoi(next());
        else if (a == "--kv-type") kv_type = next();
        else if (a == "--flash-attn") flash = next();
        else if (a == "--temp") temp = float(std::atof(next()));
        else if (a == "--top-k") top_k = std::atoi(next());
        else if (a == "--top-p") top_p = float(std::atof(next()));
        else if (a == "--min-p") min_p = float(std::atof(next()));
        else if (a == "--seed") seed = uint32_t(std::strtoul(next(), nullptr, 10));
        else if (a == "--greedy") greedy = true;
        else if (a == "--trace-prefill") trace_prefill = true;
        else if (a == "--weights") want_weights = true;
        else if (a == "--ignore-eos") ignore_eos = true;
        else if (a == "--max-seqs") max_seqs = std::atoi(next());
        else if (a == "--domain") domain = next();
        else if (a == "--quiet") quiet = true;
        else if (a == "--backend-dir") backend_dir = next();
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); usage(); return 2; }
    }
    if (model_path.empty() || out_prefix.empty()) { usage(); return 2; }
    const int sources = int(!prompts_jsonl.empty()) + int(!prompt_file.empty()) + int(!tokens_arg.empty()) + int(!tokens_file.empty()) + int(random_tokens > 0);
    if (sources != 1) { std::fprintf(stderr, "give exactly one prompt source\n"); usage(); return 2; }
    bool ok_kv = false;
    const ggml_type kvt = parse_kv_type(kv_type, ok_kv);
    if (!ok_kv) { std::fprintf(stderr, "unknown --kv-type %s\n", kv_type.c_str()); return 2; }

    // ---- collect prompts -----------------------------------------------------------------------------------
    std::vector<PromptRec> prompts;
    auto read_file = [](const std::string& p, std::string& out) {
        std::ifstream f(p, std::ios::binary);
        if (!f) return false;
        std::ostringstream ss;
        ss << f.rdbuf();
        out = ss.str();
        return true;
    };
    auto parse_ints = [](const std::string& s, std::vector<int32_t>& out) {
        std::string t = s;
        for (char& c : t) if (c == ',') c = ' ';
        std::istringstream is(t);
        long v;
        while (is >> v) out.push_back(int32_t(v));
    };
    if (!prompts_jsonl.empty()) {
        std::ifstream f(prompts_jsonl);
        if (!f) { std::fprintf(stderr, "cannot open %s\n", prompts_jsonl.c_str()); return 2; }
        std::string line;
        int ln = 0;
        while (std::getline(f, line)) {
            ++ln;
            if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
            PromptRec r;
            std::string err;
            if (!parse_json_line(line, r, err)) { std::fprintf(stderr, "%s line %d: %s\n", prompts_jsonl.c_str(), ln, err.c_str()); return 2; }
            if (!r.has_text && !r.has_tokens) { std::fprintf(stderr, "%s line %d: needs \"text\" or \"tokens\"\n", prompts_jsonl.c_str(), ln); return 2; }
            if (r.id.empty()) r.id = "line" + std::to_string(ln);
            if (r.domain.empty()) r.domain = domain;
            prompts.push_back(std::move(r));
        }
    } else if (!prompt_file.empty()) {
        PromptRec r;
        if (!read_file(prompt_file, r.text)) { std::fprintf(stderr, "cannot read %s\n", prompt_file.c_str()); return 2; }
        r.has_text = true; r.id = prompt_file; r.domain = domain;
        prompts.push_back(std::move(r));
    } else {
        PromptRec r;
        r.has_tokens = true; r.id = "tokens"; r.domain = domain;
        if (!tokens_arg.empty()) parse_ints(tokens_arg, r.tokens);
        else if (!tokens_file.empty()) {
            std::string s;
            if (!read_file(tokens_file, s)) { std::fprintf(stderr, "cannot read %s\n", tokens_file.c_str()); return 2; }
            parse_ints(s, r.tokens);
        } else {
            uint64_t x = 88172645463325252ull;
            for (int k = 0; k < random_tokens; ++k) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; r.tokens.push_back(int32_t(x % 50)); }   // ids < 50: valid for any real vocabulary too
        }
        if (r.tokens.empty()) { std::fprintf(stderr, "no token ids parsed\n"); return 2; }
        prompts.push_back(std::move(r));
    }
    if (max_seqs > 0 && int(prompts.size()) > max_seqs) prompts.resize(size_t(max_seqs));

    // ---- model and context ---------------------------------------------------------------------------------
    ggml_backend_load_all();
    if (!backend_dir.empty()) ggml_backend_load_all_from_path(backend_dir.c_str());   // GPU backends built as separate libraries live next to libllama
    llama_backend_init();
    if (!quiet) {
        std::fprintf(stderr, "backends:");
        for (size_t d = 0; d < ggml_backend_dev_count(); ++d) std::fprintf(stderr, " %s", ggml_backend_dev_name(ggml_backend_dev_get(d)));
        std::fprintf(stderr, "\n");
    }
    if (quiet) llama_log_set([](ggml_log_level lvl, const char* text, void*) { if (lvl >= GGML_LOG_LEVEL_ERROR) std::fputs(text, stderr); }, nullptr);

    // same override patterns as llama.cpp's --n-cpu-moe: experts of layers 0..N-1 in host memory
    static std::list<std::string> pattern_store;
    std::vector<llama_model_tensor_buft_override> overrides;
    for (int l = 0; l < n_cpu_moe; ++l) {
        pattern_store.push_back("blk\\." + std::to_string(l) + "\\.ffn_(up|down|gate|gate_up)_(ch|)exps");
        overrides.push_back({pattern_store.back().c_str(), ggml_backend_cpu_buffer_type()});
    }
    overrides.push_back({nullptr, nullptr});

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    if (n_cpu_moe > 0) mp.tensor_buft_overrides = overrides.data();
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { std::fprintf(stderr, "failed to load %s\n", model_path.c_str()); return 3; }
    const llama_vocab* vocab = llama_model_get_vocab(model);

    char arch_buf[128] = {0};
    llama_model_meta_val_str(model, "general.architecture", arch_buf, sizeof arch_buf);
    char ne_buf[64] = {0};
    int n_expert = 0;
    if (arch_buf[0] && llama_model_meta_val_str(model, (std::string(arch_buf) + ".expert_count").c_str(), ne_buf, sizeof ne_buf) > 0) n_expert = std::atoi(ne_buf);

    TraceState state;
    state.want_weights = want_weights;
    state.n_expert = n_expert;

    if (threads <= 0) threads = int(std::max(1u, std::thread::hardware_concurrency() / 2));   // physical-core estimate; routing does not depend on it
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = uint32_t(n_ctx);
    cp.n_batch = uint32_t(std::max(n_ubatch, 2048));
    cp.n_ubatch = uint32_t(n_ubatch);
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.type_k = kvt;
    cp.type_v = kvt;
    cp.flash_attn_type = flash == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED : (flash == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO);
    cp.cb_eval = eval_cb;
    cp.cb_eval_user_data = &state;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "failed to create a context\n"); llama_model_free(model); return 3; }

    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (greedy) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        if (top_k > 0) llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
        if (top_p < 1.0f) llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
        if (min_p > 0.0f) llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    }

    Writer w;
    w.bin.open(out_prefix + ".pltrace", std::ios::binary | std::ios::trunc);
    w.meta.open(out_prefix + ".meta.jsonl", std::ios::trunc);
    if (!w.bin || !w.meta) { std::fprintf(stderr, "cannot write %s.*\n", out_prefix.c_str()); return 3; }
    w.model_path = model_path;
    w.arch = arch_buf;
    w.n_layers = llama_model_n_layer(model);
    w.n_expert = n_expert;
    w.weights = want_weights;

    const auto t0 = std::chrono::steady_clock::now();
    uint32_t seq = 0;
    int rc = 0;
    for (const PromptRec& pr : prompts) {
        std::vector<llama_token> toks;
        if (pr.has_tokens) toks.assign(pr.tokens.begin(), pr.tokens.end());
        else {
            toks.resize(pr.text.size() + 16);
            int n = llama_tokenize(vocab, pr.text.c_str(), int32_t(pr.text.size()), toks.data(), int32_t(toks.size()), true, true);
            if (n < 0) { toks.resize(size_t(-n)); n = llama_tokenize(vocab, pr.text.c_str(), int32_t(pr.text.size()), toks.data(), int32_t(toks.size()), true, true); }
            if (n <= 0) { std::fprintf(stderr, "[seq %u] tokenization failed for %s\n", seq, pr.id.c_str()); continue; }
            toks.resize(size_t(n));
        }
        const int room = n_ctx - n_predict - 1;
        if (room < 8) { std::fprintf(stderr, "--ctx %d is too small for --n-predict %d\n", n_ctx, n_predict); rc = 2; break; }
        if (int(toks.size()) > room) toks.erase(toks.begin(), toks.end() - room);   // keep the end of an over-long prompt

        llama_memory_clear(llama_get_memory(ctx), true);
        llama_sampler_reset(smpl);

        std::string finish = "length";
        uint32_t generated = 0;
        bool failed = false;
        std::string err;

        // ---- prefill in chunks of n_batch
        size_t pos = 0;
        while (pos < toks.size()) {
            const size_t n = std::min<size_t>(toks.size() - pos, cp.n_batch);
            reset_decode_buffers(state);
            llama_batch b = llama_batch_get_one(toks.data() + pos, int32_t(n));
            if (llama_decode(ctx, b) != 0) { err = "llama_decode failed in prefill"; failed = true; break; }
            if (!state.error.empty()) { err = state.error; failed = true; break; }
            if (trace_prefill) {
                std::vector<int32_t> fed(toks.begin() + std::ptrdiff_t(pos), toks.begin() + std::ptrdiff_t(pos + n));
                if (!w.write_block(state, 0, seq, uint32_t(pos), fed, err)) { failed = true; break; }
            } else if (w.header_written == false && state.moe_layers.empty()) {
                // learn the layer set from the first decode even when prefill blocks are not stored
                for (auto& kv : state.layers) state.moe_layers.push_back(kv.first);
                std::sort(state.moe_layers.begin(), state.moe_layers.end());
            }
            pos += n;
        }

        // ---- decode
        llama_token cur = 0;
        if (!failed) cur = llama_sampler_sample(smpl, ctx, -1);
        for (int step = 0; !failed && step < n_predict; ++step) {
            if (!ignore_eos && llama_vocab_is_eog(vocab, cur)) { finish = "eog"; break; }
            reset_decode_buffers(state);
            llama_batch b = llama_batch_get_one(&cur, 1);
            if (llama_decode(ctx, b) != 0) { err = "llama_decode failed in decode"; failed = true; break; }
            if (!state.error.empty()) { err = state.error; failed = true; break; }
            std::vector<int32_t> fed{cur};
            if (!w.write_block(state, 1, seq, uint32_t(toks.size() + generated), fed, err)) { failed = true; break; }
            generated++;
            cur = llama_sampler_sample(smpl, ctx, -1);
        }
        if (failed) { finish = "error"; std::fprintf(stderr, "[seq %u] %s: %s\n", seq, pr.id.c_str(), err.c_str()); rc = 4; }

        w.meta << "{\"seq\":" << seq << ",\"id\":\"" << json_escape(pr.id) << "\",\"domain\":\"" << json_escape(pr.domain) << "\",\"prompt_tokens\":" << toks.size()
               << ",\"generated_tokens\":" << generated << ",\"finish\":\"" << finish << "\"}\n";
        w.meta.flush();
        if (!quiet) {
            const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr, "[seq %u] %-24s domain %-12s prompt %5zu tok, decoded %4u tok (%s)  total decode tokens %llu  %.0f s\n", seq, pr.id.c_str(), pr.domain.c_str(),
                         toks.size(), generated, finish.c_str(), (unsigned long long)w.decode_tokens, el);
        }
        ++seq;
        if (failed && w.blocks == 0) break;   // nothing recorded at all: do not grind through the rest
    }

    w.bin.flush();
    std::fprintf(stderr, "done: %u sequences, %llu decode tokens, %llu prefill tokens traced, %llu out-of-range values%s\n", seq, (unsigned long long)w.decode_tokens,
                 (unsigned long long)w.prefill_tokens, (unsigned long long)state.bad_values, state.bad_values ? "  <-- investigate" : "");
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return rc;
}
