// File: <project>\moe_track\pelias_serve\ref_dump.cpp
// Reference dumps for the Pelias-Serve parity stages (E1: single-layer outputs, E2: logits). It runs the STOCK llama.cpp (your pinned build) on a fixed token sequence and writes, as .npy files:
//     <prefix>_layer_<L>.npy       float32 [n_tokens, n_embd]   the output of decoder layer L (llama.cpp's tensor "l_out-L"), for each layer in --layers
//     <prefix>_logits_last.npy     float32 [K, n_vocab]         the logits of the last K positions (--logits-last K)
//     <prefix>_tokens.txt          the token ids, one line, so the executor under test runs exactly the same sequence
// Nothing is overwritten: if any target exists the tool stops before loading the model.
//
//   ref_dump -m MODEL.gguf --out-dir DIR (--prompt-file text.txt | --ids-file ids.txt | --synthetic N [--seed S]) [--layers 0,3] [--logits-last 8] [--prefix ref]
//            [-t 8] [--ngl 99] [--n-cpu-moe 28] [--ubatch 512] [--ctx 4096] [--no-repack]
//
// --prompt-file: UTF-8 text, tokenised with the model's own tokenizer (special tokens parsed, no extra BOS). --ids-file: whitespace/comma separated integers.
// --synthetic: the pseudo-random token run used by tests/logits_dump.cpp and mtp_align/dump_hidden.cpp (same generator, same seed -> same tokens).
// Build:  g++ -O2 -std=c++17 ref_dump.cpp -I<llama.cpp>/include -I<llama.cpp>/ggml/include -L<build>/bin -lllama -lggml -lggml-base -Wl,-rpath,<build>/bin -o ref_dump
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

static bool write_npy(const std::string & path, const float * data, size_t rows, size_t cols) {
    std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + std::to_string(rows) + ", " + std::to_string(cols) + "), }";
    const size_t total = 10 + dict.size() + 1;
    const size_t pad = (64 - total % 64) % 64;
    dict.append(pad, ' ');
    dict.push_back('\n');
    std::FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    std::fwrite(magic, 1, 8, f);
    const uint16_t hl = (uint16_t) dict.size();
    std::fwrite(&hl, 2, 1, f);
    std::fwrite(dict.data(), 1, dict.size(), f);
    const size_t n = rows * cols;
    const bool ok = std::fwrite(data, sizeof(float), n, f) == n;
    return std::fclose(f) == 0 && ok;
}

struct Collector {
    std::map<std::string, int> wanted;                      // tensor name -> layer
    std::map<int, std::vector<float>> rows;                 // layer -> appended rows
    std::map<int, long long> n_rows;
    int n_embd = 0;
    bool failed = false;
};

static bool eval_cb(struct ggml_tensor * t, bool ask, void * ud) {
    Collector * c = (Collector *) ud;
    auto it = c->wanted.find(t->name);
    if (it == c->wanted.end()) return ask ? false : true;
    if (ask) return true;
    if (t->ne[0] != c->n_embd) { c->failed = true; return true; }
    const int64_t n_tok = t->ne[1] * t->ne[2] * t->ne[3];
    std::vector<float> buf((size_t) c->n_embd * n_tok);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
    } else {
        const ggml_type_traits * tr = ggml_get_type_traits(t->type);
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        if (!tr || !tr->to_float || ggml_is_contiguous(t) == false) { c->failed = true; return true; }
        tr->to_float(raw.data(), buf.data(), (int64_t) buf.size());
    }
    auto & v = c->rows[it->second];
    v.insert(v.end(), buf.begin(), buf.end());
    c->n_rows[it->second] += n_tok;
    return true;
}

int main(int argc, char ** argv) {
    std::string model_path, out_dir, prompt_file, ids_file, prefix = "ref", layers_arg = "0,3";
    int threads = 4, ngl = 0, n_cpu_moe = 0, ubatch = 512, ctx_len = 4096, synthetic = 0, seed = 3, logits_last = 8;
    bool no_repack = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char * what) -> const char * {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", what); std::exit(2); }
            return argv[++i];
        };
        if (a == "-m") model_path = next("-m");
        else if (a == "--out-dir") out_dir = next("--out-dir");
        else if (a == "--prompt-file") prompt_file = next("--prompt-file");
        else if (a == "--ids-file") ids_file = next("--ids-file");
        else if (a == "--synthetic") synthetic = std::atoi(next("--synthetic"));
        else if (a == "--seed") seed = std::atoi(next("--seed"));
        else if (a == "--layers") layers_arg = next("--layers");
        else if (a == "--logits-last") logits_last = std::atoi(next("--logits-last"));
        else if (a == "--prefix") prefix = next("--prefix");
        else if (a == "-t") threads = std::atoi(next("-t"));
        else if (a == "--ngl") ngl = std::atoi(next("--ngl"));
        else if (a == "--n-cpu-moe") n_cpu_moe = std::atoi(next("--n-cpu-moe"));
        else if (a == "--ubatch") ubatch = std::atoi(next("--ubatch"));
        else if (a == "--ctx") ctx_len = std::atoi(next("--ctx"));
        else if (a == "--no-repack") no_repack = true;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    const int sources = (!prompt_file.empty()) + (!ids_file.empty()) + (synthetic > 0);
    if (model_path.empty() || out_dir.empty() || sources != 1 || ubatch < 1 || ctx_len < 8 || logits_last < 0) {
        std::fprintf(stderr, "usage: ref_dump -m MODEL.gguf --out-dir DIR (--prompt-file F | --ids-file F | --synthetic N) [--layers 0,3] [--logits-last 8] [--prefix ref] [-t N] [--ngl N] [--n-cpu-moe N] [--ubatch N] [--ctx N] [--no-repack]\n");
        return 2;
    }
    std::vector<int> layers;
    {
        std::stringstream ss(layers_arg);
        std::string tok;
        while (std::getline(ss, tok, ',')) if (!tok.empty()) layers.push_back(std::atoi(tok.c_str()));
        if (layers.empty()) { std::fprintf(stderr, "--layers is empty\n"); return 2; }
    }
    std::vector<std::string> targets;
    for (int l : layers) targets.push_back(out_dir + "/" + prefix + "_layer_" + std::to_string(l) + ".npy");
    targets.push_back(out_dir + "/" + prefix + "_logits_last.npy");
    targets.push_back(out_dir + "/" + prefix + "_tokens.txt");
    for (auto & t : targets) if (std::ifstream(t).good()) { std::fprintf(stderr, "%s already exists; refusing to overwrite it\n", t.c_str()); return 2; }

    {   // the dump folder is created if it is missing (existing files are never touched: see the check above)
        std::error_code ec;
        std::filesystem::create_directories(out_dir, ec);
        if (ec) { std::fprintf(stderr, "cannot create %s: %s\n", out_dir.c_str(), ec.message().c_str()); return 2; }
    }
    ggml_backend_load_all();
    llama_backend_init();
    llama_log_set([](ggml_log_level lvl, const char * text, void *) { if (lvl >= GGML_LOG_LEVEL_ERROR) std::fputs(text, stderr); }, nullptr);

    static std::list<std::string> pattern_store;
    std::vector<llama_model_tensor_buft_override> overrides;
    for (int l = 0; l < n_cpu_moe; ++l) {
        pattern_store.push_back("blk\\." + std::to_string(l) + "\\.ffn_(up|down|gate|gate_up)_(ch|)exps");
        overrides.push_back({pattern_store.back().c_str(), ggml_backend_cpu_buffer_type()});
    }
    overrides.push_back({nullptr, nullptr});
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    if (no_repack) mp.use_extra_bufts = false;
    if (n_cpu_moe > 0) mp.tensor_buft_overrides = overrides.data();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { std::fprintf(stderr, "failed to load %s\n", model_path.c_str()); return 3; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_embd = llama_model_n_embd(model);
    const int n_layer = llama_model_n_layer(model);
    for (int l : layers) if (l < 0 || l >= n_layer) { std::fprintf(stderr, "layer %d is outside 0..%d\n", l, n_layer - 1); return 2; }

    std::vector<int32_t> ids;
    if (synthetic > 0) {
        uint64_t s = 0x9E3779B97F4A7C15ull ^ (uint64_t) seed * 0xD1B54A32D192ED03ull;
        for (int i = 0; i < synthetic; ++i) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            ids.push_back((int32_t) (100 + (s % (uint64_t) std::max(1, n_vocab - 200))));
        }
    } else if (!ids_file.empty()) {
        std::ifstream in(ids_file);
        if (!in) { std::fprintf(stderr, "cannot read %s\n", ids_file.c_str()); return 3; }
        std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        for (char & ch : all) if (ch == ',') ch = ' ';
        std::stringstream ss(all);
        long long v;
        while (ss >> v) ids.push_back((int32_t) v);
    } else {
        std::ifstream in(prompt_file, std::ios::binary);
        if (!in) { std::fprintf(stderr, "cannot read %s\n", prompt_file.c_str()); return 3; }
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        std::vector<llama_token> toks(text.size() + 16);
        int n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), false, true);
        if (n < 0) { toks.resize((size_t) -n); n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), false, true); }
        if (n <= 0) { std::fprintf(stderr, "tokenisation failed\n"); return 3; }
        ids.assign(toks.begin(), toks.begin() + n);
    }
    for (int32_t id : ids) if (id < 0 || id >= n_vocab) { std::fprintf(stderr, "token id %d outside the vocabulary (%d)\n", id, n_vocab); return 3; }
    if ((int) ids.size() > ctx_len) ids.resize((size_t) ctx_len);
    const int n_tok = (int) ids.size();
    if (n_tok < 1) { std::fprintf(stderr, "no tokens\n"); return 3; }
    logits_last = std::min(logits_last, n_tok);

    Collector col;
    col.n_embd = n_embd;
    for (int l : layers) col.wanted["l_out-" + std::to_string(l)] = l;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) ctx_len;
    cp.n_batch = (uint32_t) ubatch;
    cp.n_ubatch = (uint32_t) ubatch;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.cb_eval = eval_cb;
    cp.cb_eval_user_data = &col;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "failed to create a context\n"); llama_model_free(model); return 3; }

    llama_batch batch = llama_batch_init(ubatch, 0, 1);
    std::vector<float> logits_out;
    for (int pos = 0; pos < n_tok; pos += ubatch) {
        const int n = std::min(ubatch, n_tok - pos);
        batch.n_tokens = n;
        for (int j = 0; j < n; ++j) {
            batch.token[j] = ids[(size_t) pos + j];
            batch.pos[j] = pos + j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = 1;                              // every position is an output, so the last layer is not pruned to the output rows
        }
        if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "llama_decode failed at position %d\n", pos); return 4; }
        for (int j = 0; j < n; ++j) {
            if (pos + j < n_tok - logits_last) continue;
            const float * lg = llama_get_logits_ith(ctx, j);
            if (!lg) { std::fprintf(stderr, "no logits at position %d\n", pos + j); return 4; }
            logits_out.insert(logits_out.end(), lg, lg + n_vocab);
        }
    }
    llama_batch_free(batch);
    if (col.failed) { std::fprintf(stderr, "a layer tensor had an unexpected shape or type\n"); return 4; }
    for (int l : layers) {
        if (col.n_rows[l] != n_tok) { std::fprintf(stderr, "layer %d: collected %lld rows, expected %d (the graph did not expose l_out-%d for every position)\n", l, col.n_rows[l], n_tok, l); return 4; }
        const auto & v = col.rows[l];
        for (float x : v) if (!std::isfinite(x)) { std::fprintf(stderr, "layer %d: non-finite value\n", l); return 4; }
    }
    int rc = 0;
    for (size_t k = 0; k < layers.size(); ++k)
        if (!write_npy(targets[k], col.rows[layers[k]].data(), (size_t) n_tok, (size_t) n_embd)) { std::fprintf(stderr, "cannot write %s\n", targets[k].c_str()); rc = 5; }
    if (logits_last > 0 && !write_npy(targets[layers.size()], logits_out.data(), (size_t) logits_last, (size_t) n_vocab)) { std::fprintf(stderr, "cannot write the logits file\n"); rc = 5; }
    if (logits_last == 0) { std::FILE * f = std::fopen(targets[layers.size()].c_str(), "wb"); if (f) std::fclose(f); }
    {
        std::ofstream t(targets.back());
        for (int i = 0; i < n_tok; ++i) t << ids[(size_t) i] << (i + 1 < n_tok ? " " : "\n");
    }
    std::fprintf(stderr, "[ref_dump] %d tokens, layers", n_tok);
    for (int l : layers) std::fprintf(stderr, " %d", l);
    std::fprintf(stderr, ", logits of the last %d positions -> %s/%s_*\n", logits_last, out_dir.c_str(), prefix.c_str());
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return rc;
}
