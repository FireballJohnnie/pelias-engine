// File: <project>\moe_track\tests\logits_dump.cpp
// Writes the logits of every position of a fixed pseudo-random token sequence, so two builds of llama.cpp (stock and Pelias-Tier) can be compared
// number by number (tests/test_phase6.py does the comparing).
//
//   logits_dump -m MODEL.gguf --out FILE.f32 [--ntok 96] [--chunk 1] [--seed 3] [-t 2] [--ngl 0] [--n-cpu-moe 0] [--no-repack]
//
//   --chunk N   tokens per llama_decode call. 1 = the decode path, 2-3 = a speculative verify batch, >= 32 = a prefill-sized batch.
//               The context's physical ubatch is set to max(chunk, 1).
//
// Output file: int32 n_pos, int32 n_vocab, then n_pos * n_vocab float32 logits (position-major).
// Build (the build script of the test does this):
//   g++ -O2 -std=c++17 logits_dump.cpp -I<llama.cpp>/include -I<llama.cpp>/ggml/include -L<build>/bin -lllama -lggml -lggml-base -Wl,-rpath,<build>/bin -o logits_dump
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::string model_path, out_path;
    int ntok = 96, chunk = 1, seed = 3, threads = 2, ngl = 0, n_cpu_moe = 0;
    bool no_repack = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char * what) -> const char * {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", what); std::exit(2); }
            return argv[++i];
        };
        if (a == "-m") model_path = next("-m");
        else if (a == "--out") out_path = next("--out");
        else if (a == "--ntok") ntok = std::atoi(next("--ntok"));
        else if (a == "--chunk") chunk = std::atoi(next("--chunk"));
        else if (a == "--seed") seed = std::atoi(next("--seed"));
        else if (a == "-t") threads = std::atoi(next("-t"));
        else if (a == "--ngl") ngl = std::atoi(next("--ngl"));
        else if (a == "--n-cpu-moe") n_cpu_moe = std::atoi(next("--n-cpu-moe"));
        else if (a == "--no-repack") no_repack = true;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || out_path.empty() || ntok < 1 || chunk < 1) {
        std::fprintf(stderr, "usage: logits_dump -m MODEL.gguf --out FILE.f32 [--ntok N] [--chunk N] [--seed N] [-t N] [--ngl N] [--n-cpu-moe N] [--no-repack]\n");
        return 2;
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

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = uint32_t(std::max(ntok + 8, 64));
    cp.n_batch = uint32_t(std::max(chunk, 1));
    cp.n_ubatch = uint32_t(std::max(chunk, 1));
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "failed to create a context\n"); llama_model_free(model); return 3; }

    // deterministic tokens (xorshift), never the special ids at the bottom of the vocabulary
    std::vector<llama_token> toks(ntok);
    uint64_t s = 0x9E3779B97F4A7C15ull ^ (uint64_t) seed * 0xD1B54A32D192ED03ull;
    for (int i = 0; i < ntok; ++i) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        toks[i] = llama_token(100 + (s % (uint64_t) std::max(1, n_vocab - 200)));
    }

    std::FILE * f = std::fopen(out_path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 3; }
    int32_t hdr[2] = { ntok, n_vocab };
    std::fwrite(hdr, sizeof(int32_t), 2, f);

    llama_batch batch = llama_batch_init(chunk, 0, 1);
    for (int pos = 0; pos < ntok; pos += chunk) {
        const int n = std::min(chunk, ntok - pos);
        batch.n_tokens = n;
        for (int j = 0; j < n; ++j) {
            batch.token[j]     = toks[pos + j];
            batch.pos[j]       = pos + j;
            batch.n_seq_id[j]  = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j]    = 1;
        }
        if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "llama_decode failed at position %d\n", pos); return 4; }
        for (int j = 0; j < n; ++j) {
            const float * lg = llama_get_logits_ith(ctx, j);
            if (!lg) { std::fprintf(stderr, "no logits for position %d\n", pos + j); return 4; }
            std::fwrite(lg, sizeof(float), (size_t) n_vocab, f);
        }
    }
    llama_batch_free(batch);
    std::fclose(f);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
