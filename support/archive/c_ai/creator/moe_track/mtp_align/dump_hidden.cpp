// File: <project>\moe_track\mtp_align\dump_hidden.cpp
// Stage 10a of the MTP-head alignment: run the FINISHED, merged Stuart model (your production GGUF) over training text and write down, for every position,
// the hidden state that llama.cpp itself would hand to the MTP draft head: the state after the model's final RMSNorm, just before the output projection.
// Training the head on exactly these numbers means it learns from what the server will really feed it (quantisation included), not from an approximation.
//
//   dump_hidden -m MODEL.gguf --in samples.ids --out hidden.bin [-t 8] [--ngl 99] [--n-cpu-moe 28] [--ubatch 512] [--ctx 4096] [--no-repack] [--limit N]
//   dump_hidden -m MODEL.gguf --synthetic 96 --seed 3 --out hidden.bin ...     (test mode: the same pseudo-random token run as tests/logits_dump.cpp)
//
// Input  (--in)  : int32 magic 0x50534944 ('PSID'), int32 n_samples, then per sample: int32 n_tok, int32 ids[n_tok]            (little endian)
// Output (--out) : int32 magic 0x50484944 ('PHID'), int32 n_embd, int32 n_samples, then per sample: int32 n_tok, uint16 bf16[n_tok][n_embd]
//                  Row t is the hidden state at position t, i.e. the state whose output distribution predicts token t+1.
// Each sample starts from an empty context (the recurrent DeltaNet state is cleared too). Prompts longer than --ctx are cut.
//
// It uses only the public llama.h API (embeddings on, pooling off, every position an output). It never writes next to the model; the output path must not exist.
// Build (the script 10_train_blk40_mtp_lora.py prints the exact line; same pattern as tests/logits_dump.cpp):
//   g++ -O2 -std=c++17 dump_hidden.cpp -I<llama.cpp>/include -I<llama.cpp>/ggml/include -L<build>/bin -lllama -lggml -lggml-base -Wl,-rpath,<build>/bin -o dump_hidden
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <list>
#include <string>
#include <vector>

static uint16_t f32_to_bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    if (std::isnan(x)) return 0x7FC0;
    u += 0x7FFFu + ((u >> 16) & 1u);          // round to nearest even
    return (uint16_t) (u >> 16);
}

int main(int argc, char ** argv) {
    std::string model_path, in_path, out_path;
    int threads = 4, ngl = 0, n_cpu_moe = 0, ubatch = 512, ctx_len = 4096, limit = 0, synthetic = 0, seed = 3;
    bool no_repack = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char * what) -> const char * {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", what); std::exit(2); }
            return argv[++i];
        };
        if (a == "-m") model_path = next("-m");
        else if (a == "--in") in_path = next("--in");
        else if (a == "--out") out_path = next("--out");
        else if (a == "-t") threads = std::atoi(next("-t"));
        else if (a == "--ngl") ngl = std::atoi(next("--ngl"));
        else if (a == "--n-cpu-moe") n_cpu_moe = std::atoi(next("--n-cpu-moe"));
        else if (a == "--ubatch") ubatch = std::atoi(next("--ubatch"));
        else if (a == "--ctx") ctx_len = std::atoi(next("--ctx"));
        else if (a == "--limit") limit = std::atoi(next("--limit"));
        else if (a == "--synthetic") synthetic = std::atoi(next("--synthetic"));
        else if (a == "--seed") seed = std::atoi(next("--seed"));
        else if (a == "--no-repack") no_repack = true;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || out_path.empty() || (in_path.empty() && synthetic <= 0) || ubatch < 1 || ctx_len < 8) {
        std::fprintf(stderr, "usage: dump_hidden -m MODEL.gguf (--in samples.ids | --synthetic N [--seed S]) --out hidden.bin [-t N] [--ngl N] [--n-cpu-moe N] [--ubatch N] [--ctx N] [--no-repack] [--limit N]\n");
        return 2;
    }
    if (std::ifstream(out_path).good()) { std::fprintf(stderr, "%s already exists; refusing to overwrite it\n", out_path.c_str()); return 2; }

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

    // ---- samples -------------------------------------------------------------------------------------------------------------------------------------------
    std::vector<std::vector<int32_t>> samples;
    if (synthetic > 0) {
        std::vector<int32_t> toks(synthetic);
        uint64_t s = 0x9E3779B97F4A7C15ull ^ (uint64_t) seed * 0xD1B54A32D192ED03ull;      // identical generator to tests/logits_dump.cpp
        for (int i = 0; i < synthetic; ++i) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            toks[i] = (int32_t) (100 + (s % (uint64_t) std::max(1, n_vocab - 200)));
        }
        samples.push_back(std::move(toks));
    } else {
        std::ifstream in(in_path, std::ios::binary);
        if (!in) { std::fprintf(stderr, "cannot read %s\n", in_path.c_str()); return 3; }
        int32_t magic = 0, n = 0;
        in.read((char *) &magic, 4); in.read((char *) &n, 4);
        if (!in || magic != 0x50534944 || n < 0) { std::fprintf(stderr, "%s is not a sample file (bad magic)\n", in_path.c_str()); return 3; }
        for (int k = 0; k < n; ++k) {
            int32_t len = 0;
            in.read((char *) &len, 4);
            if (!in || len < 0 || len > (1 << 24)) { std::fprintf(stderr, "bad sample length at sample %d\n", k); return 3; }
            std::vector<int32_t> ids((size_t) len);
            if (len) in.read((char *) ids.data(), (std::streamsize) len * 4);
            if (!in) { std::fprintf(stderr, "truncated sample file at sample %d\n", k); return 3; }
            for (int32_t id : ids) if (id < 0 || id >= n_vocab) { std::fprintf(stderr, "token id %d outside the vocabulary (%d) in sample %d\n", id, n_vocab, k); return 3; }
            samples.push_back(std::move(ids));
        }
    }
    if (limit > 0 && (int) samples.size() > limit) samples.resize((size_t) limit);

    // ---- context -------------------------------------------------------------------------------------------------------------------------------------------
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) ctx_len;
    cp.n_batch = (uint32_t) ubatch;
    cp.n_ubatch = (uint32_t) ubatch;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.embeddings = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "failed to create a context\n"); llama_model_free(model); return 3; }

    std::FILE * f = std::fopen(out_path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 3; }
    int32_t hdr[3] = { 0x50484944, n_embd, (int32_t) samples.size() };
    std::fwrite(hdr, sizeof(int32_t), 3, f);

    llama_batch batch = llama_batch_init(ubatch, 0, 1);
    std::vector<uint16_t> row((size_t) n_embd);
    double max_abs = 0.0;
    long long total = 0;
    for (size_t si = 0; si < samples.size(); ++si) {
        const std::vector<int32_t> & ids = samples[si];
        const int n_tok = (int) std::min<size_t>(ids.size(), (size_t) ctx_len);
        llama_memory_clear(llama_get_memory(ctx), true);
        int32_t nt = n_tok;
        std::fwrite(&nt, sizeof(int32_t), 1, f);
        for (int pos = 0; pos < n_tok; pos += ubatch) {
            const int n = std::min(ubatch, n_tok - pos);
            batch.n_tokens = n;
            for (int j = 0; j < n; ++j) {
                batch.token[j]     = ids[(size_t) pos + j];
                batch.pos[j]       = pos + j;
                batch.n_seq_id[j]  = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j]    = 1;
            }
            if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "llama_decode failed in sample %zu at position %d\n", si, pos); return 4; }
            for (int j = 0; j < n; ++j) {
                const float * e = llama_get_embeddings_ith(ctx, j);
                if (!e) { std::fprintf(stderr, "no embeddings for sample %zu position %d\n", si, pos + j); return 4; }
                for (int d = 0; d < n_embd; ++d) {
                    if (!std::isfinite(e[d])) { std::fprintf(stderr, "non-finite hidden value at sample %zu position %d\n", si, pos + j); return 4; }
                    max_abs = std::max(max_abs, (double) std::fabs(e[d]));
                    row[(size_t) d] = f32_to_bf16(e[d]);
                }
                std::fwrite(row.data(), sizeof(uint16_t), (size_t) n_embd, f);
            }
        }
        total += n_tok;
        if ((si + 1) % 8 == 0 || si + 1 == samples.size()) std::fprintf(stderr, "[dump_hidden] %zu/%zu samples, %lld positions\n", si + 1, samples.size(), total);
    }
    llama_batch_free(batch);
    std::fclose(f);
    std::fprintf(stderr, "[dump_hidden] wrote %s: %zu samples, %lld positions, n_embd %d, largest |value| %.2f\n", out_path.c_str(), samples.size(), total, n_embd, max_abs);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
