#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
typedef struct CUev_st* cudaEvent_t; typedef struct CUstream_st* cudaStream_t;
enum cudaError_t{cudaSuccess=0}; enum ggml_status{GGML_STATUS_SUCCESS=0};
struct ggml_cgraph{int n_nodes;}; struct ggml_backend; typedef ggml_backend* ggml_backend_t;
struct ggml_backend_cuda_context{int device; cudaStream_t stream(){return nullptr;}};
struct ggml_backend{void* context;};
#define CUDA_CHECK(x) (void)(x)
cudaError_t cudaEventElapsedTime(float*,cudaEvent_t,cudaEvent_t); cudaError_t cudaEventCreate(cudaEvent_t*); cudaError_t cudaEventRecord(cudaEvent_t,cudaStream_t);
void ggml_cuda_set_device(int);
// ---- PELIAS timeline (measurement build only; does nothing unless PELIAS_TIMELINE=<file prefix> is set) ----
// Every graph_compute call on the CUDA backend is bracketed by two CUDA events (timing enabled). When the stream is synchronized, the pending
// events are resolved and written to <prefix>.gpu.tsv:   G <seq> <t_start_ns> <t_launch_ns> <gpu_us> <n_nodes>
//   gpu_us      = GPU time between the two events (includes any idle gap inside the call; the call is one scheduler split)
//   t_start_ns  = the event's time on the host clock (steady_clock), anchored to the moment the stream synchronize returned: the LAST event of the batch
//                 is taken to have completed at that host time, the others are placed before it by their measured GPU-side distance. [ESTIMATE: error = delay between GPU idle and the sync call returning]
//   t_launch_ns = host time when the split was handed to the backend
#include <chrono>
struct pelias_tl_rec { cudaEvent_t t0; cudaEvent_t t1; int64_t host_launch_ns; int n_nodes; };
static bool pelias_tl_on() {
    static const bool on = getenv("PELIAS_TIMELINE") != nullptr;
    return on;
}
static int64_t pelias_tl_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static std::vector<pelias_tl_rec> g_pelias_tl_pending;
static std::vector<cudaEvent_t>   g_pelias_tl_free;
static FILE * g_pelias_tl_file = nullptr;
static long   g_pelias_tl_seq  = 0;

static void pelias_tl_flush() {
    if (g_pelias_tl_pending.empty()) {
        return;
    }
    const int64_t h_sync = pelias_tl_ns();
    if (!g_pelias_tl_file) {
        const char * pre = getenv("PELIAS_TIMELINE");
        std::string path = std::string(pre) + ".gpu.tsv";
        g_pelias_tl_file = fopen(path.c_str(), "w");
    }
    cudaEvent_t last = g_pelias_tl_pending.back().t1;
    for (const pelias_tl_rec & r : g_pelias_tl_pending) {
        float ms_to_last = 0.0f, ms_dur = 0.0f;
        cudaEventElapsedTime(&ms_to_last, r.t0, last);
        cudaEventElapsedTime(&ms_dur, r.t0, r.t1);
        if (g_pelias_tl_file) {
            fprintf(g_pelias_tl_file, "G\t%ld\t%lld\t%lld\t%.1f\t%d\n", ++g_pelias_tl_seq,
                    (long long) (h_sync - (int64_t) ((double) ms_to_last * 1.0e6)), (long long) r.host_launch_ns, (double) ms_dur * 1000.0, r.n_nodes);
        }
        g_pelias_tl_free.push_back(r.t0);
        g_pelias_tl_free.push_back(r.t1);
    }
    g_pelias_tl_pending.clear();
    if (g_pelias_tl_file) {
        fflush(g_pelias_tl_file);
    }
}

static enum ggml_status ggml_backend_cuda_graph_compute_impl(ggml_backend_t, ggml_cgraph*){return GGML_STATUS_SUCCESS;}
// PELIAS timeline wrapper: same behaviour as the stock call, plus two timing events around it when PELIAS_TIMELINE is set
static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    if (!pelias_tl_on() || g_pelias_tl_pending.size() >= 8192) {
        return ggml_backend_cuda_graph_compute_impl(backend, cgraph);
    }
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(cuda_ctx->device);
    pelias_tl_rec r;
    r.n_nodes = cgraph->n_nodes;
    for (cudaEvent_t * e : { &r.t0, &r.t1 }) {
        if (!g_pelias_tl_free.empty()) {
            *e = g_pelias_tl_free.back();
            g_pelias_tl_free.pop_back();
        } else {
            CUDA_CHECK(cudaEventCreate(e));
        }
    }
    r.host_launch_ns = pelias_tl_ns();
    CUDA_CHECK(cudaEventRecord(r.t0, cuda_ctx->stream()));
    const enum ggml_status st = ggml_backend_cuda_graph_compute_impl(backend, cgraph);
    CUDA_CHECK(cudaEventRecord(r.t1, cuda_ctx->stream()));
    g_pelias_tl_pending.push_back(r);
    return st;
}

