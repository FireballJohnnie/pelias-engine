// File: <project>\moe_track\pelias_serve\cold_weights_view.h
// Pass 1 / piece 1: one clean copy of every routed expert, in the GGUF's own ("plain") quantised layout, that the CPU worker pool (cold_pool.h) and the GPU
// (zero-copy kernels) read through the SAME virtual addresses.
//
// WHY A SEPARATE COPY. llama.cpp's CPU backend re-packs Q4_K/Q6_K weights into interleaved CPU_REPACK blocks that neither a GPU kernel nor ggml's plain vec_dot
// can read. This view never touches llama.cpp's loaded model: it parses the GGUF header itself (canonical layout: data starts at the next multiple of
// general.alignment after the tensor table; a tensor starts at data_start + its stored offset), reads the bytes of blk.N.ffn_{gate,up,down}_exps.weight straight
// from the file, and puts them in one anonymous arena. Every tensor sits at a 2 MiB-aligned address and takes a whole number of 2 MiB pages (for the real model
// the expert tensors are exact multiples of 2 MiB, so nothing is wasted), the arena is advised MADV_HUGEPAGE, and pin() registers each tensor with
// cudaHostRegister(Mapped|Portable) so the GPU can read them over PCIe.
//
// TWO ADDRESSES PER MATRIX (measured on this WSL2 machine, microbench/07_pin_modes): for memory registered with cudaHostRegister the pointer the GPU must use is
// NOT the host pointer; cudaHostGetDevicePointer returns a different one, and a kernel handed the host pointer faults. So every ColdMat has `base` (CPU only) and
// `dev_base` (GPU kernels only; set by pin()). Whether the two differ by a constant is reported by pin() for information; nothing relies on it.
//
// LAYOUT OF ONE TENSOR (unchanged from the file): [n_experts][rows][row_bytes]; expert e, row r starts at base + e*expert_stride + r*row_bytes.
//   gate, up : rows = n_ff (512), cols = hidden (2048)      down : rows = hidden (2048), cols = n_ff (512)
// This is exactly the addressing of cold_pool.h's `Mat`, so a ColdMat converts to one field by field (cold_pool.h is not included here: it needs ggml's headers).
//
// WHAT IS NOT DONE HERE: no dot products, no scheduling; and a "K-slice-major" repack of the down matrix (so a K-slice is contiguous) is possible later but
// deliberately not done now: the bytes stay identical to the file, which is what the tests check.
//
// Built with: cold_weights_view.cpp (needs only the C++ standard library and POSIX; CUDA calls exist only when compiled with -DPELIAS_COLD_VIEW_CUDA, or
// with -DPELIAS_CUDA_SHIM for the CPU-only stand-in used by the host tests).
#pragma once

#include "pelias_defs.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pelias {

constexpr size_t COLD_HUGE_PAGE = static_cast<size_t>(2) << 20;   // 2 MiB

// ---- GGUF quantisation geometry (values of ggml's enum ggml_type) ---------------------------------------------------------------------------------------
struct QuantGeom {
    int32_t     type_id;
    const char* name;
    int32_t     block_elems;    // elements per block (QK_K = 256 for the K-quants)
    int32_t     block_bytes;
};
const QuantGeom* quant_geom(int32_t type_id);             // nullptr for a type this reader does not know

// ---- canonical GGUF header parse ----------------------------------------------------------------------------------------------------------------------
struct GgufTensorInfo {
    std::string name;
    int32_t     type_id = 0;
    int32_t     n_dims = 0;
    uint64_t    dims[4] = {1, 1, 1, 1};                    // ggml order: dims[0] is the row length
    uint64_t    data_rel_offset = 0;                      // as stored in the file (relative to data_start)
    uint64_t    file_offset = 0;                          // data_start + data_rel_offset
    uint64_t    n_bytes = 0;
};

struct GgufTable {
    uint32_t                    version = 0;
    uint32_t                    alignment = 32;           // general.alignment, default 32
    uint64_t                    data_start = 0;
    uint64_t                    file_size = 0;
    std::string                 architecture;             // general.architecture, "" if absent
    std::vector<GgufTensorInfo> tensors;
    const GgufTensorInfo* find(const std::string& name) const;
};

// Reads only the header (never the tensor data). v2 and v3 files. Every offset is range-checked against the file size. Returns false and fills *err on any problem.
bool parse_gguf_table(const std::string& path, GgufTable* out, std::string* err);

// ---- the view -------------------------------------------------------------------------------------------------------------------------------------------
struct ColdMat {
    const uint8_t* base = nullptr;        // expert 0, row 0, CPU address. Never hand this to a GPU kernel (see TWO ADDRESSES above).
    const uint8_t* dev_base = nullptr;    // expert 0, row 0, GPU address (from cudaHostGetDevicePointer). nullptr until pin() succeeded for this matrix.
    int32_t        type_id = 0;           // ggml_type value: 12 = Q4_K, 14 = Q6_K, ...
    int64_t        rows = 0;              // output rows of one expert
    int64_t        cols = 0;              // inputs (K)
    size_t         row_bytes = 0;
    size_t         expert_stride = 0;     // = rows * row_bytes
    int64_t        n_experts = 0;
    bool           gpu_ok = false;        // registered with the driver and dev_base is valid (all three matrices of a layer are set together)
    const uint8_t* row(int e, int64_t r) const { return base + static_cast<size_t>(e) * expert_stride + static_cast<size_t>(r) * row_bytes; }
    const uint8_t* expert(int e) const { return base + static_cast<size_t>(e) * expert_stride; }
    // Same offsets on the GPU side; only call when gpu_ok.
    const uint8_t* dev_row(int e, int64_t r) const { return dev_base + static_cast<size_t>(e) * expert_stride + static_cast<size_t>(r) * row_bytes; }
    const uint8_t* dev_expert(int e) const { return dev_base + static_cast<size_t>(e) * expert_stride; }
};

struct ColdLayerView {
    int     layer = -1;
    ColdMat gate, up, down;
};

struct ColdViewOptions {
    std::string gguf_path;
    int         first_layer = 0;
    int         last_layer = -1;          // inclusive; -1 = the last layer present (blk.40, the MTP block, only with include_mtp)
    bool        include_mtp = false;      // also take layers >= N_LAYERS
    bool        strict_shape = true;      // require the qwen35moe geometry of pelias_defs.h (hidden 2048, n_ff 512, 256 experts); tiny test models set false
    int         load_threads = 4;
    bool        advise_hugepages = true;  // madvise(MADV_HUGEPAGE) on the arena
    bool        verify_after_load = false;   // read the file a second time and compare every byte
};

struct ColdLoadStats {
    size_t  n_layers = 0;
    size_t  tensor_bytes = 0;             // bytes copied from the file
    size_t  arena_bytes = 0;              // including 2 MiB padding
    double  load_seconds = 0.0;
    size_t  huge_backed_bytes = 0;        // AnonHugePages inside the arena (from /proc/self/smaps; approximate when the kernel merged neighbouring mappings)
};

struct ColdPinReport {
    size_t      regions_ok = 0;
    size_t      regions_failed = 0;
    size_t      regions_skipped = 0;      // beyond max_bytes
    size_t      pinned_bytes = 0;
    // Information only, nothing fails on these two. dev_offset = device address minus host address of the first registered region.
    bool        uva_identical = true;     // cudaHostGetDevicePointer returned the host address for every registered region
    bool        dev_offset_constant = true;   // the (device - host) difference was the same for every registered region
    int64_t     dev_offset = 0;
    std::string first_error;
};

class ColdWeightsView {
public:
    ColdWeightsView();
    ~ColdWeightsView();
    ColdWeightsView(const ColdWeightsView&) = delete;
    ColdWeightsView& operator=(const ColdWeightsView&) = delete;

    // Parse the header, allocate the arena, copy the expert tensors in. Returns false and fills *err on failure (nothing stays allocated then).
    bool open(const ColdViewOptions& opt, std::string* err);

    // Register the arena regions with the CUDA driver (cudaHostRegister Mapped|Portable), in layer order, stopping at max_bytes (0 = no limit) or at the first
    // driver failure. For each registered region it stores the device pointer (dev_base); the host and device addresses may differ. A layer is gpu_ok only if
    // all three of its tensors were registered and have a device pointer. Returns true if at least one region was registered. In a build without CUDA it registers nothing, returns false and says so in the report. Safe to call once after open().
    bool pin(size_t max_bytes, ColdPinReport* report);

    // Unregister and free everything. open() may be called again afterwards.
    void close();

    bool is_open() const { return arena_ != nullptr; }
    bool has_layer(int layer) const;
    const ColdLayerView& layer(int layer) const;           // only for layers with has_layer(layer)
    const std::vector<int>& layers() const { return layer_ids_; }
    int64_t n_experts() const { return n_experts_; }
    const ColdLoadStats& stats() const { return stats_; }
    const GgufTable& table() const { return table_; }
    // The arena as a whole (2 MiB aligned); the tests use it to check alignment.
    const uint8_t* arena_base() const { return arena_; }
    size_t arena_size() const { return arena_size_; }
    // Re-read the file and compare every byte of every copied tensor with the arena. false + *err on the first difference.
    bool verify_against_file(std::string* err, int threads = 4) const;
    std::string describe() const;

private:
    struct Region { const uint8_t* ptr; size_t bytes; size_t padded; uint64_t file_offset; int layer; int which; bool pinned; const uint8_t* dev; };
    GgufTable            table_;
    ColdViewOptions      opt_;
    uint8_t*             arena_ = nullptr;
    size_t               arena_size_ = 0;
    void*                map_base_ = nullptr;
    size_t               map_size_ = 0;
    std::vector<Region>  regions_;
    std::vector<int>     layer_ids_;
    std::vector<ColdLayerView> layers_;                     // parallel to layer_ids_
    int64_t              n_experts_ = 0;
    ColdLoadStats        stats_;
    bool                 pinned_any_ = false;
};

// True when this translation unit set was built with CUDA calls (real or the stand-in).
bool cold_view_built_with_cuda();

}  // namespace pelias
