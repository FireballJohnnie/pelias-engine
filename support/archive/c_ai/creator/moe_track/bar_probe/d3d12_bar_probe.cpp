// d3d12_bar_probe.cpp - can a D3D12 resource that the CPU can map be imported into CUDA inside WSL2?
// Build: see run_d3d12_probe.sh.  Needs the Microsoft DirectX-Headers (header only, MIT) and g++; CUDA toolkit NOT needed (libcuda via dlopen).
// For each heap kind it: creates a shared buffer (as a committed resource and as a heap+placed resource), maps it on the CPU if possible,
// writes a pattern, samples nvidia-smi BAR1 used, makes a shared handle, tries cuImportExternalMemory + cuExternalMemoryGetMappedBuffer,
// reads 4 KiB back through CUDA and compares, and times CPU stores into the mapping. Prints one RESULT line per step. Read-only toward the system:
// allocates 64 MiB per case (freed at exit), nothing is written outside its own buffers.
#define INITGUID
#include <wsl/winadapter.h>
#include <directx/d3d12.h>
#include <directx/dxcore.h>
#include <dxguids/dxguids.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <chrono>
#include <vector>
#include <immintrin.h>

// ---------------- CUDA driver API, by hand (no cuda.h) ----------------
typedef int CUresult; typedef int CUdevice; typedef void *CUcontext; typedef unsigned long long CUdeviceptr;
typedef struct { int type; union { int fd; struct { void *handle; const void *name; } win32; const void *nv; } handle; unsigned long long size; unsigned int flags; unsigned int reserved[16]; } ExtMemDesc;
typedef struct { unsigned long long offset; unsigned long long size; unsigned int flags; unsigned int reserved[16]; } ExtBufDesc;
static CUresult (*p_cuInit)(unsigned int);
static CUresult (*p_cuDeviceGet)(CUdevice *, int);
static CUresult (*p_cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice);
static CUresult (*p_cuCtxSetCurrent)(CUcontext);
static CUresult (*p_cuImportExternalMemory)(void **, const ExtMemDesc *);
static CUresult (*p_cuExternalMemoryGetMappedBuffer)(CUdeviceptr *, void *, const ExtBufDesc *);
static CUresult (*p_cuDestroyExternalMemory)(void *);
static CUresult (*p_cuMemcpyDtoH_v2)(void *, CUdeviceptr, size_t);
static CUresult (*p_cuMemsetD8_v2)(CUdeviceptr, unsigned char, size_t);
static CUresult (*p_cuCtxSynchronize)(void);
static CUresult (*p_cuGetErrorName)(CUresult, const char **);
static const char *en(CUresult r) { const char *s = NULL; if (p_cuGetErrorName && p_cuGetErrorName(r, &s) == 0 && s) return s; return "?"; }

static void bar1(const char *tag) {
    FILE *f = popen("nvidia-smi -q -d MEMORY 2>/dev/null | awk '/BAR1 Memory Usage/{f=1;next} f&&/Used/{print $3,$4;exit}'", "r");
    char b[64] = "?"; if (f) { if (!fgets(b, sizeof b, f)) strcpy(b, "?"); pclose(f); }
    b[strcspn(b, "\n")] = 0; printf("   BAR1 used [%s]: %s\n", tag, b); fflush(stdout);
}

static const size_t kSize = 64u << 20;   // 64 MiB per case
static const size_t kProbe = 4096;

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// CPU store cost into a mapped pointer: bulk (64 MiB, AVX2 non-temporal) and small (4 KiB + sfence, repeated)
static void time_cpu_stores(void *dst, const char *tag) {
    uint8_t *src = (uint8_t *)aligned_alloc(64, kSize); memset(src, 0x3C, kSize);
    uint8_t *d = (uint8_t *)dst;
    double t0 = now_s();
    for (size_t i = 0; i < kSize; i += 64) { _mm256_stream_si256((__m256i *)(d + i), _mm256_load_si256((const __m256i *)(src + i))); _mm256_stream_si256((__m256i *)(d + i + 32), _mm256_load_si256((const __m256i *)(src + i + 32))); }
    _mm_sfence(); double t1 = now_s();
    printf("   CPU store %s: 64 MiB in %.1f ms = %.2f GB/s [MEASURED, single thread, AVX2 non-temporal]\n", tag, (t1 - t0) * 1e3, kSize / (t1 - t0) / 1e9);
    const int N = 20000; double a = now_s();
    for (int k = 0; k < N; k++) { for (size_t i = 0; i < kProbe; i += 64) { _mm256_stream_si256((__m256i *)(d + i), _mm256_load_si256((const __m256i *)(src + i))); _mm256_stream_si256((__m256i *)(d + i + 32), _mm256_load_si256((const __m256i *)(src + i + 32))); } _mm_sfence(); }
    double b = now_s();
    printf("   CPU store %s: 4 KiB + sfence = %.0f ns each over %d reps [MEASURED]\n", tag, (b - a) / N * 1e9, N);
    free(src);
}

struct Kind { const char *name; D3D12_HEAP_PROPERTIES hp; D3D12_RESOURCE_STATES state; bool cpu; };

static void try_import(const char *label, HANDLE h, int type, unsigned flags, size_t size, bool cpuVisible, const uint8_t *pattern) {
    ExtMemDesc ed; memset(&ed, 0, sizeof ed); ed.type = type; ed.handle.win32.handle = (void *)h; ed.size = size; ed.flags = flags;
    void *ext = NULL; CUresult r = p_cuImportExternalMemory(&ext, &ed);
    printf("RESULT %-34s cuImportExternalMemory type %d flags %u: code=%d %s\n", label, type, flags, r, en(r)); fflush(stdout);
    if (r) return;
    ExtBufDesc bd; memset(&bd, 0, sizeof bd); bd.offset = 0; bd.size = size; CUdeviceptr dp = 0;
    r = p_cuExternalMemoryGetMappedBuffer(&dp, ext, &bd);
    printf("RESULT %-34s cuExternalMemoryGetMappedBuffer: code=%d %s  dptr=0x%llx\n", label, r, en(r), dp); fflush(stdout);
    if (!r) {
        uint8_t back[kProbe]; memset(back, 0, sizeof back);
        if (!cpuVisible) { p_cuMemsetD8_v2(dp, 0x5A, kProbe); p_cuCtxSynchronize(); }
        CUresult rc = p_cuMemcpyDtoH_v2(back, dp, kProbe);
        bool ok = true;
        if (cpuVisible) { for (size_t i = 0; i < kProbe; i++) if (back[i] != pattern[i]) { ok = false; break; } }
        else { for (size_t i = 0; i < kProbe; i++) if (back[i] != 0x5A) { ok = false; break; } }
        printf("RESULT %-34s CUDA read-back of 4 KiB: copy code=%d %s, %s\n", label, rc, en(rc), ok ? (cpuVisible ? "MATCHES the CPU-written pattern" : "MATCHES the CUDA-written pattern (control)") : "MISMATCH");
        bar1("after CUDA import + read-back");
    }
    p_cuDestroyExternalMemory(ext);
}

int main() {
    void *lc = dlopen("/usr/lib/wsl/lib/libcuda.so.1", RTLD_NOW);
    void *l12 = dlopen("/usr/lib/wsl/lib/libd3d12.so", RTLD_NOW);
    void *ldx = dlopen("/usr/lib/wsl/lib/libdxcore.so", RTLD_NOW);
    if (!lc || !l12 || !ldx) { printf("FATAL: cannot dlopen WSL libs (cuda %p, d3d12 %p, dxcore %p): %s\n", lc, l12, ldx, dlerror()); return 2; }
#define LD(h, n) p_##n = (decltype(p_##n))dlsym(h, #n); if (!p_##n) { printf("FATAL: missing symbol %s\n", #n); return 2; }
    LD(lc, cuInit) LD(lc, cuDeviceGet) LD(lc, cuDevicePrimaryCtxRetain) LD(lc, cuCtxSetCurrent) LD(lc, cuImportExternalMemory)
    LD(lc, cuExternalMemoryGetMappedBuffer) LD(lc, cuDestroyExternalMemory) LD(lc, cuMemcpyDtoH_v2) LD(lc, cuMemsetD8_v2) LD(lc, cuCtxSynchronize)
    p_cuGetErrorName = (decltype(p_cuGetErrorName))dlsym(lc, "cuGetErrorName");
    typedef HRESULT (*PFN_DXCoreCreateAdapterFactory)(REFIID, void **);
    typedef HRESULT (*PFN_D3D12CreateDevice)(IUnknown *, D3D_FEATURE_LEVEL, REFIID, void **);
    auto pDXCore = (PFN_DXCoreCreateAdapterFactory)dlsym(ldx, "DXCoreCreateAdapterFactory");
    auto pCreateDev = (PFN_D3D12CreateDevice)dlsym(l12, "D3D12CreateDevice");
    if (!pDXCore || !pCreateDev) { printf("FATAL: missing DXCoreCreateAdapterFactory or D3D12CreateDevice\n"); return 2; }

    CUresult r = p_cuInit(0); CUdevice dev = 0; CUcontext ctx = NULL;
    p_cuDeviceGet(&dev, 0); p_cuDevicePrimaryCtxRetain(&ctx, dev); p_cuCtxSetCurrent(ctx);
    printf("RESULT cuInit/primary context: cuInit code=%d\n", r);

    // adapter
    IDXCoreAdapterFactory *fac = nullptr;
    HRESULT hr = pDXCore(__uuidof(IDXCoreAdapterFactory), (void **)&fac);
    printf("RESULT DXCoreCreateAdapterFactory hr=0x%08x\n", (unsigned)hr);
    if (FAILED(hr)) return 2;
    IDXCoreAdapterList *list = nullptr;
    GUID attrs[1] = { DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS };
    hr = fac->CreateAdapterList(1, attrs, __uuidof(IDXCoreAdapterList), (void **)&list);
    printf("RESULT CreateAdapterList hr=0x%08x count=%u\n", (unsigned)hr, list ? list->GetAdapterCount() : 0);
    if (FAILED(hr) || !list) return 2;
    IDXCoreAdapter *ad = nullptr;
    for (uint32_t i = 0; i < list->GetAdapterCount(); i++) {
        IDXCoreAdapter *a = nullptr; list->GetAdapter(i, __uuidof(IDXCoreAdapter), (void **)&a);
        char desc[256] = {0}; size_t sz = 0;
        if (a && SUCCEEDED(a->GetPropertySize(DXCoreAdapterProperty::DriverDescription, &sz)) && sz < sizeof desc) a->GetProperty(DXCoreAdapterProperty::DriverDescription, sz, desc);
        printf("   adapter %u: %s\n", i, desc);
        if (!ad && strstr(desc, "NVIDIA")) ad = a;
    }
    if (!ad) { printf("FATAL: no NVIDIA adapter in DXCore list\n"); return 2; }

    ID3D12Device *dv = nullptr;
    hr = pCreateDev((IUnknown *)ad, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), (void **)&dv);
    printf("RESULT D3D12CreateDevice hr=0x%08x\n", (unsigned)hr);
    if (FAILED(hr) || !dv) return 2;

    D3D12_FEATURE_DATA_ARCHITECTURE1 ar = {}; hr = dv->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE1, &ar, sizeof ar);
    printf("RESULT ARCHITECTURE1 hr=0x%08x UMA=%d CacheCoherentUMA=%d IsolatedMMU=%d\n", (unsigned)hr, ar.UMA, ar.CacheCoherentUMA, ar.IsolatedMMU);
    D3D12_FEATURE_DATA_D3D12_OPTIONS16 o16 = {}; hr = dv->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, &o16, sizeof o16);
    printf("RESULT OPTIONS16 hr=0x%08x GPUUploadHeapSupported=%d\n", (unsigned)hr, o16.GPUUploadHeapSupported);
    bar1("start");

    auto mk = [](D3D12_HEAP_TYPE t, D3D12_CPU_PAGE_PROPERTY c, D3D12_MEMORY_POOL m) { D3D12_HEAP_PROPERTIES p = {}; p.Type = t; p.CPUPageProperty = c; p.MemoryPoolPreference = m; p.CreationNodeMask = 1; p.VisibleNodeMask = 1; return p; };
    std::vector<Kind> kinds = {
        { "DEFAULT (VRAM, control)", mk(D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN), D3D12_RESOURCE_STATE_COMMON, false },
        { "UPLOAD (system RAM, control)", mk(D3D12_HEAP_TYPE_UPLOAD, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN), D3D12_RESOURCE_STATE_GENERIC_READ, true },
        { "GPU_UPLOAD (CPU-visible VRAM)", mk(D3D12_HEAP_TYPE_GPU_UPLOAD, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN), D3D12_RESOURCE_STATE_COMMON, true },
        { "CUSTOM L1 + WRITE_COMBINE (guide)", mk(D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE, D3D12_MEMORY_POOL_L1), D3D12_RESOURCE_STATE_COMMON, true },
        { "CUSTOM L0 + WRITE_COMBINE (host WC)", mk(D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE, D3D12_MEMORY_POOL_L0), D3D12_RESOURCE_STATE_GENERIC_READ, true },
    };
    D3D12_RESOURCE_DESC rd = {}; rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; rd.Width = kSize; rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_UNKNOWN; rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; rd.Flags = D3D12_RESOURCE_FLAG_NONE;
    uint8_t pattern[kProbe]; for (size_t i = 0; i < kProbe; i++) pattern[i] = (uint8_t)(i * 31 + 7);
    const DWORD kAll = 0x10000000; // GENERIC_ALL

    for (auto &k : kinds) {
        printf("\n=== %s ===\n", k.name); fflush(stdout);
        for (int path = 0; path < 2; path++) {   // 0 = committed resource, 1 = shared heap + placed resource
            char lab[96]; snprintf(lab, sizeof lab, "%s / %s", k.name, path ? "heap+placed" : "committed");
            ID3D12Resource *res = nullptr; ID3D12Heap *heap = nullptr; void *mp = nullptr;
            if (path == 0) {
                hr = dv->CreateCommittedResource(&k.hp, D3D12_HEAP_FLAG_SHARED, &rd, k.state, nullptr, __uuidof(ID3D12Resource), (void **)&res);
            } else {
                D3D12_HEAP_DESC hd = {}; hd.SizeInBytes = kSize; hd.Properties = k.hp; hd.Alignment = 65536; hd.Flags = D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
                hr = dv->CreateHeap(&hd, __uuidof(ID3D12Heap), (void **)&heap);
                printf("RESULT %-34s CreateHeap hr=0x%08x\n", lab, (unsigned)hr);
                if (SUCCEEDED(hr)) hr = dv->CreatePlacedResource(heap, 0, &rd, k.state, nullptr, __uuidof(ID3D12Resource), (void **)&res);
            }
            printf("RESULT %-34s create resource hr=0x%08x\n", lab, (unsigned)hr); fflush(stdout);
            if (FAILED(hr) || !res) { if (heap) heap->Release(); continue; }
            if (k.cpu) {
                D3D12_RANGE none = {0, 0};
                hr = res->Map(0, &none, &mp);
                printf("RESULT %-34s Map hr=0x%08x ptr=%p\n", lab, (unsigned)hr, mp);
                if (SUCCEEDED(hr) && mp) { memcpy(mp, pattern, kProbe); _mm_sfence(); }
                bar1("after Map + 4 KiB write");
            }
            HANDLE sh = nullptr;
            ID3D12DeviceChild *obj = path ? (ID3D12DeviceChild *)heap : (ID3D12DeviceChild *)res;
            hr = dv->CreateSharedHandle(obj, nullptr, kAll, nullptr, &sh);
            printf("RESULT %-34s CreateSharedHandle hr=0x%08x handle=%p\n", lab, (unsigned)hr, (void *)sh); fflush(stdout);
            if (SUCCEEDED(hr) && sh) {
                // The shared handle looks like a small integer (0x28), i.e. a Linux fd on WSL2; on little-endian its low 32 bits are what the fd field of
                // the descriptor reads, so types 1 (OPAQUE_FD) and 9 (DMABUF_FD) get the same value as an fd.  Types: 5 D3D12_RESOURCE, 4 D3D12_HEAP,
                // 2 OPAQUE_WIN32, 3 OPAQUE_WIN32_KMT, 6/7 D3D11_RESOURCE(_KMT), 1 OPAQUE_FD, 9 DMABUF_FD.
                const int tr[] = { 5, 2, 3, 6, 7, 1, 9 }, th[] = { 4, 2, 3, 6, 7, 1, 9 };
                for (int ti = 0; ti < 7; ti++)
                    try_import(lab, sh, path == 0 ? tr[ti] : th[ti], path == 0 ? 1u : 0u, kSize, k.cpu && mp, pattern);
            }
            if (k.cpu && mp && path == 0) time_cpu_stores(mp, k.name);
            if (k.cpu && mp) res->Unmap(0, nullptr);
            res->Release(); if (heap) heap->Release();
        }
    }
    printf("\nDONE\n");
    return 0;
}
