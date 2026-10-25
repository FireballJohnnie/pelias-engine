/*
 * bar_probe.c  -  read-only CUDA driver probe for the BAR1 / host-visible-VRAM question (WSL2).
 * Loads libcuda.so.1 at run time (dlopen), so it needs no CUDA toolkit to build, only gcc.
 * Types are declared here by hand (no cuda.h) so it builds anywhere; layouts follow cuda.h
 * [DERIVED from memory, NOT checked against your cuda.h - run_bar_probe.sh greps your cuda.h
 * for the enum values, and the cuMemCreate control below validates the property layout].
 * It allocates 1 GiB of VRAM and a few small VMM handles, frees them, and exits.
 * It never reads or writes through any mapped pointer.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <dlfcn.h>

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef unsigned long long CUdeviceptr;

typedef struct {
    int type;
    union { int fd; struct { void *handle; const void *name; } win32; const void *nvSciBufObject; } handle;
    unsigned long long size;
    unsigned int flags;
    unsigned int reserved[16];
} ExtMemDesc;

typedef struct {
    int type;                 /* CU_MEM_ALLOCATION_TYPE_PINNED = 1 */
    int requestedHandleTypes; /* NONE 0, POSIX_FD 1, WIN32 2, WIN32_KMT 4 */
    struct { int type; int id; } location; /* CU_MEM_LOCATION_TYPE_DEVICE = 1 */
    void *win32HandleMetaData;
    struct { unsigned char compressionType; unsigned char gpuDirectRDMACapable; unsigned short usage; unsigned char reserved[4]; } allocFlags;
} MemProp;

typedef struct { unsigned long long p2pToken; unsigned int vaSpaceToken; } P2PTokens;

static CUresult (*p_cuInit)(unsigned int);
static CUresult (*p_cuDriverGetVersion)(int *);
static CUresult (*p_cuDeviceGet)(CUdevice *, int);
static CUresult (*p_cuDeviceGetName)(char *, int, CUdevice);
static CUresult (*p_cuDeviceGetAttribute)(int *, int, CUdevice);
static CUresult (*p_cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice);
static CUresult (*p_cuCtxSetCurrent)(CUcontext);
static CUresult (*p_cuMemAlloc_v2)(CUdeviceptr *, size_t);
static CUresult (*p_cuMemFree_v2)(CUdeviceptr);
static CUresult (*p_cuMemGetHandleForAddressRange)(void *, CUdeviceptr, size_t, int, unsigned long long);
static CUresult (*p_cuPointerGetAttribute)(void *, int, CUdeviceptr);
static CUresult (*p_cuImportExternalMemory)(void **, const ExtMemDesc *);
static CUresult (*p_cuMemGetAllocationGranularity)(size_t *, const MemProp *, int);
static CUresult (*p_cuMemCreate)(unsigned long long *, size_t, const MemProp *, unsigned long long);
static CUresult (*p_cuMemRelease)(unsigned long long);
static CUresult (*p_cuGetErrorName)(CUresult, const char **);

static const char *en(CUresult r) {
    const char *s = NULL;
    if (p_cuGetErrorName && p_cuGetErrorName(r, &s) == 0 && s) return s;
    return "?";
}

#define LOAD(name) do { void *q_ = dlsym(lib, #name); p_##name = (void *)q_; \
    printf("symbol %-34s %s\n", #name, q_ ? "present" : "MISSING"); } while (0)
#define RESULT(label, r) printf("RESULT %-44s code=%d %s\n", label, (int)(r), en(r))

static void bar1(const char *tag) {
    FILE *f = popen("nvidia-smi -q -d MEMORY 2>/dev/null | awk '/BAR1 Memory Usage/{f=1;next} f&&/Used/{print $3,$4;exit}'", "r");
    char buf[64] = "?";
    if (f) { if (!fgets(buf, sizeof buf, f)) strcpy(buf, "?"); pclose(f); }
    buf[strcspn(buf, "\n")] = 0;
    printf("BAR1 used [%s]: %s\n", tag, buf);
}

int main(void) {
    void *lib = dlopen("/usr/lib/wsl/lib/libcuda.so.1", RTLD_NOW);
    if (!lib) lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (!lib) { printf("FATAL: cannot dlopen libcuda.so.1 (%s)\n", dlerror()); return 2; }

    LOAD(cuInit); LOAD(cuDriverGetVersion); LOAD(cuDeviceGet); LOAD(cuDeviceGetName);
    LOAD(cuDeviceGetAttribute); LOAD(cuDevicePrimaryCtxRetain); LOAD(cuCtxSetCurrent);
    LOAD(cuMemAlloc_v2); LOAD(cuMemFree_v2); LOAD(cuMemGetHandleForAddressRange);
    LOAD(cuPointerGetAttribute); LOAD(cuImportExternalMemory); LOAD(cuMemGetAllocationGranularity);
    LOAD(cuMemCreate); LOAD(cuMemRelease); LOAD(cuGetErrorName);

    if (!p_cuInit || !p_cuDeviceGet || !p_cuDeviceGetAttribute) { printf("FATAL: core symbols missing\n"); return 2; }
    CUresult r = p_cuInit(0);
    RESULT("cuInit", r);
    if (r) return 2;

    int ver = 0; if (p_cuDriverGetVersion) p_cuDriverGetVersion(&ver);
    CUdevice dev = 0; r = p_cuDeviceGet(&dev, 0); RESULT("cuDeviceGet(0)", r);
    char name[128] = "?"; if (p_cuDeviceGetName) p_cuDeviceGetName(name, sizeof name, dev);
    printf("driver API version %d, device 0: %s\n", ver, name);

    static const struct { const char *n; int c; } attrs[] = {
        {"INTEGRATED", 18}, {"CAN_MAP_HOST_MEMORY", 19}, {"UNIFIED_ADDRESSING", 41},
        {"MANAGED_MEMORY", 83}, {"PAGEABLE_MEMORY_ACCESS", 88}, {"CONCURRENT_MANAGED_ACCESS", 89},
        {"HOST_REGISTER_SUPPORTED", 99}, {"VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED", 102},
        {"HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED", 103}, {"HANDLE_TYPE_WIN32_HANDLE_SUPPORTED", 104},
        {"GPU_DIRECT_RDMA_SUPPORTED", 116}, {"DMA_BUF_SUPPORTED", 124},
    };
    for (size_t i = 0; i < sizeof attrs / sizeof attrs[0]; i++) {
        int v = -1; r = p_cuDeviceGetAttribute(&v, attrs[i].c, dev);
        printf("ATTR %-46s (%3d) = %d  (ret=%d %s)\n", attrs[i].n, attrs[i].c, v, r, en(r));
    }

    CUcontext ctx = NULL;
    r = p_cuDevicePrimaryCtxRetain(&ctx, dev); RESULT("cuDevicePrimaryCtxRetain", r);
    if (!r) { r = p_cuCtxSetCurrent(ctx); RESULT("cuCtxSetCurrent", r); }

    bar1("before any allocation");

    /* 1 GiB of ordinary device memory */
    CUdeviceptr d = 0;
    r = p_cuMemAlloc_v2(&d, (size_t)1 << 30); RESULT("cuMemAlloc 1 GiB", r);
    bar1("after cuMemAlloc 1 GiB");

    if (!r) {
        if (p_cuMemGetHandleForAddressRange) {
            int fd = -1;
            r = p_cuMemGetHandleForAddressRange(&fd, d, (size_t)1 << 21, 1 /* DMA_BUF_FD */, 0);
            RESULT("cuMemGetHandleForAddressRange DMA_BUF_FD", r);
            if (!r && fd >= 0) { printf("  dma-buf fd = %d\n", fd); close(fd); }
        }
        P2PTokens tok; memset(&tok, 0, sizeof tok);
        r = p_cuPointerGetAttribute(&tok, 5 /* P2P_TOKENS */, d);
        RESULT("cuPointerGetAttribute P2P_TOKENS", r);
    }

    /* external memory import: which handle types does the driver even accept?  Dummy handles. */
    if (p_cuImportExternalMemory) {
        static const struct { const char *n; int t; } types[] = {
            {"OPAQUE_FD", 1}, {"OPAQUE_WIN32", 2}, {"OPAQUE_WIN32_KMT", 3},
            {"D3D12_HEAP", 4}, {"D3D12_RESOURCE", 5}, {"D3D11_RESOURCE", 6}, {"D3D11_RESOURCE_KMT", 7},
        };
        int devnull = open("/dev/null", O_RDONLY);
        for (size_t i = 0; i < sizeof types / sizeof types[0]; i++) {
            ExtMemDesc ed; memset(&ed, 0, sizeof ed);
            ed.type = types[i].t; ed.size = 4096;
            if (types[i].t == 1) ed.handle.fd = devnull;
            void *ext = NULL;
            char lab[80]; snprintf(lab, sizeof lab, "cuImportExternalMemory %s (dummy)", types[i].n);
            r = p_cuImportExternalMemory(&ext, &ed); RESULT(lab, r);
        }
        if (devnull >= 0) close(devnull);
    }

    /* VMM: granularity, a plain control, then an RDMA-capable request */
    if (p_cuMemGetAllocationGranularity && p_cuMemCreate) {
        MemProp pr; memset(&pr, 0, sizeof pr);
        pr.type = 1; pr.location.type = 1; pr.location.id = dev;
        size_t gran = 0;
        r = p_cuMemGetAllocationGranularity(&gran, &pr, 0); RESULT("cuMemGetAllocationGranularity minimum", r);
        printf("  granularity = %zu bytes\n", gran);
        if (!r && gran) {
            unsigned long long h = 0;
            r = p_cuMemCreate(&h, gran, &pr, 0); RESULT("cuMemCreate control (plain, no flags)", r);
            if (!r) p_cuMemRelease(h);

            MemProp pf = pr; pf.requestedHandleTypes = 1; /* POSIX FD */
            h = 0; r = p_cuMemCreate(&h, gran, &pf, 0); RESULT("cuMemCreate + POSIX_FD handle type", r);
            if (!r) p_cuMemRelease(h);

            MemProp pg = pr; pg.allocFlags.gpuDirectRDMACapable = 1;
            h = 0; r = p_cuMemCreate(&h, gran, &pg, 0); RESULT("cuMemCreate + gpuDirectRDMACapable", r);
            if (!r) p_cuMemRelease(h);
        }
    }
    bar1("after VMM handles released");

    if (d) { r = p_cuMemFree_v2(d); RESULT("cuMemFree", r); }
    bar1("after free");
    printf("DONE\n");
    return 0;
}
