// File: <project>\moe_track\lib\cuda_shim.h
// CPU-only stand-in for the parts of the CUDA runtime that moe_track code uses. TEST TOOL ONLY.
//
// Purpose: lets the .cu / .cuh sources be compiled with an ordinary C++ compiler (g++ -std=c++17 -fsyntax-only, or a
// full build) on a machine with no GPU and no nvcc, so type errors, missing includes and misuse of the API are caught,
// and so kernels that do not need block-level synchronisation can be run serially against a CPU reference.
//
// Limits (by design): there is no parallelism, no warp semantics (shuffle intrinsics return their input), no shared memory
// isolation, no streams or timing. A kernel that passes here is not thereby correct or fast on a GPU.
#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline __attribute__((always_inline))
#define __shared__ static
#define __constant__ static
#define __noinline__ __attribute__((noinline))
#define __launch_bounds__(...)
#define __restrict__ __restrict

struct dim3 {
    unsigned x, y, z;
    dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
    dim3(int x_) : x(unsigned(x_)), y(1), z(1) {}
};

inline dim3 threadIdx{0, 0, 0};
inline dim3 blockIdx{0, 0, 0};
inline dim3 blockDim{1, 1, 1};
inline dim3 gridDim{1, 1, 1};
inline const int warpSize = 32;

struct uint4 { unsigned x, y, z, w; };
struct int4 { int x, y, z, w; };
struct float4 { float x, y, z, w; };
struct uint2 { unsigned x, y; };

// ---- fp16 (software, round-to-nearest-even) ------------------------------------------------------------------
struct __half { uint16_t x; };
typedef __half half;

inline float __half2float(__half h) {
    const uint32_t sign = (h.x >> 15) & 1u, exp = (h.x >> 10) & 0x1Fu, mant = h.x & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign << 31;
        else {                       // subnormal
            int e = -1;
            uint32_t m = mant;
            do { e++; m <<= 1; } while (!(m & 0x400u));
            f = (sign << 31) | (uint32_t(127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) f = (sign << 31) | 0x7F800000u | (mant << 13);
    else f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

inline __half __float2half(float v) {
    uint32_t f;
    std::memcpy(&f, &v, 4);
    const uint32_t sign = (f >> 16) & 0x8000u;
    int exp = int((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFFu;
    __half h;
    if (((f >> 23) & 0xFF) == 0xFF) { h.x = uint16_t(sign | 0x7C00u | (mant ? 0x200u : 0u)); return h; }
    if (exp >= 31) { h.x = uint16_t(sign | 0x7C00u); return h; }
    if (exp <= 0) {
        if (exp < -10) { h.x = uint16_t(sign); return h; }
        mant |= 0x800000u;
        const int shift = 14 - exp;
        uint32_t half_m = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1), halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half_m & 1u))) half_m++;
        h.x = uint16_t(sign | half_m);
        return h;
    }
    uint32_t half_m = mant >> 13;
    const uint32_t rem = mant & 0x1FFFu;
    uint32_t bits = (uint32_t(exp) << 10) | half_m;
    if (rem > 0x1000u || (rem == 0x1000u && (bits & 1u))) bits++;
    h.x = uint16_t(sign | bits);
    return h;
}

// ---- intrinsics (single-thread meaning) ----------------------------------------------------------------------
inline void __syncthreads() {}
inline void __threadfence() {}
inline void __threadfence_system() {}
inline long long clock64() { return std::chrono::steady_clock::now().time_since_epoch().count(); }
template <class T> inline T __shfl_down_sync(unsigned, T v, int, int = 32) { return v; }
template <class T> inline T __shfl_xor_sync(unsigned, T v, int, int = 32) { return v; }
template <class T> inline T __ldg(const T* p) { return *p; }
template <class T> inline T atomicAdd(T* a, T v) { T old = *a; *a += v; return old; }
inline int __float_as_int(float f) { int i; std::memcpy(&i, &f, 4); return i; }
inline float __int_as_float(int i) { float f; std::memcpy(&f, &i, 4); return f; }
inline float __expf(float x) { return std::exp(x); }
inline float __fdividef(float a, float b) { return a / b; }

// ---- runtime API ---------------------------------------------------------------------------------------------
enum cudaError_t { cudaSuccess = 0, cudaErrorMemoryAllocation = 2, cudaErrorNotReady = 600, cudaErrorUnknown = 999 };
enum cudaMemcpyKind { cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyDefault };
typedef void* cudaStream_t;
typedef void* cudaEvent_t;

constexpr unsigned cudaHostAllocDefault = 0, cudaHostAllocPortable = 1, cudaHostAllocMapped = 2, cudaHostAllocWriteCombined = 4;
constexpr unsigned cudaStreamNonBlocking = 1, cudaEventDisableTiming = 2, cudaEventDefault = 0;
constexpr unsigned cudaDeviceMapHost = 8;
constexpr unsigned cudaHostRegisterDefault = 0, cudaHostRegisterPortable = 1, cudaHostRegisterMapped = 2, cudaHostRegisterIoMemory = 4, cudaHostRegisterReadOnly = 8;

struct cudaDeviceProp {
    char name[256] = "CPU shim (no GPU)";
    size_t totalGlobalMem = 0;
    int major = 0, minor = 0, multiProcessorCount = 1, l2CacheSize = 0, persistingL2CacheMaxSize = 0;
    int memoryClockRate = 0, memoryBusWidth = 0, maxThreadsPerBlock = 1024;
    int unifiedAddressing = 1, canMapHostMemory = 1;
};

inline const char* cudaGetErrorString(cudaError_t e) { return e == cudaSuccess ? "no error" : "shim error"; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaSetDeviceFlags(unsigned) { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) { *p = cudaDeviceProp(); return cudaSuccess; }
inline cudaError_t cudaMemGetInfo(size_t* f, size_t* t) { *f = *t = 0; return cudaSuccess; }

template <class T> inline cudaError_t cudaHostAlloc(T** p, size_t n, unsigned) {
    void* m = nullptr;
    if (posix_memalign(&m, 4096, n ? n : 1) != 0) return cudaErrorUnknown;
    std::memset(m, 0, n);
    *p = static_cast<T*>(m);
    return cudaSuccess;
}
inline cudaError_t cudaHostAlloc(void** p, size_t n, unsigned f) { return cudaHostAlloc<void>(p, n, f); }
// tests can pretend the device address differs from the host address by pl_shim_dev_offset bytes (such a pointer is only compared, never dereferenced), and can make
// the Nth cudaHostGetDevicePointer call (0-based) and every later one fail with pl_shim_getdev_fail_after
inline long long pl_shim_dev_offset = 0;
inline int pl_shim_getdev_calls = 0, pl_shim_getdev_fail_after = -1;
template <class T> inline cudaError_t cudaHostGetDevicePointer(T** d, void* h, unsigned) {
    if (pl_shim_getdev_fail_after >= 0 && pl_shim_getdev_calls++ >= pl_shim_getdev_fail_after) return cudaErrorUnknown;
    *d = reinterpret_cast<T*>(reinterpret_cast<char*>(h) + pl_shim_dev_offset);
    return cudaSuccess;
}
inline cudaError_t cudaFreeHost(void* p) { std::free(p); return cudaSuccess; }
// registration of existing memory; tests can make the Nth call (0-based) and every later one fail with pl_shim_register_fail_after
inline int pl_shim_register_calls = 0, pl_shim_register_fail_after = -1, pl_shim_registered_now = 0;
inline cudaError_t cudaHostRegister(void*, size_t, unsigned) {
    if (pl_shim_register_fail_after >= 0 && pl_shim_register_calls++ >= pl_shim_register_fail_after) return cudaErrorMemoryAllocation;
    ++pl_shim_registered_now;
    return cudaSuccess;
}
inline cudaError_t cudaHostUnregister(void*) { --pl_shim_registered_now; return cudaSuccess; }
inline cudaError_t cudaGetDevice(int* d) { *d = 0; return cudaSuccess; }
template <class T> inline cudaError_t cudaMalloc(T** p, size_t n) { return cudaHostAlloc(p, n, 0); }
inline cudaError_t cudaFree(void* p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaMemset(void* p, int v, size_t n) { std::memset(p, v, n); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) { std::memcpy(d, s, n); return cudaSuccess; }
inline cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind, cudaStream_t = nullptr) { std::memcpy(d, s, n); return cudaSuccess; }
inline cudaError_t cudaMemsetAsync(void* p, int v, size_t n, cudaStream_t = nullptr) { std::memset(p, v, n); return cudaSuccess; }
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned) { *s = nullptr; return cudaSuccess; }
inline cudaError_t cudaStreamCreate(cudaStream_t* s) { *s = nullptr; return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamQuery(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned = 0) { return cudaSuccess; }
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned) { *e = nullptr; return cudaSuccess; }
inline cudaError_t cudaEventCreate(cudaEvent_t* e) { *e = nullptr; return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = nullptr) { return cudaSuccess; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t, cudaEvent_t) { *ms = 0.f; return cudaSuccess; }
inline cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }

// ---- serial kernel launch ------------------------------------------------------------------------------------
inline void pl_shim_launch(dim3 grid, dim3 block, cudaStream_t, const std::function<void()>& body) {
    gridDim = grid;
    blockDim = block;
    for (unsigned bz = 0; bz < grid.z; ++bz)
        for (unsigned by = 0; by < grid.y; ++by)
            for (unsigned bx = 0; bx < grid.x; ++bx) {
                blockIdx = dim3(bx, by, bz);
                for (unsigned tz = 0; tz < block.z; ++tz)
                    for (unsigned ty = 0; ty < block.y; ++ty)
                        for (unsigned tx = 0; tx < block.x; ++tx) {
                            threadIdx = dim3(tx, ty, tz);
                            body();
                        }
            }
}
