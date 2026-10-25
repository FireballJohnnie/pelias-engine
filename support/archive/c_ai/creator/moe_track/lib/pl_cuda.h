// File: <project>\moe_track\lib\pl_cuda.h
// Small portability layer shared by every .cu / .cuh file in moe_track.
//
//  * Under nvcc (__CUDACC__) it is just the CUDA runtime plus two helper macros.
//  * Under a plain C++ compiler it pulls in lib/cuda_shim.h, a CPU-only stand-in that lets the host code and the
//    kernels be type-checked (and, for kernels without block synchronisation, executed serially) where there is no GPU.
//    The shim is a TEST TOOL. It is never used in a real build and says nothing about GPU speed or GPU correctness.
//
//  PL_LAUNCH(kernel, grid, block, shmem_bytes, stream, args...)   launches a kernel
//  PL_CUDA_CHECK(call)                                           aborts with file:line and the CUDA error text
#pragma once

#if defined(__CUDACC__)
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define PL_HD __host__ __device__
#define PL_LAUNCH(kernel, grid, block, shmem, stream, ...) kernel<<<(grid), (block), (shmem), (stream)>>>(__VA_ARGS__)
#else
#include "cuda_shim.h"
#define PL_HD
#define PL_LAUNCH(kernel, grid, block, shmem, stream, ...) \
    pl_shim_launch((grid), (block), (stream), [&]() { kernel(__VA_ARGS__); })
#endif

#include <cstdio>
#include <cstdlib>

#define PL_CUDA_CHECK(call)                                                                              \
    do {                                                                                                 \
        cudaError_t pl_err__ = (call);                                                                   \
        if (pl_err__ != cudaSuccess) {                                                                   \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(pl_err__)); \
            std::exit(1);                                                                                \
        }                                                                                                \
    } while (0)
