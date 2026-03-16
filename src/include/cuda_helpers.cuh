#pragma once

#include "macros.cuh"
#include "types.cuh"
#include <stdio.h>

int device_count() {
    int count;
    cuda_check(cudaGetDeviceCount(&count));
    return count;
}

void print_device_info() {
    int count = device_count();
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        cuda_check(cudaGetDeviceProperties(&prop, i));
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    }
}
