#ifndef GLASSHELIX_GLASSETCH_TYPES_CUH
#define GLASSHELIX_GLASSETCH_TYPES_CUH

#include <cuda_fp16.h>
#include <cstdint>

using fp16  = __half;
using ui16  = uint16_t;
using ui32  = uint32_t;

template<typename T>
struct DeviceMatrix {
    T*   data;
    int  ld;
    int  rows, cols;
    __device__ T& operator()(int r,int c) { return data[r*ld + c]; }
    __device__ const T& operator()(int r,int c) const { return data[r*ld + c]; }
};

#endif //GLASSHELIX_GLASSETCH_TYPES_CUH
