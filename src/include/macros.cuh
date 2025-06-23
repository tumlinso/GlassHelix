#ifndef GLASSHELIX_MACROS_CUH
#define GLASSHELIX_MACROS_CUH

#include <stdexcept>

#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #define radical __host__ __device__ __forceinline__

    inline void gpuAssert(cudaError_t code,const char* file,int line) {
        if (code!=cudaSuccess) { fprintf(stderr,"CUDA %s %d\n",file,line); exit(code);}}
    #define check(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#else
    #define radical static inline
    #include <cassert>
    #define check(ans) { assert((ans) == 0); }
#endif

#include "deviceMacros.cuh"

#endif //GLASSHELIX_MACROS_CUH
