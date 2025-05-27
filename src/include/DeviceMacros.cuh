#ifndef GLASSHELIX_DEVICEMACROS_CUH
#define GLASSHELIX_DEVICEMACROS_CUH

#include <stdexcept>
#include <cassert>
#include <cuda_runtime.h>

#define GLASSHELIX_DEVICE_CALLABLE  __device__ __forceinline__
#define GLASSHELIX_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code,const char* file,int line) {
    if (code!=cudaSuccess) { fprintf(stderr,"CUDA %s %d\n",file,line); exit(code);}
}

#define CUDA_CHECK(expr)                                                          \
    do {                                                                          \
        cudaError_t _err = (expr);                                                \
        if (_err != cudaSuccess)                                                  \
            throw std::runtime_error(cudaGetErrorString(_err));                   \
    } while (0)

#define CUSPARSE_CHECK(expr)                                                      \
    do {                                                                          \
        cusparseStatus_t _st = (expr);                                            \
        if (_st != CUSPARSE_STATUS_SUCCESS)                                       \
            throw std::runtime_error("cuSPARSE error");                           \
    } while (0)

#define CUBLASLT_CHECK(expr)                                                      \
    do {                                                                          \
        cublasStatus_t _st = (expr);                                              \
        if (_st != CUBLAS_STATUS_SUCCESS)                                         \
            throw std::runtime_error("cuBLASLt error");                           \
    } while (0)

#endif //GLASSHELIX_DEVICEMACROS_CUH
