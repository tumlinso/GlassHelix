#ifndef GLASSHELIX_DEVICEMACROS_CUH
#define GLASSHELIX_DEVICEMACROS_CUH

// CUDA Library Error-Checking Macros (no-ops on CPU)
#ifdef __CUDACC__
#ifdef USE_CUSPARSE
#include <cusparse.h>
        #define cusparse_check(expr) \
            do { \
                cusparseStatus_t _st = (expr); \
                if (_st != CUSPARSE_STATUS_SUCCESS) \
                    throw std::runtime_error("cuSPARSE error"); \
            } while (0)
#else
#define cusparse_check(expr) do { (void)(expr); } while (0)
#endif

#ifdef USE_CUBLAS
#include <cublas_v2.h>
        #define cublaslt_check(expr) \
            do { \
                cublasStatus_t _st = (expr); \
                if (_st != CUBLAS_STATUS_SUCCESS) \
                    throw std::runtime_error("cuBLASLt error"); \
            } while (0)
#else
#define cublaslt_check(expr) do { (void)(expr); } while (0)
#endif

#define cuda_check(expr) \
        do { \
            cudaError_t _err = (expr); \
            if (_err != cudaSuccess) \
                throw std::runtime_error(cudaGetErrorString(_err)); \
        } while (0)
#else
    #define cusparse_check(expr)    ((void)(expr))
    #define cublaslt_check(expr)    ((void)(expr))
    #define cuda_check(expr)        ((void)(expr))
#endif

#endif //GLASSHELIX_DEVICEMACROS_CUH
