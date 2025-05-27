#ifndef GLASSHELIX_DEVICEMACROS_HH
#define GLASSHELIX_DEVICEMACROS_HH

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

#endif //GLASSHELIX_DEVICEMACROS_HH
