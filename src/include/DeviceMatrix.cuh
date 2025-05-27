#ifndef GLASSHELIX_DEVICEMATRIX_CUH
#define GLASSHELIX_DEVICEMATRIX_CUH

#include "Types.cuh"
#include <cuda_fp16.h>

namespace glasshelix::devicematrix {
    template<typename ValueT>
    struct Dense {
        ValueT* val;
        IndexT rows, cols, ld;

        __device__ ValueT& operator()(IndexT r, IndexT c) { return data[r * ld + c]; }
        __device__ const ValueT& operator()(IndexT r, IndexT c) const { return data[r * ld + c]; }
        __device__ ValueT* address(IndexT r, IndexT c) { return &data[r * ld + c]; }
    };

    template<typename ValueT>
    struct SparseCSR {
        IndexT* rowPtr; // length = rows+1
        IndexT* colIdx; // length = nnz
        ValueT* val;    // length = nnz
        IndexT  rows, cols, nnz;
    };

    template<typename ValueT>
    struct SparseCOO {
        IndexT* rowIdx; // length = nnz
        IndexT* colIdx; // length = nnz
        ValueT* val;    // length = nnz
        IndexT  nnz;
    };

} // namespace glasshelix::devicematrix

#endif // GLASSHELIX_DEVICEMATRIX_CUH